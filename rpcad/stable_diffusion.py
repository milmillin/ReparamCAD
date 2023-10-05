import argparse, os, sys, glob
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange, repeat
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch.amp.autocast_mode import autocast

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from transformers import AutoFeatureExtractor


def load_model_from_config(config, ckpt, verbose=False) -> LatentDiffusion:
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    assert isinstance(model, LatentDiffusion)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


class StableDiffusion:
    def __init__(
        self,
        ckpt="weights/sd-v1-4.ckpt",
        ddim_steps: int = 50,
    ):
        depth = 0
        while not os.path.isfile(ckpt) and depth < 3:
            ckpt = os.path.join("..", ckpt)
            depth += 1
        config = os.path.join(os.path.dirname(__file__), "sd_config.yaml")
        config = OmegaConf.load(f"{config}")
        model = load_model_from_config(config, f"{ckpt}")
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = model.to(self.device)

        self.sampler = DDIMSampler(self.model)
        self.sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=0.0, verbose=False)

        self.ddim_steps = ddim_steps

    def encode_img(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: (b, 512, 512, 3), float 0-1
        """
        img = img * 2.0 - 1.0
        img = img.permute(0, 3, 1, 2)
        return self.model.first_stage_model.encode(img).mean * self.model.scale_factor

    def decode_img(self, samples: torch.Tensor, requires_grad: bool = False) -> torch.Tensor:
        """
        returns (b, 512, 512, 3), float 0-1
        """
        if requires_grad:
            x_samples = self.model.differentiable_decode_first_stage(samples)
        else:
            x_samples = self.model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        return x_samples.permute(0, 2, 3, 1)

    def get_conditioning(self, text: str) -> torch.Tensor:
        return self.model.get_learned_conditioning([text])

    def add_noise(self, x: torch.Tensor, t: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to latent vector x.
        returns: (x with noise, noise)
        """
        noise = torch.randn_like(x)
        x_t = torch.sqrt(self.sampler.ddim_alphas[t]) * x + self.sampler.ddim_sqrt_one_minus_alphas[t] * noise  # type: ignore
        return x_t, noise

    def one_step(
        self,
        x: torch.Tensor,
        t: int,
        c: torch.Tensor,
        uc: torch.Tensor,
        scale: float = 7.5,
    ) -> torch.Tensor:
        with torch.no_grad():
            ts = torch.full((x.shape[0],), t, device="cuda")
            x_prev, _ = self.sampler.p_sample_ddim(
                x,
                c,
                ts,
                index=t,
                use_original_steps=False,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=uc,
            )
            return x_prev

    def img2img(
        self,
        init_img: torch.Tensor,
        prompt: str,
        strength: float = 0.75,
        returns_latent: bool = False,
    ) -> torch.Tensor:
        """
        init_img: (1, 512, 512, 3), float 0-1
        returns: (1, 512, 512, 3), float 0-1
        """
        sampler = self.sampler
        ddim_steps = self.ddim_steps

        batch_size = init_img.shape[0]

        scale = 7.5
        start_code = None
        prompts = [prompt] * batch_size
        t_enc = int(strength * ddim_steps)
        init_img = init_img * 2.0 - 1.0
        init_img = init_img.permute(0, 3, 1, 2)
        init_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(init_img))
        with torch.no_grad():
            with autocast("cuda"):
                with self.model.ema_scope():
                    uc = None
                    if scale != 1.0:
                        uc = self.model.get_learned_conditioning([""] * batch_size)
                    c = self.model.get_learned_conditioning(prompts)

                    # encode (scaled latent)
                    z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * batch_size).to(self.device))
                    # decode it
                    samples = sampler.decode(
                        z_enc,
                        c,
                        t_enc,
                        unconditional_guidance_scale=scale,
                        unconditional_conditioning=uc,
                    )

                    if returns_latent:
                        return samples
                    else:
                        x_samples = self.model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                        return x_samples.permute(0, 2, 3, 1)

    def text2img(self, prompt: str) -> torch.Tensor:
        """
        returns: (1, 512, 512, 3), float 0-1
        """
        sampler = DDIMSampler(self.model)
        prompts = [prompt]
        start_code = None
        batch_size = 1
        scale = 7.5
        H = 512
        W = 512
        C = 4
        f = 8
        with torch.no_grad():
            with autocast("cuda"):
                with self.model.ema_scope():
                    uc = None
                    if scale != 1.0:
                        uc = self.model.get_learned_conditioning(batch_size * [""])
                    c = self.model.get_learned_conditioning(prompts)
                    shape = [C, H // f, W // f]
                    samples_ddim, _ = sampler.sample(
                        S=50,
                        conditioning=c,
                        batch_size=batch_size,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=scale,
                        unconditional_conditioning=uc,
                        eta=0.0,
                        x_T=start_code,
                    )
                    x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        return x_samples_ddim.permute(0, 2, 3, 1)

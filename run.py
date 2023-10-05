import hydra
from datetime import datetime
from omegaconf import OmegaConf
import os
from torchvision.transforms import GaussianBlur
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
import math
import sys
from pathlib import Path
import signal
import logging
from time import time

from rpcad.renderer import Renderer
from rpcad.util import save_image, disable_tqdm
from rpcad.stable_diffusion import StableDiffusion
from rpcad.camera import rand_pose, gen_pose, CameraPose
from rpcad.config import register_config, Config, CameraConfig, SDConfig
from rpcad.csg import Mesh
from rpcad.clip import CLIPModel

from init_models import initialize_model

disable_tqdm()
register_config()

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

REPORT_NUM = 20


def task_sd(config: Config):
    start_time = time()
    ######### STATES AND CHECKPOINT #########
    all_losses = []
    all_params = []
    all_diffused_imgs = []
    all_cam_poses = []
    time_taken = 0

    def save_state():
        torch.save(all_params, "params.pt")
        torch.save(all_cam_poses, "cam_poses.pt")
        torch.save(all_diffused_imgs, "diffused_imgs.pt")
        torch.save(all_losses, "losses.pt")
        torch.save(time_taken + time() - start_time, "time_taken.pt")

    def sigusr1_handler(signum, frame):
        nonlocal time_taken
        logging.info("Signal caught")
        sys.stdout.flush()
        save_state()
        logging.info("States saved")
        logging.info(f"TIME: {time() - start_time:.2f}")
        sys.stdout.flush()
        sys.exit()

    signal.signal(signal.SIGUSR1, sigusr1_handler)
    ######### [END] STATES AND CHECKPOINT #########

    sd_config: SDConfig = config.task  # type: ignore
    logging.info("====== Loading Stable Diffusion =====")
    sys.stdout.flush()
    sd = StableDiffusion(ddim_steps=sd_config.ddim_steps)
    logging.info("====== Stable Diffusion Loaded =====")
    logging.info("====== Loading Renderer =====")
    sys.stdout.flush()
    renderer = Renderer()
    logging.info("====== Renderer Loaded =====")
    sys.stdout.flush()
    blur_ = GaussianBlur(
        config.opt.blur_kernel_size,
        sigma=(config.opt.blur_sigma, config.opt.blur_sigma),
    )

    def blur(x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape((-1, *shape[-3:]))
        return blur_(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape(shape)

    a_cam_pose = gen_pose(20, 330, dist=2.0)
    sd_iter = config.opt.sd_per_pose
    cam_config = config.camera

    if config.opt.preprocess == "none":

        def proc(img: torch.Tensor) -> torch.Tensor:
            return img

        def proc_diffused(img: torch.Tensor) -> torch.Tensor:
            return img

    elif config.opt.preprocess == "blur":

        def proc(img: torch.Tensor) -> torch.Tensor:
            return blur(img)

        def proc_diffused(img: torch.Tensor) -> torch.Tensor:
            return img

    elif config.opt.preprocess == "laplace":
        laplace_strength = config.opt.laplace_strength

        def proc(img: torch.Tensor) -> torch.Tensor:
            return (img - laplace_strength * blur(img) + laplace_strength) / (1 + laplace_strength)

        def proc_diffused(img: torch.Tensor) -> torch.Tensor:
            return (img - laplace_strength * blur(img) + laplace_strength) / (1 + laplace_strength)

    elif config.opt.preprocess == "embed":

        def proc(img: torch.Tensor) -> torch.Tensor:
            shape = img.shape
            latent = sd.encode_img(img.reshape((-1, *shape[-3:])))
            return latent.reshape((*shape[:-3], *latent.shape[-3:]))  # (, 4, 64, 64)

        def proc_diffused(img: torch.Tensor) -> torch.Tensor:
            return img

    else:
        raise ValueError(f"Invalid preprocess mode: {config.opt.preprocess}")

    split_cam_pose = (config.opt.preprocess == "embed") and (
        torch.cuda.get_device_properties(0).total_memory < 40000000000
    )

    if config.opt.reduce_diffused == "none":

        def reduce_diffused(img: torch.Tensor, org: torch.Tensor) -> torch.Tensor:
            return img

    elif config.opt.reduce_diffused == "mean":

        def reduce_diffused(img: torch.Tensor, org: torch.Tensor) -> torch.Tensor:
            return img.mean(dim=0, keepdim=True)

    elif config.opt.reduce_diffused == "like-mean":

        def reduce_diffused(img: torch.Tensor, org: torch.Tensor) -> torch.Tensor:
            mean = img.mean(dim=0, keepdim=True)
            return img[((img - mean) ** 2).sum(dim=(1, 2, 3)).argmin(dim=0)].unsqueeze(0)

    elif config.opt.reduce_diffused == "like-org":

        def reduce_diffused(img: torch.Tensor, org: torch.Tensor) -> torch.Tensor:
            return img[((img - org) ** 2).sum(dim=(1, 2, 3)).argmin(dim=0)].unsqueeze(0)

    else:
        raise ValueError(f"Invalid reduce_diffused mode: {config.opt.reduce_diffused}")

    def get_diffused_imgs(mesh: Mesh, cam_params: CameraPose) -> torch.Tensor:
        """
        returns (sd_iter | 1, 512, 512, 3)
        """
        prompt = f"{config.prompt}, {cam_params.dir_text} view"

        # current rendering
        img = renderer.render(mesh, cam_params).detach()

        # diffuse img
        diffused_imgs = []
        for j in range(sd_iter):
            diffused_imgs.append(
                sd.img2img(
                    img,
                    prompt,
                    strength=sd_config.strength,
                    returns_latent=(config.opt.preprocess == "embed"),
                )
            )

        diffused_imgs = torch.cat(diffused_imgs, dim=0)  # (sd_iter|1, 512, 512, 3)
        if config.opt.preprocess == "embed":
            img = proc(img)
        return reduce_diffused(diffused_imgs, img)

    img_save_iters = (np.round(np.linspace(0, config.opt.max_iter, REPORT_NUM + 1)[1:]).astype(np.int32) - 1).clip(0)

    ###### START #####

    model = initialize_model(config.init.model)
    params = model.parameters()
    packed_init_params = model.pack_params(params).clone().detach()

    # load checkpoint if exists
    if Path("params.pt").exists():
        try:
            all_losses = torch.load("losses.pt")
            all_params = torch.load("params.pt")
            all_diffused_imgs = torch.load("diffused_imgs.pt")
            all_cam_poses = torch.load("cam_poses.pt")
            time_taken = torch.load("time_taken.pt")

            params: list[torch.Tensor] = all_params[-1]
            params = [p.to(device="cuda").requires_grad_() for p in params]
            start_it = len(all_params)
            logging.info(f"Resuming from iteration: {start_it}")
            sys.stdout.flush()
        except Exception as e:
            logging.info(e)
            start_it = 0
            logging.info("Starting from 0")
    else:
        start_it = 0
        logging.info("Starting from 0")

    for it in range(start_it, config.opt.max_iter):
        # make cube
        mesh = model.generate_mesh(params)
        normalize_xfrm = mesh.get_normalize_xfrm()
        mesh = mesh.transform(normalize_xfrm)

        cam_poses = []
        diffused_imgs = []
        for i in range(config.opt.cam_poses_per_iter):
            cam_params = rand_pose(cam_config, "cuda")
            diffused_img = get_diffused_imgs(mesh, cam_params)
            cam_poses.append(cam_params)
            diffused_imgs.append(diffused_img)
            if it in img_save_iters:
                if config.opt.preprocess == "embed":
                    decoded = torch.cat(
                        [sd.decode_img(dimg.unsqueeze(0)) for dimg in diffused_img],
                        dim=0,
                    )
                    save_image(decoded.mean(dim=0), f"diffused_{it:05d}_{i}.png")
                else:
                    save_image(diffused_img.mean(dim=0), f"diffused_{it:05d}_{i}.png")
        diffused_imgs = torch.stack(diffused_imgs, dim=0)
        diffused_imgs = proc_diffused(diffused_imgs)  # (n-poses, sd-iter | 1, x, y, z)

        optimizer = torch.optim.SGD(params, lr=config.opt.lr_base)
        losses = []
        for _ in range(config.opt.steps_per_iter):
            optimizer.zero_grad()

            lossf = 0.0
            if split_cam_pose:
                mesh = model.generate_mesh(params).transform(normalize_xfrm)
                for i, cam_pose in enumerate(cam_poses):
                    imgs = renderer.render(mesh, cam_pose)
                    imgs = proc(imgs)
                    loss = torch.mean((imgs - diffused_imgs[i]) ** 2)
                    loss.backward(retain_graph=True)
                    lossf += loss.item()
                    del imgs, loss
            else:
                mesh = model.generate_mesh(params).transform(normalize_xfrm)
                imgs = torch.stack([renderer.render(mesh, cam_pose) for cam_pose in cam_poses], dim=0)
                imgs = proc(imgs)

                loss = torch.mean((imgs - diffused_imgs) ** 2)  # L2 pixel loss.
                lossf += loss.item()
                loss.backward()

            if config.opt.regularizer > 0:
                packed_params = model.pack_params(params)
                reg = config.opt.regularizer * ((packed_params - packed_init_params) ** 2).mean()
                reg.backward()

            losses.append(lossf)

            optimizer.step()

        all_losses.append(losses)
        all_params.append([p.detach().cpu() for p in params])
        all_diffused_imgs.append((diffused_imgs.detach() * 255).to(torch.uint8).cpu())

        img = renderer.render(mesh, a_cam_pose)
        # all_imgs.append((img[0].detach() * 255).to(torch.uint8).cpu())
        all_cam_poses.append(cam_poses)

        if it in img_save_iters:
            save_image(img[0], f"rendered_{it:05d}.png")
            save_state()

        logging.info(f"[{it}/{config.opt.max_iter}] loss: {losses[-1]:.5f}")
        sys.stdout.flush()

    # save video
    # all_imgs = torch.stack(all_imgs, dim=0)
    # skvideo.io.vwrite("opt.mp4", all_imgs.numpy())

    # save gallery
    rim = [np.array(Image.open(f"rendered_{i:05d}.png")) for i in img_save_iters]

    n_rows = math.ceil(REPORT_NUM / 5)
    fig = plt.figure(figsize=(20.0, n_rows * 4 + 1))
    grid = ImageGrid(fig, 111, nrows_ncols=(n_rows, 5), axes_pad=0)

    for ax, im in zip(grid, rim):  # type: ignore
        ax.imshow(im)
    for ax in grid:
        ax.axis("off")
    fig.suptitle(config.prompt)
    plt.tight_layout()
    plt.savefig(f"gallery.png", bbox_inches="tight")

    # save all imgs
    # torch.save(all_imgs, "imgs.pt")

    # save diffused imgs
    all_diffused_imgs = torch.stack(all_diffused_imgs, dim=0)
    torch.save(all_diffused_imgs, "diffused_imgs.pt")

    # -------
    # Post
    logging.info("====== Extra Steps =====")
    if Path("xtra_z.png").exists():
        logging.info("Extra steps already done. Skipping.")
        return

    a_cam_pose = gen_pose(20, 330, dist=2.0)

    init_model = config.init.model
    prompt = config.prompt
    model = initialize_model(init_model)
    all_params = torch.load("params.pt")
    # all_camposes = torch.load(run / "cam_poses.pt")
    # all_diffused_imgs = torch.load(run / "diffused_imgs.pt")
    prev_param = [p.cuda() for p in all_params[-1]]
    params = [p.cuda().requires_grad_() for p in all_params[-1]]
    # diffused_imgs = all_diffused_imgs[-1].cuda().to(torch.float32) / 255
    noramlize_xfrm = model.generate_mesh(prev_param).get_normalize_xfrm()

    init_mesh = model.generate_mesh(params).transform(noramlize_xfrm)

    cam_config = CameraConfig()
    cam_poses = [gen_pose(15, i * 360 / 10) for i in range(10)] + [gen_pose(35, i * 360 / 10) for i in range(10)]
    torch.save(cam_poses, "xtra_cam_poses.pt")

    dimgs = []
    for cam_pose in cam_poses:
        with torch.no_grad():
            img = renderer.render(init_mesh, cam_pose)
            cur_dimgs = []
            for _ in range(config.opt.sd_per_pose):
                dimg = sd.img2img(img, prompt, 0.3)
                cur_dimgs.append(dimg)
            cur_dimgs = torch.cat(cur_dimgs, dim=0)
            dimg = cur_dimgs[((cur_dimgs - img) ** 2).sum(dim=(1, 2, 3)).argmin(dim=0)].unsqueeze(0)
            dimgs.append(dimg)
    dimgs = torch.cat(dimgs, dim=0)  # (n-campose, 512, 512, 3)

    torch.save((dimgs * 255).to(torch.uint8), "xtra_diffused_imgs.pt")
    save_image(
        dimgs.reshape(4, 5, 512, 512, 3).permute(0, 2, 1, 3, 4).reshape(4 * 512, 5 * 512, 3),
        str("xtra_diffused_imgs.png"),
    )

    split = torch.cuda.get_device_properties(0).total_memory < 20000000000  # 20GB

    batch_size = 5 if split else 20

    xtra_params = []
    xtra_losses = []
    optimizer = torch.optim.Adam(params, lr=1e-3)
    for it in range(500):
        optimizer.zero_grad()
        loss_f = 0.0
        for batch in np.arange(20).reshape(-1, batch_size):
            mesh = model.generate_mesh(params).transform(noramlize_xfrm)
            imgs = torch.cat([renderer.render(mesh, cam_poses[bi]) for bi in batch])
            # imgs = proc(imgs)

            loss = torch.mean((imgs - dimgs[batch]) ** 2)
            loss_f += loss.item()
            loss.backward()
        xtra_losses.append(loss_f)
        # pbar.set_description(f"loss: {loss_f}")

        optimizer.step()

        xtra_params.append([p.detach().cpu() for p in params])
        if (it + 1) % 20 == 0:
            with torch.no_grad():
                mesh = model.generate_mesh(params).transform(noramlize_xfrm)
                img = renderer.render(mesh, a_cam_pose)
                save_image(img[0], str(f"xtra_{it:05d}.png"))
    min_id = min(((v, i) for i, v in enumerate(xtra_losses)))[1]
    mesh = model.generate_mesh([p.cuda() for p in xtra_params[min_id]]).transform(noramlize_xfrm)
    img = renderer.render(mesh, a_cam_pose)
    save_image(img[0], str("xtra_z.png"))
    torch.save(xtra_params[: min_id + 1], "xtra_params.pt")
    torch.save(xtra_losses[: min_id + 1], "xtra_losses.pt")
    fig, ax = plt.subplots(1, 1)
    ax.plot(xtra_losses)
    fig.savefig(str("xtra_losses.png"))


def task_clip(config: Config):
    all_losses = []
    all_params = []
    all_cam_poses = []

    def save_state():
        torch.save(all_losses, "losses.pt")
        torch.save(all_params, "params.pt")
        torch.save(all_cam_poses, "cam_poses.pt")

    logging.info("====== Loading Renderer =====")
    sys.stdout.flush()
    renderer = Renderer()
    logging.info("====== Renderer Loaded =====")
    sys.stdout.flush()

    logging.info("====== Loading CLIP =====")
    sys.stdout.flush()
    clip = CLIPModel()
    logging.info("====== CLIP Loaded =====")
    sys.stdout.flush()

    a_cam_pose = gen_pose(20, 330, dist=2.0)
    cam_config = config.camera
    img_save_iters = (np.round(np.linspace(0, config.opt.max_iter, REPORT_NUM + 1)[1:]).astype(np.int32) - 1).clip(0)

    text_emb = F.normalize(clip.encode_text(config.prompt).detach(), dim=-1)

    model = initialize_model(config.init.model)
    params = model.parameters()

    for it in range(0, config.opt.max_iter):
        # make cube
        mesh = model.generate_mesh(params)
        normalize_xfrm = mesh.get_normalize_xfrm()
        mesh = mesh.transform(normalize_xfrm)

        cam_poses = []
        for i in range(config.opt.cam_poses_per_iter):
            cam_pose = rand_pose(cam_config, "cuda")
            cam_poses.append(cam_pose)

        optimizer = torch.optim.SGD(params, lr=config.opt.lr_base)
        losses = []
        for _ in range(config.opt.steps_per_iter):
            optimizer.zero_grad()

            mesh = model.generate_mesh(params).transform(normalize_xfrm)
            imgs = torch.cat([renderer.render(mesh, cam_pose) for cam_pose in cam_poses], dim=0)
            # imgs = proc(imgs)
            imgs_emb = F.normalize(clip.encode_image(imgs))  # b, x

            # loss = torch.mean((imgs_emb - text_emb) ** 2)  # L2 pixel loss.
            loss = -(imgs_emb * text_emb).sum(dim=-1).mean()  # -Similarity
            losses.append(loss.item())
            loss.backward()

            optimizer.step()

        all_losses.append(losses)
        all_params.append([p.detach().cpu() for p in params])

        img = renderer.render(mesh, a_cam_pose)
        all_cam_poses.append(cam_poses)

        if it in img_save_iters:
            save_image(img[0], f"rendered_{it:05d}.png")

            save_state()

        logging.info(f"[{it}/{config.opt.max_iter}] loss: {losses[-1]:.5f}")
        sys.stdout.flush()

    # save video
    # all_imgs = torch.stack(all_imgs, dim=0)
    # skvideo.io.vwrite("opt.mp4", all_imgs.numpy())

    # save gallery
    rim = [np.array(Image.open(f"rendered_{i:05d}.png")) for i in img_save_iters]

    n_rows = math.ceil(REPORT_NUM / 5)
    fig = plt.figure(figsize=(20.0, n_rows * 4 + 1))
    grid = ImageGrid(fig, 111, nrows_ncols=(n_rows, 5), axes_pad=0)

    for ax, im in zip(grid, rim):  # type: ignore
        ax.imshow(im)
    for ax in grid:
        ax.axis("off")
    fig.suptitle(config.prompt)
    plt.tight_layout()
    plt.savefig(f"gallery.png", bbox_inches="tight")


def task_distillation(config: Config):
    all_losses = []
    all_params = []
    all_cam_poses = []

    def save_state():
        torch.save(all_losses, "losses.pt")
        torch.save(all_params, "params.pt")
        torch.save(all_cam_poses, "cam_poses.pt")

    logging.info("====== Loading Renderer =====")
    sys.stdout.flush()
    renderer = Renderer()
    logging.info("====== Renderer Loaded =====")

    logging.info("====== Loading Stable Diffusion =====")
    sys.stdout.flush()
    sd = StableDiffusion()
    logging.info("====== Stable Diffusion Loaded =====")
    sys.stdout.flush()

    a_cam_pose = gen_pose(20, 330, dist=2.0)
    cam_config = config.camera
    img_save_iters = (np.round(np.linspace(0, config.opt.max_iter, REPORT_NUM + 1)[1:]).astype(np.int32) - 1).clip(0)

    uc = sd.get_conditioning("")

    model = initialize_model(config.init.model)
    params = model.parameters()
    optimizer = torch.optim.SGD(params, lr=config.opt.lr_base)
    for it in range(0, config.opt.max_iter):
        # make cube
        mesh = model.generate_mesh(params)
        normalize_xfrm = mesh.get_normalize_xfrm()
        mesh = mesh.transform(normalize_xfrm)

        t = np.random.randint(10, 40)

        cam_poses = []
        scores = []
        with torch.no_grad():
            for i in range(config.opt.cam_poses_per_iter):
                cam_pose = rand_pose(cam_config, "cuda")
                cam_poses.append(cam_pose)
                c = sd.get_conditioning(f"{config.prompt}, {cam_pose.dir_text} view")
                img = renderer.render(mesh, cam_pose).detach()
                img_latents = sd.encode_img(img)

                scores_ = []
                for _ in range(config.opt.sd_per_pose):
                    noised_img_latents, noise = sd.add_noise(img_latents, t)
                    x_prev = sd.one_step(noised_img_latents, t - 1, c, uc)
                    scores_.append(x_prev - noised_img_latents)
                scores.append(torch.cat(scores_, dim=0))
            scores = torch.stack(scores, dim=0)  # n-pose, n-sd, x, y, z

        losses = []
        for _ in range(config.opt.steps_per_iter):
            optimizer.zero_grad()

            loss_f = 0
            for i, cam_pose in enumerate(cam_poses):
                mesh = model.generate_mesh(params).transform(normalize_xfrm)
                imgs = renderer.render(mesh, cam_pose)
                img_latents = sd.encode_img(imgs).unsqueeze(1)  # n-pose, 1, x, y, z

                loss = torch.mean((img_latents - img_latents.detach() - scores[i]) ** 2) / len(
                    cam_poses
                )  # L2 pixel loss.
                loss_f += loss.item()
                loss.backward()

            losses.append(loss_f)
            optimizer.step()

        all_losses.append(losses)
        all_params.append([p.detach().cpu() for p in params])

        img = renderer.render(mesh, a_cam_pose)
        all_cam_poses.append(cam_poses)

        if it in img_save_iters:
            save_image(img[0], f"rendered_{it:05d}.png")

            save_state()

        logging.info(f"[{it}/{config.opt.max_iter}] loss: {losses[-1]:.5f}")
        sys.stdout.flush()

    # save video
    # all_imgs = torch.stack(all_imgs, dim=0)
    # skvideo.io.vwrite("opt.mp4", all_imgs.numpy())

    # save gallery
    rim = [np.array(Image.open(f"rendered_{i:05d}.png")) for i in img_save_iters]

    n_rows = math.ceil(REPORT_NUM / 5)
    fig = plt.figure(figsize=(20.0, n_rows * 4 + 1))
    grid = ImageGrid(fig, 111, nrows_ncols=(n_rows, 5), axes_pad=0)

    for ax, im in zip(grid, rim):  # type: ignore
        ax.imshow(im)
    for ax in grid:
        ax.axis("off")
    fig.suptitle(config.prompt)
    plt.tight_layout()
    plt.savefig(f"gallery.png", bbox_inches="tight")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: Config):
    logging.info(os.getcwd())
    logging.info(hydra.utils.get_original_cwd())
    logging.info(config)
    name = config.name
    prompt = config.prompt
    task = config.task
    init_model = config.init.model
    assert config.opt.preprocess in ["none", "blur", "laplace", "embed"]

    # write current job_id
    with open("jobid.txt", "w") as f:
        job_id = os.environ.get("SLURM_JOB_ID", "-1")
        f.write(job_id)

    if task.name == "sd":
        task_sd(config)
    elif task.name == "clip":
        task_clip(config)
    elif task.name == "distillation":
        task_distillation(config)
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    main()

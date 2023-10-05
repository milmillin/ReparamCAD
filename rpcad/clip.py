import torch
import numpy as np
from torchvision.transforms import ToPILImage, Normalize, Resize
import clip
from clip.model import CLIP
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.transforms import GaussianBlur

from .typing import Device
from .defaults import Default


class CLIPModel:
    """
    Wrapper of a CLIP Model
    """
    def __init__(self, name: str = "ViT-L/14", device: Device = None):
        if device is None:
            device = Default.DEVICE
        self.model, _ = clip.load(name, device)
        self.device = device
        assert isinstance(self.model, CLIP)

        self.normalizer = Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        )
        self.resizer = Resize(self.model.visual.input_resolution)

    def encode_text(self, prompt: str) -> torch.Tensor:
        """
        returns: (b, dim)
        """
        text = clip.tokenize([prompt])
        return self.model.encode_text(text.cuda())

    def encode_image(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: (b, H, W, 3), float 0-1
        returns: (b, dim)
        """
        img = self.normalizer(self.resizer(img.flip(1).permute(0, 3, 1, 2)))
        return self.model.encode_image(img)

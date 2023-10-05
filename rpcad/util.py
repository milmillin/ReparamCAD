import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import functional as T
from tqdm import tqdm
from functools import partialmethod
import colorsys
import meshplot as mp

from .defaults import Color
from .csg import Mesh


def display_image(image: torch.Tensor) -> Image.Image:
    image_np = (image * 255).detach().cpu().numpy().astype(np.uint8)
    return Image.fromarray(image_np, "RGBA" if image.shape[-1] == 4 else "RGB")


# image: H * W * 3 [0, 1]
def save_image(image: torch.Tensor, filename: str) -> None:
    display_image(image).save(filename)


def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))


def load_env(filename: str, device="cuda") -> torch.Tensor:
    img = Image.open(filename)
    img = T.pil_to_tensor(img)
    return (img.to(torch.float32) / 255.0).unsqueeze(0).to(device=device)


def disable_tqdm():
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  # type: ignore


def get_random_colors(n: int) -> list[Color]:
    dx = 1 / n
    res: list[Color] = []
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb(i * dx, 1, 1)
        res.append((round(r * 255), round(g * 255), round(b * 255)))
    return res


def display_mesh(*meshes: Mesh, shading={}) -> mp.Viewer:
    mesh, *res = meshes
    plot = mp.plot(
        mesh.V.detach().cpu().numpy(),
        mesh.F.detach().cpu().numpy(),
        c=mesh.color.detach().cpu().numpy(),
        return_plot=True,
        shading=shading,
    )
    assert isinstance(plot, mp.Viewer)
    for m in res:
        plot.add_mesh(
            m.V.detach().cpu().numpy(),
            m.F.detach().cpu().numpy(),
            c=m.color.detach().cpu().numpy(),
            shading=shading,
        )
    return plot

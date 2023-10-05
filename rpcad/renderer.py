import torch
import nvdiffrast.torch as dr
import torchvision.transforms as transforms
from torchvision.transforms.functional import resize, InterpolationMode
import random
import math
from typing import Optional, cast
import numpy as np

from .config import RenderParams
from .camera import CameraPose, gen_pose
from .csg import Mesh


def _safe_normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, dim=-1)


# Transform vertex positions to clip space
def transform_pos(t_mtx: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    # (x,y,z) -> (x,y,z,1)
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], dim=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]


blurs = [
    transforms.Compose([transforms.GaussianBlur(11, sigma=(5, 5))]),
    transforms.Compose([transforms.GaussianBlur(11, sigma=(2, 2))]),
    transforms.Compose([transforms.GaussianBlur(5, sigma=(5, 5))]),
    transforms.Compose([transforms.GaussianBlur(5, sigma=(2, 2))]),
]


def get_random_bg(h: int, w: int, device="cuda") -> torch.Tensor:
    p = torch.rand(1)

    if p > 2 / 3:
        # blur random
        background = blurs[random.randint(0, 3)](torch.rand((1, 3, h, w), device=device)).permute(0, 2, 3, 1)
    elif p > 1 / 3:
        # pattern
        size = random.randint(5, 10)
        background = torch.vstack(
            [
                torch.full((1, size, size), torch.rand(1).item() / 2, device=device),
                torch.full((1, size, size), torch.rand(1).item() / 2, device=device),
                torch.full((1, size, size), torch.rand(1).item() / 2, device=device),
            ]
        ).unsqueeze(0)

        second = torch.rand(3)

        background[:, 0, ::2, ::2] = second[0]
        background[:, 1, ::2, ::2] = second[1]
        background[:, 2, ::2, ::2] = second[2]

        background[:, 0, 1::2, 1::2] = second[0]
        background[:, 1, 1::2, 1::2] = second[1]
        background[:, 2, 1::2, 1::2] = second[2]

        # background = blurs[random.randint(0, 3)]( resize(background, out_shape=(h, w)))
        background = blurs[random.randint(0, 3)](
            resize(background, size=[h, w], interpolation=InterpolationMode.BICUBIC)
        )

        background = background.permute(0, 2, 3, 1).clamp(0, 1)

    else:
        # solid color
        background = (
            torch.vstack(
                [
                    torch.full((1, h, w), torch.rand(1).item(), device=device),
                    torch.full((1, h, w), torch.rand(1).item(), device=device),
                    torch.full((1, h, w), torch.rand(1).item(), device=device),
                ]
            )
            .unsqueeze(0)
            .permute(0, 2, 3, 1)
        )

    return background


def get_env_bg(
    env_map: torch.Tensor,  # (1, 3, H, W) -> [0, 1]
    cam_params: CameraPose,
    res: int,
    device="cuda",
) -> torch.Tensor:
    azim = cam_params.azim
    elev = cam_params.elev
    fov = cam_params.fov

    view_dir = torch.tensor(
        [
            math.cos(-azim) * math.cos(elev),
            math.sin(elev),
            math.sin(-azim) * math.cos(elev),
        ],
        device=device,
    )
    x = torch.cross(view_dir, torch.tensor([0.0, 1.0, 0.0], device=device))
    y = torch.cross(x, view_dir)

    x /= torch.norm(x, dim=-1, keepdim=True)
    y /= torch.norm(y, dim=-1, keepdim=True)

    x = x * math.sin(fov / 2)
    y = y * math.sin(fov / 2)

    dx = 1 / res
    xs = torch.arange(dx - 1, 1, dx * 2, device=device)
    ys = torch.arange(dx - 1, 1, dx * 2, device=device)

    xxs, yys = torch.meshgrid(xs, ys, indexing="xy")
    view_dirs = xxs.unsqueeze(-1) * x + yys.unsqueeze(-1) * y + view_dir
    azim = torch.atan2(view_dirs[:, :, 2], view_dirs[:, :, 0]) / np.pi

    dis = torch.sqrt(view_dirs[:, :, 2] ** 2 + view_dirs[:, :, 0] ** 2)
    elev = torch.atan2(view_dirs[:, :, 1], dis) / np.pi * 2

    uvs = torch.stack([azim, elev], dim=-1).unsqueeze(0)

    # returns (1, H, W, 3)
    return (
        torch.nn.functional.grid_sample(env_map, uvs, align_corners=False, padding_mode="reflection")
        .permute(0, 2, 3, 1)
        .flip(1)
    )


def avg_pool_nhwc(x: torch.Tensor, size) -> torch.Tensor:
    y = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
    y = torch.nn.functional.avg_pool2d(y, size)
    return y.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC

_RENDERER: Optional["Renderer"] = None

class Renderer:
    def __init__(
        self,
        use_opengl: bool = False,
        env_map: Optional[torch.Tensor] = None,  # (1, 3, H, W)
    ):
        super().__init__()
        self.glctx = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()

        self.env_map = env_map

        elev = math.radians(10)
        # angs = math.radians(90) + torch.linspace(0, np.pi * 2, self.n_lights + 1, device="cuda")[:-1]
        angs = torch.deg2rad(torch.tensor([60.0, -60.0, 180.0], device="cuda"))
        self.light_pos = (
            torch.stack(
                [
                    torch.sin(angs) * math.cos(elev),
                    torch.ones_like(angs, device="cuda") * math.sin(elev),
                    torch.cos(angs) * math.cos(elev),
                ],
                dim=-1,
            )
            * 20
        )
        self.n_lights = len(angs)
        self.light_contrib = torch.tensor([5, 2, 2, 2.5], device="cuda")
        if False:
            self.main_light_contrib = 0.0
            self.light_contrib = torch.tensor(
                [
                    *[(1.0 - self.main_light_contrib) / self.n_lights for _ in range(self.n_lights)],
                    self.main_light_contrib,
                ],
                device="cuda",
            )
        # helper
        self.zero_tensor = torch.as_tensor(0.0, dtype=torch.float32, device="cuda")
        self.one_tensor = torch.as_tensor(1.0, dtype=torch.float32, device="cuda")

    def _render_layer(
        self,
        rast: torch.Tensor,
        rast_db: torch.Tensor,
        mesh: Mesh,
        cam_pose: CameraPose,
        render_params: RenderParams,
    ) -> torch.Tensor:
        ##########################
        # Interpolate Attributes
        ##########################

        # Interpolate world space position
        V, _ = cast(tuple[torch.Tensor, torch.Tensor], dr.interpolate(mesh.V, rast, mesh.F))  # (1, res, res, 3)

        # Interpolate normal
        N, _ = cast(tuple[torch.Tensor, torch.Tensor], dr.interpolate(mesh.N, rast, mesh.F))  # (1, res, res, 3)
        N = _safe_normalize(N)

        ##########################
        # Shade
        ##########################

        max_brightness = self.light_contrib.sum() * (1 + render_params.k_s)

        campos = cam_pose.campos

        lights = torch.cat([self.light_pos, campos.unsqueeze(0)], dim=0)  # (N, 3)
        lvec = lights.unsqueeze(1).unsqueeze(1).unsqueeze(1) - V.unsqueeze(0)  # (N, 1, res, res, 3)
        lvec = _safe_normalize(lvec)
        vvec = lvec[-1:, ...]  # last light is the camera

        l_dot_n = (N.squeeze(0) * lvec).sum(dim=-1, keepdim=True)  # (N, 1, res, res, 1)

        rvec = 2 * l_dot_n * N.squeeze(0) - lvec

        diffuse = torch.clamp(l_dot_n, min=0)  # (N, 1, res, res, 1)
        specular = torch.clamp(((rvec * vvec).sum(dim=-1, keepdim=True)) ** render_params.n_s, min=0)
        intensity = diffuse + specular
        intensity = (intensity * self.light_contrib.reshape(-1, 1, 1, 1, 1)).sum(dim=0)  # (1, res, res, 1)

        tri = rast[..., -1].long() - 1  # (1, res, res)
        diffuse_color = mesh.color[tri]  # (1, res, res, 3)

        return diffuse_color * (intensity / max_brightness) ** 0.45

    # return: 1 * H * W * 3
    def render(
        self,
        mesh: Mesh,
        cam_pose: CameraPose = gen_pose(30, 30),
        render_params: RenderParams = RenderParams(),
    ) -> torch.Tensor:
        bg_mode = render_params.bg_mode
        res = render_params.res
        spp = render_params.spp

        full_res = res * spp

        mesh = mesh.cuda()
        cam_pose = cam_pose.cuda()

        V, F, N = mesh.V, mesh.F, mesh.N

        # clip space transform
        V_clip = transform_pos(cam_pose.mvp, V)

        # rasterize
        with dr.DepthPeeler(self.glctx, V_clip, F, [full_res, full_res]) as peeler:
            rast, db = cast(tuple[torch.Tensor, torch.Tensor], peeler.rasterize_next_layer())
        mask = rast[..., -1:] == 0

        # shade
        color = self._render_layer(rast, db, mesh, cam_pose, render_params)

        # add background
        if bg_mode == "random":
            bg = get_random_bg(full_res, full_res)
            color = torch.where(mask, bg, color)  # white backgrond
        elif bg_mode == "env":
            assert self.env_map is not None
            bg = get_env_bg(self.env_map, cam_pose, full_res, device="cuda")
            color = torch.where(mask, bg, color)
        elif bg_mode == "white":
            color = torch.where(mask, self.one_tensor, color)  # white backgrond
        else:
            raise ValueError(f"Invalid bg_mode: {bg_mode}")

        color = cast(torch.Tensor, dr.antialias(color.contiguous(), rast, V_clip, F)).flip(1)
        return avg_pool_nhwc(color, spp) if spp > 1 else color

    def render_vertices(
        self,
        mesh: Mesh,
        cam_pose: CameraPose = gen_pose(30, 30),
        render_params: RenderParams = RenderParams(),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        res = render_params.res

        full_res = res * 1

        mesh = mesh.cuda()
        cam_pose = cam_pose.cuda()

        V, F, N = mesh.V, mesh.F, mesh.N

        def _compute_volume(V: torch.Tensor) -> torch.Tensor:
            return torch.linalg.det(V[F.long()]).sum() / 6  # (n-f,)

        # clip space transform
        V_clip = transform_pos(cam_pose.mvp, V)

        # rasterize
        with dr.DepthPeeler(self.glctx, V_clip, F, [full_res, full_res]) as peeler:
            rast, db = cast(tuple[torch.Tensor, torch.Tensor], peeler.rasterize_next_layer())

        # scale by number of pixels shown
        fid = rast[..., -1:].long()
        f_count = torch.bincount(fid[fid >= 0])
        fscale = f_count[fid].reciprocal()  # (1, res, res, 1)

        # scale by volume jacobian
        scale_ = torch.autograd.functional.jacobian(_compute_volume, mesh.V, create_graph=False)  # (n-V, 3)
        scale, _ = cast(tuple[torch.Tensor, torch.Tensor], dr.interpolate(scale_, rast, mesh.F))  # (1, res, res, 1)
        # scale = scale * fscale

        V, _ = cast(tuple[torch.Tensor, torch.Tensor], dr.interpolate(mesh.V, rast, mesh.F, db))  # (1, res, res, 3)
        # V = dr.antialias(V.contiguous(), rast, V_clip, F).flip(1)
        return V, scale, fscale, rast[..., 2:3]

    # return: 1 * H * W * 3
    @torch.no_grad()
    def render_figure(
        self,
        mesh: Mesh,
        cam_pose: CameraPose = gen_pose(30, 30),
        render_params: RenderParams = RenderParams(),
        alpha: float = 0.8,
        num_layers: int = 4,
        transparent: bool = True,
    ) -> torch.Tensor:
        alpha = float(alpha)
        res = render_params.res
        spp = render_params.spp

        full_res = res * spp

        mesh = mesh.cuda()
        cam_pose = cam_pose.cuda()

        V, F, N = mesh.V, mesh.F, mesh.N

        # clip space transform
        V_clip = transform_pos(cam_pose.mvp, V)

        # rasterize
        with dr.DepthPeeler(self.glctx, V_clip, F, [full_res, full_res]) as peeler:
            colors: list[torch.Tensor] = []
            rasts: list[torch.Tensor] = []
            for i in range(num_layers):
                rast, db = cast(tuple[torch.Tensor, torch.Tensor], peeler.rasterize_next_layer())
                # shade
                color = self._render_layer(rast, db, mesh, cam_pose, render_params)
                colors.append(color)
                rasts.append(rast)

        # composite back to front
        if transparent:
            accum_col = torch.zeros_like(colors[0], device="cuda")
        else:
            accum_col = torch.ones_like(colors[0], device="cuda")
        accum_alpha = torch.zeros_like(colors[0][..., -1:], device="cuda")
        ones = torch.ones_like(colors[0][..., -1:], device="cuda")
        for color, rast in reversed(list(zip(colors, rasts))):
            tris = rast[..., -1:]
            alphas = (tris != 0) * alpha
            # line2 = (tris != tris.roll(3, dims=1)) | (tris != tris.roll(3, dims=2))
            # line1 = (tris != tris.roll(1, dims=1)) | (tris != tris.roll(1, dims=2))
            # color = color.minimum(1 - line2 * 0.8).minimum(~line1)
            accum_col = torch.lerp(accum_col, color, alphas)
            accum_alpha = torch.lerp(accum_alpha, ones, alphas)
            accum_col = cast(torch.Tensor, dr.antialias(accum_col.contiguous(), rast, V_clip, F))

        if transparent:
            accum_col = torch.cat([accum_col, accum_alpha], dim=-1)
        accum_col = accum_col.flip(1)
        return avg_pool_nhwc(accum_col, spp) if spp > 1 else accum_col

    @classmethod
    def get_shared(cls) -> "Renderer":
        global _RENDERER
        if _RENDERER is None:
            _RENDERER = Renderer()
        return _RENDERER

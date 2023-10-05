import torch
import numpy as np
from dataclasses import dataclass

from .config import CameraConfig
from .typing import Device
from .defaults import Default

# ----------------------------------------------------------------------------
# Projection and transformation matrix helpers.
# ----------------------------------------------------------------------------


def projection(fov: float = 45.0, ar: float = 1.0, near: float = 1.0, far: float = 50.0):
    """
    From https://github.com/rgl-epfl/large-steps-pytorch by @bathal1 (Baptiste Nicolet)

    Build a perspective projection matrix.
    Parameters
    ----------
    fov : float
        Horizontal field of view (in degrees).
    ar : float
        Aspect ratio (w/h).
    near : float
        Depth of the near plane relative to the camera.
    far : float
        Depth of the far plane relative to the camera.
    """
    fov_rad = np.radians(fov)
    tanhalffov = np.tan((fov_rad / 2))
    max_y = tanhalffov * near
    min_y = -max_y
    max_x = max_y * ar
    min_x = -max_x

    z_sign = -1.0
    proj_mat = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    proj_mat[0, 0] = 2.0 * near / (max_x - min_x)
    proj_mat[1, 1] = 2.0 * near / (max_y - min_y)
    proj_mat[0, 2] = (max_x + min_x) / (max_x - min_x)
    proj_mat[1, 2] = (max_y + min_y) / (max_y - min_y)
    proj_mat[3, 2] = z_sign

    proj_mat[2, 2] = z_sign * far / (far - near)
    proj_mat[2, 3] = -(far * near) / (far - near)

    return proj_mat


def translate(x: float, y: float, z: float) -> np.ndarray:
    return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]]).astype(np.float32)


def rotate_x(rad: float) -> np.ndarray:
    s, c = np.sin(rad), np.cos(rad)
    return np.array([[1, 0, 0, 0], [0, c, s, 0], [0, -s, c, 0], [0, 0, 0, 1]]).astype(np.float32)


def rotate_y(rad: float) -> np.ndarray:
    s, c = np.sin(rad), np.cos(rad)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]]).astype(np.float32)


DIR_TEXT = ["front", "side", "back", "side", "overhead", "bottom"]

_TWO_PI = 2 * np.pi


def get_view_direction(theta: float, phi: float, cam_config: CameraConfig) -> str:
    """
    Get viewing direction text from camera angles (in degrees).
    See also: cam_config.front_thresh and cam_config.overhead_thresh.
    """
    front_thresh = np.radians(cam_config.front_thresh)
    overhead_thresh = np.radians(cam_config.overhead_thresh)

    phi = phi % _TWO_PI
    if phi < 0:
        phi += _TWO_PI

    front_half = front_thresh / 2.0
    if phi < front_half or phi > np.pi * 2 - front_half:  # front
        res = 0  # front
    elif phi >= np.pi - front_half and phi <= np.pi + front_half:
        res = 2  # back
    elif phi < np.pi:
        res = 1  # side (left)
    else:
        res = 3  # side (right)

    # override by theta
    if theta >= overhead_thresh:
        res = 4  # overhead
    elif theta <= -overhead_thresh:
        res = 5  # bottom
    return DIR_TEXT[res]


@dataclass
class CameraPose:
    campos: torch.Tensor
    mvp: torch.Tensor
    dir_text: str
    elev: float  # radians
    azim: float  # radians
    fov: float  # radians

    def to(self, device: Device) -> "CameraPose":
        return CameraPose(
            self.campos.to(device),
            self.mvp.to(device),
            self.dir_text,
            self.elev,
            self.azim,
            self.fov,
        )

    def cuda(self) -> "CameraPose":
        return self.to("cuda")


def gen_pose(
    elev: float,  # degrees
    azim: float,  # degrees
    dist: float = 2.0,
    fov: float = 40.0,  # degrees
    orthographic: bool = False,
    cam_config: CameraConfig = CameraConfig(),
    device: Device = None,
) -> CameraPose:
    if device is None:
        device = Default.DEVICE

    # assumes elev, azim in [0, 360)
    elev = np.radians(elev)
    azim = np.radians(azim)
    fov = np.radians(fov)

    rot = np.matmul(rotate_x(-elev), rotate_y(-azim))
    if orthographic:
        left = -dist
        right = dist
        top = dist
        bottom = -dist
        near = 0
        far = 10
        proj = np.array(
            [
                [2 / (right - left), 0, 0, -(right + left) / (right - left)],
                [0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
                [0, 0, -2 / (far - near), -(far + near) / (far - near)],
                [0, 0, 0, 1],
            ]
        )
        dist = 3
    else:
        proj = projection(fov)
    mv = np.matmul(translate(0, 0, -dist), rot)
    mvp = torch.as_tensor(np.matmul(proj, mv), dtype=torch.float32, device=device)
    campos = torch.as_tensor(np.linalg.inv(mv)[:3, 3], dtype=torch.float32, device=device)
    dir_ = get_view_direction(elev, azim, cam_config)
    return CameraPose(campos, mvp, dir_, elev, azim, fov)


def rand_pose(cam_config: CameraConfig, device: Device = None) -> CameraPose:
    if device is None:
        device = Default.DEVICE

    elev = np.radians(np.random.uniform(cam_config.elev_min, cam_config.elev_max))
    azim = np.radians(np.random.uniform(cam_config.azim_min, cam_config.azim_max + 1.0))
    dist = np.random.uniform(cam_config.dist_min, cam_config.dist_max)
    fov = np.radians(np.random.uniform(cam_config.fov_min, cam_config.fov_max))

    rot = np.matmul(rotate_x(-elev), rotate_y(-azim))
    proj = projection(fov)
    if cam_config.aug_loc:
        # random offset
        limit = cam_config.shift_max
        rand_x = np.random.uniform(-limit, limit)
        rand_y = np.random.uniform(-limit, limit)
        mv = np.matmul(translate(rand_x, rand_y, -dist), rot)
    else:
        mv = np.matmul(translate(0, 0, -dist), rot)
    mvp = torch.as_tensor(np.matmul(proj, mv), dtype=torch.float32, device=device)
    campos = torch.as_tensor(np.linalg.inv(mv)[:3, 3], dtype=torch.float32, device=device)

    dir_ = get_view_direction(elev, azim, cam_config)
    return CameraPose(campos, mvp, dir_, elev, azim, fov)

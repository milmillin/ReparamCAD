from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class CameraConfig:
    fov_min: float = 40.0  # Minimum camera field of view angle during renders
    fov_max: float = 40.0  # Maximum camera field of view angle during renders
    dist_min: float = 2.0  # Minimum distance of camera from mesh during renders
    dist_max: float = 2.0  # Maximum distance of camera from mesh during renders
    elev_min: float = 5.0  # Minimum elevation angle
    elev_max: float = 50.0  # Maximum elevation angle
    azim_min: float = -180.0  # Minimum azimuth angle
    azim_max: float = 180.0  # Maximum azimuth angle
    aug_loc: bool = False  # Offset mesh from center of image?
    shift_max: float = 0.5  # Maximum shift if aug_loc
    overhead_thresh: float = 60.0  # >= x will be considered overhead
    front_thresh: float = 60.0  # [-x / 2, x / 2] will be considered front


@dataclass
class RenderParams:
    res: int = 512
    spp: int = 2
    bg_mode: str = "white"
    k_s: float = 0.04
    n_s: float = 50


@dataclass
class OptConfig:
    max_iter: int = 400
    cam_poses_per_iter: int = 5
    steps_per_iter: int = 30
    sd_per_pose: int = 3
    lr_base: float = 0.05
    lr_ramp: float = 1.0
    blur_kernel_size: int = 13
    blur_sigma: float = 2.0
    preprocess: str = "laplace"  # [none, blur, laplace, embed]
    laplace_strength: float = 0.3
    reduce_diffused: str = "like-org"  # [none, mean, like-mean, like-org]
    regularizer: float = 0.001


@dataclass
class TaskSpecificConfig:
    name: str = MISSING


@dataclass
class SDConfig(TaskSpecificConfig):
    name: str = "sd"
    ddim_steps: int = 50
    strength: float = 0.75


@dataclass
class CLIPConfig(TaskSpecificConfig):
    name: str = "clip"


@dataclass
class DistillationConfig(TaskSpecificConfig):
    name: str = "distillation"


@dataclass
class InitConfig:
    model: str = "chair"


@dataclass
class Config:
    name: str = MISSING
    prompt: str = MISSING
    task: TaskSpecificConfig = MISSING
    camera: CameraConfig = field(default_factory=CameraConfig)
    render: RenderParams = field(default_factory=RenderParams)
    opt: OptConfig = field(default_factory=OptConfig)
    init: InitConfig = field(default_factory=InitConfig)


def register_config():
    cs = ConfigStore.instance()
    cs.store(group="task", name="sd", node=SDConfig)
    cs.store(group="task", name="clip", node=CLIPConfig)
    cs.store(group="task", name="distillation", node=DistillationConfig)
    cs.store(name="base_config", node=Config)

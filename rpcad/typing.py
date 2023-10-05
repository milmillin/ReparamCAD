import torch
from typing import Union
from pathlib import Path

Device = Union[str, torch.device, None]
Color = tuple[int, int, int]
FileLike = Union[Path, str]

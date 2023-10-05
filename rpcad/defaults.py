import torch
from .typing import Device, Color


class Default:
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Colors:
    GRAY_80: Color = (187, 188, 188)
    GRAY_50: Color = (136, 139, 141)
    GRAY_30: Color = (83, 86, 90)
    BLACK: Color = (64, 64, 64)
    DEFAULT_BROWN: Color = (176, 124, 79)
    BLUE: Color = (30, 75, 217)
    RED: Color = (161, 0, 0)
    BEIGE: Color = (202, 190, 153)
    SILVER: Color = (134, 143, 152)
    LIGHT_GOLD: Color = (152, 144, 101)
    YELLOW_GOLD: Color = (255, 215, 0)
    SKY: Color = (115, 215, 255)
    EMERALD: Color = (2, 138, 15)
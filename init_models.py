import torch
from typing import Any, Callable, NamedTuple, Optional, Union as TUnion, Type, TypeVar
from functools import partial, wraps
import numpy as np

from rpcad.csg import (
    Model,
    Union,
    Cube,
    CylinderX,
    CylinderY,
    CylinderZ,
    Xfrm,
)
from rpcad.typing import Device, Color
from rpcad.defaults import Colors, Default
from rpcad.util import get_random_colors


INIT_MODELS = {}

T = TypeVar("T")


def register(name: Optional[str] = None):
    def decorator(func):
        aname = name if name is not None else func.__name__
        INIT_MODELS[aname] = func
        return func

    return decorator


def partial_ex(func: Callable[..., T], **kwargs: Callable) -> Callable[..., T]:
    """
    Returns a wrapper function that transforms arguments.
    """

    @wraps(func)
    def wrapper(**kwargs_) -> T:
        new_kwargs = {k: kwargs.get(k, lambda x: x)(v) for k, v in kwargs_.items()}
        return func(**new_kwargs)

    return wrapper


def register_detached(name: str):
    def decorator(func):
        @wraps(func)
        def org(options: InitOptions, **kwargs):
            return func(
                options=InitOptions(
                    options.device,
                    options.requires_grad,
                    with_rot=False,
                    rand_color=False,
                ),
                **kwargs,
            )

        @wraps(func)
        def rot(options: InitOptions, **kwargs):
            return func(
                options=InitOptions(
                    options.device,
                    options.requires_grad,
                    with_rot=True,
                    rand_color=False,
                ),
                **kwargs,
            )

        @wraps(func)
        def col(options: InitOptions, **kwargs):
            return func(
                options=InitOptions(
                    options.device,
                    options.requires_grad,
                    with_rot=False,
                    rand_color=True,
                ),
                **kwargs,
            )

        @wraps(func)
        def rot_col(options: InitOptions, **kwargs):
            return func(
                options=InitOptions(
                    options.device,
                    options.requires_grad,
                    with_rot=True,
                    rand_color=True,
                ),
                **kwargs,
            )

        INIT_MODELS[name] = org
        INIT_MODELS[name + "_rot"] = rot
        INIT_MODELS[name + "_col"] = col
        INIT_MODELS[name + "_rot_col"] = rot_col
        return func

    return decorator


class InitOptions(NamedTuple):
    device: Device
    requires_grad: bool
    with_rot: bool
    rand_color: bool


_Model = TypeVar("_Model", bound=Model)


def _init_detach_xfrm(
    xfrms: list[Xfrm],
    names: list[str],
    primitives: TUnion[Type[Model], list[Type[Model]]],
    colors: TUnion[Color, list[Color]],
    options: InitOptions,
) -> Model:
    N = len(xfrms)
    learnable = ["t", "s"]
    if options.with_rot:
        learnable.append("r")
    if not isinstance(primitives, list):
        primitives = [primitives] * N

    if options.rand_color:
        colors = get_random_colors(N)
    else:
        if not isinstance(colors, list):
            colors = [colors] * N

    prims = [
        Prim(
            name=name,
            trs=xfrm.to_trs(),
            color=color,
            learnable=learnable,
            requires_grad=options.requires_grad,
        )
        for xfrm, name, Prim, color in zip(xfrms, names, primitives, colors)
    ]

    return Union(prims)


def _init_detach_o_s(
    o: torch.Tensor,
    s: torch.Tensor,
    names: list[str],
    primitives: TUnion[Type[Model], list[Type[Model]]],
    colors: TUnion[Color, list[Color]],
    options: InitOptions,
) -> Model:
    xfrms = [Xfrm.device(options.device).from_o_s(o_, s_) for o_, s_ in zip(o, s)]

    return _init_detach_xfrm(xfrms, names, primitives, colors, options)


@register_detached("chair_arm")
def _chair_arm(
    options: InitOptions,
    width=1.0,
    length=1.0,
    top_thickness=0.15,
    thickness=0.15,
    height=0.7,
    arm_height=0.3,
) -> Model:
    o = torch.tensor(
        [
            [-width / 2, -height, length / 2 - thickness],
            [width / 2 - thickness, -height, length / 2 - thickness],
            [-width / 2, -height, -length / 2],
            [width / 2 - thickness, -height, -length / 2],
            [-width / 2, 0.0, -length / 2],
            [-width / 2, top_thickness, -length / 2],
            [-width / 2, top_thickness, -length / 2 + thickness],
            [width / 2 - thickness, top_thickness, -length / 2 + thickness],
        ],
        device=options.device,
    )

    s = torch.tensor(
        [
            [thickness, height, thickness],
            [thickness, height, thickness],
            [thickness, height, thickness],
            [thickness, height, thickness],
            [width, top_thickness, length],
            [width, height, thickness],
            [thickness, arm_height, length - thickness],
            [thickness, arm_height, length - thickness],
        ],
        device=options.device,
    )

    names = [
        "(left_front_leg)",
        "(right_front_leg)",
        "(left_back_leg)",
        "(right_back_leg)",
        "(seat)",
        "(back)",
        "(left_arm)",
        "(right_arm)",
    ]

    return _init_detach_o_s(o, s, names, Cube, Colors.DEFAULT_BROWN, options)


@register_detached("car")
def _car(
    options: InitOptions,
    width=1.0,
    height=0.5,
    depth=1.5,
    elev=0.15,
    thickness=0.1,
) -> Model:
    total_height = elev + height
    o = torch.tensor(
        [
            [-width / 2, -total_height / 2 + elev, -depth / 2],
            [-width / 2, -total_height / 2 + elev, -depth / 3],
            [-width / 2, -total_height / 2 + elev, depth / 6],
            [-width / 2 - thickness, -total_height / 2, -depth / 2 + thickness],
            [width / 2, -total_height / 2, -depth / 2 + thickness],
            [
                -width / 2 - thickness,
                -total_height / 2,
                depth / 2 - thickness - elev * 2,
            ],
            [width / 2, -total_height / 2, depth / 2 - thickness - elev * 2],
        ],
        device=options.device,
    )

    s = torch.tensor(
        [
            [width, height * 0.6, depth / 6],
            [width, height, depth / 2],
            [width, height * 0.6, depth / 3],
            [thickness, elev * 2, elev * 2],
            [thickness, elev * 2, elev * 2],
            [thickness, elev * 2, elev * 2],
            [thickness, elev * 2, elev * 2],
        ],
        device=options.device,
    )

    names = [
        "(back_chassis)",
        "(mid_chassis)",
        "(front_chassis)",
        "(left_back_wheel)",
        "(right_back_wheel)",
        "(left_front_wheel)",
        "(right_front_wheel)",
    ]

    prims: list[Type[Model]] = [
        Cube,
        Cube,
        Cube,
        CylinderX,
        CylinderX,
        CylinderX,
        CylinderX,
    ]

    colors = [*([Colors.GRAY_30] * 3), *([Colors.GRAY_50] * 4)]

    return _init_detach_o_s(o, s, names, prims, colors, options)


@register_detached("table")
def _table(
    options: InitOptions,
    width=1.0,
    height=0.8,
    depth=1.0,
    thickness=0.1,
    overhang=0.05,
    elev=0.35,
) -> Model:
    o = torch.tensor(
        [
            [-width / 2, height / 2 - thickness, -depth / 2],
            [-width / 2 + overhang, -height / 2 + elev, -depth / 2 + overhang],
            [-width / 2 + overhang, -height / 2, -depth / 2 + overhang],
            [width / 2 - thickness - overhang, -height / 2, -depth / 2 + overhang],
            [-width / 2 + overhang, -height / 2, depth / 2 - thickness - overhang],
            [
                width / 2 - thickness - overhang,
                -height / 2,
                depth / 2 - thickness - overhang,
            ],
        ],
        device=options.device,
    )

    s = torch.tensor(
        [
            [width, thickness, depth],
            [width - overhang * 2, thickness, depth - 2 * overhang],
            [thickness, height - thickness, thickness],
            [thickness, height - thickness, thickness],
            [thickness, height - thickness, thickness],
            [thickness, height - thickness, thickness],
        ],
        device=options.device,
    )

    names = [
        "(table_top)",
        "(table_middle)",
        "(left_back_leg)",
        "(right_back_leg)",
        "(left_front_leg)",
        "(right_front_leg)",
    ]

    return _init_detach_o_s(o, s, names, Cube, Colors.DEFAULT_BROWN, options)


@register_detached("camera")
def _camera(
    options: InitOptions,
    width=1.0,
    height=0.6,
    body_depth=0.2,
    grip_width=0.25,
    grip_depth=0.2,
    lens_r=0.25,
    lens_depth=0.5,
    flash_width=0.3,
    flash_height=0.1,
) -> Model:
    device = options.device
    xfrms = [
        Xfrm.device(device).from_o_s(
            [-width / 2, -height / 2, -0.5], [width, height, body_depth]
        ),
        Xfrm.device(device).from_o_s(
            [-width / 2, -height / 2, -0.5 + body_depth],
            [grip_width, height, grip_depth],
        ),
        Xfrm.device(device).from_o_s(
            [grip_width / 2 - lens_r, -lens_r, -0.5 + body_depth],
            [lens_r * 2, lens_r * 2, lens_depth],
        ),
        Xfrm.device(device).from_o_s(
            [grip_width / 2 - flash_width / 2, height / 2, -0.5],
            [flash_width, flash_height, body_depth],
        ),
    ]
    prims = [
        Cube,
        Cube,
        CylinderZ,
        Cube,
    ]
    names = ["body", "grip", "lens", "flash"]
    colors = [Colors.BLACK, Colors.GRAY_30, Colors.GRAY_50, Colors.GRAY_30]

    return _init_detach_xfrm(xfrms, names, prims, colors, options)


@register_detached("bottle")
def _bottle(
    options: InitOptions,
    width=0.4,
    height=1,
    height0=0.4,
    height1=0.3,
    width0=0.1,
    width1=0.2,
    width2=0.05,
):
    I = Xfrm.device(options.device)
    xfrms = [
        I.from_o_s([-width / 2, -height / 2, -width / 2], [width, height0, width]),
        I.from_o_s(
            [-width / 2, -height / 2 + height0, -width / 2], [width, height1, width]
        ),
        I.from_o_s(
            [-width1 / 2, -height / 2 + height0 + height1, -width1 / 2],
            [width1, 0.2, width1],
        ),
    ]
    prims = [
        CylinderY,
        partial_ex(partial(CylinderY, r_top=0.2), learnable=lambda x: [*x, "r_top"]),
        CylinderY,
    ]
    names = ["body", "body2", "head"]
    colors = [
        Colors.EMERALD,
        Colors.EMERALD,
        Colors.GRAY_50,
        Colors.GRAY_50,
        Colors.GRAY_50,
    ]
    return _init_detach_xfrm(xfrms, names, prims, colors, options)

def initialize_model(name: str, device: Optional[Device] = None, requires_grad: bool = True, **kwargs) -> Model:
    if device is None:
        device = Default.DEVICE
    return INIT_MODELS[name](options=InitOptions(device, requires_grad, False, False), **kwargs)

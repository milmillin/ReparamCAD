import torch
import torch.nn.functional as F
import itertools
import sympy as sp
from typing import NamedTuple
from dataclasses import dataclass

from .typing import Device
from .csg import Cube, CylinderX, CylinderY, CylinderZ, Sphere, Model, ParamMap


def compute_null_space(constraints: torch.Tensor) -> torch.Tensor:
    """
    Compute the nullspace of the constraints matrix

    constraints: (n-constraints, n-params)

    returns: (n-nullspace-dims, n-params)
    """
    return torch.tensor(
        sp.Matrix(constraints.cpu().numpy()).nullspace(),
        dtype=torch.float32,
        device=constraints.device,
    )


def project_to_null_space(params: torch.Tensor, null_space: torch.Tensor) -> torch.Tensor:
    """
    Project o_s to null_space

    flattened_o_s: (n-params) or (n-models, n-params)
    null_space: (n-nullspace-dims, n-params)

    returns: (1 or n-models, n-params)
    """
    if params.dim() == 1:
        params = params.unsqueeze(0)
    if null_space.shape[1] == params.shape[1] + 1:
        params = F.pad(params, (0, 1), "constant", 1)
    elif null_space.shape[1] != params.shape[1]:
        raise ValueError(f"Shape mismatch {null_space.shape[1]} != {params.shape[1]}")

    return torch.linalg.lstsq(null_space.T.unsqueeze(0), params.unsqueeze(-1)).solution.squeeze(-1)


def project_to_o_s(ns_params: torch.Tensor, null_space: torch.Tensor) -> torch.Tensor:
    """
    Expand to null space params to o_s

    ns_params: (n-models, n-nullspace-dims)
    null_space: (n-nullspace-dims, n-params)

    returns: (n-models, n-params - 1)
    """
    return (ns_params @ null_space)[:, :-1]


class ConstraintSet(NamedTuple):
    equations: torch.Tensor  # (n_constraint_groups, M, n_params)
    names: list[str]

    def to(self, device: Device) -> "ConstraintSet":
        return ConstraintSet(self.equations.to(device), self.names)


def pad_constraint(c: torch.Tensor, M: int) -> torch.Tensor:
    """
    To make constraints easier to handle we want them to all have the same number
    of linear constraint equations. This is achieved by adding trivial all-zero
    rows as padding

    c: (a, b)
    returns: (M, b)
    """
    return torch.nn.functional.pad(c, (0, 0, 0, M - c.shape[0]), mode="constant", value=0)


def pvar(i: int, N: int) -> torch.Tensor:
    """
    Make a parameter variable

    returns: (N,)
    """
    return torch.nn.functional.one_hot(torch.tensor(i), N)


def generate_coplanarity_constraints(
    primitives: ParamMap, parameter_count: int
) -> tuple[list[torch.Tensor], list[str]]:
    coplanarities = []
    names = []

    x_planes = []
    y_planes = []
    z_planes = []

    for p in primitives:
        name = p[0].name
        pos = [pvar(i, parameter_count) for i in p[1]["t"]]
        dim = [pvar(i, parameter_count) for i in p[1]["s"]]
        if type(p[0]) == Cube:
            x_planes.append((pos[0] - dim[0] / 2, f"left_face({name})"))
            x_planes.append((pos[0] + dim[0] / 2, f"right_face({name})"))
            y_planes.append((pos[1] - dim[1] / 2, f"bottom_face({name})"))
            y_planes.append((pos[1] + dim[1] / 2, f"top_face({name})"))
            z_planes.append((pos[2] - dim[2] / 2, f"back_face({name})"))
            z_planes.append((pos[2] + dim[2] / 2, f"front_face({name})"))
        elif type(p[0]) == CylinderX:
            x_planes.append((pos[0] - dim[0] / 2, f"left_face({name})"))
            x_planes.append((pos[0] + dim[0] / 2, f"right_face({name})"))
        elif type(p[0]) == CylinderY:
            y_planes.append((pos[1] - dim[1] / 2, f"bottom_face({name})"))
            y_planes.append((pos[1] + dim[1] / 2, f"top_face({name})"))
        elif type(p[0]) == CylinderZ:
            z_planes.append((pos[2] - dim[2] / 2, f"back_face({name})"))
            z_planes.append((pos[2] + dim[2] / 2, f"front_face({name})"))
        elif type(p[0]) == Sphere:
            pass  # no planes for a sphere

    for planes in [x_planes, y_planes, z_planes]:
        for (p1, n1), (p2, n2) in itertools.combinations(planes, 2):
            coplanarities.append(torch.stack([p1 - p2]))
            names.append(f"coplanar({n1},{n2})")

    return coplanarities, names


def generate_coaxial_constraints(primitives: ParamMap, parameter_count: int) -> tuple[list[torch.Tensor], list[str]]:
    x_axes = []
    y_axes = []
    z_axes = []
    for p in primitives:
        name = p[0].name
        pos = [pvar(i, parameter_count) for i in p[1]["t"]]
        dim = [pvar(i, parameter_count) for i in p[1]["s"]]

        if type(p[0]) == Cube:
            x_axes.append(((pos[1], pos[2]), f"x_axis({name})"))
            y_axes.append(((pos[0], pos[2]), f"y_axis({name})"))
            z_axes.append(((pos[0], pos[1]), f"z_axis({name})"))
        elif type(p[0]) == CylinderX:
            x_axes.append(((pos[1], pos[2]), f"axis({name})"))
        elif type(p[0]) == CylinderY:
            y_axes.append(((pos[0], pos[2]), f"axis({name})"))
        elif type(p[0]) == CylinderZ:
            z_axes.append(((pos[0], pos[1]), f"axis({name})"))
        elif type(p[0]) == Sphere:
            pass  # no axes for a sphere

    constraints = []
    names = []
    for axes in [x_axes, y_axes, z_axes]:
        for ((a1x, a1y), n1), ((a2x, a2y), n2) in itertools.combinations(axes, 2):
            name = f"coaxial({n1}, {n2})"
            eqns = []
            eqns.append(a1x - a2x)
            eqns.append(a1y - a2y)
            names.append(name)
            constraints.append(torch.stack(eqns))
    return constraints, names


def generate_point_constraints(primitives: ParamMap, parameter_count: int) -> tuple[list[torch.Tensor], list[str]]:
    # axis points
    points = []

    for p in primitives:
        name = p[0].name
        pos = [pvar(i, parameter_count) for i in p[1]["t"]]
        dim = [pvar(i, parameter_count) for i in p[1]["s"]]

        if type(p[0]) == Cube:
            x_names = ["left", "right"]
            y_names = ["bottom", "top"]
            z_names = ["back", "front"]
            for dx, dy, dz in itertools.product(*[[-1, 1]] * 3):
                nx = x_names[int((dx + 1) / 2)]
                ny = y_names[int((dy + 1) / 2)]
                nz = z_names[int((dz + 1) / 2)]
                point_name = f"{ny}_{nz}_{nx}({name})"
                points.append(
                    (
                        (
                            pos[0] + dx * dim[0] / 2,
                            pos[1] + dy * dim[1] / 2,
                            pos[2] + dz * dim[2] / 2,
                        ),
                        point_name,
                    )
                )
        elif type(p[0]) == CylinderX:
            points.append(((pos[0] + dim[0] / 2, pos[1], pos[2]), f"top_point({name})"))
            points.append(((pos[0] - dim[0] / 2, pos[1], pos[2]), f"bottom_point({name})"))
        elif type(p[0]) == CylinderY:
            points.append(((pos[0], pos[1] + dim[1] / 2, pos[2]), f"top_point({name})"))
            points.append(((pos[0], pos[1] - dim[1] / 2, pos[2]), f"bottom_point({name})"))
        elif type(p[0]) == CylinderZ:
            points.append(((pos[0], pos[1], pos[2] + dim[2] / 2), f"top_point({name})"))
            points.append(((pos[0], pos[1], pos[2] - dim[2] / 2), f"bottom_point({name})"))
        elif type(p[0]) == Sphere:
            points.append(((pos[0], pos[1], pos[2]), f"center({name})"))
    constraints = []
    names = []
    for ((x1, y1, z1), n1), ((x2, y2, z2), n2) in itertools.combinations(points, 2):
        eqns = []
        eqns.append(x1 - x2)
        eqns.append(y1 - y2)
        eqns.append(z1 - z2)
        constraints.append(torch.stack(eqns))
        names.append(f"coincident({n1}, {n2})")

    return constraints, names


def generate_dim_constraints(primitives: ParamMap, parameter_count: int) -> tuple[list[torch.Tensor], list[str]]:
    dims = []
    for p in primitives:
        name = p[0].name
        pos = [pvar(i, parameter_count) for i in p[1]["t"]]
        dim = [pvar(i, parameter_count) for i in p[1]["s"]]

        for i, n in zip(range(3), ["width", "height", "depth"]):
            dims.append((dim[i], f"{n}({name})"))

    constraints = []
    names = []
    for (d1, n1), (d2, n2) in itertools.combinations(dims, 2):
        constraints.append(torch.stack([d1 - d2]))
        names.append(f"equal({n1}, {n2})")
    return constraints, names


def generate_constraints(model: Model, device: Device = None) -> ConstraintSet:
    if device is None:
        device = model.device
    parameter_definitions = model.param_map()[1:]
    base_parameters = model.parameters()
    parameter_count = model.pack_params(base_parameters).shape[0]

    coplanarities, coplanarity_names = generate_coplanarity_constraints(parameter_definitions, parameter_count)
    coaxialities, coaxiality_names = generate_coaxial_constraints(parameter_definitions, parameter_count)
    coincidences, coincidence_names = generate_point_constraints(parameter_definitions, parameter_count)
    equalities, equality_names = generate_dim_constraints(parameter_definitions, parameter_count)

    constraints = coplanarities + coaxialities + coincidences + equalities
    constraint_names = coplanarity_names + coaxiality_names + coincidence_names + equality_names

    constraint_size = max([c.shape[0] for c in constraints])

    constraints = torch.stack([pad_constraint(c, constraint_size) for c in constraints]).float()

    return ConstraintSet(constraints, constraint_names)


def generate_filtered_constraints(model: Model, device: Device = None) -> ConstraintSet:
    if device is None:
        device = model.device
    constraints = generate_constraints(model)
    base_parameters = model.pack_params(model.parameters()).detach().cpu().view((-1, 1))
    # Matrix Multiple is Broadcast over the individual constraints
    constraint_filter = ((constraints.equations @ base_parameters).abs() < 1e-6).all(dim=1).flatten()
    filtered_equations = constraints.equations[constraint_filter]
    filtered_names = [constraints.names[i] for i in torch.arange(len(constraint_filter))[constraint_filter]]
    return ConstraintSet(filtered_equations, filtered_names)

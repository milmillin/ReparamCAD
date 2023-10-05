import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import NamedTuple, Union as TUnion, Optional, Generator, overload
import math
import itertools
from abc import ABC
import sympy as sp
import trimesh

from .typing import Device, Color
from .defaults import Colors, Default


def _trans_to_matrix(trans: torch.Tensor) -> torch.Tensor:
    T = torch.eye(4, dtype=torch.float32, device=trans.device)
    T[:3, 3] = trans
    return T


def _rot_trans_to_matrix(av: torch.Tensor, r: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    i, j, k = torch.unbind(av)
    device = av.device

    return torch.stack(
        [
            1.0 - 2.0 * (j * j + k * k),
            2.0 * (i * j - k * r),
            2.0 * (i * k + j * r),
            trans[0],
            2.0 * (i * j + k * r),
            1.0 - 2.0 * (i * i + k * k),
            2.0 * (j * k - i * r),
            trans[1],
            2.0 * (i * k - j * r),
            2.0 * (j * k + i * r),
            1.0 - 2.0 * (i * i + j * j),
            trans[2],
            torch.tensor(0.0, device=device),
            torch.tensor(0.0, device=device),
            torch.tensor(0.0, device=device),
            torch.tensor(1.0, device=device),
        ]
    ).reshape((4, 4))


def _scale_matrix(s: torch.Tensor) -> torch.Tensor:
    S = torch.eye(4, dtype=torch.float32, device=s.device)
    S[0, 0] = s[0]
    S[1, 1] = s[1]
    S[2, 2] = s[2]
    return S


def _matrix_to_rot(m: torch.Tensor) -> torch.Tensor:
    if m[2][2] < 0:
        if m[0][0] > m[1][1]:
            t = 1 + m[0][0] - m[1][1] - m[2][2]
            q = [t, m[0][1] + m[1][0], m[2][0] + m[0][2], m[1][2] - m[2][1]]
        else:
            t = 1 - m[0][0] + m[1][1] - m[2][2]
            q = [m[0][1] + m[1][0], t, m[1][2] + m[2][1], m[2][0] - m[0][2]]
    else:
        if m[0][0] < -m[1][1]:
            t = 1 - m[0][0] - m[1][1] + m[2][2]
            q = [m[2][0] + m[0][2], m[1][2] + m[2][1], t, m[0][1] - m[1][0]]
        else:
            t = 1 + m[0][0] + m[1][1] + m[2][2]
            q = [m[1][2] - m[2][1], m[2][0] - m[0][2], m[0][1] - m[1][0], t]
    q = torch.tensor(q, device=m.device)
    q *= 0.5 / torch.sqrt(t)

    v = torch.nn.functional.normalize(q[:3], dim=0)
    theta = 2 * torch.acos(-q[3])
    return theta * v


class Xfrm(NamedTuple):
    T: torch.Tensor
    T_inv: torch.Tensor

    def __matmul__(self, B: TUnion["Xfrm", "TRS"]):
        if isinstance(B, Xfrm):
            return Xfrm(self.T @ B.T, self.T_inv @ B.T_inv)
        else:
            return self @ B.to_xfrm()

    def to(self, device: Device) -> "Xfrm":
        return Xfrm(self.T.to(device), self.T_inv.to(device))

    def get_device(self) -> Device:
        return self.T.device

    def transform_point(self, V: torch.Tensor) -> torch.Tensor:
        V_h = F.pad(V, (0, 1), value=1)
        # V * T_fws.T
        return torch.matmul(V_h, self.T.T)[:, :-1]

    def transform_normal(self, N: torch.Tensor) -> torch.Tensor:
        # WARNING: this is not normalized
        return torch.matmul(N, self.T_inv[:3, :3])

    def transform_plane(self, plane: torch.Tensor) -> torch.Tensor:
        return torch.matmul(plane, self.T_inv)

    def transform_vector(self, vector: torch.Tensor) -> torch.Tensor:
        return torch.matmul(vector, self.T[:3, :3])

    def to_trs(self) -> "TRS":
        scale = self.T[:, :3].norm(dim=0)
        rot_mat = self.T[:3, :3] / scale.unsqueeze(0)
        rot = _matrix_to_rot(rot_mat)
        trans = self.T[:3, 3]
        return TRS(trans, rot, scale)

    @classmethod
    def device(cls, device: Device = None) -> "Xfrm":
        if device is None:
            device = Default.DEVICE
        return Xfrm(
            torch.eye(4, dtype=torch.float32, device=device),
            torch.eye(4, dtype=torch.float32, device=device),
        )

    def translate(self, trans: TUnion[list[float], tuple[float, float, float], torch.Tensor]) -> "Xfrm":
        trans = torch.as_tensor(trans, dtype=torch.float32, device=self.T.device)
        return Xfrm(_trans_to_matrix(trans), _trans_to_matrix(-trans)) @ self

    def rigid(
        self,
        rot: TUnion[list[float], tuple[float, float, float], torch.Tensor],
        trans: TUnion[list[float], tuple[float, float, float], torch.Tensor],
    ) -> "Xfrm":
        rot = torch.as_tensor(rot, dtype=torch.float32, device=self.T.device)
        trans = torch.as_tensor(trans, dtype=torch.float32, device=self.T.device)
        v = rot
        theta = torch.norm(v)

        # Convert to quaternion
        # q = [av, r]
        a = torch.sinc(theta / 2.0 / math.pi) / 2.0
        r = torch.cos(theta / 2.0)

        # Convert quaternion to matrix
        o = _rot_trans_to_matrix(a * v, r, trans)
        o_inv = _rot_trans_to_matrix(-a * v, r, -trans)
        return Xfrm(o, o_inv) @ self

    def rotate(
        self,
        rot: TUnion[list[float], tuple[float, float, float], torch.Tensor],
    ) -> "Xfrm":
        return self.rigid(rot, torch.zeros((3), device=self.T.device))

    def scale(
        self,
        scale: TUnion[list[float], tuple[float, float, float], torch.Tensor],
    ) -> "Xfrm":
        scale = torch.as_tensor(scale, dtype=torch.float32, device=self.T.device)
        return Xfrm(_scale_matrix(scale), _scale_matrix(1 / scale)) @ self

    def from_o_s(
        self,
        o: TUnion[list[float], tuple[float, float, float], torch.Tensor],
        s: TUnion[list[float], tuple[float, float, float], torch.Tensor],
    ) -> "Xfrm":
        return self.translate([0.5, 0.5, 0.5]).scale(s).translate(o)

    def from_x1_x2(
        self,
        x1: TUnion[list[float], tuple[float, float, float], torch.Tensor],
        x2: TUnion[list[float], tuple[float, float, float], torch.Tensor],
    ) -> "Xfrm":
        x1 = torch.as_tensor(x1, dtype=torch.float32, device=self.T.device)
        x2 = torch.as_tensor(x2, dtype=torch.float32, device=self.T.device)
        d = (x2 - x1).norm(dim=0)
        ax = F.normalize(x2 - x1, dim=0)
        y = torch.tensor([0.0, 1.0, 0.0], device=self.T.device)
        v = F.normalize(y.cross(ax), dim=0)
        theta = torch.acos(y.dot(ax))
        rot = theta * v
        scale = torch.ones((3), dtype=torch.float32, device=self.T.device)
        scale[1] = d
        return self.translate([0, 0.5, 0]).scale(scale).rotate(rot).translate(x1)

    def y_to_x(self) -> "Xfrm":
        return self.rotate([0, 0, -math.pi / 2])

    def y_to_z(self) -> "Xfrm":
        return self.rotate([math.pi / 2, 0, 0])


@dataclass
class TRS:
    translate: torch.Tensor
    rotate: torch.Tensor
    scale: torch.Tensor

    def to_xfrm(self) -> Xfrm:
        return Xfrm.device(self.translate.device).scale(self.scale).rigid(self.rotate, self.translate)

    def __matmul__(self, B: TUnion[Xfrm, "TRS"]) -> "TRS":
        return (self.to_xfrm() @ B).to_trs()

    def get_device(self) -> Device:
        return self.translate.device

    @classmethod
    def identity(cls, device: Device = None):
        if device is None:
            device = Default.DEVICE
        return TRS(
            translate=torch.zeros((3), dtype=torch.float32, device=device),
            rotate=torch.zeros((3), dtype=torch.float32, device=device),
            scale=torch.ones((3), dtype=torch.float32, device=device),
        )


@dataclass
class Mesh:
    V: torch.Tensor
    F: torch.Tensor
    N: torch.Tensor
    color: torch.Tensor

    def cuda(self) -> "Mesh":
        return Mesh(self.V.cuda(), self.F.cuda(), self.N.cuda(), self.color.cuda())

    def pointcloud(self, N: int) -> torch.Tensor:
        tris = self.V[self.F.long()]
        v1 = tris[:, 0, :]
        v2 = tris[:, 1, :]
        v3 = tris[:, 2, :]
        N = 1024
        tri_areas = 0.5 * torch.linalg.norm(torch.linalg.cross(v2 - v1, v3 - v1), dim=1)
        probabilities = tri_areas / tri_areas.sum()
        sample_tri_indices = torch.multinomial(probabilities, N, True)

        v1_sample = v1[sample_tri_indices]
        v2_sample = v2[sample_tri_indices]
        v3_sample = v3[sample_tri_indices]

        u = torch.rand((N, 1), device=tris.device)
        v = torch.rand((N, 1), device=tris.device)
        out_of_bounds = u + v > 1
        u[out_of_bounds] = 1 - u[out_of_bounds]
        v[out_of_bounds] = 1 - v[out_of_bounds]
        w = 1 - (u + v)

        points = v1_sample * u + v2_sample * v + w * v3_sample
        return points

    def transform(self, trans: Xfrm) -> "Mesh":
        return Mesh(
            trans.transform_point(self.V).contiguous(),
            self.F,
            trans.transform_normal(self.N),
            self.color,
        )

    def get_normalize_xfrm(self, detach: bool = True) -> Xfrm:
        min_ = self.V.min(dim=0).values
        max_ = self.V.max(dim=0).values
        if detach:
            min_ = min_.detach().clone()
            max_ = max_.detach().clone()
        center = (min_ + max_) / 2
        scale = (max_ - min_).max()
        return Xfrm.device(self.V.device).translate(-center).scale(1 / scale.repeat(3))

    @classmethod
    def union(cls, meshes: list["Mesh"]) -> "Mesh":
        V = torch.cat([mesh.V for mesh in meshes], dim=0)
        N = torch.cat([mesh.N for mesh in meshes], dim=0)
        color = torch.cat([mesh.color for mesh in meshes], dim=0)

        F = []
        offset = 0
        for mesh in meshes:
            F.append(mesh.F + offset)
            offset += mesh.V.shape[0]
        F = torch.cat(F, dim=0)
        return Mesh(V, F, N, color)

    @classmethod
    def empty(cls, device: Device) -> "Mesh":
        return Mesh(
            torch.empty((0, 3), dtype=torch.float32, device=device),
            torch.empty((0, 3), dtype=torch.int32, device=device),
            torch.empty((0, 3), dtype=torch.float32, device=device),
            torch.empty((0, 3), dtype=torch.float32, device=device),
        )

    @classmethod
    def from_trimesh(
        cls,
        trimesh: trimesh.Trimesh,
        device: Device = None,
        color: Color = Colors.DEFAULT_BROWN,
    ) -> "Mesh":
        if device is None:
            device = Default.DEVICE
        return Mesh(
            torch.tensor(trimesh.vertices, dtype=torch.float32, device=device),
            torch.tensor(trimesh.faces, dtype=torch.int32, device=device),
            torch.tensor(trimesh.vertex_normals, dtype=torch.float32, device=device),
            (torch.tensor([color], dtype=torch.float32, device=device) / 255).repeat(len(trimesh.faces), 1),
        )

    def to_trimesh(self) -> trimesh.Trimesh:
        return trimesh.Trimesh(self.V.detach().cpu().numpy(), self.F.detach().cpu().numpy())

    def volume(self) -> torch.Tensor:
        return torch.linalg.det(self.V[self.F.long()]).sum() / 6  # (n-f,)


@dataclass
class KeyPoints:
    V: torch.Tensor
    V_names: list[str]
    plane: torch.Tensor  # (.., 4)
    plane_names: list[str]
    axis: torch.Tensor  # (..., 6); (origins; dir)
    axis_names: list[str]
    dims: torch.Tensor  # (..., 3); vector
    dims_names: list[str]

    def transform(self, trans: Xfrm) -> "KeyPoints":
        return KeyPoints(
            trans.transform_point(self.V),
            self.V_names,
            trans.transform_plane(self.plane),
            self.plane_names,
            trans.transform_point(self.axis[:, :3]).row_join(  # type: ignore
                trans.transform_vector(self.axis[:, 3:])  # type: ignore
            ),
            self.axis_names,
            trans.transform_vector(self.dims),
            self.dims_names,
        )


class Parameters:
    _learnable_params_idx: list[int]
    _all_params: list[torch.Tensor]
    _names: list[str]
    _learnable: list[bool]
    _learnable_params_info: list[tuple[int, torch.Size]]
    _requires_grad: bool

    def __init__(self, requires_grad: bool):
        self._learnable_params_idx = []
        self._all_params = []
        self._names = []
        self._learnable = []
        self._learnable_params_info = []
        self._requires_grad = requires_grad

    def register(self, name: str, p: torch.Tensor, learnable: bool):
        if learnable:
            self._learnable_params_idx.append(len(self._all_params))
            self._learnable_params_info.append((p.numel(), p.size()))
            if self._requires_grad:
                p = p.clone().detach().requires_grad_(True)
        self._all_params.append(p)
        self._names.append(name)
        self._learnable.append(learnable)

    def num_all_params(self) -> int:
        return len(self._all_params)

    def num_learnable_params(self) -> int:
        return len(self._learnable_params_idx)

    def parameters(self) -> list[torch.Tensor]:
        return [self._all_params[idx] for idx in self._learnable_params_idx]

    def parameter_names(self) -> list[str]:
        return [self._names[idx] for idx in self._learnable_params_idx]

    def packed_parameters(self) -> torch.Tensor:
        return torch.cat(
            [self._all_params[idx].flatten() for idx in self._learnable_params_idx],
            dim=0,
        )

    def pack_parameters(self, params: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat([p.flatten() for p in params])

    def unpack_parameters(self, packed: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        res = []
        offset = 0
        for numel, size in self._learnable_params_info:
            res.append(packed[offset : offset + numel].reshape(size))
            offset += numel
        return res, packed[offset:]

    def all_parameters(self) -> list[torch.Tensor]:
        return self._all_params

    def forward(self, params: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(params) != len(self._learnable_params_idx):
            raise ValueError(
                f"Number of parameters doesn't match (got {len(params)} expected {len(self._learnable_params_idx)})"
            )
        res = [*self._all_params]
        for i, idx in enumerate(self._learnable_params_idx):
            res[idx] = params[i]
        return res

    def load_all_parameters(self, params: list[torch.Tensor]):
        if len(params) != len(self._all_params):
            raise ValueError(
                f"Number of parameters doesn't match (got {len(params)} expected {len(self._learnable_params_idx)})"
            )
        self._all_params = params

    def __iter__(self):
        return iter(self._all_params)

    @overload
    def __getitem__(self, idx: int) -> torch.Tensor:
        ...

    @overload
    def __getitem__(self, idx: slice) -> Generator[torch.Tensor, None, None]:
        ...

    def __getitem__(self, idx):
        return self._all_params[idx]

    def pretty(self, sw: int = 0) -> str:
        res = ""
        tabs = " " * sw
        for name, p, learnable in zip(self._names, self._all_params, self._learnable):
            a = "*" if learnable else ""
            res += f"{tabs}{name}{a}: {tuple(p.shape)}\n"
        return res

    def pretty_names(self) -> str:
        res = []
        for name, p, learnable in zip(self._names, self._all_params, self._learnable):
            a = "*" if learnable else ""
            res.append(f"{name}{a}")
        return ", ".join(res)

    def __repr__(self):
        return self.pretty()


ParamMap = list[tuple["Model", dict[str, list[int]]]]


class Model(ABC):
    name: str
    params: Parameters
    device: torch.device
    children: list["Model"]

    _learnable: set[str]
    _color: torch.Tensor
    _mesh: Optional[Mesh]

    def __init__(
        self,
        name: str,
        trs: TUnion[TRS, Xfrm],
        color: Color,
        learnable: list[str],
        requires_grad: bool,
    ):
        if isinstance(trs, Xfrm):
            trs = trs.to_trs()

        self.name = name

        self.params = Parameters(requires_grad)
        self.params.register("t", trs.translate, "t" in learnable)
        self.params.register("r", trs.rotate, "r" in learnable)
        self.params.register("s", trs.scale, "s" in learnable)
        self._learnable = set(learnable)

        self.device = trs.scale.device
        self.children = []

        self._mesh = None
        self._key_points = None
        self._color = torch.tensor([color], dtype=torch.float32, device=self.device) / 255

        # HACK for axis aligned primitives only; for Cylinder/Tube:X/Z
        self._scale_order = [0, 1, 2]
        # volume out of the bounding box
        self.volume_ratio = 0.0

    def _get_transform(self, all_params: list[torch.Tensor]) -> Xfrm:
        t, r, s = all_params[:3]
        return Xfrm.device(self.device).scale(s).rigid(r, t)

    def _generate_mesh(self, all_params: list[torch.Tensor]) -> Mesh:
        assert self._mesh is not None
        return self._mesh

    def generate_mesh(self, params: list[torch.Tensor], normalize: bool = False) -> Mesh:
        cur_params, params = self.strip_parameters(params)
        all_params = self.params.forward(cur_params)
        meshes = [self._generate_mesh(all_params)]
        for chd in self.children:
            cur_params, params = chd.strip_parameters(params)
            meshes.append(chd.generate_mesh(cur_params))
        mesh = Mesh.union(meshes).transform(self._get_transform(all_params))
        if normalize:
            min_ = mesh.V.min(dim=0).values
            max_ = mesh.V.max(dim=0).values
            center = (min_ + max_) / 2
            scale = (max_ - min_).max()
            new_V = (mesh.V - center) / scale
            return Mesh(new_V, mesh.F, mesh.N, mesh.color)
        return mesh

    def _generate_separate_meshes(self, all_params: list[torch.Tensor]) -> list[Mesh]:
        return [self._generate_mesh(all_params)]

    def generate_separate_meshes(self, params: list[torch.Tensor]) -> list[Mesh]:
        cur_params, params = self.strip_parameters(params)
        all_params = self.params.forward(cur_params)
        meshes: list[Mesh] = self._generate_separate_meshes(all_params)
        for chd in self.children:
            cur_params, params = chd.strip_parameters(params)
            meshes.extend(chd.generate_separate_meshes(cur_params))
        xfrm = self._get_transform(all_params)
        meshes = [m.transform(xfrm) for m in meshes]
        return meshes

    def _generate_trimesh(self, all_params: list[torch.Tensor]) -> Optional[trimesh.Trimesh]:
        raise NotImplementedError()

    def generate_trimesh(self, params: list[torch.Tensor]) -> list[trimesh.Trimesh]:
        cur_params, params = self.strip_parameters(params)
        all_params = self.params.forward(cur_params)
        self_mesh = self._generate_trimesh(all_params)
        meshes = [self_mesh] if self_mesh is not None else []
        for chd in self.children:
            cur_params, params = chd.strip_parameters(params)
            meshes.extend(chd.generate_trimesh(cur_params))
        for mesh in meshes:
            mesh.apply_transform(self._get_transform(all_params).T.detach().cpu().numpy())
        return meshes

    def _generate_watertight_mesh(self, all_params: list[torch.Tensor]) -> Mesh:
        raise NotImplementedError()

    def generate_watertight_mesh(self, params: list[torch.Tensor]) -> Mesh:
        cur_params, params = self.strip_parameters(params)
        all_params = self.params.forward(cur_params)
        meshes = [self._generate_watertight_mesh(all_params)]
        for chd in self.children:
            cur_params, params = chd.strip_parameters(params)
            meshes.append(chd.generate_watertight_mesh(cur_params))
        mesh = Mesh.union(meshes).transform(self._get_transform(all_params))
        return mesh

    def generate_names_o_s(self, params: list[torch.Tensor]) -> tuple[list[str], torch.Tensor]:
        raise NotImplementedError("generate_o_s is only implemented for union of cubes.")

    def parameters(self) -> list[torch.Tensor]:
        res = [*self.params.parameters()]
        for chd in self.children:
            res.extend(chd.parameters())
        return res

    def _param_map(self, pre: list[int]) -> ParamMap:
        mapping = {}
        for p, p_name in zip(self.params.parameters(), self.params.parameter_names()):
            sz = p.numel()
            mapping[p_name] = list(range(pre[0], pre[0] + sz))
            # HACK: Permute scale
            if p_name == "s":
                mapping[p_name] = [mapping[p_name][i] for i in self._scale_order]
            pre[0] += sz
        res: list[tuple["Model", dict[str, list[int]]]] = [(self, mapping)]
        for chd in self.children:
            res.extend(chd._param_map(pre))
        return res

    def param_map(self) -> ParamMap:
        return self._param_map([0])

    def parameter_names(self) -> list[str]:
        res = [*self.params.parameter_names()]
        for chd in self.children:
            res.extend(chd.parameter_names())
        return res

    def all_parameters(self) -> list[torch.Tensor]:
        res = [*self.params.all_parameters()]
        for chd in self.children:
            res.extend(chd.params.all_parameters())
        return res

    @overload
    def strip_parameters(self, params: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        ...

    @overload
    def strip_parameters(
        self, params: list[sp.ImmutableMatrix]
    ) -> tuple[list[sp.ImmutableMatrix], list[sp.ImmutableMatrix]]:
        ...

    def strip_parameters(self, params):
        num_params = self.params.num_learnable_params()
        return params[:num_params], params[num_params:]

    def load_all_parameters(self, params: list[torch.Tensor]):
        num_params = self.params.num_all_params()
        self.params.load_all_parameters(params[:num_params])
        params = params[num_params:]
        for chd in self.children:
            num_params = chd.params.num_all_params()
            chd.params.load_all_parameters(params[:num_params])
            params = params[num_params:]

    def pack_params(self, params: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat([p.flatten() for p in params], dim=0)

    def unpack_params(self, packed: torch.Tensor) -> list[torch.Tensor]:
        params, packed = self.params.unpack_parameters(packed)
        for chd in self.children:
            chd_params, packed = chd.params.unpack_parameters(packed)
            params.extend(chd_params)
        return params

    def _repr_impl(self, sw: int) -> str:
        tabs = " " * sw
        res = f'{tabs}<{self.__class__.__name__} name="{self.name}" params=({self.params.pretty_names()})>'
        for chd in self.children:
            res += "\n" + chd._repr_impl(sw + 4)
        return res

    def __repr__(self):
        return self._repr_impl(0)


class Union(Model):
    def __init__(
        self,
        primitives: list[Model],
        trs: TUnion[TRS, Xfrm, None] = None,
        learnable: list[str] = [],
        name: str = "",
        requires_grad: bool = True,
    ):
        if len(primitives) == 0:
            raise ValueError("Cannot create a Union with no primitives")
        trs = TRS.identity(primitives[0].device)
        super().__init__(name, trs, Colors.DEFAULT_BROWN, learnable, requires_grad)
        self.children = [*primitives]
        self._mesh = Mesh.empty(self.device)

    def generate_names_o_s(self, params: list[torch.Tensor]) -> tuple[list[str], torch.Tensor]:
        """
        Return o_s for backward compatibility.
        """
        names = [chd.name for chd in self.children]
        oses = []
        if not all((isinstance(chd, Cube) and "r" not in chd._learnable for chd in self.children)):
            raise NotImplementedError("generate_o_s is only valid for union of axis-aligned cubes.")

        _, params = self.strip_parameters(params)
        for chd in self.children:
            cur_params, params = chd.strip_parameters(params)
            all_params = chd.params.forward(cur_params)
            t = all_params[0]
            s = all_params[2]
            oses.append(-0.5 * s + t)  # o
            oses.append(s)
        return names, torch.cat(oses, dim=0)

    def _generate_trimesh(self, all_params: list[torch.Tensor]) -> Optional[trimesh.Trimesh]:
        return None

    def _generate_separate_meshes(self, all_params: list[torch.Tensor]) -> list[Mesh]:
        return []

    def _generate_watertight_mesh(self, all_params: list[torch.Tensor]) -> Mesh:
        return Mesh.empty(self.device)


### SPHERE ###


def _subdivide(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=0)
    # E_to_F = np.tile(np.arange(len(F)), 3)
    F_to_E = np.arange(len(E)).reshape((3, -1)).T
    EU, E_to_EU = np.unique(np.sort(E, axis=1), axis=0, return_inverse=True)
    new_V = np.concatenate([V, V[EU].mean(axis=1)], axis=0)
    DEF = E_to_EU[F_to_E] + len(V)  # (F, 3)
    FDE = DEF[:, [2, 0, 1]]
    ABC = E[F_to_E][:, :, 0]  # (F, 3)
    new_F = np.concatenate([np.stack([ABC, DEF, FDE], axis=2), DEF[:, np.newaxis, :]], axis=1).reshape((-1, 3))
    return new_V, new_F


# Vertex coordinates for a level 0 ico-sphere.
# https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/utils/ico_sphere.html
_ico_V = np.array(
    [
        [-0.5257, 0.8507, 0.0000],
        [0.5257, 0.8507, 0.0000],
        [-0.5257, -0.8507, 0.0000],
        [0.5257, -0.8507, 0.0000],
        [0.0000, -0.5257, 0.8507],
        [0.0000, 0.5257, 0.8507],
        [0.0000, -0.5257, -0.8507],
        [0.0000, 0.5257, -0.8507],
        [0.8507, 0.0000, -0.5257],
        [0.8507, 0.0000, 0.5257],
        [-0.8507, 0.0000, -0.5257],
        [-0.8507, 0.0000, 0.5257],
    ]
)


# Faces for level 0 ico-sphere
_ico_F = np.array(
    [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ]
)


def _ico_sphere(level: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if level < 0:
        raise ValueError("level must be >= 0.")
    if level == 0:
        return _ico_V, _ico_F
    else:
        V, F = _ico_sphere(level - 1)
        V, F = _subdivide(V, F)
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        return V, F


class Sphere(Model):
    def __init__(
        self,
        name: str,
        trs: TUnion[TRS, Xfrm] = TRS.identity(),
        color: Color = Colors.DEFAULT_BROWN,
        learnable: list[str] = ["t", "r", "s"],
        requires_grad: bool = True,
    ):
        super().__init__(name, trs, color, learnable, requires_grad)
        v, f = _ico_sphere(2)
        V = torch.tensor(v, dtype=torch.float32, device=self.device)
        F = torch.tensor(f, dtype=torch.int32, device=self.device)
        N = V.clone()
        V = V * 0.5
        self._mesh = Mesh(V, F, N, self._color.repeat(F.shape[0], 1))
        self.volume_ratio = torch.pi / 6  # 4/3 pi r^3


### CYLINDER ###

DEFAULT_SEGMENTS = 17


class CylinderY(Model):
    def __init__(
        self,
        name: str,
        trs: TUnion[TRS, Xfrm] = TRS.identity(),
        color: Color = Colors.DEFAULT_BROWN,
        learnable: list[str] = ["t", "r", "s"],
        requires_grad: bool = True,
        r_top: TUnion[float, torch.Tensor] = 0.5,
        segments: int = DEFAULT_SEGMENTS,
    ):
        super().__init__(name, trs, color, learnable, requires_grad)
        r_top = torch.as_tensor(r_top, dtype=torch.float32, device=self.device)
        self.params.register("r_top", r_top, "r_top" in learnable)
        self.segments = segments

        # HACK: ignore rtop for now
        self.volume_ratio = torch.pi / 4  # pi r^2
        # pi [r^2 + (rtop - r)^2 / 3 + r*(rtop - r)]

    def _generate_mesh(self, all_params: list[torch.Tensor]) -> Mesh:
        segments = self.segments
        r_top = all_params[3]
        angles = torch.linspace(0, 2 * torch.pi, segments, device=self.device, dtype=torch.float32)

        V_sides = torch.cat(
            [
                torch.stack(
                    [
                        0.5 * torch.cos(angles),
                        torch.full_like(angles, -0.5, device=self.device),
                        0.5 * torch.sin(angles),
                    ],
                    dim=1,
                ),
                torch.stack(
                    [
                        r_top * torch.cos(angles),
                        torch.full_like(angles, 0.5, device=self.device),
                        r_top * torch.sin(angles),
                    ],
                    dim=1,
                ),
            ],
            dim=0,
        )

        F_sides = torch.tensor(
            [[i + 1, i, i + 1 + segments] for i in range(segments - 1)]
            + [[i, i + segments, i + 1 + segments] for i in range(segments - 1)],
            dtype=torch.int32,
            device=self.device,
        )
        color_sides = self._color.repeat(F_sides.shape[0], 1)

        r_diff = 0.5 - r_top
        hyp = torch.sqrt(r_diff**2 + 1)
        cos_slant = 1 / hyp
        sin_slant = r_diff / hyp

        N_sides = torch.stack(
            [
                torch.cos(angles) * cos_slant,
                sin_slant.repeat(angles.shape),
                torch.sin(angles) * cos_slant,
            ],
            dim=1,
        )
        N_sides = N_sides.repeat(2, 1)

        V_top = torch.stack(
            [
                r_top * torch.cos(angles),
                torch.full_like(angles, 0.5, device=self.device),
                r_top * torch.sin(angles),
            ],
            dim=1,
        )
        F_top = torch.tensor(
            [[0, i + 2, i + 1] for i in range(segments - 3)],
            dtype=torch.int32,
            device=self.device,
        )
        N_top = torch.zeros_like(V_top, device=self.device)
        N_top[:, 1] = 1.0
        color_top = self._color.repeat(F_top.shape[0], 1)

        V_bottom = torch.stack(
            [
                0.5 * torch.cos(angles),
                torch.full_like(angles, -0.5, device=self.device),
                0.5 * torch.sin(angles),
            ],
            dim=1,
        )
        F_bottom = torch.tensor(
            [[0, i + 1, i + 2] for i in range(segments - 3)],
            dtype=torch.int32,
            device=self.device,
        )
        N_bottom = N_top * -1.0
        color_bottom = self._color.repeat(F_bottom.shape[0], 1)

        sides = Mesh(V_sides, F_sides, N_sides, color_sides)
        top = Mesh(V_top, F_top, N_top, color_top)
        bottom = Mesh(V_bottom, F_bottom, N_bottom, color_bottom)
        return Mesh.union([sides, top, bottom])

    def _generate_trimesh(self, all_params: list[torch.Tensor]) -> trimesh.Trimesh:
        segments = self.segments
        r_top = all_params[3].detach().cpu()
        # HACK
        r_top = torch.clamp(r_top, min=1e-4)
        angles = torch.linspace(0, 2 * torch.pi, segments, dtype=torch.float32)[:-1]
        V = torch.cat(
            [
                torch.stack(
                    [
                        0.5 * torch.cos(angles),
                        torch.full_like(angles, -0.5),
                        0.5 * torch.sin(angles),
                    ],
                    dim=1,
                ),
                torch.stack(
                    [
                        r_top * torch.cos(angles),
                        torch.full_like(angles, 0.5),
                        r_top * torch.sin(angles),
                    ],
                    dim=1,
                ),
            ],
            dim=0,
        )

        seg = segments - 1

        F = torch.tensor(
            [[(i + 1) % seg, i, (i + 1) % seg + seg] for i in range(seg)]
            + [[(i + 1) % seg + seg, i, i + seg] for i in range(seg)]
            + [[0, i, i + 1] for i in range(1, seg - 1)]
            + [[seg, seg + i + 1, seg + i] for i in range(1, seg - 1)],
            dtype=torch.int32,
        )
        return trimesh.Trimesh(V.numpy(), F.numpy())

    def _generate_watertight_mesh(self, all_params: list[torch.Tensor]) -> Mesh:
        segments = self.segments
        r_top = all_params[3].detach().cpu()
        # HACK
        r_top = torch.clamp(r_top, min=1e-4)
        angles = torch.linspace(0, 2 * torch.pi, segments, dtype=torch.float32, device=self.device)[:-1]
        V = torch.cat(
            [
                torch.stack(
                    [
                        0.5 * torch.cos(angles),
                        torch.full_like(angles, -0.5),
                        0.5 * torch.sin(angles),
                    ],
                    dim=1,
                ),
                torch.stack(
                    [
                        r_top * torch.cos(angles),
                        torch.full_like(angles, 0.5),
                        r_top * torch.sin(angles),
                    ],
                    dim=1,
                ),
            ],
            dim=0,
        )

        seg = segments - 1

        F = torch.tensor(
            [[(i + 1) % seg, i, (i + 1) % seg + seg] for i in range(seg)]
            + [[(i + 1) % seg + seg, i, i + seg] for i in range(seg)]
            + [[0, i, i + 1] for i in range(1, seg - 1)]
            + [[seg, seg + i + 1, seg + i] for i in range(1, seg - 1)],
            dtype=torch.int32,
            device=self.device,
        )

        # HACK: dont return normals for now
        return Mesh(
            V,
            F,
            torch.empty((0, 3), dtype=torch.float32, device=self.device),
            torch.empty((0, 3), dtype=torch.float32, device=self.device),
        )


class CylinderX(CylinderY):
    def __init__(
        self,
        name: str,
        trs: TUnion[TRS, Xfrm] = TRS.identity(),
        color: Color = Colors.DEFAULT_BROWN,
        learnable: list[str] = ["t", "r", "s"],
        requires_grad: bool = True,
        r_top: TUnion[float, torch.Tensor] = 0.5,
        segments: int = DEFAULT_SEGMENTS,
    ):
        super().__init__(
            name,
            trs @ Xfrm.device(trs.get_device()).y_to_x(),
            color,
            learnable,
            requires_grad,
            r_top,
            segments,
        )
        # HACK
        self._scale_order = [1, 0, 2]


class CylinderZ(CylinderY):
    def __init__(
        self,
        name: str,
        trs: TUnion[TRS, Xfrm] = TRS.identity(),
        color: Color = Colors.DEFAULT_BROWN,
        learnable: list[str] = ["t", "r", "s"],
        requires_grad: bool = True,
        r_top: TUnion[float, torch.Tensor] = 0.5,
        segments: int = DEFAULT_SEGMENTS,
    ):
        super().__init__(
            name,
            trs @ Xfrm.device(trs.get_device()).y_to_z(),
            color,
            learnable,
            requires_grad,
            r_top,
            segments,
        )
        # HACK
        self._scale_order = [0, 2, 1]


### TUBE ###


class TubeY(Model):
    def __init__(
        self,
        name: str,
        trs: TUnion[TRS, Xfrm] = TRS.identity(),
        color: Color = Colors.DEFAULT_BROWN,
        learnable: list[str] = ["t", "r", "s"],
        requires_grad: bool = True,
        r_top: TUnion[float, torch.Tensor] = 0.5,
        thickness: TUnion[float, torch.Tensor] = 0.2,
        segments: int = DEFAULT_SEGMENTS,
    ):
        super().__init__(name, trs, color, learnable, requires_grad)
        r_top = torch.as_tensor(r_top, dtype=torch.float32, device=self.device)
        self.params.register("r_top", r_top, "r_top" in learnable)
        thickness = torch.as_tensor(thickness, dtype=torch.float32, device=self.device)
        self.params.register("thickness", thickness, "thickness" in learnable)
        self.segments = segments

        self.volume_ratio = torch.pi * (thickness.item() - thickness.item() ** 2)  # pi [r^2 - (r - thickness)^2]

    def _generate_mesh(self, all_params: list[torch.Tensor]) -> Mesh:
        r_top = all_params[3]
        thickness = all_params[4]
        segments = self.segments
        angles = torch.linspace(0, 2 * torch.pi, segments, device=self.device, dtype=torch.float32)

        r_diff = 0.5 - r_top
        hyp = torch.sqrt(r_diff**2 + 1)
        cos_slant = 1 / hyp
        sin_slant = r_diff / hyp

        thickness = thickness / cos_slant
        r_inner = 0.5 - thickness
        r_top_inner = r_top - thickness

        V_sides = torch.cat(
            [
                torch.stack(
                    [
                        0.5 * torch.cos(angles),
                        torch.full_like(angles, -0.5, device=self.device),
                        0.5 * torch.sin(angles),
                    ],
                    dim=1,
                ),
                torch.stack(
                    [
                        r_top * torch.cos(angles),
                        torch.full_like(angles, 0.5, device=self.device),
                        r_top * torch.sin(angles),
                    ],
                    dim=1,
                ),
            ],
            dim=0,
        )
        F_sides = torch.tensor(
            [[i, i + 1, i + 1 + segments] for i in range(segments - 1)]
            + [[i, i + 1 + segments, i + segments] for i in range(segments - 1)],
            dtype=torch.int32,
            device=self.device,
        )
        N_sides = torch.stack(
            [
                torch.cos(angles) * cos_slant,
                sin_slant.repeat(angles.shape),
                torch.sin(angles) * cos_slant,
            ],
            dim=1,
        ).repeat(2, 1)
        color_sides = self._color.repeat(F_sides.shape[0], 1)

        V_inner = torch.cat(
            [
                torch.stack(
                    [
                        r_inner * torch.cos(angles),
                        torch.full_like(angles, -0.5, device=self.device),
                        r_inner * torch.sin(angles),
                    ],
                    dim=1,
                ),
                torch.stack(
                    [
                        r_top_inner * torch.cos(angles),
                        torch.full_like(angles, 0.5, device=self.device),
                        r_top_inner * torch.sin(angles),
                    ],
                    dim=1,
                ),
            ],
            dim=0,
        )
        F_inner = torch.tensor(
            [[i + 1, i, i + 1 + segments] for i in range(segments - 1)]
            + [[i + 1 + segments, i, i + segments] for i in range(segments - 1)],
            dtype=torch.int32,
            device=self.device,
        )
        N_inner = -N_sides
        color_inner = self._color.repeat(F_inner.shape[0], 1)

        V_top = torch.cat(
            [
                V_sides[segments:],
                V_inner[segments:],
            ],
            dim=0,
        )
        F_top = F_sides
        N_top = torch.zeros_like(V_top, device=self.device)
        N_top[:, 1] = 1.0
        color_top = self._color.repeat(F_top.shape[0], 1)

        V_bottom = torch.cat([V_sides[:segments], V_inner[:segments]], dim=0)
        F_bottom = F_inner
        N_bottom = torch.zeros_like(V_top, device=self.device)
        N_bottom[:, 1] = -1.0
        color_bottom = self._color.repeat(F_bottom.shape[0], 1)

        sides = Mesh(V_sides, F_sides, N_sides, color_sides)
        inner = Mesh(V_inner, F_inner, N_inner, color_inner)
        top = Mesh(V_top, F_top, N_top, color_top)
        bottom = Mesh(V_bottom, F_bottom, N_bottom, color_bottom)

        return Mesh.union([sides, inner, top, bottom])


class TubeX(TubeY):
    def __init__(
        self,
        name: str,
        trs: TUnion[TRS, Xfrm] = TRS.identity(),
        color: Color = Colors.DEFAULT_BROWN,
        learnable: list[str] = ["t", "r", "s"],
        requires_grad: bool = True,
        r_top: TUnion[float, torch.Tensor] = 0.5,
        thickness: TUnion[float, torch.Tensor] = 0.2,
        segments: int = DEFAULT_SEGMENTS,
    ):
        super().__init__(
            name,
            trs @ Xfrm.device(trs.get_device()).y_to_x(),
            color,
            learnable,
            requires_grad,
            r_top,
            thickness,
            segments,
        )
        # HACK
        self._scale_order = [1, 0, 2]


class TubeZ(TubeY):
    def __init__(
        self,
        name: str,
        trs: TUnion[TRS, Xfrm] = TRS.identity(),
        color: Color = Colors.DEFAULT_BROWN,
        learnable: list[str] = ["t", "r", "s"],
        requires_grad: bool = True,
        r_top: TUnion[float, torch.Tensor] = 0.5,
        thickness: TUnion[float, torch.Tensor] = 0.2,
        segments: int = DEFAULT_SEGMENTS,
    ):
        super().__init__(
            name,
            trs @ Xfrm.device(trs.get_device()).y_to_z(),
            color,
            learnable,
            requires_grad,
            r_top,
            thickness,
            segments,
        )
        # HACK
        self._scale_order = [0, 2, 1]


### CUBE ###


class Cube(Model):
    def __init__(
        self,
        name: str,
        trs: TUnion[TRS, Xfrm] = TRS.identity(),
        color: Color = Colors.DEFAULT_BROWN,
        learnable: list[str] = ["t", "r", "s"],
        requires_grad: bool = True,
    ):
        super().__init__(name, trs, color, learnable, requires_grad)
        V = torch.tensor(
            [
                [-0.5, -0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, 0.5, 0.5],
                [0.5, -0.5, -0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, -0.5],
                [0.5, 0.5, 0.5],
                [-0.5, -0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [-0.5, 0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [-0.5, -0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, -0.5],
                [0.5, -0.5, 0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, 0.5, 0.5],
                [0.5, 0.5, -0.5],
                [0.5, 0.5, 0.5],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        F = torch.tensor(
            [
                [0, 1, 3],
                [0, 3, 2],
                [4, 6, 7],
                [4, 7, 5],
                [8, 9, 10],
                [10, 9, 11],
                [14, 13, 12],
                [14, 15, 13],
                [16, 19, 17],
                [16, 18, 19],
                [21, 23, 20],
                [20, 23, 22],
            ],
            dtype=torch.int32,
            device=self.device,
        )
        N = torch.tensor(
            [
                [-1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self._mesh = Mesh(V, F, N, self._color.repeat(F.shape[0], 1))
        self.volume_ratio = 1
        V = torch.tensor(
            [
                [-0.5, -0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, 0.5, 0.5],
                [0.5, -0.5, -0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, -0.5],
                [0.5, 0.5, 0.5],
            ],
            device=self.device,
        )
        F = torch.tensor(
            [
                [0, 1, 3],
                [0, 3, 2],
                [4, 6, 7],
                [4, 7, 5],
                [3, 7, 2],
                [2, 7, 6],
                [0, 5, 1],
                [0, 4, 5],
                [5, 3, 1],
                [5, 7, 3],
                [4, 0, 2],
                [4, 2, 6],
            ],
            device=self.device,
        )
        self._trimesh = trimesh.Trimesh(V.cpu().numpy(), F.cpu().numpy())
        # HACK
        self._watertight_mesh = Mesh(
            V,
            F,
            torch.empty((0, 3), dtype=torch.float32, device=self.device),
            torch.empty((0, 3), dtype=torch.float32, device=self.device),
        )

    def _generate_trimesh(self, all_params: list[torch.Tensor]) -> trimesh.Trimesh:
        return self._trimesh.copy()

    def _generate_watertight_mesh(self, all_params: list[torch.Tensor]) -> Mesh:
        return self._watertight_mesh


class Pyramid(Model):
    def __init__(
        self,
        name: str,
        trs: TUnion[TRS, Xfrm] = TRS.identity(),
        color: Color = Colors.DEFAULT_BROWN,
        learnable: list[str] = ["t", "r", "s"],
        requires_grad: bool = True,
        s_top: TUnion[float, torch.Tensor] = 0.5,
    ):
        super().__init__(name, trs, color, learnable, requires_grad)
        s_top = torch.as_tensor(s_top, dtype=torch.float32, device=self.device)
        self.params.register("s_top", s_top, "s_top" in learnable)
        # HACK: to fix
        self.volume_ratio = 1

    def _generate_mesh(self, all_params: list[torch.Tensor]) -> Mesh:
        s_top = all_params[3] / 2
        V = torch.tensor(
            [
                [-0.5, -0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, 0.5, 0.5],
                [0.5, -0.5, -0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, -0.5],
                [0.5, 0.5, 0.5],
                [-0.5, -0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [-0.5, 0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [-0.5, -0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, -0.5],
                [0.5, -0.5, 0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, 0.5, 0.5],
                [0.5, 0.5, -0.5],
                [0.5, 0.5, 0.5],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        V[[2, 3, 9, 13, 20, 21], 0] = -s_top
        V[[6, 7, 11, 15, 22, 23], 0] = s_top
        V[[2, 6, 9, 11, 20, 22], 2] = -s_top
        V[[3, 7, 13, 15, 21, 23], 2] = s_top

        F = torch.tensor(
            [
                [0, 1, 3],
                [0, 3, 2],
                [4, 6, 7],
                [4, 7, 5],
                [8, 9, 10],
                [10, 9, 11],
                [14, 13, 12],
                [14, 15, 13],
                [16, 19, 17],
                [16, 18, 19],
                [21, 23, 20],
                [20, 23, 22],
            ],
            dtype=torch.int32,
            device=self.device,
        )

        r_diff = 0.5 - s_top
        hyp = torch.sqrt(r_diff**2 + 1)
        cos_slant = 1 / hyp
        sin_slant = r_diff / hyp

        zero = torch.tensor(0.0, device=self.device)

        N = torch.cat(
            [
                torch.stack([-cos_slant, sin_slant, zero]).repeat(4, 1),
                torch.stack([cos_slant, sin_slant, zero]).repeat(4, 1),
                torch.stack([zero, sin_slant, -cos_slant]).repeat(4, 1),
                torch.stack([zero, sin_slant, cos_slant]).repeat(4, 1),
                torch.tensor([0.0, -1.0, 0.0], device=self.device).repeat(4, 1),
                torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(4, 1),
            ],
            dim=0,
        )

        return Mesh(V, F, N, self._color.repeat(F.shape[0], 1))

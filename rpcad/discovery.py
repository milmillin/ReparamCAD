import torch
import torch.nn.functional as F
from dataclasses import dataclass
import scipy
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union
from PIL import ImageDraw, ImageFont, Image
import trimesh
from tqdm import tqdm
import cv2
from tqdm.contrib.concurrent import process_map


from .renderer import Renderer, RenderParams
from .csg import Model, ParamMap, Xfrm, Mesh
from .constraints import (
    pvar,
    ConstraintSet,
    generate_filtered_constraints,
    generate_constraints,
    project_to_null_space,
)
from .mesh_boolean import union, intersection
from .typing import FileLike, Device
from .camera import CameraPose, gen_pose
from .util import display_image


def _compute_volume(mesh_t: Optional[trimesh.Trimesh]) -> float:
    if mesh_t is None:
        return 0
    return mesh_t.volume


def _compute_iou(mesh12: tuple[list[trimesh.Trimesh], list[trimesh.Trimesh]], depth: int = 0) -> float:
    """
    Compute iou between two meshes
    """
    try:
        mesh1, mesh2 = mesh12
        mesh1_ = union(mesh1)
        mesh2_ = union(mesh2)
        intersection_t = intersection([mesh1_, mesh2_])
        union_t = union([mesh1_, mesh2_])
        res = _compute_volume(intersection_t) / _compute_volume(union_t)
        assert -1e-5 <= res <= 1 + 1e-5
        return res
    except Exception as e:
        # HACK
        if depth < 20:
            print(f"Warning: retry {depth + 1}")
            return _compute_iou(mesh12, depth + 1)
        else:
            # i_vol = _compute_volume(intersection_t)
            # u_vol = max(mesh1_.volume, mesh2_.volume, union_t.volume)
            # return i_vol / u_vol
            return 0


def _compute_scale_factor(param_map: ParamMap, params: torch.Tensor) -> torch.Tensor:
    """
    Compute the weighting for each key point (the area of that plane)

    params: (n-dim) or (m, n-dim)

    returns: (1 or m, n-kp)
    """
    if params.dim() == 1:
        params = params.unsqueeze(0)

    entries = []
    for prim, p_map in param_map:
        if "s" in p_map and "t" in p_map:
            si = p_map["s"]
            for i in [[1, 2], [0, 2], [0, 1]]:
                col = [si[j] for j in i]
                entries.append(params[:, col].prod(dim=-1) * prim.volume_ratio)
                entries.append(params[:, col].prod(dim=-1) * prim.volume_ratio)
    return torch.stack(entries, dim=1)  # (m, n-kp)


def _construct_key_point_map(param_map: ParamMap, param_len: int, device: Device) -> torch.Tensor:
    """
    Create a map from param to key points (coordinate of planes)

    param_len: n-dim

    return (n-kp, n-dim)
    """
    entries = []
    for prim, p_map in param_map:
        if "s" in p_map and "t" in p_map:
            si = p_map["s"]
            ti = p_map["t"]
            for i in range(3):
                pti = pvar(ti[i], param_len)
                psi = pvar(si[i], param_len)
                entries.append(pti - 0.5 * psi)
                entries.append(pti + 0.5 * psi)
    return torch.stack(entries, dim=0).to(device=device)


def apply_constraint_linear(
    org_params: torch.Tensor,
    param_map: ParamMap,
    key_point_map: torch.Tensor,
    constraints: torch.Tensor,
) -> torch.Tensor:
    """
    org_params: (n-dim) or (m, n-dim)
    constraints: (n-constraints, n-dim)
    key_point_map: (n-kp, n-dim)

    returns (1 or m, n-dim)
    """
    device = org_params.device

    if org_params.dim() == 1:
        org_params = org_params.unsqueeze(0)  # (m, n-dim)

    ns_basis = torch.tensor(scipy.linalg.null_space(constraints.cpu().numpy()), device=device).T  # (n-ns, n-dim)

    kp_from_ns = key_point_map @ ns_basis.T  # (n-kp, n-ns)

    org_kp = org_params @ key_point_map.T  # (m, n-kp)

    scale_factor = _compute_scale_factor(param_map, org_params).unsqueeze(-1)  # (m, n-kp, 1)

    # minimize (kp_from_ns) @ P - org_kp
    driver = "gels" if scale_factor.is_cuda else "gelsd"
    sol = torch.linalg.lstsq(
        kp_from_ns * scale_factor, org_kp.unsqueeze(-1) * scale_factor, driver=driver
    ).solution.squeeze(
        -1
    )  # (m, n-ns)
    return sol @ ns_basis  # (m, n-dim)


def apply_constraint_image(
    org_params: torch.Tensor,
    constraints: torch.Tensor,
    model: Model,
    images: torch.Tensor,
    cam_poses: list[CameraPose],
    xfrm: Xfrm,
) -> torch.Tensor:
    """
    org_params: (n-dim,)
    constraints: (n-constraints, n-dim)
    images: (n-poses, 512, 512, 3)
    cam_poses: (n-poses)
    model: has to be cuda

    returns (n-dim,)
    """
    if org_params.dim() >= 2:
        raise ValueError("can only accept one model at a time")
    if images.shape[-3] != images.shape[-2]:
        raise ValueError("can only accept square images")

    assert model.device.type == "cuda"

    renderer = Renderer.get_shared()
    device = org_params.device

    ns_basis = torch.tensor(scipy.linalg.null_space(constraints.cpu().numpy()), device=device).T  # (n-ns, n-dim)

    y = project_to_null_space(org_params, ns_basis).squeeze(0)
    y = y.detach().clone().requires_grad_()  # (n-ns)

    render_params = RenderParams(res=images.shape[-2], spp=1)

    losses = []
    for j in range(5):
        optimizer = torch.optim.Adam([y], lr=3e-3)
        for i in range(30):
            optimizer.zero_grad()

            x = (y * ns_basis.T).sum(dim=-1)  # (n-dim)
            mesh = model.generate_mesh(model.unpack_params(x)).transform(xfrm)
            imgs = torch.cat(
                [renderer.render(mesh, pose, render_params) for pose in cam_poses],
                dim=0,
            )

            loss = ((imgs - images) ** 2).mean()
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
    return (y.detach().clone() * ns_basis.T).sum(dim=-1)  # (n-dim)


def generate_mesh_and_apply_flag(model: Model, params: torch.Tensor, flags: Union[torch.Tensor, list[bool]]) -> Mesh:
    meshes = model.generate_separate_meshes(model.unpack_params(params))
    return Mesh.union([m for m, f in zip(meshes, flags) if f])


def _generate_trimesh(model_param: tuple[Model, torch.Tensor]) -> list[trimesh.Trimesh]:
    model, param = model_param
    return model.generate_trimesh(model.unpack_params(param))


def score_iou(
    params: torch.Tensor,
    org_params: torch.Tensor,
    model: Model,
    concurrent: bool = True,
) -> torch.Tensor:
    """
    Score based on IoU with original mesh (defined by org_params)

    params: (n-dim) or (m, n-dim)
    org_params: (n-dim) or (m, n-dim)
    returns: (m,)
    """
    if params.dim() == 1:
        params = params.unsqueeze(0)
    if org_params.dim() == 1:
        org_params = org_params.unsqueeze(0)

    print("Generating mesh")
    if len(org_params) > 100:
        o_meshes = process_map(_generate_trimesh, [(model, param) for param in org_params], chunksize=32)
    else:
        o_meshes = [model.generate_trimesh(model.unpack_params(param)) for param in org_params]
    if len(params) > 100:
        n_meshes = process_map(_generate_trimesh, [(model, param) for param in params], chunksize=32)
    else:
        n_meshes = [model.generate_trimesh(model.unpack_params(param)) for param in params]

    if len(params) % len(org_params) == 0:
        o_meshes = o_meshes * (len(params) // len(org_params))
    elif len(org_params) % len(params) == 0:
        n_meshes = n_meshes * (len(org_params) // len(params))
    elif len(params) != len(org_params):
        raise ValueError(f"dimension mismatch {len(params)} and {len(org_params)}")

    print("Scoring")
    if concurrent:
        scores = process_map(_compute_iou, list(zip(o_meshes, n_meshes)), chunksize=32)
    else:
        scores = []
        for on in tqdm(list(zip(o_meshes, n_meshes))):
            scores.append(_compute_iou(on))
    return torch.tensor(scores, device=params.device)


def score_image(
    param: torch.Tensor,
    model: Model,
    diffused_imgs: torch.Tensor,
    cam_poses: list[CameraPose],
    xfrm: Xfrm,
) -> torch.Tensor:
    """
    Score based on the given images

    params: (n-dim)

    returns: ()
    """
    assert param.dim() == 1, "Only one param is supported"

    renderer = Renderer.get_shared()
    mesh = model.generate_mesh(model.unpack_params(param)).transform(xfrm).cuda()

    render_params = RenderParams(res=diffused_imgs.shape[-2], spp=1)

    imgs = torch.cat([renderer.render(mesh, pose, render_params).to(device=param.device) for pose in cam_poses])
    return ((imgs - diffused_imgs) ** 2).mean()


def _map_apply_constraints_linear(
    params: torch.Tensor,
    constraints: torch.Tensor,
    param_map: ParamMap,
    key_point_map: torch.Tensor,
    base_constraints: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    For each c in constraints, apply (base_constraints + c).

    params: (n-dim) or (n-params, n-dim)
    constraints: (n-eqn, n-dim) or (n-constraints, n-eqn, n-dim)
    key_point_map: (n-kp, n-dim)
    base_constraints: (n-base-constraints, n-dim)

    returns: (1 or n-constraints, 1 or n-params, dim)
    """
    if params.dim() == 1:
        params = params.unsqueeze(0)
    if constraints.dim() == 2:
        constraints = constraints.unsqueeze(0)

    print("Applying constraints")
    projected_params = []
    for constraint in tqdm(constraints):
        cat_constraints = constraint
        if base_constraints is None:
            cat_constraints = constraint
        else:
            cat_constraints = torch.cat([constraint, base_constraints], dim=0)
        projected_params.append(
            apply_constraint_linear(params, param_map, key_point_map, cat_constraints)  # (n-params, n-dim)
        )
    return torch.stack(projected_params, dim=0)  # (n-constraints, n-params, n-dim)


#############################################
# Utils


def find_change(x: torch.Tensor) -> int:
    """
    Returns the index at which the mean of x changes most significantly.

    Returns k such that the following total residual error is smallest:
    (x[:k] - x[:k].mean())**2).sum() + ((x[k:] - x[k:].mean())**2).sum()

    See also: https://www.mathworks.com/help/signal/ref/findchangepts.html
    """
    assert x.dim() == 1
    assert len(x) > 1
    n = len(x)
    res = []
    for k in range(1, n - 1):
        res.append(((x[:k] - x[:k].mean()) ** 2).sum() + ((x[k:] - x[k:].mean()) ** 2).sum())
    res = torch.stack(res)
    return int(res.argmin().item()) + 1


###############################################################################
# Algorithm


@dataclass
class DiscoveryInput:
    model: Model
    """
    Model
    """
    variation_parameters: torch.Tensor
    """
    (n-var, n-dim)  Variation parameters
    """
    variation_names: list[str]
    """
    (n-var,)  List of variation names
    """
    diffused_images: torch.Tensor
    """
    (n-var, n-poses, 512, 512, dim)  Diffused images
    """
    cam_poses: list[CameraPose]  # (n-poses)
    """
    (n-poses,)  Camera poses used to generate diffused_images
    """
    xfrms: list[Xfrm]  # (n-var)
    """
    (n-var,)  Render offset for each variation
    """


def load_output_dir(runs_dir: FileLike, model_name: str, model: Model) -> DiscoveryInput:
    device = model.device
    constraints = generate_filtered_constraints(model)
    full_constraints = generate_constraints(model)
    constraints = ConstraintSet(constraints.equations.to(device=device), constraints.names)
    full_constraints = ConstraintSet(full_constraints.equations.to(device=device), full_constraints.names)
    runs_dir = Path(runs_dir)
    prompts = []
    variation_params = []
    diffused_images = []
    cam_poses = []
    xfrms = []
    for run in runs_dir.iterdir():
        _, init_model, prompt = run.name.split("--")
        if init_model != model_name:
            continue
        all_params = torch.load(run / "params.pt")
        params = [p.to(device) for p in all_params[-1]]
        variation_params.append(model.pack_params(params))
        prompts.append(prompt)
        diffused_images.append(torch.load(run / "xtra_diffused_imgs.pt").to(device=device, dtype=torch.float32) / 255)
        cam_poses.append([p.to(device) for p in torch.load(run / "xtra_cam_poses.pt")])
        xfrms.append(model.generate_mesh(params).get_normalize_xfrm().to(device))
    diffused_images = torch.stack(diffused_images, dim=0)
    return DiscoveryInput(
        model=model,
        variation_parameters=torch.stack(variation_params).to(device=device),
        variation_names=prompts,
        diffused_images=diffused_images,
        cam_poses=cam_poses,
        xfrms=xfrms,
    )


@dataclass
class DiscoveryOutput:
    step_params: torch.Tensor
    """
    (n-steps + 1, n-variation, n-dim) Param at each step. step_params[0] is original param.
    """

    step_image_scores: torch.Tensor
    """
    (n-steps + 1, n-variation) Score with diffused images at each step. step_score[0] is score of original param.
    """

    constraints: ConstraintSet
    """
    (n-constraints, n-eqn, n-dim) Constraints considered
    """

    added_constraint_indices: list[int]
    """
    (n-steps) List of constraint indices added
    """

    cutoff: int
    """
    Cutoff step. params at cutoff = step_params[cutoff]. constraints added = added_constraints[:cutoff]
    """

    optional_parts: torch.Tensor
    """
    (n-parts, n-variations) Boolean indicating whether a part is optional.
    """


@dataclass
class GreedyAndResortOutput(DiscoveryOutput):
    all_params: list[torch.Tensor]  # n-steps, (n-rmd-constraints*, n-variation, n-dim)
    all_scores: list[torch.Tensor]  # n-steps, (n-rmd-constraints*, n-variation)
    step_scores: torch.Tensor  # (n-steps, n-variation)
    step_idx: list[list[int]]  # (n-steps, n-rmd-constraint*)
    scores_from_org: torch.Tensor  # (n-steps, n-variation)


def _score_image_batch(step_params: torch.Tensor, data: DiscoveryInput) -> torch.Tensor:
    """
    step_params: (n-steps + 1, n-var, n-dim)

    returns: (n-steps + 1, n-var)
    """
    all_scores_image = []
    for params in step_params:
        all_scores_ = []
        for i, param in enumerate(params):
            all_scores_.append(score_image(param, data.model, data.diffused_images[i], data.cam_poses, data.xfrms[i]))
        all_scores_image.append(torch.stack(all_scores_, dim=0))
    return torch.stack(all_scores_image)  # (n-steps + 1, n-var)


def _score_image_render_with_flags(
    data: DiscoveryInput, param: torch.Tensor, org_param: torch.Tensor, flags: Union[torch.Tensor, list[bool]]
) -> torch.Tensor:
    model = data.model
    cam_poses = data.cam_poses
    renderer = Renderer.get_shared()

    org_mesh = model.generate_mesh(model.unpack_params(org_param))
    xfrm = org_mesh.get_normalize_xfrm()
    org_mesh = org_mesh.transform(xfrm)
    mesh = generate_mesh_and_apply_flag(model, param, flags).transform(xfrm)

    org_imgs = torch.cat([renderer.render(org_mesh, pose) for pose in cam_poses])

    imgs = torch.cat([renderer.render(mesh, pose) for pose in cam_poses])

    return ((imgs - org_imgs) ** 2).mean()


def find_optional_parts(data: DiscoveryInput) -> torch.Tensor:
    all_scores_from_rendered = []
    for flag in ~torch.eye(len(data.model.children), dtype=torch.bool):
        scores_ = []
        for i, param in enumerate(data.variation_parameters):
            scores_.append(_score_image_render_with_flags(data, param, param, flag))
        all_scores_from_rendered.append(torch.stack(scores_))
    all_scores_from_rendered = torch.stack(all_scores_from_rendered)
    return all_scores_from_rendered < 1e-4  # (n_children, n_variations)


def run_greedy_and_resort(
    data: DiscoveryInput,
    data_gpu: DiscoveryInput,
) -> GreedyAndResortOutput:
    """
    Returns: added constraints, params, scores
    """
    constraints_set = generate_filtered_constraints(data.model)
    param_map = data.model.param_map()
    device = data.model.device
    param_len = data.variation_parameters[0].shape[0]
    key_point_map = _construct_key_point_map(param_map, param_len, device)

    constraints = constraints_set.equations
    remaining: set[int] = set(range(constraints.shape[0]))

    added_constraints: list[int] = []
    all_params = []
    all_scores = []
    step_params = [data.variation_parameters]  # (n-var, n-dim)
    step_scores = []
    step_idx = []
    while len(remaining) > 0:
        current_order = list(remaining)

        projected_params = _map_apply_constraints_linear(
            step_params[-1],
            constraints=constraints[current_order],
            base_constraints=constraints[added_constraints].flatten(0, 1) if len(added_constraints) > 0 else None,
            param_map=param_map,
            key_point_map=key_point_map,
        )  # (len(current_order), n-var, n-dim)

        # Scores relative to previous step
        scores = score_iou(projected_params.flatten(0, 1), step_params[-1], data.model).reshape(
            projected_params.shape[:2]
        )  # (len(current_order), n-var)

        all_scores.append(scores)
        all_params.append(projected_params)
        reduced_scores = scores.prod(dim=-1)  # (len(current_order),)
        cidx = int(reduced_scores.argmax().item())
        idx = current_order[cidx]
        print(f"Adding {idx}: {constraints_set.names[idx]}")
        added_constraints.append(idx)
        remaining.remove(idx)

        step_scores.append(scores[cidx])
        step_params.append(projected_params[cidx])
        step_idx.append(current_order)

        # check if remaining is a linear combination of added constraints
        current_rank = torch.linalg.matrix_rank(constraints[added_constraints].flatten(0, 1))
        for r in list(remaining):
            r_rank = torch.linalg.matrix_rank(constraints[added_constraints + [r]].flatten(0, 1))
            if current_rank == r_rank:
                print(f"-> Implies {r}: {constraints_set.names[r]}")
                remaining.remove(r)
    step_params = torch.stack(step_params)
    step_scores = torch.stack(step_scores)  # iou

    scores_from_org = score_iou(step_params[1:].flatten(0, 1), data.variation_parameters, data.model).reshape(
        -1, data.variation_parameters.shape[0]
    )

    step_image_scores = _score_image_batch(step_params, data)  # (n-steps + 1, n-var)

    cutoff = find_change(step_image_scores.mean(dim=-1)) - 1

    optional_parts = find_optional_parts(data)

    return GreedyAndResortOutput(
        constraints=constraints_set,
        step_params=step_params,
        step_image_scores=step_image_scores,
        cutoff=cutoff,
        added_constraint_indices=added_constraints,
        optional_parts=optional_parts,
        # extra
        all_params=all_params,
        all_scores=all_scores,
        step_scores=step_scores,
        step_idx=step_idx,
        scores_from_org=scores_from_org,
    )


@dataclass
class SingleSortOutput(DiscoveryOutput):
    indiv_params: torch.Tensor  # (n-constraints, n-var, n-dim)
    indiv_scores: torch.Tensor  # (n-constraints, n-var)
    cum_params: torch.Tensor  # (n-reduced-constraints, n-var, n-dim)
    cum_scores: torch.Tensor  # (n-reduced-constraints, n-var)


def run_single_sort(data: DiscoveryInput, data_gpu: DiscoveryInput, res: int = 256) -> SingleSortOutput:
    """
    Returns: added constraints, params, scores
    """
    constraints_set = generate_filtered_constraints(data_gpu.model)

    constraints = constraints_set.equations  # gpu

    if 512 % res != 0:
        raise ValueError("accept resolution as factor of 512 for now")

    images = F.avg_pool2d(data_gpu.diffused_images.flatten(0, 1).permute(0, 3, 1, 2), 512 // res).permute(0, 2, 3, 1)
    images = images.reshape(
        *data_gpu.diffused_images.shape[:2], *images.shape[-3:]
    )  # (n-mod, n-pose, res, res, 3), gpu

    print("Applying params")
    all_params = []
    for constraint in tqdm(constraints):
        params_ = []
        for i, param in enumerate(data.variation_parameters):
            projected_params = apply_constraint_image(
                param, constraint, data_gpu.model, images[i], data_gpu.cam_poses, data_gpu.xfrms[i]
            )
            params_.append(projected_params)
        all_params.append(torch.stack(params_, dim=0))
    all_params = torch.stack(all_params, dim=0)  # (n-constraints, n-var, n-dim), gpu

    print("Scoring")
    indiv_scores = score_iou(all_params.flatten(0, 1).cpu(), data.variation_parameters, model=data.model).reshape(
        all_params.shape[:2]
    )  # (n-constraints, n-var), cpu

    reduced_scores = indiv_scores.prod(dim=-1)  # cpu
    order = reduced_scores.argsort(descending=True)  # cpu

    rank = torch.tensor(
        [torch.linalg.matrix_rank(constraints[order[: i + 1]].flatten(0, 1)) for i in range(len(constraints))]
    )
    transition_mask = rank != rank.roll(1)  # gpu

    filtered_order = order[transition_mask.cpu()]  # cpu

    print("Applying cummulative constraints")
    cum_params = []
    for j in tqdm(range(len(filtered_order))):
        cum_params_ = []
        for i, param in enumerate(data_gpu.variation_parameters):
            cum_params_.append(
                apply_constraint_image(
                    param,
                    constraints[filtered_order[: j + 1]].flatten(0, 1),
                    data_gpu.model,
                    images[i],
                    data_gpu.cam_poses,
                    data_gpu.xfrms[i],
                )
            )
        cum_params.append(torch.stack(cum_params_, dim=0))
    cum_params = torch.stack(cum_params, dim=0)  # (n-step, n-var, n-dim), gpu

    print("Cumulative Scoring")
    cum_scores = score_iou(cum_params.flatten(0, 1).cpu(), data.variation_parameters, model=data.model).reshape(
        cum_params.shape[:2]
    )  # cpu

    step_params = torch.cat(
        [data_gpu.variation_parameters.unsqueeze(0), cum_params], dim=0
    ).cpu()  # (n-step + 1, n-var, n-dim)

    step_image_scores = _score_image_batch(step_params, data)
    cutoff = find_change(step_image_scores.mean(dim=-1)) - 1

    optional_parts = find_optional_parts(data)

    return SingleSortOutput(
        constraints=constraints_set.to("cpu"),
        step_params=step_params,
        step_image_scores=step_image_scores,
        cutoff=cutoff,
        added_constraint_indices=[int(x.item()) for x in filtered_order],
        optional_parts=optional_parts,
        # extra
        indiv_params=all_params.cpu(),
        indiv_scores=indiv_scores,
        cum_params=cum_params.cpu(),
        cum_scores=cum_scores,
    )


#############################################
# Visualization


def create_image_grid(model: Model, all_params: torch.Tensor, all_names: list[str]) -> Image.Image:
    """
    all_params: (n-constraint, n-model, 48)
    all_names: (n-constraint)
    """
    assert len(all_params) == len(all_names)
    renderer = Renderer.get_shared()
    imgs = []
    with torch.no_grad():
        for params in tqdm(all_params.reshape(-1, all_params.shape[-1])):
            imgs.append(
                renderer.render(model.generate_mesh(model.unpack_params(params), normalize=True)).detach().cpu()
            )
    imgs = torch.cat(imgs, dim=0).reshape(*all_params.shape[:2], 512, 512, 3).permute(0, 2, 1, 3, 4)
    grid = display_image(imgs.cpu().flatten(0, 1).flatten(1, 2))

    drawer = ImageDraw.Draw(grid)
    drawer.font = ImageFont.truetype(  # type: ignore
        str(Path(list(cv2.__path__)[0]) / "qt/fonts/DejaVuSans.ttf"), size=40
    )
    for i, name in enumerate(all_names):
        drawer.text(
            (10, 10 + i * 512),
            f"{i}. {name}",
            fill=(0, 0, 0),
        )
    return grid


def plot_scores(scores: torch.Tensor, labels: list[str]):
    """
    scores: (n-constraints, m)
    """
    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    for score, label in zip(scores.T, labels):
        ax.plot(score.numpy(), label=label)
    # ax.plot(reduce(scores).numpy(), label="prod")
    # ax.set_ylim(0, 1)
    ax.legend()
    return fig, ax

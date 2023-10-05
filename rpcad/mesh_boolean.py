import trimesh
import tempfile
import subprocess
import os
from typing import Optional
from pathlib import Path

_MESH_BOOLEAN_EXEC = (
    Path(__file__).parent.parent.parent
    / "InteractiveAndRobustMeshBooleans/build/mesh_booleans"
).absolute()


def boolean(op, meshes: list[trimesh.Trimesh]) -> Optional[trimesh.Trimesh]:
    with tempfile.TemporaryDirectory() as tmp:
        files = [os.path.join(tmp, f"{i:02d}.obj") for i in range(len(meshes))]
        for mesh, fn in zip(meshes, files):
            mesh.export(fn, file_type="obj")
        outfile = os.path.join(tmp, "out.obj")
        a = subprocess.run([".", op, *files, outfile], executable=_MESH_BOOLEAN_EXEC)
        if a.returncode != 0:
            print(a.stdout.decode("utf-8".strip()))
            print(a.stderr.decode("utf-8".strip()))
            return None
        res = trimesh.load(outfile, file_type="obj", force="mesh")
    return res if isinstance(res, trimesh.Trimesh) else None


def union(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    res = boolean("union", meshes)
    assert res is not None
    return res


def intersection(meshes: list[trimesh.Trimesh]) -> Optional[trimesh.Trimesh]:
    return boolean("intersection", meshes)


def subtraction(meshes: list[trimesh.Trimesh]) -> Optional[trimesh.Trimesh]:
    return boolean("subtraction", meshes)


def xor(meshes: list[trimesh.Trimesh]) -> Optional[trimesh.Trimesh]:
    return boolean("xor", meshes)

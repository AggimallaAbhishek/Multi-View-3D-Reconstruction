"""Surface mesh from the dense cloud: screened Poisson reconstruction (Open3D, external;
legacy API, the tensor API has no Poisson). Normals are estimated from neighbours and
oriented towards the cameras that saw each point; the thinly supported fringe Poisson
extrapolates (lowest vertex densities) is cut off."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import DenseConfig
from src.dense.fusion import DenseCloud


def poisson_mesh(cloud: DenseCloud, config: DenseConfig):
    """Open3D ``TriangleMesh`` (legacy) with vertex colours."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud.points))
    pcd.colors = o3d.utility.Vector3dVector(cloud.colors / 255.0)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(config.normal_neighbours))
    normals = np.asarray(pcd.normals)
    flip = np.sum(normals * cloud.views, axis=1) < 0
    normals[flip] *= -1
    pcd.normals = o3d.utility.Vector3dVector(normals)
    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=config.poisson_depth, n_threads=config.poisson_threads
    )
    density = np.asarray(density)
    mesh.remove_vertices_by_mask(density < np.quantile(density, config.mesh_density_quantile))
    return mesh


def write_mesh(path: str | Path, mesh) -> Path:
    import open3d as o3d

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(path), mesh):
        raise OSError(f"could not write {path}")
    return path

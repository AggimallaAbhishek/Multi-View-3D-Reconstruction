"""Dense reconstruction from the sparse result's cameras: rectified two-view stereo per
image (``stereo``), fused into one cloud by multi-view depth consistency (``fusion``),
optionally meshed (``meshing``, Open3D Poisson: external)."""

from src.dense.fusion import DenseCloud, fuse_depth_maps, voxel_downsample
from src.dense.stereo import ViewPlan, pair_depth, plan_views

__all__ = [
    "DenseCloud",
    "ViewPlan",
    "fuse_depth_maps",
    "pair_depth",
    "plan_views",
    "voxel_downsample",
]

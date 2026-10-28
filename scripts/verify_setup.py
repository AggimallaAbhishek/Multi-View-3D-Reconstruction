"""Check that the Python environment can run the pipeline.

Usage: python scripts/verify_setup.py
"""

import platform
import sys


def main() -> int:
    import cv2
    import matplotlib
    import numpy as np
    import open3d as o3d
    import pandas
    import scipy

    print(f"Python      {platform.python_version()} ({platform.machine()})")
    print(f"OpenCV      {cv2.__version__}")
    print(f"NumPy       {np.__version__}")
    print(f"SciPy       {scipy.__version__}")
    print(f"Open3D      {o3d.__version__}")
    print(f"Matplotlib  {matplotlib.__version__}")
    print(f"pandas      {pandas.__version__}")

    failures = []

    if not hasattr(cv2, "SIFT_create"):
        failures.append("cv2.SIFT_create is missing (install only opencv-contrib-python)")
    else:
        img = (np.random.default_rng(0).random((128, 128)) * 255).astype(np.uint8)
        keypoints = cv2.SIFT_create().detect(img, None)
        print(f"SIFT        ok ({len(keypoints)} keypoints on noise image)")

    try:
        pcd = o3d.t.geometry.PointCloud(o3d.core.Tensor(np.zeros((4, 3), dtype=np.float32)))
        assert pcd.point.positions.shape[0] == 4
        print("Open3D      ok (tensor point cloud)")
    except Exception as exc:  # noqa: BLE001 - report any import/runtime failure
        failures.append(f"Open3D point cloud creation failed: {exc}")

    if failures:
        print("\nEnvironment verification FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nEnvironment verification successful.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

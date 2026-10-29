"""Registered cameras of an incremental reconstruction, keyed by image index."""

from __future__ import annotations

from src.geometry.camera import Camera


class CameraRegistry:
    """Image index -> camera pose, remembering the order images were registered in.

    That order is also the view index of each camera when the reconstruction is
    exported for triangulation or bundle adjustment.
    """

    def __init__(self) -> None:
        self._cameras: dict[int, Camera] = {}

    def register(self, image: int, camera: Camera) -> None:
        if image in self._cameras:
            raise ValueError(f"image {image} is already registered")
        self._cameras[image] = camera

    def update(self, image: int, camera: Camera) -> None:
        """Replace the pose of a registered image (e.g. after bundle adjustment)."""
        if image not in self._cameras:
            raise KeyError(f"image {image} is not registered")
        self._cameras[image] = camera

    def camera(self, image: int) -> Camera:
        return self._cameras[image]

    def __contains__(self, image: object) -> bool:
        return image in self._cameras

    def __len__(self) -> int:
        return len(self._cameras)

    @property
    def images(self) -> list[int]:
        """Registered image indices in registration order."""
        return list(self._cameras)

"""Learned features and matching — **external**: kornia's DISK and LightGlue (optional extra).

DISK (Tyszkiewicz et al., NeurIPS 2020) detects and describes keypoints with a U-Net trained
for matching; LightGlue (Lindenberger et al., ICCV 2023) matches two sets of keypoints with
attention over both images at once instead of comparing descriptors one by one, which keeps
working at wider baselines than the ratio test. Both are pretrained networks whose weights
kornia downloads on first use. Install with ``uv pip install -e ".[learned]"``.

They plug into the classical pipeline unchanged: ``DISKExtractor`` returns the same
``Features`` arrays as SIFT (strongest first, pixel-centre coordinates), ``LightGlueMatcher``
the same ``Matches`` / ``MatchStats`` as ``DescriptorMatcher``, with ``1 - confidence`` as
the distance. Verification, tracks and registration do not know which produced them.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np

from src.config import MatchConfig
from src.data.preprocessing import to_gray
from src.features.extractor import FEATURE_DEFAULTS, FeatureExtractor, Features
from src.features.matcher import Matches, MatchStats

try:
    import kornia
    import torch
    from kornia.feature import DISK, LightGlue
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        'learned features need the optional extra: uv pip install -e ".[learned]"'
    ) from exc

# DISK's descriptors are matched by LightGlue weights trained on DISK; nothing else fits them.
LIGHTGLUE_FEATURES = "disk"


def torch_device(name: str = "auto") -> torch.device:
    """``auto``: CUDA, else Apple's MPS, else the CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DISKExtractor(FeatureExtractor):
    """DISK keypoints (``depth`` checkpoint) on the grayscale image, copied to 3 channels.

    Detection runs on the grayscale working image like SIFT, so the feature cache and the
    rest of the pipeline treat it the same. DISK places keypoints on whole pixels (heatmap
    maxima); its precision is therefore ~0.3 px, below SIFT's sub-pixel refinement.
    """

    name = "disk"
    descriptor_dim = 128
    descriptor_dtype = np.float32

    def __init__(self, max_features: int = FEATURE_DEFAULTS.max_features, device: str = "auto"):
        super().__init__(max_features)
        self.device = torch_device(device)
        self.model = DISK.from_pretrained("depth", device=self.device).eval()

    def _detector(self):  # pragma: no cover - DISK does not go through cv2.Feature2D
        raise NotImplementedError

    def signature(self) -> str:
        # The device type is part of it: GPU kernels can reorder near-equal scores, so a
        # cache filled on one device is not assumed valid on another.
        return (
            f"disk-n{self.max_features}-depth-kornia{kornia.__version__}"
            f"-torch{torch.__version__}-{self.device.type}"
        )

    def detect(self, image: np.ndarray) -> Features:
        gray = to_gray(image)
        tensor = torch.from_numpy(gray).to(self.device, torch.float32) / 255.0
        tensor = tensor.expand(1, 3, *gray.shape)
        with torch.inference_mode():
            (out,) = self.model(tensor, n=self.max_features, pad_if_not_divisible=True)
        points = out.keypoints.cpu().numpy().astype(np.float64)
        if len(points) == 0:
            return self._empty()
        scores = out.detection_scores.cpu().numpy().astype(np.float32)
        order = np.argsort(-scores, kind="stable")[: self.max_features]
        n = len(order)
        return Features(
            points=points[order],
            descriptors=out.descriptors.cpu().numpy().astype(np.float32)[order],
            # No scale or orientation: the size drawn is the 5 px non-maximum window.
            sizes=np.full(n, 5.0, dtype=np.float32),
            angles=np.full(n, -1.0, dtype=np.float32),  # OpenCV's "not computed"
            responses=scores[order],
            detector=self.name,
        )


def _learned_version() -> str:
    """kornia's and torch's versions: LightGlue's weights ship with kornia and run on
    torch, so either can change its matches."""
    from importlib.metadata import PackageNotFoundError, version

    def get(package: str) -> str:
        try:
            return version(package)
        except PackageNotFoundError:
            return "none"

    return f"kornia-{get('kornia')}-torch-{get('torch')}"


class LightGlueMatcher:
    """LightGlue on DISK features; same interface as ``DescriptorMatcher``.

    LightGlue normalises keypoints by the image size, so it is given every image's
    ``(width, height)`` up front, indexed like the features.
    """

    def __init__(
        self, config: MatchConfig, image_sizes: Sequence[tuple[int, int]], device: str = "auto"
    ):
        self.config = config
        self.image_sizes = list(image_sizes)
        self.device = torch_device(device)
        # Its weights ship with kornia and run on torch, and GPU kernels can reorder
        # near-equal scores (as for the DISK feature cache): all part of its cache key.
        self.cache_tag = f"{_learned_version()}|{self.device.type}"
        self.model = (
            LightGlue(LIGHTGLUE_FEATURES, filter_threshold=config.lightglue_filter_threshold)
            .to(self.device)
            .eval()
        )

    def _side(self, features: Features, index: int) -> dict:
        def tensor(a: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(self.device)[None]

        return {
            "keypoints": tensor(features.points),
            "descriptors": tensor(features.descriptors),
            "image_size": tensor(np.asarray(self.image_sizes[index], dtype=np.float32)),
        }

    def match(
        self, features_i: Features, features_j: Features, i: int = 0, j: int = 1
    ) -> tuple[Matches, MatchStats]:
        start = time.perf_counter()
        for f in (features_i, features_j):
            if f.detector != LIGHTGLUE_FEATURES:
                raise ValueError(
                    f"LightGlue here is trained on {LIGHTGLUE_FEATURES} features, got "
                    f"{f.detector!r}: set features.detector: {LIGHTGLUE_FEATURES}"
                )
        if len(features_i) == 0 or len(features_j) == 0:
            matches = Matches.empty(i, j)
        else:
            with torch.inference_mode():
                out = self.model(
                    {"image0": self._side(features_i, i), "image1": self._side(features_j, j)}
                )
            pairs = out["matches"][0].cpu().numpy().astype(np.int32)
            scores = out["scores"][0].float().cpu().numpy()
            order = np.argsort(pairs[:, 0], kind="stable")  # ascending idx_i, as elsewhere
            pairs, scores = pairs[order], scores[order]
            distances = np.clip(1.0 - scores, 0.0, 1.0).astype(np.float32)
            matches = Matches(i, j, pairs[:, 0].copy(), pairs[:, 1].copy(), distances)
        n = len(matches)
        stats = MatchStats(
            i=i,
            j=j,
            features_i=len(features_i),
            features_j=len(features_j),
            raw=n,  # LightGlue has no separate nearest-neighbour or ratio stage
            after_ratio=n,
            final=n,
            time_ms=(time.perf_counter() - start) * 1000,
        )
        return matches, stats

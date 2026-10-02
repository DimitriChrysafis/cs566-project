"""NYU Depth v2 test split (Eigen 654-image protocol).

Layout under root (default: ``<repo>/data/nyu``):
    rgb/NNNNN.png    uint8, (480, 640, 3), BGR-free RGB ordering
    depth/NNNNN.npy  float32, (480, 640), depth in meters, 0 = invalid

Files are named by the original labeled-dataset index (e.g. 01441).
Depth is metric (Kinect, ~0.5-10 m); pixels with value 0 are missing.
"""

from pathlib import Path

import cv2
import numpy as np


def _repo_root():
    return Path(__file__).resolve().parents[3]


def nyu_test_dir():
    return _repo_root() / "data" / "nyu"


def iter_nyu(root=None, limit=None):
    root = Path(root) if root else nyu_test_dir()
    depth_dir = root / "depth"
    rgb_dir = root / "rgb"
    names = sorted(p.stem for p in depth_dir.glob("*.npy"))
    if limit:
        names = names[: int(limit)]
    for name in names:
        depth = np.load(depth_dir / f"{name}.npy")
        rgb = cv2.imread(str(rgb_dir / f"{name}.png"), cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(rgb_dir / f"{name}.png")
        yield {
            "name": name,
            "rgb": cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
            "depth": depth.astype(np.float32),
        }

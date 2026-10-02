"""DDFF 12-Scene benchmark (Hazirbas et al., ACCV 2018).

what the mirrors actually ship (vision.in.tum.de/webarchive/hazirbas/ddff12scene):

    ddff-dataset-test.h5        'stack_test': (120, 10, 383, 552, 3) uint8 focal
                                stacks for the 6 test scenes. official test gt
                                was withheld for the benchmark leaderboard, so
                                `depth` is None for these scenes.
    ddff-dataset-trainval.h5    optional (12.6 gb): 'stack_train'/'disp_train',
                                'stack_val'/'disp_val', 224x224 crops.
    DepthRegistered/<split>/<scene>/DEPTH_XXXX.png   uint16, meters * 1000,
                                rgb-d depth registered to the lightfield view.
                                only the 6 train scenes are populated.
    calib/                      lytro illum intrinsics + stereo calib yml.

conventions:
    - focal stacks were rendered by refocusing lytro illum lightfields at 10
      disparities linearly sampled in [0.28, 0.02] px (paper sec. 4), which the
      authors equate to focus depths of [0.5, 7] m, i.e. depth_m = 0.14 / disp_px
      for this rig. `focdist` is returned in meters under the assumption that
      slice index order follows ascending focus distance; flip it if a
      focus-measure check says otherwise.
    - gt 'disp' arrays (trainval h5) are lightfield disparity in px; convert to
      meters with depth = DISP_TO_M / disp.
    - registered depth maps carry a different viewpoint crop than the h5
      stacks; they are exposed as their own scenes ('reg/train/<scene>/<idx>')
      with stack=None.
"""

from pathlib import Path

import cv2
import h5py
import numpy as np

DISP_TO_M = 0.14
STACK_FOCUSDIST_M = np.linspace(0.5, 7.0, 10).astype(np.float32)


def _repo_root():
    return Path(__file__).resolve().parents[3]


def ddff_dir():
    return _repo_root() / "data" / "ddff"


def _h5_scenes(root):
    scenes = []
    for f in sorted(Path(root).glob("*.h5")):
        with h5py.File(f, "r") as h5:
            for key in h5.keys():
                if not key.startswith("stack_"):
                    continue
                split = key.split("_", 1)[1]
                for i in range(h5[key].shape[0]):
                    scenes.append((f"{split}_{i:03d}", f.name, split, i))
    return scenes


def _reg_scenes(root):
    scenes = []
    reg = Path(root) / "DepthRegistered"
    if reg.exists():
        for p in sorted(reg.rglob("DEPTH_*.png")):
            if "colored" in p.stem:
                continue
            rel = p.relative_to(reg)
            scenes.append((f"reg/{rel.parents[1].name}/{rel.parents[0].name}/{p.stem[6:]}", p, None, None))
    return scenes


def list_scenes(root=None):
    root = Path(root) if root else ddff_dir()
    names = [s[0] for s in _h5_scenes(root)] + [s[0] for s in _reg_scenes(root)]
    return names


def _camera(root):
    calib = sorted(str(p.relative_to(root)) for p in (Path(root) / "calib").rglob("*") if p.is_file())
    return {
        "device": "lytro illum + rgb-d (registered)",
        "refocus_disp_px": (0.28, 0.02),
        "disp_to_m": DISP_TO_M,
        "calib_files": calib,
    }


def load_dff_scene(root, name):
    """name: e.g. 'test_007' or 'reg/train/studentlab/0043' (see list_scenes)."""
    root = Path(root) if root else ddff_dir()
    if name.startswith("reg/"):
        _, split, scene, idx = name.split("/")
        png = root / "DepthRegistered" / split / scene / f"DEPTH_{idx}.png"
        depth = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(png)
        return {
            "stack": None,
            "depth": depth.astype(np.float32) / 1000.0,
            "focdist": None,
            "camera": _camera(root),
        }
    split, idx = name.rsplit("_", 1)
    idx = int(idx)
    for f in sorted(root.glob("*.h5")):
        with h5py.File(f, "r") as h5:
            stack_key, disp_key = f"stack_{split}", f"disp_{split}"
            if stack_key not in h5:
                continue
            stack = np.asarray(h5[stack_key][idx])
            disp = np.asarray(h5[disp_key][idx]) if disp_key in h5 else None
            depth = (DISP_TO_M / np.maximum(disp, 1e-6)).astype(np.float32) if disp is not None else None
            if disp is not None:
                depth[disp <= 0] = 0.0
            return {
                "stack": stack,
                "depth": depth,
                "focdist": STACK_FOCUSDIST_M.copy(),
                "camera": _camera(root),
            }
    raise KeyError(f"no stack_{split} in {root}")

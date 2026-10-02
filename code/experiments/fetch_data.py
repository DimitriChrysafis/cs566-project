"""fetch benchmark data into <repo>/data.

    python fetch_data.py --list          show what is present
    python fetch_data.py nyu             NYU Depth v2 Eigen test split (654 imgs)
    python fetch_data.py ddff            DDFF 12-Scene test h5 (~761 MB)
    python fetch_data.py ddff --trainval also grab the 12.6 GB trainval h5
    python fetch_data.py all             nyu + ddff

nyu sources are tried in order; the first two are the historical hosts and
are usually dead, the HF mirror of the fast-depth preprocessing is what
actually works today. ddff comes from the TUM vision webarchive.
"""

import argparse
import io
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

import cv2
import h5py
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
NYU_DIR = ROOT / "data" / "nyu"
DDFF_DIR = ROOT / "data" / "ddff"

NYU_ZIP_URL = "https://s3-eu-west-1.amazonaws.com/densedepth/nyu_test.zip"
NYU_MAT_URL = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
NYU_HF_BASE = "https://huggingface.co/datasets/sayakpaul/nyu_depth_v2/resolve/main/data"
NYU_HF_VAL_TARS = ["val-000000.tar", "val-000001.tar"]
NYU_N_TEST = 654

DDFF_BASE = "https://vision.in.tum.de/webarchive/hazirbas/ddff12scene"
DDFF_TEST = "ddff-dataset-test.h5"
DDFF_TRAINVAL = "ddff-dataset-trainval.h5"
DDFF_DEPTH = "depthregistered.tar.gz"
DDFF_CAL_BASE = "https://raw.githubusercontent.com/hazirbas/ddff-toolbox/master/caldata"
DDFF_CAL_FILES = [
    "lfcalib/IntParamLF.mat",
    "lfcalib/IntParamLF_SubAperture.mat",
    "stereocalib/stereoparams.yml",
]


def log(msg):
    print(msg, flush=True)


def download(url, dest, timeout=30):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    head = requests.head(url, timeout=timeout, allow_redirects=True)
    if head.status_code >= 400:
        raise RuntimeError(f"{url}: http {head.status_code}")
    total = int(head.headers.get("Content-Length") or 0)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        encoded = "Content-Encoding" in r.headers
        n, nxt = 0, 50 * 2**20
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                n += len(chunk)
                if n >= nxt:
                    log(f"    {n / 2**20:.0f}/{total / 2**20:.0f} mb")
                    nxt += 200 * 2**20
    if total and not encoded and tmp.stat().st_size != total:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{url}: short read")
    tmp.replace(dest)


def _save_nyu_pair(out, name, rgb, depth):
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out / "rgb" / f"{name}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"could not write rgb for {name}")
    np.save(out / "depth" / f"{name}.npy", np.asarray(depth, dtype=np.float32))


def _depth_to_meters(arr):
    arr = np.asarray(arr)
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 1000.0
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0 * 10.0
    return arr.astype(np.float32)


def _nyu_from_zip(out):
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
        tmp = Path(f.name)
    try:
        download(NYU_ZIP_URL, tmp)
        zf = zipfile.ZipFile(tmp)
        groups = {}
        for info in zf.infolist():
            stem = Path(info.filename).stem.lower()
            kind = "depth" if "depth" in stem else ("rgb" if ("color" in stem or "rgb" in stem) else None)
            if kind:
                groups.setdefault(Path(info.filename).stem.split("_")[0], {})[kind] = info
        n = 0
        for key in sorted(groups):
            g = groups[key]
            if "rgb" not in g or "depth" not in g:
                continue
            rgb = cv2.imdecode(np.frombuffer(zf.read(g["rgb"]), np.uint8), cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            raw = cv2.imdecode(np.frombuffer(zf.read(g["depth"]), np.uint8), cv2.IMREAD_UNCHANGED)
            _save_nyu_pair(out, key.zfill(5), rgb, _depth_to_meters(raw))
            n += 1
        if n == 0:
            raise RuntimeError("nyu_test.zip: no rgb/depth pairs found")
        log(f"    {n} pairs from nyu_test.zip")
    finally:
        tmp.unlink(missing_ok=True)


def _nyu_from_mat(out):
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
        tmp = Path(f.name)
    try:
        download(NYU_MAT_URL, tmp)
        with h5py.File(tmp, "r") as h5:
            images = h5["images"]
            depths = h5["depths"]
            n_total = images.shape[0]
            # eigen test indices are not shipped; fall back to last 654 in index order
            idxs = list(range(n_total - NYU_N_TEST, n_total))
            log("    warning: canonical eigen index list unavailable, using last 654 indices")
            for j, i in enumerate(idxs):
                rgb = np.asarray(images[i]).transpose(2, 1, 0)
                _save_nyu_pair(out, f"{i:05d}", rgb, np.asarray(depths[i]).T)
                if j % 100 == 0:
                    log(f"    {j}/{len(idxs)}")
        log(f"    {len(idxs)} pairs from labeled mat")
    finally:
        tmp.unlink(missing_ok=True)


def _nyu_from_hf(out):
    for tar_name in NYU_HF_VAL_TARS:
        with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as f:
            tmp = Path(f.name)
        try:
            download(f"{NYU_HF_BASE}/{tar_name}", tmp)
            n = 0
            with tarfile.open(tmp, "r:") as tf:
                for member in tf:
                    if not member.name.endswith(".h5"):
                        continue
                    raw = tf.extractfile(member).read()
                    with h5py.File(io.BytesIO(raw), "r") as h5:
                        rgb = np.asarray(h5["rgb"]).transpose(1, 2, 0)
                        depth = np.asarray(h5["depth"])
                    name = Path(member.name).stem
                    _save_nyu_pair(out, name, rgb, depth)
                    n += 1
            log(f"    {n} pairs from {tar_name}")
        finally:
            tmp.unlink(missing_ok=True)


def _nyu_stats(out):
    depths = sorted((out / "depth").glob("*.npy"))
    if not depths:
        return
    d = np.load(depths[0])
    log(f"    sample {depths[0].stem}: {d.shape} {d.dtype} min={d.min():.3f} max={d.max():.3f} m")


def fetch_nyu():
    have = len(list((NYU_DIR / "depth").glob("*.npy"))) if (NYU_DIR / "depth").exists() else 0
    if have >= NYU_N_TEST:
        log(f"nyu: {have} images already present, skip")
        return
    for label, fn in [("densedepth zip", _nyu_from_zip), ("labeled mat", _nyu_from_mat), ("hf mirror", _nyu_from_hf)]:
        log(f"nyu: trying {label}")
        try:
            fn(NYU_DIR)
            _nyu_stats(NYU_DIR)
            have = len(list((NYU_DIR / "depth").glob("*.npy")))
            if have < NYU_N_TEST:
                raise RuntimeError(f"incomplete: {have}/{NYU_N_TEST}")
            log(f"nyu: ok, {have} images")
            return
        except Exception as e:
            log(f"nyu: {label} failed ({e})")
    raise SystemExit("nyu: all sources failed")


def _ddff_depth_pack():
    dest = DDFF_DIR / "DepthRegistered"
    if dest.exists() and any(dest.rglob("DEPTH_*.png")):
        log("ddff: registered depth pack already present, skip")
        return
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
        tmp = Path(f.name)
    try:
        log("ddff: downloading registered depth maps (~58 mb)")
        download(f"{DDFF_BASE}/{DDFF_DEPTH}", tmp)
        n = 0
        with tarfile.open(tmp, "r:gz") as tf:
            for member in tf:
                if not member.isfile() or "colored" in member.name or not member.name.endswith(".png"):
                    continue
                rel = Path(member.name).relative_to("DepthRegistered")
                out = DDFF_DIR / "DepthRegistered" / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(tf.extractfile(member).read())
                n += 1
        log(f"ddff: {n} depth maps -> {dest} (test dirs are empty: gt withheld)")
    finally:
        tmp.unlink(missing_ok=True)


def _ddff_calib():
    for rel in DDFF_CAL_FILES:
        dest = DDFF_DIR / "calib" / rel
        if dest.exists():
            continue
        try:
            download(f"{DDFF_CAL_BASE}/{rel}", dest)
        except Exception as e:
            log(f"ddff: calib {rel} failed ({e})")


def fetch_ddff(with_trainval=False):
    DDFF_DIR.mkdir(parents=True, exist_ok=True)
    dest = DDFF_DIR / DDFF_TEST
    if dest.exists() and dest.stat().st_size > 0:
        log(f"ddff: {dest.name} already present, skip")
    else:
        log(f"ddff: downloading {DDFF_TEST} (~761 mb)")
        download(f"{DDFF_BASE}/{DDFF_TEST}", dest)
        log(f"ddff: ok -> {dest}")
    _ddff_depth_pack()
    _ddff_calib()
    if with_trainval:
        dest = DDFF_DIR / DDFF_TRAINVAL
        log(f"ddff: downloading {DDFF_TRAINVAL} (~12.6 gb, this is large)")
        download(f"{DDFF_BASE}/{DDFF_TRAINVAL}", dest)
        log(f"ddff: ok -> {dest}")


def show():
    for name, d in [("nyu", NYU_DIR), ("ddff", DDFF_DIR)]:
        if not d.exists():
            log(f"{name}: absent")
            continue
        if name == "nyu":
            rgb = list((d / "rgb").glob("*.png")) if (d / "rgb").exists() else []
            dep = list((d / "depth").glob("*.npy")) if (d / "depth").exists() else []
            log(f"{name}: {len(rgb)} rgb, {len(dep)} depth")
        else:
            for f in sorted(d.glob("*.h5")):
                with h5py.File(f, "r") as h5:
                    keys = {k: tuple(h5[k].shape) for k in h5.keys()}
                log(f"{name}: {f.name} {f.stat().st_size / 2**20:.0f} mb {keys}")
            reg = d / "DepthRegistered"
            nreg = len(list(reg.rglob("DEPTH_*.png"))) if reg.exists() else 0
            log(f"{name}: {nreg} registered depth maps")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", nargs="?", choices=["nyu", "ddff", "all"])
    p.add_argument("--list", action="store_true", help="show what is present")
    p.add_argument("--trainval", action="store_true", help="ddff: also fetch the 12.6 gb trainval h5")
    args = p.parse_args()
    if args.list or not args.cmd:
        show()
        return
    if args.cmd in ("nyu", "all"):
        fetch_nyu()
    if args.cmd in ("ddff", "all"):
        fetch_ddff(args.trainval)


if __name__ == "__main__":
    sys.exit(main())

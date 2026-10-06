#!/usr/bin/env python
"""dav2 model-size + metric-head comparison on nyu eigen test split (654 imgs).

relative checkpoints (small/base/large): per-image affine alignment of
predicted inverse depth to gt inverse depth (lsq / huber-irls / median
scale-only). metric checkpoint: raw output is depth in meters, evaluated
unaligned (none) and with the lsq path for reference.

note: requested id depth-anything/Depth-Anything-V2-Metric-Hypersim-Small-hf
does not exist on hf; the hf-format hypersim (indoor) metric checkpoints are
named Metric-Indoor-*-hf, so we use ...-Metric-Indoor-Small-hf.

crops: full = all valid px (gt in (0,10]); eigen = rows 45:472, cols 41:602.
alignment is fit inside each crop's valid mask, matching eval protocol.
"""

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthbench.dav2 import DAV2
from depthbench.data.nyu import iter_nyu, nyu_test_dir

REPO_ROOT = Path(__file__).resolve().parents[2]
GT_MAX = 10.0
EIGEN_SLICES = (slice(45, 472), slice(41, 602))

MODELS = [
    ("dav2s", "depth-anything/Depth-Anything-V2-Small-hf", "rel"),
    ("dav2b", "depth-anything/Depth-Anything-V2-Base-hf", "rel"),
    ("dav2l", "depth-anything/Depth-Anything-V2-Large-hf", "rel"),
    ("hypersim", "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", "metric"),
]
VARIANTS = {
    "dav2s": ("lsq", "irls", "median"),
    "dav2b": ("lsq",),
    "dav2l": ("lsq",),
    "hypersim": ("none", "lsq"),
}
METHOD_ORDER = [f"{t}_{v}" for t, _, _ in MODELS for v in VARIANTS[t]]


class MetricDepth:
    """same loading pattern as depthbench.dav2 but returns raw meters."""

    def __init__(self, model_id, device):
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict_depth(self, rgb):
        pix = self.processor(images=[rgb], return_tensors="pt")["pixel_values"]
        pix = pix.to(self.device)
        pred = self.model(pixel_values=pix).predicted_depth.float()
        d = F.interpolate(
            pred[:, None], size=rgb.shape[:2], mode="bilinear", align_corners=False
        )[0, 0]
        return d.cpu().numpy().astype(np.float32)


def _sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


def _fit_lsq(x, y):
    a = np.stack([x, np.ones_like(x)], 1)
    s, t = np.linalg.lstsq(a, y, rcond=None)[0]
    return float(s), float(t)


def _fit_irls(x, y, iters=5, delta=1.345):
    a = np.stack([x, np.ones_like(x)], 1)
    beta = np.linalg.lstsq(a, y, rcond=None)[0]
    for _ in range(iters):
        r = y - a @ beta
        mad = np.median(np.abs(r - np.median(r)))
        u = r / max(1.4826 * mad, 1e-12)
        w = np.minimum(1.0, delta / np.abs(u))
        sw = np.sqrt(w)
        beta = np.linalg.lstsq(a * sw[:, None], y * sw, rcond=None)[0]
    return float(beta[0]), float(beta[1])


def _fit_median(x, y):
    ok = x > 1e-9
    if not ok.any():
        return 0.0, 0.0
    return float(np.median(y[ok] / x[ok])), 0.0


def _metrics(p, g):
    p = np.clip(p, 1e-3, GT_MAX)
    d = p - g
    ratio = np.maximum(p / g, g / p)
    return {
        "absrel": float(np.mean(np.abs(d) / g)),
        "rmse": float(np.sqrt(np.mean(d * d))),
        "delta1": float(np.mean(ratio < 1.25)),
    }


def _crop_masks(gt):
    base = np.isfinite(gt) & (gt > 0) & (gt <= GT_MAX)
    eig = np.zeros_like(base)
    eig[EIGEN_SLICES] = True
    return {"full": base, "eigen": base & eig}


def _inv_to_depth(q):
    return 1.0 / np.clip(q, 1e-3, None)


def _aggregate(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["crop"])].append(r)
    print(f"\n{'method':<15s} {'crop':<6s} {'absrel':>7s} {'rmse':>7s} {'d1':>7s} {'ms/img':>8s} n")
    for m in METHOD_ORDER:
        for c in ("full", "eigen"):
            g = groups.get((m, c))
            if not g:
                continue
            row = [np.mean([r[k] for r in g]) for k in ("absrel", "rmse", "delta1", "runtime_ms")]
            print(f"{m:<15s} {c:<6s} {row[0]:7.4f} {row[1]:7.4f} {row[2]:7.4f} {row[3]:8.1f} {len(g)}")
    return groups


def _bar_fig(groups, path):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })
    methods = [m for m in METHOD_ORDER if (m, "eigen") in groups]
    x = np.arange(len(methods))
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.6, 2.3), constrained_layout=True)
    for i, (crop, color) in enumerate((("full", "#8da0cb"), ("eigen", "#4a6fa5"))):
        vals = [np.mean([r["delta1"] for r in groups[(m, crop)]]) for m in methods]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, label=crop, color=color, edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=5.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel(r"$\delta_1$ (higher better)")
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path)
    plt.close(fig)


def _qual_fig(gal, order, path):
    n = len(order)
    fig, axes = plt.subplots(n, 4, figsize=(8.6, 2.0 * n), constrained_layout=True)
    if n == 1:
        axes = axes[None, :]
    titles = ["rgb", "gt depth (m)", "dav2s lsq (m)", "dav2l lsq (m)"]
    for i, name in enumerate(order):
        g = gal[name]
        ims = [g["rgb"], g["gt"], g["dav2s"], g["dav2l"]]
        for j, im in enumerate(ims):
            ax = axes[i, j]
            if j == 0:
                ax.imshow(im)
            else:
                ax.imshow(np.where(g["mask"], im, np.nan), cmap="turbo_r", vmin=0, vmax=GT_MAX)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(titles[j], fontsize=8)
        axes[i, 0].set_ylabel(name, fontsize=6, rotation=0, ha="right", va="center")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--nyu-root", default=None)
    ap.add_argument("--out", default=str(REPO_ROOT / "results"))
    args = ap.parse_args()

    root = Path(args.nyu_root) if args.nyu_root else nyu_test_dir()
    out = Path(args.out)
    fig_dir = out / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}  root: {root}")
    print("note: Metric-Hypersim-Small-hf not on hf -> using Metric-Indoor-Small-hf (hypersim indoor)")

    rows = []
    gal, gal_order = {}, []

    for tag, model_id, kind in MODELS:
        if kind == "rel":
            model = DAV2(model_id, device=device)
            predict = model.predict_inv
        else:
            model = MetricDepth(model_id, device)
            predict = model.predict_depth
        warmup = True
        for item in tqdm(iter_nyu(root, limit=args.limit), desc=tag, total=args.limit):
            rgb, gt = item["rgb"], item["depth"]
            if warmup:
                predict(rgb)
                _sync(model.device)
                warmup = False
            t0 = time.perf_counter()
            raw = predict(rgb)
            _sync(model.device)
            rt = (time.perf_counter() - t0) * 1e3

            inv = _inv_to_depth(raw) if kind == "metric" else raw.astype(np.float64)
            masks = _crop_masks(gt)
            for crop, m in masks.items():
                n = int(m.sum())
                if n < 100:
                    continue
                x = inv[m].astype(np.float64)
                y = 1.0 / gt[m].astype(np.float64)
                fits = {}
                for v in VARIANTS[tag]:
                    if v == "lsq":
                        fits[v] = _fit_lsq(x, y)
                    elif v == "irls":
                        fits[v] = _fit_irls(x, y)
                    elif v == "median":
                        fits[v] = _fit_median(x, y)
                for v in VARIANTS[tag]:
                    if v == "none":
                        p = raw[m].astype(np.float64)
                    else:
                        s, t = fits[v]
                        p = _inv_to_depth(s * x + t)
                    mt = _metrics(p, gt[m].astype(np.float64))
                    rows.append({
                        "method": f"{tag}_{v}",
                        "model": model_id,
                        "crop": crop,
                        "img": item["name"],
                        "absrel": mt["absrel"],
                        "rmse": mt["rmse"],
                        "delta1": mt["delta1"],
                        "runtime_ms": rt,
                    })

            if tag in ("dav2s", "dav2l"):
                if len(gal_order) < 3 and item["name"] not in gal and tag == "dav2s":
                    m = masks["full"]
                    s, t = _fit_lsq(inv[m].astype(np.float64), 1.0 / gt[m].astype(np.float64))
                    gal[item["name"]] = {
                        "rgb": rgb,
                        "gt": np.clip(gt, 0, GT_MAX),
                        "mask": m,
                        "dav2s": np.clip(_inv_to_depth(s * inv + t), 0, GT_MAX),
                    }
                    gal_order.append(item["name"])
                elif tag == "dav2l" and item["name"] in gal and "dav2l" not in gal[item["name"]]:
                    m = masks["full"]
                    s, t = _fit_lsq(inv[m].astype(np.float64), 1.0 / gt[m].astype(np.float64))
                    gal[item["name"]]["dav2l"] = np.clip(_inv_to_depth(s * inv + t), 0, GT_MAX)

        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    if not rows:
        sys.exit("no images scored")
    csv_path = out / "nyu_models.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "model", "crop", "img", "absrel", "rmse", "delta1", "runtime_ms"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} rows)")

    groups = _aggregate(rows)

    fig_path = fig_dir / "nyu_models.pdf"
    _bar_fig(groups, fig_path)
    print(f"wrote {fig_path}")

    if gal_order:
        qual_path = fig_dir / "nyu_models_qual.pdf"
        _qual_fig(gal, gal_order, qual_path)
        print(f"wrote {qual_path}")


if __name__ == "__main__":
    main()

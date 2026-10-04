#!/usr/bin/env python
"""da-v2 baseline on nyu test split.

for each image: predict inverse depth, fit s,t by least squares so
s*pred_inv + t ~= 1/gt on valid pixels, evaluate resulting metric depth.
also evaluates a cheap median-ratio scale-only fit for comparison.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthbench.dav2 import DAV2
from depthbench.data.nyu import iter_nyu, nyu_test_dir

REPO_ROOT = Path(__file__).resolve().parents[2]
GT_MAX = 10.0
METRICS = ["absrel", "sqrel", "rmse", "rmse_log", "d1", "d2", "d3"]


def _valid_mask(gt):
    return np.isfinite(gt) & (gt > 0) & (gt <= GT_MAX)


def _fit_affine(pred_inv, gt, mask):
    # least squares for s*pred_inv + t ~= 1/gt
    x = pred_inv[mask].astype(np.float64)
    y = 1.0 / gt[mask].astype(np.float64)
    a = np.stack([x, np.ones_like(x)], axis=1)
    (s, t), *_ = np.linalg.lstsq(a, y, rcond=None)
    return float(s), float(t)


def _fit_median(pred_inv, gt, mask):
    # scale-only fit: s = median(gt_inv / pred_inv), t = 0
    x = pred_inv[mask].astype(np.float64)
    y = 1.0 / gt[mask].astype(np.float64)
    ok = x > 1e-6
    if not ok.any():
        return 0.0
    return float(np.median(y[ok] / x[ok]))


def _inv_to_depth(inv_aligned):
    return 1.0 / np.clip(inv_aligned, 1e-3, None)


def _metrics(pred, gt, mask):
    p = np.clip(pred[mask].astype(np.float64), 1e-3, GT_MAX)
    g = gt[mask].astype(np.float64)
    diff = p - g
    ratio = np.maximum(p / g, g / p)
    return {
        "absrel": float(np.mean(np.abs(diff) / g)),
        "sqrel": float(np.mean(diff**2 / g)),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "rmse_log": float(np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))),
        "d1": float(np.mean(ratio < 1.25)),
        "d2": float(np.mean(ratio < 1.25**2)),
        "d3": float(np.mean(ratio < 1.25**3)),
    }


def _score(item, pred_inv):
    gt = item["depth"]
    mask = _valid_mask(gt)
    n = int(mask.sum())
    if n < 100:
        return None
    s, t = _fit_affine(pred_inv, gt, mask)
    pred_depth = _inv_to_depth(s * pred_inv + t)
    m = _metrics(pred_depth, gt, mask)
    s_med = _fit_median(pred_inv, gt, mask)
    med_depth = _inv_to_depth(s_med * pred_inv)
    m_med = _metrics(med_depth, gt, mask)
    row = {"name": item["name"], "n_valid": n, "s": s, "t": t, "s_med": s_med}
    row.update(m)
    row.update({f"med_{k}": v for k, v in m_med.items()})
    return row, gt, mask, pred_depth


def _write_csv(rows, path):
    cols = (
        ["name", "n_valid", "s", "t"]
        + METRICS
        + ["s_med"]
        + [f"med_{m}" for m in METRICS]
    )
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _print_aggregate(rows):
    print(f"\n{len(rows)} images")
    for prefix in ("", "med_"):
        label = "least-squares affine" if not prefix else "median-ratio scale-only"
        print(f"  {label}")
        for m in METRICS:
            v = np.array([r[f"{prefix}{m}"] for r in rows])
            print(f"    {m:>9s}  {v.mean():.4f} +- {v.std():.4f}")


def _gallery(gallery, path):
    n = len(gallery)
    fig, axes = plt.subplots(n, 4, figsize=(11, 2.1 * n), constrained_layout=True)
    if n == 1:
        axes = axes[None, :]
    titles = ["rgb", "gt depth (m)", "aligned pred (m)", "rel err"]
    for i, (rgb, gt, pred, err) in enumerate(gallery):
        for j, im in enumerate([rgb, gt, pred, err]):
            ax = axes[i, j]
            if j == 0:
                ax.imshow(rgb)
            elif j == 3:
                ax.imshow(np.clip(err, 0, 0.5), cmap="magma", vmin=0, vmax=0.5)
            else:
                ax.imshow(im, cmap="turbo_r", vmin=0, vmax=GT_MAX)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(titles[j], fontsize=9)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--nyu-root", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--save-preds", action="store_true")
    ap.add_argument("--out", default=str(REPO_ROOT / "results"))
    args = ap.parse_args()

    root = Path(args.nyu_root) if args.nyu_root else nyu_test_dir()
    out = Path(args.out)
    fig_dir = out / "figs"
    pred_dir = out / "preds" / "nyu_dav2"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if args.save_preds:
        pred_dir.mkdir(parents=True, exist_ok=True)

    model = DAV2(device=args.device)
    print(f"device: {model.device}  root: {root}")

    rows = []
    gallery = []
    batch = []

    def flush():
        invs = model.predict_inv([b["rgb"] for b in batch])
        for b, inv in zip(batch, invs):
            res = _score(b, inv)
            if res is None:
                print(f"  skip {b['name']}: <100 valid px")
                continue
            row, gt, mask, pred_depth = res
            rows.append(row)
            if len(gallery) < 8:
                err = np.full(gt.shape, np.nan, np.float32)
                err[mask] = np.abs(pred_depth - gt)[mask] / gt[mask]
                gallery.append((b["rgb"], np.where(mask, gt, np.nan), np.where(mask, pred_depth, np.nan), err))
            if args.save_preds:
                np.savez_compressed(
                    pred_dir / f"{b['name'].replace('/', '_')}.npz",
                    pred_inv=inv.astype(np.float32),
                    s=row["s"],
                    t=row["t"],
                )
        batch.clear()

    for item in tqdm(iter_nyu(root, limit=args.limit), total=args.limit):
        batch.append(item)
        if len(batch) >= args.batch_size:
            flush()
    flush()

    if not rows:
        sys.exit("no images scored")
    csv_path = out / "nyu_dav2.csv"
    _write_csv(rows, csv_path)
    print(f"wrote {csv_path}")
    _print_aggregate(rows)
    gal_path = fig_dir / "nyu_dav2_gallery.pdf"
    _gallery(gallery, gal_path)
    print(f"wrote {gal_path}")


if __name__ == "__main__":
    main()

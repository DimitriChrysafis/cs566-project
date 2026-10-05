import csv
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, sobel

METRIC_KEYS = ("absrel", "sqrel", "rmse", "rmse_log", "delta1", "delta2", "delta3", "n")


def valid_mask(gt, min_depth=1e-3, max_depth=10.0):
    gt = np.asarray(gt)
    return np.isfinite(gt) & (gt >= min_depth) & (gt <= max_depth)


def align_inv_to_gt(pred_inv, gt, mask=None):
    m = valid_mask(gt) & np.isfinite(pred_inv)
    if mask is not None:
        m &= mask
    a = np.asarray(pred_inv)[m].astype(np.float64)
    y = 1.0 / np.asarray(gt)[m].astype(np.float64)
    if a.size < 2 or np.ptp(a) < 1e-9:
        return (float(y.mean() / max(a.mean(), 1e-9)) if a.size else 1.0), 0.0
    A = np.stack([a, np.ones_like(a)], 1)
    s, t = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(s), float(t)


def align_median(pred_inv, gt, mask=None):
    m = valid_mask(gt) & np.isfinite(pred_inv) & (np.abs(pred_inv) > 1e-9)
    if mask is not None:
        m &= mask
    if m.sum() == 0:
        return 1.0
    r = (1.0 / np.asarray(gt)[m]) / np.asarray(pred_inv)[m]
    return float(np.median(r))


def inverse_to_depth(q, eps=1e-3):
    return 1.0 / np.clip(np.asarray(q, dtype=np.float64), eps, None)


def evaluate_depth(pred, gt, mask=None, min_depth=1e-3, max_depth=10.0):
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    m = valid_mask(gt, min_depth, max_depth) & np.isfinite(pred) & (pred > 0)
    if mask is not None:
        m &= np.asarray(mask, bool)
    n = int(m.sum())
    out = dict.fromkeys(METRIC_KEYS, np.nan)
    out["n"] = n
    if n == 0:
        return out
    p, g = pred[m], gt[m]
    d = p - g
    out["absrel"] = float(np.mean(np.abs(d) / g))
    out["sqrel"] = float(np.mean(d * d / g))
    out["rmse"] = float(np.sqrt(np.mean(d * d)))
    out["rmse_log"] = float(np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2)))
    ratio = np.maximum(p / g, g / p)
    for k, key in ((1, "delta1"), (2, "delta2"), (3, "delta3")):
        out[key] = float(np.mean(ratio < 1.25**k))
    return out


def summarize(rows):
    rows = list(rows)
    out = {"n_scenes": len(rows)}
    if not rows:
        return out
    keys = [k for k in rows[0] if k not in ("scene", "method", "split", "param")]
    for k in keys:
        vals = np.array([r[k] for r in rows if k in r], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[k] = float(vals.mean())
            out[k + "_std"] = float(vals.std())
    return out


def texture_split(rgb, thresh=None):
    # fixed rule (chosen before evaluation, not tuned per scene):
    # sobel gradient magnitude on gray, smoothed with gaussian sigma=1.5 px,
    # threshold at the median magnitude. low = below median, high = at/above.
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim == 3:
        g = rgb @ np.array([0.299, 0.587, 0.114])
    else:
        g = rgb
    mag = np.hypot(sobel(g, 0), sobel(g, 1))
    mag = gaussian_filter(mag, 1.5)
    if thresh is None:
        thresh = float(np.median(mag))
    high = mag >= thresh
    return ~high, high


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    for r in rows[1:]:
        fields += [k for k in r if k not in fields]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with Path(path).open() as f:
        return [{k: _to_float(v) for k, v in r.items()} for r in csv.DictReader(f)]

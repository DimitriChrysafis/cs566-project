from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.6})

_CMAP = {"depth": "turbo_r", "err": "magma", "conf": "viridis"}
_KINDS = ("img", "depth", "err", "conf")


def _apply_cmap(arr, cmap, vmin, vmax):
    arr = np.asarray(arr, np.float64)
    norm = np.clip((arr - vmin) / max(vmax - vmin, 1e-12), 0, 1)
    rgba = plt.get_cmap(cmap)(norm)
    rgba[~np.isfinite(arr), :3] = 0.0
    return (rgba[..., :3] * 255).astype(np.uint8)


def depth_rgb(depth, vmin=None, vmax=None):
    d = np.asarray(depth, np.float64)
    if vmin is None:
        vmin = float(np.nanmin(d))
    if vmax is None:
        vmax = float(np.nanmax(d))
    return _apply_cmap(d, _CMAP["depth"], vmin, vmax)


def err_rgb(rel, vmax=0.5):
    return _apply_cmap(np.asarray(rel), _CMAP["err"], 0.0, vmax)


def conf_rgb(conf):
    return _apply_cmap(np.asarray(conf), _CMAP["conf"], 0.0, 1.0)


def _panel_spec(item):
    for k in _KINDS:
        if k in item:
            return k, item[k]
    raise KeyError(f"panel needs one of {_KINDS}, got {sorted(item)}")


def _show(ax, item):
    kind, data = _panel_spec(item)
    if kind == "img":
        ax.imshow(np.asarray(data))
        return None
    if kind == "depth":
        im = ax.imshow(data, cmap=_CMAP["depth"], vmin=item.get("vmin"), vmax=item.get("vmax"))
    elif kind == "err":
        im = ax.imshow(data, cmap=_CMAP["err"], vmin=0, vmax=item.get("vmax", 0.5))
    else:
        im = ax.imshow(data, cmap=_CMAP["conf"], vmin=0, vmax=1)
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.03)
    cb = plt.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=5, length=1.5, pad=1)
    if "ticks" in item:
        cb.set_ticks(item["ticks"])
    return im


def save_panels(items, path, ncols=None, suptitle=None):
    n = len(items)
    ncols = ncols or int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.42 * ncols, 1.42 * nrows), squeeze=False)
    for i, item in enumerate(items):
        ax = axes[i // ncols, i % ncols]
        _show(ax, item)
        ax.set_title(item.get("title", ""), fontsize=7, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
    for i in range(n, nrows * ncols):
        axes[i // ncols, i % ncols].axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=8, y=1.0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_gallery(rows, path):
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(5.8, 1.45 * n), squeeze=False)
    for i, (rgb, gt, pred, err) in enumerate(rows):
        lo = float(np.nanmin(gt[np.isfinite(gt)]))
        hi = float(np.nanmax(gt[np.isfinite(gt)]))
        panels = [
            dict(img=rgb, title="rgb" if i == 0 else ""),
            dict(depth=gt, vmin=lo, vmax=hi, title="gt depth" if i == 0 else ""),
            dict(depth=pred, vmin=lo, vmax=hi, title="pred depth" if i == 0 else ""),
            dict(err=err, vmax=0.5, title="rel err" if i == 0 else ""),
        ]
        for j, item in enumerate(panels):
            ax = axes[i, j]
            _show(ax, item)
            ax.set_title(item["title"], fontsize=7, pad=2)
            ax.set_xticks([])
            ax.set_yticks([])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

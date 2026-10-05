import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from depthbench.data.ddff import list_scenes, load_dff_scene
except Exception:
    print("ddff not available")
    sys.exit(0)

from depthbench.dff import depth_from_focus

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
FIGS = RES / "figs"

plt.rcParams.update({"font.size": 6, "font.family": "serif", "axes.linewidth": 0.5})


def _get(d, k):
    return d.get(k) if isinstance(d, dict) else getattr(d, k, None)


def _absrel(est, gt, m):
    return float(np.mean(np.abs(est - gt)[m] / gt[m]))


def _rmse(est, gt, m):
    return float(np.sqrt(np.mean((est - gt)[m] ** 2)))


def _delta1(est, gt, m, thr=1.25):
    r = np.maximum(est / gt, gt / est)
    return float(np.mean(r[m] < thr))


def _fig(name, stack, gt, est, conf, path):
    fig = plt.figure(figsize=(7.2, 1.55))
    gs = fig.add_gridspec(1, 5, wspace=0.18)
    ax = [fig.add_subplot(gs[0, i]) for i in range(5)]
    ax[0].imshow(np.clip(stack[0], 0, 1) if stack.ndim == 4 else stack[0], cmap="gray")
    ax[1].imshow(np.clip(stack[-1], 0, 1) if stack.ndim == 4 else stack[-1], cmap="gray")
    if gt is not None:
        ax[2].imshow(gt, cmap="turbo_r")
        ax[3].imshow(est, cmap="turbo_r")
    else:
        ax[2].imshow(est, cmap="turbo_r")
        ax[3].imshow(est, cmap="turbo_r")
    cf = ax[4].imshow(conf, cmap="viridis", vmin=0, vmax=1)
    for a, t in zip(ax, ["frame 0", f"frame {len(stack) - 1}", "gt/est", "est", "conf"]):
        a.set_title(t, fontsize=6, pad=2)
        a.set_xticks([])
        a.set_yticks([])
    cax = make_axes_locatable(ax[4]).append_axes("right", size="6%", pad=0.03)
    cb = fig.colorbar(cf, cax=cax)
    cb.ax.tick_params(labelsize=4.5, length=1.5, pad=1)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    scenes = list(list_scenes() or [])
    if not scenes:
        print("ddff not available")
        return
    FIGS.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in scenes:
        d = load_dff_scene(None, name)
        if _get(d, "stack") is None:
            continue
        stack = np.asarray(_get(d, "stack"), np.float32)
        zfs = np.asarray(_get(d, "focdist"), np.float64)
        gt = _get(d, "depth")
        gt = np.asarray(gt, np.float32) if gt is not None else None

        est, conf = depth_from_focus(stack, zfs)
        row = dict(scene=str(name), n_frames=len(zfs), conf=float(conf.mean()))
        if gt is not None:
            valid = np.isfinite(gt) & (gt > 0)
            row.update(absrel=_absrel(est, gt, valid), rmse=_rmse(est, gt, valid),
                       delta1=_delta1(est, gt, valid), n_px=int(valid.sum()))
        else:
            row.update(absrel=np.nan, rmse=np.nan, delta1=np.nan, n_px=int(est.size))
        rows.append(row)
        print(f"{name}: frames={row['n_frames']} absrel={row['absrel']:.3f} "
              f"rmse={row['rmse']:.3f} delta1={row['delta1']:.3f} conf={row['conf']:.3f}")
        if len(rows) <= 6:
            _fig(str(name), stack, gt, est, conf, FIGS / f"ddff_dff_{name}.pdf")

    with open(RES / "ddff_dff.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["scene", "n_frames", "absrel", "rmse", "delta1", "conf", "n_px"])
        wtr.writeheader()
        for r in rows:
            wtr.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {RES / 'ddff_dff.csv'}")


if __name__ == "__main__":
    main()

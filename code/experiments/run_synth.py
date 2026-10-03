import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depthbench.synth import make_scene, render_stack, SCENE_KINDS
from depthbench.dff import depth_from_focus

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
FIGS = RES / "figs"
ZFS = np.geomspace(0.9, 7.0, 10)

plt.rcParams.update({"font.size": 6, "font.family": "serif", "axes.linewidth": 0.5})


def _absrel(est, gt, m):
    return float(np.mean(np.abs(est - gt)[m] / gt[m]))


def _rmse(est, gt, m):
    return float(np.sqrt(np.mean((est - gt)[m] ** 2)))


def _delta1(est, gt, m, thr=1.25):
    r = np.maximum(est / gt, gt / est)
    return float(np.mean(r[m] < thr))


def _metrics(est, gt, m):
    return _absrel(est, gt, m), _rmse(est, gt, m), _delta1(est, gt, m)


def _fig(scene, stack, est, conf, path):
    rel = np.abs(est - scene.depth) / scene.depth
    z_lo, z_hi = float(scene.depth.min()), float(scene.depth.max())
    i_near = int(np.argmin(np.abs(ZFS - 1.1)))
    i_far = int(np.argmin(np.abs(ZFS - 5.0)))

    fig = plt.figure(figsize=(8.6, 1.55))
    gs = fig.add_gridspec(1, 6, wspace=0.18)
    ax = [fig.add_subplot(gs[0, i]) for i in range(6)]
    ax[0].imshow(stack[i_near])
    ax[1].imshow(stack[i_far])
    im = ax[2].imshow(scene.depth, cmap="turbo_r", vmin=z_lo, vmax=z_hi)
    ax[3].imshow(est, cmap="turbo_r", vmin=z_lo, vmax=z_hi)
    cf = ax[4].imshow(conf, cmap="viridis", vmin=0, vmax=1)
    er = ax[5].imshow(rel, cmap="magma", vmin=0, vmax=0.5)
    titles = [f"(a) focus {ZFS[i_near]:.1f} m", f"(b) focus {ZFS[i_far]:.1f} m",
              "(c) true depth", "(d) dff depth", "(e) conf", "(f) rel. error"]
    for a, t in zip(ax, titles):
        a.set_title(t, fontsize=6, pad=2)
        a.set_xticks([])
        a.set_yticks([])
    for i, ima in [(2, im), (4, cf), (5, er)]:
        cax = make_axes_locatable(ax[i]).append_axes("right", size="6%", pad=0.03)
        cb = fig.colorbar(ima, cax=cax)
        cb.ax.tick_params(labelsize=4.5, length=1.5, pad=1)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    rows = []
    for kind in SCENE_KINDS:
        scene = make_scene(kind, seed=0)
        stack = render_stack(scene, ZFS, noise=0.004, seed=0)
        est, conf = depth_from_focus(stack, ZFS)

        tl = scene.masks["textureless"]
        regions = {"all": np.ones_like(tl), "textured": ~tl, "textureless": tl}
        for reg, m in regions.items():
            a, r, d = _metrics(est, scene.depth, m)
            rows.append(dict(scene=kind, region=reg, absrel=a, rmse=r, delta1=d,
                             conf=float(conf[m].mean()), n_px=int(m.sum())))
            print(f"{kind:10s} {reg:12s} absrel={a:.3f} rmse={r:.3f} delta1={d:.3f} conf={conf[m].mean():.3f}")

        _fig(scene, stack, est, conf, FIGS / f"synth_dff_{kind}.pdf")
        np.savez_compressed(RES / f"synth_{kind}.npz", depth=scene.depth, z_est=est,
                            conf=conf, zfs=ZFS,
                            **{f"mask_{k}": v for k, v in scene.masks.items()})
        print(f"  -> {FIGS / f'synth_dff_{kind}.pdf'}")

    with open(RES / "synth_dff.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["scene", "region", "absrel", "rmse", "delta1", "conf", "n_px"])
        wtr.writeheader()
        for r in rows:
            wtr.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {RES / 'synth_dff.csv'}")


if __name__ == "__main__":
    main()

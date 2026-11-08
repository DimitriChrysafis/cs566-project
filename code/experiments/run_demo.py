import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from depthbench import dff, metrics, refine, synth, viz
import run_refine as base

RESULTS = Path(__file__).resolve().parents[2] / "results"
KIND, SEED = "slanted", 0
W, H, T = 480, 320, 10
ZF_LO, ZF_HI = 0.9, 7.0


def main():
    t0 = time.time()
    scene = synth.make_scene(KIND, seed=SEED, w=W, h=H)
    zfs = np.geomspace(ZF_LO, ZF_HI, T)
    stack = synth.render_stack(scene, zfs, noise=0.004, seed=SEED)
    print(f"scene: {KIND} {W}x{H}, {T} frames, focus {ZF_LO}-{ZF_HI} m "
          f"(rendered {time.time() - t0:.1f}s)")

    pipe = dff.dff_pipeline(stack, zfs)
    rgb = base.sharp_ref(stack)
    gt = np.asarray(scene.depth, np.float64)
    d_inv = base.learned_inv(rgb, gt, SEED)
    mask = metrics.valid_mask(gt)

    fs = refine.fit_scale(d_inv, pipe["z"], pipe["conf"])
    ref = refine.refine(d_inv, pipe["z"], pipe["conf"], lam=base.LAM)
    print(f"scale fit: s={fs['s']:.3f} t={fs['t']:.3f} coverage={fs['coverage']:.2f} "
          f"calibrated={ref['calibrated']} niter={ref['niter']}")

    preds = {
        "focus_raw": pipe["z_raw"],
        "focus": pipe["z"],
        "scale_fit": refine.to_depth(fs["s"] * d_inv + fs["t"]),
        "refined": refine.to_depth(ref["q"]),
    }
    print(f"\n{'method':<12s} {'absrel':>7s} {'rmse':>7s} {'delta1':>7s} {'n':>8s}")
    for name, z in preds.items():
        r = metrics.evaluate_depth(z, gt, mask)
        print(f"{name:<12s} {r['absrel']:7.4f} {r['rmse']:7.3f} "
              f"{r['delta1']:7.4f} {r['n']:8d}")

    z = preds["refined"]
    err = np.abs(z - gt) / gt
    fig = RESULTS / "figs" / f"demo_{KIND}.pdf"
    viz.save_panels([
        dict(img=rgb, title="sharpest frame"),
        dict(depth=gt, vmin=1, vmax=6, title="gt depth"),
        dict(depth=pipe["z"], vmin=1, vmax=6, title="focus $z_F$"),
        dict(conf=pipe["conf"], title="conf $w$"),
        dict(depth=z, vmin=1, vmax=6, title="refined"),
        dict(err=err, vmax=0.5, title="rel err"),
    ], fig, ncols=6, suptitle=f"{KIND}: dff + learned prior + refinement")
    print(f"\nwrote {fig}\ntotal {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

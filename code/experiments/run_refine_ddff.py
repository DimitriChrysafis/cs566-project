import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthbench.data.ddff import list_scenes, load_dff_scene
from depthbench.dav2 import DAV2
from depthbench.dff import depth_from_focus, to_gray
from depthbench.refine import refine, to_depth
from depthbench.viz import save_panels

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
FIGS = RES / "figs"


def sharpest_frame(stack):
    fm = np.stack([np.abs(np.diff(to_gray(stack)[i])) for i in range(len(stack))])
    return int(np.argmax(fm.reshape(len(stack), -1).mean(1)))


def focdist_corr(z_f, d_inv, conf):
    """rank corr between focus depth and learned inverse-depth prior; used to
    verify the assumed ascending order of ddff refocus slices (they are not
    recorded in the test archive)."""
    m = np.isfinite(z_f) & np.isfinite(d_inv) & (conf > np.percentile(conf, 75))
    if m.sum() < 1000:
        return 0.0
    return float(spearmanr(-z_f[m].ravel(), d_inv[m].ravel()).statistic)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default="test_000,test_007,test_013,test_044,test_058,test_072")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    FIGS.mkdir(parents=True, exist_ok=True)
    model = DAV2(device=args.device)
    available = set(list_scenes())
    order_checked = False
    for name in [s.strip() for s in args.scenes.split(",") if s.strip() in available]:
        d = load_dff_scene(None, name)
        stack = np.asarray(d["stack"], np.float32) / 255.0
        zfs = np.asarray(d["focdist"], np.float64)

        k = sharpest_frame(stack)
        rgb = (np.clip(stack[k], 0, 1) * 255).astype(np.uint8)
        d_inv = model.predict_inv(rgb)

        z_f, conf = depth_from_focus(stack, zfs)
        z_rev, conf_rev = depth_from_focus(stack, zfs[::-1])
        fwd, rev = focdist_corr(z_f, d_inv, conf), focdist_corr(z_rev, d_inv, conf_rev)
        if rev > fwd:
            z_f, conf = z_rev, conf_rev
        if not order_checked:
            print(f"focdist order check ({name}): forward rho={fwd:.3f} reversed rho={rev:.3f}"
                  f" -> using {'reversed' if rev > fwd else 'forward'}")
            order_checked = True

        out = refine(d_inv, z_f, conf, norm=True)
        z_ref = to_depth(out["q"])

        print(f"{name}: calibrated={out['calibrated']} s={out['s']:.3f} t={out['t']:.4f} "
              f"conf={conf.mean():.3f} rho_fwd={fwd:.3f} rho_rev={rev:.3f}")
        save_panels([
            dict(img=rgb, title="sharpest frame"),
            dict(depth=z_f, title="focus z_F"),
            dict(conf=conf, title="conf"),
            dict(depth=z_ref, title="dav2+refine"),
        ], FIGS / f"ddff_refine_{name}.pdf")


if __name__ == "__main__":
    main()

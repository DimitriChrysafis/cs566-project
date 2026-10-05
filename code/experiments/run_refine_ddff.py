import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthbench.data.ddff import list_scenes, load_dff_scene
from depthbench.dav2 import DAV2
from depthbench.dff import depth_from_focus, to_gray
from depthbench.metrics import texture_split
from depthbench.refine import fit_scale, refine, to_depth
from depthbench.viz import save_panels

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
FIGS = RES / "figs"


def sharpest_frame(stack):
    fm = np.stack([np.abs(np.diff(to_gray(stack)[i])) for i in range(len(stack))])
    return int(np.argmax(fm.reshape(len(stack), -1).mean(1)))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default="test_000,test_007,test_013,test_044")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    FIGS.mkdir(parents=True, exist_ok=True)
    model = DAV2(device=args.device)
    available = set(list_scenes())
    for name in [s.strip() for s in args.scenes.split(",") if s.strip() in available]:
        d = load_dff_scene(None, name)
        stack = np.asarray(d["stack"], np.float32) / 255.0
        zfs = np.asarray(d["focdist"], np.float64)
        z_f, conf = depth_from_focus(stack, zfs)

        k = sharpest_frame(stack)
        rgb = (np.clip(stack[k], 0, 1) * 255).astype(np.uint8)
        d_inv = model.predict_inv(rgb)

        fit = fit_scale(d_inv, z_f, conf)
        out = refine(d_inv, z_f, conf)
        z_ref = to_depth(out["q"])
        low, _ = texture_split(rgb)

        print(f"{name}: calibrated={out['calibrated']} s={out['s']:.3f} t={out['t']:.4f} "
              f"conf={conf.mean():.3f} lowtex={low.mean():.2f}")
        save_panels([
            dict(img=rgb, title="sharpest frame"),
            dict(depth=z_f, title="focus z_F"),
            dict(conf=conf, title="conf"),
            dict(depth=z_ref, title="dav2+refine"),
        ], FIGS / f"ddff_refine_{name}.pdf")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Regenerate the qualitative PNG assets in webpage/assets/ from results.

Recomputes the pipeline for the two showcase synthetic scenes and one
ddff stack so the figures always reflect the current code. Nyu panels
reuse the stored per-image predictions from run_dav2_nyu --save-preds.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "experiments"))

from depthbench import metrics, viz
from depthbench.data.nyu import iter_nyu

RES = ROOT / "results"
ASSETS = ROOT / "webpage" / "assets"
NYU_IDS = ["00001", "00009", "00014"]


def nyu_panels():
    rows = []
    want = set(NYU_IDS)
    for item in iter_nyu():
        if item["name"] not in want:
            continue
        pred = np.load(RES / "preds" / "nyu_dav2" / f"{item['name']}.npz")
        gt = np.asarray(item["depth"], np.float64)
        z = metrics.inverse_to_depth(pred["s"] * pred["pred_inv"] + pred["t"])
        err = np.abs(z - gt) / np.clip(gt, 1e-6, None)
        err[gt <= 0] = 0
        rows.append((item["rgb"], gt, z, err))
    viz.save_gallery(rows, ASSETS / "nyu_qualitative.png")


def synth_panels():
    from run_refine import predictions, run_scene
    for kind, fname in (("corridor", "synth_refinement.png"),
                        ("slanted", "failure_cases.png")):
        data = run_scene(kind, seed=1 if kind == "corridor" else 2)
        preds, ref = predictions(data)
        gt, z = data["gt"], preds["combined"]
        err = np.abs(z - gt) / gt
        viz.save_panels([
            dict(img=data["rgb"], title="rgb"),
            dict(depth=gt, vmin=1, vmax=6, title="gt depth"),
            dict(depth=data["z_f"], vmin=1, vmax=6, title="focus $z_F$"),
            dict(conf=data["conf"], title="conf $w$"),
            dict(depth=z, vmin=1, vmax=6,
                 title="combined" if ref["calibrated"] else "combined (fell back to focus)"),
            dict(err=err, vmax=0.5, title="rel err"),
        ], ASSETS / fname, ncols=6,
            suptitle=f"{data['name']} (calibrated={ref['calibrated']})")


def ddff_panels():
    from depthbench.data.ddff import load_dff_scene
    from depthbench.dav2 import DAV2
    from depthbench.dff import depth_from_focus
    from depthbench.refine import refine, to_depth
    from run_refine_ddff import focdist_corr, sharpest_frame

    d = load_dff_scene(None, "test_013")
    stack = np.asarray(d["stack"], np.float32) / 255.0
    zfs = np.asarray(d["focdist"], np.float64)
    rgb = (np.clip(stack[sharpest_frame(stack)], 0, 1) * 255).astype(np.uint8)
    d_inv = DAV2().predict_inv(rgb)
    z_f, conf = depth_from_focus(stack, zfs)
    z_rev, conf_rev = depth_from_focus(stack, zfs[::-1])
    if focdist_corr(z_rev, d_inv, conf_rev) > focdist_corr(z_f, d_inv, conf):
        z_f, conf = z_rev, conf_rev
    out = refine(d_inv, z_f, conf, norm=True)
    viz.save_panels([
        dict(img=rgb, title="sharpest frame"),
        dict(depth=z_f, title="focus $z_F$"),
        dict(conf=conf, title="conf"),
        dict(depth=to_depth(out["q"]), title="dav2+refine"),
    ], ASSETS / "ddff_qualitative.png")


def main():
    ASSETS.mkdir(exist_ok=True)
    nyu_panels()
    synth_panels()
    ddff_panels()
    import shutil
    mid = ROOT / "report" / "midterm_report.pdf"
    if mid.exists():
        shutil.copy(mid, ASSETS / "midterm.pdf")
    print("assets written to", ASSETS)


if __name__ == "__main__":
    main()

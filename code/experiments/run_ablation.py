import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from depthbench import metrics, refine
import run_refine as base

RESULTS = Path(__file__).resolve().parents[2] / "results"
ABL_SCENES = sorted(base.TUNE)


def eval_rows(scene, param, method, z, gt, splits, calibrated=True):
    return [dict(scene=scene, param=param, method=method, split=name,
                 calibrated=calibrated, **metrics.evaluate_depth(z, gt, sm))
            for name, sm in splits.items()]


def abl_stacksize():
    from depthbench import dff, synth
    rows = []
    for i, kind in enumerate(ABL_SCENES):
        scene = synth.make_scene(kind, seed=i, w=base.W, h=base.H)
        gt = np.asarray(scene.depth, np.float64)
        mask = metrics.valid_mask(gt)
        for n in (2, 3, 5, 10):
            zfs = np.geomspace(base.ZF_LO, base.ZF_HI, n)
            stack = synth.render_stack(scene, zfs, noise=0.004, seed=i)
            pipe = dff.dff_pipeline(stack, zfs)
            z_f, conf = pipe["z"], pipe["conf"]
            rgb = base.sharp_ref(stack)
            low, high = metrics.texture_split(rgb)
            splits = {"all": mask, "high_tex": mask & high}
            d = base.learned_inv(rgb, gt, i)
            ref = refine.refine(d, z_f, conf, lam=base.LAM)
            rows += eval_rows(kind, n, "focus", z_f, gt, splits)
            rows += eval_rows(kind, n, "refined", refine.to_depth(ref["q"]), gt,
                              splits, ref["calibrated"])
        print(f"{kind}: stacksize done")
    return rows


def abl_conf():
    rows = []
    for i, kind in enumerate(ABL_SCENES):
        d = base.run_scene(kind, seed=i)
        gt = d["gt"]
        for tag, w in (("conf", d["conf"]), ("uniform", np.ones_like(d["conf"]))):
            ref = refine.refine(d["d_inv"], d["z_f"], w, lam=base.LAM)
            rows += eval_rows(kind, tag, "refined", refine.to_depth(ref["q"]), gt,
                              {k: d["splits"][k] for k in ("all", "high_tex")},
                              ref["calibrated"])
        print(f"{kind}: conf done")
    return rows


def abl_lam():
    rows = []
    for i, kind in enumerate(ABL_SCENES):
        d = base.run_scene(kind, seed=i)
        for lam in (0.0, 0.3, 1.0, 3.0, 10.0, 30.0):
            ref = refine.refine(d["d_inv"], d["z_f"], d["conf"], lam=lam)
            rows += eval_rows(kind, lam, "refined", refine.to_depth(ref["q"]), d["gt"],
                              {k: d["splits"][k] for k in ("all", "high_tex")},
                              ref["calibrated"])
        print(f"{kind}: lam done")
    return rows


def abl_mu():
    rows = []
    for i, kind in enumerate(ABL_SCENES):
        d = base.run_scene(kind, seed=i)
        for mu in (0.0, 1e-3, 1e-2, 1e-1):
            ref = refine.refine(d["d_inv"], d["z_f"], d["conf"], lam=base.LAM, mu=mu)
            rows += eval_rows(kind, mu, "refined", refine.to_depth(ref["q"]), d["gt"],
                              {k: d["splits"][k] for k in ("all", "high_tex")},
                              ref["calibrated"])
        print(f"{kind}: mu done")
    return rows


ABLATIONS = {"stacksize": abl_stacksize, "conf": abl_conf, "lam": abl_lam, "mu": abl_mu}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", required=True, choices=ABLATIONS)
    args = ap.parse_args()
    rows = ABLATIONS[args.ablation]()
    metrics.write_csv(rows, RESULTS / f"ablation_{args.ablation}.csv")
    for param in dict.fromkeys(r["param"] for r in rows):
        for method in dict.fromkeys(r["method"] for r in rows if r["param"] == param):
            sel = [r for r in rows if r["param"] == param and r["method"] == method
                   and r["split"] == "all"]
            s = metrics.summarize(sel)
            print(f"{args.ablation}={param} {method:8s} absrel={s['absrel']:.4f} "
                  f"rmse={s['rmse']:.3f} delta1={s['delta1']:.3f}")


if __name__ == "__main__":
    main()

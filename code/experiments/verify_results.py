import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from depthbench import metrics
import run_refine as base

RESULTS = Path(__file__).resolve().parents[2] / "results"
LEARNED_METHODS = ("aligned_dav2", "metric_dav2", "scale_fit", "refined", "combined")
METHODS = ("const", "focus_raw", "focus") + LEARNED_METHODS
SPLITS = ("all", "low_tex", "high_tex")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="corridor", choices=base.SCENES)
    ap.add_argument("--seed", type=int, default=None,
                    help="default: index of scene in run_refine.SCENES")
    ap.add_argument("--csv", default=str(RESULTS / "refine_synth.csv"))
    ap.add_argument("--tol-absrel", type=float, default=5e-3)
    ap.add_argument("--tol-delta1", type=float, default=1e-2)
    ap.add_argument("--tol-rmse", type=float, default=2e-2)
    ap.add_argument("--exact-tol", type=float, default=1e-6,
                    help="tolerance for learned-free rows (proved bit-exact)")
    ap.add_argument("--strict", action="store_true",
                    help="fail learned rows when the da-v2 stand-in is in use")
    ap.add_argument("--from-preds", action="store_true",
                    help="verify committed results/refine_preds/<scene>.npz "
                         "instead of re-running the learned models; "
                         "deterministic on any machine")
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else base.SCENES.index(args.scene)
    committed = {}
    for r in metrics.read_csv(args.csv):
        if r["scene"] == args.scene:
            committed[(r["method"], r["split"])] = r
    if not committed:
        sys.exit(f"no committed rows for {args.scene} in {args.csv}")

    if args.from_preds:
        f = RESULTS / "refine_preds" / f"{args.scene}.npz"
        if not f.exists():
            sys.exit(f"{f} missing; run run_refine.py once with the learned "
                     "models available to commit the priors")
        saved = np.load(f)
        data = base.run_scene(args.scene, seed=seed, load_models=False)
        data["d_inv"] = saved["d_inv"]
        if saved["z_metric"].size:
            data["z_metric"] = saved["z_metric"]
        standin = False
    else:
        data = base.run_scene(args.scene, seed=seed)
        standin = base.get_dav2() is None
    preds, ref = base.predictions(data)

    tols = {"absrel": args.tol_absrel, "sqrel": args.tol_absrel,
            "rmse": args.tol_rmse, "rmse_log": args.tol_delta1,
            "delta1": args.tol_delta1, "delta2": args.tol_delta1,
            "delta3": args.tol_delta1, "n": 0.0}
    keys = list(metrics.METRIC_KEYS)

    n_pass = n_fail = n_skip = 0
    print(f"scene={args.scene} seed={seed} csv={args.csv}")
    print(f"{'method':<13s} {'split':<9s} {'metric':<9s} {'computed':>11s} "
          f"{'committed':>11s} {'diff':>10s} {'tol':>8s}  status")
    for method in METHODS:
        learned = method in LEARNED_METHODS
        for split in SPLITS:
            c = committed.get((method, split))
            p = preds.get(method)
            if p is None:
                n_skip += 1
                print(f"{method:<13s} {split:<9s} {'':<9s} {'no model':>11s} "
                      f"{'':>11s} {'':>10s} {'':>8s}  SKIP")
                continue
            r = metrics.evaluate_depth(p, data["gt"], data["splits"][split])
            for k in keys:
                if c is None:
                    print(f"{method:<13s} {split:<9s} {k:<9s} {r[k]:>11.6f} "
                          f"{'missing':>11s} {'':>10s} {'':>8s}  FAIL")
                    n_fail += 1
                    continue
                diff = abs(r[k] - float(c[k]))
                tol = tols[k] if learned else args.exact_tol
                ok = diff <= tol
                if ok:
                    status, n_pass = "PASS", n_pass + 1
                elif learned and standin and not args.strict:
                    status, n_skip = "SKIP", n_skip + 1
                else:
                    status, n_fail = "FAIL", n_fail + 1
                print(f"{method:<13s} {split:<9s} {k:<9s} {r[k]:>11.6f} "
                      f"{float(c[k]):>11.6f} {diff:>10.3e} {tol:>8.1e}  {status}")

    print(f"\ncalibrated={ref['calibrated']} niter={ref['niter']} "
          f"committed_calibrated={committed.get(('refined', 'all'), {}).get('calibrated')}")
    print(f"{n_pass} pass  {n_fail} fail  {n_skip} skip")
    print("VERIFY", "FAIL" if n_fail else "PASS")
    if standin:
        print("note: da-v2 unavailable, stand-in prior used; learned-dependent "
              "rows "
              + ("FAILED" if args.strict else "are marked SKIP")
              + " (committed csv used real da-v2). "
                "rerun with torch + hf checkpoint to verify them.")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()

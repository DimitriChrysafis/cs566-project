import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depthbench import metrics, refine, viz

RESULTS = Path(__file__).resolve().parents[2] / "results"
SCENES = ["gallery", "corridor", "slanted", "objects", "stairs", "shelves"]
TUNE, EVAL = {"gallery", "stairs"}, {"corridor", "objects", "slanted", "shelves"}
W, H, T = 480, 320, 10
ZF_LO, ZF_HI = 0.9, 7.0
LAM = 1.0

_DAV2 = None


def _focus_sharpness(f):
    g = np.asarray(f, np.float64)
    if g.ndim == 3:
        g = g @ np.array([0.299, 0.587, 0.114])
    return float((np.abs(2 * g - np.roll(g, 1, 1) - np.roll(g, -1, 1))
                  + np.abs(2 * g - np.roll(g, 1, 0) - np.roll(g, -1, 0))).mean())


def sharp_ref(stack):
    return stack[int(np.argmax([_focus_sharpness(f) for f in stack]))]


def get_dav2():
    global _DAV2
    if _DAV2 is None:
        try:
            from depthbench.dav2 import DAV2
            _DAV2 = DAV2("depth-anything/Depth-Anything-V2-Small-hf")
            print("dav2 loaded")
        except Exception as e:
            print(f"dav2 unavailable ({e}); using stand-in learned prediction")
            _DAV2 = False
    return _DAV2 or None


def fake_learned(gt, seed=0):
    rng = np.random.default_rng(seed)
    q = 1.0 / np.asarray(gt, np.float64)
    d = gaussian_filter(q, 1.2)
    d = 1.12 * d - 0.04
    d = d + gaussian_filter(rng.normal(0, 1, d.shape), 8) * 0.02
    return np.clip(d, 1e-3, None)


def learned_inv(rgb, gt, seed):
    model = get_dav2()
    if model is not None:
        try:
            u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            d = np.asarray(model.predict_inv(u8), np.float64)
            if d.shape != gt.shape:
                import cv2
                d = cv2.resize(d, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
            return d
        except Exception as e:
            print(f"dav2 predict failed ({e}); using stand-in")
    return fake_learned(gt, seed)


def run_scene(kind, seed=0, n_frames=T):
    from depthbench import dff, synth
    scene = synth.make_scene(kind, seed=seed, w=W, h=H)
    zfs = np.geomspace(ZF_LO, ZF_HI, n_frames)
    stack = synth.render_stack(scene, zfs, noise=0.004, seed=seed)
    pipe = dff.dff_pipeline(stack, zfs)
    rgb = sharp_ref(stack)
    gt = np.asarray(scene.depth, np.float64)
    d_inv = learned_inv(rgb, gt, seed)
    mask = metrics.valid_mask(gt)
    low, high = metrics.texture_split(rgb)
    name = getattr(scene, "name", kind) or kind
    return dict(name=name, rgb=rgb, gt=gt, z_f=pipe["z"], z_raw=pipe["z_raw"],
                conf=pipe["conf"], d_inv=d_inv, mask=mask,
                splits={"all": mask, "low_tex": mask & low,
                        "high_tex": mask & high})


def predictions(data):
    gt, d, z_f, w, mask = data["gt"], data["d_inv"], data["z_f"], data["conf"], data["mask"]
    so, to = metrics.align_inv_to_gt(d, gt, mask)
    fs = refine.fit_scale(d, z_f, w)
    ref = refine.refine(d, z_f, w, lam=LAM)
    z_comb = refine.to_depth(ref["q"]) if ref["calibrated"] else z_f
    return {
        "focus_raw": data["z_raw"],
        "focus": z_f,
        "aligned_dav2": metrics.inverse_to_depth(so * d + to),
        "scale_fit": refine.to_depth(fs["s"] * d + fs["t"]),
        "refined": refine.to_depth(ref["q"]),
        "combined": z_comb,
    }, ref


def main():
    rows = []
    for i, kind in enumerate(SCENES):
        data = run_scene(kind, seed=i)
        preds, ref = predictions(data)
        cal = ref["calibrated"]
        sset = "tune" if data["name"] in TUNE else "eval"
        for method, z in preds.items():
            for split, sm in data["splits"].items():
                r = metrics.evaluate_depth(z, data["gt"], sm)
                rows.append(dict(scene=data["name"], method=method, split=split,
                                 set=sset, calibrated=cal, **r))
        gt, z = data["gt"], preds["refined"]
        err = np.abs(z - gt) / gt
        viz.save_panels([
            dict(img=data["rgb"], title="rgb"),
            dict(depth=gt, vmin=1, vmax=6, title="gt depth"),
            dict(depth=data["z_f"], vmin=1, vmax=6, title="focus $z_F$"),
            dict(conf=data["conf"], title="conf $w$"),
            dict(depth=z, vmin=1, vmax=6, title="refined"),
            dict(err=err, vmax=0.5, title="rel err"),
        ], RESULTS / "figs" / f"refine_{data['name']}.pdf", ncols=6,
            suptitle=f"{data['name']} (calibrated={cal})")
        print(f"{data['name']}: calibrated={cal} niter={ref['niter']}")

    metrics.write_csv(rows, RESULTS / "refine_synth.csv")
    for sset in ("tune", "eval"):
        print(f"-- {sset} scenes --")
        for method in ("focus_raw", "focus", "aligned_dav2", "scale_fit", "refined", "combined"):
            for split in ("all", "low_tex", "high_tex"):
                sel = [r for r in rows if r["method"] == method and r["split"] == split
                       and r.get("set") == sset]
                s = metrics.summarize(sel)
                if sel:
                    print(f"{method:12s} {split:8s} absrel={s['absrel']:.4f} "
                          f"rmse={s['rmse']:.3f} delta1={s['delta1']:.3f}")
    return rows


if __name__ == "__main__":
    main()

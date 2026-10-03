import numpy as np
from scipy.ndimage import uniform_filter

_LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def to_gray(stack):
    stack = np.asarray(stack, np.float32)
    if stack.ndim == 4 and stack.shape[-1] == 3:
        return stack @ _LUMA
    return stack


def focus_measure(gray, win=11):
    g = np.pad(np.asarray(gray, np.float32), 1, mode="edge")
    ml = np.abs(2 * g[1:-1, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]) + \
         np.abs(2 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1])
    return uniform_filter(ml, win)


def depth_from_focus(stack, zfs, win=11, return_fm=False):
    """modified-laplacian depth from focus (nayar & nakagawa 1994).

    stack: (T,H,W,3) or (T,H,W) focal stack; zfs: (T,) focus distances in m.
    per-pixel argmax over the focus-measure volume, then parabolic sub-frame
    interpolation evaluated on the inverse-depth axis.

    confidence in [0,1]:
        prom = clip((fm[k] - mean_{t!=k} fm[t]) / (fm[k] + eps), 0, 1)
        curv = clip(2 * (2 fm[k] - fm[k-1] - fm[k+1]) / (fm[k-1] + 2 fm[k] + fm[k+1] + eps), 0, 1)
        tex  = fm[k] / (fm[k] + 0.02)
        conf = sqrt(prom * curv) * tex
    prom is the peak's prominence over the other frames, curv its sharpness,
    tex an absolute gate so noise-floor regions (fm ~ 0.01) get conf ~ 0.
    """
    gray = to_gray(stack)
    zfs = np.asarray(zfs, np.float64)
    fm = np.stack([focus_measure(f, win) for f in gray])
    T = len(zfs)
    k = np.clip(fm.argmax(0), 1, T - 2)
    yi, xi = np.indices(k.shape)
    a, b, c = fm[k - 1, yi, xi], fm[k, yi, xi], fm[k + 1, yi, xi]

    off = np.clip(0.5 * (a - c) / (a - 2 * b + c + 1e-9), -0.5, 0.5)
    t = np.clip(k + off, 0, T - 1)
    i0 = t.astype(int)
    frac = t - i0
    inv = 1.0 / zfs
    q = inv[i0] * (1 - frac) + inv[np.minimum(i0 + 1, T - 1)] * frac
    z_est = (1.0 / q).astype(np.float32)

    rest = (fm.sum(0) - b) / max(T - 1, 1)
    prom = np.clip((b - rest) / (b + 1e-9), 0, 1)
    curv = np.clip(2.0 * (2 * b - a - c) / (a + 2 * b + c + 1e-9), 0, 1)
    tex = b / (b + 0.02)
    conf = (np.sqrt(prom * curv) * tex).astype(np.float32)

    if return_fm:
        return z_est, conf, fm
    return z_est, conf

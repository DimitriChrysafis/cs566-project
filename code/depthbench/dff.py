import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

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


def all_focus(stack, fm=None, win=11):
    """approximate all-focused frame: per-pixel fm-weighted blend."""
    gray = to_gray(stack)
    if fm is None:
        fm = np.stack([focus_measure(f, win) for f in gray])
    w = fm / np.maximum(fm.sum(0, keepdims=True), 1e-12)
    if stack.ndim == 4:
        return (stack * w[..., None]).sum(0).astype(np.float32)
    return (gray * w).sum(0).astype(np.float32)


def smooth_depth(z_est, conf, guide=None, sig_s=4.0, sig_r=0.08, blend=True):
    """normalized convolution on inverse depth (knutsson & westin 1993):
    q_s = sum_j c_j K_s K_r q_j / sum_j c_j K_s K_r, q = 1/z_est.

    c is the focus confidence, K_s a spatial gaussian, K_r a range kernel
    on the guide image (all-focus gray) so confident edges survive.
    blend: keep each pixel's own estimate in proportion to conf so that
    high-confidence structure is not blurred.
    """
    q = 1.0 / np.maximum(np.asarray(z_est, np.float64), 1e-3)
    c = np.clip(np.asarray(conf, np.float64), 0, 1)
    h, w = q.shape
    rad = int(np.ceil(2.5 * sig_s))
    num = np.zeros_like(q)
    den = np.zeros_like(q)
    g = None
    if guide is not None:
        g = np.asarray(guide, np.float64)
        if g.ndim == 3:
            g = g @ _LUMA
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            ks = np.exp(-(dy * dy + dx * dx) / (2 * sig_s * sig_s))
            ys, xs = slice(max(0, dy), min(h, h + dy)), slice(max(0, dx), min(w, w + dx))
            src = (slice(max(0, -dy), min(h, h - dy)), slice(max(0, -dx), min(w, w - dx)))
            kj = ks * c[src]
            if g is not None:
                kj = kj * np.exp(-((g[src] - g[ys, xs]) ** 2) / (2 * sig_r * sig_r))
            num[ys, xs] += kj * q[src]
            den[ys, xs] += kj
    q_s = num / np.maximum(den, 1e-12)
    if blend:
        q_s = c * q + (1 - c) * q_s
    return (1.0 / np.maximum(q_s, 1e-3)).astype(np.float32)


def dff_pipeline(stack, zfs, win=11, sig_s=4.0, sig_r=0.08):
    """raw dff + confidence + normalized-convolution smoothed depth."""
    z_raw, conf, fm = depth_from_focus(stack, zfs, win=win, return_fm=True)
    guide = all_focus(stack, fm=fm)
    z_sm = smooth_depth(z_raw, conf, guide, sig_s=sig_s, sig_r=sig_r)
    return dict(z=z_sm, z_raw=z_raw, conf=conf, guide=guide)

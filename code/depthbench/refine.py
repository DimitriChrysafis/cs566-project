"""defocus-informed scale fitting and refinement.

objective (project proposal, "proposed method"):

    q(x) = s * d_hat(x) + t + r(x),        z(x) = q(x)^-1
    min_{s,t,r} sum_x w(x) * rho(q(x) - 1/z_F(x))
              + lam * sum_x ||grad r(x)||^2
              + mu  * sum_x r(x)^2
    s.t.  q(x) >= eps > 0

d_hat is DA-V2 relative inverse depth (higher = closer), z_F the focus
depth, w in [0,1] its confidence, rho a Huber loss. s,t are fit first
with r = 0 (IRLS), then r is solved over all pixels via a linearized
system (W + lam*L + mu*I) r = b with a 4-neighbor graph Laplacian L.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

EPS = 1e-3


def _focus_valid(z_f, w, mask=None):
    m = np.isfinite(z_f) & (z_f > EPS) & np.isfinite(w) & (w > 0)
    if mask is not None:
        m &= np.asarray(mask, bool)
    return m


def _huber_w(res, delta, sigma):
    # standardized-residual huber weight psi(e)/e, e = res/sigma
    a = np.abs(res)
    return np.minimum(1.0, delta * sigma / np.maximum(a, 1e-12))


def _robust_sigma(res):
    mad = np.median(np.abs(res - np.median(res)))
    if mad > 1e-12:
        return 1.4826 * mad
    return max(float(np.sqrt(np.mean(res * res))), 1e-6)


def fit_scale(d_inv, z_f, w, mask=None, huber_delta=1.345, iters=10, min_conf_cov=0.02):
    """IRLS for s,t on confident focus pixels with r = 0."""
    d_inv = np.asarray(d_inv, np.float64)
    z_f = np.asarray(z_f, np.float64)
    w = np.asarray(w, np.float64)
    m = _focus_valid(z_f, w, mask)
    coverage = float(m.sum() / m.size)
    a, y, wv = d_inv[m], 1.0 / z_f[m], w[m]
    s, t = 1.0, 0.0
    ok = coverage >= min_conf_cov and m.sum() >= 8
    if ok:
        for _ in range(iters):
            res = a * s + t - y
            sig = _robust_sigma(res)
            u = wv * _huber_w(res, huber_delta, sig)
            sa, s1 = np.sum(u * a), np.sum(u)
            A = np.array([[np.sum(u * a * a), sa], [sa, s1]])
            b = np.array([np.sum(u * a * y), np.sum(u * y)])
            try:
                s, t = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                ok = False
                break
        if s <= 0:  # degenerate: d anticorrelates -> bounded t-only fallback
            s, t = 0.0, float(np.sum(wv * y) / np.sum(wv))
            ok = False
    elif m.sum() > 0:  # too few pixels for affine: constant at weighted mean
        s, t = 0.0, float(np.sum(wv * y) / np.sum(wv))
    # else: no focus info at all -> s=1, t=0, stays relative
    return dict(s=float(s), t=float(t), coverage=coverage, ok=bool(ok))


def _laplacian(h, w):
    idx = np.arange(h * w).reshape(h, w)
    i = np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()])
    j = np.concatenate([idx[:, 1:].ravel(), idx[1:, :].ravel()])
    e = sp.coo_matrix((np.ones(i.size), (i, j)), shape=(h * w, h * w))
    e = e + e.T
    return sp.diags(np.asarray(e.sum(0)).ravel()) - e


def refine(d_inv, z_f, w, lam=3.0, mu=1e-2, huber_delta=1.345, cg_iters=400,
           s=None, t=None, mask=None, eps=EPS, irls_iters=3):
    """full objective: affine scale fit then weighted-smooth correction r."""
    d_inv = np.asarray(d_inv, np.float64)
    z_f = np.asarray(z_f, np.float64)
    w = np.asarray(w, np.float64)
    h, wd = d_inv.shape
    n = h * wd

    calibrated = True
    if s is None or t is None:
        fs = fit_scale(d_inv, z_f, w, mask=mask, huber_delta=huber_delta)
        s, t = fs["s"], fs["t"]
        calibrated = fs["ok"]
    if not calibrated:
        q = np.clip(s * d_inv + t, eps, None)
        return dict(q=q, s=float(s), t=float(t), r=np.zeros_like(d_inv),
                    ok=False, niter=0, calibrated=False)

    m = _focus_valid(z_f, w, mask)
    y = np.where(m, 1.0 / np.maximum(z_f, eps), 0.0)
    base = s * d_inv + t
    c = (base - y).ravel()
    wf = np.where(m, w, 0.0).ravel()
    r = np.zeros(n)
    niter = 0

    A_reg = lam * _laplacian(h, wd) + mu * sp.identity(n, format="csc")
    for _ in range(irls_iters):
        res = c + r
        u = np.where(wf > 0, wf * _huber_w(res, huber_delta, _robust_sigma(res[m.ravel()])), 0.0)
        A = sp.diags(u + 1e-9) + A_reg
        b = -u * c
        diag = A.diagonal()
        M = spla.aslinearoperator(sp.diags(1.0 / np.maximum(diag, 1e-9)))

        cnt = [0]
        r, info = spla.cg(A, b, x0=r, rtol=1e-7, atol=1e-9, maxiter=cg_iters,
                          M=M, callback=lambda xk: cnt.__setitem__(0, cnt[0] + 1))
        niter += cnt[0]
        if info < 0:
            break

    q = np.clip(base + r.reshape(h, wd), eps, None)
    return dict(q=q, s=float(s), t=float(t), r=r.reshape(h, wd),
                ok=bool(info == 0), niter=niter, calibrated=True)


def to_depth(q, eps=EPS):
    return 1.0 / np.clip(np.asarray(q, np.float64), eps, None)

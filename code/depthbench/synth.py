from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter

from .camera import Camera

ASSETS = Path(__file__).resolve().parent / "assets"


@dataclass
class Scene:
    color: np.ndarray
    depth: np.ndarray
    masks: dict = field(default_factory=dict)
    name: str = ""


def _load(name, w, h):
    im = Image.open(ASSETS / name).convert("RGB").resize((w, h), Image.LANCZOS)
    return np.asarray(im, np.float32) / 255.0


def _smooth_noise(rng, h, w, sigma=2.0):
    return gaussian_filter(rng.normal(0, 1, (h, w)), sigma).astype(np.float32)


def _lowtex(color, thr=0.012, win=15):
    g = color @ np.array([0.299, 0.587, 0.114], np.float32)
    gy, gx = np.gradient(g)
    return uniform_filter(np.hypot(gx, gy), win) < thr


def _rect(h, w, y0, y1, x0, x1):
    yy, xx = np.mgrid[0:h, 0:w]
    return (yy >= y0 * h) & (yy < y1 * h) & (xx >= x0 * w) & (xx < x1 * w)


def _disc(h, w, cy, cx, r):
    yy, xx = np.mgrid[0:h, 0:w]
    d2 = ((yy - cy * h) ** 2 + (xx - cx * w) ** 2) / (r * min(h, w)) ** 2
    return d2 <= 1.0, d2


def _gallery(seed, w, h):
    yy, _ = np.mgrid[0:h, 0:w]
    depth = (6.0 - 4.2 * (yy / (h - 1))).astype(np.float32)
    color = _load("nyc.png", w, h)

    card = _rect(h, w, 0.125, 0.55, 0.625, 0.917)
    depth[card] = 3.0
    color[card] = np.array([0.62, 0.60, 0.57], np.float32)

    painting = _rect(h, w, 0.297, 0.953, 0.083, 0.437)
    py, px = np.where(painting)
    pw, ph = px.max() - px.min() + 1, py.max() - py.min() + 1
    depth[painting] = 1.1
    color[painting] = _load("Vincent_van_Gogh.png", pw, ph).reshape(-1, 3)

    masks = {
        "card": card,
        "painting": painting,
        "textureless": card | _lowtex(color),
    }
    return Scene(color, depth, masks, "gallery")


def _corridor(seed, w, h):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy, fc = w / 2.0, 0.45 * h, 350.0
    H_floor, H_ceil, X_wall, Z_back = 1.6, 1.1, 1.2, 6.0

    def _plane(num, den, valid):
        return np.divide(num, den, out=np.full((h, w), np.inf, np.float32), where=valid)

    z_floor = _plane(fc * H_floor, yy - cy, yy > cy)
    z_ceil = _plane(fc * H_ceil, cy - yy, yy < cy)
    z_left = _plane(fc * X_wall, cx - xx, xx < cx)
    z_right = _plane(fc * X_wall, xx - cx, xx > cx)
    planes = np.stack([z_floor, z_ceil, z_left, z_right, np.full((h, w), Z_back, np.float32)])
    which = planes.argmin(0)
    depth = np.take_along_axis(planes, which[None], 0)[0].astype(np.float32)

    X = depth * (xx - cx) / fc
    Y = depth * (cy - yy) / fc
    color = np.zeros((h, w, 3), np.float32)

    floor = which == 0
    chk = ((np.floor(X / 0.22) + np.floor(depth / 0.22)) % 2)[floor]
    fcol = np.array([0.42, 0.38, 0.34], np.float32) + 0.30 * chk[:, None]
    fcol += 0.16 * _smooth_noise(rng, h, w, 0.8)[floor][:, None]
    fcol += 0.10 * rng.normal(0, 1, (floor.sum(), 1)).astype(np.float32)
    color[floor] = fcol

    ceil = which == 1
    color[ceil] = np.array([0.80, 0.79, 0.76], np.float32)

    walls = (which == 2) | (which == 3)
    stripe = (np.floor(depth / 0.5) % 2)[walls]
    wcol = np.array([0.55, 0.50, 0.44], np.float32) - 0.22 * stripe[:, None]
    band = (np.abs(np.mod(Y, 0.4) - 0.2) < 0.025)[walls]
    wcol += 0.10 * band[:, None]
    wcol += 0.15 * _smooth_noise(rng, h, w, 0.8)[walls][:, None]
    wcol += 0.10 * rng.normal(0, 1, (walls.sum(), 1)).astype(np.float32)
    color[walls] = wcol

    back = which == 4
    color[back] = np.array([0.66, 0.62, 0.56], np.float32) + 0.015 * _smooth_noise(rng, h, w, 2.0)[back][:, None]

    by, bx = np.where(back)
    y0, y1 = by.min() + (np.ptp(by) // 5), by.max() - (np.ptp(by) // 6)
    x0 = bx.min() + (np.ptp(bx) // 12)
    x1 = x0 + (np.ptp(bx) // 3)
    poster = back & (yy >= y0) & (yy < y1) & (xx >= x0) & (xx < x1)
    if poster.any():
        pimg = _load("I_Love_New_York.png", x1 - x0, y1 - y0)
        sub = poster[y0:y1, x0:x1]
        color[y0:y1, x0:x1][sub] = pimg[sub]

    masks = {
        "floor": floor,
        "ceiling": ceil,
        "walls": walls,
        "back": back,
        "poster": poster,
        "textureless": ceil | (back & ~poster) | _lowtex(color),
    }
    return Scene(np.clip(color, 0, 1), depth, masks, "corridor")


def _slanted(seed, w, h):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    depth = (2.0 + 2.8 * (xx / (w - 1)) + 0.6 * (yy / (h - 1))).astype(np.float32)

    pat = 0.5 + 0.22 * np.sin(xx / 7.0) * np.sin(yy / 9.0) + 0.14 * _smooth_noise(rng, h, w, 1.0)
    color = np.stack([pat, 0.55 + 0.2 * np.sin(xx / 11.0 + 1.3) + 0.1 * _smooth_noise(rng, h, w, 1.0), pat], -1)
    color = np.clip(color, 0, 1).astype(np.float32)

    blank = _rect(h, w, 0.12, 0.42, 0.55, 0.85)
    color[blank] = np.array([0.78, 0.76, 0.72], np.float32)

    plane = np.ones((h, w), bool)
    spheres = np.zeros((h, w), bool)
    for cyx, r, zc, col in [
        ((0.32, 0.28), 0.13, 1.5, (0.72, 0.25, 0.20)),
        ((0.62, 0.62), 0.17, 2.6, (0.25, 0.55, 0.35)),
        ((0.30, 0.80), 0.09, 4.0, (0.25, 0.35, 0.70)),
    ]:
        d, d2 = _disc(h, w, cyx[0], cyx[1], r)
        bulge = 0.45 * np.sqrt(np.clip(1 - d2, 0, 1))
        depth[d] = zc - bulge[d]
        speck = _smooth_noise(rng, h, w, 0.8)[d][:, None]
        color[d] = np.clip(np.array(col, np.float32) + 0.25 * speck + 0.25 * bulge[d][:, None], 0, 1)
        spheres |= d
        plane &= ~d

    masks = {
        "plane": plane,
        "spheres": spheres,
        "blank": blank & ~spheres,
        "textureless": (blank & ~spheres) | _lowtex(color),
    }
    return Scene(color, depth, masks, "slanted")


def _objects(seed, w, h):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    depth = np.full((h, w), 6.5, np.float32)
    color = np.ones((h, w, 3), np.float32) * np.array([0.74, 0.72, 0.68], np.float32)
    color += (0.015 * _smooth_noise(rng, h, w, 2.5))[..., None]
    color = np.clip(color + 0.03 * (yy / (h - 1))[..., None], 0, 1).astype(np.float32)

    flat = _rect(h, w, 0.18, 0.42, 0.40, 0.60)
    depth[flat] = 3.0
    color[flat] = np.array([0.70, 0.62, 0.50], np.float32)

    d, _ = _disc(h, w, 0.58, 0.20, 0.17)
    depth[d] = 1.3
    dot = _smooth_noise(rng, h, w, 0.7)[d][:, None]
    color[d] = np.clip(np.array([0.30, 0.45, 0.65], np.float32) + 0.5 * dot, 0, 1)

    stripe = _rect(h, w, 0.55, 0.85, 0.52, 0.78)
    depth[stripe] = 2.5
    s = (np.floor((xx - 0.52 * w) / (0.018 * w)) % 2)[stripe][:, None]
    color[stripe] = np.clip(np.array([0.60, 0.30, 0.25], np.float32) + 0.2 * s + 0.06 * _smooth_noise(rng, h, w, 1.0)[stripe][:, None], 0, 1)

    card = _rect(h, w, 0.30, 0.72, 0.80, 0.98)
    depth[card] = 4.2
    py, px = np.where(card)
    color[card] = _load("I_Love_New_York.png", px.max() - px.min() + 1, py.max() - py.min() + 1).reshape(-1, 3)

    objs = flat | d | stripe | card
    bg = ~objs
    masks = {
        "objects": objs,
        "flat_obj": flat,
        "textureless": bg | flat | _lowtex(color),
    }
    return Scene(np.clip(color, 0, 1), depth, masks, "objects")


_BUILDERS = {"gallery": _gallery, "corridor": _corridor, "slanted": _slanted, "objects": _objects}
SCENE_KINDS = tuple(_BUILDERS)


def make_scene(kind, seed=0, w=480, h=320):
    if kind not in _BUILDERS:
        raise ValueError(f"unknown scene kind {kind!r}; choose from {SCENE_KINDS}")
    return _BUILDERS[kind](seed, w, h)


def render_frame(scene, zf, cam=None, bins=28):
    cam = cam or Camera()
    color, depth = scene.color, scene.depth
    inv = 1.0 / depth
    edges = np.linspace(inv.min(), inv.max() + 1e-6, bins + 1)
    out = np.zeros_like(color)
    for i in range(bins):
        m = ((inv >= edges[i]) & (inv < edges[i + 1])).astype(np.float32)
        if not m.any():
            continue
        s = float(cam.sigma(1.0 / (0.5 * (edges[i] + edges[i + 1])), zf))
        mb = gaussian_filter(m, s) if s > 0.3 else m
        cb = np.stack([gaussian_filter(color[..., c] * m, s) if s > 0.3 else color[..., c] * m for c in range(3)], -1)
        out = out * (1 - mb[..., None]) + cb
    return np.clip(out, 0, 1)


def render_stack(scene, zfs, cam=None, noise=0.004, seed=0):
    rng = np.random.default_rng(seed)
    cam = cam or Camera()
    frames = []
    for zf in zfs:
        f = render_frame(scene, float(zf), cam)
        frames.append(np.clip(f + rng.normal(0, noise, f.shape), 0, 1).astype(np.float32))
    return np.stack(frames)

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

HERE = Path(__file__).resolve().parent
HW1 = HERE.parents[2] / "HW1" / "provided" / "CS566_HW1_Python"

plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.6})

F = 0.025
N = 2.0
F_PX = 1200.0
W, H = 480, 320
SIGMA_MAX = 9.0


def sigma_px(z, zf):
    return np.minimum(F_PX * F / (2 * N * (zf - F)) * np.abs(1.0 - zf / z), SIGMA_MAX)


def load(name, w, h):
    return np.asarray(Image.open(HW1 / name).convert("RGB").resize((w, h), Image.LANCZOS), np.float32) / 255.0


def build_scene():
    yy, xx = np.mgrid[0:H, 0:W]
    depth = 6.0 - 4.2 * (yy / (H - 1))
    color = load("nyc.png", W, H)

    card = (yy > 40) & (yy < 175) & (xx > 300) & (xx < 440)
    depth[card] = 3.0
    color[card] = np.array([0.62, 0.60, 0.57])

    pw, ph = 170, 210
    painting = (yy >= 95) & (yy < 95 + ph) & (xx >= 40) & (xx < 40 + pw)
    depth[painting] = 1.1
    color[painting] = load("Vincent_van_Gogh.png", pw, ph).reshape(-1, 3)
    return color, depth, card, painting


def render(color, depth, zf, bins=28):
    inv = 1.0 / depth
    edges = np.linspace(inv.min(), inv.max() + 1e-6, bins + 1)
    out = np.zeros_like(color)
    for i in range(bins):
        m = ((inv >= edges[i]) & (inv < edges[i + 1])).astype(np.float32)
        if not m.any():
            continue
        s = float(sigma_px(1.0 / (0.5 * (edges[i] + edges[i + 1])), zf))
        mb = gaussian_filter(m, s) if s > 0.3 else m
        cb = np.stack([gaussian_filter(color[..., c] * m, s) if s > 0.3 else color[..., c] * m for c in range(3)], -1)
        out = out * (1 - mb[..., None]) + cb
    return np.clip(out, 0, 1)


def focus_measure(img):
    g = img @ np.array([0.299, 0.587, 0.114], np.float32)
    ml = np.abs(2 * g - np.roll(g, 1, 1) - np.roll(g, -1, 1)) + np.abs(2 * g - np.roll(g, 1, 0) - np.roll(g, -1, 0))
    return uniform_filter(ml, 11)


def depth_from_focus(stack, zfs):
    fm = np.stack([focus_measure(f) for f in stack])
    k = np.clip(fm.argmax(0), 1, len(zfs) - 2)
    yi, xi = np.indices(k.shape)
    a, b, c = fm[k - 1, yi, xi], fm[k, yi, xi], fm[k + 1, yi, xi]
    off = np.clip(0.5 * (a - c) / (a - 2 * b + c + 1e-9), -0.5, 0.5)
    inv = 1.0 / zfs
    return 1.0 / np.interp(k + off, np.arange(len(zfs)), inv)


def fig_blur_curve():
    z = np.linspace(0.55, 8, 400)
    fig, ax = plt.subplots(figsize=(3.0, 1.55))
    for zf, c in [(1.0, "#1F4037"), (2.0, "#B8860B"), (4.0, "#9C3D2E")]:
        ax.plot(z, sigma_px(z, zf), color=c, lw=1.2, label=f"focused at {zf:g} m")
        ax.axvline(zf, color=c, lw=0.5, ls=":")
    ax.set_xlabel("scene depth $z$ (m)")
    ax.set_ylabel(r"blur $\sigma$ (px)")
    ax.set_xlim(0.55, 8)
    ax.set_ylim(0, SIGMA_MAX + 0.3)
    ax.legend(frameon=False, fontsize=6, loc="center right", bbox_to_anchor=(1.0, 0.5))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.2)
    fig.savefig(HERE / "blur_curve.pdf")
    plt.close(fig)


def fig_preliminary():
    color, depth, card, painting = build_scene()
    rng = np.random.default_rng(0)
    zfs = np.geomspace(0.9, 7.0, 10)
    stack = [np.clip(render(color, depth, zf) + rng.normal(0, 0.004, color.shape), 0, 1).astype(np.float32) for zf in zfs]

    est = depth_from_focus(stack, zfs)
    rel = np.abs(est - depth) / depth
    textured = ~card
    abs_tex, abs_card = rel[textured].mean(), rel[card].mean()
    d1_tex = (np.maximum(est / depth, depth / est)[textured] < 1.25).mean()

    near = stack[int(np.argmin(np.abs(zfs - 1.1)))]
    far = stack[int(np.argmin(np.abs(zfs - 5.0)))]

    fig = plt.figure(figsize=(7.0, 1.38))
    gs = fig.add_gridspec(1, 5, wspace=0.2)
    ax = [fig.add_subplot(gs[0, i]) for i in range(5)]
    cax1 = make_axes_locatable(ax[3]).append_axes("right", size="5%", pad=0.03)
    cax2 = make_axes_locatable(ax[4]).append_axes("right", size="5%", pad=0.03)
    ax[0].imshow(near)
    ax[1].imshow(far)
    im = ax[2].imshow(depth, cmap="turbo_r", vmin=1, vmax=6)
    ax[3].imshow(est, cmap="turbo_r", vmin=1, vmax=6)
    er = ax[4].imshow(rel, cmap="magma", vmin=0, vmax=0.5)
    zn, zfar = zfs[int(np.argmin(np.abs(zfs - 1.1)))], zfs[int(np.argmin(np.abs(zfs - 5.0)))]
    titles = [f"(a) focus at {zn:.1f} m", f"(b) focus at {zfar:.1f} m",
              "(c) true depth", "(d) our DFF", "(e) rel. error"]
    for a, t in zip(ax, titles):
        a.set_title(t, fontsize=7, pad=2)
        a.set_xticks([])
        a.set_yticks([])
    ax[4].annotate("textureless\ncard", xy=(370, 175), xytext=(370, 262), color="white", fontsize=5.5,
                   ha="center", va="center", arrowprops=dict(arrowstyle="-", color="white", lw=0.5))
    cb1 = fig.colorbar(im, cax=cax1)
    cb1.set_ticks([1, 2, 4, 6])
    cb1.ax.tick_params(labelsize=5, length=1.5, pad=1)
    cb2 = fig.colorbar(er, cax=cax2)
    cb2.set_ticks([0, 0.25, 0.5])
    cb2.ax.tick_params(labelsize=5, length=1.5, pad=1)
    fig.savefig(HERE / "dff_preliminary.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return abs_tex, abs_card, d1_tex, len(zfs)


if __name__ == "__main__":
    fig_blur_curve()
    a, b, d1, n = fig_preliminary()
    print(f"frames={n} absrel_textured={a:.3f} absrel_textureless={b:.3f} delta1_textured={d1:.3f}")

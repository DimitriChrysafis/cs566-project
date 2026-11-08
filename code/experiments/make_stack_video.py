"""looping clip for the webpage: step a synthetic focal stack through its
focus settings while the dff depth estimate fills in, ending on the final
smoothed depth map + all-focus composite. writes webpage/assets/stack_step.mp4
(+ stack_step_poster.png)."""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depthbench.synth import make_scene, render_stack
from depthbench.dff import (all_focus, depth_from_focus, focus_measure,
                            smooth_depth, to_gray)
from depthbench.viz import conf_rgb, depth_rgb

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "webpage" / "assets"
FFMPEG = "/opt/homebrew/bin/ffmpeg"

SCENE, W, H = "corridor", 480, 320
ZFS = np.geomspace(0.9, 7.0, 10)
FPS = 24
HOLD, FADE, END_HOLD, LOOP_FADE = 12, 4, 40, 8

CW, CH = 720, 470
IMG_BOX = (18, 44, 430, 287)      # x, y, w, h  (aspect 1.5)
DEP_BOX = (462, 44, 242, 161)
CONF_BOX = (462, 249, 242, 161)
ACCENT = (28, 118, 196)
INK = (28, 28, 30)
GRAY = (120, 120, 124)
TRACK = (210, 210, 214)

FONT = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
f_head = ImageFont.truetype(str(FONT), 20)
f_sub = ImageFont.truetype(str(FONT), 16)
f_small = ImageFont.truetype(str(FONT), 13)
f_tiny = ImageFont.truetype(str(FONT), 11)


def _panel(arr, box):
    x, y, w, h = box
    return Image.fromarray(arr).resize((w, h), Image.LANCZOS)


def _dots(d, k):
    xs = np.linspace(40, 424, len(ZFS))
    y = 384
    d.line([(xs[0], y), (xs[-1], y)], fill=TRACK, width=3)
    for i, x in enumerate(xs):
        r = 7 if i == k else 5
        fill = ACCENT if i <= k else (255, 255, 255)
        d.ellipse([x - r, y - r, x + r, y + r], fill=fill,
                  outline=ACCENT if i <= k else TRACK, width=2)
    d.text((xs[0] - 14, y + 14), f"{ZFS[0]:.1f} m", font=f_tiny, fill=GRAY)
    d.text((xs[-1] - 14, y + 14), f"{ZFS[-1]:.1f} m", font=f_tiny, fill=GRAY)


def make_canvas(img, dep, conf, header, k, mean_conf):
    c = Image.new("RGB", (CW, CH), (255, 255, 255))
    c.paste(_panel(img, IMG_BOX), IMG_BOX[:2])
    c.paste(_panel(dep, DEP_BOX), DEP_BOX[:2])
    c.paste(_panel(conf, CONF_BOX), CONF_BOX[:2])
    d = ImageDraw.Draw(c)
    d.text((18, 14), header, font=f_head, fill=INK)
    d.text((462, 18), "estimated depth", font=f_sub, fill=INK)
    d.text((462, 223), "confidence", font=f_sub, fill=INK)
    d.text((18, 352), "focus sweep", font=f_small, fill=GRAY)
    _dots(d, k)
    d.text((18, 414), "mean confidence", font=f_small, fill=GRAY)
    bx, by, bw = 150, 416, 180
    d.rounded_rectangle([bx, by, bx + bw, by + 10], 5, outline=TRACK, width=1)
    if mean_conf > 0.01:
        d.rounded_rectangle([bx, by, bx + bw * min(mean_conf, 1.0), by + 10],
                            5, fill=ACCENT)
    d.text((bx + bw + 10, by - 4), f"{mean_conf:.2f}", font=f_small, fill=INK)
    d.text((18, 448), "depth from focus: per-pixel argmax of the "
                      "modified-laplacian focus measure", font=f_tiny, fill=GRAY)
    return np.asarray(c)


def running_est(fms, k):
    f = fms[:k + 1]
    idx = f.argmax(0)
    z = ZFS[idx].astype(np.float32)
    b = f.max(0)
    if k == 0:
        conf = 0.4 * b / (b + 0.02)
    else:
        rest = (f.sum(0) - b) / k
        prom = np.clip((b - rest) / (b + 1e-9), 0, 1)
        conf = prom * b / (b + 0.02)
    return z, np.clip(conf, 0, 1).astype(np.float32)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    z_lo, z_hi = float(ZFS.min()), float(ZFS.max())

    scene = make_scene(SCENE, seed=0, w=W, h=H)
    stack = render_stack(scene, ZFS, noise=0.004, seed=0)
    fms = np.stack([focus_measure(g) for g in to_gray(stack)])

    z_raw, conf = depth_from_focus(stack, ZFS)
    guide = all_focus(stack, fm=fms)
    z_sm = smooth_depth(z_raw, conf, guide)

    canvases = []
    for k in range(len(ZFS)):
        z_run, c_run = running_est(fms, k)
        canvases.append(make_canvas(
            (stack[k] * 255).astype(np.uint8),
            depth_rgb(z_run, z_lo, z_hi), conf_rgb(c_run),
            f"focus {k + 1}/10  ·  zf = {ZFS[k]:.2f} m", k, float(c_run.mean())))
    canvases.append(make_canvas(
        (guide * 255).astype(np.uint8),
        depth_rgb(z_sm, z_lo, z_hi), conf_rgb(conf),
        "all-focus composite  ·  final depth", len(ZFS) - 1, float(conf.mean())))

    tmp = Path(tempfile.mkdtemp(prefix="stack_frames_"))
    try:
        n = 0

        def emit(arr, count):
            nonlocal n
            u8 = np.clip(arr + 0.5, 0, 255).astype(np.uint8)
            for _ in range(count):
                Image.fromarray(u8).save(tmp / f"frame_{n:04d}.png")
                n += 1

        def dissolve(a, b, count, include_end=False):
            steps = count + (1 if include_end else 0)
            for i in range(1, count + 1):
                t = i / steps
                emit((a * (1 - t) + b * t).astype(np.uint8), 1)

        cvs = [c.astype(np.float32) for c in canvases]
        for k in range(len(ZFS)):
            emit(cvs[k], HOLD)
            dissolve(cvs[k], cvs[k + 1], FADE)
        emit(cvs[-1], END_HOLD)
        dissolve(cvs[-1], cvs[0], LOOP_FADE - 1, include_end=True)
        dur = n / FPS
        print(f"{n} frames -> {dur:.2f} s @ {FPS} fps")

        shutil.copy(tmp / "frame_0000.png", OUT_DIR / "stack_step_poster.png")
        mp4 = OUT_DIR / "stack_step.mp4"
        r = subprocess.run(
            [FFMPEG, "-y", "-framerate", str(FPS), "-i", str(tmp / "frame_%04d.png"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21",
             "-movflags", "+faststart", str(mp4)],
            capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-2000:])
            print("mp4 failed; falling back to gif")
            subprocess.run(
                [FFMPEG, "-y", "-framerate", str(FPS), "-i", str(tmp / "frame_%04d.png"),
                 "-vf", "scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];"
                        "[s1][p]paletteuse", str(OUT_DIR / "stack_step.gif")],
                capture_output=True, text=True, check=True)
        print(f"wrote {mp4 if mp4.exists() else OUT_DIR / 'stack_step.gif'}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

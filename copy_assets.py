#!/usr/bin/env python3
"""Copy the reports, figures and csvs the page links to into webpage/assets/.

Only webpage/ is deployed, so anything the page references has to live
there. Pdf figures are rendered to png with sips (macOS) or pdftoppm.
Missing sources are skipped. Stdlib only, idempotent.
"""

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "webpage" / "assets"
RES = ROOT / "results"

# source -> name under assets/
PDFS = {
    "proposal/project_proposal.pdf": "proposal.pdf",
    "report/midterm_report.pdf": "midterm.pdf",
    "report/final_report.pdf": "final.pdf",
}
# pdf figures rendered to png for <img>
FIGS = {
    "proposal/figures/blur_curve.pdf": "blur_curve.png",
    "results/figs/synth_dff_corridor.pdf": "synth_dff_corridor.png",
    "results/figs/nyu_models.pdf": "nyu_models.png",
    "results/figs/nyu_models_qual.pdf": "nyu_models_qual.png",
    "results/figs/refine_objects.pdf": "refine_objects.png",
    "results/figs/refine_shelves.pdf": "refine_shelves.png",
    "results/figs/refine_stairs.pdf": "refine_stairs.png",
    "results/figs/refine_gallery.pdf": "refine_gallery.png",
    "results/figs/ddff_refine_test_000.pdf": "ddff_000.png",
    "results/figs/ddff_refine_test_044.pdf": "ddff_044.png",
    "results/figs/ddff_refine_test_072.pdf": "ddff_072.png",
}


def pdf_to_png(src, dst, width=1600):
    if shutil.which("sips"):
        cmd = ["sips", "-s", "format", "png", "-Z", str(width), str(src), "--out", str(dst)]
    elif shutil.which("pdftoppm"):
        cmd = ["pdftoppm", "-png", "-singlefile", "-scale-to-x", str(width),
               "-scale-to-y", "-1", str(src), str(dst.with_suffix(""))]
    else:
        print("skip", src.name, "(no sips or pdftoppm)")
        return
    subprocess.run(cmd, check=True, capture_output=True)


def main():
    ASSETS.mkdir(exist_ok=True)
    for src, name in PDFS.items():
        if (ROOT / src).exists():
            shutil.copy(ROOT / src, ASSETS / name)
    for src, name in FIGS.items():
        if (ROOT / src).exists():
            pdf_to_png(ROOT / src, ASSETS / name)
    data = ASSETS / "data"
    data.mkdir(exist_ok=True)
    for csv in RES.glob("*.csv"):
        shutil.copy(csv, data / csv.name)
    print("copied into", ASSETS)


if __name__ == "__main__":
    main()

# v1.0 release checklist

One tagged release so the webpage "download" link points at something
stable instead of a moving branch zip.

## before tagging

- [ ] `run_demo.sh` exists at the repo root and runs end to end on one
      synthetic scene (renders the stack, runs dff + da-v2 + refine,
      writes a figure). It is in progress; do not tag without it.
- [ ] `pip install -r code/requirements.txt` works from a clean venv.
- [ ] README commands match the scripts (order, `--ablation` choices).
- [ ] `python webpage/build.py` regenerates the tables in `index.html`
      with no diff, so the site and the CSVs agree.
- [ ] `data/`, `results/preds/`, `results/*.npz` stay untracked.

## tag

    git tag -a v1.0 -m "v1.0: code release"
    git push origin v1.0

Then draft the GitHub release at
https://github.com/DimitriChrysafis/cs566-project/releases/new.

## artifacts to attach

`cs566-v1.0-code.zip`: `code/` plus `run_demo.sh` (and `README.md` if we
want the zip self-contained).

    zip -r cs566-v1.0-code.zip code run_demo.sh README.md \
        -x "*/__pycache__/*" "*.pyc"

`cs566-v1.0-results.zip`: the CSVs only, same layout as `results/`:

    nyu_models.csv        table 1, per-image rows for all variants/crops
    nyu_dav2.csv          per-image alignment detail for da-v2 small
    refine_synth.csv      table 2, per-scene rows, all/low/high-tex splits
    synth_dff.csv         raw vs smoothed dff on the synthetic scenes
    ddff_dff.csv          ddff per-stack confidences (no gt, absrel is nan)
    ablation_stacksize.csv, ablation_conf.csv, ablation_lam.csv,
    ablation_mu.csv       tuning-scene ablations

Do not attach `results/figs/*.pdf`, `results/*.npz`, `results/preds/`, or
`data/`; all regenerable or too large.

## link from the webpage

The Code section of `index.html` gains a release link next to the branch
zip: `https://github.com/DimitriChrysafis/cs566-project/releases/tag/v1.0`,
with the two zips listed under it. Keep the existing
`archive/refs/heads/main.zip` link as the "latest source" option.

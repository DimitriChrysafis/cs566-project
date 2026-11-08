# cs566 final project

monocular depth estimation: benchmarking depth anything v2 against a
classical depth-from-focus baseline, plus a defocus-informed scale
refinement. a single rgb image does not determine absolute scale, so we
align da-v2 relative depth with confident focus measurements from a
focal stack: an affine fit on the confident pixels, then a regularized
local correction, with rejection and fallback to the focus estimate when
the two disagree. evaluated on the nyu-depth v2 test split (654 images,
eigen crop), six rendered synthetic focal-stack scenes (tune/eval
split), and the ddff 12-scene real stacks (qualitative only; the public
test split ships no ground truth).

site: https://dimitrichrysafis.github.io/cs566-project/
repo: https://github.com/DimitriChrysafis/cs566-project

## headline results

nyu-depth v2 test set (654 images, eigen crop), `results/nyu_models.csv`:

| method | absrel | rmse | delta1 |
|---|---|---|---|
| da-v2 small (scale+shift) | 0.062 | 0.297 | 0.956 |
| da-v2 small (robust irls) | 0.058 | 0.292 | 0.958 |
| da-v2 small (median scale) | 0.494 | 2.349 | 0.393 |
| da-v2 base (scale+shift) | 0.057 | 0.283 | 0.959 |
| da-v2 large (scale+shift) | 0.055 | 0.277 | 0.961 |
| da-v2 metric indoor (unaligned) | 0.218 | 0.650 | 0.670 |
| da-v2 metric indoor (aligned) | 0.077 | 0.353 | 0.936 |

held-out synthetic scenes (corridor, objects, slanted, shelves),
`results/refine_synth.csv`:

| method | absrel | rmse | delta1 |
|---|---|---|---|
| dff raw | 0.233 | 1.552 | 0.732 |
| dff + all-focus smooth | 0.221 | 1.510 | 0.702 |
| aligned da-v2 | 0.271 | 1.524 | 0.548 |
| da-v2 + scale fit | 0.308 | 1.614 | 0.459 |
| refined (ours) | 0.259 | 1.483 | 0.575 |
| combined (ours) | 0.206 | 1.284 | 0.704 |

## demo

    bash run_demo.sh          # renders one synthetic stack, runs the full pipeline

## setup

    pip install -r code/requirements.txt
    python code/experiments/fetch_data.py all      # nyu + ddff (see datasets below)

experiments, in order:

    python code/experiments/run_synth.py           # synthetic stacks, raw vs smoothed dff
    python code/experiments/run_dav2_nyu.py        # da-v2 small baseline on nyu test split
    python code/experiments/run_nyu_models.py      # da-v2 s/b/l + metric indoor on nyu
    python code/experiments/run_refine.py          # scale fit + refinement, tune/eval split
    python code/experiments/run_ablation.py --ablation stacksize   # also conf, lam, mu
    python code/experiments/run_dff_ddff.py        # dff on ddff real stacks (qualitative)
    python code/experiments/run_refine_ddff.py     # full pipeline on ddff stacks

    python webpage/build.py                        # regenerate results tables in index.html
    bash publish.sh                                # deploy webpage/ to github pages (gh-pages branch)

the nyu scripts take `--limit N` for a quick pass and `--device` to pick
cpu/mps/cuda. the ablation script runs one ablation per call, over the
two tuning scenes.

## layout

    code/depthbench/    library: data loaders, synth stacks, dff, dav2, refine, metrics
    code/experiments/   runnable scripts that write to results/
    data/               datasets (not tracked, see below)
    docs/               site outline, release checklist
    proposal/           proposal pdf + tex
    report/             midterm report pdf + tex
    results/            csvs, figures, predictions (large files not tracked)
    webpage/            static site source, deployed as-is
    run_demo.sh         one-command demo on one synthetic scene

## reports

- proposal: `proposal/project_proposal.pdf`
- midterm report: `report/midterm_report.pdf`
- final report: `report/` after submission, also posted on the site

## datasets

`data/` is not tracked. `fetch_data.py` pulls:

- nyu-depth v2 eigen test split: 654 rgb+depth pairs, ~1 gb unpacked
  (hf mirror of the fast-depth preprocessing)
- ddff 12-scene test h5: ~761 mb, plus calibration files. `--trainval`
  also grabs the 12.6 gb trainval h5; not needed for this project

the synthetic focal stacks are rendered by `code/depthbench/synth.py`,
no download. da-v2 checkpoints are pulled from the hub on first run.

## license

course project for compsci 566 (uw-madison, fall 2026). no license file;
cite the project page or the report if you reuse the code.

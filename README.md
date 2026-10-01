# cs566 final project

monocular depth estimation: benchmarking depth anything v2 against a
classical depth-from-focus baseline, plus a defocus-informed scale
refinement.

proposal in `proposal/`, code in `code/`, site in `webpage/`.

## layout

    code/depthbench/    library: data loaders, synth stacks, dff, dav2, refine, metrics
    code/experiments/   runnable scripts that write to results/
    data/               datasets (not tracked)
    results/            csvs, figures, predictions (large files not tracked)

## setup

    pip install -r code/requirements.txt
    python code/experiments/fetch_data.py          # downloads datasets into data/
    python code/experiments/run_synth.py           # synthetic focal stacks + dff check
    python code/experiments/run_dav2_nyu.py        # da-v2 baseline on nyu test split
    python code/experiments/run_refine.py          # scale fit + refinement

# repro notes

audit of whether a grader can rerun this repo and get the committed numbers.
tested on macOS arm64, python 3.14, numpy 2.4.3, scipy 1.18.0. verification
tool: `code/experiments/verify_results.py` (see bottom).

## headline

the learned-free half of the synthetic pipeline is bit-for-bit reproducible.
the da-v2-dependent half is not, on any machine without torch + the hf
checkpoint; and even with them it is device-dependent (mps vs cpu vs cuda).

rerunning corridor exactly as `run_refine.py` does (seed=1, i.e. the scene's
index in `SCENES`):

- `focus_raw` / `focus` rows: recomputed metrics match `refine_synth.csv`
  with max abs diff 0.0 over all 7 metric columns and all 3 splits.
- `aligned_dav2` / `scale_fit` / `refined` / `combined` rows: do not match
  when da-v2 is unavailable, because `learned_inv` silently falls back to
  `fake_learned`, a gt-derived stand-in that is much stronger than the real
  model. stand-in vs committed (corridor, split=all): aligned_dav2 absrel
  0.0019 vs 0.0629, scale_fit 0.0173 vs 0.0964, refined 0.0312 vs 0.0541.
  the committed csv was therefore produced by real da-v2 inference
  (identical aligned_dav2 numbers in both committed versions of the csv,
  c6a9901 and d2720ff, confirm the prior never changed).

## determinism audit, per script

| script | deterministic? | notes |
|---|---|---|
| run_synth.py | yes | seed=0 for every scene; writes synth_dff.csv + synth_*.npz |
| run_refine.py | focus rows yes, learned rows env-dependent | seeds are `enumerate(SCENES)` indices: gallery=0 corridor=1 slanted=2 objects=3 stairs=4 shelves=5. `learned_inv` -> real da-v2 if importable else seeded stand-in |
| run_ablation.py | same caveat | seeds are indices into `sorted(TUNE)` = gallery=0 stairs=1, so ablation stairs is a different noise realization than refine_synth stairs (seed 4). csvs are self-consistent but not cross-comparable for that scene |
| run_demo.py / run_demo.sh | yes | exports HF_HUB_OFFLINE=1, forces the stand-in; demo numbers will not match the learned rows of refine_synth.csv by design |
| run_dav2_nyu.py | given same data+ckpt+device | no rng; `runtime_ms` column varies per run (harmless). needs data/nyu + torch + model download |
| run_nyu_models.py | same | as above |
| run_dff_ddff.py | yes given data | sorted scene order; exits 0 cleanly when ddff is missing |
| run_refine_ddff.py | given model+data | hard-fails at import without torch (`from depthbench.dav2 import DAV2` unguarded); data-driven focdist reversal check is deterministic |
| make_stack_video.py | yes | synth only; needs ffmpeg at the hardcoded path |
| webpage/make_assets.py | same caveat as run_refine | needs nyu preds npz, ddff data, and da-v2 for ddff_panels |
| webpage/build.py | yes | pure function of results/*.csv; missing csv -> placeholder kept |
| fetch_data.py | n/a | network pulls; content fixed by upstream mirrors |

library: `synth.py` seeds every builder and `render_stack` via
`np.random.default_rng(seed)`; `_gallery` ignores its seed (uses none, so
harmless). `dff.py`, `refine.py`, `metrics.py`, `camera.py` contain no rng,
no set/dict-order dependence, no clock reads. the global `np.random` and
`random` modules are never used. only time-dependence anywhere is
`runtime_ms` in nyu_models.csv and timing prints.

### the one real hazard

`learned_inv` is the single nondeterminism source: which predictor you get
depends on whether torch/transformers and the hf checkpoint resolve, and on
the inference device. there is no unseeded rng to fix.

fixed after this audit: `run_refine.py` now commits the learned priors it
used (`results/refine_preds/<scene>.npz`: `d_inv`, `z_metric`, `prior`),
and `verify_results.py --from-preds` re-evaluates those committed maps
instead of re-running inference — the learned rows then verify
deterministically on any machine, no torch needed. a `prior` column in
`refine_synth.csv` records whether each scene used real da-v2 or the
stand-in (committed rows are all `dav2`).

## run-order dependencies (readme trace)

- `run_demo.sh`: self-contained, no downloads. works.
- `fetch_data.py all`: needed only by the four dataset scripts
  (run_dav2_nyu, run_nyu_models, run_dff_ddff, run_refine_ddff).
  run_synth, run_refine, run_ablation, run_demo do not need it — they are
  self-contained on synthetic scenes. readme's order is correct and
  complete; a grader short on bandwidth can run the synthetic subset
  without fetching anything.
- `run_dav2_nyu.py`: needs `fetch_data.py nyu` output + torch + a da-v2
  checkpoint (hf cache or network). `--save-preds` is required if you later
  want `webpage/make_assets.py` nyu panels; readme does not mention this.
- `run_nyu_models.py`: same dataset + 4 checkpoints.
- `run_refine.py` / `run_ablation.py`: no dataset; see da-v2 caveat above.
- `run_dff_ddff.py`: needs `fetch_data.py ddff` + h5py + cv2; degrades
  gracefully.
- `run_refine_ddff.py`: needs ddff data and torch at import time.
  `--all --csv ddff_refine.csv` runs every test stack and persists
  per-stack rho_fwd/rho_rev/calibrated rows (the basis for the site's
  15-of-120 reversed / 110-of-120 calibrated numbers).
- `webpage/build.py`: stdlib only, reads results/nyu_models.csv (falls back
  to nyu_dav2.csv), results/refine_synth.csv, and the four
  results/ablation_*.csv. all six RESULTS markers exist in index.html.
  it does not read synth_dff.csv or ddff_dff.csv (no tables for them).
- `publish.sh`: git subtree push to gh-pages; deploy step, not part of
  verification.

## csv / asset drift

- `webpage/assets/data/*.csv` vs `results/*.csv`: all 9 files are
  numerically identical; the only difference is line endings (results LF,
  assets CRLF), so a raw `diff` shows every line changed but there is no
  numeric drift. produced by `webpage/copy_assets.py` glob of results/*.csv.
- index.html links only `assets/data/refine_synth.csv` of the csvs;
  synth_dff.csv and ddff_dff.csv are shipped but unlinked and unread by
  build.py. harmless dead weight.
- `assets/presentation.pdf` is linked but not produced by copy_assets.py
  (its PDFS map covers proposal/midterm/final reports only). present in
  assets anyway, likely copied by hand; rerunning copy_assets will not
  recreate it if deleted.
- `assets/{ddff_qualitative,failure_cases,nyu_qualitative,synth_refinement}.png`
  come from make_assets.py (heavy deps), not copy_assets.py.
- `copy_assets.py` maps `report/final_report.pdf` which does not exist yet;
  skipped gracefully.
- seed mismatch worth knowing, not a bug: `results/synth_corridor.npz` and
  `synth_dff.csv` come from run_synth (seed 0), while the corridor rows of
  `refine_synth.csv` come from run_refine (seed 1). same scene name,
  different noise realization. e.g. dff raw absrel 0.117417 (seed 0) vs
  0.117376 (seed 1).

## how a grader reruns this

1. `pip install -r code/requirements.txt`
2. `python code/experiments/verify_results.py` — one corridor slice vs
   refine_synth.csv. PASS rows must be exact (tol 1e-6); learned rows need
   torch + the da-v2-small checkpoint and are checked at absrel +-0.005,
   delta1 +-0.01 (rmse +-0.02). without da-v2 they print SKIP with the
   measured diffs; `--strict` turns skips into failures. exit nonzero on
   any FAIL. `--from-preds` instead scores the committed priors in
   `results/refine_preds/`, which verifies every row on any machine.
3. `python code/experiments/run_synth.py` — full synthetic dff table,
   deterministic; expect byte-equal metrics vs results/synth_dff.csv
   modulo its 4-decimal rounding.
4. `python code/experiments/run_refine.py` — rewrites refine_synth.csv;
   learned rows reproduce only with real da-v2 on the same device class.
5. `python code/experiments/fetch_data.py all`, then the nyu/ddff scripts —
   expect ~1-2% level variation across devices for model rows; runtime_ms
   always differs.
6. `python webpage/build.py` — regenerates the index.html tables from the
   csvs; then `python webpage/copy_assets.py` to resync assets/data.

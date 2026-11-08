# Research review — CS 566 final project audit

Reviewer: Dr. Petra (PC audit). Scope: `webpage/index.html`, `presentation/notes.md`,
`presentation/qa_prep.md`, `presentation/slides.tex`, `README.md` vs. `results/*.csv`
and `code/`. Arithmetic re-checked independently; all CSV values below are as committed.

Verified-correct items (for the record): every Table 1 number matches
`nyu_models.csv` eigen-crop means over exactly 654 images (dav2s_lsq 0.0618/0.297/0.956,
dav2s_irls 0.0575/0.292/0.958, dav2s_median 0.494/2.349/0.393, dav2b 0.057/0.283/0.959,
dav2l 0.055/0.277/0.961, metric unaligned 0.218/0.650/0.670, aligned 0.077/0.353/0.936).
All Table 2 means match the `set=eval, split=all` rows of `refine_synth.csv`, and every
ablation-table number in Tables 3-6 matches its CSV. The tune/eval split is real
(gallery/stairs = tune in the CSV `set` column; ablations only ever ran on `TUNE`).
Per-image alignment is disclosed next to Table 1. DDFF no-GT is disclosed. Rejection
uses no ground truth.

## Findings

### 1. [BLOCKER] "The method never does worse than either of its components" is false on the reported eval set

- **Claim.** index.html:81 — fallback "so the method never does worse than either of
  its components." Slide 7 (slides.tex:178) titles the results "combined beats both
  parts" and claims "Per scene it matches the better component" (slides.tex:204);
  notes.md slide 7 repeats "It beats both of its components."
- **Evidence** (`refine_synth.csv`, `set=eval, split=all`):
  - **shelves**: combined AbsRel 0.2120 vs. focus (DFF+smooth) 0.1876, `calibrated=True`.
    The accepted merge is 13% worse than the focus component it would have returned on
    rejection. δ1: combined 0.784 vs. focus_raw 0.926 / focus 0.888.
  - **objects**: combined 0.4718 is worse than *scale_fit* 0.4393, worse than
    *focus_raw* 0.4567, and far worse than *aligned_dav2* 0.3453. Only the (already
    degraded) smoothed focus 0.5025 is worse. δ1: combined 0.219 vs. focus_raw 0.328.
  - Aggregate δ1: combined 0.704 < DFF raw 0.732 — worse in the mean too, not just
    per scene. (Acknowledged privately in qa_prep:34-39, not on the page.)
  - Under a median-over-scenes aggregation, combined AbsRel (0.150) also loses to
    DFF+smooth (0.147): the headline win is carried by the `objects` outlier scene.
- The rejection gate only fires on a degenerate fit (s ≤ 0, coverage < 2%, or a
  linear-solve failure — `refine.py:fit_scale`). Accepted merges can and do lose to
  a component input; nothing in the mechanism enforces the claimed invariant.
- **Fix.** Delete or scope the guarantee: state it as "on the eval-set mean AbsRel,"
  with an explicit per-scene win/loss table (shelves and objects lose). If the
  invariant is wanted for real, select per scene between focus and merged output
  using a GT-free criterion and show it works.

### 2. [SHOULD-FIX] DDFF focus distances do not implement the disclosed reconstruction

- **Claim.** index.html:92 and `ddff.py` docstring: slices are disparities linear in
  [0.28, 0.02] px mapped to focus distances through z = 0.14/disp.
- **Evidence.** `data/ddff.py:37` sets `STACK_FOCUSDIST_M = np.linspace(0.5, 7.0, 10)`
  = [0.5, 1.22, 1.94, 2.67, 3.39, 4.11, 4.83, 5.56, 6.28, 7.0]. The documented mapping
  gives 0.14/linspace(0.28,0.02,10) = [0.5, 0.56, 0.63, 0.72, 0.85, 1.03, 1.31, 1.80,
  2.86, 7.0] — linear in disparity is hyperbolic in depth. Only the endpoints agree.
  All DDFF DFF and refinement outputs were computed against materially wrong focus
  distances. Compounding it, `run_dff_ddff.py:82-86` computed the reported conf
  values with the unverified forward slice order (the Spearman order check exists
  only in `run_refine_ddff.py:56-63`).
- **Fix.** Set focdist to `0.14 / np.linspace(0.28, 0.02, 10)` (or show the paper
  actually intends linear-in-depth refocusing and fix the docstring instead), then
  re-run the DDFF figures and restate the confidence range.

### 3. [SHOULD-FIX] Unsupported quantitative claims about DDFF stacks

- **Claim.** qa_prep.md:21 — "about a third of the stacks prefer the reversed
  order"; slides.tex:234 / notes.md slide 9 — "calibration is accepted on most
  stacks."
- **Evidence.** `run_refine_ddff.py` defaults to 6 scenes
  (`--scenes test_000,007,013,044,058,072`); the order check result is printed for
  the first scene only and no CSV/log persists per-stack ordering or calibration
  outcomes. Nothing in `results/` substantiates "a third" or "most." The midterm
  report correctly hedges ("about a third of the stacks *we checked*"); the talk
  materials present it as a property of all 120. Additionally, acceptance there
  relies on `norm=True` confidence renormalization (`refine.py:44-52`), so it is not
  comparable to the synthetic-set rejection rule anyway.
- **Fix.** Persist the per-stack `rho_fwd`/`rho_rev`/`calibrated` triple to a CSV
  over all 120 stacks, or qualify the claims to the subset actually run.

### 4. [SHOULD-FIX] Runtime claim contradicts the recorded data

- **Claim.** qa_prep.md:52 — "DA-V2 ViT-S runs about 55 ms per image on a laptop GPU."
- **Evidence.** `nyu_models.csv` `runtime_ms` for dav2s over the 654 images:
  mean 105.0 ms, median 126.9 ms, p25 55.2 ms, min 48.4 ms. "~55 ms" describes the
  fastest quartile, not the typical runtime. (The slides' "4× runtime" for ViT-L is
  consistent with the CSV: 422/105 ≈ 4.0.)
- **Fix.** Report ~105 ms mean (or quote the distribution and device); record the
  device in the CSV.

### 5. [SHOULD-FIX] Missing baselines a metric-depth claim needs

- **Evidence.** Table 2 pits the stack-based system against DFF and an
  oracle-aligned relative model, but the **DA-V2 Metric Indoor checkpoint — a
  monocular model that outputs meters with no stack — was evaluated on NYU and never
  run on the synthetic reference frames**, where it is the natural control for "does
  the focal stack add anything." Also missing: a constant/mean-depth sanity baseline
  (relevant because `fit_scale`'s s≤0 fallback already emits a weighted-mean constant
  map — slanted scale_fit 0.297) and a scale-only fit on confident focus pixels to
  isolate the affine-vs-scalar point Table 1 makes. Separately, the learned
  "component" in Table 2 is oracle-aligned to GT (webpage says "Aligned DA-V2";
  only the slides say "oracle") — generous labeling for a comparison the deployed
  system cannot perform.
- **Fix.** Add a "DA-V2 Metric Indoor (unaligned)" row to Table 2, plus a
  constant-depth row; label the aligned row "oracle-aligned."

### 6. [SHOULD-FIX] Single-realization results with no variance; margins within seed noise

- **Evidence.** Scene and stack noise are seeded per scene index, and the same
  nominal configuration gives materially different numbers across realizations:
  stairs `refined,all` AbsRel is **0.090 in `refine_synth.csv` (seed 4) vs. 0.188 in
  `ablation_mu.csv` at the same μ=0.01 (seed 1)** — a 2× swing. `synth_dff.csv`
  (seed 0) and `refine_synth.csv` (seed i) also disagree on identical nominal
  quantities (e.g., objects dff_raw 0.4524 vs. 0.4567; corridor dff_raw RMSE 1.1706
  vs. 1.1617). The headline margins (combined 0.206 vs. focus 0.221/0.233, gaps of
  0.015-0.027) are the same order as this per-scene realization noise, and only 4
  eval scenes exist.
- **Fix.** Report mean ± std over ≥3 seeds per scene, or at minimum state the number
  of realizations and the per-scene table so readers can judge spread.

### 7. [SHOULD-FIX] δ1 regression is reported but never explained on the page

- **Claim (implicit).** The abstract and slide 7 call combined "the best of anything
  here" on the headline numbers.
- **Evidence.** Table 2 itself shows combined δ1 0.704 < DFF raw 0.732, and the
  per-scene eval rows show δ1 losses on objects (0.219 vs. 0.328) and shelves
  (0.784 vs. 0.926). The only explanation offered lives in `qa_prep.md:34-39`
  (an internal Q&A answer), not in the graded write-up.
- **Fix.** Add one sentence to the webpage discussion acknowledging the δ1 trade-off
  and why AbsRel is the primary metric.

### 8. [SHOULD-FIX] The confidence-weighting design choice loses its own ablation

- **Claim.** Table 4 (index.html:182) — "Uniform weights are a little better on
  gallery and a little worse on stairs; the means are nearly tied"; learned `w` is
  the default behind Table 2.
- **Evidence.** `ablation_conf.csv` mean: uniform AbsRel 0.162 < conf 0.176 and
  uniform δ1 0.719 > conf 0.717; on gallery alone uniform is 0.114 vs. 0.164 (30%
  better, not "a little"). The chosen default is not the winner of its own tuning-set
  ablation on the headline metric.
- **Fix.** Justify the choice (e.g., per-scene worst-case behavior or DDFF
  robustness), or report the uniform-weights variant in Table 2.

### 9. [NIT] Abstract implies quantitative evaluation on DDFF

- index.html:26 lists DDFF among the datasets "reporting AbsRel, RMSE, and δ1";
  `ddff_dff.csv` metrics are all `nan` by design. Say "qualitative on DDFF" up front.

### 10. [NIT] DDFF confidence range is misstated on the page

- index.html:246 says confidences are "roughly 0.03-0.05"; `ddff_dff.csv` actual range
  is 0.0248-0.0508. The slides/midterm say 0.02-0.05. Pick one (the latter is right).

### 11. [NIT] "Extreme residuals" is not a rejection trigger

- index.html:81 lists "nonpositive fitted slope, or extreme residuals" as causing
  rejection. `refine.py:fit_scale` rejects on s ≤ 0, coverage < 0.02, or solve
  failure only; residuals are trimmed (worst 10% dropped, refit once), not rejected.

### 12. [NIT] "Every prediction is aligned to ground truth before scoring"

- index.html:61 — Table 1 itself contains the unaligned metric-indoor row. Rephrase
  to "aligned scores."

### 13. [NIT] DDFF runs different pipeline settings than reported

- `run_refine_ddff.py:65` calls `refine(d_inv, z_f, conf, norm=True)`, which uses the
  library default `lam=3.0` (`refine.py:113`), not the λ=1 reported for Table 2, plus
  a per-scene confidence renormalization. The normalization is disclosed in slides
  and midterm but not on the webpage; the λ difference is disclosed nowhere.

### 14. [NIT] The "monocular" reference image is selected using the whole stack

- `run_refine.py:29,75` feeds DA-V2 the sharpest frame of the stack (`sharp_ref`),
  chosen by a global focus measure over all 10 frames. Legitimate, but the
  "reference image" wording (index.html:72) glosses over stack-dependent selection.

### 15. [NIT] Corridor "textureless ceiling" number covers more than the ceiling

- index.html:243 — the 0.42 figure is `synth_dff.csv` corridor `dff_raw,textureless`
  = 0.4203, where the mask is ceiling ∪ back-wall ∪ low-texture regions
  (`synth.py:129`), not the ceiling alone.

### 16. [NIT] Silent GT-derived fallback in the learned branch

- `run_refine.py:45-66` — if DA-V2 weights are unavailable, `learned_inv` returns
  `fake_learned` (blurred inverse GT + affine + noise) with only a stdout notice, and
  `refine_synth.csv` records no provenance. The committed CSV provably used real
  DA-V2 (a stand-in would produce near-perfect aligned rows; see `docs/repro_notes.md`),
  but a grader without torch would silently regenerate flattered numbers.
  Fix: hard-fail or write a `prior` column (the uncommitted diff in progress adds one).

### 17. [NIT] Eigen crop boundary is off by one pixel vs. common convention

- `run_nyu_models.py:42` uses rows 45:472, cols 41:602; the conventional Eigen crop is
  45:471 × 41:601 (one extra row/column, ~0.5% more pixels). Immaterial to the
  numbers; note it for protocol fidelity.

## Verdict

The benchmark half of this project is solid: Table 1 reproduces to the fourth decimal
against `nyu_models.csv` on the correct 654-image Eigen protocol, alignment variants are
honestly disclosed, and the tune/eval split on the synthetic scenes is genuinely held
out. But the contribution section overclaims its central mechanism: the fallback gate is
presented as a guarantee that the combined system "never does worse than either of its
components," and that is demonstrably false on the project's own eval CSV — on shelves
the accepted merge loses to the focus estimate (0.212 vs. 0.188), on objects it loses to
every alternative component reading, and on δ1 it loses even in aggregate. The real-data
leg is weaker than presented: the implemented DDFF focus distances contradict the
documented reconstruction, and the two most-quoted DDFF numbers (reversed-order rate,
acceptance rate) have no supporting artifact. Combined with single-realization results
whose margins sit inside observable seed noise, a missing monocular-metric baseline, and
an ablation that does not support the chosen confidence weighting, the paper needs a
revised claims pass — not new ideas, just honesty about what the data shows.

**Counts: 1 BLOCKER, 7 SHOULD-FIX, 9 NIT.**
Worst three: #1 (false component-dominance guarantee — the method's headline claim),
#2 (DDFF focus distances contradict the disclosed reconstruction),
#6 (headline margins within unreported realization noise).

# Speaker notes — 5 minute video

Target: about 700 spoken words at a normal pace. Read it out loud once before
recording; if you run long, cut the parentheticals. Speaker changes are marked.
Dimitri takes slides 1-3 and 7-8, Vincent takes 4-6 and 9-10.

---

## Slide 1 — Title (0:00-0:12) — Dimitri

Hi, we're Dimitri and Vincent. Our project is on monocular depth estimation.
We benchmarked Depth Anything V2 against classical depth from focus, and then
we built a refinement that uses focus measurements to recover metric scale.

## Slide 2 — Problem (0:12-0:50) — Dimitri

One photo cannot tell you meters. A small object up close looks the same as a
big one far away, so learned models predict depth only up to scale and shift.
To get real distances you have to align each prediction to something. Defocus
is different. Blur is set by lens geometry, so a focal stack carries an
absolute scale cue. The plot shows blur versus scene depth for three focus
settings. The curve lives in meters, not relative units. That is the signal we
use.

## Slide 3 — Pipeline (0:50-1:20) — Dimitri

Here is the system. The top branch runs Depth Anything on one image and gives
relative inverse depth. The bottom branch runs depth from focus on a focal
stack and gives a metric estimate plus a confidence map. Our part is the box in
the middle: it fits the learned output to the focus measurements and adds a
regularized local correction. If the fit looks degenerate we reject it and keep
the focus estimate instead of shipping a bad scale.

---> HAND OFF TO VINCENT <---

## Slide 4 — Learned baseline (1:20-2:00) — Vincent

First the benchmark. On 654 NYU test images, the aligned ViT-L gets AbsRel
0.055 and delta one 0.961. Two things stand out. Median scale alignment
collapses to 0.494, because a single scalar cannot fix the offset in inverse
depth. You need at least an affine fit. And the metric indoor checkpoint,
which is supposed to give meters directly, scores 0.218 unaligned. Its scale
prior is biased on NYU. So learned depth is strong but the scale problem is
real. We use ViT-S as the refinement backbone since bigger models add little.

## Slide 5 — Depth from focus (2:00-2:35) — Vincent

For the classical branch, each pixel gets a focus measure per frame using the
modified Laplacian. We fit a parabola around the peak for sub-frame depth, and
the curve shape gives a confidence. Smoothing with the all-focus image fixes
flat regions. Where texture exists the estimate is accurate. Where it does
not, the confidence reads low, which is exactly what the next stage needs.

## Slide 6 — Scale fitting (2:35-3:10) — Vincent

Our contribution fits the learned inverse depth to the focus estimate. We
solve for a global affine scale and shift plus a local correction r. The fit
is confidence weighted with a Huber loss, so only trustworthy focus pixels
pull on it. Two penalties keep the correction small and smooth where the
confidence is weak. And if the prediction anticorrelates with focus, the scale
goes negative, we detect that and mark the output uncalibrated.

---> HAND OFF TO DIMITRI <---

## Slide 7 — Synthetic results (3:10-3:45) — Dimitri

Results on the four held-out synthetic scenes, scored in meters with no
alignment. Raw depth from focus gets 0.233. Oracle-aligned Depth Anything gets
0.271, worse, because our renderer is far from its training data. Our refined
output alone gets 0.259. The combined system, which keeps the focus estimate
when calibration is rejected, gets 0.206, the best mean error of anything
here. It is not uniformly better though: on shelves the fallback would have
been cleaner, and delta-one is a touch below raw DFF. The honest summary is
that it buys mean error where the stack has texture.

## Slide 8 — Failure case (3:45-4:10) — Dimitri

This slide is why the fallback exists. On the slanted scene the learned
prediction anticorrelates with the focus estimate, so the fitted scale goes
negative. Scale fit alone scores 0.297 against 0.087 for focus. Instead of
emitting that, the system marks the scene uncalibrated and returns the focus
depth, which you can see is flat in the refined panel. That rejection is most
of the combined method's gain.

---> HAND OFF TO VINCENT <---

## Slide 9 — Real stacks (4:10-4:40) — Vincent

On real data we used the DDFF 12-Scene dataset. The test split ships 120 Lytro
focal stacks but no ground truth, so this part is qualitative. Focus distances
are not recorded either; we rebuilt them from the paper's refocus geometry.
Real stacks have much lower confidence than synthetic ones, but with per-scene
normalization the calibration is accepted on 110 of the 120 stacks and the
refined maps look plausible.

## Slide 10 — Takeaway (4:40-5:00) — Dimitri

To wrap up: aligned Depth Anything is excellent on real indoor scenes, but
metric checkpoints and naive scaling are not safe defaults. Focus gives true
scale where texture exists, and confidence-weighted fitting turns it into a
usable prior. Remaining limits are texture dependence, veiling at edges, and
no DDFF labels. Thanks, and we will take questions live.

---

(~705 spoken words; roughly 5:00 at 140 wpm. If rehearsal runs over 5:05,
drop the last bullet of slide 4 and the parenthetical on slide 7.)

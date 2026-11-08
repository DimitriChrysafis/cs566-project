# Q&A prep — likely questions

Two minutes of live Q&A after the video. Short answers; defer anything deep to
the report or the webpage tables.

**Why does median scale alignment fail so badly (0.494)?**
A single scalar can rescale the prediction but cannot fix the offset in
inverse depth, and the median is easily pulled by outliers and sky or wall
regions. An affine fit q = s*d + t absorbs both and is the minimum that works.

**Why does your combined method beat DA-V2 on synthetic scenes while DA-V2 wins
on NYU?**
NYU is indoor RGB-D data close to DA-V2's training distribution. Our rendered
scenes are far from it, so even oracle-aligned DA-V2 degrades there. We read
the synthetic benchmark as a stress test, not a real-data ranking.

**How do you get focus distances for DDFF if they are not recorded?**
We reconstructed them from the DDFF paper's refocus geometry: disparities
linear in [0.02, 0.28] px, z approximately 0.14/d. We verified the
frame-to-depth ordering per stack with Spearman correlation against the
learned prior; about a third of the stacks prefer the reversed order.

**What happens when a scene has no texture?**
The focus curve is flat, so the confidence goes to about zero and those pixels
exert no pull on the fit. The regularizers then pin the correction to zero and
the output stays near the global affine fit; if too little reliable signal
exists at all, the fit is rejected and we return the focus estimate.

**Why fit in inverse depth instead of depth?**
Relative models like DA-V2 output disparity-like inverse depth, where the
ambiguity is affine. Fitting q = s*d + t in that space is the natural
parameterization; z = 1/q converts back to meters.

**DFF raw has a higher delta-one than your combined output. Why claim it is
worse?**
Raw per-pixel maxima stay sharp where texture exists, which keeps delta one
high (0.732 vs 0.704). But its textureless outliers inflate AbsRel to 0.233
versus our 0.206. We optimize for the mean error, and the combined system is
the best on that.

**What does the confidence map actually measure?**
Peak sharpness and prominence of the per-pixel focus curve; a flat or
ambiguous curve gives low w. It tracks texture well: about 0.06-0.26 mean on
textureless regions versus 0.26-0.66 on textured ones.

**How did you pick lambda and mu, and did you tune on the test scenes?**
Both were tuned only on the two tuning scenes (gallery, stairs); the four
reported scenes were never touched during tuning. lambda = 1.0 and mu = 0.01
sat in a flat region of the ablation.

**How fast is it?**
DA-V2 ViT-S runs about 55 ms per image on a laptop GPU. The DFF and
refinement stages are CPU numpy over the stack and run in seconds per scene.

**What is defocus veiling?**
Blurred texture from a near surface bleeds over a flat region behind it and
lifts the focus measure there, so confidence can read high at occlusion
boundaries where the estimate is actually wrong. It is a real failure mode of
the measurement, not a renderer bug.

# MERF: Maximum-Entropy Residual Flow for CZSL

Status: leading generative candidate; closed-world evidence is promising, while
open-world calibration is not solved.

## Motivation

A seen visual composition embedding contains primitive main effects, reusable
instance variation, and pair-specific interaction leakage. Empirically resampling
seen residuals can copy the last component into unseen pairs; using only the
deterministic additive/flow mean loses diversity. The desired initial distribution
should preserve residual energy and coverage without retaining the identity of a
seen composition.

## Method hypothesis

1. Fit an ANOVA-style additive prior to seen composition features:
   `m(a,o) = global + attribute(a) + object(o)`.
2. Collect residuals `r = z - m(a,o)` and retain their global first moment and
   average squared radius.
3. Replace the empirical residual bank by the isotropic Gaussian with those
   moments. Under a fixed second-moment constraint, this is the maximum-entropy
   residual prior; isotropy deliberately removes pair-specific directions.
4. Warm-start each target pair at `m(a,o) + sigma * epsilon` and learn a
   primitive-text-conditioned rectified flow to transport this anonymized prior to
   the seen visual distribution.
5. Generate balanced support for unseen pairs, train a lightweight cosine
   classifier, and inject only validation-approved residual evidence into frozen
   CAMS logits.

The intended claim is not merely that Gaussian noise regularizes training. It is
that maximum-entropy residual anonymization resolves a CZSL-specific conflict:
retain distributional coverage while preventing seen interaction leakage from
being mistaken for transferable primitive structure.

## Evidence so far

- UT-Zappos CAMS seed0, three Gaussian generation seeds: mean closed test AUC/HM
  `45.04/57.43` versus CAMS `43.76/56.56` (`+1.28/+0.87`).
- Independent CAMS seed1, three generation seeds: mean closed test AUC/HM
  `42.48/54.98` versus `40.45/53.97` (`+2.03/+1.00`).
- MIT-States CAMS seed0: `+0.29/+0.23` closed test AUC/HM.
- Independently trained MIT-States CAMS seed1, three Gaussian generation seeds:
  mean closed test AUC/HM gain `+0.186/+0.099` point, with every seed improving
  both metrics. The effect is stable but much smaller than UT.
- Deterministic residual-scale-zero ablation on UT seed0: only `+0.32/-0.21`;
  stochastic residual coverage is necessary.
- A parameter-matched direct endpoint regressor (same network and sampling head,
  three generator seeds) averages `43.65/56.46` test AUC/HM, slightly below CAMS.
  The eight-step flow averages `44.94/57.41`. A one-step flow diagnostic reaches
  only `43.97/56.40`. Thus the gain is not explained by adding another MLP;
  learned multi-step transport dynamics are necessary.
- A held-out nearest-centroid identity probe predicts 83 seen pairs from empirical
  residuals at `18.40%` accuracy versus `1.20%` chance. Moment-matched diagonal and
  isotropic Gaussian residuals score `1.32%` and `1.35%`, respectively. Empirical
  residuals therefore retain strong pair identity, while Gaussian resampling
  anonymizes it almost completely. Between-pair centroids account for `6.07%` of
  total empirical residual energy.
- The same leakage is stronger on MIT-States: `52.66%` accuracy over 1,262 seen
  pairs versus `0.079%` chance, with `23.52%` between-pair residual energy. On a
  fixed 256-pair C-GQA subset, empirical accuracy is `17.01%` versus `0.39%`
  chance, and between-pair energy is `26.62%`; Gaussian probes again fall to chance.
- Axis-wise variance does not support the hypothesis that C-GQA uniquely requires
  an anisotropic prior. Diagonal participation-rank fractions are UT `0.869`, MIT
  `0.983`, and C-GQA `0.972`: UT is actually the most anisotropic dataset. The more
  plausible C-GQA bottleneck is statistical density and label-space scale (roughly
  4.8 real images per seen pair across 5,592 seen pairs), not loss of a few dominant
  residual axes.
- The C-GQA diagonal-Gaussian control confirms this diagnosis. Validation AUC gain
  is only `+0.059` point and frozen test AUC/HM changes by `-0.046/-0.268`; it is
  disabled by the uniform `+0.20` validation gate. Retaining per-axis variance is
  insufficient.
- C-GQA still contains substantial complementarity: a validation-label per-image
  oracle improves fixed-fusion AUC/HM by `+4.02/+6.01`. Output-only numeric routing
  reaches `0.740` pair-held-out AUROC for margin-improvement labels but yields only
  `+0.018` AUC point under its best sparse fusion; primitive-topology biases reduce
  AUROC to `0.595`. The remaining headroom cannot be captured by post-hoc scalar
  reliability.
- On two edge-preserving, training-only C-GQA pair-held-out folds, standard MERF
  classifies real pseudo-unseen pairs at `43.48/64.11` and `46.38/67.35` AUC/HM,
  far above text-only `27.22/44.40` on fold0. This confirms that the generator has
  compositional transfer capacity even where formal benchmark fusion is weak.
- Nested primitive/interaction flow improves those folds only marginally, and its
  formal C-GQA validation gain is just `+0.076` AUC point. Barycentric recentering
  strongly improves the two held-out folds but gives only `+0.062` on formal
  C-GQA validation and `+0.035/-0.256` AUC/HM on UT test. Both extensions are
  diagnostics, not components of the promoted method.
- C-GQA has no reliable gain and should be disabled by a uniform validation gate.
- UT open-world currently fails (`-0.79/-1.81` test AUC/HM); interaction-only
  post-processing worsens the split shift and is rejected.

## Novelty boundary

- FlowComposer composes learned primitive velocity fields and exploits leaked
  features. MERF instead makes residual anonymization the central object: it
  discards pair-identified residual directions using a maximum-entropy prior before
  conditional transport.
- DIFFCZSL uses diffusion-model intermediate representations as a regularizer;
  MERF generates task-space composition support and does not use diffusion features.
- Visual Proxy, ClusPro, and LGPRO learn proxies/prototypes; MERF models a full
  transferable residual distribution rather than a finite set of centers.
- Generic feature generation is not enough for a claim. The paper must isolate
  maximum-entropy anonymization, conditional transport, and fusion in matched
  ablations against empirical bootstrap, diagonal Gaussian, deterministic mean,
  additive-only, direct endpoint regression, and one-step/flow-free sampling.

## Promotion requirements

1. Finish independently trained CAMS seed2 baselines on UT and MIT; seed0/1 already
   show positive closed-world trends. Replay one shared protocol and report
   mean/std across base seeds, not only generation seeds.
2. Diagnose C-GQA by residual anisotropy, sample scarcity, and candidate count;
   either obtain a validation-predictable gain or uniformly abstain.
3. Solve open-world candidate-space calibration without using UT test for formula
   selection. Develop on a fresh validation protocol and freeze before evaluation.
4. Compare directly with FlowComposer and DIFFCZSL in mechanism, accuracy, memory,
   sampling time, and inference overhead.
5. Report residual pair-identifiability before/after anonymization. A successful
   maximum-entropy prior should reduce the accuracy of a seen-pair probe while
   retaining total residual energy and improving unseen classification.

## Next high-value diagnostic

Complete the official FlowComposer Stage-1/Stage-2 comparison under its released
UT-Zappos configuration, then report accuracy, memory, wall time, and whether its
feature-leak augmentation is required. In parallel, quantify held-out-pair
generation quality versus samples per pair and candidate count. Any C-GQA successor
must change representation training under pair-held-out episodes; do not spend more
trials on scalar interaction scales, covariance shape, recentering, or output gates.

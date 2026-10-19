# MECoFlow verified paper numbers

Generated directly from archived JSON files by `research/verify_paper_results.py`.

## Closed-world main and cross-checkpoint results

| Setting | CAMS AUC | MECoFlow AUC | Delta AUC | CAMS HM | MECoFlow HM | Delta HM | Shared setting |
|---|---:|---:|---:|---:|---:|---:|---|
| MIT-States / CAMS checkpoint | 23.124 | 23.416 +/- 0.029 | +0.293 | 39.895 | 40.130 +/- 0.073 | +0.235 | update x 0.2 |
| UT-Zappos / CAMS checkpoint | 43.758 | 45.038 +/- 0.126 | +1.280 | 56.556 | 57.429 +/- 0.313 | +0.874 | update x 1 |
| UT-Zappos / CAMS seed-1 | 40.448 | 42.482 +/- 0.581 | +2.034 | 53.972 | 54.975 +/- 0.407 | +1.003 | update x 1 |
| MIT-States / CAMS seed-1 | 23.297 | 23.517 +/- 0.043 | +0.220 | 40.329 | 40.398 +/- 0.103 | +0.069 | update x 0.5 |

C-GQA is gated to CAMS because its validation gain is only +0.051 AUC points (< 0.2); raw test delta is -0.244 AUC.

## Fixed source-ensemble upgrade

These fixed-setting source ensembles are checkpoint-level evidence, not a dataset-independent confidence interval:

| Setting | CAMS AUC | Fused AUC | Delta AUC | CAMS HM | Fused HM | Delta HM | Protocol |
|---|---:|---:|---:|---:|---:|---:|---|
| UT diagonal-source mean ensemble | 43.758 | 45.127 | +1.369 | 56.556 | 57.649 | +1.093 | diagonal mean, update x 1.25 |
| UT seed-1 diagonal-source mean ensemble | 40.448 | 42.122 | +1.674 | 53.972 | 54.724 | +0.752 | diagonal mean, update x 0.75 |
| UT six-source median ensemble | 43.758 | 45.022 | +1.263 | 56.556 | 57.555 | +1.000 | isotropic+diagonal median, update x 1.25 |
| UT inverse-consensus gated ensemble | 43.758 | 44.870 | +1.111 | 56.556 | 57.602 | +1.046 | inverse agreement gate, update x 0.75 |
| UT support-aware consensus ensemble | 43.758 | 45.063 | +1.305 | 56.556 | 57.788 | +1.232 | harmonic support + inverse agreement, update x 0.75 |
| MIT seed-1 isotropic-source median ensemble | 23.297 | 23.502 | +0.206 | 40.329 | 40.441 | +0.112 | source median, update x 0.75 |
| UT seed-1 support-only ensemble | 40.448 | 42.309 | +1.861 | 53.972 | 54.941 | +0.969 | min image support, update x 0.75 |
| MIT seed-1 support-only ensemble | 23.297 | 23.573 | +0.276 | 40.329 | 40.354 | +0.026 | min image support, update x 0.5 |
| UT factorized object-support ensemble | 43.758 | 45.228 | +1.470 | 56.556 | 57.508 | +0.953 | object support mean, update x 1.0 |
| UT confidence-gated ensemble | 43.758 | 45.264 | +1.506 | 56.556 | 57.585 | +1.029 | low base entropy gate, update x 1.0 |
| MIT seed-1 confidence-gated ensemble | 23.297 | 23.498 | +0.202 | 40.329 | 40.493 | +0.164 | high base entropy gate, update x 0.5 |
| UT joint support-confidence ensemble | 43.758 | 45.031 | +1.273 | 56.556 | 57.392 | +0.836 | object support x high margin, update x 1.0 |
| MIT seed-1 joint support-confidence ensemble | 23.297 | 23.458 | +0.161 | 40.329 | 40.351 | +0.022 | attribute support + low entropy, update x 0.25 |
| UT adaptive source-posterior ensemble | 43.758 | 44.748 | +0.990 | 56.556 | 57.273 | +0.718 | median + disagreement/confidence posterior, update x 1.5 |
| MIT seed-1 adaptive source-posterior ensemble | 23.297 | 23.553 | +0.257 | 40.329 | 40.610 | +0.282 | mean + disagreement/entropy posterior, update x 0.75 |
| MIT original adaptive source-posterior ensemble | 23.124 | 23.486 | +0.362 | 39.895 | 40.073 | +0.177 | mean + disagreement/entropy posterior, update x 0.75 |
| UT graph-conditioned adaptive posterior ensemble | 43.758 | 42.861 | -0.897 | 56.556 | 56.296 | -0.260 | attribute support x source posterior, strength 1.5 |
| MIT seed-1 graph-conditioned adaptive posterior ensemble | 23.297 | 23.565 | +0.268 | 40.329 | 40.461 | +0.133 | min support + source posterior, strength 0.5 |
| UT locked reliability residual mixture | 43.758 | 44.711 | +0.952 | 56.556 | 57.273 | +0.718 | adaptive/support residuals, alpha 0.875 |
| MIT seed-1 locked reliability residual mixture | 23.297 | 23.602 | +0.305 | 40.329 | 40.644 | +0.315 | adaptive/support residuals, alpha 0.5 |
| MIT original locked reliability residual mixture | 23.124 | 23.479 | +0.355 | 39.895 | 40.036 | +0.140 | imported alpha 0.5, adaptive/support residuals |
| UT interaction hyper-classifier | 43.758 | 43.950 | +0.192 | 56.556 | 56.636 | +0.080 | pair-specific low-rank weights, full x 0.2 |
| UT pair-level source-family router | 43.758 | 45.134 | +1.376 | 56.556 | 57.649 | +1.093 | train support^0.25 route, update x 1.25 |
| UT seed-1 pair-level source-family router | 40.448 | 42.632 | +2.184 | 53.972 | 55.156 | +1.184 | inverse train support route, update x 1.0 |
| MIT original pair-level source-family router | 23.124 | 23.459 | +0.335 | 39.895 | 40.001 | +0.105 | inverse family-disagreement route, update x 0.5 |
| UT selective unseen-pair source router | 43.758 | 44.418 | +0.659 | 56.556 | 57.320 | +0.765 | pair support^0.5, unseen-only, full x 1.5 |
| UT seed-1 selective unseen-pair source router | 40.448 | 42.184 | +1.736 | 53.972 | 55.270 | +1.298 | pair support^0.25, unseen-only, full x 1.5 |
| MIT selective pair source router | 23.124 | 23.459 | +0.335 | 39.895 | 40.001 | +0.105 | pair support^0.25, all-pair, update x 0.5 |
| UT seed-1 soft selective pair router | 40.448 | 42.272 | +1.824 | 53.972 | 55.427 | +1.455 | pair support^0.25, seen lambda 0.25, full x 1.5 |
| MIT soft selective pair router | 23.124 | 23.384 | +0.260 | 39.895 | 40.001 | +0.105 | pair support^0.5, seen lambda 0.75, update x 0.5 |
| UT seed-1 Pareto route blend | 40.448 | 42.586 | +2.138 | 53.972 | 55.241 | +1.269 | 0.5 inverse-pair update + 0.5 soft selective full |
| UT seed-1 instance-adaptive route blend | 40.448 | 46.745 | +6.298 | 53.972 | 60.951 | +6.979 | unseen-pair CAMS entropy schedules inverse/soft routes |
| UT seed-1 sigmoid-threshold route blend | 40.448 | 46.892 | +6.444 | 53.972 | 61.235 | +7.263 | unseen-pair sigmoid(8*(entropy-0.5)) inverse/soft routes |
| UT seed-1 candidate-conditioned route | 40.448 | 48.808 | +8.360 | 53.972 | 65.835 | +11.863 | pair-score sigmoid(-4*(z-0)) inverse/soft routes |
| UT original instance-adaptive route blend | 43.758 | 49.041 | +5.283 | 56.556 | 62.126 | +5.571 | unseen-pair CAMS entropy schedules inverse/soft routes |
| UT original sigmoid-threshold route blend | 43.758 | 49.082 | +5.324 | 56.556 | 62.179 | +5.624 | unseen-pair hard entropy threshold 0.7 inverse/soft routes |
| UT original candidate-conditioned route | 43.758 | 51.790 | +8.031 | 56.556 | 67.756 | +11.200 | pair-score sigmoid(-4*(z-0.5)) inverse/soft routes |
| MIT seed-1 soft selective diagonal replay | 23.297 | 25.362 | +2.066 | 40.329 | 45.579 | +5.250 | constant soft route, pair support^0.25, seen lambda 0.25 |
| MIT seed-1 threshold soft-route replay | 23.297 | 25.371 | +2.074 | 40.329 | 45.579 | +5.250 | reverse hard entropy threshold 0.9, soft route |
| MIT seed-1 candidate-conditioned route | 23.297 | 26.677 | +3.381 | 40.329 | 49.354 | +9.025 | pair-score sigmoid(-4*(z-0.5)) inverse/soft routes |
| C-GQA threshold selective transport | 16.325 | 16.555 | +0.231 | 34.780 | 36.743 | +1.964 | reverse hard entropy threshold 0.8, inverse/soft routes |
| C-GQA structural-risk selective transport | 16.325 | 16.607 | +0.283 | 34.780 | 36.899 | +2.119 | entropy + support risk (beta .5, slope 8, threshold .8) |
| C-GQA candidate-conditioned route (single-source exploratory) | 16.325 | 17.455 | +1.130 | 34.780 | 40.386 | +5.606 | pair-score sigmoid(-8*(z+0.5)) inverse/soft routes |
| C-GQA candidate-conditioned route (three-source) | 16.325 | 17.546 | +1.222 | 34.780 | 40.722 | +5.942 | pair-score sigmoid(-8*(z+0.5)) inverse/soft routes; 3 source pairs |

All fixed candidates were selected from validation-only evidence; their audit JSONs record test_search_after_lock=false.
The machine-readable UT checkpoint summary is `output/research/warmflow_ut/ut_diagonal_ensemble_checkpoint_summary.json`;
the detailed fixed-test/audit pairs are retained under the corresponding warmflow_ut and warmflow_mit_seed1final directories.
The UT six-source median, inverse-consensus gate, and support-aware consensus gate each have 6/6 positive source-level AUC/HM gains;
the MIT seed-1 median has positive aggregate HM but only 1/3 positive source-level HM gains; the UT seed-1 support-only branch has 3/3 positive source-level AUC/HM gains; the MIT seed-1 support-only branch has 3/3 positive AUC and 2/3 positive HM source gains; the UT factorized object-support and confidence-gated branches have 6/6 positive source-level AUC/HM gains; the MIT confidence-gated branch has 3/3 positive AUC and positive aggregate HM.
The UT seed-1 joint support-confidence candidate is retained separately as a negative-transfer boundary: its validation gate passed, but fixed test changed by -2.416 AUC / -1.633 HM points with 0/3 positive source runs.
The joint gate is therefore a mechanism ablation with checkpoint-specific transfer risk, not a universal replacement for the main support-aware setting.
The adaptive source-posterior branch uses only per-image source disagreement and source confidence; it improves the MIT seed-1 fixed test by +0.257 AUC / +0.282 HM points, but remains a checkpoint-level mechanism ablation.
The same locked MIT adaptive candidate transfers to the original MIT checkpoint with +0.362 AUC / +0.177 HM points, providing a second checkpoint-level positive replay.
The graph-conditioned adaptive posterior is retained as a mixed transfer boundary: MIT seed-1 is +0.268 AUC / +0.133 HM, while UT fixed test is -0.897 / -0.260 despite +3.894 / +2.590 validation gains.
The new instance-adaptive route blend uses only validation-calibrated CAMS entropy on graph-unseen pairs to interpolate two already-frozen route residuals; it reaches +6.298 AUC / +6.979 HM on UT seed-1, with 3/3 positive source seeds. The test run only reuses validation entropy normalization and records test_search_after_lock=false.
The preregistered sigmoid-threshold variant (sigmoid(8*(entropy-0.5))) reaches +6.444 AUC / +7.263 HM on UT seed-1, with 3/3 positive source seeds; it is retained as the stronger threshold-schedule variant.
The new candidate-conditioned route uses frozen per-image, per-pair CAMS confidence (sigmoid slope -4, threshold 0) to select between the inverse-support AUC route and soft-support HM route. It reaches +8.360 AUC / +11.863 HM on UT seed-1, with 3/3 positive source-pair runs; source means are +7.872/+11.438.
The same mechanism independently reaches +5.283 AUC / +5.571 HM on the original UT checkpoint, also with 3/3 positive source seeds; the schedule is selected separately by validation.
The candidate-conditioned route independently reaches +8.031 AUC / +11.200 HM on the original UT checkpoint; its source mean is +8.049/+11.128, with all three source-pair runs positive.
The original UT checkpoint independently selects hard entropy threshold 0.7 from the same preregistered family and reaches +5.324 AUC / +5.624 HM, with 3/3 positive source seeds.
MIT seed-1 has a separate diagonal-source replay whose validation selects the constant soft-selective route; fixed test gains are +2.066 AUC / +5.250 HM with 3/3 positive source seeds. It is reported as cross-dataset soft-route evidence, not as an entropy-schedule gain.
MIT seed-1 also passes the threshold-schedule replay with +2.074 AUC / +5.250 HM and 3/3 positive source seeds; the small AUC difference from the constant route is retained as a checkpoint-specific schedule ablation.
The same candidate-conditioned route independently reaches +3.381 AUC / +9.025 HM on MIT seed-1; its three source-pair mean is +3.177/+8.888, with all source-pair gains positive. The candidate was fixed from validation before test access.
C-GQA has a limited positive selective-transport replay: validation locks reverse hard entropy threshold 0.8 and fixed test changes by +0.231 AUC / +1.964 HM. This uses one available source seed and is reported as preliminary transfer evidence, not a multi-seed universal claim.
C-GQA structural-risk selective transport combines frozen CAMS entropy with graph support deficiency; validation locks beta=0.5, slope=8, threshold=0.8 and fixed test changes by +0.283 AUC / +2.119 HM. It remains single-source preliminary evidence, not a multi-seed universal claim.
C-GQA structural-risk source audit replays the locked candidate on three explicitly listed source pairs with 3/3 positive test AUC and 3/3 positive HM; mean gains are +0.266 AUC / +1.640 HM. Two pairs use existing generation-seed artifacts and fail the validation AUC gate, so this is source-level robustness evidence rather than a new main result.
C-GQA candidate-conditioned routing is a separate single-source exploratory transfer: it reaches +1.130 AUC / +5.606 HM, but it is not promoted to the C-GQA main result or a multi-source claim.
C-GQA three-source candidate-conditioned routing upgrades this to locked multi-source evidence: fixed test changes are +1.222 AUC / +5.942 HM, with 3/3 source-pair AUC/HM gains positive and source means +1.135/+5.788. The candidate was selected on the six-source validation probe before test access.

## Joint reliability transfer boundary

The same validation-only joint support-confidence procedure was independently applied to the UT seed-1 checkpoint.
It passed the validation AUC gate (+0.430 AUC / +2.425 HM points), but the locked fixed test was negative:
**-2.416 AUC / -1.633 HM**, with 0/3 positive source runs.
This audited negative boundary is retained to prevent cross-checkpoint overclaiming.

## Source-distribution ablation

| Source | MIT AUC | MIT HM | UT AUC | UT HM |
|---|---:|---:|---:|---:|
| Empirical bootstrap | 23.417 +/- 0.012 | 40.133 +/- 0.028 | 44.458 +/- 0.158 | 56.620 +/- 0.229 |
| Diagonal moment-matched Gaussian | 23.420 +/- 0.012 | 40.130 +/- 0.073 | 45.030 +/- 0.152 | 57.429 +/- 0.313 |
| Isotropic max-entropy Gaussian | 23.416 +/- 0.029 | 40.130 +/- 0.073 | 45.038 +/- 0.126 | 57.429 +/- 0.313 |

On UT, isotropic exceeds diagonal by only +0.009 AUC points; the defensible conclusion is parity with lower seed variance, not a significant accuracy advantage.

## Stochasticity control

| Setting | CAMS AUC | MECoFlow AUC | Delta AUC | CAMS HM | MECoFlow HM | Delta HM | Shared setting |
|---|---:|---:|---:|---:|---:|---:|---|
| UT deterministic source (s=0) | 43.758 | 44.094 +/- 0.053 | +0.336 | 56.556 | 56.613 +/- 0.276 | +0.057 | full x 0.5 |
| UT stochastic max-entropy source | 43.758 | 45.038 +/- 0.126 | +1.280 | 56.556 | 57.429 +/- 0.313 | +0.874 | update x 1 |

The stochastic source contributes +0.945 AUC and +0.816 HM over s=0 with the same flows.

## Transport-depth and no-flow control

| Setting | CAMS AUC | MECoFlow AUC | Delta AUC | CAMS HM | MECoFlow HM | Delta HM | Shared setting |
|---|---:|---:|---:|---:|---:|---:|---|
| UT direct endpoint regressor | 43.758 | 43.934 +/- 0.111 | +0.176 | 56.556 | 56.620 +/- 0.196 | +0.064 | full x 0.2 |
| UT 1-step raw | 43.758 | 44.618 +/- 0.093 | +0.860 | 56.556 | 56.899 +/- 0.210 | +0.344 | update x 0.5 |
| UT 1-step gated | 43.758 | 43.758 +/- 0.000 | +0.000 | 56.556 | 56.556 +/- 0.000 | +0.000 | update x 0 |
| UT 8-step main | 43.758 | 45.038 +/- 0.126 | +1.280 | 56.556 | 57.429 +/- 0.313 | +0.874 | update x 1 |

The deterministic no-flow source-augmentation diagnostic reaches 40.984 AUC / 54.507 HM (-2.774/-2.049 versus CAMS). It has no generator-seed variance and is reported as a diagnostic, not a three-seed estimate.

One-step is positive on all three test seeds, but its validation gain is only +0.149 AUC points and therefore fails the 0.2-point reliability gate; eight-step gains +1.120 validation AUC points and passes.

With exactly the same network parameters, source, auxiliary loss, generated classifier, and selection protocol, eight-step rectified flow exceeds direct endpoint regression by +1.105 AUC and +0.809 HM.

## Source-location control

| Setting | CAMS AUC | MECoFlow AUC | Delta AUC | CAMS HM | MECoFlow HM | Delta HM | Shared setting |
|---|---:|---:|---:|---:|---:|---:|---|
| MIT zero-centered source | 23.124 | 23.335 +/- 0.051 | +0.211 | 39.895 | 39.986 +/- 0.090 | +0.090 | update x 0.2 |
| MIT additive compositional center | 23.124 | 23.416 +/- 0.029 | +0.293 | 39.895 | 40.130 +/- 0.073 | +0.235 | update x 0.2 |

The additive compositional center contributes +0.081 AUC and +0.144 HM over the same-energy zero-centered source.

## Open-world boundary

| Setting | CAMS AUC | MECoFlow AUC | Delta AUC | CAMS HM | MECoFlow HM | Delta HM | Shared setting |
|---|---:|---:|---:|---:|---:|---:|---|
| MIT open-world raw | 8.643 | 8.792 +/- 0.044 | +0.149 | 22.269 | 22.449 +/- 0.110 | +0.179 | update x 0.5 |
| MIT open-world gated | 8.643 | 8.643 +/- 0.000 | +0.000 | 22.269 | 22.269 +/- 0.000 | +0.000 | update x 0 |

UT raw single-seed test changes by -0.789 AUC and -1.812 HM; open-world is not a paper claim.

## Efficiency

- Flow parameters: 21.169M
- Classifier parameters: 1.507M
- Flow training: 28.1s
- Generation, classifier, and evaluation: 50.9s
- Peak CUDA memory: 1.51 GiB
- Batched inference: 11.6ms for 12,995 images (0.00089ms/image)

All consistency assertions passed.

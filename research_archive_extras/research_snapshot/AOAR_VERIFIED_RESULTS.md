# AOAR-Replay verified results

Generated directly from archived JSON files by `research/verify_aoar_results.py`.

## Shared validation selection

| Variant | AUC | Delta AUC | HM | Delta HM | Positive AUC seeds | Shared setting |
|---|---:|---:|---:|---:|---:|---|
| random / Gaussian sample | 45.931 +/- 0.096 | +0.760 | 62.025 +/- 0.255 | +0.378 | 3/3 | update x 1 |
| raster / Gaussian sample | 45.608 +/- 0.832 | +0.436 | 61.981 +/- 0.515 | +0.334 | 1/3 | update x 1 |
| random / conditional mean | 46.095 +/- 0.111 | +0.924 | 62.075 +/- 0.068 | +0.429 | 3/3 | update x 1 |

## Fixed shared-selection test

| Variant | CAMS AUC | AOAR AUC | Delta AUC | AUC-positive seeds | CAMS HM | AOAR HM | Delta HM | HM-positive seeds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random / Gaussian sample | 43.758 | 44.378 +/- 0.036 | +0.620 | 3/3 | 56.556 | 56.208 +/- 0.327 | -0.347 | 0/3 |
| random / conditional mean | 43.758 | 44.466 +/- 0.065 | +0.708 | 3/3 | 56.556 | 56.376 +/- 0.359 | -0.180 | 1/3 |

Conditional-mean generation improves AUC and reduces the HM loss relative to Gaussian sampling, but does not produce a stable HM gain.

All AOAR protocol and consistency assertions passed.

# MECoFlow validation-gate sensitivity audit

Generated from archived JSONs by `research/gate_sensitivity.py`. Thresholds and gains are in AUC percentage points.

## Validation evidence and raw test outcome

| Setting | Role | Validation gain | Raw test Delta AUC | Raw test Delta HM |
|---|---|---:|---:|---:|
| MIT | closed main | +0.257 | +0.293 | +0.235 |
| UT | closed main | +1.120 | +1.280 | +0.874 |
| UT seed-1 | closed checkpoint | +0.361 | +2.034 | +1.003 |
| MIT seed-1 | closed checkpoint | +0.273 | +0.220 | +0.069 |
| C-GQA | closed boundary | +0.051 | -0.244 | -0.347 |
| UT 1-step | ablation | +0.149 | +0.860 | +0.344 |
| UT direct | capacity control | +0.304 | +0.176 | +0.064 |
| MIT OW | open boundary | +0.149 | +0.149 | +0.179 |
| UT OW | open boundary | +1.204 | -0.789 | -1.812 |

## Decision sensitivity

`pass` means only that the raw generator would be enabled at that validation threshold; the table does not reselect fusion hyperparameters.

| Gate | MIT | UT | UT seed-1 | MIT seed-1 | C-GQA | UT 1-step | UT direct | MIT OW | UT OW |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| 0.05 | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| 0.10 | pass | pass | pass | pass | fallback | pass | pass | pass | pass |
| 0.15 | pass | pass | pass | pass | fallback | fallback | pass | fallback | pass |
| 0.20 | pass | pass | pass | pass | fallback | fallback | pass | fallback | pass |
| 0.25 | pass | pass | pass | pass | fallback | fallback | pass | fallback | pass |
| 0.30 | fallback | pass | pass | fallback | fallback | fallback | pass | fallback | pass |
| 0.40 | fallback | pass | fallback | fallback | fallback | fallback | fallback | fallback | pass |
| 1.00 | fallback | pass | fallback | fallback | fallback | fallback | fallback | fallback | pass |

## Interpretation

- On the closed-world main/checkpoint evidence, every threshold from 0.10 through 0.25 AUC points gives the same decision: keep all four positive settings and reject C-GQA. The chosen 0.20 is therefore not a knife-edge value.
- The 0.20 gate deliberately rejects the positive but weak-validation UT 1-step diagnostic; it trades recall for conservative evidence.
- UT open-world has a large positive validation gain but negative test transfer. No scalar validation-gain threshold that preserves the closed-world UT result can fix this domain shift, so the gate is not a universal reliability guarantee and open-world remains outside the claim.
- This is a post-hoc sensitivity audit of one uniformly applied protocol, not evidence that 0.20 was theoretically derived or preregistered.

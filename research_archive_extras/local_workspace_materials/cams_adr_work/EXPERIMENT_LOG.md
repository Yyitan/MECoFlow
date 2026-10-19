# ADR-CAMS Experiment Log

This file is the running record for code changes and experiments in `CAMS-ADR`.
Every new idea should add an entry with the git commit, commands, results, and conclusion.

## Baseline

- Date: 2026-07-27
- Commit: `0576a76`
- Source: clean copy of `./lyt/czsl/CAMS-main/CAMS`
- Notes: copied without `output/`, `weight/`, `__pycache__/`, `*.pyc`; initialized as an independent git repository.

## Pair-Affinity Soft Label Learning

- Status: seed0 sweep completed; multi-seed verification pending
- Code commit: `da78a17` (`feat: add pair-affinity soft label training`)
- Motivation: original CAMS treats all wrong compositions as equally negative. CZSL negatives have semantic structure: same-attribute, same-object, and text-similar pairs should be softer negatives than unrelated pairs.
- First target: UT-Zappos seed0
- Sweep:
  - `affinity_eps=0.05`
  - `affinity_eps=0.10`
  - `affinity_eps=0.15`
- Fixed parameters:
  - `affinity_tau=0.07`
  - `affinity_attr_weight=0.35`
  - `affinity_obj_weight=0.35`
  - `affinity_text_weight=0.30`
- Baseline reference, UT-Zappos seed0:
  - Closed: S 68.62, U 76.10, HM 56.56, AUC 43.76
  - Open: S 68.62, U 64.83, HM 52.13, AUC 36.25

### Implementation Notes

- Files changed:
  - `model/cams.py`
  - `flags.py`
  - `configs/utzappos_affinity_eps005.yml`
  - `configs/utzappos_affinity_eps010.yml`
  - `configs/utzappos_affinity_eps015.yml`
  - `run_affinity_ut_seed0.sh`
- Training change:
  - `com_loss` and `glb_loss` can use soft cross entropy with pair-affinity targets.
  - `att_loss` and `obj_loss` remain hard-label CE to keep primitive supervision stable.
- Pair-affinity target:
  - `(1 - eps) * one_hot(gt_pair) + eps * softmax(affinity / tau)`
  - `affinity = w_attr * same_attr + w_obj * same_obj + w_text * pair_text_cosine`
- Verification:
  - Remote `python -m py_compile model/cams.py flags.py` passed.

### UT-Zappos Seed0 Results

All values are percentages. Deltas are against the original CAMS seed0 result.

| Setting | Closed S | Closed U | Closed HM | Closed AUC | Open S | Open U | Open HM | Open AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CAMS baseline | 68.62 | 76.10 | 56.56 | 43.76 | 68.62 | 64.83 | 52.13 | 36.25 |
| eps=0.05 | 68.72 | 76.15 | 56.61 | 43.86 | 68.72 | 65.42 | 52.38 | 36.54 |
| eps=0.10 | 68.72 | 76.10 | 56.46 | 43.83 | 68.72 | 65.15 | 52.05 | 36.36 |
| eps=0.15 | 68.82 | 75.52 | 56.36 | 43.64 | 68.82 | 64.83 | 52.09 | 36.30 |

Best setting: `eps=0.05`.

- Closed delta: S +0.10, U +0.05, HM +0.05, AUC +0.10.
- Open delta: S +0.10, U +0.59, HM +0.25, AUC +0.29.
- Larger smoothing is worse: `eps=0.15` reduces closed U/HM/AUC.
- Conclusion: the direction is promising, especially for open-world unseen accuracy, but the gain is small and requires paired multi-seed verification.

Run record:

Run script:

```bash
cd ./lyt/czsl/CAMS-ADR
bash run_affinity_ut_seed0.sh
```

Completed from 2026-07-28 10:01 CST to 14:45 CST on GPU 2. Raw logs and checkpoints are under `output/ut_affinity_eps*_seed0/`.

## Plan: eps=0.05 Multi-Seed Verification and Target Diagnostics

- Date: 2026-08-01
- Status: completed
- Purpose:
  - compare affinity and original CAMS under the same seeds;
  - determine whether the nominal smoothing mass reaches non-ground-truth pairs;
  - avoid expanding the hyperparameter search before confirming stability.
- Experiments: UT-Zappos `eps=0.05`, seeds 1 and 2, sequentially on GPU 2.
- Paired original CAMS references:

| Seed | Closed S | Closed U | Closed HM | Closed AUC | Open S | Open U | Open HM | Open AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 69.40 | 75.67 | 55.16 | 44.10 | 69.40 | 63.51 | 50.59 | 36.09 |
| 2 | 68.62 | 74.83 | 55.48 | 43.40 | 68.62 | 64.62 | 50.99 | 36.81 |

- Target diagnostics (printed once per run):
  - ground-truth affinity/target probability;
  - total and top-1 non-ground-truth target mass;
  - mass assigned to same-attribute and same-object negatives;
  - target entropy.
- Output directories:
  - `output/ut_affinity_eps005_seed1/`
  - `output/ut_affinity_eps005_seed2/`
- Launch command:

```bash
cd ./lyt/czsl/CAMS-ADR
nohup bash run_affinity_ut_multiseed.sh > output/affinity_multiseed/nohup.log 2>&1 &
```

Completed from 2026-08-01 18:28 CST to 20:58 CST on GPU 2. Both runs completed without errors.

### Per-Seed Results

| Seed | Method | Closed S | Closed U | Closed HM | Closed AUC | Open S | Open U | Open HM | Open AUC |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | CAMS | 68.62 | 76.10 | 56.56 | 43.76 | 68.62 | 64.83 | 52.13 | 36.25 |
| 0 | affinity | 68.72 | 76.15 | 56.61 | 43.86 | 68.72 | 65.42 | 52.38 | 36.54 |
| 1 | CAMS | 69.40 | 75.67 | 55.16 | 44.10 | 69.40 | 63.51 | 50.59 | 36.09 |
| 1 | affinity | 69.50 | 75.73 | 55.32 | 44.29 | 69.50 | 63.41 | 50.35 | 36.15 |
| 2 | CAMS | 68.62 | 74.83 | 55.48 | 43.40 | 68.62 | 64.62 | 50.99 | 36.81 |
| 2 | affinity | 68.72 | 75.15 | 55.35 | 43.64 | 68.72 | 64.73 | 51.21 | 36.95 |

### Three-Seed Summary (mean +/- sample standard deviation)

| Method | Closed S | Closed U | Closed HM | Closed AUC | Open S | Open U | Open HM | Open AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CAMS | 68.88 +/- 0.45 | 75.53 +/- 0.65 | 55.73 +/- 0.73 | 43.75 +/- 0.35 | 68.88 +/- 0.45 | 64.32 +/- 0.71 | 51.24 +/- 0.80 | 36.38 +/- 0.38 |
| affinity | 68.98 +/- 0.45 | 75.68 +/- 0.50 | 55.76 +/- 0.74 | 43.93 +/- 0.33 | 68.98 +/- 0.45 | 64.52 +/- 1.02 | 51.31 +/- 1.02 | 36.55 +/- 0.40 |
| paired delta | +0.10 | +0.14 | +0.03 | +0.18 | +0.10 | +0.20 | +0.08 | +0.16 |

Paired AUC deltas are positive for every seed:

- Closed AUC: `+0.10`, `+0.19`, `+0.24`.
- Open AUC: `+0.29`, `+0.06`, `+0.14`.

HM is less stable: closed HM deltas are `+0.05`, `+0.16`, `-0.13`; open HM deltas are `+0.25`, `-0.24`, `+0.22`.

### Target-Distribution Diagnostics

| Seed | GT target prob. | Non-GT mass | Top non-GT | Same-attr mass | Same-obj mass | Entropy |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.997133 | 0.002867 | 0.000254 | 0.001546 | 0.001261 | 0.027656 |
| 2 | 0.997061 | 0.002939 | 0.000260 | 0.001736 | 0.001142 | 0.028252 |

### Conclusion

- The method produces a small but consistent AUC improvement: mean `+0.18` closed and `+0.16` open, with positive AUC deltas on all three seeds.
- Mean unseen accuracy also improves (`+0.14` closed, `+0.20` open), but HM changes are small and seed-dependent.
- The nominal `eps=0.05` only assigns about `0.29%` target mass to all non-GT pairs because the GT pair dominates the affinity softmax.
- Next recommended experiment: exclude the GT pair from the affinity softmax so the full epsilon is distributed over structured negatives, then use a conservative epsilon sweep because this changes the effective smoothing strength substantially.

## Plan: GT-Excluded Pair-Affinity Targets

- Date: 2026-08-23
- Status: prepared; results pending
- Motivation: in the first implementation, the GT pair dominates the affinity softmax. With nominal `eps=0.05`, only about `0.29%` total target mass reaches non-GT pairs.
- Change: add `affinity_exclude_gt`. When enabled, the GT affinity logit is masked before softmax, so the target is exactly:
  - GT pair: `1 - eps`
  - structured negative pairs together: `eps`
- Backward compatibility: the flag defaults to false, so all previous configs/checkpoints retain the original behavior.
- Dataset and seed: UT-Zappos seed0.
- Conservative sweep: `eps=0.005`, `0.01`, `0.02`.
- Fixed parameters: `tau=0.07`, attribute/object/text weights `0.35/0.35/0.30`.
- Selection criterion: validation AUC; report both closed- and open-world S/U/HM/AUC against the original CAMS and the previous affinity method.
- GPU: 2, sequential execution.
- Output directories:
  - `output/ut_affinity_xgt_eps0005_seed0/`
  - `output/ut_affinity_xgt_eps001_seed0/`
  - `output/ut_affinity_xgt_eps002_seed0/`
- Launch command:

```bash
cd ./lyt/czsl/CAMS-ADR
nohup bash run_affinity_xgt_ut_seed0.sh > output/affinity_xgt_sweep/nohup.log 2>&1 &
```

The raw logs, target diagnostics, final metrics, comparison, and conclusion must be appended here after completion.

# ADR-CAMS Experiment Log

This file is the running record for code changes and experiments in `CAMS-ADR`.
Every new idea should add an entry with the git commit, commands, results, and conclusion.

## Baseline

- Date: 2026-07-27
- Commit: `0576a76`
- Source: clean copy of `./lyt/czsl/CAMS-main/CAMS`
- Notes: copied without `output/`, `weight/`, `__pycache__/`, `*.pyc`; initialized as an independent git repository.

## Plan: Pair-Affinity Soft Label Learning

- Status: implemented, not yet trained
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

### Pending Results

Run script:

```bash
cd ./lyt/czsl/CAMS-ADR
bash run_affinity_ut_seed0.sh
```

Results should be appended here after the sweep finishes.

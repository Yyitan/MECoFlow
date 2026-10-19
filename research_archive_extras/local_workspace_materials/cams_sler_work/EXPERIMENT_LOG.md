# ADR-CAMS Experiment Log

This file is the running record for code changes and experiments in `CAMS-ADR`.
Every new idea should add an entry with the git commit, commands, results, and conclusion.

## 2026-08-24: Reliability-Gated WarmFlow Evidence (Rejected)

Motivation: unconditional structured posterior correction and fixed generated-logit
fusion expose the same failure mode: auxiliary evidence is not uniformly reliable.
We froze the WarmFlow classifiers and selected a per-image reliability gate from
validation data only, jointly over three generation seeds. The gate used only
label-free inference quantities (generated margin, base/generated score agreement,
and text/generated agreement); the cached test split was not accessed or evaluated
during selection.

- UT-Zappos validation selected generated-margin gating, q=0.25,
  temperature=0.25 IQR, full generated evidence, strength=1.0. All three validation
  seeds improved; mean AUC/HM gains were +0.43/+0.58 percentage points over CAMS.
- Frozen UT-Zappos test: AUC 43.76 -> 44.11 (+0.35), HM 56.56 -> 56.45 (-0.10).
  This is worse than fixed WarmFlow fusion (AUC 44.46, HM 56.62).
- MIT-States validation selected base-agreement gating, q=0.25,
  temperature=0.25 IQR, text-update evidence, strength=0.5. All validation seeds
  improved; mean AUC/HM gains were +0.30/+0.43 points over CAMS.
- Frozen MIT-States test: AUC 23.12 -> 23.36 (+0.23), HM 39.90 -> 40.11 (+0.22).
  Fixed WarmFlow remains marginally better (AUC 23.42, HM 40.13).

Decision: reject output-only scalar reliability gating as a paper contribution. It
retains some WarmFlow gain but does not improve fixed fusion, and the UT
validation-to-test shift remains. Next diagnose the oracle complementarity of the
base and generated branches; if headroom is low, improve generation rather than add
more calibration heuristics.

## 2026-08-24: WarmFlow Oracle Complementarity Diagnostic

Protocol: validation labels were used diagnostically to choose a fusion strength
independently for each image from {0, .05, .1, .2, .5, 1, 2, 4}. This is an oracle
upper bound and is not a valid method result.

- UT-Zappos fixed fusion: AUC 45.57, HM 62.09. Per-image oracle: AUC 49.28,
  HM 66.02, leaving +3.71/+3.94 points over fixed fusion. The oracle uses generated
  evidence for about 41% of images and chooses almost exclusively strength 0 or 4.
- MIT-States fixed fusion: AUC 26.74, HM 43.94. Per-image oracle: AUC 32.80,
  HM 50.84, leaving +6.06/+6.89 points. It uses generated evidence for about
  29--34% of images.
- Existing label-free margin/agreement features do not identify the useful subset:
  their AUROC is at most about .61 on UT and near chance (roughly .45--.49) on MIT.

Conclusion: the generated branch contains substantial complementary information;
the bottleneck is estimating candidate/sample reliability, not a low information
ceiling. This motivates uncertainty estimated from independent generative members.

## 2026-08-24: Candidate-Wise Generative Uncertainty (Rejected)

Three independent generated classifiers estimate per-image, per-composition mean
and epistemic standard deviation. Validation compares the ensemble mean, a lower
confidence bound, and variance shrinkage.

- UT-Zappos validation selected variance shrinkage with risk=.5, full evidence,
  strength=.5: AUC/HM gains +0.71/+0.66 points over CAMS.
- Frozen UT-Zappos test: AUC 43.76 -> 44.23 (+0.47), HM 56.56 -> 56.55 (flat).
  This is more stable than scalar gating but still below fixed WarmFlow AUC 44.46
  and does not improve HM.
- MIT-States validation rejected all variance penalties and selected the plain
  three-member mean, with AUC/HM gains +0.23/+0.33. Frozen test reached AUC 23.43
  and HM 40.16 (+0.30/+0.27 over CAMS), only +0.01/+0.03 above the fixed WarmFlow
  summary after rounding.

Decision: generated-classifier variance is not an effective candidate-level risk
signal. Keep the ensemble mean as a stronger low-variance baseline, but do not claim
uncertainty-aware fusion as the method. The oracle gap requires reliability learned
from structure or from held-out pseudo-compositions, not post-hoc score dispersion.

## 2026-08-24: Pair-Held-Out Reliability Router (Rejected)

A 19-dimensional label-free reliability representation (margins, entropy, JS
divergence, top-k agreement, primitive agreement and three-member uncertainty) was
fed to a small MLP. Every prediction was out-of-fold for the complete ground-truth
composition, so the diagnostic measured cross-composition rather than within-class
memorization.

- UT-Zappos OOF AUROC 0.518; best OOF fusion changed AUC by only +0.01 and HM by
  +0.06 percentage points.
- MIT-States OOF AUROC 0.598; best OOF fusion changed AUC/HM by +0.15/+0.33,
  still below fixed WarmFlow validation fusion.

Decision: even a learned output router does not transfer reliably across pair labels.
Stop score-level routing and move structural supervision into representation training.

## 2026-08-24: Leave-One-Composition-Out Prototype Completion (Rejected)

For each seen pair, its own samples were removed from its attribute neighborhood,
object neighborhood and global mean. A shared gated residual network reconstructed
the held-out visual centroid with primitive-sharing hard-negative contrastive loss.
At inference the same network completed unseen pair prototypes.

- UT validation selected the non-learned additive hybrid and showed a misleading
  AUC/HM +2.51/+1.87. Frozen test collapsed to +0.04/-0.51.
- MIT validation selected the learned hybrid (+0.27/+0.36); frozen test was only
  +0.19/+0.03.
- C-GQA validation selected learned all-pair completion (+0.19/+0.29); frozen test
  was +0.07/-0.19.

Decision: prototype completion improves unseen potential but does not preserve the
seen/unseen calibration curve. This reproduces the failure of post-hoc generated
scores and is not sufficient as the main method.

## 2026-08-24: Structured Marginal Coherence (Promoted to Test)

The next method moves global candidate structure into CAMS training. Composition and
global pair logits are marginalized with differentiable log-sum-exp over all pairs
sharing each attribute or object. The resulting row/column marginals receive ground
truth primitive CE plus reliability-weighted distillation from CAMS primitive
branches. Unlike test-time SPR, wrong primitive posteriors are not imposed after the
fact: all branches can learn a coherent representation during supervised training,
and inference remains unchanged. Initial UT seed0 run uses marginal weight .2,
distillation weight .1 and temperature 2, with test evaluation disabled during
training.

UT-Zappos seed0 validation trajectory (the test split remained disabled):

| Epoch | Val S | Val U | Val HM | Val AUC |
|---:|---:|---:|---:|---:|
| 1 | 68.53 | 70.90 | 60.18 | 44.78 |
| 2 | 65.45 | 74.15 | 61.41 | 45.58 |
| 3 | 70.01 | 73.43 | **63.28** | **48.26** |
| 4 | 68.99 | 70.22 | 60.74 | 45.15 |

The epoch-3 checkpoint improves validation AUC/HM by approximately +3.09/+1.63
percentage points over the original CAMS reference (45.17/61.65). It also raises
both the best seen and unseen accuracies, so the gain is not a one-sided bias trade.
Epoch 5 was interrupted at 82% before validation because two concurrent WarmFlow
jobs had started on GPUs 1 and 2; stopping SMC on GPU 3 restored the user-mandated
two-GPU limit. `output/utzappos_smc_seed0/val_best.pt` is the saved epoch-3 model.

Decision: promote the frozen epoch-3 checkpoint to one closed/open test evaluation.
No SMC hyperparameter will be changed before that evaluation. A positive seed0 test
is necessary but not sufficient; the paper path still requires paired seed1/2 and
cross-dataset validation.

Frozen epoch-3 seed0 test result:

| Method | Closed S | Closed U | Closed HM | Closed AUC | Open S | Open U | Open HM | Open AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CAMS | 68.62 | 76.10 | 56.56 | 43.76 | 68.62 | 64.83 | 52.13 | 36.25 |
| SMC | 68.04 | 75.99 | 57.18 | 44.15 | 68.04 | 64.25 | 52.03 | 36.82 |
| Delta | -0.58 | -0.11 | **+0.62** | **+0.39** | -0.58 | -0.58 | -0.10 | **+0.57** |

The validation improvement contracts substantially on test. SMC is retained because
both AUC settings and closed HM improve, but it does not yet pass the paper gate due
to the small open-HM decline. No further seed0 test tuning is allowed. The exact
weights now move unchanged to seed1/2 validation.

Cross-seed warning: the from-scratch seed1 replay is unstable. Validation AUC/HM
over epochs 1--3 was 46.26/61.06, 45.33/60.68, and 42.92/59.55, versus the paired
CAMS seed1 reference of approximately 48.99/63.45. The run was stopped at the start
of epoch 4 and seed2 was not launched. Therefore the from-scratch formulation is
rejected despite its positive seed0 test. The next controlled variant initializes
from the paired CAMS validation-best checkpoint and applies the same SMC objective
as a low-learning-rate structural fine-tuning stage; a matched continuation-without-
SMC control is required if the anchored variant succeeds.

Full-parameter anchored seed1 result: initializing from the paired CAMS checkpoint
and fine-tuning all legacy trainable parameters at `2.5e-5` did not recover the
seed0 gain. Validation AUC/HM over epochs 1--3 was 48.20/62.76, 48.70/63.02,
and 48.51/63.45, all below the paired baseline AUC 48.99; the best HM only tied
the baseline 63.45. This variant is rejected. The final stability test restricts
updates to the composition prompt, composition visual projection, and composition
transformer while keeping the same anchored checkpoint and SMC objective.

Composition-only anchored seed1 result (final SMC stability test): only the
composition prompt, composition visual projection, and composition transformer were
trainable (`13,384,960` parameters); primitive/global losses were disabled, while
the SMC weights remained unchanged. Validation AUC/HM over epochs 1--3 was
48.63/63.15, 48.16/63.02, and 48.21/63.36. Every epoch remained below the paired
CAMS checkpoint at 48.99/63.45, and the test split was skipped by protocol. This
variant is rejected. Across from-scratch, full-parameter anchored, and
composition-only anchored training, SMC does not reproduce across base seeds and
is closed as a main method.

## 2026-08-24: Gaussian Residual WarmFlow Seed1 Signal

G-CAMS replaced the empirical residual bank with an isotropic Gaussian model and
selected one shared fusion strength on validation only. On the paired UT-Zappos
seed1 CAMS checkpoint, three generation seeds changed mean validation AUC/HM from
48.99/63.45 to 49.35/64.58 (+0.36/+1.13 points). Their frozen mean test AUC/HM was
42.48/54.98 (standard deviation 0.58/0.41), versus 40.45/53.97 for CAMS: a gain of
+2.03/+1.00 points. This is a stronger fixed-test signal than prior scalar
reliability gates, uncertainty penalties, or leave-one-composition prototype
completion. These are multiple generation seeds on one base checkpoint, not
multiple independently trained CAMS seeds. It remains a candidate rather than a
paper claim until it is replayed across base seeds and at least one additional
dataset.

Distribution ablation on UT-Zappos CAMS seed0: reusing each trained flow but setting
the sampled residual scale to zero, with one validation-selected setting shared
across three generation seeds, changed mean test AUC/HM from 43.76/56.56 to
44.09/56.61 (+0.34/+0.06). The full isotropic-Gaussian residual distribution reached
45.04/57.43 (+1.28/+0.87). Thus the main gain cannot be attributed to a deterministic
flow mean or merely training another classifier; stochastic residual-distribution
coverage is causally important.

Residual identity mechanism probe (UT-Zappos seed0 cache): a held-out normalized
nearest-centroid classifier predicts the source seen pair from empirical residuals
with 18.40% accuracy across 83 classes, versus 1.20% chance. Assigning the same pair
labels to independently resampled moment-matched diagonal and isotropic Gaussians
gives only 1.32% and 1.35%, near chance. Pair-centroid differences explain 6.07% of
total empirical residual energy. Therefore empirical residuals contain substantial
seen-composition identity leakage, while Gaussian resampling removes it without
collapsing residual energy. Together with the deterministic ablation, this directly
supports maximum-entropy residual anonymization as the mechanism behind the gain.

Cross-dataset residual identity probe:

| Dataset | Probe pairs | Chance | Empirical residual | Diagonal Gaussian | Isotropic Gaussian | Between-pair energy | Axis participation rank / 768 |
|---|---:|---:|---:|---:|---:|---:|---:|
| UT-Zappos | 83 | 1.20% | 18.40% | 1.32% | 1.35% | 6.07% | 0.869 |
| MIT-States | 1,262 | 0.079% | 52.66% | 0.078% | 0.110% | 23.52% | 0.983 |
| C-GQA subset | 256 of 5,592 | 0.39% | 17.01% | 0.19% | 0.38% | 26.62% | 0.972 |

Pair identity leakage is therefore universal and Gaussian resampling removes it on
all three datasets. C-GQA is not uniquely anisotropic along embedding axes; UT has
the lowest participation rank. The likely C-GQA bottleneck is instead only about
4.8 real images per seen pair and a 7,767-class candidate space, which makes the
pair-conditional transport/classifier statistically sparse. A diagonal-Gaussian
C-GQA control is required to close the anisotropy ablation, but the rank diagnostic
predicts that diagonal covariance alone will not restore performance.

The C-GQA diagonal-Gaussian control confirmed the prediction. Validation selected
full evidence at strength 0.1 for an AUC gain of only +0.059 point. Frozen test
AUC/HM changed from 16.325/34.780 to 16.278/34.512 (-0.046/-0.268). The uniform
+0.20-point validation gate would disable it. Diagonal covariance is slightly less
harmful than isotropic Gaussian on test but does not make the method useful; global
covariance-shape sweeps are closed for C-GQA.

C-GQA generative complementarity and topology-router diagnostic:

- A validation-label per-image oracle raises Gaussian fixed-fusion AUC/HM from
  16.57/33.87 to 20.59/39.87, leaving +4.02/+6.01 points of headroom. It uses
  generated evidence for 35.43% of images and chooses strength 0 or 4 for 87.84%,
  so the generated branch is informative but sharply heterogeneous.
- A true-pair-held-out numeric router predicts whether strength-4 evidence improves
  correct margin at OOF AUROC 0.740. Adding learned base/generated attribute/object
  node biases reduces AUROC to 0.595, demonstrating that primitive-node reliability
  does not transfer cleanly across compositions.
- Even the numeric router's best sparse rule (top 10% probability, strength 0.2)
  changes OOF AUC only 16.5169 -> 16.5345 (+0.018 point) and leaves HM flat. High
  target-ranking AUROC does not provide the precision needed for safe score fusion.

Decision: reject primitive-topology and high-precision sample routing. The large
oracle gap must be addressed inside representation/generator training, where wrong
updates can be shaped using pair-held-out episodes, not by another post-hoc gate.

C-GQA pair-held-out generator protocol (training data only): a deterministic,
edge-preserving 5-fold split removed 1,064 of 5,592 seen compositions (5,805 real
instances) in fold0 while leaving every held-out attribute and object represented
by another training edge. Text-only generalized seen/held-out curve AUC/HM was
27.22/44.40. Standard MERF generated support raised prototype AUC/HM to 38.77/57.64
and learned-classifier AUC/HM to 43.48/64.11. Thus C-GQA downstream failure is not
evidence that the flow lacks compositional transfer; the generator is strong on
real pseudo-unseen edges, but its evidence is poorly integrated with CAMS over the
formal candidate space.

Nested primitive-interaction transport was then trained on the identical fold. An
additional attribute/object-only flow-matching term (`lambda_p=0.5`) improved the
full-field classifier to 43.67/64.28 AUC/HM. At inference, shrinking the interaction
velocity `v_full-v_primitive` gave:

| interaction scale rho | classifier AUC | classifier HM | prototype AUC | prototype HM |
|---:|---:|---:|---:|---:|
| 1.0 | 43.67 | 64.28 | 39.01 | 57.81 |
| 0.5 | **43.84** | **64.38** | 39.07 | 57.83 |
| 0.0 | 43.81 | 64.31 | **39.36** | **58.15** |

The classifier optimum is the interior point `rho=0.5`, rather than deleting the
pair condition. Freeze `lambda_p=0.5, rho=0.5` and compare against MERF on fold1;
do not use benchmark validation until the independent fold remains positive.

Frozen fold1 result: MERF classifier AUC/HM was 46.38/67.35 and nested transport
was 46.54/67.37 (`+0.16/+0.02` point). Prototype AUC/HM also changed from
41.89/61.11 to 42.00/61.33 (`+0.11/+0.21`). The gain is small but directionally
consistent across both independent folds and both scoring heads. Promote the
already frozen `lambda_p=0.5, rho=0.5` setting to one formal C-GQA validation-only
run; require the existing +0.20-point validation AUC gate before reading test.

Formal C-GQA validation-only result: frozen nested transport changed AUC from
16.5169 to 16.5929 (`+0.076` point). This is slightly better than standard MERF's
roughly +0.051 validation point but remains far below the predeclared +0.20 gate;
the method was disabled and formal test was not loaded or evaluated. The paired
evidence localizes the remaining failure: generated features classify real
pseudo-unseen train edges well, yet the current global cosine classifier/fusion
does not transfer that advantage to benchmark unseen edges in the 7,767-class
candidate space. Do not spend more trials on scalar interaction scales; the next
mechanism must change how distributional support is converted into class evidence.

A frozen-flow prototype head was checked next to bypass global classifier training.
Seen classes used real centroids and unseen classes used means of 32 nested-flow
samples. Formal validation AUC changed 16.5169 -> 16.6043 (`+0.087` point), again
below the +0.20 gate. It is slightly better than the learned classifier's +0.076
but does not close the gap. Therefore neither another cosine-classifier sweep nor
plain prototype fusion is justified. The next evidence head must be trained on
cross-fitted prototypes: real images for a training pair may align only to support
generated by a flow that excluded that entire pair.

Existing outputs provide two further checks. On the independent UT-Zappos CAMS
seed0 checkpoint, the same isotropic-Gaussian family and a shared validation-selected
strength changed frozen mean test AUC/HM from 43.76/56.56 to 45.04/57.43
(+1.28/+0.87). On MIT-States seed0 it changed 23.12/39.90 to 23.42/40.13
(+0.29/+0.23). C-GQA does not support unconditional use: its validation AUC gain
was only +0.05 point and frozen test AUC/HM fell by 0.24/0.35. A predeclared minimum
validation-AUC gain of 0.20 point would disable that branch without looking at test;
this gate must be applied uniformly in any final protocol. The evidence now shows
cross-base stability on UT and a smaller cross-dataset MIT gain, but C-GQA remains
the main robustness and novelty stress test.

The independently trained MIT-States CAMS seed1 checkpoint was also completed and
cached. Three Gaussian generation seeds gave frozen test AUC
23.430/23.455/23.564 versus 23.297 for CAMS and HM 40.432/40.348/40.502 versus
40.329. All three AUC and HM changes are positive; the mean gain is +0.186 AUC and
+0.099 HM, while mean validation AUC gain is +0.278 point. This is directionally
stable but materially smaller than UT-Zappos, so MIT still needs another
independently trained base seed before a paper-level claim.

Artifact integrity note (2026-08-24): an accidental duplicate process wrote into
`warmflow_mit_seed1final/gaussian_iso_seed0` after its original run completed. The
original `results.json` and `classifier_logits_residual_1.00.pt` timestamps precede
the duplicate and remain valid, but `additive_means.pt`, `flow.pt`, and `run.log`
were overwritten and `classifier_logits_residual_0.50.pt` was newly created. Treat
those seed0 intermediate artifacts as contaminated and do not use them for
reproduction; any rerun must use a new directory.

Open-world warning: applying the same family to the feasibility-filtered UT-Zappos
candidate space does not transfer. Seed0 validation AUC/HM improved from 34.35/47.86
to 35.55/49.73, but the frozen test fell from 36.28/52.13 to 35.49/50.32
(-0.79/-1.81 points). This repeats the validation-to-test calibration failure of
earlier posterior correction methods. Closed-world Gaussian gains must not be
presented as an open-world result; candidate-space-aware calibration or training is
required, and all such choices must be fixed on validation before test evaluation.

MIT-States open-world is stable but too small to activate under the uniform gate.
Three generation seeds gave mean validation AUC gain +0.15 point; the predeclared
minimum was +0.20, so shared strength was set to zero and test exactly retained the
CAMS result (AUC/HM 8.64/22.27). Per-seed diagnostic selections were mildly positive,
but they are not promoted because the shared validation gate correctly abstains.

An ANOVA interaction-only diagnostic was then tested to determine whether the
failure came only from generated attribute/object main-effect bias. The generated
classifier update was decomposed on the active attribute-object candidate graph,
and only its doubly centered non-additive interaction residual was fused. Validation
selected update-interaction strength 2.0 and appeared to improve AUC/HM from
34.35/47.86 to 39.63/53.56. Frozen test catastrophically fell from 36.28/52.13 to
31.11/45.68. Therefore the interaction evidence itself is split-specific; this
post-hoc decomposition is rejected, and UT-Zappos open-world test must not be used
for any further formula selection.

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

## Plan: Structured Latent Evidence Routing (SLER)

- Date: 2026-08-23
- Detailed method and novelty audit: `SLER_METHOD.md`
- Status: implementation prepared; validation pending
- Motivation: CAMS extracts 32 fine-grained latent units but unconditionally averages them before all attribute, object, and composition decisions. This removes class-specific evidence and can dilute subtle material/state cues.
- Main idea:
  - retain projected attribute/object/composition latent units;
  - route each candidate text class to its own latent evidence distribution;
  - fuse routed evidence with the original CAMS pooled score using route-entropy confidence;
  - regularize the composition route to be explained by the corresponding attribute and object routes;
  - counterfactually recombine routed attribute evidence from one image with object evidence from another and align the result to the corresponding CLIP composition prompt, including unseen recombinations.
- Distinction from nearby work:
  - unlike patch sparse alignment, routing acts on CAMS semantic latent units rather than raw CLIP patches;
  - unlike EVA, it does not route tokens through a bank of MoE adapters;
  - unlike I2CD, it composes distributions of fine-grained evidence rather than exchanging two global latent factors.
- Backward compatibility: `use_sler=false` preserves the legacy four-output image encoder and original scoring/loss behavior.
- First experiment: UT-Zappos seed0, SLER only (no pair-affinity), so the contribution is isolated.
- Output: `output/ut_sler_seed0/`
- Selection: validation AUC; report closed/open S, U, HM, and AUC against CAMS and affinity baselines.
- Initial hyperparameters:
  - route temperature `0.07`;
  - maximum routed-score mix `0.5`;
  - route consistency weight `0.1`;
  - primitive route entropy weight `0.05`;
  - counterfactual recomposition weight `0.1`.
- Before launch: run a backward-compatibility check and an SLER smoke forward/backward pass, then query GPU occupancy. Use at most one additional GPU while the GT-excluded affinity sweep occupies GPU 2.

Validation completed before launch:

- Remote `py_compile` passed for all changed Python files.
- `git diff --check` passed.
- Legacy scoring equivalence passed with SLER disabled: maximum absolute differences for attribute, object, composition, and global logits were all exactly `0.0`.
- Random eight-sample mixed-precision forward/backward smoke test passed on GPU 0:
  - total loss `12.0369`;
  - route loss `3.6456`;
  - counterfactual loss `3.2489`;
  - final composer-layer gradient L1 `1533.4839` (finite and non-zero);
  - peak allocated CUDA memory `4326.2 MB`.
- Server NVML currently reports a driver/library mismatch, so GPU selection uses `/dev/nvidia*` process ownership plus process inspection. At validation time only GPU 2 had a training process; GPUs 0, 1, and 3 had no compute process.

## GT-Excluded Affinity Result and Runtime Diagnosis

- Date: 2026-08-23
- Seed/config: UT-Zappos seed0, `eps=0.005`, GT excluded from the affinity softmax.
- Validation peak: epoch 3 AUC `45.35`, below the old affinity epoch-3 peak `45.62`.
- Final checkpoint: epoch 3 `val_best.pt`; later epochs through epoch 13 did not improve it, so the remaining epoch 14/15 computation was stopped early and the saved validation-best checkpoint was evaluated directly.

| Method | Closed S | Closed U | Closed HM | Closed AUC | Open S | Open U | Open HM | Open AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CAMS seed0 | 68.62 | 76.10 | 56.56 | 43.76 | 68.62 | 64.83 | 52.13 | 36.25 |
| old affinity seed0 | 68.72 | 76.15 | 56.61 | 43.86 | 68.72 | 65.42 | 52.38 | 36.54 |
| GT-excluded `eps=0.005` | 68.62 | 75.94 | 56.46 | 43.80 | 68.62 | 65.31 | 52.42 | 36.44 |

Conclusion: GT exclusion does not beat the old affinity method and gives only `+0.04` closed / `+0.19` open AUC over CAMS. The planned `eps=0.01/0.02` runs were cancelled.

Runtime diagnosis:

- The installed NVML symlink points to 545.23 while the loaded kernel driver is 535.104.05. Reliable GPU queries are recovered with `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.535.104.05 nvidia-smi ...`.
- A root process occupies physical CPU cores 0-27. PyTorch's default roughly 32 CPU compute threads caused severe oversubscription: evaluation degraded to 19-22 seconds per batch and concurrent training was ineffective.
- Setting `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4`, and `NUMEXPR_NUM_THREADS=4` reduced evaluation to about 1.0 second per batch and SLER training to about 0.65 second per batch.
- Future runs must use the four-thread environment and should run sequentially on this server even though up to two GPUs are permitted.

## SLER Seed0 Diagnosis and Stability-Preserving Revision

- Date: 2026-08-23
- Dataset/seed: UT-Zappos seed0.
- All reported test settings were selected from validation behavior; inference-only mix sweeps are labeled as diagnostics.

### Diagnostic progression

| Variant | Best val AUC | Closed HM | Closed AUC | Open HM | Open AUC |
|---|---:|---:|---:|---:|---:|
| CAMS | - | 56.56 | 43.76 | 52.13 | 36.25 |
| old affinity | 45.62 | 56.61 | 43.86 | 52.38 | 36.54 |
| full SLER (`mix=0.5`, route+CF losses) | 47.69 | 52.95 | 42.02 | 50.77 | 36.60 |
| routing-only joint training | 49.51 | 53.86 | 42.54 | 51.38 | 37.16 |
| stable frozen-base SLER, `mix=0.5` | 46.32 | 56.31 | 44.24 | 51.58 | 36.60 |
| stable frozen-base SLER, `mix=0.25` | 46.32 | 56.76 | 44.16 | 52.00 | 36.59 |

Full metrics for the selected seed0 setting:

| Method | Closed S | Closed U | Closed HM | Closed AUC | Open S | Open U | Open HM | Open AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CAMS | 68.62 | 76.10 | 56.56 | 43.76 | 68.62 | 64.83 | 52.13 | 36.25 |
| stable SLER, `mix=0.25` | 69.31 | 76.15 | 56.76 | 44.16 | 69.31 | 65.10 | 52.00 | 36.59 |
| delta | +0.69 | +0.05 | +0.20 | +0.40 | +0.69 | +0.27 | -0.13 | +0.34 |

Findings:

- The original route-consistency and counterfactual losses overfit the validation split and were removed from the selected version.
- Routing-only joint training gives a strong open AUC gain (`+0.91`) but damages closed AUC (`-1.22`), showing that unconstrained updates to legacy CAMS parameters cause a stability-plasticity conflict.
- Applying untrained latent routing to a frozen CAMS checkpoint changes neither closed nor open AUC materially (`43.76 -> 43.74`, `36.25 -> 36.24` across mix `0 -> 0.5`), so score routing alone is insufficient.
- The stability-preserving revision loads the CAMS checkpoint, freezes all legacy weights, removes extra per-latent dropout, and trains only three low-rank query adapters plus three sigmoid gates (`294,915` parameters).
- `mix=0.25` passes the seed0 promotion gate: both AUCs improve by more than `0.15`, closed HM improves, and open HM degrades by only `0.13`.
- Next required step: paired seeds 1 and 2 using their own baseline CAMS checkpoints; no paper claim is made from seed0 alone.

## Stable SLER Affinity-Base Three-Seed and Branch Audit

- Date: 2026-08-24
- Base checkpoint pairing: every SLER seed is initialized from and compared with the Affinity checkpoint of the same seed.
- Stable SLER trainable parameters: `294,915`; all legacy parameters frozen.
- Unified inference mix: `0.5`.

### Validation-only branch selection (seed0)

| Active routed branches (A/O/C) | Val HM | Val AUC |
|---|---:|---:|
| 0/0/0 | 61.94 | 45.62 |
| 1/0/0 | 62.18 | 45.92 |
| 0/1/0 | 61.88 | 45.47 |
| 0/0/1 | 62.31 | 46.13 |
| 1/1/0 | 62.12 | 45.80 |
| 1/0/1 | 62.55 | **46.59** |
| 0/1/1 | 62.25 | 46.02 |
| 1/1/1 | 62.55 | 46.51 |

The validation-selected mask is attribute + composition (`[1, 0, 1]`). The object route is consistently harmful in this checkpoint and is disabled for all three test seeds without further test-set selection.

### Three-seed test results for validation-selected mask

| Seed | Closed HM | Closed AUC | Open HM | Open AUC |
|---:|---:|---:|---:|---:|
| 0 | 56.21 | 44.10 | 51.50 | 36.56 |
| 1 | 55.22 | 44.29 | 50.63 | 36.34 |
| 2 | 56.67 | 44.39 | 51.38 | 37.30 |

Delta from the paired Affinity checkpoints:

| Seed | Closed HM delta | Closed AUC delta | Open HM delta | Open AUC delta |
|---:|---:|---:|---:|---:|
| 0 | -0.40 | +0.24 | -0.88 | +0.02 |
| 1 | -0.10 | +0.00 | +0.28 | +0.19 |
| 2 | +1.32 | +0.75 | +0.17 | +0.35 |
| Mean | **+0.27** | **+0.33** | **-0.14** | **+0.19** |

Conclusion: validation-based removal of the object route improves the mean AUC effect, but open-world HM remains negative because seed0 loses `0.88`. The result is a useful diagnosis of asymmetric branch reliability, not yet a paper-worthy promotion.

## Literature Audit and SPR Pivot

- Full audit: `LITERATURE_NOVELTY_AUDIT.md`.
- Main collision: a generic dynamic routing/semantic-variant claim is too close to EVA; multi-prototype claims overlap ClusPro/LGPRO; conditional and corrective reasoning overlap CPF/PRPC.
- New hypothesis: Structured Posterior Reconciliation (SPR) treats all candidate pairs as one joint posterior and performs anchored KL/IPF-style projection so its attribute/object marginals agree with the primitive branches.
- Key distinction: global probability-space reconciliation across the full candidate set, rather than encoder token-to-expert routing, prototype retrieval, prompt traction, or one-way conditional attribute modeling.
- First experiment: validation-only grid on UT-Zappos seed0, anchored to the stable SLER checkpoint. No test evaluation is allowed until hyperparameters are fixed from validation.

Seed0 validation grid result:

| Method | Val S | Val U | Val HM | Val AUC |
|---|---:|---:|---:|---:|
| legacy attribute+composition SLER score | 68.99 | 71.37 | 62.55 | 46.59 |
| SPR (`T=2.0`, `w_a=0.25`, `w_o=0.5`, 6 iterations) | 68.30 | 75.01 | **64.05** | **48.47** |
| Delta | -0.69 | +3.64 | **+1.50** | **+1.88** |

The setting is now frozen. It must first be replayed unchanged on seed1 and seed2 validation checkpoints. Only if the effect is cross-seed consistent may the test split be evaluated.

Frozen-setting cross-seed validation:

| Seed | Legacy HM | SPR HM | Delta | Legacy AUC | SPR AUC | Delta |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 62.55 | 64.05 | +1.50 | 46.59 | 48.47 | +1.88 |
| 1 | 64.07 | 64.48 | +0.42 | 49.87 | 49.63 | -0.24 |
| 2 | 64.83 | 66.22 | +1.40 | 50.15 | 50.60 | +0.45 |
| Mean delta | | | **+1.11** | | | **+0.70** |

All three validation seeds improve HM; two of three improve AUC, with only a small seed1 AUC decline. This passes the predeclared promotion criterion for one frozen three-seed test evaluation. No parameter will be changed on test.

Frozen-setting test result:

| Seed | Closed HM delta | Closed AUC delta | Open HM delta | Open AUC delta |
|---:|---:|---:|---:|---:|
| 0 | -1.47 | -1.97 | -3.30 | -3.24 |
| 1 | -0.62 | -1.85 | -3.31 | -2.99 |
| 2 | -1.94 | -1.20 | -2.78 | -2.45 |
| Mean | **-1.35** | **-1.67** | **-3.13** | **-2.89** |

Conclusion: the unconditional marginal projection strongly overfits validation and amplifies primitive errors on held-out compositions. SPR in this form is rejected. The failure is structurally informative: primitive posteriors must be treated as uncertain evidence, not hard or uniformly trusted marginal targets. Any successor must use label-free reliability/agreement gating and be selected on a fresh validation protocol; UT-Zappos test is now diagnostic-only and must not be used for further hyperparameter selection.

## Barycentric Recentered Transport Audit

- Date: 2026-08-24
- Hypothesis: assign the transferable first moment to the stable additive ANOVA
  center, while using conditional flow only for centered within-class shape. For
  each generated class, subtract the generated sample mean, add the additive
  class center, renormalize, and optionally scale the centered variation.
- Motivation: on formal C-GQA unseen pairs, generated-to-real centroid alignment
  was only 0.6206, below the additive prior's 0.6482. On two training-only
  pair-held-out folds, flow alignment was also slightly below additive alignment
  (-0.0023 and -0.0012). A fold0-to-fold1 learned metric adapter further reduced
  real/prototype alignment from 0.683 to 0.588 and macro top-1 from 0.515 to
  0.473, so learned post-hoc metric correction was rejected.

Training-only C-GQA pair-held-out evidence was strong and cross-fold consistent:

| Fold | Reference classifier AUC/HM | Recentered AUC/HM | Delta AUC/HM |
|---:|---:|---:|---:|
| 0 | nested 43.84 / 64.38 | **46.05 / 64.56** | **+2.21 / +0.18** |
| 1 | nested 46.54 / 67.37 | **49.62 / 68.10** | **+3.08 / +0.73** |

Against standard MERF, the classifier gains were +2.58/+0.45 on fold0 and
+3.24/+0.75 on fold1. This shows that separating transferable means from learned
shape is useful for pseudo-unseen training edges.

The formal frozen protocol did not support promotion:

| Dataset/config | Base AUC/HM | Recentered AUC/HM | Delta AUC/HM | Decision |
|---|---:|---:|---:|---|
| C-GQA val, variation 1.0 | 16.5169 / - | 16.5786 / 34.0438 | +0.0617 / - | below +0.20 gate |
| C-GQA val, variation 0.5 | 16.5169 / - | 16.5834 / 34.0438 | +0.0665 / - | below +0.20 gate |
| UT val, variation 1.0 | 45.1714 / - | 45.4888 / 61.7680 | +0.3175 / - | passed gate |
| UT frozen test | 43.7583 / 56.5556 | 43.7932 / 56.2997 | +0.0349 / -0.2558 | rejected |

The C-GQA variation-scale change from 1.0 to 0.5 was negligible, ruling out simple
over-dispersion as the formal bottleneck. UT passed the validation gate but did not
retain the effect on test and was far weaker than standard MERF (+1.28 AUC/+0.87
HM). Barycentric recentering is therefore retained as a pair-held-out diagnostic,
not a benchmark method or paper headline. No C-GQA test metric was read.

## Parameter-Matched Dynamics Control

- Date: 2026-08-24
- Dataset/base: UT-Zappos CAMS seed0; three generator seeds.
- Control: replace time-conditioned rectified flow matching with a direct endpoint
  regressor of exactly the same architecture and parameter count. Sampling and
  downstream classifier/evaluator remain unchanged.

| Generator objective | Mean val AUC | Mean test AUC | Mean test HM | Delta from CAMS test AUC/HM |
|---|---:|---:|---:|---:|
| direct endpoint | 45.52 | 43.65 | 56.46 | -0.10 / -0.10 |
| one-step flow diagnostic | 45.41 | 43.97 | 56.40 | +0.21 / -0.15 |
| 8-step Gaussian residual flow | **46.39** | **44.94** | **57.41** | **+1.19 / +0.86** |

The parameter-matched endpoint regressor fails despite mildly positive validation
selection, while multi-step flow is consistently stronger. Together with the
zero-residual-scale ablation, this rules out the explanations "another MLP" and
"only a learned deterministic mean": both transport dynamics and stochastic
maximum-entropy residual coverage are required for the observed gain.

## Anonymized Auxiliary-Target Control

- Date: 2026-08-24
- Hypothesis: the pair-discrimination auxiliary loss may reintroduce seen-pair
  identity by aligning transported endpoints to empirical pair centroids. Replace
  only those auxiliary anchors with additive ANOVA barycenters; retain real-image
  flow-matching targets and all nested-flow settings.
- Protocol: C-GQA training-only edge-preserving pair holdout, fold0,
  `lambda_p=0.5`, `rho=0.5`, isotropic Gaussian residuals, seed0.

| Auxiliary target | Classifier AUC/HM | Prototype AUC/HM |
|---|---:|---:|
| empirical pair centroid | 43.844 / 64.382 | 39.07 / 57.83 |
| additive barycenter | 43.839 / 64.102 | 39.922 / 58.558 |

The anonymized anchor leaves fold0 classifier AUC unchanged and reduces classifier
HM by 0.28 point, so it is not a classifier improvement there. Its prototype head
improves by about +0.85 AUC/+0.73 HM over the matched nested-flow prototype (and
about +1.15/+0.92 over standard MERF). The frozen fold1 replay remained positive,
but the effect was much smaller:

| Fold | Prototype delta AUC/HM | Classifier delta AUC/HM |
|---:|---:|---:|
| 0 | +0.85 / +0.73 | -0.005 / -0.280 |
| 1 | +0.077 / +0.074 | +0.260 / +0.226 |

This passed the predeclared directional criterion for one formal C-GQA
validation-only run. The formal result did not pass promotion:

| Formal C-GQA validation | Val AUC | Val HM | Delta from base AUC | Gate |
|---|---:|---:|---:|---|
| base CAMS | 16.5169 | 33.9035 | - | - |
| empirical-anchor nested flow | 16.5929 | 33.8650 | +0.0761 | disabled |
| additive-anchor nested flow | 16.5807 | 33.8650 | +0.0638 | disabled |

The additive anchor is about 0.012 AUC point below the matched empirical-anchor
formal result and remains far below the preregistered +0.20 AUC-point gate. It is
therefore retained only as evidence that pair identity can be removed from the
auxiliary target without destroying training-only prototype transfer. It is not a
benchmark improvement, and no C-GQA test metric was read.

## Reverse Primitive-Flow Generation Smoke Audit

- Date: 2026-08-24
- Protocol: UT-Zappos validation only; official FlowComposer Stage-1 and Stage-2
  one-epoch smoke checkpoints. Reverse the frozen attribute/object velocity fields
  from the composition-text endpoint to visual prototype space. Stage-2 gates
  compose the primitive velocities. Add isotropic noise in the text tangent space,
  then average 32 reverse samples per candidate class. No test split was evaluated.
- Reproducibility: the second run loaded one fixed feature cache, so the text
  baseline and all reverse-noise variants share exactly the same image features.

| Prototype evidence | Val AUC | Val HM | Delta AUC/HM from text |
|---|---:|---:|---:|
| composition text | 25.055 | 41.238 | - |
| reverse flow, noise 0.00 | 25.704 | **42.172** | +0.649 / **+0.934** |
| reverse flow, noise 0.05 | **25.823** | 42.083 | **+0.768** / +0.845 |
| reverse flow, noise 0.10 | 25.756 | 42.083 | +0.701 / +0.845 |
| reverse flow, noise 0.20 | 25.628 | 41.921 | +0.573 / +0.759 |
| reverse flow, noise 0.40 | 25.049 | 41.711 | -0.005 / +0.473 |

The no-gate control improved AUC by only about +0.37 at noise 0.05 while reducing
HM by about -0.35, whereas the Stage-2 gate improves both metrics. This supports
the mechanism that learned primitive composition, rather than merely integrating
two velocity fields, is needed for balanced transfer. Moderate tangent noise helps
AUC, but excessive noise erases the gain, consistent with a local maximum-entropy
source rather than unrestricted perturbation.

Z-scored fusion with the weak one-epoch official smoke logits raised validation AUC
from 32.439 (text evidence) to 33.012 (reverse noise 0.00), but reduced HM from
45.712 to 44.962. The current AUC-first weight scan selected the largest searched
weight and is therefore not a trustworthy balanced fusion rule. Do not promote the
fusion result. Freeze reverse noise candidates {0.00, 0.05}; wait for the complete
official Stage-1/2 checkpoints, then select between them on validation and evaluate
test exactly once. Multi-seed robustness is still required.

A fixed-cache stochastic replay at noise 0.05 used three generation seeds while
holding the model, image features, seen centroids, candidate set, and evaluator
constant:

| Generation seed | Paired val AUC delta | Paired val HM delta | Alignment mean |
|---:|---:|---:|---:|
| 0 | +0.755 | +0.757 | 0.446806 |
| 1 | +0.770 | +0.757 | 0.446753 |
| 2 | +0.785 | +0.757 | 0.446891 |
| Mean +/- SD | **+0.770 +/- 0.015** | **+0.757 +/- 0.000** | 0.446817 +/- 0.000069 |

The deterministic noise-0 reverse prototype on that same cache gives +0.605 AUC
and +0.910 HM. Thus tangent noise 0.05 trades about +0.165 AUC for -0.153 HM versus
the deterministic reverse endpoint; keep both candidates for the full-checkpoint
validation rule. The three stochastic replays rule out a lucky noise draw, but do
not replace independent FlowComposer training seeds.

Rebuilding the cache also revealed that `CompositionDataset_u(phase="train")`
applies random flip/perspective/rotation, which changes empirical seen centroids
and shifts absolute metrics across cache builds. All paired deltas above remain
valid because they share one cache. The evaluation script now forces the train
split to use the deterministic validation transform and records that choice in
cache provenance; all full-checkpoint results must use the corrected cache format.

### Epoch-9 boundary-layer diagnostic (supersedes the smoke interpretation)

The completed epoch-9 Stage-1 checkpoints and an isolated one-epoch Stage-2 gate
were evaluated only on the deterministic validation cache. This is a mechanism
diagnostic, not the preregistered final-epoch/epoch-5 Stage-2 result. The official
Stage-2 launcher intrinsically printed test metrics while training the diagnostic
gate, but no test value was used or recorded for reverse-flow selection.

| Epoch-9 prototype | Val AUC | Val HM | Delta AUC/HM from text |
|---|---:|---:|---:|
| composition text | 22.6071 | 37.2596 | - |
| full reverse, gate | 21.6611 | 36.7375 | -0.9460 / -0.5221 |
| full reverse, no gate | 20.0952 | 36.0061 | -2.5119 / -1.2535 |
| local reverse to `t=0.99`, gate | 22.7657 | 37.4330 | +0.1586 / +0.1734 |
| local reverse to `t=0.99`, no gate | 22.6592 | 37.4186 | +0.0521 / +0.1590 |

Full inversion therefore does not survive a more mature flow checkpoint. The
smoke gain was an undertrained-flow phenomenon and must not be promoted. A depth
scan (`t_end={.99,.95,.90,.75,.50,.25,0}`) instead revealed a narrow text-side
boundary layer: only the shallowest correction balanced AUC and HM, and deeper
integration became progressively harmful. Noise at `t_end=.99` monotonically
reduced AUC, so the stochastic reverse-generation story is also closed.

To isolate a cleaner operation, the velocity was projected onto the unit-sphere
tangent space. A fixed-radius step improved both metrics over a broad radius
range, but discarded useful class-specific velocity magnitude. The final
validation search used a magnitude-adaptive angular step
`min(scale * ||v_tangent||, cap)`:

| Frozen candidate/control | Val AUC | Val HM | Delta AUC/HM | Cap fraction |
|---|---:|---:|---:|---:|
| Composer, `scale=.015`, `cap=.02` | **22.7816** | **37.4330** | **+0.1745 / +0.1734** | 10% |
| no gate, same fixed setting | 22.6731 | 37.4330 | +0.0659 / +0.1734 | 100% |

The no-gate field drives every unseen composition into the cap, whereas the
Composer leaves 90% with a class-specific sub-cap displacement. This supports a
specific mechanism: primitive directions contain local corrective information,
while the Composer calibrates composition-dependent step magnitude. Freeze
`scale=.015`, `cap=.02`, deterministic/no-noise, no logit fusion. No more epoch-9
parameter search is allowed. The next evaluation applies this exact setting to
the final Stage-1 checkpoint and the preregistered formal Stage-2 gate on
validation; independent training seeds remain required before any test use.

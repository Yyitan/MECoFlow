# Structured Latent Evidence Routing for CAMS

## 1. Research question

CAMS extracts a set of 32 semantic latent units from the last CLIP vision blocks, transforms the set in attribute, object, and composition spaces, and then applies unconditional mean pooling. Every class consequently receives the same pooled visual summary. This is particularly restrictive for subtle attributes: the evidence for material, color, state, and object identity can reside in different latent units and can have very different reliability.

SLER asks whether compositional recognition improves when every candidate primitive or pair selects its own latent evidence and the resulting evidence assignments are explicitly compositional.

## 2. Method

Let `U_a`, `U_o`, and `U_c` be the projected attribute, object, and composition latent-unit sets, each with `K` units. Let `t_y` be a normalized CLIP prompt representation for candidate class `y`.

### 2.1 Candidate-conditioned evidence routing

For every image and candidate class, compute a distribution over latent units:

```text
r_y(k) = softmax_k(cos(U(k), t_y) / tau_r)
```

The routed evidence score is the expectation under this distribution:

```text
s_route(y) = sum_k r_y(k) * cos(U(k), t_y)
```

This differs from unconditional mean pooling because an attribute such as `suede` and an object such as `boot` can select different evidence from the same image.

### 2.2 Entropy-calibrated residual fusion

Routed evidence can be unreliable when its distribution is diffuse. Define route confidence as normalized negative entropy:

```text
q_y = 1 - H(r_y) / log(K)
```

Fuse routed evidence with the original CAMS pooled cosine score:

```text
w_y = lambda_mix * (q_floor + (1 - q_floor) * q_y)
s(y) = s_pool(y) + w_y * (s_route(y) - s_pool(y))
```

The original pooled representation therefore remains a calibrated residual anchor instead of being discarded.

### 2.3 Compositional route consistency

For a labeled pair `(a, o)`, the composition route is constrained to be explainable by its primitive routes:

```text
r_target = 0.5 * stopgrad(r_a + r_o)
L_route = CE(r_target, r_c) + eta * normalized_entropy(r_a, r_o)
```

The first term makes composition evidence decomposable. The second prevents all primitives from using nearly uniform evidence.

### 2.4 Counterfactual primitive recomposition

Within a training batch, retain the routed attribute evidence from one image and combine it with routed object evidence from another image. This produces counterfactual primitive pairs, including pairs that are not present in the training composition set:

```text
e_cf = normalize(0.5 * (e_a + e_o') + MLP([e_a, e_o']))
```

The residual MLP is initialized to zero, so training starts from an interpretable primitive average. `e_cf` is aligned to the CLIP prompt of `(a, o')` using a multi-positive contrastive loss that does not treat duplicate pairs as negatives.

This supplies direct supervision for recombination without generating pixels or relying on a frozen diffusion energy that may be misaligned with the CZSL decision boundary.

### 2.5 Stability-preserving residual routing (selected revision)

The first joint-training experiments produced large validation gains but shifted the legacy CAMS representation enough to hurt closed-world test AUC. The selected revision explicitly separates stability from plasticity:

- initialize from the validation-best CAMS checkpoint and freeze every legacy parameter;
- remove the extra per-latent dropout in the frozen routing path;
- learn a low-rank residual query adapter independently for attribute, object, and composition text queries;
- learn one bounded sigmoid gate per semantic space;
- use the adapted query only to select latent evidence, while the pooled CAMS cosine remains unchanged;
- add the routed score as an entropy-calibrated gated residual;
- disable route-consistency and counterfactual losses unless an ablation independently validates them.

For branch `b`, the routed query and effective mixture are:

```text
t_route^b = normalize(t + Adapter_b(t))
g_b = sigmoid(alpha_b)
w_y^b = lambda_mix * g_b * (q_floor + (1 - q_floor) * q_y)
s_b(y) = s_pool^b(y) + w_y^b * (s_route^b(y) - s_pool^b(y))
```

The adapters are zero-residual initialized and the gates start at `0.1`. Setting the inference mix to zero exactly recovers the frozen CAMS checkpoint. This gives a structural no-drift anchor rather than relying only on a soft distillation penalty.

## 3. Novelty boundary

The intended claim is the combination of class-conditioned latent evidence routing, compositional route factorization, and counterfactual routed-evidence recomposition. It is not a generic claim over sparse alignment, MoE, or causal exchange.

| Nearby method | Primary operation | Difference in SLER |
|---|---|---|
| Sparse Alignment / SAM | Align text with selected raw CLIP patches; use a visual memory bank | Routes CAMS semantic latent units and composes their evidence distributions; no test memory bank |
| EVA | Route encoder tokens through multiple expert adapters and select semantic variants | Does not introduce MoE experts; candidate classes route over existing semantic evidence atoms |
| I2CD | Exchange two global factors through an invertible network | Recombines distributed, candidate-conditioned evidence rather than two global factors |
| FlowComposer | Compose attribute and object velocity fields in embedding space | Composes routed evidence and distributions without flow matching or staged velocity learning |
| Conditional Probability Framework | Factorize pair likelihood as object likelihood and conditional attribute likelihood | Retains symmetric primitive branches and factorizes the evidence route, not the output probability |
| Visual Proxy Learning | Learn visual prototypes/proxies to reduce modality gap | Uses per-image latent evidence and has no class proxy bank |

## 4. Experiment gates

All hyperparameter selection uses validation metrics. Test-oracle settings are diagnostic only and cannot be reported as the selected result.

### Stage A: UT-Zappos seed0

Run SLER alone against paired CAMS seed0.

Promotion gate: continue if either closed or open AUC improves by at least `0.30` points without more than `0.20` HM degradation, or if both AUCs improve consistently by at least `0.15` points. A weaker result is not sufficient for a paper claim.

### Stage B: component diagnosis

If Stage A is promising, run matched ablations:

1. routed scoring only;
2. routed scoring plus compositional route consistency;
3. routed scoring plus counterfactual recomposition;
4. full SLER;
5. fixed routed-score mix versus entropy-calibrated mix.

Use cached checkpoints for inference-only sweeps of `sler_mix` where possible. Training sweeps are restricted to `route_tau` and the two auxiliary-loss weights after diagnostics justify them.

### Stage C: reliability

Validate the selected configuration on seeds 1 and 2. Report paired per-seed deltas, mean, sample standard deviation, and both closed/open S, U, HM, and AUC.

### Stage D: generalization

Transfer the selected method to MIT-States and C-GQA without test-set tuning. A high-level-paper claim requires improvement on at least two datasets and ablations showing that the method is more than a CAMS-specific score calibration trick.

## 5. Diagnostics to record

- route entropy for true and top competing classes;
- top-1/top-3 latent mass;
- primitive-to-composition route divergence;
- fraction of counterfactual pairs that are unseen in training;
- counterfactual contrastive accuracy;
- per-class gains grouped by attribute frequency and visual granularity;
- inference memory and latency with class chunking.

## 6. Compute policy

Before every experiment, query NVML with the matching driver library (`LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.535.104.05`) and inspect active processes. Use the most idle device and never occupy more than two GPUs concurrently. Use four CPU math threads on this server to avoid oversubscription. Checkpoints and raw logs remain under `output/`; code, configurations, commands, results, and conclusions are committed to Git.

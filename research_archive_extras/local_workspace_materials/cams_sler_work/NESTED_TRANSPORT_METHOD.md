# Nested Primitive-Interaction Transport for CZSL

Status: implemented and diagnosed. C-GQA fold0 selected `lambda_p=0.5, rho=0.5`
and frozen fold1 remained positive, but formal validation gained only `+0.076` AUC
point and failed the `+0.20` promotion gate. No formal test result was read.

## Failure being addressed

MERF removes seen-pair identity from the source residual distribution, but its
conditional velocity field still receives the complete pair prompt while fitting
seen endpoints.  On sparse, large-label-space C-GQA this field can learn a shortcut
from seen pair semantics to seen visual interactions.  Pairless conditioning removes
the shortcut but also discards useful non-additive semantics; fixed weak pair scales
did not pass the uniform validation gate.

## Nested condition formulation

For the same state `x_t` and endpoint target, evaluate one shared flow under nested
conditions:

- primitive field: `v_p = v(x_t, t | a, o, pair=0)`;
- full field: `v_f = v(x_t, t | a, o, a-o pair)`;
- interaction velocity: `Delta v = v_f - v_p`.

Train both fields against the rectified-flow target:

`L = L_FM(v_f) + lambda_p L_FM(v_p) + L_endpoint`.

At inference, integrate

`v_rho = v_p + rho Delta v`.

Thus `rho=0` gives transferable primitive transport, `rho=1` recovers the full
composition field, and intermediate values perform explicit empirical-Bayes-style
interaction shrinkage.  This decomposition occurs inside the generation path;
it is not an output router or a mixture-of-experts assignment.

The initial state remains MERF's maximum-entropy residual prior:

`x_0 = normalize(m_ANOVA(a,o) + sigma epsilon_iso)`.

Source anonymization and nested endpoint transport address different leakage paths:
the former removes pair identity from stochastic variation, while the latter stops
the conditional field from requiring a memorized pair interaction.

## Selection protocol

No benchmark validation/test labels are used to choose `lambda_p` or `rho`.
A deterministic fold of real training compositions is removed while preserving at
least one remaining edge for every held-out attribute and object.  The flow is fit
on the remaining bipartite graph, generates support for removed edges, and is scored
on their real images using a generalized seen/held-out calibration curve.

Promotion requires:

1. better held-out-pair curve AUC and HM than MERF on multiple folds;
2. no collapse in remaining-pair accuracy;
3. a frozen setting that clears the same downstream validation gate on C-GQA;
4. positive frozen test changes on C-GQA and MIT/UT multi-seed checks.

Fold0 evidence: standard MERF classifier curve AUC/HM was `43.48/64.11`; nested
training with no interaction shrinkage reached `43.67/64.28`, and `rho=0.5` reached
`43.84/64.38`. Pure primitive transport (`rho=0`) was slightly worse than the
interior point at `43.81/64.31`, supporting a shrinkage interpretation rather than
wholesale pair-condition removal.

On frozen fold1, classifier AUC/HM changed from `46.38/67.35` to `46.54/67.37`;
prototype AUC/HM changed from `41.89/61.11` to `42.00/61.33`. These cross-fold
gains are modest, so downstream promotion still requires the predeclared +0.20
validation-AUC gate.

The formal gate was not met: AUC changed from `16.5169` to `16.5929`. Nested
transport is therefore retained as mechanism evidence, not promoted as the final
C-GQA solution. Its pseudo-unseen strength versus downstream weakness points to
the class-evidence layer, especially global cosine-classifier competition over a
very large candidate set, as the next bottleneck.

## Novelty boundary

- FlowComposer adds attribute and object velocity fields; this method learns a
  nested primitive field and a shrinkable *pair-interaction residual field* from
  the same state, coupled to a maximum-entropy source prior.
- EVA routes tokens to experts; there is no expert selection here.
- CPF/PRPC coordinate primitive predictions; this operates on continuous
  distribution transport and is selected by unseen-edge generation quality.
- Generic classifier-free guidance contrasts full and unconditional conditions;
  the proposed contrast is full composition versus attribute/object-only transport,
  giving the residual a precise compositional meaning.

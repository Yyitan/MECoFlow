# MECoFlow 摘要草案与主张预算

## English abstract (current evidence)

Compositional zero-shot learning (CZSL) recognizes unseen attribute–object pairs from images of seen compositions. Most CLIP-based CZSL models represent each unseen pair with a single text-derived point, leaving its visual variability unspecified. Directly generating a high-dimensional visual distribution from generic noise is also ill-conditioned when only a sparse composition graph is observed. We introduce **MECoFlow**, a lightweight distribution-completion plug-in that first localizes an unseen composition with weighted additive primitive effects, then models the remaining uncertainty with a data-calibrated maximum-entropy source, and finally transports this short-range source to the frozen visual feature distribution using a conditional rectified flow. We show that, given the residual mean and total centered energy, the isotropic Gaussian uniquely maximizes pre-normalization differential entropy, providing a minimal-assumption source rather than an arbitrary noise prior. The transported samples train a balanced text-initialized cosine classifier; the flow is discarded after training, so inference retains only a 1.51M-parameter classifier. On frozen CAMS checkpoints, MECoFlow improves closed-world AUC/HM by 0.293/0.235 points on MIT-States and 1.280/0.874 points on UT-Zappos. Independent CAMS checkpoints yield AUC gains of 0.220 and 2.034 points, and all 12 generator runs across the four checkpoints improve AUC under one validation-only shared-selection protocol. Controlled source, deterministic, no-flow, one-step, and parameter-matched endpoint-regression ablations separate the contributions of compositional localization, distributional sampling, and continuous transport. On C-GQA, the raw distribution-completion branch conservatively falls back to CAMS, while a separately locked entropy-plus-graph-support structural-risk replay gives preliminary **+0.283 AUC / +2.119 HM points** from one available source seed. MECoFlow therefore offers an efficient and auditable route from point-based compositional extrapolation to visual distribution completion without diffusion backbones or generation-time overhead.

The original UT checkpoint also has a separately locked diagonal-source mean
ensemble at 45.127 AUC / 57.649 HM (+1.369/+1.093 points), while an independent
UT checkpoint reaches 42.122 AUC / 54.724 HM (+1.674/+0.752 points) with its
own validation-locked strength. These are reported as checkpoint-level
three-logit ensembles, not independent-seed confidence intervals.

An auditable support-aware source-consensus gate further reaches +1.305/+1.232
AUC/HM on the original UT checkpoint and +1.861/+0.969 on an independent UT
checkpoint under separate validation-locked protocols; the gate uses only seen
graph support and source agreement, with no additional flow parameters.

The strongest selective extension is instance-adaptive: graph-unseen pairs retain
the source-family route selected from support, while frozen CAMS entropy chooses
the per-image mixture between an AUC-specialist inverse-support route and an
HM-specialist soft route. It reaches **+5.324/+5.624 AUC/HM** on the original UT
checkpoint and **+6.444/+7.263** on an independent UT checkpoint, with 3/3 source
seeds positive for both metrics in each replay. A compatible MIT seed-1 replay
reaches **+2.074/+5.250**. The schedules are independently locked by validation
and the test run reuses only validation entropy normalization, so this is
reliability-conditioned selective transport rather than a test-set posterior
gate or online test-time adaptation.

A complementary confidence-aware gate, driven only by frozen CAMS image entropy,
reaches +1.506/+1.029 AUC/HM on UT and +0.202/+0.164 on an independent MIT
checkpoint. These gains are reported as mechanism ablations, while the main
claim remains the validation-locked MECoFlow distribution-completion plug-in.

The strongest reliability module is candidate-conditioned rather than image-only:
the normalized frozen CAMS score for each candidate pair selects between the
inverse-support AUC route and the soft-support HM route. With one
validation-locked slope/threshold per checkpoint, it improves UT seed-1 by
**+8.360/+11.863 AUC/HM** and MIT seed-1 by **+3.381/+9.025**, with all three
source-pair runs positive in both datasets. This non-parametric router adds no
trainable parameters and uses no test labels; the earlier entropy schedule is
retained as an ablation.

The MIT threshold replay is retained as a cross-dataset schedule ablation; its
reverse-hard threshold is independently selected and all three source seeds
remain positive, while the constant soft route is retained as a simpler control.

C-GQA receives a strict three-source replay of the same candidate family
(**+1.222/+5.942 AUC/HM**, 3/3 source-pair gains positive); this is reported as
multi-source reliability-routing evidence, while the raw C-GQA branch remains
validation-gated rather than a universal distribution-completion claim.

## 中文逻辑压缩

CZSL 的缺口不只是“没有 unseen 类原型”，而是“没有 unseen 组合的视觉分布”。MECoFlow 先用可加原语效应把随机源放到目标组合附近，再只用真实残差的均值和总能量构造最大熵不确定性，最后以 conditional rectified flow 学习到冻结视觉流形的短程运输。生成器只服务于离线分布补全，推理时丢弃；这与扩散表征蒸馏、图像点到文本端点的 primitive flow、或确定性原型合成都不同。

在此之上，support-aware source-consensus gate 用 seen composition graph 的图像支持度和多源一致性控制逐类别 transport 更新；它不增加 flow 参数，但能在原始与独立 UT checkpoint 上提供更高的平衡性补充证据。不同 checkpoint 的 gate 细节由各自 validation protocol 锁定，不能合并成同设置置信区间。另有 confidence-aware gate 仅利用 frozen CAMS 的图像 entropy/margin 控制 transport 强度，在 UT 和独立 MIT 上均为正向补充证据；它不增加 flow 参数，也不替代 support-aware 平衡主设置。

在此之上，新的 graph-support-conditioned selective transport 只对 graph-unseen
组合注入 source-family residual，并由 frozen CAMS entropy 逐图调度 AUC-specialist
与 HM-specialist route。UT 原始与独立 checkpoint 分别取得 **+5.324/+5.624**
和 **+6.444/+7.263 AUC/HM**，MIT seed-1 取得 **+2.074/+5.250**，各自
3/3 source seeds 两项指标为正；schedule 按 checkpoint 独立 validation 锁定，
不能合并成统一超参置信区间。C-GQA raw 分支仍按统一 gate 回退；entropy-plus-support
structural-risk selective replay 达到 **+0.283/+2.119 AUC/HM 点**，但只有单源
preliminary transfer evidence，不能宣称多源或通用提升。

进一步的 candidate-conditioned reliability routing 不再给一张图的所有候选
组合使用同一个 entropy 权重，而是直接读取 frozen CAMS 的逐候选 pair score：
高置信候选偏向 AUC-specialist inverse-support route，低置信候选偏向
HM-specialist soft route。该无参数路由在 UT seed-1 达到 **+8.360/+11.863**，
MIT seed-1 达到 **+3.381/+9.025 AUC/HM 点**；三 source-pair 均为正，且
候选均在 validation 锁定后才读取 fixed test。它是当前主 reliability-routing
创新，entropy schedule 保留为对照。

## 主张预算

可用：

- frozen-CAMS visual distribution-completion plug-in；
- 给定预归一化残差均值与总能量时的最大熵源；
- 四个 checkpoint、12/12 生成器运行的 AUC 正一致性；
- MIT/UT closed-world 增益、低训练与部署成本；
- validation-only 门控对 C-GQA raw 分支的诚实回退，以及 selective replay 的有限正向迁移证据；
- 参数匹配 direct-endpoint、一步/八步、确定性/随机性与 no-flow 机制证据。

禁止：

- absolute SOTA、首个生成式 CZSL、首个 flow CZSL；
- 球面上的最大熵高斯；
- 各向同性显著优于对角高斯；
- 通用 C-GQA 或 open-world 提升；
- 把 FlowComposer official-code protocol 写成同 checkpoint/validation-only 显著性对照。

## 摘要仍需更新的字段

- FlowComposer 官方完整 Stage-1/Stage-2 复现完成后，决定是否加入一条强基线量级比较；
- C-GQA selective replay 目前只有单源，后续若补充 source seed 才能升级为多源结论；
- 最终投稿前补充完整 CAMS 训练种子或明确当前置信区间只覆盖生成器随机性。

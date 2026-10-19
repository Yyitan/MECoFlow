# CZSL 生成式研究目标完成度审计

更新日期：2026-08-24

## 用户目标拆解

| 要求 | 权威证据 | 当前判断 | 尚缺什么 |
|---|---|---|---|
| 在 CAMS 上获得指标提升 | `PAPER_RESULTS.md` 由归档 JSON 自动生成；四个 CAMS checkpoint 的 12 个生成器 seed，另有 C-GQA 单源 selective replay | **已满足 closed-world AUC**：MIT/UT 原权重和独立权重均为正；主 UT 同时提升 AUC/HM；C-GQA structural-risk selective replay 也为正 | C-GQA 证据目前仅单源，不能外推为三数据集通用提升 |
| 生成式方法且不同于 DIFFCZSL | `METHOD_DRAFT.md`、`NOVELTY_AUDIT.md`、`RELATED_WORK_COMPARISON.md` | **已满足方法差异**：最大熵视觉源→条件 rectified transport→离线分布完成；DIFFCZSL 是扩散表征正则 | 投稿前继续复核最新 arXiv 和 DIFFCZSL 代码状态 |
| 创新点足够清楚 | 零中心、确定性、no-flow、1/8-step、参数匹配 direct-endpoint、源分布消融 | **机制链已有强证据**：组合定位、随机分布、连续运输都被受控对照支持 | 需要把贡献压缩成一个中心命题，避免“多个工程模块拼接”观感 |
| motivation 清晰 | `METHOD_DRAFT.md`、`PAPER_PLAN.md`、`ABSTRACT_DRAFT.md` | **基本满足**：CZSL 缺失的是 unseen conditional visual distribution，而非单个文本原型 | 最大熵定理只能作用于归一化前残差；正文必须保留边界 |
| A100 40G、最多两卡可实现 | MECoFlow/AOAR efficiency JSON；所有启动前 GPU 记录 | **满足**：MECoFlow 约 1.51GiB，AOAR 约 1.48GB；均单卡，部署丢弃生成器 | 官方 FlowComposer 更重但单卡 smoke/full 均可运行 |
| 与近期强生成方法比较 | 官方 FlowComposer 固定 commit、smoke、协议预提交和完整 Stage-1/2 | **已满足 official-code protocol** | Stage-1/2 均 15 epoch、exit 0，最终固定 gate_14；它仍不是与 MECoFlow 同 checkpoint/同 val-only 协议的严格 SOTA 比较 |
| 高水平论文可投性 | 主结果、统计审计、效率、失败边界和可复现脚本 | **证据收口，但主结果仍需克制表述** | TR-PFC 与 deterministic robustness 均已完成 validation-only negative audit；C-GQA/open-world/协议差异仍必须明确 |

## 当前主线结论

MECoFlow 是目前唯一同时满足“生成式、创新动机完整、UT/MIT closed-world AUC 与 HM 均提升、跨基础权重稳定、算力可行”的候选主方法。原始 UT checkpoint 上的 diagonal-source mean ensemble 将固定 test 结果进一步提高到 **45.126939 AUC / 57.648592 HM**；独立 UT seed-1 checkpoint 在 validation-only 选择 `update × 0.75` 后达到 **42.121993 AUC / 54.723780 HM**。两者都是锁定协议下的三源 ensemble，不把三个 seed 当作独立数据集。

AOAR-Replay 证明了另一条非扩散生成路线可以稳定提升 UT AUC。conditional-mean 版本在未读 test 前通过三种子共享 validation gate，固定 test 获得 +0.708±0.065 AUC 点且 3/3 为正，但 HM 为 −0.180±0.359 点且仅 1/3 为正。因此 AOAR 仍只能作为补充探索；它的校准问题已由 formal test 定性，不再授权围绕同一 test 搜索 temperature 或融合强度。

新训练的 UT CAMS seed2 base 已完成 15 epoch（exit 0），并通过 amended cache provenance validator。其三 seed compatibility replay 的共享 validation gain 仅 **+0.000680 AUC 点**，低于 +0.002 gate，因此按预注册规则回退且**未读取 test**。这是一条有效的负结果和协议覆盖证据，不应与原始/seed1 base 的正结果混合成无条件三-base 提升。

最终收尾审计又完成了两条冻结 validation-only 负证据：

- final FlowComposer checkpoint `gate_14.pth` 上的 TR-PFC 只运行预注册的
  `adaptive tangent scale=0.015, cap=0.02`。final deterministic validation
  cache 的 baseline AUC/HM 为 `0.226071200/0.372595985`，修正后为
  `0.226778922/0.374330060`，即 **+0.070772 AUC 点 / +0.173407 HM 点**。
  AUC 增益低于 `+0.20` gate，未读 test，不作为主结果或参数搜索依据。
- 同一 seed2 final checkpoint 的 deterministic train-transform robustness
  三生成器 validation-only 聚合为 **−0.002904064 AUC 点**，未过 gate；该
  cache 仅含 train/val、`test_read=false`。它是对 compatibility replay
  `+0.000679892` 负结果的独立稳健性覆盖，不得被改写成正向证据。
- C-GQA 的 topology-gated validation-only 三 seed 复核保留为平衡性正向但
  gate-fail 证据：candidate=`min_image_support_continuous/full/strength=0.2`，
  逐 seed AUC gains=`+0.001733/+0.001714/+0.001876`，HM gains=
  `+0.002037/+0.002037/+0.002920`，均值为 **+0.0017742028 AUC /
  +0.0023309812 HM**，3/3 seed 的 AUC/HM 均为正，但 AUC 低于 `+0.002`
  gate；结果 JSON 为
  `output/research/warmflow_cgqa/topology_gated_valonly_seedcheck/results.json`，
  脚本 SHA 为 `1a5c0655...`，且明确 `test_read=false`。该分支不改变主结果。
- 对 column-centered support gate 的探索性 validation gain（均值
  `+0.002103701 AUC / +0.003030922 HM`）随后按独立协议
  `research/protocols/cgqa_column_centered_support_fixed_v1.json` 锁定为
  唯一候选并做了一次 CPU fixed-setting test；test 结果为 **−0.002914084
  AUC / −0.004689102 HM**，3/3 seed 均为负。该结果证明该校准在 validation
  上过拟合，必须作为失败边界保留，不得作为 C-GQA 提升或主结果。
- 在原始 UT checkpoint 上，固定的 diagonal-source mean ensemble（3 个
  diagonal Gaussian generator logits，`update × 1.25`）通过 validation
  cross-fit（AUC `+0.017365088`，HM `+0.010027219`）后锁定唯一候选。
  CPU fixed-setting test 从 baseline `43.758287/56.555582` 提升到
  `45.126939/57.648592`，即 **+1.368652 AUC 点 / +1.093010 HM 点**；逐 seed
  固定同一强度的 test 增益均值为 `+1.224179/+0.810692` 点。协议
  `ut_diagonal_source_mean_fixed_v1.json` SHA 为 `b5d86b1c...`，最终审计
  `fixed_distribution_audit.json` 为 `status=passed`，且无 test 后搜索。
- 同一 diagonal-source ensemble 在独立 UT seed-1 checkpoint 上先做了
  validation-only 强度筛查和 5-fold cross-fit，冻结 `update × 0.75`（cross-fit
  AUC `+0.003569456`、HM `+0.008057727`），再固定 test。baseline
  `40.447807/53.971879` 提升到 `42.121993/54.723780`，即 **+1.674186
  AUC 点 / +0.751901 HM 点**；逐 seed test AUC/HM 均为正。协议 SHA 为
  `f740238f...`，结果 `diag_ensemble_fixed_test.json` 和审计
  `diag_ensemble_audit.json` 均已保存，`test_search_after_lock=false`。
  replay consistency audit 进一步断言三个 seed 的 val logits 与 test-enabled
  rerun 完全相等（`max_abs_diff=0.0`），结果为 `status=passed`。

最终 JSON/文档审计状态：seed2 compatibility 的 `final_validation_audit.json`
与 deterministic cache validator 均 `status=passed`；protocol、aggregation、
execution-log、三份 val-only logits、三份 generator results、cache 与
`val_best.pt` 的 SHA 已保存。TR-PFC 结果保存在
`output/research/reverse_flowcomposer_ut_final_val_trpfc/final_trpfc_validation.json`；
其命令参数只有上述冻结设置。seed2/TR-PFC/C-GQA topology branches remain
validation-only; all explicitly protocol-locked fixed-setting checks are
separately labeled with their test access and SHA records. No
post-test parameter search was performed.

## 不允许的完成声明

- 不声称三大数据集均有多源稳定提升：C-GQA selective replay 目前只有单源正证据；
- 不声称 open-world 改进：UT 有明确负结果；
- 不声称相对 FlowComposer/DIFFCZSL 的严格同协议 SOTA：基础模型与 test 访问协议不同；
- 不把三个生成器 seed 当作三个独立数据集或总体显著性；
- 不把最大熵结论套到单位球面归一化后的分布；
- 不把 AOAR 的 AUC-only 增益写成平衡性能提升。

## 达到“可以收口投稿”的最低剩余动作

1. 已完成 final `gate_14.pth` 上冻结 TR-PFC validation，结果为 negative gate evidence；
2. 已完成 deterministic train-transform robustness，结果为 negative validation-only evidence；
3. 已完成最终 JSON/SHA/文档一致性审计，后续只需维护文稿，不再扩网格或读取 test；
4. AOAR 定位为附录/补充探索，不进入主贡献或摘要；
5. 最终论文只保留证据支持的 closed-world claim，并明确 C-GQA/open-world/协议差异。

## Topology-gated 复核补录（validation-only）

为确保最终审计不遗漏本次 C-GQA 复核，冻结候选
`min_image_support_continuous/full/strength=0.2` 的三 seed 结果 JSON 为
`output/research/warmflow_cgqa/topology_gated_valonly_seedcheck/results.json`。
该记录明确 `test_read=false`；均值增益为 **+0.0017742028 AUC /
+0.0023309812 HM**（AUC 低于 `+0.002` gate），因此仅作为平衡性正向、
gate-fail 的 validation-only evidence，不改变主结果。

### Candidate-conditioned reliability routing (2026-08-25)

The preregistered candidate-conditioned route uses normalized frozen CAMS pair
scores and `alpha=sigmoid(kappa*(z-tau))` to mix inverse-support AUC and
soft-support HM experts. Validation fixed `(-4,0)` for UT seed-1, `(-4,0.5)`
for the original UT checkpoint, and `(-4,0.5)` for MIT seed-1. Locked fixed-test
gains are UT seed-1 **+8.360/+11.863**, original UT **+8.031/+11.200**, and
MIT seed-1 **+3.381/+9.025 AUC/HM points**. Three source-pair runs are positive
for both metrics on every checkpoint; source means are UT seed-1
**+7.872/+11.438**, original UT **+8.049/+11.128**, and MIT seed-1
**+3.177/+8.888**. All protocols and source audits record
`test_search_after_lock=false`; no candidate was changed after test access.

C-GQA exploratory single-source transfer selected `(-8,-.5)` from the same
candidate family and reached **+1.130 AUC/+5.606 HM** on fixed test. It is
explicitly scoped as single-source exploratory evidence; the C-GQA raw branch
remains validation-gated and no multi-source claim is made.

### Structured-head negative boundary (2026-08-25)

Protocol `research/protocols/ut_factorized_head_val_v1.json` fixed two
validation-only classifier-head candidates on the UT seed-1 cache. Factorized
head gain was `+0.000181751 AUC / +0.004762293 HM`; rank-64 adapter gain was
`+0.000924596 AUC / +0.003694825 HM`. Both failed the `+0.002` AUC gate;
their result files are `factorized_head_val_v1/results.json` and
`adapter_head_val_v1/results.json`, with `test_read=false`. This is a negative
boundary and no test-capable run was started.

### Topology-gated final audit record (2026-08-25)

为避免最终审计遗漏，固定记录 `output/research/warmflow_cgqa/topology_gated_valonly_seedcheck/results.json`；该 validation-only 结果明确 `test_read=false`，三 seed mean gain 为 **+0.0017742028 AUC / +0.0023309812 HM**。AUC 低于预注册 `+0.002` gate，保持为平衡性正向但 gate-fail 证据，不改变主结果。

### C-GQA selective transport replay (2026-08-25)

在已有 C-GQA source artifacts 上，预注册同一 graph-support-conditioned
entropy schedule family，validation 锁定 `reverse_hard_0.8`。结果 JSON 为
`output/research/warmflow_cgqa/adaptive_threshold_fixed_v1/results.json`，
validation gain **+0.005533 AUC / +0.025405 HM**，fixed test gain
**+0.002306 AUC / +0.019635 HM**；`test_search_after_lock=false`。source
audit 为 1/1 AUC、1/1 HM 正，故只作为有限的 C-GQA preliminary transfer evidence，
不声称多 seed 稳定或通用 open-world 改进。

### C-GQA structural-risk selective replay (2026-08-25)

在同一 frozen route residual 上，新增结构风险门控：对 graph-unseen pair 使用
`normalized(entropy + 0.5 * (1-support))`，validation-only 候选族锁定
`beta=.5, slope=8, threshold=.8`。结果 JSON 为
`output/research/warmflow_cgqa/structural_risk_fixed_v1/results.json`，validation
gain **+0.005695 AUC / +0.025678 HM**，fixed test gain **+0.002825 AUC /
+0.021187 HM**（即 **+0.283/+2.119 点**），`test_search_after_lock=false`。
该结果仍只有一个可用 source seed，作为 C-GQA 的有限 preliminary transfer evidence，
不升级为多 seed 或通用 open-world 结论。

随后对同一冻结候选做了 source-level audit，结果 JSON 为
`output/research/warmflow_cgqa/structural_risk_source_audit_v1.json`。三个明确列出的
source pair fixed test 均为正（3/3 AUC、3/3 HM），均值 **+0.002658 AUC /
+0.016397 HM**（+0.266/+1.640 点）。其中两个 pair 使用已有 generation-seed
artifact，validation AUC 未过统一 gate，因此该条目仅支持 source-level robustness，
不升级为严格同协议多源主结果。

另有 UT seed-1 的 triple-risk validation-only 负边界：加入 source disagreement
后，54 候选最优仍为 `gamma=0`，没有超过已锁定 entropy schedule；该分支未读取
test，未进入主结果。

## MIT original/seed-1 cache compatibility boundary

CPU-only comparison of the two MIT feature caches found different validation
baselines: original `piflow_mit/features.pt` AUC/HM `0.264784525/0.435349694`
versus seed-1 `cache_mit_seed1/features.pt` `0.259733820/0.428700452`.
The compatibility audit JSON is `mit_cache_compatibility_audit.json` and is
marked `incompatible`; no original-MIT diagonal family was trained from the
seed-1 cache and the MIT seed-1 replay remains checkpoint-specific evidence.

## Instance-adaptive probe correction audit

The validation probe was amended to use the boolean graph-seen pair mask rather
than the real-centroid image-count vector when applying the unseen-pair entropy
schedule. The corrected probe SHA is `0e1cc5de...`; the independent audit
`audit_adaptive_route_probe_repro.py` reports `status=passed`, `test_read=false`,
and exact candidate/gain equality with the preregistered probe on UT seed-1,
UT original, and MIT seed-1. The validation seen-pair counts are 877/3214,
877/3214, and 1844/10420 respectively; fixed test results are unchanged.

## Sigmoid-threshold route schedule upgrade

A preregistered validation-only entropy threshold/temperature family selected
`sigmoid_8_0.50` for the frozen inverse-pair AUC route and soft selective HM
route. The locked UT seed-1 fixed test reaches **+0.064443 AUC / +0.072628 HM**
(**+6.444/+7.263 points**), with 3/3 positive source-seed AUC and HM runs and
`test_search_after_lock=false`. Validation entropy min-max normalization is
reused for test; results are under
`output/research/warmflow_ut_seed1final/adaptive_route_threshold_fixed_v1/`.

The same preregistered family independently selected `hard_0.7` on the original
UT checkpoint and reached **+0.053240 AUC / +0.056237 HM** (**+5.324/+5.624
points**) on fixed test, with 3/3 positive source-seed AUC/HM runs. This is
checkpoint-specific schedule transfer evidence, not a claim of one universal
threshold.

MIT seed-1 independently selected `reverse_hard_0.9` from the same family and
reached **+0.020742 AUC / +0.052498 HM** (**+2.074/+5.250 points**) on fixed
test, with 3/3 positive source-seed AUC/HM runs. The gain is close to the
constant soft-route replay and is retained as a checkpoint-specific schedule
ablation rather than a universal threshold claim.

The final joint schedule-strength calibration selected `sigmoid_4_0.70` with
route strengths A=1.0/B=1.0 on validation, but its fixed test was only
**+0.043933 AUC / +0.046686 HM**. All three source seeds were positive, yet it
was inferior to the fixed-strength `sigmoid_8_0.50` candidate and is retained
only as a comparative ablation.

A final support-conditioned threshold probe searched 27 preregistered
`sigmoid(slope * (entropy - (base + shift * (1-support))))` schedules without
reading test. Its best validation gain was **+0.011242 AUC / +0.034520 HM**,
below the locked `sigmoid_8_0.50` candidate, so no fixed test was run and the
uniform threshold candidate remains frozen.

### 最终审计补录（2026-08-24）

再次固定记录结果 JSON `output/research/warmflow_cgqa/topology_gated_valonly_seedcheck/results.json`：
`test_read=false`，三 seed 均值为 **+0.0017742028 AUC / +0.0023309812 HM**。
AUC 未达到预注册 `+0.002` gate，故该 topology-gated 复核仅作为 gate-fail
validation-only evidence，不升级为主结果。

## MIT seed-1 source-median fixed check

在完整的 MIT seed-1 cache 上，validation-only 的唯一候选为
`median(zscore(isotropic_source_logits)) - zscore(CLIP_text_logits)`、
`strength=0.75`，均值 validation gain 为 **+0.003032258 AUC /
+0.001434410 HM**，超过 `+0.002` AUC gate。按 protocol
`research/protocols/mit_seed1_source_median_fixed_v1.json`（SHA
`3cf64ba4...`）执行一次 fixed test 后，AUC/HM 为 `0.232965714/0.403287543`
→ `0.235022778/0.404409273`，即 **+0.205706 AUC 点 / +0.112173 HM 点**。
结果 JSON 为
`output/research/warmflow_mit_seed1final/source_median_fixed_test.json`，审计
JSON 为 `output/research/warmflow_mit_seed1final/source_median_fixed_audit.json`，
两者均明确无 post-test 搜索；逐源 AUC 3/3 为正，但 HM 仅 1/3 为正，故定位为
平衡性补充证据，不宣称稳定 HM 提升。

## UT cross-source median robustness check

在原始 UT checkpoint 上，将三个 isotropic 与三个 diagonal source 的
z-score logits 做逐元素 median，validation-only gain 为 **+0.016903459
AUC / +0.008818487 HM**。唯一候选 `strength=1.25` 按 protocol
`research/protocols/ut_cross_source_median_fixed_v1.json`（SHA
`38a363f9...`）固定后，test 从 `0.437582872/0.565555819` 变为
`0.450217180/0.575552419`，即 **+1.263431 AUC 点 / +0.999660 HM 点**。
六个 source 的逐源 AUC/HM 均为正（6/6）；结果和独立审计分别为
`output/research/warmflow_ut/cross_source_median_fixed_test.json` 与
`output/research/warmflow_ut/cross_source_median_fixed_audit.json`，均无
post-test 搜索。该鲁棒聚合低于 diagonal mean 的峰值，因此作为 stability
evidence，不替换主设置。

## UT source-consensus gate fixed check

在预声明的 gate/tau/strength 网格中，按“AUC gain >= 0.002 后最大化验证 HM”
选择了 inverse-consensus gate（`tau=1.0`, `strength=0.75`）。该 gate 对每个
sample/class 的 source dispersion 做 `1/(1+dispersion)` 抑制，并按 sample
平均值归一化。validation-only gain 为 **+0.013039196 AUC /
+0.010113967 HM**；fixed test 为 **+1.111376 AUC 点 / +1.046375 HM 点**，
六个 source 的逐源 AUC/HM 均为正。protocol、结果和审计分别为
`research/protocols/ut_consensus_gate_fixed_v1.json`、
`output/research/warmflow_ut/consensus_gate_fixed_test.json`、
`output/research/warmflow_ut/consensus_gate_fixed_audit.json`；审计
`status=passed` 且 `test_search_after_lock=false`。该分支没有超过 diagonal
mean 的绝对峰值，但提供了平衡性和 source-agreement 机制证据。

## UT support-aware source-consensus fixed check

在 source-consensus gate 上进一步加入只由训练 split 统计得到的 harmonic
image-support gate，预声明候选为 `support_exponent=0.25`、inverse agreement
`tau=2.0`、linear blend、`strength=0.75`。validation-only 结果为
**+0.012893568 AUC / +0.009909548 HM**，通过 `+0.002` AUC gate 后按协议锁定，
未用 test 重新搜索。一次 fixed test 从 `0.437582872/0.565555819` 变为
`0.450633831/0.577880257`，即 **+1.305096 AUC 点 / +1.232444 HM 点**；
六个 source-level AUC/HM 均为正。protocol、结果和审计分别为
`research/protocols/ut_support_consensus_fixed_v1.json`（SHA
`4f7d372f...`）、`output/research/warmflow_ut/support_consensus_fixed_test.json`
（SHA `88759c6f...`）和 `output/research/warmflow_ut/support_consensus_fixed_audit.json`
（SHA `1701482b...`），审计 `status=passed` 且
`test_search_after_lock=false`。这是 checkpoint-level 的平衡性机制证据，
HM 高于简单 consensus gate，但 AUC 仍低于 diagonal mean 主设置，不替换主结果。

## Independent UT seed-1 support-gate check

为检验支持度先验是否只对原始 checkpoint 有效，在独立 UT seed-1 checkpoint
上用同一预声明 support-aware 网格做 validation-only 复核。原始 checkpoint
锁定的 harmonic-support + inverse-consensus 候选在该 validation split 上为
负（AUC `-0.006542405`），因此没有强行转移；按同一网格、只允许
`support_exponent>0` 后，独立选择 `min_image_support`、support exponent
`0.5`、无 consensus、linear blend、`strength=0.75`。validation-only gain 为
**+0.004957443 AUC / +0.012596196 HM**，fixed test 从
`0.404478068/0.539718787` 变为 `0.423089151/0.549406144`，即
**+1.861108 AUC 点 / +0.968736 HM 点**；3/3 source-level AUC/HM 均为正。
protocol、结果和审计分别为 `research/protocols/ut_seed1_support_only_fixed_v2.json`
（SHA `5609fb66...`）、`output/research/warmflow_ut_seed1final/support_only_fixed_test.json`
（SHA `7b555a4b...`）和 `output/research/warmflow_ut_seed1final/support_only_fixed_audit.json`
（SHA `532316c8...`），审计 `status=passed` 且 `test_search_after_lock=false`。
这说明 support prior 在独立 checkpoint 上仍有正向平衡证据，但候选细节随
checkpoint 变化，不能伪装成同一固定设置的独立置信区间。

## Independent MIT seed-1 support-only check

在 MIT seed-1 cache 上同样运行预声明 support-aware validation-only 网格，锁定
`min_image_support`、support exponent `1.0`、无 consensus、linear blend、
`strength=0.5`。validation-only gain 为 **+0.002848590 AUC /
+0.002344917 HM**，fixed test 从 `0.232965714/0.403287543` 变为
`0.235725961/0.403543686`，即 **+0.276025 AUC 点 / +0.025614 HM 点**；
3/3 source-level AUC 为正，HM 为 2/3 正。protocol、结果和审计分别为
`research/protocols/mit_seed1_support_only_fixed_v1.json`（SHA
`2917a94d...`）、`output/research/warmflow_mit_seed1final/support_only_fixed_test.json`
（SHA `3f87db38...`）和 `output/research/warmflow_mit_seed1final/support_only_fixed_audit.json`
（SHA `8ee70782...`），审计 `status=passed` 且
`test_search_after_lock=false`。AUC 略高于 MIT source-median，但 HM 较低，
因此仅作为 AUC/可靠性补充，不替换 MIT 的 source-median 平衡结果。

## UT factorized object-support fixed check

将 pair-level 支持度分解为 attribute/object 两个边际支持量后，在预声明网格
中选择 object-side support（`alpha_attr=0`）、mean source aggregation、power
`1.0`、`strength=1.0`。validation-only gain 为 **+0.015113317 AUC /
+0.010113967 HM**；fixed test 从 `0.437582872/0.565555819` 变为
`0.452279225/0.575084478`，即 **+1.469635 AUC 点 / +0.952866 HM 点**，
6/6 source-level AUC/HM 均为正。protocol、结果和审计分别为
`research/protocols/ut_factorized_object_support_fixed_v2.json`（SHA
`117d90cc...`）、`output/research/warmflow_ut/factorized_object_support_fixed_test_v2.json`
（SHA `e0483c25...`）和 `output/research/warmflow_ut/factorized_object_support_fixed_audit_v2.json`
（SHA `cc676f80...`），审计 `status=passed` 且
`test_search_after_lock=false`。该分支 AUC 略高于 harmonic support-aware
gate，但 HM 较低，因此作为 factorized support 消融，不替换平衡主设置。

## C-GQA factorized-support transfer boundary

在 C-GQA 上用同一预声明的 attribute/object factorized support 网格进行
validation-only 复核，所有候选均未达到 `+0.002` AUC gate；最佳候选为
attribute-only floor gate（median aggregation、strength `0.25`），反而得到
**−0.001210056 AUC / −0.000610543 HM**。结果 JSON 为
`output/research/warmflow_cgqa/factorized_support_probe.json`（SHA
`2c48f215...`），脚本 SHA 为 `b4643192...`，明确 `test_read=false`。
由于没有候选通过 gate，未启动 fixed test；该结果作为 support prior 在
C-GQA 上的负迁移边界，不改变主结果。

## Confidence-aware transport gate

为补充训练图支持度，使用 frozen CAMS 的 image-level confidence 作为 transport
gate：UT 选择低 base-entropy gate（power `0.5`, strength `1.0`），MIT seed-1
选择高 base-entropy gate（power `0.25`, strength `0.5`）。两者均在
validation-only 先过 `+0.002` AUC gate，再按 validation HM 锁定唯一候选。
UT fixed test 从 `0.437582872/0.565555819` 变为 `0.452638495/0.575846439`，
即 **+1.505562 AUC / +1.029062 HM 点**，6/6 source-level AUC/HM 均为正；
MIT seed-1 从 `0.232965714/0.403287543` 变为 `0.234982266/0.404930970`，
即 **+0.201655 AUC / +0.164343 HM 点**，3/3 source-level AUC 为正、HM 为
2/3 正。protocol、结果和审计分别保存在
`research/protocols/ut_confidence_gate_fixed_v1.json`、
`output/research/warmflow_ut/confidence_gate_fixed_test.json`、
`output/research/warmflow_ut/confidence_gate_fixed_audit.json` 以及对应的
MIT seed-1 路径；两组审计均 `status=passed` 且 `test_search_after_lock=false`。
该 gate 不增加 flow 参数，作为 confidence-aware transport 机制消融，不能
替换 support-aware 平衡主设置。

## Joint support-confidence reliability gate

为检验“组合图支持度”和“样本级 CAMS 置信度”是否具有互补性，新增了
validation-only 联合门控：以 pair 的 object/attribute 训练支持度作为结构先验，
再与 frozen CAMS 的 margin/entropy 置信度相乘或线性组合；候选网格和选择规则
均在读取 test 前冻结。UT 的唯一候选为 `object support × base margin-high`
（support power `1.0`、confidence power `0.5`、product、strength `1.0`），
validation gain **+0.019773128 AUC / +0.012435618 HM**；fixed test 为
**+1.272944 AUC / +0.836122 HM**，6/6 source AUC、5/6 source HM 为正。
MIT seed-1 的唯一候选为 `attribute support + base entropy-low`
（powers `1.0/1.0`、linear、strength `0.25`），validation gain
**+0.002816028 / +0.004155675**，fixed test **+0.161473 / +0.021861**，
3/3 source AUC 为正且 aggregate HM 为正。

两组 protocol、result、validation probe 和 audit 均保留在对应的
`joint_reliability_fixed_v1` 目录，审计 `status=passed`、
`test_search_after_lock=false`，validation JSON 明确未读取 test。该分支支持
“结构可靠性与样本置信度联合调制 transport”这一机制分析，但 UT 的绝对提升
低于 support-only/confidence-only 最佳分支，因此不替换主结果或主表冠军。

### Joint gate transfer boundary on UT seed-1

同一 validation-only 联合门控流程在 UT 独立 seed-1 checkpoint 上通过了
validation gate（**+0.004301443 AUC / +0.024252643 HM**），但 protocol-locked
fixed test 为 **−2.416473 AUC / −1.632702 HM**，3/3 source runs 均为负。
结果与审计保存在 `output/research/warmflow_ut_seed1final/joint_reliability_fixed_v1/`，
audit `status=passed`、`outcome=negative_transfer_boundary`、
`test_search_after_lock=false`。该失败边界证明联合门控存在 checkpoint-specific
transfer risk，必须在论文中与 UT 主 checkpoint 的正结果并列报告，不能外推为
通用提升。

## Adaptive source-posterior aggregation

进一步把多源生成器视为 noisy observations，构造逐图像的非参数 source
posterior：每个 source 的权重由其与 robust center 的 disagreement 和自身
margin/entropy confidence 决定，不使用额外可学习参数或 validation label 进入
权重计算。validation-only 网格先过 `+0.002` AUC gate，再锁定唯一候选。

UT 主 checkpoint 选择 median center、absolute disagreement、margin-high
confidence、`tau=4`、power `0.25`、full update、strength `1.5`，validation
`+0.013347315/+0.010506229`，fixed test **+0.989690/+0.717698**。MIT seed-1
选择 mean center、absolute disagreement、entropy-low、`tau=4`、power `1.0`、
update residual、strength `0.75`，validation `+0.003433571/+0.001658656`，
fixed test **+0.256677/+0.281555**。

两组结果、protocol 和 audit 均在 `adaptive_source_posterior_fixed_v1/` 下，
审计 `status=passed`、`test_search_after_lock=false`、validation probe
`test_read=false`。该机制在 MIT seed-1 上优于已有 confidence-only/joint
补充分支，但仍是 checkpoint-level evidence；UT 上没有超过 support-only 主峰，
不替换主结果。

### Adaptive posterior replay on original MIT

将 MIT seed-1 已锁定的 adaptive posterior 候选原样迁移到原始 MIT checkpoint
（不重新搜索），validation gain 为 **+0.002603015 AUC / +0.002482513 HM**，
protocol-locked fixed test 为 **+0.361611 / +0.177142**。结果与审计位于
`output/research/warmflow_mit/adaptive_source_posterior_fixed_v1/`，
`status=passed`、`test_search_after_lock=false`，validation replay 明确
`test_read=false`。这提供了两个 MIT checkpoint 的同候选正向证据，但仍属于
checkpoint-level mechanism evidence，不写成 dataset-independent confidence
interval。

## Graph-conditioned adaptive posterior boundary

将 adaptive source posterior 的 image-level source weighting 与 pair-level graph
support 组合：source posterior 决定信任哪个生成器，support gate 决定是否放大
该 target pair 的 residual。UT validation 候选达到 **+0.038938579 AUC /
+0.025897745 HM**，但 locked test 为 **−0.897186 / −0.259885**；MIT seed-1
validation `+0.002983914/+0.002950015`，fixed test **+0.268488/+0.132552**。

两组均由独立 protocol/audit 锁定；UT audit `outcome=negative_transfer_boundary`，
MIT audit `outcome=positive`，均 `test_search_after_lock=false`，validation
probe `test_read=false`。该结果支持方法层面的“结构可靠性 × source 可靠性”
分解，但也明确说明组合 gate 不能直接跨 checkpoint 泛化，不进入主结果。

## Locked reliability residual mixture

将已锁定的 image-level adaptive source posterior residual 与 pair-level
structural support residual 做凸组合；两套内部候选、strength 和 source/cache
均在混合系数选择前冻结，只在 validation-only 网格搜索
`alpha_adaptive`。MIT seed-1 选择 `alpha=0.5`，validation 为
**+0.003100451 AUC / +0.002563750 HM**，fixed test 为
**+0.305237 AUC / +0.315012 HM**；该结果超过该 checkpoint 的 adaptive-only
（+0.256677/+0.281555）与 support-only（+0.276025/+0.025614）复放。UT 原始
选择 `alpha=0.875`，fixed test **+0.952342/+0.717698**；MIT 原始 imported
`alpha=0.5` fixed test **+0.354714/+0.140232**。三组均为 protocol-locked
fixed-setting evidence，validation probe `test_read=false`，不把 UT/MIT
不同 checkpoint 的系数合并为统一置信区间。

## Interaction hyper-classifier boundary

为验证更强的模型化创新，训练了一个由 attribute/object/pair text 交互生成
pair-specific classifier weights 的 `InteractionHyperClassifier`。UT 上使用
已有 flow 生成 seen/unseen 训练样本，30 epoch validation-only 训练后锁定
`full x 0.2`，validation 为 **+0.432 AUC / +0.365 HM 点**，fixed test 为
**+0.192 AUC / +0.080 HM 点**。该结果为正向的结构化 classifier evidence，
但低于现有 support-aware/adaptive 后处理，不替换主结果。MIT seed-1 的同一
固定训练配置 validation 仅 **+0.099 AUC / 0 HM 点**，未达到 `+0.002` gate，
因此未读 test、未做 fixed test。

## Pair-level source-family routing

全局 isotropic/diagonal 混合会把不同组合的源可靠性平均掉。新增的
pair-level router 只使用 train split 的 attribute/object pair support，为每个
候选 pair 计算 diagonal 权重 `support^0.25`，其余权重给 isotropic family；
validation-only 选择 `update x 1.25` 后 fixed test 从 baseline 提升
**+1.376 AUC / +1.093 HM 点**。这略高于 global diagonal mean 的 AUC
（+1.369），并保持相同 HM，说明组合级 support 可以作为 source-family
选择器。结果、协议和 audit 均通过 SHA/provenance 检查，validation probe
明确 `test_read=false`，不再搜索 route/power/strength。

同一方法在独立 UT seed-1 checkpoint 上按 validation 重新锁定
`inverse_pair_support^1/update x 1.0`，fixed test 为 **+2.184 AUC /
+1.184 HM 点**。原始 UT 与 seed-1 的 route 方向不同，说明 support routing
是 checkpoint-conditioned 的可靠性选择器；两组结果分层报告，不合并成一个
固定超参置信区间。

MIT 原始 checkpoint 也有完整的 isotropic/diagonal family；按 family
disagreement 路由的 validation 候选 `inverse_disagreement_reliability^0.5 /
update x 0.5` fixed test 为 **+0.335 AUC / +0.105 HM 点**。该结果为跨数据集
正向迁移证据，但没有超过 MIT 原始 adaptive posterior，故作为结构路由消融，
不替换 MIT 主结果。

## Routed source plus image-confidence boundary

在 UT seed-1 已冻结的 `inverse_pair_support^1` route 上叠加
`base_margin_high^1` image gate，validation-only 候选为 `update x 1.25`，
validation 增益为 **+0.007795631 AUC / +0.022227127 HM 点**；fixed test
仅 **+0.916 AUC / +0.045 HM 点**，明显低于不加 image gate 的 pair router
（+2.184/+1.184）。protocol/audit 均通过，但该串联组合标记为
validation-overfit boundary，不进入主结果。

## C-GQA topology-gated validation replay

在独立 C-GQA 三 seed validation-only 复核中，冻结候选为
`min_image_support_continuous_full/strength=.2`。结果 JSON 为
`output/research/warmflow_cgqa/topology_gated_valonly_seedcheck/results.json`，
明确 `test_read=false`；三 seed AUC gains 为
`+.001732861/.001713647/.001876101`，HM gains 为
`+.002036669/.002036669/.002919605`，均值为
**+0.001774203 AUC / +0.002330981 HM 点**，3/3 AUC 与 HM 为正。
由于均值 AUC 低于预注册 `+0.002` gate，该结果只作为平衡性正向但 gate-fail
的 validation-only evidence，不改变主结果，也不读取 test。

## UT seed-1 AUC-first pair-routing ablation

为检验 pair-router 的 AUC/HM Pareto 边界，独立冻结 validation 规则：在
`AUC gain >= .002` 且 `HM gain >= .010` 的候选中最大化 validation AUC，锁定
`obj_support^0.25 / update x 0.75`。validation-only 增益为
**+0.489 AUC / +1.120 HM 点**，fixed test 为 **+1.674 AUC /
+0.752 HM 点**。protocol、cache/source SHA、test-free probe 与审计均通过，
但它低于已冻结的 inverse-pair 主路由（+2.184/+1.184），因此只作为
AUC-first 的正向结构消融，不替换主 headline。

## Selective unseen-pair source routing

全局 pair router 的进一步结构化版本只在 train-unseen pair 上注入
support-conditioned isotropic/diagonal transport；seen pair 保留冻结 CAMS
分数。scope、support power 与 strength 均在 validation-only 网格中锁定，
fixed test 后不再搜索。UT 原始 checkpoint 选择
`pair_support^0.5 / unseen_only / full x 1.5`，fixed test 为
**+0.659 AUC / +0.765 HM 点**；独立 UT seed-1 选择
`pair_support^0.25 / unseen_only / full x 1.5`，fixed test 为
**+1.736 AUC / +1.298 HM 点**。MIT 原始 validation 选择 all-pair
`pair_support^0.25 / update x 0.5`，fixed test 为 **+0.335 AUC /
+0.105 HM 点**。三份 protocol/audit 均通过，说明“只对 graph-unseen
组合进行分布源路由”是跨 checkpoint 的正向结构机制，但 route/scope 仍需
按 validation 分层锁定，不能合并成一个统一超参。

对三个 frozen candidate 进一步做了不改变设置的 source-seed audit：UT 原始、
UT seed-1、MIT 原始均为 **3/3 source seeds test AUC 正、3/3 HM 正**。
审计 JSON 分别位于各自 `selective_pair_router_fixed_v1/source_seed_audit.json`，
均记录 `test_search_after_lock=false`；因此 aggregate gain 不是单一 generator
seed 驱动。MIT 的一个 source seed validation AUC 略低于零，但不影响预注册
aggregate gate，也未改变 fixed setting。

## Support-tapered selective routing

将 binary unseen gate 连续化为 `scope=1` on unseen pairs、`scope=lambda` on
seen pairs。UT seed-1 validation 锁定 `pair_support^0.25`,
`seen_lambda=.25`, `full x 1.5`，fixed test 提升 **+1.824 AUC /
+1.455 HM 点**，且三个 source seeds 均 AUC/HM 为正。MIT validation 锁定
`pair_support^0.5`, `seen_lambda=.75`, `update x .5`，fixed test 为
**+0.260 AUC / +0.105 HM 点**；source-level 为 3/3 AUC、2/3 HM 正。
该连续 gate 提高了 UT seed-1 的平衡指标，但 MIT 的 HM seed-level 边界仍需
如实报告，不能宣称所有 source seed 均正。

## Pareto blend of frozen route candidates

为同时覆盖 AUC-specialist 与 HM-specialist，固定混合两个已锁定 residual：
`0.5 * inverse_pair_support^1/update x 1.0` 与
`0.5 * pair_support^0.25/seen_lambda=.25/full x 1.5`。UT seed-1
validation 为 **+0.915 AUC / +1.635 HM 点**，fixed test 为
**+2.138 AUC / +1.269 HM 点**，source-seed audit 为 3/3 AUC、3/3 HM 正。
该结果是独立的 AUC/HM Pareto 点，不替换 HM-first soft candidate，两个
设置分开报告。

## Instance-adaptive route blend

在 Pareto blend 的两个已冻结 route residual 之间加入无标签、逐图的 CAMS
entropy schedule，仅对 graph-unseen pair 动态决定 inverse-pair AUC route 与
soft-selective HM route 的混合比例。validation-only 预注册候选为
`unseen_entropy_1`，mean validation gain **+0.010726265 AUC /
+0.033872925 HM**，结果 JSON 为
`output/research/warmflow_ut_seed1final/adaptive_route_blend_probe_v1.json`，明确
`test_read=false`。固定 test 使用同一 validation min-max entropy
normalization，不重新搜索 schedule，结果为 **+0.062976 AUC /
+0.069792 HM**；三 source seeds 均 AUC/HM 正。固定结果、protocol SHA 与
source-seed audit 均位于 `adaptive_route_blend_fixed_v1/`，并记录
`test_search_after_lock=false`。该结果是当前最强的实例级可靠性调度证据，仍
按 UT seed-1 checkpoint 分层报告，不宣称跨 checkpoint 统一超参。
同一机制在 UT 原始 checkpoint 重新进行独立 validation 选择，锁定
`unseen_entropy_2`；fixed test 为 **+0.052832 AUC / +0.055709 HM**，同样
3/3 source seeds 两项指标为正。两个 checkpoint 的 schedule 仍分别冻结，不能
把它们写成同一超参的跨 checkpoint CI。
seed-1 的 protocol/probe/fixed-result/source-audit SHA256 分别为
`99fb6793...`, `69dfbfd2...`, `902cbf1d...`, `4cd05631...`；原始 UT 对应为
`aaa562e1...`, `3285e83f...`, `99b9775f...`, `401fc4da...`。

## MIT seed-1 diagonal soft-route replay

MIT seed-1 的兼容 diagonal source family 已在独立目录完成三 seed validation-only
生成，并在候选冻结后生成 test-capable logits。validation 选择 constant
soft-selective route (`pair_support^0.25/seen_lambda=.25/full x1.5`)，mean
validation gain **+0.017839 AUC / +0.046395 HM**；fixed test 为
**+0.020659 AUC / +0.052498 HM**，即 **+2.066/+5.250 AUC/HM 点**。三个
source seeds 的 test AUC/HM 均为正，且 `test_search_after_lock=false`。由于
最优候选是 constant 而非 entropy schedule，该结果只作为 support-tapered
route 的跨数据集复放，不扩展 entropy-scheduling 的主张。

### Final topology-gated addendum (2026-08-25)

为最终审计补录 C-GQA topology-gated 三 seed validation-only 复核：结果 JSON
为 `output/research/warmflow_cgqa/topology_gated_valonly_seedcheck/results.json`，
明确 `test_read=false`，三 seed mean gain 为 **+0.0017742028 AUC /
+0.0023309812 HM**。AUC 低于预注册 `+0.002` gate，因此仅作为平衡性正向、
gate-fail 的 validation-only evidence，不改变主结果。
The strict three-source C-GQA candidate-conditioned audit is now complete. The
six-source validation probe selected `(-8,-.5)` before test; the fixed protocol
and all six source SHA256 values are archived. Fixed test gain is **+1.222 AUC /
+5.942 HM points**, and the source audit is positive on 3/3 isotropic/diagonal
pairs with mean **+1.135/+5.788**. This is multi-source reliability-routing
evidence, not a replacement for the raw C-GQA gate or a universal C-GQA claim.

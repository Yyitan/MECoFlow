# MECoFlow 论文方案（工作稿）

## 工作标题

**Maximum-Entropy Compositional Flow for Generating Unseen Visual Distributions**

当前方法名：MECoFlow（Maximum-Entropy Compositional Flow，暂定）。它是 ReCoFlow 原型经过受控源分布消融后的简化与增强版本；截至 2026-08-24 的精确名称检索未发现同名方法，但定稿前仍需复检。

## 核心问题与 motivation

主流 CLIP-CZSL 方法最终仍以一个文本点原型表示未见属性–物体组合。这个做法忽略两个事实：

1. 未见组合在视觉空间中不是一个点，而是包含视角、材质、形状、背景等变化的条件分布；
2. 从零均值球面噪声直接生成该分布，对样本有限且类别数很多的 CZSL 来说条件跨度过大，实验证明会出现条件坍缩；
3. 确定性可加原型虽然缩短了生成路径，却完全丢失类内变化；需要一个既靠近目标组合、又不对不可验证的高阶残差结构作额外假设的随机源；
4. 已见组合已经提供可复用的视觉残差总能量，真正缺失的是属性与物体交互后的条件运输规律。

因此，将生成目标从“文本/噪声到视觉特征”改写为：

> 先以属性–物体加性均值定位未见组合，再以真实残差能量构造最大熵随机源，最后学习从该短程源分布到真实视觉分布的条件运输。

## 方法

### 1. 可加原语先验

对已见组合质心执行带样本数权重的两因素分解：

\[
\mu_{a,o}=\mu_0+u_a+v_o,
\]

其中 `u_a` 与 `v_o` 通过交替加权残差更新估计。它为任何未见组合提供一个有视觉数据支撑的初始均值。

### 2. 数据校准的最大熵源分布

先从真实训练实例估计残差总能量：

\[
r_i=f_i-\mu_{a_i,o_i},\qquad
\sigma^2=\frac{1}{Nd}\sum_i\|r_i-\bar r\|_2^2.
\]

在只约束均值与总方差、且不假设未见组合高阶残差结构的条件下，最大熵源为各向同性高斯：

\[
\epsilon\sim\mathcal N(0,I),\qquad
x_0=\operatorname{norm}(\mu_{a,o}+s(\bar r+\sigma\epsilon)).
\]

形式化理由是：固定协方差时高斯最大化微分熵；总能量约束固定 `tr(Σ)`；在固定迹下，AM-GM 说明 `det(Σ)` 在全部特征值相等时最大，因此 `Σ=σ²I`。这把各向同性源从经验性选择提升为“只有总变异量可知时的最少额外假设”。

`s∈{0,0.5,1}` 在 flow 训练中随机采样，生成尺度由验证集选择。受控三种子消融表明，经验 bootstrap、逐维方差高斯和等总方差各向同性高斯在 MIT 上统计等价；UT 上各向同性明显优于经验 bootstrap，但与逐维高斯持平。主方法采用各向同性形式的理由是总能量约束下的最大熵最小假设、更少统计量和不增加种子方差，而不是宣称其显著优于所有高斯源。

固定相同 flow、仅在生成时令 `s=0` 的确定性对照在 UT 上只有 44.094 AUC / 56.613 HM，而随机最大熵源达到 45.038 / 57.429；这有力支持分布采样是收益来源之一，而不是把运输后的单个原型重复 32 次。

### 3. 原语条件 rectified transport

给定真实目标 `x1=f_i`，在线性路径

\[
x_t=(1-t)x_0+t x_1
\]

上学习速度场：

\[
\mathcal L_{FM}=\mathbb E\left\|v_\theta(x_t,t,e_a,e_o,e_{a,o})-(x_1-x_0)\right\|_2^2.
\]

轻量 MLP 使用 AdaLN 式条件调制；训练中附加已见组合质心分类约束。主版本用 8 个 Euler 步，1 步版本作为效率/方差消融。

### 4. 分布完成与平衡分类

对每个未见组合采样 `M=32` 个视觉特征，与每个已见类均衡抽取的真实特征共同训练文本初始化余弦分类器。生成分数作为 CAMS 的补充分支：

\[
S=S_{CAMS}+\lambda S_{gen}
\]

或去除生成分类器继承的原始文本分量，只注入学习到的视觉修正：

\[
S=S_{CAMS}+\lambda(S_{gen}-S_{text}).
\]

`full/update` 与 `λ` 只由验证 AUC 选择。生成器不参与测试时采样，推理额外成本仅为已缓存分类器权重的一次矩阵乘。

### 5. 可靠性门控

当验证 AUC 的绝对增益低于 0.002 时，令 `λ=0` 回退基线。这不是用于制造提升的后验筛选，而是一个预先固定的安全机制，用来避免生成分支在组合极多、单类极稀疏的数据集上负迁移。

## 预期贡献写法

1. 提出 CZSL 的**视觉分布缺失**视角：未见组合的核心缺口不是单一语义原型，而是条件视觉分布；
2. 提出**可加原语定位 + 数据校准最大熵扰动**的组合源分布，在无未见图像时以最小分布假设同时保留组合位置与类内多样性；
3. 提出从该短程源到真实视觉分布的原语条件 rectified transport，以及去除文本初始化项的视觉修正融合；
4. 方法是训练期分布完成、测试期轻量分类器的 plug-in；跨生成器种子和两个独立 CAMS 权重均获得一致提升，且单卡显存远低于 A100 40G 限制。

## 与近期工作的边界

| 方法 | 生成/运输对象 | 关键操作 | 与 MECoFlow 的区别 |
|---|---|---|---|
| DIFFCZSL (2026) | 预训练扩散表征 | CleanDIFT→CLIP 对比蒸馏 | 不生成未见视觉分布；依赖 SD2.1 表征 |
| FlowComposer (CVPR 2026) | 视觉点→文本点 | 原语速度场组合 | 点级跨模态对齐，而非先验分布→视觉分布 |
| LPD (KBS 2026) | 未见特征 | triplet + learnable primitive disparity | 不使用最大熵组合源与连续分布运输 |
| SemanticFlowNet (GZSL, 2026) | 语义→视觉特征 | decoupled flow matching、可选 CFG | 面向常规 GZSL；仍是语义/噪声条件生成，不建模属性–物体组合残差源分布 |
| DeFA (2025) | batch 实例属性/物体特征的 Cartesian 重组 | 残差 MLP + 频率去偏 | 多样性来自实例重组、与主干联合训练；无最大熵随机源和 ODE 运输 |
| Task-Aware Feature Generation (2019) | 未见特征分布 | 层级 task-conditioned noise | 无 CLIP/CAMS、无组合加性位置与残差能量约束；因此不能声称首个生成式 CZSL |
| ADiVA (GZSL, 2026) | instance-level 属性条件分布 | 可迁移属性分布 + 视觉引导语义对齐 | 建模普通 ZSL 的语义条件分布，而非 CZSL 的最大熵视觉源与组合运输 |

方法公式、最大熵定理边界、算法和审稿质疑见 `research/METHOD_DRAFT.md`；详细逐项核对、允许/禁止主张及所需消融见 `research/NOVELTY_AUDIT.md`；论文报告数值、插件增益和协议可比性见 `research/RELATED_WORK_COMPARISON.md`；统一门控的阈值敏感性及 open-world 反例见 `research/GATE_AUDIT.md`。

## 当前可支持的结论

- 闭世界、统一三生成器种子选参：MIT AUC/HM 为 23.416±0.029 / 40.130±0.073（+0.293/+0.235）；UT 为 45.038±0.126 / 57.429±0.313（+1.280/+0.874）；C-GQA raw 分支由门控安全回退，structural-risk selective replay 有限正向；
- 无 flow UT 测试大幅下降，支持提升不是简单的平衡分类或添加伪样本；
- UT 的 1-step 原始三 seed AUC 为 44.618±0.093（+0.860）且全部为正，但验证增益 +0.149 点未过门控；8-step 达到 45.038±0.126（+1.280），验证增益 +1.120 点并通过门控，支持多步运输提高效应强度与可选择可靠性；
- 与主 flow 完全参数匹配的 direct endpoint regressor 仅达到 43.934±0.111 AUC / 56.620±0.196 HM；8-step rectified flow 额外提高 +1.105/+0.809，说明收益不是由网络容量本身解释；
- 当前不能声称 open-world 改进：UT open-world 存在明确负迁移；
- 第二个独立 UT CAMS checkpoint 上，三个生成器统一选择 `update × 1.0` 后，AUC/HM 为 42.482±0.581 / 54.975±0.407（+2.034/+1.003）；平均验证增益也超过可靠性门槛，证明提升跨基础模型权重保持；
- 新训练的 MIT CAMS seed-1 上，三个生成器统一选择 `update × 0.5` 后，AUC/HM 为 23.517±0.043 / 40.398±0.103（+0.220/+0.069）；平均验证增益 +0.273 AUC 点，通过 0.2 点安全门槛，三个测试 AUC 均为正；
- 四个 CAMS checkpoint 的 12 个生成器运行全部获得正 AUC 增益，逐 checkpoint 的 95% seed-t 匀间下界均为正；这只证明优化/采样随机性下的一致性，不能当作独立数据集总体显著性。HM 为 11/12 正，不扩大主张；
- 经验残差、对角高斯、各向同性高斯在 MIT 上统计等价；UT 上各向同性与对角高斯也基本持平（45.038 vs 45.030 AUC），但各向同性种子方差更低。不能声称高阶非高斯结构或各向同性带来显著精度增益；主张已修正为组合均值定位、真实残差能量校准与最小假设。
- 同残差能量的零中心源在 MIT 上为 23.335±0.051 AUC / 39.986±0.090 HM；加性组合中心额外带来 +0.081 AUC / +0.144 HM，并降低种子方差，支持“先定位、再运输”的设计。
- MIT open-world 在标准验证选参下三种子 AUC/HM 为 8.792±0.044 / 22.449±0.110（+0.149/+0.179），三次均为正；但验证增益低于安全门槛，门控协议回退。UT open-world 仍为负，不能作通用 open-world 主张。
- 独立的 AOAR-Replay 非扩散探索中，Gaussian-sampling 版本获得 44.378±0.036 AUC（+0.620，3/3 正）但 HM 为 56.208±0.327（−0.347，0/3 正）；validation 预选的 conditional-mean 版本进一步达到 44.466±0.065 AUC（+0.708，3/3 正），HM 仍为 56.376±0.359（−0.180，仅 1/3 正）。因此它只作为“生成式特征回放可稳定改善 AUC，但平衡校准仍未解决”的补充证据，不并入 MECoFlow 主贡献或摘要。
- 官方 FlowComposer Stage-1/Stage-2 已按固定 commit 和 official-code protocol 完成；Stage-2 固定最终 `gate_14.pth`，不按逐 epoch test 曲线选优。该结果只能作官方代码协议的外部强生成参照，不能宣称与冻结 CAMS/validation-gated MECoFlow 是严格同协议 SOTA 对比。
- 其机器可读最终 endpoint evaluation 为 `AUC=0.4101/HM=0.5360`（`stage2_metrics.json`，`gate_14` 规则）；正文应把它作为 official-code 量级参照，并明确其 test-access protocol 与 MECoFlow 不同。
- 新 UT CAMS seed2 base 的 amended provenance cache 通过全部 SHA/split/transform 审计，但三 seed compatibility replay 共享 validation gain 仅 +0.000680 AUC 点，低于 gate，按协议不读 test；该结果作为负边界与防泄漏证据报告。

## Final validation closure

The final-checkpoint validation sweep is closed without additional parameter
search. On the official FlowComposer final `gate_14.pth`, the only permitted
TR-PFC setting (`adaptive tangent scale=0.015`, `cap=0.02`) changed the final
deterministic validation result from AUC/HM `0.226071200/0.372595985` to
`0.226778922/0.374330060`, namely **+0.070772 AUC points / +0.173407 HM
points**. Because the AUC gain is below the fixed `+0.20` gate, TR-PFC is a
mechanism diagnostic and negative-boundary result, not a main-paper selected
setting; test was not read.

The separate deterministic train-transform robustness branch for the amended
seed2 checkpoint also failed the same frozen validation gate: three generator
seeds gave shared mean validation AUC gain **-0.002904064**. Its cache contained
train/val only and recorded `test_read=false`; it did not modify the parent
protocol or the compatibility replay. These negative results should be shown as
failure boundaries/safety-gate evidence, not combined with the earlier positive
UT/MIT checkpoint results as a new universal claim.

The C-GQA topology-gated validation-only cross-seed check is a bounded positive
balance signal but still gate-fail. With
`min_image_support_continuous/full/strength=0.2`, the three AUC gains were
`+0.001733/+0.001714/+0.001876` and the HM gains were
`+0.002037/+0.002037/+0.002920` (3/3 positive for both); the shared mean was
**+0.0017742028 AUC / +0.0023309812 HM**, below the fixed `+0.002` AUC gate.
The result is validation-only (`test_read=false`) and is recorded at
`output/research/warmflow_cgqa/topology_gated_valonly_seedcheck/results.json`
with script SHA prefix `1a5c0655`; it does not alter the main result.

An additional column-centered support calibration looked positive on the
validation split (`+0.002103701 AUC / +0.003030922 HM` mean), but its unique
candidate was frozen before a single fixed-setting test under protocol SHA
`bab3c4f8...`. Test transfer was negative (`−0.002914084 AUC /
−0.004689102 HM`, 0/3 positive), so this mechanism is a validation-overfit
failure boundary and must not be presented as a C-GQA improvement.

## Fixed diagonal-source ensemble upgrade

The original UT checkpoint also supports a stronger, separately locked source
ensemble: average three diagonal Gaussian generator logits, subtract the CLIP
text score (`update`), and use strength `1.25`. The candidate was frozen after
validation-only 5-fold cross-fit (`+0.017365088 AUC / +0.010027219 HM`, all
folds positive), then evaluated once under protocol SHA `b5d86b1c...`. The
three-logit ensemble reaches **45.126939 AUC / 57.648592 HM**, relative to
`43.758287/56.555582`, for **+1.368652 AUC / +1.093010 HM points**. The same
frozen strength applied per generator gives mean gains `+1.224179/+0.810692`
points, so the ensemble and the per-seed sensitivity must be reported
separately. This is a fixed source-ensemble calibration result, not a claim
that generator seeds are independent datasets; protocol/result/audit hashes are
stored and no post-test search was performed.

The paper's final evidence table must therefore retain the original closed-world
MECoFlow results as the main result, report C-GQA and UT open-world fallback, and
place seed2 compatibility, deterministic robustness, and final TR-PFC in the
reproducibility/failure-boundary section. No further GPU sweep or test-dependent
search is justified by these outcomes.

The diagonal-source ensemble now has an independent UT checkpoint replication.
The original checkpoint uses validation-locked `update x 1.25` and reaches
`+1.368652 AUC / +1.093010 HM` points; UT seed-1 uses validation-locked
`update x 0.75` and reaches `+1.674186 AUC / +0.751901 HM` points. Both are
three-logit ensembles with 3/3 positive per-generator test gains under their
respective fixed protocols. Report them as checkpoint-level evidence, not as a
single same-strength confidence interval.
The seed-1 test-enabled reruns are bitwise identical to their validation-only
logits (`max_abs_diff=0.0` for all three seeds), as recorded by the replay
consistency audit.

## 投稿判断门槛

在以下条件满足前，不建议直接宣称顶会级完成：

1. 跨基础权重证据已补齐：原始 MIT/UT checkpoint、UT seed-1 与新训练 MIT seed-1 四个数据集–checkpoint 设置的平均 AUC 都为正；后续若算力允许再增加完整 CAMS 训练种子以收紧置信区间；
2. C-GQA structural-risk selective replay 已有单源 **+0.283 AUC/+2.119 HM 点**正增益，但仍需更多 source seed/checkpoint 才能升级为可信多源结论；
3. MIT open-world 已有稳定小幅正结果，但强度不足；主文应以 closed-world 为核心，open-world 放补充材料并明确 UT 负结果；
4. 继续争取 DIFFCZSL 的同 CAMS checkpoint 对照；FlowComposer 官方复现单独标为 official-code protocol，LPD/DeFA 保留 external-reported 对照，不伪称同协议；
5. 时间/显存/参数与源分布、随机性、源位置、no-flow、1/8-step 消融已完成并由 JSON 验证器统一生成；接下来需要把机制链条压缩成清晰的主表，避免“只是换成 flow matching”的审稿印象。

### Topology-gated 复核审计条目

将 C-GQA topology-gated 三 seed 复核作为 validation-only 补充证据：结果
JSON 为 `output/research/warmflow_cgqa/topology_gated_valonly_seedcheck/results.json`，
`test_read=false`，均值为 **+0.0017742028 AUC / +0.0023309812 HM**。该均衡
正信号仍低于 `+0.002` AUC gate，论文中标为 gate-fail，不改主结果或结论。

### 最终 topology-gated 补录（2026-08-24）

审计再次锁定 `output/research/warmflow_cgqa/topology_gated_valonly_seedcheck/results.json`，
其中 `test_read=false`，三 seed mean gain 为 **+0.0017742028 AUC /
+0.0023309812 HM**。AUC 未过预注册 gate，因此只作为 validation-only 的
平衡性正向、gate-fail 证据，不纳入主结果或用于 test 选参。

### Joint support-confidence reliability gate

新增一个可解释的机制消融：pair-level graph support 描述组合结构的可靠性，
base classifier 的 entropy/margin 描述当前图像样本的可靠性，二者共同调节
distribution-completion transport 的残差幅度。所有候选先在 validation-only
网格中通过 AUC gate，再以 HM 锁定唯一设置；fixed test 不重新搜索。

UT 的 `object support × base margin-high` 得到 validation `+0.019773128 /
+0.012435618`，fixed test `+1.272944 AUC / +0.836122 HM`；MIT seed-1 的
`attribute support + base entropy-low` 得到 validation `+0.002816028 /
+0.004155675`，fixed test `+0.161473 / +0.021861`。该联合门控有明确机制
解释和完整 provenance，但没有超过 UT support-only 或 confidence-only 的
绝对峰值，因此放入机制/消融表，不改主结果。对应结果已由
`research/verify_paper_results.py` 自动纳入 `research/PAPER_RESULTS.md`。

UT seed-1 必须作为反例同表报告：同一联合 gate 在 validation 上为
`+0.004301443/+0.024252643`，但 locked test 为 **−2.416473 AUC /
−1.632702 HM**，3/3 source 均负。论文应将该结果解释为 checkpoint-specific
transfer risk，并明确 validation gate 不能单独保证跨 checkpoint 泛化；这条
负边界是审稿时防止“只挑正例”的关键证据。

### Adaptive source-posterior aggregation

将多个 flow source 解释为 noisy observations，并按每个图像的 source
disagreement 与 source confidence 估计非参数 posterior，再聚合 residual。
该机制不增加 flow 参数，且每张图像的权重不依赖标签；唯一候选仍由统一
validation AUC/HM gate 锁定。UT fixed test 为 `+0.989690 AUC / +0.717698 HM`，
MIT seed-1 为 **+0.256677 / +0.281555**。MIT 分支是当前最有价值的新平衡性
补充，但 UT 未超过 support-only 主分支，故主表中列为 checkpoint-level
mechanism ablation，不改 MECoFlow 主结果。

同一 MIT seed-1 候选原样迁移到原始 MIT checkpoint 后仍为正：fixed test
`+0.361611 AUC / +0.177142 HM`。因此 adaptive posterior 可以作为两个 MIT
checkpoint 的一致机制补充；但它不是对所有 dataset/checkpoint 都锁定同一
超参数的主方法，UT 采用了不同 validation 候选，论文中需保持 checkpoint-level
范围声明。

Graph-conditioned adaptive posterior 只放机制消融：MIT seed-1 的 fixed test
为 `+0.268488 AUC / +0.132552 HM`，但 UT 的强 validation 信号转为
`−0.897186/−0.259885`。这条反例说明 source reliability 与 graph support 的
组合存在 checkpoint-specific calibration，不能作为统一主算法或跨数据集结论。

机器可读的 checkpoint-level 汇总为
`output/research/warmflow_ut/ut_diagonal_ensemble_checkpoint_summary.json`。
该文件记录两组 protocol/result/audit SHA 链、明确的
`checkpoint_level_not_dataset_independent` 范围，以及 validation-only 的
跨分布负边界（`test_read=false`）。

MIT seed-1 的 isotropic-source median 也通过了 validation gate，并在一次
protocol-locked test 中得到 **+0.205706 AUC / +0.112173 HM** 点。它补强了
“源分布聚合而非单一随机样本”的机制证据，但由于逐源 HM 仅 1/3 为正，主文中
应放在平衡性补充表，不与 UT 的稳定三源 HM 结果混为一谈。结果和审计路径为
`output/research/warmflow_mit_seed1final/source_median_fixed_test.json` 与
`output/research/warmflow_mit_seed1final/source_median_fixed_audit.json`。

UT 的六源 median（isotropic×3 + diagonal×3）在 fixed test 上达到
**+1.263431 AUC / +0.999660 HM**，且 6/6 source-level AUC/HM 均为正。
它低于 diagonal mean 主设置的绝对提升，因此应作为鲁棒性和方差控制证据，
不替换主表的 diagonal mean。结果和审计路径为
`output/research/warmflow_ut/cross_source_median_fixed_test.json` 与
`output/research/warmflow_ut/cross_source_median_fixed_audit.json`。

另外，source-consensus inverse gate（AUC gate 后按 validation HM 选取）在
UT fixed test 达到 **+1.111376 AUC / +1.046375 HM**，六个 source-level
结果全部为正。它补足了“source agreement 可用于抑制不确定更新”的机制链，
但绝对 AUC 低于 diagonal mean，故放在方法消融/鲁棒性表，不改变主设置。结果
和审计路径为 `output/research/warmflow_ut/consensus_gate_fixed_test.json`
与 `output/research/warmflow_ut/consensus_gate_fixed_audit.json`。

Support-aware source-consensus is the stronger balance-oriented mechanism
ablation. It multiplies source agreement by a harmonic image-support prior
computed only from the seen training graph, with a predeclared
`support_exponent=0.25`, inverse agreement `tau=2.0`, linear blend, and update
strength `0.75`. The validation-only gain is **+0.012893568 AUC /
+0.009909548 HM**; the protocol-locked fixed test reaches **+1.305096 AUC /
+1.232444 HM**, with 6/6 source-level AUC/HM gains positive. The protocol,
result, and passed audit are
`research/protocols/ut_support_consensus_fixed_v1.json`,
`output/research/warmflow_ut/support_consensus_fixed_test.json`, and
`output/research/warmflow_ut/support_consensus_fixed_audit.json`. It improves
HM over simple consensus gating but remains below the diagonal-mean AUC peak,
so it belongs in the mechanism/robustness table rather than replacing the main
setting.

On the independent UT seed-1 checkpoint, the original harmonic-support plus
inverse-consensus candidate was rejected by validation (`-0.006542405 AUC`),
which is an important anti-overclaim control. Applying the same preregistered
support-aware grid to that checkpoint selected a support-only candidate
(`min_image_support`, exponent `0.5`, strength `0.75`) with validation gain
**+0.004957443 AUC / +0.012596196 HM** and fixed-test gain **+1.861108 AUC /
+0.968736 HM**; all three source-level AUC/HM gains were positive. Keep this as
independent transfer evidence, not as a same-setting confidence interval.
The result and audit are
`output/research/warmflow_ut_seed1final/support_only_fixed_test.json` and
`output/research/warmflow_ut_seed1final/support_only_fixed_audit.json` under
protocol `research/protocols/ut_seed1_support_only_fixed_v2.json`.

MIT seed-1 gives a complementary AUC-oriented check: the same support-aware
grid selects `min_image_support` with exponent `1.0` and strength `0.5`, yielding
validation-only **+0.002848590 AUC / +0.002344917 HM** and fixed-test
**+0.276025 AUC / +0.025614 HM**. All three source AUC gains are positive, while
source HM is positive for 2/3 runs. Because its HM is below the MIT source-median
result, report it as a reliability/AUC supplement rather than a new main setting.
The locked artifacts are
`output/research/warmflow_mit_seed1final/support_only_fixed_test.json` and
`output/research/warmflow_mit_seed1final/support_only_fixed_audit.json` under
`research/protocols/mit_seed1_support_only_fixed_v1.json`.

The factorized-support ablation on original UT separates attribute and object
reliability. The validation-locked object-side variant (`alpha_attr=0`, mean
aggregation, power `1.0`, strength `1.0`) gives **+1.469635 AUC / +0.952866 HM**
on fixed test with 6/6 positive source-level AUC/HM gains. Its AUC is marginally
higher but HM lower than the harmonic support-aware gate, so it should be shown
as a factorized reliability ablation, not selected as the balanced headline.
Artifacts: `output/research/warmflow_ut/factorized_object_support_fixed_test_v2.json`,
`output/research/warmflow_ut/factorized_object_support_fixed_audit_v2.json`, and
protocol `research/protocols/ut_factorized_object_support_fixed_v2.json`.

C-GQA is an explicit negative transfer control for this mechanism: the same
factorized support grid produced no validation candidate above the `+0.002` AUC
gate, with the best candidate at **-0.001210 AUC / -0.000611 HM**. Keep this
validation-only result in the boundary/ablation table and do not run or report a
fixed C-GQA test for it.

Confidence-aware transport is a complementary mechanism ablation. Using only
frozen CAMS image confidence, the UT low-entropy gate reaches **+1.505562 AUC /
+1.029062 HM**, while the independent MIT seed-1 high-entropy gate reaches
**+0.201655 AUC / +0.164343 HM**. Both are protocol-locked after validation-only
selection; 6/6 UT source-level metrics and 3/3 MIT AUC gains are positive. It
adds no flow parameters and should be reported beside, not merged into, the
support-aware headline.

### Reliability residual mixture

把 source-posterior reliability 与 pair-support reliability 从“串联 gate”改为
两个已冻结 residual expert 的凸组合，只用 validation 选择一个混合系数。MIT
seed-1 的 locked `alpha=0.5` 在 fixed test 达到 **+0.305 AUC / +0.315 HM 点**，
并在原始 MIT 复放达到 **+0.355/+0.140**；UT 原始为 **+0.952/+0.718**。该结果
提供了比单一 confidence 或 support gate 更平衡的机制证据，但由于候选随
checkpoint 固定，不应把三组数字汇总成同一设置的置信区间，也不替换闭世界主表。

### Interaction hyper-classifier ablation

Pair-specific hyper-classifier 用 attribute/object/pair 的低秩交互生成分类器
权重，是比后处理 gate 更强的结构化创新候选。UT fixed test 为 **+0.192 AUC /
+0.080 HM 点**，但低于 support-aware 主分支；MIT seed-1 validation 未过 gate。
因此作为方法消融和负边界报告，不纳入主 headline。

### Pair-level source-family routing

全局 source ensemble 的新补充是组合级路由：用 train pair 的 attribute/object
support 为每个候选组合分配 isotropic/diagonal source 权重。UT fixed test 为
**+1.376 AUC / +1.093 HM 点**，略高于 global diagonal mean 的 AUC 且 HM 相同。
这是当前最清晰的“结构可靠性驱动分布源选择”证据，建议作为主方法的结构化
消融或增强模块；protocol 只允许 validation 选择 `support^0.25, update x 1.25`。

在独立 UT seed-1 上，validation 选择了相反的 `inverse_pair_support^1` 路由，
fixed test 达到 **+2.184 AUC / +1.184 HM 点**。这支持“按 checkpoint 的组合
结构选择分布源”的观点，但也要求论文明确：route/power 通过 validation
独立锁定，不能把两个 checkpoint 的结果汇总成同一个 fixed-setting CI。

MIT 原始也复用了完整 isotropic/diagonal family，按 family disagreement 的
`inverse^0.5` 路由取得 **+0.335 AUC / +0.105 HM 点**。它支持 source routing
机制跨数据集有效，但增益小于 MIT adaptive posterior，因此应放在结构路由
消融表，不作为 MIT 主 headline。

在 UT seed-1 上把 `base_margin_high^1` image-confidence gate 叠加到
`inverse_pair_support^1` 路由后，fixed test 仅 **+0.916 AUC / +0.045 HM**，
未复现 pair router 的 **+2.184/+1.184**；该组合应作为 validation-overfit
负边界，不进入主方法。

C-GQA 的 topology-gated 三 seed validation-only 复核（结果 JSON：
`output/research/warmflow_cgqa/topology_gated_valonly_seedcheck/results.json`）
在 `test_read=false` 下取得 mean **+0.001774203 AUC / +0.002330981 HM**，
三 seed 的 AUC/HM 均为正，但 AUC 低于预注册 `+0.002` gate。因此将其作为
“平衡性正向但 gate-fail”的补充证据，不升级为主结果，也不进行 fixed test。

另有一个预注册的 AUC-first pair-routing 消融：在 validation 同时要求
`AUC gain >= .002`、`HM gain >= .010` 后最大化 AUC，锁定
`obj_support^0.25/update x 0.75`，fixed test 为 **+1.674 AUC /
+0.752 HM 点**。它验证了 source routing 的 AUC/HM Pareto 结构，但低于
`inverse_pair_support^1` 的 **+2.184/+1.184**，因此放入消融表而不作为
主 headline。

### Selective unseen-pair source routing

将 source-family routing 约束到 graph-unseen 组合，是当前最有解释力的增强：
seen pair 保留真实图像监督下的 CAMS 分数，只有 unseen pair 接收由 train
support 决定的 isotropic/diagonal mixture。UT 原始 fixed test 为
**+0.659 AUC/+0.765 HM**，UT seed-1 为 **+1.736/+1.298**，MIT 原始为
**+0.335/+0.105**。validation 会锁定 scope（UT 两个 checkpoint 为
unseen-only，MIT 为 all）、support power 与 strength；论文应分层报告这些
冻结设置，避免声称存在一个跨 checkpoint 的统一 route。该模块可作为
MECoFlow 的结构主增强，核心贡献表述为“graph-support-conditioned selective
transport”，而不是简单的后验置信度 gate。

source-seed audit 进一步支持该主线：三个 checkpoint 各自的三个 generator
source 都保持 test AUC/HM 正增益（9/9 AUC、9/9 HM）。因此主表可以报告
aggregate fixed gain，补充表报告 source-level positivity；MIT 个别 seed 的
validation 波动应如实保留，不把它隐藏成统一 validation 保证。

### Support-tapered selective transport

binary unseen-only gate 可以连续化为 `scope=1`（unseen）与
`scope=lambda`（seen）。UT seed-1 的 frozen soft candidate
(`pair_support^0.25`, `lambda=.25`, `full x 1.5`) 达到 **+1.824 AUC /
+1.455 HM**，并且 3/3 source seeds 均正；这是当前最强的平衡设置。MIT
的 validation 选择 `lambda=.75/update x .5`，fixed test **+0.260 AUC /
+0.105 HM**，但 source-level HM 为 2/3，作为正向但有边界的跨数据集证据。
论文应将 lambda 解释为 graph-support uncertainty taper，而不是任意的
confidence threshold。

另保留一个 validation-locked Pareto blend：将 inverse-pair AUC specialist
与 soft selective HM specialist 各取 0.5，UT seed-1 fixed test 为
**+2.138 AUC/+1.269 HM**，3/3 source seeds 全部正。该结果适合放入
主增强的 Pareto 消融表；headline 可按论文偏好的 AUC-first 或 HM-first
目标分别选择，但不能把两个 blend 权重重新按 test 调整。

### Instance-adaptive selective transport

在两个已经冻结的 route residual 之间加入逐图 CAMS entropy schedule。仅对
graph-unseen pair 使用该 schedule，seen pair 保持 CAMS；schedule 由
validation-only 选择并固定 validation 的 min-max normalization，test 不再
搜索。UT seed-1 锁定 `unseen_entropy_1` 后 fixed test 达到 **+0.062976
AUC/+0.069792 HM**，3/3 source seeds 两项指标均正。该结果提供了比固定
convex blend 更强的实例级可靠性调度证据，论文中可作为 selective transport
的主要机制扩展；但它仍是 checkpoint-conditioned evidence，不应与 MIT/UT
原始 checkpoint 的不同 route 设置合并成统一置信区间。
在 UT 原始 checkpoint 上独立 validation 选择 `unseen_entropy_2` 后，fixed
test 为 **+0.052832 AUC/+0.055709 HM**，3/3 source seeds 两项指标均正；这
支持机制级迁移，但 schedule 仍按 checkpoint 分层锁定。

MIT seed-1 通过独立 diagonal source replay 进一步验证 support-tapered route：
validation 选择 constant soft route，fixed test **+2.066 AUC/+5.250 HM**，
3/3 source seeds 两项指标均正。该结果放在 cross-dataset route replay 表中，
不作为 entropy schedule 的证据。

### Final topology-gated audit addendum (2026-08-25)

最终审计记录固定 JSON `output/research/warmflow_cgqa/topology_gated_valonly_seedcheck/results.json`，
`test_read=false`，三 seed mean gain 为 **+0.0017742028 AUC / +0.0023309812 HM**。
该结果仍低于 `+0.002` AUC gate，只作为 validation-only 的平衡性正向证据，不改变主结果。

### MIT cache compatibility boundary

原始 MIT 与 seed-1 feature cache 的 validation baseline 不一致：AUC/HM 分别为
`0.264784525/0.435349694` 和 `0.259733820/0.428700452`。因此不从 seed-1 cache
派生原始 MIT diagonal 结果，也不把 MIT seed-1 replay 写成跨 checkpoint 统一证据；
该决定记录在 `mit_cache_compatibility_audit.json`。

### Instance-adaptive probe reproducibility amendment

instance-adaptive route probe 已修正 pair-seen mask 的实现，并在三个 checkpoint
上做了 validation-only 逐位复现。修正前后候选和 validation gains 完全一致，
独立审计 `adaptive_route_blend_probe_repro_audit.json` 为 passed、`test_read=false`；
因此现有 fixed gains 仍有效，但论文应引用修正后的脚本 SHA `0e1cc5de...`。

### Sigmoid-threshold instance-adaptive schedule

UT seed-1 在预注册的熵阈值/温度族中锁定 `sigmoid_8_0.50`，仍只对
graph-unseen pair 在 inverse-pair 与 soft-selective residual 之间调度。fixed
test 达到 **+6.444 AUC / +7.263 HM 点**，3/3 source seeds 两项指标均正，且
`test_search_after_lock=false`。该结果替代原 `unseen_entropy_1` 作为当前 UT
seed-1 的 strongest instance-adaptive candidate；原候选保留为 power-law 对照。

同一预注册 schedule family 在 UT 原始 checkpoint 选择 `hard_0.7`，fixed test
为 **+5.324 AUC / +5.624 HM 点**，3/3 source seeds 两项均正。这支持
instance-adaptive selective transport 的机制级迁移，但阈值仍按 checkpoint
分别锁定，不应写成统一超参。

MIT seed-1 的同 family 复放选择 `reverse_hard_0.9`，fixed test 为
**+2.074 AUC / +5.250 HM 点**，3/3 source seeds 两项均正。该结果与 constant
soft route 基本持平，放入 cross-dataset schedule ablation，不作为统一阈值结论。

C-GQA 在同一 graph-support-conditioned entropy family 下 validation 锁定
`reverse_hard_0.8`，fixed test **+0.002306 AUC / +0.019635 HM 点**，
validation **+0.005533/+0.025405**，`test_search_after_lock=false`。source
audit 为 1/1 AUC、1/1 HM 正；该结果是有限 preliminary transfer evidence，不能
与 UT/MIT 三源结果合并成多源置信区间，也不改变 open-world 边界。

在同一 frozen route residual 上，structural-risk selective transport 将 CAMS entropy
与 graph support deficiency 联合为无标签 gate；validation 锁定
`beta=.5, slope=8, threshold=.8`，fixed test **+0.002825 AUC / +0.021187 HM**
（+0.283/+2.119 点），`test_search_after_lock=false`。该结果优于 C-GQA 的
entropy-only selective replay，但仍只有单 source seed，论文中应标为 preliminary
transfer evidence，不与 UT/MIT 三源结果合并。

补充的 source-level audit 在冻结候选下得到 3/3 positive test AUC 与 3/3 positive
HM，mean **+0.266/+1.640 点**；但两个 generation-source pair 的 validation AUC
未过统一 gate，故只作为 robustness supplement，不宣称严格同协议多源置信区间。

联合 schedule-strength 网格的最优 validation 候选在 fixed test 仅为
**+4.393 AUC/+4.669 HM 点**，低于 `sigmoid_8_0.50`，因此不作为主结果；
它作为“强度与可靠性调度不能任意联合放大”的消融边界。

support-conditioned threshold 的 validation-only 复核未超过统一 sigmoid
threshold（最佳 `+0.011242 AUC/+0.034520 HM`），因此不进行 fixed test；论文主线
使用统一 entropy threshold，并把 support-conditioned 版本作为未通过门控的边界。

### Topology-gated final audit record (2026-08-25)

最终审计固定引用 `output/research/warmflow_cgqa/topology_gated_valonly_seedcheck/results.json`；`test_read=false`，三 seed mean gain **+0.0017742028 AUC / +0.0023309812 HM**。该 validation-only 结果低于 `+0.002` AUC gate，论文中仅作为平衡性正向但 gate-fail 的补充证据，不纳入主结果或 fixed-test 选参。

### Structured-head boundary

UT seed-1 validation-only comparison of a factorized composition head and a
rank-64 visual adapter did not pass the AUC gate: gains were respectively
`+0.000181751/+0.004762293` and `+0.000924596/+0.003694825` (AUC/HM), with
`test_read=false`. This negative control strengthens the mechanism claim that
the improvement comes from maximum-entropy distribution completion plus the
frozen CAMS update, rather than from an arbitrary classifier-head replacement.

### Candidate-conditioned reliability routing

The strongest current extension is a class-conditional route mixer. For each
image/candidate pair, normalize the frozen CAMS pair logit across candidate
classes and use `alpha=sigmoid(-4*(z-tau))` to interpolate the frozen
inverse-support AUC expert and soft-support HM expert; seen pairs retain the
HM route. Validation locks `tau=0` on UT seed-1 and `tau=0.5` on MIT seed-1.
Fixed test gains are **+8.360 AUC/+11.863 HM** and **+3.381/+9.025** points,
respectively. Three explicit source-pair runs are positive in both datasets,
with source means **+7.872/+11.438** and **+3.177/+8.888**. This is the primary
reliability-routing innovation because it is candidate-specific, non-parametric,
and does not add a trainable inference module; the image-only entropy schedule
remains an ablation.

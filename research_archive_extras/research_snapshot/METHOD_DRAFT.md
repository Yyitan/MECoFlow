# MECoFlow 方法章节草案

## 1. 问题动机

现有 CLIP/CAMS 类 CZSL 方法主要学习组合的判别分数，但一个未见组合 `(a,o)` 在训练集中没有视觉实例，因此没有可用于平衡分类器的类内分布。直接从标准白噪声学习到高维冻结视觉空间的运输跨度过长；直接复用 seen residual 又保留了不必要的实例和组合特异结构。我们希望构造一个只编码三条可由 seen 数据支持的知识的源分布：

1. 未见组合在视觉空间中的大致位置可由属性和物体边际效应组合得到；
2. seen 视觉特征相对该位置仍有非零类内不确定性；
3. 除残差均值和总能量外，不对未见组合的协方差方向作无法验证的假设。

MECoFlow 因而把问题拆成“先定位、再补不确定性、最后短程运输”：加性组合先验给出位置，最大熵残差给出最小假设随机源，条件 rectified flow 把该源运输到 CAMS 视觉特征流形。

## 2. 加性组合位置

设冻结 CAMS composition 特征为 `x_i∈R^d`，训练标签为 `y_i=(a_i,o_i)`，均经过 L2 归一化。先计算每个 seen pair 的图像数加权质心 `c_ao`，并拟合二因素加性模型

`mu_ao = mu_0 + u_a + v_o`。

实现通过 30 次交替加权边际更新（backfitting）近似求解加权最小二乘问题

`min_{mu_0,{u_a},{v_o}} sum_(a,o) n_ao ||c_ao - mu_0 - u_a - v_o||_2^2`，

其中只对 seen pair 求和，并对属性/物体效应作中心化约束。最终对 `mu_ao` 做 L2 归一化。因为 CZSL 假设测试组合的属性和物体原语都在训练中出现，该位置可对 unseen pair 直接计算，不使用 unseen 图像。

## 3. 最大熵残差源

对每个训练实例定义预归一化残差 `r_i=x_i-mu_(a_i,o_i)`，并估计

`m_r=(1/N)sum_i r_i`，

`E_r=(1/N)sum_i ||r_i-m_r||_2^2`。

令 `sigma^2=E_r/d`，采样

`z=m_r+sigma epsilon, epsilon~N(0,I_d)`。

组合 `(a,o)`、随机尺度 `s` 下的 flow 源点为

`x_0=Norm(mu_ao+s z)`。

训练时 `s` 从 `{0,0.5,1}` 均匀采样，使速度场同时观察确定性中心、局部和完整能量源；主版本生成时使用 `s=1`。当前实现用固定随机种子先物化与训练 residual 数量相同的高斯 bank，再 bootstrap 采样，以保证 empirical/diagonal/isotropic 消融除分布假设外完全一致。

### 定理 1：给定均值和总能量的最大熵残差

在所有满足 `E_q[z]=m_r` 且 `E_q||z-m_r||_2^2=E_r` 的 `R^d` 连续分布 `q` 中，

`q*(z)=N(m_r,(E_r/d)I_d)`

具有最大微分熵，且

`h(q)<=(d/2)log(2 pi e E_r/d)`。

证明：记 `Sigma=Cov_q[z]`，则 `tr(Sigma)=E_r`。固定协方差时，高斯分布最大化微分熵，所以

`h(q)<=(1/2)log((2 pi e)^d det(Sigma))`。

设 `lambda_j` 为 `Sigma` 的特征值。由算术–几何平均不等式，

`det(Sigma)=prod_j lambda_j<=(sum_j lambda_j/d)^d=(E_r/d)^d`，

等号当且仅当 `Sigma=(E_r/d)I_d`；结合固定协方差下的高斯等号条件即得。

边界必须写清：该定理作用于归一化前的残差 `z`。`x_0=Norm(mu+s z)` 是它经平移、缩放和球面归一化后的 push-forward，不能称为“球面上的最大熵分布”。

## 4. 条件 rectified flow

对训练目标 `x_1=x_i` 和上述源点 `x_0`，采样 `t~U[0,1]`，定义线性概率路径

`x_t=(1-t)x_0+t x_1`，真实条件速度为 `u_t=x_1-x_0`。

速度网络 `v_theta` 接收 `x_t,t` 以及冻结 CAMS 的属性、物体和 pair prompt：

`v_theta(x_t,t,e_a,e_o,e_ao)`。

三个 prompt 分别线性投影到条件空间后相加，时间使用 sinusoidal embedding；主干为四个条件 residual blocks。flow matching 损失为

`L_FM=E||v_theta(x_t,t,e_a,e_o,e_ao)-(x_1-x_0)||_2^2`。

为使预测终点保留组合判别性，计算

`x_hat_1=Norm(x_t+(1-t)v_theta(...))`，

并用所有 seen pair 真实质心构成 cosine classifier，加入

`L_aux=CE(20 x_hat_1 C_seen^T,y)`。

总损失 `L=L_FM+0.05 L_aux`。

训练采样器按 pair 频率的倒数重采样，因此每个 seen composition 对速度场贡献近似平衡。主版本不启用 primitive auxiliary loss、CFG 或局部 residual bank；这些扩展已在 C-GQA 诊断中排除。

## 5. 未见分布生成和分类器

对每个 unseen pair 采样 `M=32` 个最大熵源点，并用 `K=8` 步显式 Euler 积分

`x_(k+1)=x_k+(1/K)v_theta(x_k,k/K,e_a,e_o,e_ao)`。

最终对输出 L2 归一化。seen 类从真实训练特征中每类平衡抽样 32 个实例；unseen 类使用 32 个生成实例。用 pair text embedding 初始化权重，训练一个全类平衡 cosine classifier

`S_gen(x,c)=exp(gamma)<Norm(x),Norm(w_c)>`。

该分类器不是替换 CAMS，而是估计“若所有 seen/unseen 类都有平衡视觉样本时”的补充分数。

## 6. 文本残差融合与可靠回退

对每张图分别在类别维做 z-score。主版本的 `update` 融合使用

`S_final=S_CAMS+lambda[Z(S_gen)-Z(S_text)]`，

其中 `S_text` 是冻结 CAMS composition feature 与原始 pair text embedding 的相似度。减去文本分数可把生成头中已存在的文本初始化先验消掉，只注入由生成视觉分布学到的更新量。

为降低单次 transport source 的偶然性，UT 的增强版本在固定候选中对三个
diagonal residual source generator 的 z-scored logits 做均值，再计算同一
text-residual update：

`S_ens = (1/3) sum_k Z(S_gen^(k))`,
`S_final = S_CAMS + 1.25 [S_ens - Z(S_text)]`。

这里的 ensemble 是对 transport-source marginalization 的推理近似，不把
generator seeds 当作独立数据集；候选和强度必须在 validation-only cross-fit
后冻结。UT fixed-setting test 的 ensemble 增益为 +1.369 AUC / +1.093 HM
点，逐 seed 固定强度的均值增益为 +1.224/+0.811 点。

候选 `full/update` 和离散 `lambda` 只按验证 AUC 选择，AUC 完全相同时才以 HM 打破平局。三个生成器 seed 使用平均验证 AUC 选择同一设置；若平均验证增益低于 0.002（0.2 AUC 点），正式协议令 `lambda=0` 回退 CAMS。该门控是统一的保守评估协议，不应描述为理论保证或事前注册的阈值。

## 7. 训练与推理算法

```text
Input: frozen CAMS train features/prompts, pair counts, M=32, K=8
1: Fit weighted additive effects mu_0, u_a, v_o from seen pair centroids
2: Compute residual mean m_r and total centered energy E_r
3: Construct isotropic residual source N(m_r, E_r/d I)
4: Repeat for 20 epochs with pair-balanced sampling:
5:     sample seen instance x_1 and scale s in {0,.5,1}
6:     sample x_0 = Norm(mu_y + s z), t ~ U[0,1]
7:     update v_theta with flow-matching + seen-centroid auxiliary loss
8: For each unseen pair, transport M source samples with K Euler steps
9: Combine generated unseen samples and balanced real seen samples
10: Train text-initialized all-class cosine classifier
11: Select one shared validation-only residual fusion across generator seeds
Output: lightweight classifier logits and conservative CAMS fusion
```

部署时不需要加性先验、随机源或 flow；只保留训练好的 cosine classifier，并与 CAMS 分数作两路矩阵乘法融合。

## 8. 贡献表述（可用版本）

1. 提出一个面向 CZSL 未见类分布完成的最大熵组合源：加性原语效应定位类别，残差总能量刻画数据尺度，而各向同性协方差在给定约束下具有严格最大熵解释。
2. 提出从该源到冻结 VLM 视觉特征流形的条件 rectified flow，以训练期生成的未见类分布学习平衡视觉分类器，推理时无需运行生成器。
3. 通过源位置、随机性、分布族、no-flow、运输步数、跨 CAMS checkpoint 和安全回退等受控实验，区分定位、分布采样和连续运输各自的贡献与失效边界。
4. 提出一个可审计的 transport-source marginalization：对异方差 diagonal
   residual transports 做固定 seed ensemble，并在 text-residual 空间注入
   更新；该增强只在有 cross-fit validation 证据时启用。
5. 提出 graph-support-conditioned selective transport：只对 graph-unseen
   pair 注入 source residual，再由冻结 CAMS entropy 进行逐图路由。该模块
   不做 test-time adaptation，不修改 CAMS 参数，也不需要生成时运行 flow。

作为 ensemble 的鲁棒扩展，令 `m=median_k Z(S_gen^k)`，并以 source
dispersion `r_c=mean_k |Z(S_gen^k)_c-m_c|` 构造共识门控
`g_c=(1+tau*r_c/median(r))^{-1}`，再按每个样本的类别均值归一化。最终更新为
`S_final=S_CAMS+lambda*g*(m-Z(S_text))`。这是一种不增加 flow 参数的
source-agreement 控制：不确定类别的 transport 更新被压低，而整体更新能量保持。
在 UT 上，预声明网格并按 AUC gate 后的 validation HM 选择 `tau=1,lambda=.75`
得到 fixed test **+1.111/+1.046 AUC/HM 点**；它低于 diagonal mean 的峰值，
所以应作为可审计机制消融，而不是主结果替代。

进一步的 support-aware 变体把训练图像支持度作为第二个可靠性先验。令
`q_p` 为组合 pair 的 harmonic attribute/object image support，经 `log1p` 和
全局归一化后得到 `q`，令 `g_agree` 为上式 source-agreement gate，则
`g=Normalize(0.5*g_agree+0.5*q^{0.25})`，最终仍使用
`S_final=S_CAMS+lambda*g*(m-Z(S_text))`。该 gate 不引入 flow 参数，也不访问
validation/test 标签以外的统计；在预注册的 `tau=2, lambda=.75` 候选上，UT
validation-only 为 **+0.012894/+0.009910 AUC/HM**，fixed test 为
**+1.305/+1.232 AUC/HM 点**，6/6 source-level 指标为正。它提高 HM 而不超过
diagonal mean 的 AUC，因此作为平衡性机制证据，不作为主设置替代。

在独立 UT seed-1 checkpoint 上，原始 harmonic-support + inverse-consensus
候选的 validation AUC 为负，因此不把同一参数硬套到另一 checkpoint；同一
预声明网格改选 support-only gate 后仍得到 **+1.861/+0.969 AUC/HM 点**，
3/3 source-level 指标为正。该结果支持“支持度是可迁移的可靠性信号”，同时
也说明 gate 细节必须由每个冻结 checkpoint 的 validation protocol 确定，不能
伪装成跨 checkpoint 的统一置信区间。

MIT seed-1 上同一类 support-only gate 也保持正 AUC（**+0.276 AUC 点**），
但 HM 仅 **+0.026** 且 source-level HM 为 2/3 正，因此该分支只作为 AUC/
reliability 补充，不替代 MIT source-median 的平衡结果。

支持度还可以分解为 attribute/object 边际项。UT 上 object-side support 与
mean source aggregation 的 factorized gate 达到 **+1.470/+0.953 AUC/HM 点**，
6/6 source-level 指标为正；它略提高 AUC 却降低 HM，说明支持度先验确实参与
可靠性控制，但不能只按单一指标选 gate，正文应把 harmonic 版本作为平衡主消融。

另一条互补机制是 confidence-aware transport：以 frozen CAMS 图像的 entropy
或 top-2 margin 生成逐样本 gate，再乘到 `m-Z(S_text)` 更新上。UT 的低
base-entropy gate 达到 **+1.506/+1.029 AUC/HM 点**，MIT seed-1 的高
base-entropy gate 达到 **+0.202/+0.164**；两者均不增加 flow 参数。它说明
transport 可以同时利用数据支持度与当前分类器不确定性，但 confidence gate
仍是机制消融，不替换 support-aware 平衡主设置。

禁止写成“首个生成式 CZSL”“首个 flow matching CZSL”“各向同性显著优于对角高斯”或“通用 open-world/C-GQA 提升”。

## 9. 审稿人可能质疑及当前回答

- **最大熵高斯没有显著超过 diagonal。** 这正是最小假设主张：它只用一个总能量标量，在 MIT/UT 不损失性能且 UT 种子方差略低；创新不依赖声称 isotropic 更准。
- **归一化后已不是高斯。** 定理只约束预归一化 residual，正文必须明确 push-forward 边界。
- **为何不用 diffusion？** 冻结特征空间、有限 seen pairs 与两张 A100 条件下，rectified flow 提供直接速度监督和 8 步生成；DIFFCZSL 则是外部 diffusion representation regularization，研究对象不同。
- **是否只是加性 prototype？** UT 的同 flow `s=0` 比随机源低 0.945 AUC/0.816 HM；无 flow 源增强反而低于 CAMS 2.774 AUC。
- **是否只是 flow 结构带来参数量？** 参数完全匹配的 direct endpoint regressor 只达到 43.934±0.111 AUC / 56.620±0.196 HM；8-step rectified flow 额外高 +1.105/+0.809，且 no-flow 源增强更差。因此增益不能由 MLP 容量本身解释。
- **是否依赖一个 checkpoint？** 原始 MIT/UT 和各自独立 seed-1 checkpoint 的平均 AUC都为正；两个 seed-1 均通过验证门控。
- **是否由生成器随机种子碰巧驱动？** 四个冻结 CAMS checkpoint 的 12 个生成器运行全部获得正 AUC 增益，逐 checkpoint 的 95% seed-t 区间也均在零以上；但这些种子共享数据，证据只支持优化/采样稳定性，不冒充数据集总体显著性。HM 为 11/12 正，因此主张聚焦 AUC。
- **与 FlowComposer/WARM-CAT 是否重复？** FlowComposer 联合训练 primitive flows 和 Composer，将真实视觉点运输到属性/物体文本端点并在推理时组合速度；WARM-CAT 在测试流上在线更新文本/视觉原型。MECoFlow 不训练 Composer、不访问测试流、不在线改原型，而是从 seen composition graph 推断 unseen visual distribution，离线完成 source-to-visual-manifold transport，随后只保留轻量 classifier 和冻结的可靠性路由。因此 entropy threshold 是无测试时适应的可靠性调度，不是 test-time adaptation。
- **C-GQA 的边界是什么？** raw distribution-completion 分支仍因 validation gate 回退；entropy 与 graph-support deficiency 联合的 structural-risk selective replay 在主 source pair 上取得 **+0.283 AUC/+2.119 HM 点**，冻结候选的三个显式 source pair 复放均为正（mean **+0.266/+1.640**），但其中两个 generation-source pair 的 validation AUC 未过 gate。因此这仍是 preliminary transfer/robustness evidence，而不是严格同协议多源或通用 C-GQA/open-world 结论。

### Reliability-conditioned transport (mechanism ablation)

The transport update can be conditioned on two independent reliability signals:
the graph-support prior of the target composition and the confidence of the frozen
CAMS image classifier. For target pair (p=(a,o)), we compute a normalized support
score (s_p) from seen-image counts of its attribute/object primitives. For image
(x), we compute a normalized confidence score (c_x) from the base-logit margin or
entropy. The residual update is then

\[
\Delta z_{x,p}=\lambda\,g(s_p,c_x)\,[q_{x,p}-t_{x,p}],
\]

where (g) is a predeclared product or linear combination of powered scores. This
keeps the graph prior and image uncertainty explicit rather than hiding them in a
learned post-hoc reweighting. A validation-only gate selects the largest HM among
candidates with AUC gain at least 0.002; the resulting fixed tests are reported as
mechanism evidence, not as a replacement for the main support-aware setting.
The independent UT seed-1 replay is intentionally retained as a negative-transfer
control (validation positive, locked test negative), so the gate is not presented
as a universally transferable heuristic.

### Non-parametric source posterior

Rather than treating all generated source logits as equally reliable, we compute a
per-image posterior over sources. Let (r_{s,x}) be the normalized residual score
from source (s), and let (m_x) be the median or mean source center. We define

\[
w_{s,x}\propto\exp\{-\tau,d(r_{s,x},m_x)\}\,c_{s,x}^{\gamma},
\qquad \bar r_x=\sum_s w_{s,x}r_{s,x},
\]

where (d) is an absolute or RMS disagreement and (c_{s,x}) is a source-level
margin or inverse entropy confidence. The posterior is non-parametric and computed
offline from frozen logits; it therefore adds no trainable inference module. The
validation gate selects one setting before test access, and the MIT seed-1 result
provides positive balanced evidence while the UT result is retained as a lower-gain
cross-check rather than a universal claim.
The MIT seed-1 candidate also replays positively on the original MIT checkpoint,
which is useful cross-checkpoint evidence but does not justify a dataset-independent
confidence interval.

An optional structural conditioner multiplies this image-level posterior by a
normalized seen-graph support score for the target pair. This separates two
uncertainties—whether the generator source is reliable for the image and whether
the composition is structurally supported. The controlled replay shows why this
factorization is reported as an ablation: it is positive on MIT seed-1 but negative
on UT after a much stronger validation gain.

### Graph-support-conditioned selective transport

The source-family residual should not be injected uniformly into every image and
pair. Let `q_p` be normalized train-image support for composition `p`, and let
`u_x` be the entropy of frozen CAMS logits for image `x`, normalized using
validation statistics only. We retain two frozen residual experts: an
inverse-pair route `d_A` and a soft support-tapered route `d_B`. For graph-unseen
pairs we use `alpha_x = u_x^r`, while seen pairs retain the frozen CAMS score:

`S_final(x,p) = S_CAMS(x,p) + (1-alpha_x) d_A(x,p) + alpha_x d_B(x,p)`.

The sigmoid threshold is selected on validation, entropy normalization is frozen
before test access, and all source logits are generated offline. Thus this is an
instance-level reliability schedule, not a learned test-time gate. Separate UT
validation locks give `sigmoid_8_0.50` on the independent checkpoint and `hard_0.7`
on the original checkpoint; fixed gains are **+6.444/+7.263** and
**+5.324/+5.624 AUC/HM**, respectively, with 3/3 source seeds positive for both
metrics. MIT seed-1 selects `reverse_hard_0.9` and reaches **+2.074/+5.250**.
These are checkpoint-conditioned replays, not one universal hyperparameter claim.

### Candidate-conditioned reliability routing

The global image entropy gate still assigns one route weight to every candidate
pair. We replace it with a class-conditional reliability map derived from the
frozen CAMS score itself. Let
`z_{x,p} = (l^CAMS_{x,p} - mean_p l^CAMS_{x,p}) / std_p l^CAMS_{x,p}`.
For graph-seen pairs we retain the HM-specialist route; for unseen pairs we use

`alpha_{x,p} = sigmoid(kappa (z_{x,p} - tau))`,

and mix the two already-frozen experts as
`Delta(x,p) = (1-alpha_{x,p}) d_A(x,p) + alpha_{x,p} d_B(x,p)`.
With `kappa=-4`, high-confidence candidate scores favor the inverse-support
AUC expert, while low-confidence candidates favor the support-tapered HM
expert. This is a non-parametric, candidate-specific reliability router: it
adds no trainable parameters, sees no test labels, and only reuses the frozen
CAMS logits plus offline source logits. Validation fixes one `(kappa,tau)` per
checkpoint before test access. It yields **+8.360 AUC/+11.863 HM** on UT seed-1
and **+3.381/+9.025** on MIT seed-1; all three explicit source pairs are
positive on both metrics in both checkpoints. The corresponding source means are
**+7.872/+11.438** and **+3.177/+8.888** points. This replaces the weaker
image-only entropy schedule as the primary reliability-routing innovation,
while keeping the entropy route as an ablation.

### Structural-risk selective transport

Entropy alone measures image-level ambiguity but cannot distinguish a difficult
image from a structurally unsupported composition. We therefore define a frozen,
label-free risk score for graph-unseen pairs:

`r(x,p) = Normalize(u_x + beta * (1 - q_p))`,

where `u_x` is validation-normalized CAMS entropy and `q_p` is normalized training
support of the target pair's attribute/object primitives. The route mixture is
`alpha_x = sigmoid(kappa * (tau - r(x,p)))`; seen pairs retain `alpha=1`, while
unseen pairs interpolate the inverse-support AUC expert and the soft HM expert.
This is a structural reliability conditioner, not test-time adaptation and not a
new trainable module. The preregistered C-GQA replay locks `beta=.5, kappa=8,
tau=.8` and reaches **+0.283 AUC / +2.119 HM points** on fixed test, while the
same risk family remains below the already locked settings on UT/MIT; this
checkpoint-specific contrast is retained as evidence for the mechanism boundary.
- **为何融合阈值是 0.002？** 它是所有数据集/seed 统一使用的保守门槛，用于拒绝弱验证信号；不能包装为理论最优，需在补充材料报告阈值敏感性或固定阈值下的全部结果。

完整阈值审计已由 `research/gate_sensitivity.py` 从归档 JSON 生成到 `research/GATE_AUDIT.md`：closed-world 的 0.10–0.25 AUC 点阈值给出相同主决策，但 UT open-world 是无法由该标量门控修复的反例。

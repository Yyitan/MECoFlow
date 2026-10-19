# Reverse Primitive-Flow Generation

更新时间：2026-08-24

## 1. 研究问题

FlowComposer 学习从真实视觉组合特征到组合文本端点的属性/对象速度场，并在
正向推理中把这些场用于判别。这里研究一个不同问题：冻结这些已学习的判别式
速度场，是否能从组合文本端点反向积分，生成可迁移的未见组合视觉类支持？

该问题与 MERF 正交。MERF 从 ANOVA 可组合中心附近的匿名随机源学习新的条件
运输；Reverse Primitive-Flow Generation 不重新训练运输模型，而是把已有的
视觉→文本 primitive fields 反向解释成一个零额外训练参数的生成器。

## 2. 方法

设归一化组合文本端点为 `e_(a,o)`，FlowComposer 的属性和对象速度场分别为
`v_a(z,t)` 与 `v_o(z,t)`，Stage-2 Composer/gater 为 `C`。在文本端点的切空间
中采样各向同性方向：

```text
u = normalize((I - e e^T) epsilon),  epsilon ~ N(0, I)
z_K = normalize(e + sigma u)
```

在固定角半径约束下，`u` 是切空间球面上的均匀方向；这给出不携带 seen-pair
身份的最大熵方向扰动。随后用冻结速度场做反向 Euler 积分：

```text
t_k = k / K
v_k = C(v_a(z_k, t_k), v_o(z_k, t_k))
z_(k-1) = normalize(z_k - v_k / K)
```

每个候选组合生成 `M` 个样本，归一化均值作为未见类原型。seen 类仍使用真实
训练图像中心，避免为了形式统一而丢弃可靠监督。最终可单独使用生成原型分类，
也可在验证集预先冻结权重后与官方组合 logits 做 z-score 融合。

## 3. 可检验机制

1. **反向可用性**：若正向场确实学习了视觉到语义端点的连续对应，反向原型应
   比原始文本原型更接近真实未见类中心，并提高 AUC/HM。
2. **原语组合必要性**：去掉 Stage-2 gate、直接相加两个速度场，应削弱平衡
   迁移收益。
3. **局部熵而非任意噪声**：小切空间扰动可以覆盖类内方向并改善 AUC；过大
   扰动应破坏语义并使收益消失。
4. **多步动力学必要性**：在完整 checkpoint 上比较 1/2/4/8/16 步；若一步已
   达到相同效果，则不能把连续反向运输作为贡献。

## 4. 与现有工作的精确边界

- 不能主张首次将 flow matching 用于 CZSL；FlowComposer 已覆盖该点。
- 不能把属性/对象流、Composer、endpoint contrastive loss 当作新模块；它们
  都来自冻结的官方模型。
- 新操作是把原本用于视觉→文本判别推理的 primitive fields 反向积分，从组合
  文本端抽取视觉类支持，并用身份匿名的切空间扰动表达多样性。
- 与普通文本原型、无 gate 的速度场相加、正向官方 logits、直接端点映射及
  MERF 独立生成器的对照必须同时存在。

在论文叙述中，这应被称为“discriminative-to-generative field inversion”或
“reverse primitive-flow support generation”，而不是新的 FlowComposer 变体。

## 5. 已有 validation-only smoke 证据

官方 UT-Zappos 1-epoch Stage-1/Stage-2 smoke checkpoint，固定图像特征缓存，
32 samples/class，8 个反向步。最可靠的结果是同一缓存内相对文本原型的配对
差值：

| Evidence | 配对 Val AUC 增益 | 配对 Val HM 增益 |
|---|---:|---:|
| reverse, gate, sigma=0 | +0.605 | **+0.910** |
| reverse, gate, sigma=0.05，3 generation seeds | **+0.770 +/- 0.015** | +0.757 +/- 0.000 |
| reverse, no gate, sigma=0.05 | 约 +0.37 | 约 -0.35 |
| reverse, gate, sigma=0.4 | 约 -0.005 | 约 +0.47 |

这只支持继续实验，不构成正式结果。1-epoch 官方 logits 很弱，当前融合提高
AUC 但降低 HM，不能用于主张整体模型提升。

一次缓存重建审计发现官方 train transform 含随机 flip/perspective/rotation，
会改变 seen 训练中心并导致跨缓存绝对指标漂移。脚本已改为用 deterministic
validation transform 提取 train evaluation features，并把该选择写入 provenance。
正式结果只允许使用修正后的确定性缓存；旧缓存仅保留其内部配对差值。

## 6. 冻结验证协议

1. 等待完整官方 Stage-1 和 Stage-2 checkpoint；不得用 smoke 结果读取 test。
2. 完整模型先只跑 validation，候选 `sigma={0,0.05}`；反向步数先固定为 8。
3. 选择规则：优先要求 AUC 与 HM 均不低于文本原型，再在可行候选中最大化
   AUC；若无候选同时满足，则该数据集关闭反向生成。
4. 融合必须分别报告 validation-best-AUC 与 validation-best-HM 权重，并保存
   完整权重网格；最终主表只采用预注册的一套规则。
5. 至少两个独立官方训练种子方向一致后，才允许冻结一次 test 评估。
6. test 后不得再修改 `sigma`、反向步数、样本数或融合权重。

## 7. 论文级最低证据

- UT-Zappos、MIT-States、C-GQA 三数据集；至少三个种子，均值与标准差。
- 文本原型、无 gate、单步、多步、不同噪声半径、不同样本数、正向官方模型、
  MERF 和直接端点生成对照。
- 报告生成原型到真实中心的 alignment、类间 margin、seen/unseen 校准、AUC、
  best HM、运行时间与显存。
- 证明收益不是只来自把真实 seen 中心与文本 unseen 原型混合的 evaluator：所有
  对照共享相同 seen 中心、候选集和偏置扫描。

## 8. 风险与否决条件

- FlowComposer 场只在训练耦合路径附近被监督，从文本端反向可能离开训练支持。
- Stage-2 gate 在正向目标上训练，未必保证反向 ODE 稳定或可逆。
- 固定半径球面噪声不是任意协方差高斯；论文必须准确描述为切空间最大熵方向，
  避免把实现包装成未实现的 moment-matched Gaussian。
- 若完整 checkpoint、独立种子或第二数据集不能同时提高 AUC/HM，本扩展降级为
  分析实验，不作为主贡献。

## 9. Epoch-9 诊断后的方法修订

较成熟的 epoch-9 checkpoint 否定了“完整反向生成”作为主方法：完整反向积分
相对文本原型下降 `0.946 AUC / 0.522 HM` 点，无 gate 时下降更大。深度扫描表明
收益只存在于文本端点附近的窄边界层，`t=1 -> 0.99` 为正，继续反向则逐渐变差；
加入切空间噪声也单调降低 AUC。因此，第 2 节的完整随机反向生成保留为被否定的
科学假设，不再作为候选贡献。

修订后的候选称为 **Trust-Region Primitive-Field Correction (TR-PFC)**。它只在
组合文本端点计算一次冻结场，不声称“partial flow”本身是创新（官方
FlowComposer 已在正向推理中使用 partial integration）：

```text
v = C(v_a(e, 1), v_o(e, 1))
v_tan = (I - e e^T) v
r(e) = min(alpha ||v_tan||, r_max)
e' = normalize(e - r(e) normalize(v_tan))
```

这里的新操作边界是：把判别式视觉→文本 primitive field 在**文本端局部切空间**
解释为原型校正方向，并用 Composer 输出范数形成组合自适应、上界受控的角步长。
它不是新的 flow 训练目标，也不是一般意义的 partial flow。

epoch-9 validation-only 网格冻结 `alpha=0.015, r_max=0.02`：相对文本原型
`+0.1745 AUC / +0.1734 HM` 点。相同参数去掉 gate 后只剩 `+0.0659 AUC`，且
100% 未见组合触发上限；有 gate 时只有 10% 触顶。这一对照支持 Composer 的
作用是校准组合级步长，而非仅提供一个统一方向。

后续协议随之修订：固定无噪声、单次文本边界校正、`alpha=.015`、`r_max=.02`、
不融合官方 logits；原样迁移到 final-epoch Stage-1 与预注册正式 Stage-2 gate 的
validation。只有独立训练种子和第二数据集同时保持 AUC/HM 正向，才进入一次性
test。epoch-9/one-epoch gate 仅是机制诊断，不改变正式 checkpoint 规则。

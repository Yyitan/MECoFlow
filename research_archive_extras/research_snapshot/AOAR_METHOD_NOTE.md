# AOAR-Replay：文献动机、方法与论证边界

## 定位

AOAR-Replay 是一条与 MECoFlow 互补的非扩散生成路线。它保留 CAMS 的冻结视觉空间与部署形式，只在训练期学习 seen composition residual 的轻量 any-order autoregressive decoder，再离线生成 unseen composition 特征用于平衡分类器训练。

这条路线回答一个不同于 MECoFlow 的问题：如果组合残差可以被拆成多个连续 latent token，固定的 PCA token 顺序是否会诱导不必要的局部条件捷径，而随机顺序训练能否让生成器学习更对称、更可组合的残差分解？

## 从近期工作中可迁移的证据

### Order-marginalized autoregressive classification

近期 any-order AR generative classifier 明确使用 position-instruction token，并通过多个随机 token order 聚合条件似然。其实验显示固定 raster order 更容易依赖局部判别线索，而 order marginalization 改善分类和分布偏移鲁棒性；其推理相对 diffusion classifier 更快。

AOAR-Replay 只迁移以下结构原则：

- 用显式 position instruction 指定当前需要预测的 latent token；
- 每个训练样本随机采样 token permutation；
- causal mask 保证预测当前 token 时看不到其值；
- unseen replay 时也随机采样生成顺序。

不迁移图像级 VQ tokenizer、ImageNet 条件预训练或用生成似然直接遍历类别做分类。原工作的效率结论来自大型图像生成器，不能作为当前特征生成器的实测效率结论。

any-order 建模本身不是新概念。已有工作指出，穷举式 AO-ARM 会用多条分解路径重复定义同一联合分布；2026 年的进一步分析还指出，长序列单流 any-order attention 存在结构邻近与语义邻近竞争。当前 pilot 的序列只有 16 个连续 token，并把 position instruction 与 value 放在不同序列位置，因此先用最小实现验证信号；若失败，不能仅凭理论预期宣称 random order 有效，也不会在同一 test 上临时改成复杂 two-stream 模型。

### Generation is required for data-efficient perception

该工作把生成式 compositional generalization 写成对 decoder 的反演，并研究 online search 与 offline generative replay。它为“生成 unseen latent combinations 后训练识别器”提供了清晰的上层动机，也强调 decoder 的结构性归纳偏置会决定回放是否有效。

AOAR-Replay 对应 offline replay，但不是该论文图像 VAE 的复现：这里的 decoder 只生成冻结 CAMS 特征中的低维 residual，组合中心由属性与物体的加性统计先验提供，AR 只需建模 composition-specific variation。

## 方法

对 seen 图像特征 $v$ 和 pair $p=(a,o)$，先估计加性组合中心 $\mu_p$，再定义残差

$$r=v-\mu_p.$$

在 seen residual 上拟合 PCA，取前 $d=128$ 维并分成 $N=16$ 个 8 维连续 token。给定属性、物体与 pair prompt，Transformer 输入为

$$[c, q_{\pi_1}, z_{\pi_1},\ldots,q_{\pi_N},z_{\pi_N}],$$

其中 $q_{\pi_i}$ 是显式位置指令，$\pi$ 对每个样本独立随机。模型在 $q_{\pi_i}$ 位置预测随后连续 token 的 Gaussian mean/log-scale，并最小化 Gaussian NLL。严格 causal mask 使 $z_{\pi_i}$ 在自身预测时不可见。

对每个 unseen pair，模型采样 32 个随机顺序和连续 residual，PCA 逆变换后加回 $\mu_p$ 并归一化。seen/unseen 各 pair 等量回放用于训练文本初始化余弦分类器，最终仍使用 CAMS logits 与生成分类器更新项的 validation-selected fusion。

## 计算可行性

- 16 个 token、33 个输入位置；
- hidden 256、4 层、8 heads；
- 单卡训练与离线生成；
- 不训练像素生成器，不更新 CAMS backbone；
- 生成器部署时丢弃。

这与原图像级研究的计算量不是一个量级，但具体训练时间、显存和收益必须由当前实现实测，不能先验声称。

## 可证伪预测与必要消融

如果 order-symmetry motivation 正确，则 random-order 应在共享 validation 选择下优于 fixed raster order。连续 token 噪声是否必要则由 stochastic sampling 与条件均值对照直接证伪：若条件均值更优，就不能把 Gaussian sampling 写成收益来源。若 random-order 与 raster parity，AOAR 只能作为普通特征 AR replay，不能把 order symmetry 写成核心贡献。

首轮 UT validation-only 采用预注册的 +0.2 AUC 点门槛。未过门即停止且不读 test；过门后才补三个生成种子、固定共享融合参数，并做 random-vs-raster、mean-vs-sampling 对照。

## 当前实证结论

主 random-order + Gaussian-sampling 版本通过共享 validation gate，固定设置后的 test AUC 三种子均为正，平均 +0.620 点；但 HM 平均 −0.347 点且 0/3 正，因此它尚不是平衡提升。

在严格 validation-only 消融中，random-order 的平均增益与稳定性均优于 raster：+0.760 vs +0.436 AUC 点，且共享设置逐 seed 正增益为 3/3 vs 1/3。该结果支持顺序随机化的稳定性动机，但 raster 均值仍为正，不能写成固定顺序完全无效。

移除 Gaussian token noise 后，conditional-mean 版本的共享 validation 增益提高到 +0.924 点。冻结该选择后，test AUC 为 44.466±0.065（+0.708±0.065，3/3 seed 正），HM 为 56.376±0.359（−0.180±0.359，仅 1/3 正）。由此，当前证据不支持“连续 token 随机采样是 AOAR 收益来源”；random permutation 已提供顺序多样性，去噪提高了 AUC 并减小 HM 降幅，但没有解决 seen/unseen 平衡。

## 论证边界

- 不声称 any-order AR 原论文已经验证 CZSL；它验证的是图像级 generative classification。
- 不把另一篇论文的 theorem 直接套到 CAMS 特征空间；其生成器可逆性、slot 交互和数据机制假设在本项目中尚未建立。
- 不声称 AOAR 比 diffusion/flow 更快，除非在同一缓存、同一硬件和同一评估协议下实测。
- 即使真实 GPU validation 通过门槛，只要固定 test 的 HM 未形成稳定正增益，AOAR 仍只作为补充探索，不进入论文主结果或摘要。

## 截至 2026-08-24 的直接新颖性检索

使用 arXiv、CVF Open Access 的 `compositional zero-shot + autoregressive / any-order / generative replay / feature generation` 组合检索，未找到把 any-order continuous residual AR 用于标准 attribute-object CZSL 的公开论文。检索结果中的 AR 工作研究一般生成分类或图像生成；CZSL 工作主要是判别式 prompt/条件概率方法、FlowComposer 的 flow transport，以及 DIFFCZSL 的扩散表征正则。

这只能支持“未发现直接先例”，不能证明绝对首创。论文最终应写成据我们检索所知，并在投稿前再次复核最新 arXiv。

相关一手来源：

- https://arxiv.org/abs/2205.13554
- https://arxiv.org/abs/2602.16092
- https://arxiv.org/abs/2608.19871

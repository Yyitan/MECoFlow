# MECoFlow / CAMS 生成式 CZSL 实验记录

更新日期：2026-08-24

## 1. 当前主方法

MECoFlow（Maximum-Entropy Compositional Flow）在冻结的 CAMS 视觉特征空间中完成未见组合的分布补全：

1. 用两因素加性模型从已见组合质心估计属性效应与物体效应，得到未见组合先验 `mu(a,o)`；
2. 从真实残差总能量估计标量尺度 `sigma`，构造最小假设的最大熵随机起点 `x0 = normalize(mu(a,o) + s*(mean(r)+sigma*epsilon))`，其中 `epsilon~N(0,I)`；
3. 用原语条件 rectified flow 将 `x0` 运输到真实组合特征 `x1`；
4. 为未见组合生成平衡样本，并训练文本初始化的余弦分类器；
5. 在验证集上选择生成分数与 CAMS 分数的融合方式；生成器只在训练阶段运行，推理仅增加一次轻量线性分类分数。

它与 DIFFCZSL 不同：DIFFCZSL 用预训练 Stable Diffusion / CleanDIFT 特征蒸馏正则化真实图像表征；MECoFlow 不调用图像扩散模型，而是在任务特征空间显式生成未见组合的视觉分布。它与 FlowComposer 也不同：FlowComposer 学习视觉点到文本点的原语速度场；MECoFlow 学习最大熵组合源分布到视觉组合分布的短程运输。

最接近的生成式先工作是 2026 KBS 的 learnable primitive disparity（LPD）：LPD 依靠 metric-learning triplet 与可学习类差异合成特征；MECoFlow 的区别不能只写成“生成未见特征”，而应强调最大熵组合源、连续 rectified transport、以及对每个未见组合进行分布采样。受控消融已证明经验非高斯高阶结构不是必要因素，因此不再作该主张。

## 2. 固定评估协议

- 所有融合超参数只由验证 AUC 选择，HM 仅用于 AUC 完全相同时的次级排序；
- 三个生成器随机种子使用同一个、按平均验证指标选择的融合超参数；
- 验证 AUC 增益低于 0.002 时自动回退 CAMS，防止弱验证信号在测试集负迁移；
- 快速闭世界 evaluator 已与 CAMS 原 evaluator 做过逐项等价验证；
- 当前三种子改变 flow、生成采样和生成分类器随机性；冻结的 CAMS checkpoint 保持一致。

## 3. 闭世界主结果（统一三种子选参）

| 数据集 | CAMS AUC | MECoFlow AUC | ΔAUC | CAMS HM | MECoFlow HM | ΔHM | 统一设置 |
|---|---:|---:|---:|---:|---:|---:|---|
| MIT-States | 23.124 | 23.416 ± 0.029 | +0.293 | 39.895 | 40.130 ± 0.073 | +0.235 | update, 0.2 |
| UT-Zappos | 43.758 | 45.038 ± 0.126 | +1.280 | 56.556 | 57.429 ± 0.313 | +0.874 | update, 1.0 |
| C-GQA raw MECoFlow | 16.325 | 16.325 | +0.000 | 34.780 | 34.780 | +0.000 | 可靠性门控回退 |
| C-GQA selective replay | 16.325 | 16.555 | +0.231 | 34.780 | 36.743 | +1.964 | reverse hard entropy threshold 0.8 |
| C-GQA structural-risk selective replay | 16.325 | 16.607 | +0.283 | 34.780 | 36.899 | +2.119 | entropy + support risk, beta .5 |

结果文件：

- `output/research/warmflow_mit_gaussian_iso_shared.json`
- `output/research/warmflow_ut/gaussian_iso_shared.json`
- `output/research/warmflow_cgqa/gaussian_iso_seed0/results.json`（验证增益仅 +0.051 AUC 点，可靠性门控回退）

## 4. 已排除路线

1. 未校准标准白噪声条件 flow：生成分类器条件坍缩；后续同能量零中心受控版本不再坍缩，但仍显著弱于加性组合中心，见第 5 节；
2. 仅文本/加性原型与无 flow 数据增强：验证可出现较大表面增益，但 UT 测试从 43.758 降到 40.984，说明 flow 的作用不是简单增加样本；
3. 条件扩散能量、完整 latent-query 集合去噪：排序可学习，但与 CAMS 融合无增益或负增益；
4. 稀疏 query 去噪：MIT 仅约 +0.059 AUC，效应太小；
5. cross-primitive instance flow：MIT 小幅正增益，UT 验证过拟合且测试崩溃；
6. 语义 ridge/kNN 交互残差补全、因子化分类器、低秩适配器、图平滑：验证选择零权重，或验证正而测试负；
7. C-GQA pairless flow：验证信号消失，说明 pair prompt 提供的信息虽弱但必要；
8. 一步 Euler flow：UT 三种子共享选择的原始测试 AUC 为 44.618±0.093（+0.860），三次均正；但平均验证增益只有 +0.149 AUC 点，低于 +0.2 安全门槛，正式协议回退。它不是“测试不稳定”，而是验证证据不足、弱于 8-step 的 +1.120 验证增益与 +1.280 测试增益；
9. 语义局部残差池：C-GQA 分类器验证仅 +0.066 AUC；生成均值最高约 +0.192，但测试负迁移；
10. 局部 flow 的属性/物体层级辅助损失、生成置信度掩码：均未把 C-GQA 验证增益迁移到测试集。
11. CAMS `glb` 空间 flow 与 `com+glb` 二维融合：C-GQA 的 `glb` 验证增益仅 +0.044 AUC 点，多空间融合最终仍只选择 `com`；
12. 冻结生成分类器的已见类权重：C-GQA 验证增益仅 +0.069 AUC 点，未解决大类别空间负迁移。
13. classifier-free guidance（`cond_drop=0.1`，`guidance=1.0/1.5`）：C-GQA 验证增益分别仅 +0.076/+0.078 AUC 点，均弱于无 CFG 主版本；条件 dropout 与 guidance 放大均无效。
14. 生成原型驱动的共享 composition prompt refinement：`lr=1e-3` 明显过冲，`lr=1e-5` 的最佳验证 AUC 仍下降 0.069 点；共享上下文会同时移动全部组合，容量与选择性不足，正式排除。
15. 类条件低秩 interaction hyper-classifier：在 C-GQA 上采用 class-sampled contrastive 训练后，验证 AUC 增益为 +0.183 点，但原始测试 AUC/HM 分别下降 0.048/0.083 点；更强的类别选择性仍未解决 split 外推。
16. 训练图拓扑门控：验证选择 `min_image_support` 连续门控（+0.203 AUC 点），但测试 AUC 从 16.325 降至 16.035、HM 从 34.780 降至 34.387；原语度数/图像支持度无法预测生成器在新组合上的可靠性。
17. leave-composition-out marginal prior：训练起点对目标 seen pair 排除其统计量，显式模拟 unseen 推理；验证仅 +0.0695 AUC 点，原始测试仍下降 0.041 点，说明先验统计泄漏不是主要瓶颈。
18. 三种子生成分数不确定性收缩与 top-K 反向流能量：前者验证 +0.120、测试 −0.054 AUC 点；后者验证 +0.144、测试 −0.459 AUC 点。生成条件信号在 C-GQA val 上可用，但继续复杂化后处理会放大 val/test 组合分布偏移。
19. 完整 pair prompt 条件收缩：`pair_cond_scale=0.5/0.25` 的验证增益仅 +0.076/+0.083 AUC 点，均弱于默认条件；保留属性/物体锚点并削弱 pair 交互仍不能解决 C-GQA，训练期条件容量搜索停止。
20. composition cross-fit flow：仅用三分之二 seen 组合训练 fold-0 时，C-GQA 生成分类器验证 AUC 降至 5.784，融合仅 +0.049 AUC 点；组合覆盖损失大于防记忆收益，不继续另外两个 fold。

## 5. 源分布受控消融与方法升级

在 MIT 上以三个生成器种子、统一验证选参比较经验 bootstrap、逐维矩匹配高斯与等总方差各向同性高斯：三者 AUC/HM 统计等价，说明高阶非高斯结构不是增益来源。UT 上各向同性明显优于经验 bootstrap，但与逐维高斯持平；主方法选择 MECoFlow 是因为它在性能不损失的同时只需总能量标量，并具有最大熵最小假设解释，而不是因为显著优于所有高斯基线。

| 源分布 | MIT AUC | MIT HM | UT AUC | UT HM |
|---|---:|---:|---:|---:|
| 经验残差 bootstrap | 23.417 ± 0.012 | 40.133 ± 0.028 | 44.458 ± 0.158 | 56.620 ± 0.229 |
| 对角矩匹配高斯 | 23.420 ± 0.012 | 40.130 ± 0.073 | 45.030 ± 0.152 | 57.429 ± 0.313 |
| 最大熵各向同性高斯 | 23.416 ± 0.029 | 40.130 ± 0.073 | **45.038 ± 0.126** | **57.429 ± 0.313** |

UT 对角高斯与各向同性高斯的差距只有 +0.009 AUC 点，HM 在三位小数处相同；各向同性的 AUC 种子标准差更低（0.126 vs 0.152）。因此不能声称各向同性显著更准，只能声称它在更少统计量、更强最小假设下保持性能且略降方差。各向同性版本不需保存实例残差库，只需加性组合均值、残差均值和一个总能量标量；论文 motivation 应为“组合定位 + 数据尺度校准 + 最大熵最小假设”，不能再强调经验非高斯性。

### 组合位置是否必要

把训练和生成源中心从加性 `mu(a,o)` 改成零，同时保持残差均值、总能量、条件、flow 架构、优化器和三种子统一选参不变：

| 源中心 | MIT AUC | ΔAUC | MIT HM | ΔHM | 统一设置 |
|---|---:|---:|---:|---:|---|
| 零中心 | 23.335 ± 0.051 | +0.211 | 39.986 ± 0.090 | +0.090 | update, 0.2 |
| 加性组合中心 | **23.416 ± 0.029** | **+0.293** | **40.130 ± 0.073** | **+0.235** | update, 0.2 |

加性定位在同能量随机源上额外贡献 **+0.081 AUC / +0.144 HM**，且降低种子方差。零中心仍有小幅正增益，说明文本条件 flow 能做一定长程运输；但把源预先放到未见组合邻域更有效、更稳定。

结果文件：`output/research/warmflow_mit/zero_center_shared.json`。

### 随机分布源是否必要

固定完全相同的三条已训练 flow，只把生成尺度设为 `s=0`，使每个未见组合的 32 个输入退化为同一个确定性原型。统一验证选参后，UT 的确定性版本只达到 44.094 ± 0.053 AUC / 56.613 ± 0.276 HM；随机最大熵版本达到 45.038 ± 0.126 / 57.429 ± 0.313。随机分布采样相对确定性原型额外贡献 **+0.945 AUC、+0.816 HM**，说明方法收益来自分布完成，而不是重复运输一个类原型。

结果文件：`output/research/warmflow_ut/gaussian_iso_deterministic_shared.json`。

### 连续运输与步数是否必要

在 UT 上使用同一 checkpoint、同一最大熵源和三个生成器 seed：

| 运输设置 | 测试 AUC | ΔAUC | 测试 HM | ΔHM | 验证增益 | 门控 |
|---|---:|---:|---:|---:|---:|---|
| 无 flow 源增强（确定性诊断） | 40.984 | −2.774 | 54.507 | −2.049 | — | 不适用 |
| 参数匹配 direct endpoint regressor | 43.934 ± 0.111 | +0.176 | 56.620 ± 0.196 | +0.064 | +0.304 | 通过 |
| 1-step Euler（raw） | 44.618 ± 0.093 | +0.860 | 56.899 ± 0.210 | +0.344 | +0.149 | 未通过 |
| 1-step Euler（正式） | 43.758 | +0.000 | 56.556 | +0.000 | +0.149 | 回退 CAMS |
| 8-step ODE（主版本） | **45.038 ± 0.126** | **+1.280** | **57.429 ± 0.313** | **+0.874** | **+1.120** | 通过 |

无 flow 结果没有生成器随机性，因此只作为机制诊断，不报告伪造的三 seed 方差。direct endpoint 使用与 flow 完全相同的 21.169M 网络、最大熵源、训练实例、辅助损失、生成分类器和验证协议，只把随机时间路径训练改为 `t=0` 的终点位移回归，并单步生成；8-step rectified flow 比它额外提高 **+1.105 AUC / +0.809 HM**，排除“只是更大 MLP 容量”的解释。1-step flow 的三次测试 AUC 全部为正，但验证信号未过统一门槛；结论是时间条件 flow 训练和多步积分共同提高效应强度与可验证可靠性，而不是声称一步法必然负迁移。

结果文件：

- `output/research/genclf_ut/results.json`
- `output/research/warmflow_ut/one_step_shared_raw.json`
- `output/research/warmflow_ut/one_step_shared_gated.json`
- `output/research/warmflow_ut/direct_endpoint_shared_raw.json`
- `output/research/warmflow_ut/direct_endpoint_shared_gated.json`

## 6. 跨 CAMS checkpoint 稳健性

服务器现存的 UT seed-1 checkpoint 与新训练的 MIT seed-1 checkpoint 均和各自原 CAMS checkpoint 在测试输出上明显不同。每个 checkpoint 使用三个生成器随机实例，并只按平均验证 AUC 选择一组共享融合设置：

| 基础模型 | CAMS AUC | MECoFlow AUC | ΔAUC | CAMS HM | MECoFlow HM | ΔHM |
|---|---:|---:|---:|---:|---:|---:|
| UT 原 checkpoint | 43.758 | 45.038 ± 0.126 | +1.280 | 56.556 | 57.429 ± 0.313 | +0.874 |
| UT CAMS seed-1 | 40.448 | 42.482 ± 0.581 | +2.034 | 53.972 | 54.975 ± 0.407 | +1.003 |
| MIT 原 checkpoint | 23.124 | 23.416 ± 0.029 | +0.293 | 39.895 | 40.130 ± 0.073 | +0.235 |
| MIT CAMS seed-1 | 23.297 | 23.517 ± 0.043 | +0.220 | 40.329 | 40.398 ± 0.103 | +0.069 |

最大熵版本在 UT/MIT seed-1 checkpoint 上的平均验证增益分别为 +0.361/+0.273 AUC 点，均高于保守门槛 0.2 点，因此标准选参与安全门控协议给出相同结果；四个数据集–checkpoint 设置的平均 AUC 都为正。MIT seed-1 三个测试 AUC 均为正，但 HM 的单 seed 并非全部为正，论文只报告均值和方差，不扩大主张。

结果文件：

- `output/research/warmflow_ut_seed1final/gaussian_iso_shared_raw.json`
- `output/research/warmflow_ut_seed1final/gaussian_iso_shared_gated.json`
- `output/research/warmflow_mit_seed1final/gaussian_iso_shared_raw.json`
- `output/research/warmflow_mit_seed1final/gaussian_iso_shared_gated.json`

MIT seed-1 的正式 seed0 证据来自 `gaussian_iso_strict_seed0`。旧 `gaussian_iso_seed0` 目录的原始 `results.json` 与 `classifier_logits_residual_1.00.pt` 未被改写，但一次重复任务污染了其中的 `flow.pt`、`additive_means.pt` 和 `run.log`，因此不再用该目录的中间件复现。

## 7. 当前状态

- C-GQA 的大网格后处理、训练期条件收缩、cross-fit 与最大熵源均已停止；raw 主结果仍回退 CAMS，但新增 structural-risk selective replay（+0.283/+2.119 点）作为单源 preliminary transfer evidence。
- MIT CAMS seed-1 已完成训练、特征缓存和三生成器复核；门控通过，正式结果已归档。
- UT 对角高斯三种子已完成；当前无本任务 GPU 进程。

### Open-world 边界

MIT open-world 的三个生成器种子使用同一验证选择 `update × 0.5`，原始结果由 CAMS 的 8.643 AUC / 22.269 HM 提升至 **8.792 ± 0.044 / 22.449 ± 0.110**（+0.149/+0.179），三次测试均为正；但平均验证增益 +0.149 AUC 点低于统一设定的 0.2 点安全门槛，因此保守协议回退 CAMS。UT open-world 单种子虽有 +1.204 验证 AUC 点，测试却由 36.280/52.130 降至 35.491/50.318（−0.789/−1.812）。因此主文只主张 closed-world，MIT open-world 小幅正结果可放补充材料，不能声称通用 open-world 改进。

结果文件：

- `output/research/warmflow_mit_ow/gaussian_iso_shared_raw.json`
- `output/research/warmflow_mit_ow/gaussian_iso_shared_gated.json`
- `output/research/warmflow_ut_ow/gaussian_iso_seed0/results.json`

## 8. 计算成本（MIT-States，单张 A100）

- rectified flow：21.169M 参数，20 epoch 训练约 28.1 秒；
- 8 步未见类生成 + 余弦分类器训练 + 快速评估：约 50.9 秒；
- 完整研究脚本（含读取 499 MB cache、构建原始 dataset/evaluator）：约 151.7 秒；
- 训练/生成阶段峰值显存约 1.55 GB（1.51 GiB）；
- 推理不运行 flow，只保留 1.507M 参数的余弦分类器；12,995 张测试图、1,962 类的两路打分与融合平均 11.6 ms，即约 0.00089 ms/图（批量矩阵乘法），缓存全部特征/分数时峰值约 646 MiB。

结果文件：

- `output/research/warmflow_mit_efficiency_full/results.json`
- `output/research/warmflow_mit_efficiency/inference_benchmark.json`

## 9. 仍需补齐的论文级实验

- 若资源允许，继续增加完整 CAMS 训练种子并报告跨基础模型置信区间；当前已完成 UT 与 MIT 各一个独立 seed-1 checkpoint；
- 零中心、确定性源、三种随机源、无 flow、一步/八步已由验证器整理成一致的论文表；下一步只需决定主文与补充材料的版面分配；
- open-world 明确限定为非主张：MIT 为稳定小幅正、UT 明显负；后续只在出现新的机制性假设时再探索，不以测试集反复调参；
- 与 DIFFCZSL、FlowComposer、DeFA、CAMS 及 2026 强基线在相同 checkpoint/协议下对比。

## 10. AOAR-Replay 预注册探索

基于近期 any-order autoregressive generative classification 与 generative replay 工作，新增一条不依赖扩散/flow 的轻量特征生成路线。实现位于 `research/anyorder_ar_replay.py`：把加性中心化后的 CAMS residual 压缩为 16 个连续 PCA token，用属性、物体和 pair prompt 条件的 random-order causal Transformer 建模，再对 unseen pair 做离线平衡回放。

当前状态：

- crdiff `py_compile`、CLI 和 CPU 合成 teacher-forcing / NLL backward / sequential generation 单测通过；
- 方法动机、可证伪预测与论文论证边界记录于 `research/AOAR_METHOD_NOTE.md`；
- UT 原 checkpoint 的三种子 validation-only 已完成，共享选择 `update × 1.0`，平均 validation AUC 增益 +0.760 点，通过 +0.2 点门槛；
- 固定共享选择后 test AUC 为 44.378±0.036（+0.620，3/3 正），HM 为 56.208±0.327（−0.347，0/3 正）；因此不是平衡提升，不进入主方法摘要；
- fixed raster 的共享 validation 增益为 +0.436 点但方差很大、共享设置仅 1/3 seed 正；random order 的 +0.760 点为 3/3 正，支持其稳定性作用；
- conditional-mean 的共享 validation 增益为 +0.924 点；固定共享选择后的 test AUC 为 44.466±0.065（+0.708±0.065，3/3 正），HM 为 56.376±0.359（−0.180±0.359，仅 1/3 正）。它提高 AUC 并缓解但未消除 HM 代价，当前不支持连续 token 噪声是收益来源；
- 3.762M AOAR 参数，单 seed 训练约 40 秒，生成+分类+评估约 4.7 秒，CUDA max allocated 约 1.48GB；
- 完整数值、逐 seed 符号、消融和论证边界见 `research/AOAR_RESULTS.md`。

## 11. 官方 FlowComposer 与第三 UT base seed 收尾

- 官方 FlowComposer 固定 commit `b4619f8...` 的 Stage-1/Stage-2 均按官方代码协议完成 15 epoch、exit 0；Stage-2 固定加载 epoch-5 initializer，最终固定 `gate_14.pth`，不按逐 epoch test 曲线选优。解析文件为 `output/research/official_flowcomposer_ut/stage1_metrics.json` 与 `stage2_metrics.json`，预提交 protocol SHA 未改变。
- 新训练 UT CAMS seed2 base 15 epoch exit 0；amended cache 使用 extraction seed=2 固定 train 随机增强，val/test transform deterministic。provenance validator 对 cache/checkpoint/config/builder/amendment/dataset/piflow 的 SHA、split 和 transform 全部通过。
- seed2 compatibility replay 三 generator seeds 的共享 validation 选择仍固定为 `update x 1.0`，但 mean AUC gain 仅 **+0.000679892**，低于 +0.002 gate，按协议回退且不读取 test。三份 logits 键集合严格为 `{'val'}`；最终 audit exit 0，aggregation 不含 test，结果均绑定同一 cache SHA `931092...` 与 val-best SHA `b40fa...`。
- seed2 是 amended reproducible pipeline；旧 seed0/1 cache builder 未固定 extraction seed，三者不宣称为完全同 builder 协议。该 negative result 保留用于覆盖边界和防泄漏审计。

## 12. Final validation-only closure (2026-08-24)

### 12.1 Seed2 compatibility replay: negative gate evidence

The amended seed2 final checkpoint completed 15 epochs with exit status 0. The
cache provenance validator passed with extraction seed 2, deterministic val/test
transforms, fixed-RNG train transforms, and split sizes train/val/test =
22998/3214/2914. The three frozen generator runs used Gaussian isotropic
residuals, residual scale 1.0, additive ANOVA source, 20 flow epochs, 30
classifier epochs, 32 samples/class, 8 sampling steps, and the preregistered
`update x 1.0` fusion protocol.

The validation gains were seed0 `-0.002105104`, seed1 `+0.003325352`, and seed2
`+0.000860660` AUC absolute. Shared aggregation was `+0.000679892`, below the
fixed `+0.002` gate, so the setting was disabled and test was not read. The
final audit passed: all three logits files contained exactly `{'val'}`, the
candidate set was unique, every generator result and the aggregation pointed to
the same seed2 cache/checkpoint provenance, and the aggregation contained no
test fields.

### 12.2 Deterministic train-transform robustness: negative evidence

As a separate validation-only robustness branch, train features were re-extracted
with deterministic Resize/CenterCrop/RGB/CLIP-normalized transforms. The cache
contained train/val only and explicitly recorded `test_read=false`; it did not
replace the compatibility replay or alter the parent fusion protocol. Three
generator runs all failed the gate (approximately `-0.00413` AUC each); shared
validation aggregation was **-0.002904064 AUC**, with the frozen fusion disabled.
No test logits or test metrics were produced.

### 12.3 Final-checkpoint TR-PFC: frozen negative evidence

Official FlowComposer Stage-1/Stage-2 completed under the fixed protocol with
final `gate_14.pth` (gate_14 SHA starts `b9f78bdf`). A single final-checkpoint
validation-only TR-PFC run used final deterministic validation features and the
only permitted setting `adaptive tangent scale=0.015, cap=0.02`; no grid or test
evaluation was run. Baseline AUC/HM were `0.226071200/0.372595985`; TR-PFC
obtained `0.226778922/0.374330060`, or **+0.070772 AUC points / +0.173407 HM
points**. The AUC gain did not reach the predeclared `+0.20` gate, so this is
mechanism/negative-boundary evidence rather than a selected main result.

Final artifacts are retained in the independent seed2 and TR-PFC output
directories. The protocol, cache, checkpoint, validation aggregation, execution
logs, generator results, and audit JSONs are hash-recorded; no subsequent GPU
job is authorized by this closure entry.

### 12.4 C-GQA topology-gated validation-only cross-seed check

The frozen `min_image_support_continuous/full/strength=0.2` topology gate was
replayed on three independent generator logits without reading test. Per-seed
AUC gains were `+0.001733/+0.001714/+0.001876` and HM gains were
`+0.002037/+0.002037/+0.002920`; all three seeds were positive for both
metrics. The shared mean was **+0.0017742028 AUC / +0.0023309812 HM**, so the
balanced signal is positive but the AUC mean remains below the fixed `+0.002`
gate and is not promoted to the main result. The audit JSON is
`output/research/warmflow_cgqa/topology_gated_valonly_seedcheck/results.json`,
the script SHA starts `1a5c0655`, and the record explicitly sets
`test_read=false`.

### 12.5 C-GQA column-centered calibration: fixed-test failure

An exploratory pair-column calibration combined with the frozen support gate
gave validation-only mean gains of `+0.002103701 AUC / +0.003030922 HM` on the
same three logits. To test transfer without post-test search, the unique
candidate (`column_centered_min_image_support_full`, strength `0.2`) was frozen
in `research/protocols/cgqa_column_centered_support_fixed_v1.json` (protocol
SHA `bab3c4f8...`) before one CPU fixed-setting test. Test gains were
**−0.002914084 AUC / −0.004689102 HM**, with 0/3 positive for either metric.
This is validation overfit evidence, not an improvement claim; the candidate
is rejected and no further search is authorized under this branch.

### 12.6 UT diagonal-source mean ensemble: fixed positive upgrade

The original UT checkpoint was re-evaluated with a distribution-source ensemble
that averages the three diagonal moment-matched Gaussian generator logits after
row-wise z-scoring, then applies the fixed `update` delta against CLIP text
logits. The only candidate was `strength=1.25`; it was frozen after a
validation-only screen and 5-fold cross-fit validation (`+0.017365088 AUC /
+0.010027219 HM`, all five folds positive). The fixed protocol is
`research/protocols/ut_diagonal_source_mean_fixed_v1.json` with SHA
`b5d86b1c...`; no source/strength/seed search was performed after test access.

The CPU fixed-setting test changed UT baseline AUC/HM
`0.437582872/0.565555819` to `0.451269393/0.576485920`, namely **+1.368652
AUC points / +1.093010 HM points**. Applying the same frozen strength to each
generator separately gave mean test gains `+1.224179 AUC / +0.810692 HM`
(population standard deviations `0.124755/0.269899` points); the larger number
above is the explicitly reported three-logit ensemble. The result JSON is
`output/research/warmflow_ut/fixed_distribution_test.json` (SHA
`0e7d3df5...`) and the audit JSON is `fixed_distribution_audit.json` (status
`passed`, SHA `c205a056...`).

### 12.7 Independent UT seed-1 diagonal-source ensemble

The diagonal-source ensemble was re-run from the independent UT seed-1 CAMS
checkpoint in isolated validation and test directories. The validation-only
screen considered the predeclared strengths `{0.2, 0.5, 0.75, 1.0, 1.25}`;
`0.75` was the retained candidate after the validation aggregate, and 5-fold
cross-fit remained positive (`+0.003569456 AUC / +0.008057727 HM`). The fixed
protocol `ut_seed1_diagonal_source_mean_fixed_v1.json` has SHA
`f740238f...`; each test run loaded the corresponding validation-trained flow,
and the external aggregator applied only the frozen `update x 0.75` ensemble.

The seed-1 baseline AUC/HM `0.404478068/0.539718787` became
`0.421219926/0.547237796`, or **+1.674186 AUC points / +0.751901 HM points**.
The same frozen setting applied per generator separately produced mean gains
`+1.629256 AUC / +0.662026 HM` (population SD `0.470953/0.356969` points),
with 3/3 positive for both metrics. The result JSON is
`output/research/warmflow_ut_seed1final/diag_ensemble_fixed_test.json` (SHA
`5db2736d...`), and `diag_ensemble_audit.json` is `status=passed` (the audit
also records `test_search_after_lock=false`).
The separate replay-consistency audit is also `status=passed`: all three
validation tensors in the test-enabled reruns are exactly equal to their
validation-only counterparts (`max_abs_diff=0.0`), so test access did not alter
the selection logits.

### 12.8 Topology-gated 复核补录（validation-only）

结果 JSON：`output/research/warmflow_cgqa/topology_gated_valonly_seedcheck/results.json`；
冻结候选为 `min_image_support_continuous/full/strength=0.2`，三 seed 均未读取
test（`test_read=false`）。均值增益为 **+0.0017742028 AUC /
+0.0023309812 HM**，AUC 未达到 `+0.002` gate，故不升级为主结果。

### 12.8a 最终审计补录（2026-08-24）

结果 JSON 固定为 `output/research/warmflow_cgqa/topology_gated_valonly_seedcheck/results.json`；
本次三 seed 复核明确 `test_read=false`，mean gain 为 **+0.0017742028 AUC /
+0.0023309812 HM**。该结果保持为 topology-gated 的平衡性正向但 gate-fail
validation-only evidence，不改变主结果。

### 12.9 MIT seed-1 source-median fixed check

MIT seed-1 完整 cache 的 validation-only 探针选择唯一候选
`source_median_update/strength=0.75`，AUC/HM gain 为
`+0.003032258/+0.001434410`。固定 test 在 protocol
`research/protocols/mit_seed1_source_median_fixed_v1.json`（SHA
`3cf64ba4...`）下执行一次：基线 `0.232965714/0.403287543`，融合后
`0.235022778/0.404409273`，即 **+0.205706 AUC 点 / +0.112173 HM 点**。
结果为 `output/research/warmflow_mit_seed1final/source_median_fixed_test.json`，
审计为 `output/research/warmflow_mit_seed1final/source_median_fixed_audit.json`，
`status=passed`、`test_search_after_lock=false`。逐源 AUC 3/3 正，HM 1/3 正。

### 12.10 UT cross-source median robustness check

原始 UT checkpoint 的六源（isotropic×3 + diagonal×3）median 聚合在
validation-only 上得到 `+0.016903459 AUC / +0.008818487 HM`，随后锁定
`strength=1.25` 做一次 fixed test。结果为 **+1.263431 AUC 点 /
+0.999660 HM 点**；六个 source 的逐源 AUC/HM 均为正。protocol 为
`research/protocols/ut_cross_source_median_fixed_v1.json`（SHA
`38a363f9...`），结果为
`output/research/warmflow_ut/cross_source_median_fixed_test.json`，审计为
`output/research/warmflow_ut/cross_source_median_fixed_audit.json`，
`status=passed` 且 `test_search_after_lock=false`。该结果作为鲁棒性补充，
主结果仍使用 validation 选择的 diagonal mean。

### 12.11 UT source-consensus gate fixed check

validation-only 在预声明网格中以 AUC gain `>=0.002` 为门槛、最大化 HM，锁定
inverse gate（`tau=1.0`, `strength=0.75`）。该 gate 的 test 结果为
**+1.111376 AUC 点 / +1.046375 HM 点**，六个 source-level AUC/HM 均为正。
protocol SHA 为 `6239ccf1...`，结果和审计分别为
`output/research/warmflow_ut/consensus_gate_fixed_test.json` 与
`output/research/warmflow_ut/consensus_gate_fixed_audit.json`，均为
`status=passed`，且 `test_search_after_lock=false`。它作为 source-agreement
机制与平衡性证据，不替换 diagonal mean 主设置。

### 12.12 UT support-aware source-consensus fixed check

在 source-consensus 上加入由 train split 图像支持度计算的 harmonic support
gate；候选网格预先固定为 support exponent、agreement tau、blend 和 strength，
并限制 `support_exponent > 0` 后按 validation HM 选择。锁定的候选为
`support_exponent=0.25`、inverse agreement `tau=2.0`、linear blend、
`strength=0.75`，validation-only gain 为 **+0.012893568 AUC /
+0.009909548 HM**。fixed test 结果为 **+1.305096 AUC 点 /
+1.232444 HM 点**，六个 source-level AUC/HM 均为正。protocol SHA 为
`4f7d372f...`，结果和审计分别为
`output/research/warmflow_ut/support_consensus_fixed_test.json` 与
`output/research/warmflow_ut/support_consensus_fixed_audit.json`，其
`status=passed`、`test_search_after_lock=false`。该分支的 HM 超过简单
source-consensus gate，但 AUC 仍低于 diagonal mean，因此定位为平衡性机制
消融，不替换主设置。

### 12.13 Independent UT seed-1 support-only transfer

在独立 UT seed-1 checkpoint 上复用同一预声明 support-aware 网格。原始
harmonic-support + inverse-consensus 候选在 validation-only 上为
`-0.006542405 AUC`，没有转移该候选；在 `support_exponent > 0` 子网格中按
validation HM 选择 `min_image_support`、无 consensus、exponent `0.5`、
linear blend、`strength=0.75`。validation-only gain 为
`+0.004957443 AUC / +0.012596196 HM`，fixed test 为
**+1.861108 AUC 点 / +0.968736 HM 点**，3/3 source-level AUC/HM 均为正。
protocol SHA 为 `5609fb66...`，结果和审计分别为
`output/research/warmflow_ut_seed1final/support_only_fixed_test.json` 与
`output/research/warmflow_ut_seed1final/support_only_fixed_audit.json`，均为
`status=passed` 且 `test_search_after_lock=false`。这是独立 checkpoint 的
补充 transfer evidence，不是与原始 checkpoint 合并的同设置置信区间。

### 12.14 Independent MIT seed-1 support-only transfer

MIT seed-1 上的 support-aware validation-only 网格锁定
`min_image_support`、support exponent `1.0`、无 consensus、linear blend、
`strength=0.5`。validation-only gain 为 `+0.002848590 AUC /
+0.002344917 HM`；fixed test 为 **+0.276025 AUC 点 / +0.025614 HM 点**。
三份 source 的 AUC 均为正，HM 为 2/3 正。protocol SHA 为 `2917a94d...`，
结果和审计分别为
`output/research/warmflow_mit_seed1final/support_only_fixed_test.json` 与
`output/research/warmflow_mit_seed1final/support_only_fixed_audit.json`，均为
`status=passed` 且 `test_search_after_lock=false`。它的 AUC 高于 MIT
source-median，但 HM 较低，只作为补充可靠性证据。

### 12.15 UT factorized object-support gate

将 pair support 分解为 attribute/object 边际统计，在预声明
`aggregation × alpha_attr × transform × strength` 网格中按 validation HM 选择
object-side support（`alpha_attr=0`）、mean aggregation、power `1.0`、
`strength=1.0`。validation-only gain 为 `+0.015113317 AUC /
+0.010113967 HM`；fixed test 为 **+1.469635 AUC 点 / +0.952866 HM 点**，
6/6 source-level AUC/HM 均为正。protocol SHA 为 `117d90cc...`，结果和审计
分别为 `output/research/warmflow_ut/factorized_object_support_fixed_test_v2.json`
与 `output/research/warmflow_ut/factorized_object_support_fixed_audit_v2.json`，
均为 `status=passed` 且 `test_search_after_lock=false`。AUC 略高、HM 略低于
harmonic support-aware gate，定位为 factorized support 消融。

### 12.16 C-GQA factorized-support transfer boundary

C-GQA 上复用 attribute/object factorized support validation-only 网格时，没有
候选达到 `+0.002` AUC gate；最佳为 attribute-only floor gate、median
aggregation、strength `0.25`，结果为 `-0.001210056 AUC / -0.000610543 HM`。
结果 JSON 为 `output/research/warmflow_cgqa/factorized_support_probe.json`
（SHA `2c48f215...`），`test_read=false`，因此没有进入 fixed test。这是
support prior 的负迁移边界，不可写成 C-GQA 提升。

### 12.18 Joint support-confidence reliability gate

在 frozen CAMS logits 和 frozen source ensemble 上运行 validation-only 联合门控
探针，候选同时覆盖 pair graph support（min/harmonic/object/attribute）与
image-level confidence（entropy/margin），先要求 validation AUC gain >= 0.002，
再以 validation HM 锁定唯一候选。UT 锁定 `object × base margin-high`、
`powers=1.0/0.5`、`product`、`strength=1.0`：validation
`+0.019773128/+0.012435618`，fixed test `+1.272944/+0.836122`，6/6 source
AUC、5/6 source HM 为正。MIT seed-1 锁定 `attribute + base entropy-low`、
`powers=1.0/1.0`、`linear`、`strength=0.25`：validation
`+0.002816028/+0.004155675`，fixed test `+0.161473/+0.021861`，3/3 source
AUC 为正，aggregate HM 为正。

协议分别为 `research/protocols/ut_joint_reliability_fixed_v1.json` 和
`research/protocols/mit_seed1_joint_reliability_fixed_v1.json`；结果与审计位于
各自 `joint_reliability_fixed_v1/` 目录。两组审计均 `status=passed`，
`test_search_after_lock=false`，validation probe 的 `test_read=false`。该实验
作为“graph support × instance confidence”机制消融记录，不把它的结果与
support-only/confidence-only 分支拼成新的主结果。

### 12.19 Joint gate negative transfer on UT seed-1

沿用同一预注册联合门控网格，在 UT seed-1 validation 上选择
`attribute × margin-high`（`support_power=0.5`、`confidence_power=1.0`、
product、strength `1.25`），validation 增益为 `+0.004301443 AUC /
+0.024252643 HM`。固定 test 反向迁移至 **−2.416473 AUC / −1.632702 HM**，
0/3 source runs 为正。该结果已由 `audit_joint_reliability.py --allow-negative`
作为 `negative_transfer_boundary` 审计通过；不是隐藏或丢弃的运行失败，而是
联合可靠性先验无法跨 checkpoint 稳定迁移的正式边界。

### 12.20 Adaptive source-posterior aggregation

将 isotropic/diagonal flow outputs 视为 noisy source observations，按每张图像
的 source-to-center disagreement 与 source confidence 计算归一化权重，再聚合
source logits。权重计算不读取标签，只有候选锁定使用 validation gate。UT 主
checkpoint 的固定候选为 `median + abs disagreement + margin-high`, `tau=4`,
power `0.25`, full strength `1.5`：validation `+0.013347315/+0.010506229`，
test `+0.989690/+0.717698`。MIT seed-1 为 `mean + abs disagreement +
entropy-low`, `tau=4`, power `1.0`, update strength `0.75`：validation
`+0.003433571/+0.001658656`，test **+0.256677/+0.281555**。

protocol 分别为 `ut_adaptive_source_posterior_fixed_v1.json` 和
`mit_seed1_adaptive_source_posterior_fixed_v1.json`；两组 audit 均通过且
`test_search_after_lock=false`。该机制提供比固定 mean/median 更有解释力的
逐样本源可靠性估计，但只作为机制/补充结果，不能把 UT 与 MIT 的不同候选合并
为同设置置信区间。

### 12.21 Original MIT adaptive posterior replay

将 MIT seed-1 已锁定的 `mean + abs disagreement + entropy-low`, `tau=4`,
power `1.0`, update strength `0.75` 原样迁移到原始 MIT checkpoint。validation
`+0.002603015/+0.002482513`，fixed test **+0.361611/+0.177142**。该复放不
重新搜索参数，protocol/result/audit 独立保存；audit `status=passed`，
`test_search_after_lock=false`，validation replay `test_read=false`。这是
adaptive source posterior 在两个 MIT checkpoint 上的一致正向证据。

### 12.22 Graph-conditioned adaptive posterior

在 adaptive source posterior 之上增加 pair-level graph support，形成 image-level
source reliability 与 composition-level structural reliability 的双层 gate。
UT 的 validation 候选为 `attribute support^0.5 × median/abs/margin posterior`
（strength `1.5`），validation `+0.038938579/+0.025897745`，但 test
**−0.897186/−0.259885**。MIT seed-1 的 `min support^1.0 + mean/abs/entropy
posterior`（linear、strength `0.5`）validation `+0.002983914/+0.002950015`，
test **+0.268488/+0.132552**。

该分支的正负结果均保留：UT 是强 validation 过拟合的负边界，MIT 是小幅正向
机制证据。protocol、结果和 audit 独立保存，均满足 `test_read=false` 的
validation 不变量与 `test_search_after_lock=false`。

### 12.17 Confidence-aware transport gate

使用 frozen CAMS 的 image-level entropy/margin 作为逐样本 gate。UT
validation-only 选择 `base_entropy_low^0.5`, strength `1.0`，fixed test 为
**+1.505562 AUC / +1.029062 HM 点**，6/6 source-level 指标为正；MIT seed-1
选择 `base_entropy_high^0.25`, strength `0.5`，fixed test 为
**+0.201655 AUC / +0.164343 HM 点**，3/3 AUC 正、HM 2/3 正。两组 protocol
均在 test 前锁定，审计 `status=passed` 且 `test_search_after_lock=false`。
该分支不增加 flow 参数，是 support-aware gate 之外的 confidence-aware
transport 消融。

### 12.23 Locked reliability residual mixture

新增一个不改变 flow 的机制组合：把 validation 前已锁定的 adaptive source
posterior residual 与 support-aware structural residual 按凸系数相加，只在
validation-only 选择 `alpha`。MIT seed-1 的 `alpha_adaptive=0.5` 给出
validation `+0.003100451/+0.002563750`，fixed test **+0.305237 AUC 点 /
+0.315012 HM 点**；UT 原始 `alpha=0.875` fixed test **+0.952342/+0.717698**，
MIT 原始 imported `alpha=0.5` fixed test **+0.354714/+0.140232**。所有候选均由
独立 protocol 锁定，结果分别位于
`output/research/warmflow_ut/residual_mixture_fixed_v1/results.json`、
`output/research/warmflow_mit_seed1final/residual_mixture_fixed_v1/results.json`
和 `output/research/warmflow_mit/residual_mixture_fixed_v1/results.json`；三份
validation probe 均明确 `test_read=false`，fixed test 不再搜索 alpha 或内部
候选。该分支作为“source reliability × structural reliability”平衡性消融，
不替换 UT 主 headline。

### 12.24 Interaction hyper-classifier

新增模型化消融：`InteractionHyperClassifier` 用 attribute/object/pair text
的低秩交互生成 pair-specific classifier weights，再用 frozen flow 生成样本
训练 classifier。UT 30 epoch validation-only 选择 `full x 0.2`，validation
`+0.004324853/+0.003646179`，fixed test **+0.191834 AUC 点 /
+0.080275 HM 点**。该结构提供了比标量 residual gate 更强的创新证据，但指标
低于现有 support-aware/adaptive 结果，故只列为模型化消融。MIT seed-1 15 epoch
validation 为 `+0.000994804/0.000000000`，AUC gate-fail，未读取 test。

### 12.25 Pair-level source-family routing

对 isotropic 与 diagonal 三 seed family 做 pair-level 路由：权重只由 train
split 的组合 support 给出，`diagonal_weight=support^0.25`，再使用 frozen
`update x 1.25`。validation gain 为 `+0.017223295/+0.008168948`，fixed
test **+1.375993 AUC 点 / +1.093010 HM 点**。相比 global diagonal mean
（+1.368652/+1.093010），AUC 有小幅提升且 HM 不变；该结果支持“组合级
结构可靠性选择分布源”的方法解释。结果位于
`output/research/warmflow_ut/pair_source_router_fixed_v1/results.json`，
audit `status=passed`，validation `test_read=false`，test 后未搜索参数。

独立 UT seed-1 复放重新按 validation 锁定 `inverse_pair_support^1/update x 1.0`
（不是原始 UT 的 `pair_support^0.25/update x 1.25`），validation
`+0.003108242/+0.012596196`，fixed test **+2.184035 AUC 点 /
+1.183657 HM 点**。该 checkpoint-conditioned route 结果与原始 UT 的
`+1.375993/+1.093010` 均为正，但 route 方向不同，不能写成统一固定设置。

### Candidate-conditioned reliability routing (2026-08-25)

A validation-only probe replaced the image-wide entropy weight with a
candidate-specific frozen CAMS pair-score gate. The preregistered family used
`alpha=sigmoid(kappa*(z_{x,p}-tau))`, with `kappa` in `{-8,-4,4,8}` and
`tau` in `{-0.5,0,0.5}`; seen pairs were fixed to the HM route. Validation
selected `kappa=-4,tau=0` on UT seed-1 (`+1.765/+4.843` AUC/HM points) and
`kappa=-4,tau=0.5` on MIT seed-1 (`+2.972/+8.667`). No test tensors were read
during selection.

Protocol-locked fixed tests then reached **+8.360 AUC/+11.863 HM** on UT seed-1
and **+3.381/+9.025** on MIT seed-1. The source-pair audits are independently
positive for all three pairs in both datasets: UT mean **+7.872/+11.438** and
MIT mean **+3.177/+8.888** AUC/HM points. Results are in
`candidate_conditioned_route_fixed_v1.json` and
`candidate_conditioned_route_source_audit_v1.json` under the corresponding
checkpoint directories; every audit records `test_search_after_lock=false`.
This is now the primary candidate-conditioned reliability-routing innovation;
the earlier image-only entropy schedule remains an ablation.

The same locked candidate-conditioned route was independently replayed on the
original UT checkpoint. Validation selected `kappa=-4,tau=.5`; fixed test gain
was **+8.031 AUC/+11.200 HM** and the three source-pair mean was
**+8.049/+11.128**, with all source pairs positive. This gives two independent
UT checkpoints plus MIT seed-1 under the same mechanism family, while keeping
checkpoint-specific thresholds frozen separately.

C-GQA was then evaluated as a deliberately separate single-source exploratory
transfer using the same candidate family. Validation selected `kappa=-8,tau=-.5`
(`+1.344/+6.395` AUC/HM); the locked fixed test gave **+1.130 AUC/+5.606 HM**.
This result remains preliminary single-source evidence and does not change the
C-GQA raw gate or become a multi-source main claim.

### 12.32 UT structured-head negative boundary

按预注册协议 `research/protocols/ut_factorized_head_val_v1.json`，在同一
UT seed-1 cache、同一 flow checkpoint、`residual_scale=1.0` 和同一生成 seed
下，validation-only 比较 factorized composition head 与 rank-64 visual
adapter。factorized head 相对 CAMS base 的 AUC/HM gain 为
`+0.000181751/+0.004762293`，adapter 为 `+0.000924596/+0.003694825`；
两者均低于 `+0.002` AUC gate，且两个 results JSON 均明确 `test_read=false`。
该负边界支持把主方法收益归因于最大熵分布完成与冻结 CAMS 融合，而不是任意
替换分类头；不启动 fixed test，不改变主结果。

### 12.36 Topology-gated final audit record (2026-08-25)

最终审计补录固定结果 JSON `output/research/warmflow_cgqa/topology_gated_valonly_seedcheck/results.json`；该条目为 validation-only，明确 `test_read=false`，三 seed mean gain **+0.0017742028 AUC / +0.0023309812 HM**。AUC 未过 `+0.002` gate，因此只保留为平衡性正向、gate-fail 证据。

### 12.37 C-GQA structural-risk selective replay

在 graph-unseen pair 上将 frozen CAMS entropy 与训练图支持不足联合为结构风险分数，
validation 锁定 `beta=.5, slope=8, threshold=.8`。结果位于
`output/research/warmflow_cgqa/structural_risk_fixed_v1/`：validation
`+0.005695/+0.025678`，fixed test **+0.002825 AUC / +0.021187 HM**，即
**+0.283/+2.119 点**，且 `test_search_after_lock=false`。这是单 source seed 的
有限迁移证据；同一候选在 UT/MIT 作为机制消融均未超过各自已锁定主设置。

### 12.38 C-GQA structural-risk source audit

在不重新搜索 beta/slope/threshold 的前提下，对冻结候选做三个显式 source pair
复放。结果 JSON：`output/research/warmflow_cgqa/structural_risk_source_audit_v1.json`。
fixed test 三个 pair 的 AUC/HM 均为正，3/3、3/3，均值 **+0.266/+1.640 点**。
后两个 pair 使用已有 generation-seed artifact 且 validation AUC 未过统一 gate，
所以这是 source-level robustness evidence，不替代主结果的严格多源协议。

### 12.39 UT seed-1 triple-risk negative boundary

为检验 source disagreement 是否能补充 image entropy 与 graph support，预注册
54 个 `entropy + beta*(1-support) + gamma*source_disagreement` 门控候选，
validation-only 结果最优为 `gamma=0`（即退化为已知 structural-risk gate），
gain **+0.868655 AUC / +2.996748 HM 点**，低于已锁定 sigmoid entropy 主设置的
validation gain **+1.145472/+3.387293 点**。该分支未读 test、未做 fixed replay，
作为 source-disagreement 无额外收益的负边界保留。

### 12.31 Final topology-gated audit addendum (2026-08-25)

最终审计再次固定结果 JSON `output/research/warmflow_cgqa/topology_gated_valonly_seedcheck/results.json`；
三 seed 复核明确 `test_read=false`，mean gain 为 **+0.0017742028 AUC /
+0.0023309812 HM**。该条目保持为平衡性正向但 AUC gate-fail 的 validation-only
证据，不读 test、不改主结果。

### 12.32 MIT cache compatibility boundary (CPU-only)

为避免把不同 CAMS checkpoint 的特征缓存混合，比较了原始 MIT
`piflow_mit/features.pt` 与 seed-1 `cache_mit_seed1/features.pt`。validation
AUC/HM 分别为 `0.264784525/0.435349694` 与 `0.259733820/0.428700452`，故标记
`incompatible`。未从 seed-1 cache 训练原始 MIT diagonal family；审计 JSON 为
`mit_cache_compatibility_audit.json`，MIT seed-1 结果保持 checkpoint-specific。

### 12.33 Instance-adaptive probe correction audit

修正 probe 中 graph-seen mask 的来源：使用 pair-level boolean seen mask，避免把
real-centroid 的图像计数误当作布尔值。修正后分别重跑 UT seed-1、UT 原始和 MIT
seed-1 validation-only probe，候选、所有 row gains 与原始结果逐位一致；独立审计
JSON `adaptive_route_blend_probe_repro_audit.json` 均为 `status=passed` 且
`test_read=false`，固定 test 结果不变。修正脚本 SHA 前缀为 `0e1cc5de`。

### 12.35 C-GQA selective transport replay

基于已有 C-GQA isotropic/diagonal source artifacts，validation-only 预注册
entropy threshold family，锁定 `reverse_hard_0.8`。validation gain 为
`+0.005533/+0.025405 AUC/HM`，fixed test 为 **+0.002306 AUC/+0.019635 HM**；
结果位于 `output/research/warmflow_cgqa/adaptive_threshold_fixed_v1/`，
`test_search_after_lock=false`。当前仅有 1 个 source seed，audit 1/1 AUC、
1/1 HM 正，故不升级为 C-GQA 多 seed 或通用结论。

### 12.34 Sigmoid-threshold route schedule upgrade

在冻结 inverse-pair AUC route 与 soft-selective HM route 后，预注册 entropy
threshold/temperature family 并仅用 validation 选参。`sigmoid_8_0.50` 通过
`+0.002` AUC gate；fixed test 为 **+6.444 AUC / +7.263 HM 点**，source-seed
audit 为 3/3 AUC、3/3 HM 正，且 `test_search_after_lock=false`。这比原
`unseen_entropy_1` 的 +6.298/+6.979 更高；结果目录为
`adaptive_route_threshold_fixed_v1/`。此前 route-strength 校准候选为
`+4.498/+4.768`，作为补充消融，不替换 sigmoid schedule。

最后的联合 `schedule x strength` validation-only 网格选择了
`sigmoid_4_0.70/A=1.0/B=1.0`，但 fixed test 仅为 **+4.393 AUC/+4.669 HM
点**（3/3 source seeds 正），低于固定强度的 `sigmoid_8_0.50`
**+6.444/+7.263**。该结果作为联合校准边界，不替换当前 strongest candidate，
并停止继续扩网格。

最后的 support-conditioned threshold 27 候选 validation-only 搜索最佳为
`sigmoid_8_0.70` with `support_shift=.2`，gain `+0.011242/+0.034520`，仍低于
当前 `sigmoid_8_0.50`，未读取 test、未启动 fixed replay；算法搜索到此停止。

同一 threshold family 在 UT 原始 checkpoint 独立 validation 锁定 `hard_0.7`，
fixed test 为 **+5.324 AUC/+5.624 HM 点**，source-seed audit 3/3 AUC、3/3 HM
正。两个 UT checkpoint 的 schedule 分别由各自 validation 锁定，不能合并成
统一 threshold 的置信区间。

MIT seed-1 同一 family 选择 `reverse_hard_0.9`，fixed test 为
**+2.074 AUC/+5.250 HM 点**，source-seed audit 3/3 AUC、3/3 HM 正；相对
constant soft route 仅有小幅 AUC 差异，作为 checkpoint-specific schedule
消融记录，不升级为统一阈值。

### 12.31 Pareto blend of frozen routes

在 validation-only 中对两个已冻结候选做 convex blend：A=`inverse_pair_support^1`
的 update x1.0，B=`pair_support^0.25` soft selective full x1.5；锁定
`blend_soft=.5`。validation `+0.009151286/+0.016350749`，fixed test
**+2.138209 AUC/+1.269095 HM** 点，三 source seeds 均 AUC/HM 正。
该点 AUC 高于 soft-only、HM 高于 inverse-only，是可报告的 Pareto
trade-off，但不与 HM-first soft candidate 合并成单一主设置。

### 12.32 Instance-adaptive route blend

在两个已冻结的 route residual 之间引入逐图 entropy schedule：seen pair 保持
CAMS，unseen pair 根据 CAMS entropy 在 `inverse_pair_support^1/update x1.0`
与 `pair_support^0.25/seen_lambda=.25/full x1.5` 之间插值。validation-only
仅搜索预声明 schedule，锁定 `unseen_entropy_1`，mean validation gain 为
`+0.010726265 AUC/+0.033872925 HM`，且 `test_read=false`。

固定 test 只复用 validation entropy 的 min-max normalization，未在 test 后
搜索：AUC **+0.062976**、HM **+0.069792** 点；source-seed audit 为 3/3
AUC、3/3 HM 正，protocol/audit 均 `status=passed`，
`test_search_after_lock=false`。结果位于
`output/research/warmflow_ut_seed1final/adaptive_route_blend_fixed_v1/`。
该分支说明“组合支持度决定可用 transport family，实例不确定性决定 route
强度”可作为 selective transport 的进一步机制，而不是新的后验 test gate。

同一 validation-only 脚本在 UT 原始 checkpoint 独立复放并锁定
`unseen_entropy_2`：validation `+0.020522070/+0.032502036`，fixed test
`+0.052832/+0.055709 AUC/HM`，source-seed audit 为 3/3、3/3 正。该复放只
说明机制跨 CAMS checkpoint 保持正向，不把 schedule 强行统一。

### 12.33 MIT seed-1 diagonal source replay

为补齐 MIT 的 source-family 对照，在独立目录用同一 20-epoch flow 协议训练
三个 `gaussian_diag` source seed；先生成 validation-only logits，再在候选冻结
后用同一 flow checkpoint 生成 test-capable logits。adaptive probe 的
validation 选择 constant soft-selective route，gain `+0.017839/+0.046395`
（AUC/HM）；fixed test 为 **+2.066/+5.250 AUC/HM**，source-seed audit 为
3/3 AUC、3/3 HM 正，且 `test_search_after_lock=false`。由于最优 schedule 是
constant，而非 entropy，结果作为 support-tapered route 的 MIT seed-1 复放，
不写成实例熵调度的跨数据集提升。
seed-1 与原始 UT 的 protocol/result/audit SHA 前缀已同步写入 completion audit，
用于最终 provenance 复核。

### 12.26 Routed source plus image confidence boundary

在 UT seed-1 的 `inverse_pair_support^1` 路由上叠加
`base_margin_high^1` image gate，validation-only 选择 `update x 1.25`，
validation 为 `+0.007795631/+0.022227127`，但 fixed test 仅
`+0.916 AUC/+0.045 HM` 点，未复现 pair router 的 `+2.184/+1.184`。
protocol/audit 通过；该组合记录为 validation-overfit boundary，不纳入主结果。

### 12.27 C-GQA topology-gated validation replay

独立目录 `output/research/warmflow_cgqa/topology_gated_valonly_seedcheck/results.json`
完成三 seed validation-only 复核，候选为
`min_image_support_continuous_full/strength=.2`。三 seed AUC gains
`+.001732861/.001713647/.001876101`，HM gains
`+.002036669/.002036669/.002919605`，mean gain
`+0.001774203 AUC/+0.002330981 HM`；3/3 AUC/HM 正，且 `test_read=false`。
均值 AUC 低于 `+0.002` gate，故标记为平衡性正向但 gate-fail 的
validation-only evidence，未读取 test、未改变主结果。

### 12.28 UT seed-1 AUC-first pair-routing ablation

在 validation-only 网格中预注册 AUC-first 规则：候选须满足 AUC gain
`>=.002` 与 HM gain `>=.010`，然后最大化 AUC。锁定
`obj_support^0.25/update x 0.75` 后，validation 为
`+0.004894246/+0.011200076`，fixed test 为
`+1.674186 AUC/+0.751901 HM` 点。结果位于
`output/research/warmflow_ut_seed1final/pair_source_router_aucfirst_v1/`，
audit `status=passed`；不与 inverse-pair 主路由合并，作为 AUC/HM Pareto
结构消融报告。

### 12.29 Selective unseen-pair source routing

新增 scope 维度，validation-only 网格比较 `all/seen_only/unseen_only`，
只在 train-unseen pair 注入 source-family residual，其余 pair 保持原 CAMS
分数。UT 原始锁定 `pair_support^0.5/unseen_only/full x 1.5`：validation
`+0.025995220/+0.020311324`，fixed test **+0.659371 AUC /
+0.764892 HM** 点。UT seed-1 锁定 `pair_support^0.25/unseen_only/full x 1.5`：
validation `+0.015368756/+0.014496908`，fixed test **+1.735855 AUC /
+1.298395 HM** 点。MIT 原始 validation 选择
`pair_support^0.25/all/update x 0.5`：validation `+0.002354507/+0.003322819`，
fixed test **+0.335326 AUC/+0.105430 HM** 点。结果目录分别为
`output/research/warmflow_ut/selective_pair_router_fixed_v1/`、
`output/research/warmflow_ut_seed1final/selective_pair_router_fixed_v1/` 和
`output/research/warmflow_mit/selective_pair_router_fixed_v1/`，三项 audit
均 `status=passed`；scope/power/strength 由各自 validation 锁定，不跨 checkpoint
合并成统一 CI。

同一 frozen candidate 的 source-seed audit 显示：UT 原始、UT seed-1 和 MIT
原始均为 3/3 test AUC 正、3/3 test HM 正；对应 JSON 为三个
`selective_pair_router_fixed_v1/source_seed_audit.json`。MIT seed-1 的单个
source validation AUC 为轻微负值，但 test 仍为正，且没有在 test 后重新搜索。

### 12.30 Support-tapered selective routing

将 binary unseen-only scope 扩展为连续 `seen_lambda`：unseen pair 保持完整
transport，seen pair 只接收 lambda 比例。UT seed-1 validation 锁定
`pair_support^0.25/seen_lambda=.25/full x 1.5`，validation
`+0.011133677/+0.015733895`，fixed test **+1.823992 AUC/+1.454767 HM**；
source-seed audit 为 3/3 AUC、3/3 HM 正。MIT 锁定
`pair_support^0.5/seen_lambda=.75/update x .5`，validation
`+0.002618165/+0.004221242`，fixed test **+0.260322 AUC/+0.105430 HM**；
source-seed audit 为 3/3 AUC、2/3 HM 正。UT soft candidate 是当前 strongest
balanced setting；MIT 结果作为连续 gate 的正向但 seed-level HM 边界。

MIT 原始 checkpoint 的 isotropic/diagonal family 复放选择
`inverse_disagreement_reliability^0.5/update x 0.5`，validation
`+0.002354728/+0.003322819`，fixed test **+0.335386 AUC 点 /
+0.105430 HM 点**。audit `status=passed`；结果为正向跨数据集结构路由证据，
但低于该 checkpoint 的 adaptive posterior，保持为补充消融。

独立 UT seed-1 复放重新按 validation 锁定 `inverse_pair_support^1/update x 1.0`
（不是原始 UT 的 `pair_support^0.25/update x 1.25`），validation
`+0.003108242/+0.012596196`，fixed test **+2.184035 AUC 点 /
+1.183657 HM 点**。该 checkpoint-conditioned route 结果与原始 UT 的
`+1.375993/+1.093010` 均为正，但 route 方向不同，不能写成统一固定设置。
### 12.39 C-GQA candidate-conditioned route: strict three-source fixed audit

The candidate-conditioned reliability route was preregistered from a six-source
validation probe with `kappa=-8`, `tau=-.5`. The fixed protocol locked the
candidate before test, listed all six source-logit SHA256 values, and recorded
`test_search_after_lock=false`. Fixed test improved C-GQA by **+1.222 AUC /
+5.942 HM points** (16.325 -> 17.546 AUC; 34.780 -> 40.722 HM). The independent
source audit is positive for all 3 isotropic/diagonal source pairs, with mean
**+1.135 AUC / +5.788 HM points**. This upgrades the earlier single-source
exploratory replay to strict three-source transfer evidence, but does not change
the raw C-GQA validation-gated main branch.

# AOAR-Replay 实验审计

更新日期：2026-08-24

## 1. 协议

- 数据与基座：UT-Zappos 原 CAMS checkpoint 的冻结 feature cache；
- 主版本：128 维 residual PCA，16×8 维连续 token，hidden 256、4 layers、8 heads，random token order，Gaussian token sampling；
- 每个 unseen pair 离线生成 32 个特征；
- 三个生成/训练 seed 均先运行 `validation-only`；
- 在所有 seed 的 validation AUC 上选择一组共享 `full/update × strength`；
- 平均 validation AUC 增益达到预注册的 +0.2 点后，才固定该设置读取 test；
- 正式 test 运行必须通过 `--shared-selection` 加载已启用的共享 JSON，脚本拒绝无共享选择的 test。

## 2. 主版本

三种子共享 validation 选择为 `update × 1.0`：

| split | CAMS AUC | AOAR AUC | ΔAUC | CAMS HM | AOAR HM | ΔHM |
|---|---:|---:|---:|---:|---:|---:|
| validation | 45.171 | 45.931 ± 0.096 | **+0.760** | 61.646 | 62.025 ± 0.255 | **+0.378** |
| test | 43.758 | 44.378 ± 0.036 | **+0.620** | 56.556 | 56.208 ± 0.327 | **−0.347** |

逐 seed test AUC 增益为 +0.586/+0.616/+0.657 点，3/3 为正；HM 增益为 −0.724/−0.146/−0.172 点，0/3 为正。

结论：AOAR-Replay 产生了非常稳定的 AUC 增益，但在当前 Gaussian-sampling 版本上没有得到平衡的 HM 提升。它不能替代 MECoFlow 主结果，也不能写成“AUC/HM 全面提升”；可作为独立非扩散生成路线、AUC 专向信号或补充材料负结果。

机器可读结果：

- `output/research/aoar_ut/shared_val_selection.json`
- `output/research/aoar_ut/shared_test_results.json`

## 3. 机制消融与冻结 test 扩展

所有消融均保持 latent 维度、网络、训练轮数、classifier 和候选融合网格不变，并各自做三种子共享选择。raster 只做 validation-only；conditional-mean 在未读 test 前通过共享 gate，随后用冻结的 `update × 1.0` 读取三种子 test。

| token order | generation | 共享设置 | validation AUC | ΔAUC | seed std | 共享设置逐 seed 正增益 |
|---|---|---|---:|---:|---:|---:|
| random | Gaussian sample | update × 1.0 | 45.931 ± 0.096 | +0.760 | 0.096 | 3/3 |
| raster | Gaussian sample | update × 1.0 | 45.608 ± 0.832 | +0.436 | 0.832 | 1/3 |
| random | conditional mean | update × 1.0 | **46.095 ± 0.111** | **+0.924** | 0.111 | 3/3 |

### 顺序效应

raster 的平均增益仍为正，因此不能声称固定顺序完全失败。但它主要由 seed2 的强正结果驱动，seed0/1 在共享设置下略负，且 AUC seed 标准差约为 random-order 的 8.6 倍。当前证据支持“random order 提高跨 seed 稳定性并改善平均 validation AUC”，而不是“只有 random order 才能生成”。

### 连续 token 噪声

去掉 Gaussian token noise、只使用每一步条件均值后，validation 增益从 +0.760 提高到 +0.924 点，HM validation 均值也略升。因此当前数据不支持“AR 内部随机采样是收益来源”；多样性仍可来自 random permutation，而额外连续噪声可能过度扩散 unseen classifier，和 test HM 下降方向一致。

冻结设置后的 conditional-mean test AUC 为 **44.466 ± 0.065**，相对 CAMS **+0.708 ± 0.065** 点，三个 seed 增益为 +0.712/+0.640/+0.770，3/3 为正。HM 为 **56.376 ± 0.359**，相对 CAMS **−0.180 ± 0.359** 点，逐 seed 为 −0.553/−0.149/+0.162，仅 1/3 为正。它比 Gaussian sampling 的 AUC 更高、HM 降幅更小，但仍未解决平衡校准，不能升级为主方法。

机器可读结果：

- `output/research/aoar_ut/raster_shared_val_selection.json`
- `output/research/aoar_ut/mean_shared_val_selection.json`
- `output/research/aoar_ut/mean_shared_test_results.json`

## 4. 效率

主版本 seed0 的脚本内测量：

- AOAR 参数：3,761,680；
- 余弦 classifier 参数：89,089；
- AR 训练约 40.0 秒；
- unseen generation + classifier + evaluation 约 4.7 秒；
- CUDA max allocated 约 1,474.7 MiB；
- `/usr/bin/time` 完整墙钟约 51 秒，远低于单张 A100 40G 限制。

生成器仍只用于离线 replay，部署时可丢弃；但正式推理成本需和 MECoFlow 用同一 benchmark 脚本再测，不能只凭结构声称更快。

## 5. 代码与可复现性

- `research/anyorder_ar_replay.py`：主训练/生成/评估；
- `research/aggregate_aoar_validation.py`：只接受 validation-only 结果并做共享选择；
- `research/aggregate_aoar_test.py`：验证所有运行共享同一选择后汇总 test；
- `research/RUN_AOAR_REPLAY.md`：预注册运行协议；
- `research/AOAR_METHOD_NOTE.md`：文献动机、可证伪预测和论证边界。

正式 test 使用独立目录按相同 seed 重训，以避免覆盖 validation-only 证据。逐 seed 比较表明，test 与先前 validation-only 目录中的 `anyorder_ar.pt` 和 `residual_pca.pt` SHA256 完全一致（3/3）；共享设置下的 validation 指标也逐 seed 完全一致。因此 test 阶段只增加了固定选择的 test evaluator，没有发生生成器或 PCA 漂移。

conditional-mean 扩展也通过同一检查：三个 seed 的 validation-only 与 formal-test `anyorder_ar.pt`、`residual_pca.pt` SHA256 均逐项一致（6/6 文件对），排除了在看到 test 后重训漂移或改参。

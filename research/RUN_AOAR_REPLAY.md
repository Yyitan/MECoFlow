# AOAR-Replay 可行性实验

## 1. 目的

AOAR-Replay（Any-Order Autoregressive Residual Replay）是与 MECoFlow 独立的非扩散生成路线。它不训练图像级 VQ/AR 模型，而是在冻结 CAMS `com` 特征空间中：

1. 用与 MECoFlow 相同的加性组合均值定位属性–物体组合；
2. 对 seen 实例残差拟合 128 维 PCA，并分成 16 个、每个 8 维的连续 token；
3. 以属性、物体和 pair prompt 为条件，用显式位置指令训练 random-order causal Transformer；
4. 对每个 unseen pair 以随机 token 顺序采样 32 个残差，解码回 CAMS 特征并做离线 generative replay；
5. 复用与 MECoFlow 相同的平衡文本初始化余弦分类器、`full/update` 融合网格和 +0.2 AUC 点 validation gate。

随机顺序不是普通数据打乱：模型序列为 `condition, position_1, value_1, ..., position_N, value_N`，在每个 position instruction 上预测随后 token 的 Gaussian 参数。causal mask 保证预测不能看到当前值；不同顺序让生成器学习多种残差因子分解，降低任意 PCA 轴顺序带来的归纳偏置。

## 2. 计算预算

- 默认模型：hidden 256、4 layers、8 heads、序列长度 33，远小于图像级 RandAR；
- PCA/AR 训练和 unseen replay 均单卡完成；
- 生成器离线丢弃，推理仍只保留平衡余弦分类器；
- 任何 GPU 启动前仍需查询显卡与相关进程，且与其他任务合计不超过两张卡。

## 3. 预先固定的首轮协议

第一轮只在 UT-Zappos 原 checkpoint 上运行 seed 0、`--validation-only`，不构造 test dataset：

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<least-idle> \
./anaconda3/envs/crdiff/bin/python -u \
research/anyorder_ar_replay.py \
  --cache output/research/warmflow_ut/features.pt \
  --dataset-path data/ut-zap50k \
  --work-dir output/research/aoar_ut/seed0_val \
  --latent-dim 128 --token-dim 8 \
  --hidden-dim 256 --layers 4 --heads 8 \
  --epochs 20 --batch-size 512 --lr 2e-4 \
  --generation-temperature 1.0 --residual-scale 1.0 \
  --samples-per-class 32 \
  --classifier-epochs 30 --classifier-batch-size 2048 \
  --min-val-auc-gain 0.002 --validation-only --seed 0
```

决策规则在运行前固定：

- seed-0 validation AUC 增益低于 +0.2 点：停止该路线，不读取 test；
- 达到门槛：补 seed 1/2 的 validation-only 运行，用三种子平均 validation AUC 选择一组共享融合参数；
- 只有共享选择仍达到门槛，才固定参数读取三个 seed 的 test；
- 不搜索 latent/token/temperature 后再用同一 test 挑版本；后续超参变化必须先在 validation-only 阶段比较。

## 4. 必要对照

若通过首轮门槛，至少报告：

- 固定 PCA raster order vs random order；
- 确定性 AR mean vs Gaussian sampling；
- AOAR-Replay vs MECoFlow vs parameter-matched direct endpoint；
- AR 参数、训练/生成时间和峰值显存；
- 三生成种子共享 validation 选择；
- MIT/UT closed-world，C-GQA 仍先 validation-only。

## 5. 当前状态

- `research/anyorder_ar_replay.py` 已在 crdiff 通过 `py_compile`；
- CPU 合成单测已验证 teacher forcing、Gaussian NLL backward、逐 token 生成和 scatter 复原；
- 主 random+sample 版本三种子共享 validation 选择为 `update × 1.0`，+0.760 AUC 点并通过门槛；
- 固定共享选择的 test 为 44.378±0.036 AUC（+0.620，3/3 正）/56.208±0.327 HM（−0.347，0/3 正）；
- raster validation-only 为 +0.436 点且高方差；conditional-mean validation-only 为 +0.924 点；
- conditional-mean 在 test 前冻结 `mean_shared_val_selection.json` 后得到 44.466±0.065 AUC（+0.708，3/3 正）/56.376±0.359 HM（−0.180，仅 1/3 正），因此改善但未解决平衡代价；
- 任何非 validation-only 运行现在都必须传入 `--shared-selection`，否则脚本拒绝读取 test；
- 完整审计见 `research/AOAR_RESULTS.md`。所有预注册 AOAR validation/test 扩展均已完成，GPU2 已释放。

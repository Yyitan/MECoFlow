# MECoFlow 复现实验命令

在 `./CAMS` 下执行。每次启动 GPU 程序前先查询：

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.535.104.05 \
/usr/bin/nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
  --format=csv,noheader,nounits
```

Python 环境：

```bash
PY=./anaconda3/envs/crdiff/bin/python
DATA=./8222/all_data
```

## 1. 构建冻结 CAMS 特征缓存

```bash
CUDA_VISIBLE_DEVICES=<gpu> $PY -u research/build_feature_cache.py \
  --config configs/utzappos.yml \
  --dataset-path $DATA/ut-zap50k \
  --checkpoint output/utzappos_crdiff/final_model.pt \
  --output output/research/warmflow_ut/features.pt
```

MIT-States 与 C-GQA 分别替换 config、数据目录、checkpoint 和输出目录。

## 2. 训练主版本（8 步、三个生成器种子）

```bash
for seed in 0 1 2; do
  CUDA_VISIBLE_DEVICES=<gpu> $PY -u research/warmstart_flow_classifier.py \
    --cache output/research/warmflow_ut/features.pt \
    --dataset-path $DATA/ut-zap50k \
    --work-dir output/research/warmflow_ut/gaussian_iso_seed${seed} \
    --residual-distribution gaussian_isotropic \
    --residual-scales 1.0 \
    --sample-steps 8 \
    --samples-per-class 32 \
    --flow-epochs 20 \
    --classifier-epochs 30 \
    --seed $seed
done
```

不要直接把该循环提交到繁忙服务器；应在每个实际启动前重新查询 GPU，并且全服务器同时最多占两张卡。

## 3. 三种子统一验证选参

```bash
CUDA_VISIBLE_DEVICES=<gpu> $PY -u research/shared_fusion_selection.py \
  --cache output/research/warmflow_ut/features.pt \
  --dataset-path $DATA/ut-zap50k \
  --classifier-logits \
    output/research/warmflow_ut/gaussian_iso_seed0/classifier_logits_residual_1.00.pt \
    output/research/warmflow_ut/gaussian_iso_seed1/classifier_logits_residual_1.00.pt \
    output/research/warmflow_ut/gaussian_iso_seed2/classifier_logits_residual_1.00.pt \
  --output output/research/warmflow_ut/gaussian_iso_shared.json \
  --min-val-auc-gain 0.002
```

脚本按三个种子的平均验证 AUC 选择一个共同的 `full/update` 与融合强度；只有 AUC 完全相同时才用 HM 打破平局。门控失败时测试强度自动置零。

## 4. 关键消融

经验残差 bootstrap 与逐维矩匹配高斯：

```bash
... research/warmstart_flow_classifier.py ... --residual-distribution empirical
... research/warmstart_flow_classifier.py ... --residual-distribution gaussian_diag
```

主版本使用 `gaussian_isotropic`：它以真实训练残差总能量校准最大熵源，不保存实例残差库。MIT 三种子显示三类数据校准源统计等价；UT 上它明显优于经验 bootstrap，与对角高斯基本持平但种子方差略低。选择它的依据是最小假设与统计简洁性，不能声称显著优于对角高斯。

与主网络参数完全相同、但不采样中间时间且直接回归终点位移的容量控制：

```bash
... research/warmstart_flow_classifier.py ... \
  --transport-objective direct_endpoint \
  --sample-steps 8
```

`sample-steps` 在该对照中会被忽略，结果 JSON 将记录 `flow_sampling_steps=1`。该实验只移除时间条件路径/连续积分，保留最大熵源、网络参数、训练数据、辅助损失、生成分类器和验证协议。

去除组合位置、只保留相同残差统计的零中心对照：

```bash
... research/warmstart_flow_classifier.py ... \
  --residual-distribution gaussian_isotropic \
  --source-center zero
```

该开关只把训练与生成源中心从加性 `mu(a,o)` 改为零；残差均值、总能量、网络、条件和训练协议全部不变，用于隔离组合定位的必要性。

一步采样：

```bash
... research/warmstart_flow_classifier.py ... --sample-steps 1
```

无 pair 条件：

```bash
... research/warmstart_flow_classifier.py ... --pair-cond-scale 0
```

语义局部残差池：

```bash
... research/warmstart_flow_classifier.py ... \
  --residual-bank-mode local --semantic-neighbors 16 --primitive-bonus 0.25
```

冻结已见类生成分类器权重：

```bash
... research/warmstart_flow_classifier.py ... --freeze-seen-classifier
```

全局 CAMS 特征空间：

```bash
... research/warmstart_flow_classifier.py ... --feature-key glb
```

后四项目前是诊断开关，不属于已确认的主版本。

## 5. 主要产物

- `flow.pt`：训练后的条件速度场；
- `additive_means.pt`：可加原语先验；
- `classifier_logits_residual_*.pt`：验证/测试生成分类器 logits，用于统一选参而无需重新训练；
- `results.json`：单次运行的基础指标、所有验证候选、门控选择和测试指标；
- `shared_auc_selection.json`：三种子统一协议的最终结果。

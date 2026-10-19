# MECoFlow —— 基于最大熵分布补全的组合零样本学习研究存档

> **定位声明**：本仓库是一个**研究存档（research archive）**，用于完整记录在
> [CAMS](https://github.com/ybyangjing/CAMS)（arXiv:2511.16378）基座上开展的
> MECoFlow（Maximum-Entropy Composition Flow）方向的研究过程与产物。
> 仓库内的 CAMS 基础代码版权归原作者所有；本存档仅用于记录个人研究过程，
> 不构成对 CAMS 或 FlowComposer 等上游工作的替代发布。

## 1. 研究动机

组合零样本学习（Compositional Zero-Shot Learning, CZSL）要求模型识别训练中从未
共同出现过的属性–物体组合。多数基于 CLIP 的 CZSL 方法把每个 unseen 组合表示为
**单个文本导出的点原型**，无法刻画该组合的视觉分布；而在稀疏组合图上直接从通用
噪声生成高维视觉分布又是病态的。MECoFlow 的出发点是：

> CZSL 的缺口不只是"没有 unseen 类原型"，而是"没有 unseen 组合的**视觉分布**"。

因此 MECoFlow 把问题重构为**分布补全（distribution completion）**：先用可加原语
效应把随机源定位到目标组合附近，再用数据校准的最大熵源刻画剩余不确定性，最后用
conditional rectified flow 把这个短程源运输到冻结的视觉特征流形上。

## 2. 方法概要

MECoFlow 是一个轻量的**冻结 CAMS 分布补全插件**，训练与推理分四步：

1. **加性组合定位**：用属性/物体可加原语效应构造未见组合的先验中心 `mu(a, o)`；
2. **最大熵残差源**：在给定真实残差的均值与总中心能量的前提下，各向同性高斯
   唯一最大化预归一化微分熵——以此构造最小假设的源分布，而非任意噪声先验；
3. **条件 rectified flow**：以 pair 条件学习从最大熵源到冻结视觉特征分布的
   短路连续运输（8 步采样）；
4. **生成式重放训练**： transported 样本训练一个平衡的 text-initialized
   cosine 分类器；**flow 在推理时丢弃**，最终只保留约 1.51M 参数的分类器，
   无扩散骨干、无生成期开销。

在主路线之上，存档还记录了两条扩展线（详见 `research/` 文档）：

- **可靠性路由（reliability routing）**：support-aware source-consensus gate、
  confidence-aware gate、graph-support-conditioned selective transport，以及
  作为主创新的 **candidate-conditioned reliability routing**——直接读取冻结
  CAMS 的逐候选 pair score，在 AUC-specialist 与 HM-specialist 路由间做无参数
  切换，不增加可训练参数、不使用测试标签；
- **AOAR-Replay**：与 MECoFlow 独立的非扩散对照路线（Any-Order Autoregressive
  Residual Replay），在冻结 CAMS `com` 特征空间中对 PCA 残差 token 做
  random-order causal Transformer 生成重放，复用同一套验证门控与融合协议。

## 3. 仓库结构

```
MECoFlow/
├── code_snapshot/            # 完成时的代码快照（CAMS 基座 + 研究期改动）
│   ├── UPSTREAM_README.md    #   上游 CAMS README 原样保留（原 README.md 改名）
│   ├── LITERATURE_NOVELTY_AUDIT.md   # 文献与创新边界审计
│   ├── model/ clip_modules/  #   CAMS 模型与 CLIP 封装
│   ├── configs/              #   三个数据集 × 多种子的配置
│   ├── external/FlowComposer_official/   # FlowComposer 官方代码快照（对照基线）
│   ├── run_all3.sh / run_seeds.sh        # CAMS 基线多种子训练脚本
│   └── train.py / test.py    #   CAMS 训练与评估入口
├── research/                 # MECoFlow 全部研究脚本与文档（核心）
│   ├── RUN_RECOFLOW.md       #   主路线复现命令
│   ├── RUN_AOAR_REPLAY.md    #   AOAR-Replay 复现命令
│   ├── METHOD_DRAFT.md / ABSTRACT_DRAFT.md / PAPER_PLAN.md
│   ├── EXPERIMENT_LOG.md / PAPER_RESULTS.md / RESEARCH_COMPLETION_AUDIT.md
│   ├── NOVELTY_AUDIT.md / RELATED_WORK_COMPARISON.md
│   └── warmstart_flow_classifier.py 等 120+ 实验脚本与冻结协议 JSON
├── research_archive_extras/  # 归档补充材料
│   ├── configs_snapshot/     #   配置独立快照
│   ├── project_root_snapshot/ / research_snapshot/   # 项目根与研究目录快照
│   ├── local_workspace_materials/            # 本地工作区材料索引
│   └── ARCHIVE_METADATA.txt / SOURCE_LINKS.txt / MANIFEST_SHA256.txt
├── patches_from_upstream/    # 相对上游 CAMS 的已知差异（仅 1 个 config 保存路径 diff）
├── ARCHIVE_MANIFEST.txt      # 原始完整存档的文件清单与字节数
└── ARCHIVE_SHA256SUMS.txt    # 原始完整存档的 SHA256 校验和
```

**注意**：`ARCHIVE_MANIFEST.txt` 与 `ARCHIVE_SHA256SUMS.txt` 描述的是 2026-08-25
建立的**原始完整存档**（含 `output/` 下的特征缓存、checkpoint、训练日志等约 94GB
产物）。本仓库仅收录代码快照与研究文档，**不含数据集、模型权重与实验输出**。

## 4. 复现说明

### 4.1 环境与数据

- PyTorch + CLIP ViT-L/14（存档所用 conda 环境名为 `crdiff`，依赖与 CAMS 一致，
  可参考 `code_snapshot/UPSTREAM_README.md`）；
- 数据集：MIT-States、UT-Zappos、C-GQA（下载方式见上游 CAMS README）。

### 4.2 CAMS 基线（可选，用于复现冻结 checkpoint）

`code_snapshot/run_all3.sh` 与 `run_seeds.sh` 记录了三个数据集 × 5 种子的完整
基线训练流程（train.py 闭集训练 + test.py 开放世界评估）。脚本中的服务器本地
路径（conda 环境、数据目录）需按自己的环境修改。

### 4.3 MECoFlow 主路线

逐步命令见 [`research/RUN_RECOFLOW.md`](research/RUN_RECOFLOW.md)，流程为：

```bash
# 1) 构建冻结 CAMS 特征缓存
python research/build_feature_cache.py --config configs/utzappos.yml \
    --dataset-path <data>/ut-zap50k --checkpoint <cams_ckpt> \
    --output output/research/warmflow_ut/features.pt

# 2) 训练主版本（8 步采样、三个生成器种子）
python research/warmstart_flow_classifier.py --cache <cache> \
    --residual-distribution gaussian_isotropic --sample-steps 8 \
    --samples-per-class 32 --flow-epochs 20 --classifier-epochs 30 --seed <s>

# 3) 三种子统一验证选参（validation-only 门控）
python research/shared_fusion_selection.py --classifier-logits <seed0> <seed1> <seed2> ...
```

关键消融（经验 bootstrap / 对角高斯源、`direct_endpoint` 参数匹配对照、
`--source-center zero` 组合定位消融、一步采样等）同样记录在该文档中。

### 4.4 AOAR-Replay 对照路线

见 [`research/RUN_AOAR_REPLAY.md`](research/RUN_AOAR_REPLAY.md)：`research/anyorder_ar_replay.py`
在冻结特征空间训练 random-order AR 残差生成器，脚本强制 `--validation-only`
先行，未通过预注册门槛不读取 test。

## 5. 已验证结果（摘自存档）

以下数字直接摘自归档时的验证记录（`README_ARCHIVE.md`，完整表格见
[`research/PAPER_RESULTS.md`](research/PAPER_RESULTS.md) 与
`research/RESEARCH_COMPLETION_AUDIT.md`），均为相对冻结 CAMS checkpoint 的
AUC / HM 点数增益，全部候选由 validation-only 协议锁定：

| 设置 | 增益（AUC / HM，点） |
|---|---|
| UT-Zappos seed-1 candidate-conditioned route | +8.360 / +11.863 |
| UT-Zappos 原 checkpoint candidate-conditioned route | +8.031 / +11.200 |
| MIT-States seed-1 candidate-conditioned route | +3.381 / +9.025 |
| C-GQA 严格三源重放 | +1.222 / +5.942 |

存档同时明确了**不主张**的范围：不声称绝对 SOTA，不声称各向同性高斯显著优于
对角高斯，C-GQA 原始分布补全分支保持 validation-gated 回退，open-world 无通用
提升结论。

## 6. 引用

本存档的基座工作为 CAMS，如使用本仓库中的 CAMS 代码，请引用：

```bibtex
@article{yang2025cams,
  title={CAMS: Towards Compositional Zero-Shot Learning via Gated Cross-Attention and Multi-Space Disentanglement},
  author={Yang, Pan and Deng, Cheng and Yang, Jing and Zhao, Han and Liu, Yun and Chen, Yuling and Ruan, Xiaoli and Chen, Yanping},
  journal={arXiv preprint arXiv:2511.16378},
  year={2025}
}
```

## 7. License

本存档基于 CAMS（见上游仓库与其许可声明）。`code_snapshot/` 中与上游相同的
文件版权归原作者所有；本存档**新增**的研究脚本、文档与配置以 MIT 协议发布
（见 [LICENSE](LICENSE)，该许可仅覆盖新增内容，不改变上游代码的权属）。

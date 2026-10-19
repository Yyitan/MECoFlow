# FlowComposer 官方代码复现与协议审计

更新日期：2026-08-24

## 1. 复现对象

- 官方仓库：`HKUST-LongGroup/FlowComposer`
- 固定提交：`b4619f8eed546e7408a46d34ae44b4386f23da20`
- 代码目录：`external/FlowComposer_official/code`
- 数据：UT-Zappos 标准 closed-world split
- 环境：现有 `crdiff`（Python 3.10），单张 A100 40GB
- CLIP：服务器已有的 OpenAI ViT-L/14 checkpoint

本复现不改模型代码。冒烟配置只把 `train.epochs` 从 15 改为 1；正式运行使用仓库原始 `config/troika/ut-zappos.yml` 和 `run.sh` 中的 Stage-1 参数。

## 2. 冒烟结果

| 项目 | 结果 |
|---|---:|
| 退出状态 | 0 |
| 单 epoch 墙钟 | 14分42.51秒 |
| GPU 峰值（`nvidia-smi` 观测） | 约 17.8 GiB |
| 主进程最大 RSS | 3,477,916 KiB |
| epoch-0 AUC / HM | 38.59 / 53.29 |
| checkpoint | `troika_0.pt`、`cfm_attribute_0.pt`、`cfm_obj_0.pt` 均完整生成 |

该数值只用于确认依赖、数据、训练、评估与保存链路可运行，不能作为正式方法比较。

## 3. 官方代码协议审计

### Stage-1

`s1_flow.py` 虽构造 `val_dataset`，但训练循环没有对其调用评估。每个 epoch 直接把 `test_dataset` 传给 `evaluate`：前 `epoch_pt_end` 阶段评估一步，其后同时评估一步和八步；随后保存该 epoch 的 Troika、attribute-flow 与 object-flow 权重。

### Stage-2

`s2_gate.py` 同样构造但不使用 `val_dataset`。它在 gate 训练前先评估一次 test，并在每个 epoch 后对 test 做八步组合评估和一个端点诊断，然后保存 `gate_{epoch}.pth`。

### checkpoint 选择

官方 `run.sh` 对 UT-Zappos 固定 `UT_CHECKPOINT_EPOCH=5`，Stage-2 加载 `troika_5.pt`、`cfm_attribute_5.pt`、`cfm_obj_5.pt`，gate 学习率为 `1e-7`。代码中没有提供由 validation 指标自动选出 epoch 5 的路径。

因此，后续正式数值只能称为 **official-code protocol reproduction**。它能验证官方实现是否在当前硬件/数据上重现论文量级，但不能当作与 MECoFlow 的严格 val-only 同协议显著性比较，也不能据其逐 epoch test 曲线选择最优 epoch。

## 3.1 Stage-2 冒烟

使用 Stage-1 smoke 的 epoch-0 checkpoint 和同一 1-epoch YAML，在独立目录运行官方 `s2_gate.py`：

| 项目 | 结果 |
|---|---:|
| 退出状态 | 0 |
| 墙钟 | 9分24.19秒 |
| GPU 峰值（`nvidia-smi` 观测） | 约 7.6 GiB |
| 主进程最大 RSS | 3,476,844 KiB |
| checkpoint | `gate_0.pth` 完整生成 |

解析器得到四条预期记录：初始非组合 step-1 为 38.59 AUC；一轮 gate 后 full-endpoint compose step-1/8 为 6.33/9.06 AUC；`end_points=0.1` 诊断为 39.84 AUC。一轮 gate 显然未收敛，这些数值只验证 initial evaluation、gate training、两种积分步数、短端点诊断和保存链路均可运行，不用于性能比较。

## 4. 正式运行状态

- Stage-1：原始 15 epoch 配置、seed 0 已在 GPU3 正常完成；退出状态 0，`done!` 存在，墙钟 2:16:43，最大 RSS 3,478,236 KiB。
- Stage-1 产物核验：`troika_0..14.pt`、`cfm_attribute_0..14.pt`、`cfm_obj_0..14.pt` 各 15 个，解析得到 29 条官方 test 评估记录；这些曲线不用于选优。
- 输出目录：`output/research/official_flowcomposer_ut`
- Stage-2：已在重新检查 GPU 占用后启动并正常完成；严格加载官方指定 `troika_5.pt`、`cfm_attribute_5.pt`、`cfm_obj_5.pt`，原始 15 epoch 配置，`lr_gate=1e-7`、`lr_fm=2e-4`，日志为 `stage2.log`。
- Stage-2 产物核验：`gate_0..14.pth` 各完整存在；`stage2_metrics.json` 解析为 `completed_marker=true`、`exit_status=0`、46 条官方评估记录、墙钟 1:55:23、最大 RSS 3,476,792 KiB。
- 固定最终 gate：`gate_14.pth` SHA256 为 `b9f78bdf9ba61c5fdc153738cfb8910e5d2dc38ca8b67686f167dd71bad42aa1`。预提交 protocol SHA 仍为 `38569a1fd257657afbe6fa29703dfda1139a2749a4177c3f191de428cb5d0dda`。
- 正式汇报预先固定为 Stage-2 的最后一个 epoch 和官方指定的 Stage-1 epoch-5 初始化；逐 epoch test 结果只作诊断，不用于回头选优。
- 机器可读预提交：`output/research/protocol/official_flowcomposer_ut_protocol_precommit.json`，在任何 Stage-2 曲线产生前固定上述规则；SHA256 为 `38569a1fd257657afbe6fa29703dfda1139a2749a4177c3f191de428cb5d0dda`。
- Stage-1 机器可读解析：`output/research/official_flowcomposer_ut/stage1_metrics.json`。
- Stage-2 机器可读解析：`output/research/official_flowcomposer_ut/stage2_metrics.json`。

### 4.1 最终 gate_14 机器可读指标

按预提交规则固定 `gate_14.pth`，Stage-2 最后一条官方 endpoint evaluation
（`reported_epoch=15`, `compose=true`, `end_points=0.1`）为：AUC **0.4101**、HM
**0.5360**、best-seen **0.7019**、best-unseen **0.6949**。该数值来自
`stage2_metrics.json`，不是从逐 epoch test 曲线反向选优；它只能作为
official-code protocol 的量级参照，不能与 MECoFlow 的 frozen-CAMS validation-gated
结果合并成同协议比较。

## 5. 与 MECoFlow 的解释边界

FlowComposer 是 Troika 上联合训练的 primitive endpoint flow 与 velocity composer；MECoFlow 是冻结 CAMS 后的最大熵源到视觉分布运输，训练后丢弃 flow。即使两者使用同一数据 split，基础模型、训练形态与 test 访问协议仍不同。正文应保留两类证据：

1. 官方 FlowComposer 复现：证明与公开强生成/流式 CZSL 的绝对量级差距；
2. MECoFlow 的冻结 CAMS 严格插件表：用共享 validation 选择、三生成种子和安全门控证明自身增益。

两张表不能合并成“同 checkpoint SOTA”结论。

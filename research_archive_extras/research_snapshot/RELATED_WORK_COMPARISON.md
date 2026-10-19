# MECoFlow 与 2025–2026 生成式/流式 CZSL 的可比性审计

## 1. 先给结论

MECoFlow 当前不能宣称绝对 SOTA。它的最强可辩护位置是：**在冻结 CAMS checkpoint 上，以极低训练和部署成本完成未见组合的视觉分布补全，并在四个数据集–checkpoint 设置上获得正的平均 AUC；其生成机制、训练信号和推理形态均不同于 DIFFCZSL、FlowComposer、DeFA 与 LPD。**

跨论文绝对数值只可作为标准 split 下的参考，因为各工作使用的基础模型、复现权重、联合训练方式和验证超参数并不相同。最公平的横向量是“相对各自基础模型的插件增益”，但这仍不是同 checkpoint 对照。

## 2. 标准 closed-world 论文报告数值（非同 checkpoint）

| 方法 | 所接基础模型 / 训练方式 | MIT AUC / HM | UT AUC / HM | C-GQA AUC / HM |
|---|---|---:|---:|---:|
| MECoFlow（本项目） | 冻结 CAMS checkpoint，三生成器共享验证选择 | 23.416±0.029 / 40.130±0.073 | 45.038±0.126 / 57.429±0.313 | 16.325 / 34.780（门控回退） |
| DIFF-CAMS | CAMS 联合训练 + SD2.1/CleanDIFT 训练期教师 | 24.0 / 40.7 | 46.1 / 57.9 | 16.2 / 34.2 |
| FlowComposer | Troika + primitive flows + Composer + leakage augmentation | 23.5 / 40.2 | 46.8 / 58.6 | 15.9 / 34.0 |
| DeFA | LoRA 微调 CLIP + 实例 Cartesian 重组和残差融合 | 22.8 / 39.3 | 46.1 / 58.6 | 14.6 / 32.3 |
| LPD（CLIP 版本） | CSP 风格 CLIP + learnable primitive disparity | 不报告 | 40.9 / 54.1 | 7.3 / 21.8 |

这些绝对数值不能排序后直接写成同协议 SOTA：例如本项目的 C-GQA CAMS 基础分数 16.325/34.780 已高于上述几篇论文中的多个完整方法，说明 checkpoint/实现差异足以改变排名。

## 3. 相对各自基础模型的插件增益

| 插件 | 基础模型 | MIT ΔAUC / ΔHM | UT ΔAUC / ΔHM | C-GQA ΔAUC / ΔHM |
|---|---|---:|---:|---:|
| MECoFlow | 本项目 CAMS | +0.293 / +0.235 | +1.280 / +0.874 | +0.000 / +0.000（门控；raw AUC −0.244） |
| DIFFCZSL | 论文复现 CAMS | +0.6 / +0.4 | +1.1 / +1.3 | +0.4 / +0.5 |
| FlowComposer | 论文复现 Troika | +1.4 / +1.0 | +5.0 / +3.2 | +3.4 / +4.3 |
| DeFA | 论文自身 UT baseline | 未给同表组件基线 | +4.9 / +6.6 | 未给同表组件基线 |

因此，MECoFlow 在 UT 的 AUC 插件增益与 DIFF-CAMS 同量级且略高，但 HM 增益更低；MIT 增益更小；C-GQA 尚未建立增益。FlowComposer 和 DeFA 的论文增益明显更大，但建立在不同基础模型和联合训练方案上。主文不能用这些数字声称 MECoFlow 优于它们。

本地官方代码复现提供了一个额外的可核查锚点：UT Stage-2 固定最终
`gate_14.pth` 的 endpoint evaluation 为 **AUC 0.4101 / HM 0.5360**。
该结果属于 official-code protocol（训练和评估脚本逐 epoch 读取 test，且使用
Troika epoch-5 初始化），所以只用于说明公开强生成方法的运行量级，不替代同
checkpoint、validation-only 的严格对照。

## 4. 方法边界与真正差异

| 方法 | 生成/运输对象 | 训练信号 | 未见组合如何出现 | 推理时额外路径 |
|---|---|---|---|---|
| MECoFlow | 最大熵视觉源 → CAMS 视觉特征分布 | 冻结 CAMS seen 特征上的 conditional rectified flow | 加性组合位置 + 残差总能量随机源，随后条件 ODE 运输并采样 | 不运行 flow；仅轻量余弦分类头 |
| DIFFCZSL | 不生成未见视觉样本；使用扩散表征作正则 | SD2.1/CleanDIFT 与视觉/文本分支对齐 | 仍由原 CZSL 识别模型外推 | 无扩散路径，但训练需外部扩散教师 |
| FlowComposer | 图像特征 → 属性/物体文本端点的 primitive velocities | primitive flow matching + velocity Composer + leakage supervision | 在 embedding space 组合属性/物体速度场 | 运行 primitive flow/Composer |
| DeFA | batch 内属性/物体实例特征 Cartesian 重组 | 重建、pair augmentation、contrastive augmentation、频率去偏 | 从训练实例重组并经残差融合网络合成 | 联合识别/增强网络保留 |
| LPD | 文本组合点 + learnable primitive disparity | triplet/重建/primitive refinement | 学习组合 disparity 并合成视觉向量 | 其编码/解码与分类路径保留 |

MECoFlow 的核心新意不能写成“首个生成式 CZSL”或“首个 flow CZSL”。可写成：**在给定组合加性位置、残差均值和总能量约束下，以最大熵源表达未知类内不确定性，再学习到冻结视觉空间的短程分布运输；这与把图像点运输到文本端点的 primitive flow 是相反方向和不同目标。**

## 5. 计算形态

| 方法 | 训练期额外依赖/参数 | 论文报告的推理形态 |
|---|---|---|
| MECoFlow | flow 21.169M；单 A100 约 28.1 s 训练；峰值约 1.51 GiB | flow 离线丢弃；分类器 1.507M；冻结特征上两路批量评分约 0.00089 ms/图 |
| DIFF-CAMS | +17.7M trainable；SD2.1/CleanDIFT 教师；CAMS 每 epoch 由 7m35s 增至 9m18s（MIT） | 扩散教师不进入推理 |
| FlowComposer | FM 74.06M + Composer 25.97M | Troika 推理约 17.6→19.2 ms/图（MIT）、11.0→12.7（UT）、69.8→74.4（C-GQA） |
| DeFA | ViT-L/14 LoRA + 多个投影/融合模块，论文在 RTX 4090 联合训练 | 合成与融合属于完整识别模型的一部分 |
| LPD | 300 epoch；编码器/解码器与 disparity 学习 | 非冻结 CAMS 后处理插件 |

计算表不能用 MECoFlow 的缓存后处理时间与其他方法的端到端训练时间作“速度倍数”结论；只能说明 MECoFlow 适合在已训练基础模型上低成本追加。

## 6. 投稿表格建议

主文应拆成两张表：

1. **标准 split 参考表**：列出各论文报告的 absolute AUC/HM，并加脚注明“不同基础模型/论文实现，不作同 checkpoint 显著性比较”。
2. **严格插件表**：只放本项目四个 CAMS checkpoint、三个生成器 seed、统一验证选择与门控结果；DIFF-CAMS 等只报告论文 delta，标为 external reported。

若要获得真正的同 checkpoint 强比较，优先级是：

1. 获得 DIFFCZSL 官方代码后，在当前 CAMS checkpoint 上运行其 DIFF-CAMS；
2. 若无官方代码，只实现论文关键正则会产生“非官方弱复现”，必须单独标注，不能作为否定该方法的证据；
3. FlowComposer/DeFA 改接 CAMS 会改变其原始联合训练结构，成本高且不再是原方法，宜使用作者代码与其基础模型复现实验，而不是做一个名字相同的 proxy；
4. 当前已有 cross-primitive flow、no-flow 与 interaction 等内部对照可用于机制消融，但不能冒充官方 FlowComposer/DeFA。

FlowComposer 官方代码现已公开，固定提交、运行环境、冒烟结果和 test-set 访问协议另见 `research/OFFICIAL_FLOWCOMPOSER_AUDIT.md`。正式复现必须标为 official-code protocol：公开 Stage-1/Stage-2 均逐 epoch 读取 test，且 UT 的 Stage-2 初始化由脚本固定为 Stage-1 epoch 5；不能把这一结果描述为严格 validation-only 对照。

DIFFCZSL 代码状态于 2026-08-24 复核：arXiv v1（<https://arxiv.org/abs/2608.19871>）未列项目页或官方仓库，按精确标题、缩写和作者检索也未发现可验证的官方实现。因此当前只引用论文数值；不把自行猜测的正则实现命名为 DIFF-CAMS。

## 7. 来源定位

- DIFFCZSL：Table 1、Table 7、Table 8。
- FlowComposer：Table 1、Table 7、Table 11、Table 13。
- DeFA：Table 2、Table 4 与 implementation details。
- LPD：Table 1、Table 3 与 training setup。
- 本项目：`research/PAPER_RESULTS.md`，由 `research/verify_paper_results.py` 从归档 JSON 自动生成。

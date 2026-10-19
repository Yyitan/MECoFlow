# Awesome Compositional Zero-Shot Learning 2024-2026

A curated list of recent papers on Compositional Zero-Shot Learning (CZSL), with an emphasis on top conferences and journals from 2024 to 2026.

This repository focuses on attribute-object compositional learning, open-vocabulary CZSL, CLIP/VLM-based CZSL, prototype learning, disentanglement, context modeling, and related survey resources.

## Contents

- [2026 Papers](papers/2026.md)
- [2025 Papers](papers/2025.md)
- [2024 Papers](papers/2024.md)
- [Surveys](surveys.md)
- [Related Earlier Works](papers/related-earlier-works.md)

## Main Papers

| Year | Venue | Title | Paper | Code | Setting | Keywords |
|---|---|---|---|---|---|---|
| 2026 | CVPR | Bridging the Modality Gap in Compositional Zero-Shot Learning via Sparse Alignment and Unimodal Memory Bank | [paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_Bridging_the_Modality_Gap_in_Compositional_Zero-Shot_Learning_via_Sparse_CVPR_2026_paper.pdf) | - | CZSL | modality gap, patch features, memory bank |
| 2026 | ICLR | Decoupling Primitive with Experts Dynamic Feature Alignment for Compositional Zero-Shot Learning | [paper](https://openreview.net/pdf?id=hUtTGobe1r) | - | CZSL | primitive ambiguity, MoE, expert alignment |
| 2026 | IJCV | MMPR: Multi-Modal Primitive Retrieval for Compositional Zero-Shot Learning | [paper](https://ieeexplore.ieee.org/document/11027124) | - | CZSL | retrieval, primitive representation |
| 2026 | AAAI | I2CD: An Invertible Causal Framework for Compositional Zero-Shot Learning via Disentangle-Compose-Disentangle | [paper](https://ojs.aaai.org/index.php/AAAI/article/view/38221) | - | CZSL | disentanglement, causality, context |
| 2026 | CVPR | FlowComposer: Composable Flows for Compositional Zero-Shot Learning | [paper](https://arxiv.org/pdf/2603.03815) | [code](https://hkust-longgroup.github.io/FlowComposer/) | CZSL | flow matching, disentanglement |
| 2025 | EAAI | A Novel Compositional Zero-Shot Learning Approach Based on Hierarchical Multi-Scale Feature Fusion | [paper](https://www.sciencedirect.com/science/article/pii/S0952197625016355) | - | CZSL | multi-scale feature fusion, patch selection |
| 2025 | ACM MM | Compositional Zero-Shot Learning via Progressive Language-based Observations | [paper](https://dl.acm.org/doi/abs/10.1145/3746027.3755208) | - | CZSL | language observation, context |
| 2025 | IJCAI | MSCI: Addressing CLIP's Inherent Limitations for Compositional Zero-Shot Learning | [paper](https://www.ijcai.org/proceedings/2025/224) | - | VLM-based CZSL | CLIP, intermediate visual features |
| 2025 | ICCV | A Conditional Probability Framework for Compositional Zero-shot Learning | [paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Wu_A_Conditional_Probability_Framework_for_Compositional_Zero-shot_Learning_ICCV_2025_paper.pdf) | - | CZSL | conditional probability, semantic constraint, context |
| 2025 | ICCV | Learning Visual Proxy for Compositional Zero-Shot Learning | [paper](https://arxiv.org/pdf/2501.13859) | [code](https://github.com/codefish12-09/VP_CMJL) | CZSL | modality gap, visual proxy |
| 2025 | NeurIPS | TOMCAT: Test-time Comprehensive Knowledge Accumulation for Compositional Zero-Shot Learning | [paper](https://neurips.cc/virtual/2025/loc/san-diego/poster/117606) | [code](https://github.com/xud-yan/TOMCAT) | CZSL | test-time adaptation, knowledge accumulation |
| 2025 | CVPR | LOGICZSL: Exploring Logic-induced Representation for Compositional Zero-shot Learning | [paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_LOGICZSL_Exploring_Logic-induced_Representation_for_Compositional_Zero-shot_Learning_CVPR_2025_paper.pdf) | - | CZSL | logic-induced representation, semantic hierarchy |
| 2025 | ICLR | ClusPro: Learning Clustering-based Prototypes for Compositional Zero-shot Learning | [paper](https://arxiv.org/abs/2502.06501) | - | CZSL | multi-prototype learning, clustering, HSIC |
| 2024 | CVPR | Troika: Multi-Path Cross-Modal Traction for Compositional Zero-Shot Learning | [paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_Troika_Multi-Path_Cross-Modal_Traction_for_Compositional_Zero-Shot_Learning_CVPR_2024_paper.pdf) | [code](https://github.com/bighuang624/Troika) | VLM-based CZSL | cross-modal traction, context |
| 2024 | CVPR | Beyond Seen Primitive Concepts for Attribute-Object Compositional Learning | [paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Saini_Beyond_Seen_Primitive_Concepts_and_Attribute-Object_Compositional_Learning_CVPR_2024_paper.pdf) | [code](https://github.com/ov-czsl/ov-czsl) | OV-CZSL | unseen primitive concepts, open vocabulary |
| 2024 | ECCV | PLID: Prompting Language-Informed Distribution for Compositional Zero-Shot Learning | [paper](https://arxiv.org/pdf/2305.14428) | [code](https://github.com/Cogito2012/PLID) | CZSL | prompt learning, language-informed distribution |
| 2024 | IEEE | Learning Conditional Prompt for Compositional Zero-Shot Learning | [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10688263) | - | CZSL | conditional prompt |
| 2024 | TPAMI | LIDF: Disentangling before Composing: Learning Invariant Disentangled Features for Compositional Zero-Shot Learning | [paper](https://ieeexplore.ieee.org/document/10737100) | - | CZSL | invariant disentangled features |

## Preprints and Workshops

| Year | Venue | Title | Paper | Code | Keywords |
|---|---|---|---|---|---|
| 2026 | arXiv | Progressive Reasoning with Primitive Correction for Compositional Zero-Shot Learning | [paper](https://arxiv.org/pdf/2607.05911) | - | primitive correction, reasoning |
| 2026 | CompLearn Workshop at ICML | LGPRO: Language-Guided Prototype Discovery for Compositional Zero-Shot Learning | [paper](https://openreview.net/forum?id=WvvVykgPDL) | - | language-guided prototype discovery |
| 2026 | arXiv | H2EM: Learning Hierarchical Hyperbolic Embeddings for Compositional Zero-Shot Learning | [paper](https://arxiv.org/pdf/2512.20029) | - | hyperbolic embedding, semantic hierarchy |
| 2026 | arXiv | Structure-aware Prompt Adaptation from Seen to Unseen for Open-Vocabulary Compositional Zero-Shot Learning | [paper](https://arxiv.org/pdf/2603.03815) | [code](https://github.com/ZHlo-404/SPA) | OV-CZSL, prompt adaptation |
| 2025 | arXiv | CAMS: Towards Compositional Zero-Shot Learning via Gated Cross-Attention and Multi-Space Disentanglement | [paper](https://arxiv.org/pdf/2511.16378) | [code](https://github.com/ybyangjing/CAMS) | gated cross-attention, multi-space disentanglement |

## How to Contribute

Pull requests are welcome. Please include the paper title, venue, year, paper link, code link if available, and a short keyword summary.

## Acknowledgement

This list is inspired by earlier CZSL paper collections and is intended to focus on recent work from 2024 to 2026.


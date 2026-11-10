---
id: M-DL-03
title: 神经网络训练与优化算法
type: module
notion_page_id: 410663e8-0333-8167-8029-ed29dc5d00c6
notion_url: https://app.notion.com/p/410663e8033381678029ed29dc5d00c6
notion_db: 课程模块
学科: 深度学习
排序: 3
当前状态: 未开始
学习目标: Epoch/Batch/学习率，以及 SGD/Momentum/Adam 的差别与选用条件。
所属路线: [RM-005]
知识点: [DL-023, DL-024, DL-025, DL-026, DL-027, DL-028, DL-029, DL-030, DL-031]
tags: [深度学习]
---

# M-DL-03 神经网络训练与优化算法

> 本笔记由 Notion「课程模块」数据库导出生成（Obsidian 侧原无对应笔记）。

## 学习目标

Epoch/Batch/学习率，以及 SGD/Momentum/Adam 的差别与选用条件。

---

## 🔗 关联

> 以下内容由 Notion「AI Knowledge System · 课程模块」数据库按知识 ID 合并而来（导出于 2026-11-09）。

| 属性 | 值 |
|---|---|
| 学科 | 深度学习 |
| 排序 | 3 |
| 当前状态 | 未开始 |

- **所属路线**：[RM-005 深度学习路线](../../00_%E7%9F%A5%E8%AF%86%E9%A6%96%E9%A1%B5/%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF/RM-005%20%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF.md)
- **知识点**：[DL-023 神经网络训练的完整流程](../05%20%E8%AE%AD%E7%BB%83%E6%8A%80%E5%B7%A7/DL-023%20%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E8%AE%AD%E7%BB%83%E7%9A%84%E5%AE%8C%E6%95%B4%E6%B5%81%E7%A8%8B.md) · [DL-024 Epoch、Batch 与 Mini-batch](../05%20%E8%AE%AD%E7%BB%83%E6%8A%80%E5%B7%A7/DL-024%20Epoch%E3%80%81Batch%20%E4%B8%8E%20Mini-batch.md) · [DL-025 Batch Size 与 Learning Rate](../05%20%E8%AE%AD%E7%BB%83%E6%8A%80%E5%B7%A7/DL-025%20Batch%20Size%20%E4%B8%8E%20Learning%20Rate.md) · [DL-026 SGD 与 Momentum](../05%20%E8%AE%AD%E7%BB%83%E6%8A%80%E5%B7%A7/DL-026%20SGD%20%E4%B8%8E%20Momentum.md) · [DL-027 Nesterov、AdaGrad 与 RMSprop](../05%20%E8%AE%AD%E7%BB%83%E6%8A%80%E5%B7%A7/DL-027%20Nesterov%E3%80%81AdaGrad%20%E4%B8%8E%20RMSprop.md) · [DL-028 Adam 与 AdamW](../05%20%E8%AE%AD%E7%BB%83%E6%8A%80%E5%B7%A7/DL-028%20Adam%20%E4%B8%8E%20AdamW.md) · [DL-029 学习率调度与 Warmup](../05%20%E8%AE%AD%E7%BB%83%E6%8A%80%E5%B7%A7/DL-029%20%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E5%BA%A6%E4%B8%8E%20Warmup.md) · [DL-030 权重初始化：Xavier 与 He](../05%20%E8%AE%AD%E7%BB%83%E6%8A%80%E5%B7%A7/DL-030%20%E6%9D%83%E9%87%8D%E5%88%9D%E5%A7%8B%E5%8C%96%E2%80%94%E2%80%94Xavier%20%E4%B8%8E%20He.md) · [DL-031 梯度裁剪与训练稳定性](../05%20%E8%AE%AD%E7%BB%83%E6%8A%80%E5%B7%A7/DL-031%20%E6%A2%AF%E5%BA%A6%E8%A3%81%E5%89%AA%E4%B8%8E%E8%AE%AD%E7%BB%83%E7%A8%B3%E5%AE%9A%E6%80%A7.md)

---
id: M-DL-04
title: 模型泛化、正则化与训练技巧
type: module
notion_page_id: 410663e8-0333-81d4-95b2-e94ba744b423
notion_url: https://app.notion.com/p/410663e8033381d495b2e94ba744b423
notion_db: 课程模块
学科: 深度学习
排序: 4
当前状态: 未开始
学习目标: 过拟合怎么识别、Dropout/BN/LN 各自解决什么问题、如何做可复现实验。
所属路线: [RM-005]
知识点: [DL-032, DL-033, DL-034, DL-035, DL-036, DL-037, DL-038, DL-039]
tags: [深度学习]
---

# M-DL-04 模型泛化、正则化与训练技巧

> 本笔记由 Notion「课程模块」数据库导出生成（Obsidian 侧原无对应笔记）。

## 学习目标

过拟合怎么识别、Dropout/BN/LN 各自解决什么问题、如何做可复现实验。

---

## 🔗 关联

> 以下内容由 Notion「AI Knowledge System · 课程模块」数据库按知识 ID 合并而来（导出于 2026-11-09）。

| 属性 | 值 |
|---|---|
| 学科 | 深度学习 |
| 排序 | 4 |
| 当前状态 | 未开始 |

- **所属路线**：[RM-005 深度学习路线](../../00_%E7%9F%A5%E8%AF%86%E9%A6%96%E9%A1%B5/%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF/RM-005%20%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF.md)
- **知识点**：[DL-032 过拟合与欠拟合（深度学习视角）](../05%20%E8%AE%AD%E7%BB%83%E6%8A%80%E5%B7%A7/DL-032%20%E8%BF%87%E6%8B%9F%E5%90%88%E4%B8%8E%E6%AC%A0%E6%8B%9F%E5%90%88%E2%80%94%E2%80%94%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E8%A7%86%E8%A7%92.md) · [DL-033 偏差-方差与泛化能力](../05%20%E8%AE%AD%E7%BB%83%E6%8A%80%E5%B7%A7/DL-033%20%E5%81%8F%E5%B7%AE-%E6%96%B9%E5%B7%AE%E4%B8%8E%E6%B3%9B%E5%8C%96%E8%83%BD%E5%8A%9B.md) · [DL-034 L1/L2 正则化与 Weight Decay](../05%20%E8%AE%AD%E7%BB%83%E6%8A%80%E5%B7%A7/DL-034%20L1%E3%80%81L2%20%E6%AD%A3%E5%88%99%E5%8C%96%E4%B8%8E%20Weight%20Decay.md) · [DL-035 Dropout](../05%20%E8%AE%AD%E7%BB%83%E6%8A%80%E5%B7%A7/DL-035%20Dropout.md) · [DL-036 Batch Normalization](../05%20%E8%AE%AD%E7%BB%83%E6%8A%80%E5%B7%A7/DL-036%20Batch%20Normalization.md) · [DL-037 Layer Normalization 与 BN 的区别](../05%20%E8%AE%AD%E7%BB%83%E6%8A%80%E5%B7%A7/DL-037%20Layer%20Normalization%20%E4%B8%8E%20BN%20%E7%9A%84%E5%8C%BA%E5%88%AB.md) · [DL-038 Early Stopping 与模型检查点](../05%20%E8%AE%AD%E7%BB%83%E6%8A%80%E5%B7%A7/DL-038%20Early%20Stopping%20%E4%B8%8E%E6%A8%A1%E5%9E%8B%E6%A3%80%E6%9F%A5%E7%82%B9.md) · [DL-039 数据增强、训练日志与可复现性](../05%20%E8%AE%AD%E7%BB%83%E6%8A%80%E5%B7%A7/DL-039%20%E6%95%B0%E6%8D%AE%E5%A2%9E%E5%BC%BA%E3%80%81%E8%AE%AD%E7%BB%83%E6%97%A5%E5%BF%97%E4%B8%8E%E5%8F%AF%E5%A4%8D%E7%8E%B0%E6%80%A7.md)

---
id: M-LLM-02
title: Attention 机制与数学原理
type: module
notion_page_id: 411663e8-0333-8155-8f26-c93869a86df7
notion_url: https://app.notion.com/p/411663e8033381558f26c93869a86df7
notion_db: 课程模块
学科: 大语言模型
排序: 2
当前状态: 未开始
学习目标: QKV、缩放点积、多头、掩码：把 Attention 的每一步算清楚，不只背公式。
所属路线: [RM-006]
知识点: [LLM-011, LLM-012, LLM-013, LLM-014, LLM-015, LLM-016, LLM-017, LLM-018, LLM-019, LLM-020]
tags: [大语言模型]
---

# M-LLM-02 Attention 机制与数学原理

> 本笔记由 Notion「课程模块」数据库导出生成（Obsidian 侧原无对应笔记）。

## 学习目标

QKV、缩放点积、多头、掩码：把 Attention 的每一步算清楚，不只背公式。

---

## 🔗 关联

> 以下内容由 Notion「AI Knowledge System · 课程模块」数据库按知识 ID 合并而来（导出于 2026-11-09）。

| 属性 | 值 |
|---|---|
| 学科 | 大语言模型 |
| 排序 | 2 |
| 当前状态 | 未开始 |

- **所属路线**：[RM-006 大语言模型路线](../../00_%E7%9F%A5%E8%AF%86%E9%A6%96%E9%A1%B5/%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF/RM-006%20%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E8%B7%AF%E7%BA%BF.md)
- **知识点**：[LLM-011 Attention 的基本思想](../03%20Attention%E6%9C%BA%E5%88%B6/LLM-011%20Attention%20%E7%9A%84%E5%9F%BA%E6%9C%AC%E6%80%9D%E6%83%B3.md) · [LLM-012 Query、Key、Value 与 QKV 的来源](../03%20Attention%E6%9C%BA%E5%88%B6/LLM-012%20Query%E3%80%81Key%E3%80%81Value%20%E4%B8%8E%20QKV%20%E7%9A%84%E6%9D%A5%E6%BA%90.md) · [LLM-013 点积注意力与 Scaled Dot-Product Attention](../03%20Attention%E6%9C%BA%E5%88%B6/LLM-013%20%E7%82%B9%E7%A7%AF%E6%B3%A8%E6%84%8F%E5%8A%9B%E4%B8%8E%20Scaled%20Dot-Product%20Attention.md) · [LLM-014 Softmax 在 Attention 中的作用](../03%20Attention%E6%9C%BA%E5%88%B6/LLM-014%20Softmax%20%E5%9C%A8%20Attention%20%E4%B8%AD%E7%9A%84%E4%BD%9C%E7%94%A8.md) · [LLM-015 Self-Attention](../03%20Attention%E6%9C%BA%E5%88%B6/LLM-015%20Self-Attention.md) · [LLM-016 Cross-Attention](../03%20Attention%E6%9C%BA%E5%88%B6/LLM-016%20Cross-Attention.md) · [LLM-017 Multi-Head Attention](../03%20Attention%E6%9C%BA%E5%88%B6/LLM-017%20Multi-Head%20Attention.md) · [LLM-018 掩码：Causal、Padding 与 Attention Mask](../03%20Attention%E6%9C%BA%E5%88%B6/LLM-018%20%E6%8E%A9%E7%A0%81%E2%80%94%E2%80%94Causal%E3%80%81Padding%20%E4%B8%8E%20Attention%20Mask.md) · [LLM-019 Attention 的张量维度与实现细节](../03%20Attention%E6%9C%BA%E5%88%B6/LLM-019%20Attention%20%E7%9A%84%E5%BC%A0%E9%87%8F%E7%BB%B4%E5%BA%A6%E4%B8%8E%E5%AE%9E%E7%8E%B0%E7%BB%86%E8%8A%82.md) · [LLM-020 Attention 的计算复杂度与局限](../03%20Attention%E6%9C%BA%E5%88%B6/LLM-020%20Attention%20%E7%9A%84%E8%AE%A1%E7%AE%97%E5%A4%8D%E6%9D%82%E5%BA%A6%E4%B8%8E%E5%B1%80%E9%99%90.md)

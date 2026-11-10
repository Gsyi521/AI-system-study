---
id: M-LLM-03
title: Transformer 完整架构
type: module
notion_page_id: 411663e8-0333-819a-89a7-c974d0c45442
notion_url: https://app.notion.com/p/411663e80333819a89a7c974d0c45442
notion_db: 课程模块
学科: 大语言模型
排序: 3
当前状态: 未开始
学习目标: Encoder/Decoder/Block 的数据流与维度变化，含残差、LayerNorm、FFN、位置编码。
所属路线: [RM-006]
知识点: [LLM-021, LLM-022, LLM-023, LLM-024, LLM-025, LLM-026, LLM-027, LLM-028, LLM-029, LLM-030]
tags: [大语言模型]
---

# M-LLM-03 Transformer 完整架构

> 本笔记由 Notion「课程模块」数据库导出生成（Obsidian 侧原无对应笔记）。

## 学习目标

Encoder/Decoder/Block 的数据流与维度变化，含残差、LayerNorm、FFN、位置编码。

---

## 🔗 关联

> 以下内容由 Notion「AI Knowledge System · 课程模块」数据库按知识 ID 合并而来（导出于 2026-11-09）。

| 属性 | 值 |
|---|---|
| 学科 | 大语言模型 |
| 排序 | 3 |
| 当前状态 | 未开始 |

- **所属路线**：[RM-006 大语言模型路线](../../00_%E7%9F%A5%E8%AF%86%E9%A6%96%E9%A1%B5/%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF/RM-006%20%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E8%B7%AF%E7%BA%BF.md)
- **知识点**：[LLM-021 Transformer 总体架构](../04%20Transformer%E6%9E%B6%E6%9E%84/LLM-021%20Transformer%20%E6%80%BB%E4%BD%93%E6%9E%B6%E6%9E%84.md) · [LLM-022 Encoder 与 Encoder Block](../04%20Transformer%E6%9E%B6%E6%9E%84/LLM-022%20Encoder%20%E4%B8%8E%20Encoder%20Block.md) · [LLM-023 Decoder 与 Decoder Block](../04%20Transformer%E6%9E%B6%E6%9E%84/LLM-023%20Decoder%20%E4%B8%8E%20Decoder%20Block.md) · [LLM-024 位置编码](../04%20Transformer%E6%9E%B6%E6%9E%84/LLM-024%20%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81.md) · [LLM-025 残差、LayerNorm 与 Pre-LN/Post-LN](../04%20Transformer%E6%9E%B6%E6%9E%84/LLM-025%20%E6%AE%8B%E5%B7%AE%E3%80%81LayerNorm%20%E4%B8%8E%20Pre-LN%E3%80%81Post-LN.md) · [LLM-026 Multi-Head 的变体：MQA 与 GQA](../04%20Transformer%E6%9E%B6%E6%9E%84/LLM-026%20Multi-Head%20%E7%9A%84%E5%8F%98%E4%BD%93%E2%80%94%E2%80%94MQA%20%E4%B8%8E%20GQA.md) · [LLM-027 Encoder-Only、Decoder-Only 与 Encoder-Decoder](../04%20Transformer%E6%9E%B6%E6%9E%84/LLM-027%20Encoder-Only%E3%80%81Decoder-Only%20%E4%B8%8E%20Encoder-Decoder.md) · [LLM-028 RoPE 与长上下文外推](../04%20Transformer%E6%9E%B6%E6%9E%84/LLM-028%20RoPE%20%E4%B8%8E%E9%95%BF%E4%B8%8A%E4%B8%8B%E6%96%87%E5%A4%96%E6%8E%A8.md) · [LLM-029 FFN、激活函数与 SwiGLU](../04%20Transformer%E6%9E%B6%E6%9E%84/LLM-029%20FFN%E3%80%81%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%E4%B8%8E%20SwiGLU.md) · [LLM-030 Transformer 参数量与计算量估算](../04%20Transformer%E6%9E%B6%E6%9E%84/LLM-030%20Transformer%20%E5%8F%82%E6%95%B0%E9%87%8F%E4%B8%8E%E8%AE%A1%E7%AE%97%E9%87%8F%E4%BC%B0%E7%AE%97.md)

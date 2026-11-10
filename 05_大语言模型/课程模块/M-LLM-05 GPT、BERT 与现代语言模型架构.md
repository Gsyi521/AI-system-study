---
id: M-LLM-05
title: GPT、BERT 与现代语言模型架构
type: module
notion_page_id: 411663e8-0333-813d-80a3-e97e26c28b03
notion_url: https://app.notion.com/p/411663e80333813d80a3e97e26c28b03
notion_db: 课程模块
学科: 大语言模型
排序: 5
当前状态: 未开始
学习目标: Encoder-only / Decoder-only / Encoder-Decoder 的训练目标与适用场景。
所属路线: [RM-006]
知识点: [LLM-042, LLM-043, LLM-044, LLM-045, LLM-046, LLM-047, LLM-048, LLM-049, LLM-050, LLM-051, LLM-052]
实战项目: [PROJ-021]
tags: [大语言模型]
---

# M-LLM-05 GPT、BERT 与现代语言模型架构

> 本笔记由 Notion「课程模块」数据库导出生成（Obsidian 侧原无对应笔记）。

## 学习目标

Encoder-only / Decoder-only / Encoder-Decoder 的训练目标与适用场景。

---

## 🔗 关联

> 以下内容由 Notion「AI Knowledge System · 课程模块」数据库按知识 ID 合并而来（导出于 2026-11-09）。

| 属性 | 值 |
|---|---|
| 学科 | 大语言模型 |
| 排序 | 5 |
| 当前状态 | 未开始 |

- **所属路线**：[RM-006 大语言模型路线](../../00_%E7%9F%A5%E8%AF%86%E9%A6%96%E9%A1%B5/%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF/RM-006%20%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E8%B7%AF%E7%BA%BF.md)
- **知识点**：[LLM-042 BERT 的预训练任务](../05%20%E9%A2%84%E8%AE%AD%E7%BB%83%E4%B8%8E%E5%BE%AE%E8%B0%83/LLM-042%20BERT%20%E7%9A%84%E9%A2%84%E8%AE%AD%E7%BB%83%E4%BB%BB%E5%8A%A1.md) · [LLM-043 GPT 系列的演进](../05%20%E9%A2%84%E8%AE%AD%E7%BB%83%E4%B8%8E%E5%BE%AE%E8%B0%83/LLM-043%20GPT%20%E7%B3%BB%E5%88%97%E7%9A%84%E6%BC%94%E8%BF%9B.md) · [LLM-044 T5 与 Span Corruption](../05%20%E9%A2%84%E8%AE%AD%E7%BB%83%E4%B8%8E%E5%BE%AE%E8%B0%83/LLM-044%20T5%20%E4%B8%8E%20Span%20Corruption.md) · [LLM-045 预训练目标对比：MLM、CLM 与 Span Corruption](../05%20%E9%A2%84%E8%AE%AD%E7%BB%83%E4%B8%8E%E5%BE%AE%E8%B0%83/LLM-045%20%E9%A2%84%E8%AE%AD%E7%BB%83%E7%9B%AE%E6%A0%87%E5%AF%B9%E6%AF%94%EF%BC%9AMLM%E3%80%81CLM%20%E4%B8%8E%20Span%20Corruption.md) · [LLM-046 双向上下文与单向生成](../05%20%E9%A2%84%E8%AE%AD%E7%BB%83%E4%B8%8E%E5%BE%AE%E8%B0%83/LLM-046%20%E5%8F%8C%E5%90%91%E4%B8%8A%E4%B8%8B%E6%96%87%E4%B8%8E%E5%8D%95%E5%90%91%E7%94%9F%E6%88%90.md) · [LLM-047 上下文窗口与位置外推](../05%20%E9%A2%84%E8%AE%AD%E7%BB%83%E4%B8%8E%E5%BE%AE%E8%B0%83/LLM-047%20%E4%B8%8A%E4%B8%8B%E6%96%87%E7%AA%97%E5%8F%A3%E4%B8%8E%E4%BD%8D%E7%BD%AE%E5%A4%96%E6%8E%A8.md) · [LLM-048 微调范式：全参与参数高效](../05%20%E9%A2%84%E8%AE%AD%E7%BB%83%E4%B8%8E%E5%BE%AE%E8%B0%83/LLM-048%20%E5%BE%AE%E8%B0%83%E8%8C%83%E5%BC%8F%EF%BC%9A%E5%85%A8%E5%8F%82%E4%B8%8E%E5%8F%82%E6%95%B0%E9%AB%98%E6%95%88.md) · [LLM-049 零样本、少样本与上下文学习](../05%20%E9%A2%84%E8%AE%AD%E7%BB%83%E4%B8%8E%E5%BE%AE%E8%B0%83/LLM-049%20%E9%9B%B6%E6%A0%B7%E6%9C%AC%E3%80%81%E5%B0%91%E6%A0%B7%E6%9C%AC%E4%B8%8E%E4%B8%8A%E4%B8%8B%E6%96%87%E5%AD%A6%E4%B9%A0.md) · [LLM-050 提示的结构与格式敏感性](../05%20%E9%A2%84%E8%AE%AD%E7%BB%83%E4%B8%8E%E5%BE%AE%E8%B0%83/LLM-050%20%E6%8F%90%E7%A4%BA%E7%9A%84%E7%BB%93%E6%9E%84%E4%B8%8E%E6%A0%BC%E5%BC%8F%E6%95%8F%E6%84%9F%E6%80%A7.md) · [LLM-051 模型规模与能力的关系](../05%20%E9%A2%84%E8%AE%AD%E7%BB%83%E4%B8%8E%E5%BE%AE%E8%B0%83/LLM-051%20%E6%A8%A1%E5%9E%8B%E8%A7%84%E6%A8%A1%E4%B8%8E%E8%83%BD%E5%8A%9B%E7%9A%84%E5%85%B3%E7%B3%BB.md) · [LLM-052 现代 Decoder-only 架构的差异](../05%20%E9%A2%84%E8%AE%AD%E7%BB%83%E4%B8%8E%E5%BE%AE%E8%B0%83/LLM-052%20%E7%8E%B0%E4%BB%A3%20Decoder-only%20%E6%9E%B6%E6%9E%84%E7%9A%84%E5%B7%AE%E5%BC%82.md)
- **实战项目**：[PROJ-021 从零实现一个 Transformer 语言模型](../../07_%E5%AE%9E%E6%88%98%E9%A1%B9%E7%9B%AE/05%20%E5%A4%A7%E6%A8%A1%E5%9E%8B%E9%A1%B9%E7%9B%AE/PROJ-021%20%E4%BB%8E%E9%9B%B6%E5%AE%9E%E7%8E%B0%E4%B8%80%E4%B8%AA%20Transformer%20%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.md)

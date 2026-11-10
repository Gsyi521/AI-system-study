---
id: M-LLM-04
title: Tokenizer、Embedding 与文本表示
type: module
notion_page_id: 411663e8-0333-81cb-b31c-eeb347861e32
notion_url: https://app.notion.com/p/411663e8033381cbb31ceeb347861e32
notion_db: 课程模块
学科: 大语言模型
排序: 4
当前状态: 未开始
学习目标: BPE/WordPiece/SentencePiece、词表与特殊符号、为什么 token 数决定成本。
所属路线: [RM-006]
知识点: [LLM-031, LLM-032, LLM-033, LLM-034, LLM-035, LLM-036, LLM-037, LLM-038, LLM-039, LLM-040, LLM-041]
tags: [大语言模型]
---

# M-LLM-04 Tokenizer、Embedding 与文本表示

> 本笔记由 Notion「课程模块」数据库导出生成（Obsidian 侧原无对应笔记）。

## 学习目标

BPE/WordPiece/SentencePiece、词表与特殊符号、为什么 token 数决定成本。

---

## 🔗 关联

> 以下内容由 Notion「AI Knowledge System · 课程模块」数据库按知识 ID 合并而来（导出于 2026-11-09）。

| 属性 | 值 |
|---|---|
| 学科 | 大语言模型 |
| 排序 | 4 |
| 当前状态 | 未开始 |

- **所属路线**：[RM-006 大语言模型路线](../../00_%E7%9F%A5%E8%AF%86%E9%A6%96%E9%A1%B5/%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF/RM-006%20%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E8%B7%AF%E7%BA%BF.md)
- **知识点**：[LLM-031 Tokenizer 基础与子词切分](../02%20Token%E4%B8%8EEmbedding/LLM-031%20Tokenizer%20%E5%9F%BA%E7%A1%80%E4%B8%8E%E5%AD%90%E8%AF%8D%E5%88%87%E5%88%86.md) · [LLM-032 BPE 算法](../02%20Token%E4%B8%8EEmbedding/LLM-032%20BPE%20%E7%AE%97%E6%B3%95.md) · [LLM-033 WordPiece 与 SentencePiece](../02%20Token%E4%B8%8EEmbedding/LLM-033%20WordPiece%20%E4%B8%8E%20SentencePiece.md) · [LLM-034 特殊 Token 与 Attention Mask](../02%20Token%E4%B8%8EEmbedding/LLM-034%20%E7%89%B9%E6%AE%8A%20Token%20%E4%B8%8E%20Attention%20Mask.md) · [LLM-035 Token 数量、上下文窗口与成本](../02%20Token%E4%B8%8EEmbedding/LLM-035%20Token%20%E6%95%B0%E9%87%8F%E3%80%81%E4%B8%8A%E4%B8%8B%E6%96%87%E7%AA%97%E5%8F%A3%E4%B8%8E%E6%88%90%E6%9C%AC.md) · [LLM-036 Embedding 与向量表示](../02%20Token%E4%B8%8EEmbedding/LLM-036%20Embedding%20%E4%B8%8E%E5%90%91%E9%87%8F%E8%A1%A8%E7%A4%BA.md) · [LLM-037 语义空间与相似度](../02%20Token%E4%B8%8EEmbedding/LLM-037%20%E8%AF%AD%E4%B9%89%E7%A9%BA%E9%97%B4%E4%B8%8E%E7%9B%B8%E4%BC%BC%E5%BA%A6.md) · [LLM-038 从静态词向量到上下文向量](../02%20Token%E4%B8%8EEmbedding/LLM-038%20%E4%BB%8E%E9%9D%99%E6%80%81%E8%AF%8D%E5%90%91%E9%87%8F%E5%88%B0%E4%B8%8A%E4%B8%8B%E6%96%87%E5%90%91%E9%87%8F.md) · [LLM-039 Embedding 与模型参数](../02%20Token%E4%B8%8EEmbedding/LLM-039%20Embedding%20%E4%B8%8E%E6%A8%A1%E5%9E%8B%E5%8F%82%E6%95%B0.md) · [LLM-040 Token 嵌入与位置嵌入的区别](../02%20Token%E4%B8%8EEmbedding/LLM-040%20Token%20%E5%B5%8C%E5%85%A5%E4%B8%8E%E4%BD%8D%E7%BD%AE%E5%B5%8C%E5%85%A5%E7%9A%84%E5%8C%BA%E5%88%AB.md) · [LLM-041 多语言 Tokenizer 的取舍](../02%20Token%E4%B8%8EEmbedding/LLM-041%20%E5%A4%9A%E8%AF%AD%E8%A8%80%20Tokenizer%20%E7%9A%84%E5%8F%96%E8%88%8D.md)

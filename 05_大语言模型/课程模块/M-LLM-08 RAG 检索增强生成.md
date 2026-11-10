---
id: M-LLM-08
title: RAG 检索增强生成
type: module
notion_page_id: 411663e8-0333-8108-b62d-e0a01beef38e
notion_url: https://app.notion.com/p/411663e803338108b62de0a01beef38e
notion_db: 课程模块
学科: 大语言模型
排序: 8
当前状态: 未开始
学习目标: 切分、Embedding、向量检索、重排、上下文构建与评估的完整工程。
所属路线: [RM-006]
知识点: [LLM-076, LLM-077, LLM-078, LLM-079, LLM-080, LLM-081, LLM-082, LLM-083, LLM-084, LLM-085, LLM-086, LLM-087]
实战项目: [PROJ-023, PROJ-026]
tags: [大语言模型]
---

# M-LLM-08 RAG 检索增强生成

> 本笔记由 Notion「课程模块」数据库导出生成（Obsidian 侧原无对应笔记）。

## 学习目标

切分、Embedding、向量检索、重排、上下文构建与评估的完整工程。

---

## 🔗 关联

> 以下内容由 Notion「AI Knowledge System · 课程模块」数据库按知识 ID 合并而来（导出于 2026-11-09）。

| 属性 | 值 |
|---|---|
| 学科 | 大语言模型 |
| 排序 | 8 |
| 当前状态 | 未开始 |

- **所属路线**：[RM-006 大语言模型路线](../../00_%E7%9F%A5%E8%AF%86%E9%A6%96%E9%A1%B5/%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF/RM-006%20%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E8%B7%AF%E7%BA%BF.md)
- **知识点**：[LLM-076 RAG 的基本架构](../07%20RAG%E4%B8%8EAgent/LLM-076%20RAG%20%E7%9A%84%E5%9F%BA%E6%9C%AC%E6%9E%B6%E6%9E%84.md) · [LLM-077 文档切块策略](../07%20RAG%E4%B8%8EAgent/LLM-077%20%E6%96%87%E6%A1%A3%E5%88%87%E5%9D%97%E7%AD%96%E7%95%A5.md) · [LLM-078 向量化与嵌入模型](../07%20RAG%E4%B8%8EAgent/LLM-078%20%E5%90%91%E9%87%8F%E5%8C%96%E4%B8%8E%E5%B5%8C%E5%85%A5%E6%A8%A1%E5%9E%8B.md) · [LLM-079 向量检索与相似度索引](../07%20RAG%E4%B8%8EAgent/LLM-079%20%E5%90%91%E9%87%8F%E6%A3%80%E7%B4%A2%E4%B8%8E%E7%9B%B8%E4%BC%BC%E5%BA%A6%E7%B4%A2%E5%BC%95.md) · [LLM-080 混合检索：BM25 与向量](../07%20RAG%E4%B8%8EAgent/LLM-080%20%E6%B7%B7%E5%90%88%E6%A3%80%E7%B4%A2%EF%BC%9ABM25%20%E4%B8%8E%E5%90%91%E9%87%8F.md) · [LLM-081 Rerank 与检索结果精排](../07%20RAG%E4%B8%8EAgent/LLM-081%20Rerank%20%E4%B8%8E%E6%A3%80%E7%B4%A2%E7%BB%93%E6%9E%9C%E7%B2%BE%E6%8E%92.md) · [LLM-082 提示词组装与上下文预算](../07%20RAG%E4%B8%8EAgent/LLM-082%20%E6%8F%90%E7%A4%BA%E8%AF%8D%E7%BB%84%E8%A3%85%E4%B8%8E%E4%B8%8A%E4%B8%8B%E6%96%87%E9%A2%84%E7%AE%97.md) · [LLM-083 RAG 的评估](../07%20RAG%E4%B8%8EAgent/LLM-083%20RAG%20%E7%9A%84%E8%AF%84%E4%BC%B0.md) · [LLM-084 RAG 的失败模式与修复](../07%20RAG%E4%B8%8EAgent/LLM-084%20RAG%20%E7%9A%84%E5%A4%B1%E8%B4%A5%E6%A8%A1%E5%BC%8F%E4%B8%8E%E4%BF%AE%E5%A4%8D.md) · [LLM-085 Agent 的基本循环](../07%20RAG%E4%B8%8EAgent/LLM-085%20Agent%20%E7%9A%84%E5%9F%BA%E6%9C%AC%E5%BE%AA%E7%8E%AF.md) · [LLM-086 工具调用与函数调用](../07%20RAG%E4%B8%8EAgent/LLM-086%20%E5%B7%A5%E5%85%B7%E8%B0%83%E7%94%A8%E4%B8%8E%E5%87%BD%E6%95%B0%E8%B0%83%E7%94%A8.md) · [LLM-087 Agent 的规划、记忆与多步执行](../07%20RAG%E4%B8%8EAgent/LLM-087%20Agent%20%E7%9A%84%E8%A7%84%E5%88%92%E3%80%81%E8%AE%B0%E5%BF%86%E4%B8%8E%E5%A4%9A%E6%AD%A5%E6%89%A7%E8%A1%8C.md)
- **实战项目**：[PROJ-023 搭建一个最小可用的 RAG 问答系统](../../07_%E5%AE%9E%E6%88%98%E9%A1%B9%E7%9B%AE/05%20%E5%A4%A7%E6%A8%A1%E5%9E%8B%E9%A1%B9%E7%9B%AE/PROJ-023%20%E6%90%AD%E5%BB%BA%E4%B8%80%E4%B8%AA%E6%9C%80%E5%B0%8F%E5%8F%AF%E7%94%A8%E7%9A%84%20RAG%20%E9%97%AE%E7%AD%94%E7%B3%BB%E7%BB%9F.md) · [PROJ-026 端到端 LLM 应用：客服问答](../../07_%E5%AE%9E%E6%88%98%E9%A1%B9%E7%9B%AE/05%20%E5%A4%A7%E6%A8%A1%E5%9E%8B%E9%A1%B9%E7%9B%AE/PROJ-026%20%E7%AB%AF%E5%88%B0%E7%AB%AF%20LLM%20%E5%BA%94%E7%94%A8%EF%BC%9A%E5%AE%A2%E6%9C%8D%E9%97%AE%E7%AD%94.md)

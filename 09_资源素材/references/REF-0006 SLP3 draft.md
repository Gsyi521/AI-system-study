---
id: REF-0006
title: "Speech and Language Processing (3rd ed. draft)"
type: BOOK
authors: [Jurafsky, Martin]
year: 2024
venue: Stanford University (online draft)
arxiv_id: {arxiv}
isbn: ""
doi: ""
source_url: https://web.stanford.edu/~jurafsky/slp3/
local_path: ""
zotero_key: ""
subject: 大语言模型
importance: 参考
reading_status: 未读
related_courses: [M-LLM-01]
related_knowledge: [LLM-001, LLM-002, LLM-035]
related_projects: []
related_refs: [REF-0001]
tags: [nlp, 教材, 语言模型]
last_verified: 2026-11-09
verification_note: "离线录入：来源 URL 未在审计阶段访问（只读约束），首次联网时请复核并更新 last_verified"
notion_page_id: 413663e8-0333-8168-9d68-f30d54e7cb8a
notion_url: https://app.notion.com/p/SLP3-draft-413663e8033381689d68f30d54e7cb8a
created: 2026-11-09
updated: 2026-11-09
notion_db: 参考资料
学科: 大语言模型
类型: BOOK
重要度: 参考
阅读状态: 未读
作者: "Jurafsky, Martin"
年份: 2024
来源/出版方: Stanford University (online draft)
原文链接: "https://web.stanford.edu/~jurafsky/slp3/"
标签: [nlp, 教材, 语言模型]
最后核对: 2026-11-09
关联知识点: [LLM-001, LLM-002, LLM-035]
关联课程模块: [M-LLM-01]
关联资料: [REF-0001]
---

# REF-0006 Speech and Language Processing (3rd ed. draft)

> **类型**：BOOK ｜ **年份**：2024 ｜ **来源**：Stanford University (online draft)
> **原文**：<https://web.stanford.edu/~jurafsky/slp3/>
> **重要度**：参考 ｜ **阅读状态**：未读

## 1. 为什么读它（一句话）

NLP 与语言模型的系统教材，用来补课程之外的背景与术语。

## 2. 它解决什么问题

课程偏工程与实验，缺少完整的学科脉络。

## 3. 核心结论

1. 第 3 章 n-gram 语言模型给出平滑与困惑度的标准定义。
2. 第 7/8 章覆盖神经语言模型与 Transformer 的过渡。
3. 术语（perplexity、smoothing、tokenization）权威且稳定。

## 4. 与我的知识库的关系

LLM-002 的 n-gram 加一平滑实验（α=0.1→PPL 4.09）可用本书第 3 章的平滑框架解释。

- 关联课程模块：`[M-LLM-01]`
- 关联知识点：`[LLM-001, LLM-002, LLM-035]`
- 关联项目：`[]`
- 关联资料：`[REF-0001]`

## 5. 关键页 / 关键片段

Ch.3 N-gram Language Models；Ch.7 Neural Networks；Ch.8 Transformers

## 6. 我的疑问与反驳

在线 draft 持续更新，章节号可能变化；未记录具体版本日期。

## 7. 可复用的图表 / 公式

- `scaled dot-product attention`：$\mathrm{softmax}(QK^\top/\sqrt{d_k})V$（见 REF-0001 §3.2.1）
- 本资料可直接支撑的推导：见第 4 节列出的知识点。

## 8. 阅读记录

| 日期 | 动作 | 备注 |
|---|---|---|
| 2026-11-09 | 建条目（离线） | 元数据来自公开书目信息，URL 未访问 |

---

## 🔗 Notion 同步信息

> 以下内容由 Notion「AI Knowledge System · 参考资料」数据库按知识 ID 合并而来（导出于 2026-11-09）。

| 属性 | 值 |
|---|---|
| 学科 | 大语言模型 |
| 类型 | BOOK |
| 重要度 | 参考 |
| 阅读状态 | 未读 |
| 作者 | Jurafsky, Martin |
| 年份 | 2024 |
| 来源/出版方 | Stanford University (online draft) |
| 原文链接 | https://web.stanford.edu/~jurafsky/slp3/ |
| 标签 | nlp、教材、语言模型 |
| 最后核对 | 2026-11-09 |

- **关联知识点**：[LLM-001 什么是语言模型](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/01%20NLP%E5%9F%BA%E7%A1%80/LLM-001%20%E4%BB%80%E4%B9%88%E6%98%AF%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.md) · [LLM-002 统计语言模型与 N-gram](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/01%20NLP%E5%9F%BA%E7%A1%80/LLM-002%20%E7%BB%9F%E8%AE%A1%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E4%B8%8E%20N-gram.md) · [LLM-035 Token 数量、上下文窗口与成本](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/02%20Token%E4%B8%8EEmbedding/LLM-035%20Token%20%E6%95%B0%E9%87%8F%E3%80%81%E4%B8%8A%E4%B8%8B%E6%96%87%E7%AA%97%E5%8F%A3%E4%B8%8E%E6%88%90%E6%9C%AC.md)
- **关联课程模块**：[M-LLM-01 大语言模型基础与发展历史](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-LLM-01%20%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%9F%BA%E7%A1%80%E4%B8%8E%E5%8F%91%E5%B1%95%E5%8E%86%E5%8F%B2.md)
- **关联资料**：[REF-0001 Attention Is All You Need](REF-0001%20Attention%20Is%20All%20You%20Need.md)

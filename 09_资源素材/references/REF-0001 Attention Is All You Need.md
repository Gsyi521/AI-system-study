---
id: REF-0001
title: "Attention Is All You Need"
type: PAPER
authors: [Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin]
year: 2017
venue: NeurIPS
arxiv_id: 1706.03762
isbn: ""
doi: ""
source_url: https://arxiv.org/abs/1706.03762
local_path: ""
zotero_key: ""
subject: 大语言模型
importance: 核心
reading_status: 精读
related_courses: [M-LLM-02, M-LLM-03]
related_knowledge: [LLM-004, LLM-013, LLM-021]
related_projects: []
related_refs: []
tags: [transformer, attention, 原始论文]
last_verified: 2026-11-09
verification_note: "离线录入：来源 URL 未在审计阶段访问（只读约束），首次联网时请复核并更新 last_verified"
notion_page_id: 413663e8-0333-81fa-9ca3-ec4fe539a791
notion_url: https://app.notion.com/p/Attention-Is-All-You-Need-413663e8033381fa9ca3ec4fe539a791
created: 2026-11-09
updated: 2026-11-09
notion_db: 参考资料
学科: 大语言模型
类型: PAPER
重要度: 核心
阅读状态: 精读
作者: "Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin"
年份: 2017
来源/出版方: NeurIPS
原文链接: "https://arxiv.org/abs/1706.03762"
标签: [transformer, attention, 原始论文]
最后核对: 2026-11-09
关联知识点: [LLM-004, LLM-013, LLM-021]
关联课程模块: [M-LLM-02, M-LLM-03]
---

# REF-0001 Attention Is All You Need

> **类型**：PAPER ｜ **年份**：2017 ｜ **来源**：NeurIPS
> **原文**：<https://arxiv.org/abs/1706.03762>
> **重要度**：核心 ｜ **阅读状态**：精读

## 1. 为什么读它（一句话）

Transformer 的原始论文：自注意力、多头、位置编码的全部出处。

## 2. 它解决什么问题

解决 RNN/CNN 无法并行、长依赖衰减的问题。

## 3. 核心结论

1. 提出完全基于注意力的序列转换架构，去掉循环与卷积。
2. Scaled dot-product attention：除以 √d_k 抑制 softmax 饱和。
3. 多头注意力把表示空间切成 H 份并行关注不同子空间。
4. 位置编码用不同频率的正弦/余弦，使模型感知顺序。

## 4. 与我的知识库的关系

课程里 LLM-013 手算的公式、LLM-021 的总体架构均以此文为准；本库对 512 维/8 头的推导与该文一致。

- 关联课程模块：`[M-LLM-02, M-LLM-03]`
- 关联知识点：`[LLM-004, LLM-013, LLM-021]`
- 关联项目：`[]`
- 关联资料：`[]`

## 5. 关键页 / 关键片段

§3.2.1 Scaled Dot-Product Attention；§3.2.2 Multi-Head；§3.5 Positional Encoding；图 1（编码器-解码器结构）

## 6. 我的疑问与反驳

论文报告的 BLEU 提升在多大程度来自多头 vs 残差+LayerNorm，文中未做完整消融。

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
| 类型 | PAPER |
| 重要度 | 核心 |
| 阅读状态 | 精读 |
| 作者 | Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin |
| 年份 | 2017 |
| 来源/出版方 | NeurIPS |
| 原文链接 | https://arxiv.org/abs/1706.03762 |
| 标签 | transformer、attention、原始论文 |
| 最后核对 | 2026-11-09 |

- **关联知识点**：[LLM-004 Attention 的历史背景与 Transformer 的提出](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/01%20NLP%E5%9F%BA%E7%A1%80/LLM-004%20Attention%20%E7%9A%84%E5%8E%86%E5%8F%B2%E8%83%8C%E6%99%AF%E4%B8%8E%20Transformer%20%E7%9A%84%E6%8F%90%E5%87%BA.md) · [LLM-013 点积注意力与 Scaled Dot-Product Attention](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/03%20Attention%E6%9C%BA%E5%88%B6/LLM-013%20%E7%82%B9%E7%A7%AF%E6%B3%A8%E6%84%8F%E5%8A%9B%E4%B8%8E%20Scaled%20Dot-Product%20Attention.md) · [LLM-021 Transformer 总体架构](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/04%20Transformer%E6%9E%B6%E6%9E%84/LLM-021%20Transformer%20%E6%80%BB%E4%BD%93%E6%9E%B6%E6%9E%84.md)
- **关联课程模块**：[M-LLM-02 Attention 机制与数学原理](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-LLM-02%20Attention%20%E6%9C%BA%E5%88%B6%E4%B8%8E%E6%95%B0%E5%AD%A6%E5%8E%9F%E7%90%86.md) · [M-LLM-03 Transformer 完整架构](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-LLM-03%20Transformer%20%E5%AE%8C%E6%95%B4%E6%9E%B6%E6%9E%84.md)

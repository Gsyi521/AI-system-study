---
id: REF-0002
title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
type: PAPER
authors: [Devlin, Chang, Lee, Toutanova]
year: 2019
venue: NAACL-HLT
arxiv_id: "1810.04805"
isbn: ""
doi: ""
source_url: https://arxiv.org/abs/1810.04805
local_path: ""
zotero_key: ""
subject: 大语言模型
importance: 核心
reading_status: 已读
related_courses: [M-LLM-05]
related_knowledge: [LLM-042, LLM-046]
related_projects: []
related_refs: [REF-0001]
tags: [bert, mlm, 预训练]
last_verified: 2026-11-09
verification_note: "离线录入：来源 URL 未在审计阶段访问（只读约束），首次联网时请复核并更新 last_verified"
notion_page_id: 413663e8-0333-8181-8051-fd9030bd0659
notion_url: https://app.notion.com/p/BERT-413663e8033381818051fd9030bd0659
created: 2026-11-09
updated: 2026-11-09
notion_db: 参考资料
学科: 大语言模型
类型: PAPER
重要度: 核心
阅读状态: 已读
作者: "Devlin, Chang, Lee, Toutanova"
年份: 2019
来源/出版方: NAACL-HLT
原文链接: "https://arxiv.org/abs/1810.04805"
标签: [bert, mlm, 预训练]
最后核对: 2026-11-09
关联知识点: [LLM-042, LLM-046]
关联课程模块: [M-LLM-05]
关联资料: [REF-0001]
---

# REF-0002 BERT

> **类型**：PAPER ｜ **年份**：2019 ｜ **来源**：NAACL-HLT ｜ **arXiv**：1810.04805
> **原文**：<https://arxiv.org/abs/1810.04805>
> **重要度**：核心 ｜ **阅读状态**：已读

## 1. 为什么读它（一句话）

理解型模型的范式起点：MLM + NSP 的双向预训练。

## 2. 它解决什么问题

此前预训练是单向的（只能看左侧），无法同时利用左右上下文。

## 3. 核心结论

1. MLM 随机遮 15%，只对被遮位置计损失。
2. 遮盖策略 80% 换 `[MASK]`、10% 随机词、10% 保持原词，缓解预训练-微调不一致。
3. NSP 判句对是否相邻；后续 RoBERTa 证明其收益有限。
4. `[CLS]/[SEP]/[MASK]` 三个特殊 token 承担句表示与边界。

## 4. 与我的知识库的关系

LLM-042 的 15% 遮盖实测（20 字符遮 2 个 = 0.100）与本文的随机独立采样一致，差异来自短序列方差。

- 关联课程模块：`[M-LLM-05]`
- 关联知识点：`[LLM-042, LLM-046]`
- 关联项目：`[]`
- 关联资料：`[REF-0001]`

## 5. 关键页 / 关键片段

- §3.1 Pre-training BERT（MLM / NSP）
- §3.2 输入表示（segment + position + token 三段相加）
- 附录 A.2 遮盖策略

## 6. 我的疑问与反驳

15% 的最优性缺乏系统消融；80/10/10 也是经验值。

## 7. 可复用的图表 / 公式

- MLM 损失：只对 $\mathcal{M}$（被遮集合）求和，$L=-\sum_{i\in\mathcal{M}}\log P(w_i\mid \text{上下文})$
- 可支撑：LLM-042 的遮盖比例与二项分布推导

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
| 阅读状态 | 已读 |
| 作者 | Devlin, Chang, Lee, Toutanova |
| 年份 | 2019 |
| 来源/出版方 | NAACL-HLT |
| 原文链接 | https://arxiv.org/abs/1810.04805 |
| 标签 | bert、mlm、预训练 |
| 最后核对 | 2026-11-09 |

- **关联知识点**：[LLM-042 BERT 的预训练任务](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/05%20%E9%A2%84%E8%AE%AD%E7%BB%83%E4%B8%8E%E5%BE%AE%E8%B0%83/LLM-042%20BERT%20%E7%9A%84%E9%A2%84%E8%AE%AD%E7%BB%83%E4%BB%BB%E5%8A%A1.md) · [LLM-046 双向上下文与单向生成](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/05%20%E9%A2%84%E8%AE%AD%E7%BB%83%E4%B8%8E%E5%BE%AE%E8%B0%83/LLM-046%20%E5%8F%8C%E5%90%91%E4%B8%8A%E4%B8%8B%E6%96%87%E4%B8%8E%E5%8D%95%E5%90%91%E7%94%9F%E6%88%90.md)
- **关联课程模块**：[M-LLM-05 GPT、BERT 与现代语言模型架构](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-LLM-05%20GPT%E3%80%81BERT%20%E4%B8%8E%E7%8E%B0%E4%BB%A3%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84.md)
- **关联资料**：[REF-0001 Attention Is All You Need](REF-0001%20Attention%20Is%20All%20You%20Need.md)

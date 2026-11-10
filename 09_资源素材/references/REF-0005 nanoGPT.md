---
id: REF-0005
title: "nanoGPT (karpathy)"
type: GITHUB
authors: [Andrej Karpathy]
year: 2023
venue: GitHub
arxiv_id: {arxiv}
isbn: ""
doi: ""
source_url: https://github.com/karpathy/nanoGPT
local_path: ""
zotero_key: ""
subject: 大语言模型
importance: 重要
reading_status: 在读
related_courses: [M-LLM-03]
related_knowledge: [LLM-021, LLM-030, LLM-047]
related_projects: [PROJ-021]
related_refs: [REF-0001]
tags: [gpt, 从零实现, 教学代码]
last_verified: 2026-11-09
verification_note: "离线录入：来源 URL 未在审计阶段访问（只读约束），首次联网时请复核并更新 last_verified"
notion_page_id: 413663e8-0333-81dd-8193-ceec900c101c
notion_url: https://app.notion.com/p/nanoGPT-413663e8033381dd8193ceec900c101c
created: 2026-11-09
updated: 2026-11-09
notion_db: 参考资料
学科: 大语言模型
类型: GITHUB
重要度: 重要
阅读状态: 在读
作者: Andrej Karpathy
年份: 2023
来源/出版方: GitHub
原文链接: "https://github.com/karpathy/nanoGPT"
标签: [gpt, 从零实现, 教学代码]
最后核对: 2026-11-09
关联知识点: [LLM-021, LLM-030, LLM-047]
关联项目: [PROJ-021]
关联课程模块: [M-LLM-03]
关联资料: [REF-0001]
---

# REF-0005 nanoGPT (karpathy)

> **类型**：GITHUB ｜ **年份**：2023 ｜ **来源**：GitHub
> **原文**：<https://github.com/karpathy/nanoGPT>
> **重要度**：重要 ｜ **阅读状态**：在读

## 1. 为什么读它（一句话）

最短的 GPT 训练/推理可运行实现，适合对照课程代码。

## 2. 它解决什么问题

论文与教材代码量大，初学者难以端到端跑通。

## 3. 核心结论

1. train.py 与 model.py 合计约 600 行即覆盖 GPT 训练全流程。
2. 显式实现 causal self-attention 与 weight tying。
3. 用 nanoGPT 的规模做实验成本极低（单卡可跑）。

## 4. 与我的知识库的关系

PROJ-021 从零实现字符级 Transformer 时，结构对照本仓库（Pre-LN、权重共享、causal mask）。

- 关联课程模块：`[M-LLM-03]`
- 关联知识点：`[LLM-021, LLM-030, LLM-047]`
- 关联项目：`[PROJ-021]`
- 关联资料：`[REF-0001]`

## 5. 关键页 / 关键片段

model.py（GPT 定义）、train.py（训练循环）、config/

## 6. 我的疑问与反驳

仓库更新频繁；本条目未锁定 commit（license 为 MIT）。

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
| 类型 | GITHUB |
| 重要度 | 重要 |
| 阅读状态 | 在读 |
| 作者 | Andrej Karpathy |
| 年份 | 2023 |
| 来源/出版方 | GitHub |
| 原文链接 | https://github.com/karpathy/nanoGPT |
| 标签 | gpt、从零实现、教学代码 |
| 最后核对 | 2026-11-09 |

- **关联知识点**：[LLM-021 Transformer 总体架构](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/04%20Transformer%E6%9E%B6%E6%9E%84/LLM-021%20Transformer%20%E6%80%BB%E4%BD%93%E6%9E%B6%E6%9E%84.md) · [LLM-030 Transformer 参数量与计算量估算](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/04%20Transformer%E6%9E%B6%E6%9E%84/LLM-030%20Transformer%20%E5%8F%82%E6%95%B0%E9%87%8F%E4%B8%8E%E8%AE%A1%E7%AE%97%E9%87%8F%E4%BC%B0%E7%AE%97.md) · [LLM-047 上下文窗口与位置外推](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/05%20%E9%A2%84%E8%AE%AD%E7%BB%83%E4%B8%8E%E5%BE%AE%E8%B0%83/LLM-047%20%E4%B8%8A%E4%B8%8B%E6%96%87%E7%AA%97%E5%8F%A3%E4%B8%8E%E4%BD%8D%E7%BD%AE%E5%A4%96%E6%8E%A8.md)
- **关联项目**：[PROJ-021 从零实现一个 Transformer 语言模型](../../07_%E5%AE%9E%E6%88%98%E9%A1%B9%E7%9B%AE/05%20%E5%A4%A7%E6%A8%A1%E5%9E%8B%E9%A1%B9%E7%9B%AE/PROJ-021%20%E4%BB%8E%E9%9B%B6%E5%AE%9E%E7%8E%B0%E4%B8%80%E4%B8%AA%20Transformer%20%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.md)
- **关联课程模块**：[M-LLM-03 Transformer 完整架构](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-LLM-03%20Transformer%20%E5%AE%8C%E6%95%B4%E6%9E%B6%E6%9E%84.md)
- **关联资料**：[REF-0001 Attention Is All You Need](REF-0001%20Attention%20Is%20All%20You%20Need.md)

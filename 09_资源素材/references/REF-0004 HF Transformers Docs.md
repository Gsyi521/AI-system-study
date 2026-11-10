---
id: REF-0004
title: "Hugging Face Transformers Documentation"
type: DOCUMENTATION
authors: [Hugging Face]
year: 2024
venue: Hugging Face
arxiv_id: {arxiv}
isbn: ""
doi: ""
source_url: https://huggingface.co/docs/transformers/index
local_path: ""
zotero_key: ""
subject: 大语言模型
importance: 重要
reading_status: 在读
related_courses: [M-LLM-09]
related_knowledge: [LLM-030, LLM-088, LLM-089]
related_projects: [PROJ-024]
related_refs: []
tags: [transformers, api, 文档]
last_verified: 2026-11-09
verification_note: "离线录入：来源 URL 未在审计阶段访问（只读约束），首次联网时请复核并更新 last_verified"
notion_page_id: 413663e8-0333-8192-90a6-ca3f80869814
notion_url: https://app.notion.com/p/HF-Transformers-Docs-413663e80333819290a6ca3f80869814
created: 2026-11-09
updated: 2026-11-09
notion_db: 参考资料
学科: 大语言模型
类型: DOCUMENTATION
重要度: 重要
阅读状态: 在读
作者: Hugging Face
年份: 2024
来源/出版方: Hugging Face
原文链接: "https://huggingface.co/docs/transformers/index"
标签: [transformers, api, 文档]
最后核对: 2026-11-09
关联知识点: [LLM-030, LLM-088, LLM-089]
关联项目: [PROJ-024]
关联课程模块: [M-LLM-09]
---

# REF-0004 Hugging Face Transformers Documentation

> **类型**：DOCUMENTATION ｜ **年份**：2024 ｜ **来源**：Hugging Face
> **原文**：<https://huggingface.co/docs/transformers/index>
> **重要度**：重要 ｜ **阅读状态**：在读

## 1. 为什么读它（一句话）

课程所有 transformers 代码的官方依据（AutoModel、generate、缓存接口）。

## 2. 它解决什么问题

API 版本差异大，需要权威且可查的用法说明。

## 3. 核心结论

1. AutoModelForCausalLM.from_pretrained 加载权重与配置。
2. generate 支持 greedy/beam/采样与 use_cache。
3. attn_implementation='eager' 才返回注意力权重。
4. past_key_values 即 KV Cache 的逐层张量。

## 4. 与我的知识库的关系

LLM-088/089 里 generate 与手写循环逐位相同的结论，依据本文对 generate 默认参数的说明。

- 关联课程模块：`[M-LLM-09]`
- 关联知识点：`[LLM-030, LLM-088, LLM-089]`
- 关联项目：`[PROJ-024]`
- 关联资料：`[]`

## 5. 关键页 / 关键片段

Quickstart；Generation 策略文档；Cache 文档；Attention 实现选择

## 6. 我的疑问与反驳

文档随版本变化，引用时须记录 version（本条目未锁定版本号）。

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
| 类型 | DOCUMENTATION |
| 重要度 | 重要 |
| 阅读状态 | 在读 |
| 作者 | Hugging Face |
| 年份 | 2024 |
| 来源/出版方 | Hugging Face |
| 原文链接 | https://huggingface.co/docs/transformers/index |
| 标签 | transformers、api、文档 |
| 最后核对 | 2026-11-09 |

- **关联知识点**：[LLM-030 Transformer 参数量与计算量估算](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/04%20Transformer%E6%9E%B6%E6%9E%84/LLM-030%20Transformer%20%E5%8F%82%E6%95%B0%E9%87%8F%E4%B8%8E%E8%AE%A1%E7%AE%97%E9%87%8F%E4%BC%B0%E7%AE%97.md) · [LLM-088 自回归推理的基本流程](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/06%20%E6%8E%A8%E7%90%86%E4%B8%8E%E9%83%A8%E7%BD%B2/LLM-088%20%E8%87%AA%E5%9B%9E%E5%BD%92%E6%8E%A8%E7%90%86%E7%9A%84%E5%9F%BA%E6%9C%AC%E6%B5%81%E7%A8%8B.md) · [LLM-089 Prefill 与 Decode 两阶段](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/06%20%E6%8E%A8%E7%90%86%E4%B8%8E%E9%83%A8%E7%BD%B2/LLM-089%20Prefill%20%E4%B8%8E%20Decode%20%E4%B8%A4%E9%98%B6%E6%AE%B5.md)
- **关联项目**：[PROJ-024 结构化输出与约束解码](../../07_%E5%AE%9E%E6%88%98%E9%A1%B9%E7%9B%AE/05%20%E5%A4%A7%E6%A8%A1%E5%9E%8B%E9%A1%B9%E7%9B%AE/PROJ-024%20%E7%BB%93%E6%9E%84%E5%8C%96%E8%BE%93%E5%87%BA%E4%B8%8E%E7%BA%A6%E6%9D%9F%E8%A7%A3%E7%A0%81.md)
- **关联课程模块**：[M-LLM-09 大语言模型推理、部署与优化](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-LLM-09%20%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86%E3%80%81%E9%83%A8%E7%BD%B2%E4%B8%8E%E4%BC%98%E5%8C%96.md)

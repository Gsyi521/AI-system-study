---
id: REF-0003
title: "LoRA: Low-Rank Adaptation of Large Language Models"
type: PAPER
authors: [Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen]
year: 2022
venue: ICLR
arxiv_id: "2106.09685"
isbn: ""
doi: ""
source_url: https://arxiv.org/abs/2106.09685
local_path: ""
zotero_key: ""
subject: 大语言模型
importance: 核心
reading_status: 精读
related_courses: [M-LLM-07]
related_knowledge: [LLM-048, LLM-065, LLM-075]
related_projects: [PROJ-022]
related_refs: [REF-0001]
tags: [lora, peft, 微调]
last_verified: 2026-11-09
verification_note: "离线录入：来源 URL 未在审计阶段访问（只读约束），首次联网时请复核并更新 last_verified"
notion_page_id: 413663e8-0333-81f8-9f70-e5913e222433
notion_url: https://app.notion.com/p/LoRA-413663e8033381f89f70e5913e222433
created: 2026-11-09
updated: 2026-11-09
notion_db: 参考资料
学科: 大语言模型
类型: PAPER
重要度: 核心
阅读状态: 精读
作者: "Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen"
年份: 2022
来源/出版方: ICLR
原文链接: "https://arxiv.org/abs/2106.09685"
标签: [lora, peft, 微调]
最后核对: 2026-11-09
关联知识点: [LLM-048, LLM-065, LLM-075]
关联项目: [PROJ-022]
关联课程模块: [M-LLM-07]
关联资料: [REF-0001]
---

# REF-0003 LoRA

> **类型**：PAPER ｜ **年份**：2022 ｜ **来源**：ICLR ｜ **arXiv**：2106.09685
> **原文**：<https://arxiv.org/abs/2106.09685>
> **重要度**：核心 ｜ **阅读状态**：精读

## 1. 为什么读它（一句话）

参数高效微调的主流方案，课程 LoRA 实验的理论依据。

## 2. 它解决什么问题

全参微调 7B 需要约 100 GB 显存，成本不可承受。

## 3. 核心结论

1. 把权重更新约束为低秩分解 $\Delta W = BA$，只训练 $A$、$B$。
2. 秩 $r$ 远小于 $d$ 时，可训练参数降到 0.1%~1%。
3. 推理时可把 $BA$ 合并回 $W$，**不增加延迟**。
4. 多个任务上接近全参微调效果。

## 4. 与我的知识库的关系

LLM-065 实测 $r=8$ 挂 `c_attn` 为 **294,912** 参数（**0.2364%**），合并后 logits 差 **9.9e-05**，与本文"可训练参数极少、合并后等价"的结论一致。

- 关联课程模块：`[M-LLM-07]`
- 关联知识点：`[LLM-048, LLM-065, LLM-075]`
- 关联项目：`[PROJ-022]`
- 关联资料：`[REF-0001]`

## 5. 关键页 / 关键片段

- §4 方法（Fig.1：冻结 $W$，旁路 $A\to B$）
- §4.2 应用位置（attention 的 $W_q/W_v$ 性价比最高）
- §5 实验（GLUE / E2E / DART）

## 6. 我的疑问与反驳

$r$ 的最优值依赖任务与数据量，论文未给出选择规则；$r$ 与 target_modules 的交互也缺少系统指导。

## 7. 可复用的图表 / 公式

- 前向：$h = W_0x + \Delta Wx = W_0x + BAx$，其中 $B\in\mathbb{R}^{d\times r}$、$A\in\mathbb{R}^{r\times k}$
- 参数量：$r(d+k)$ 对比全参 $dk$
- 可支撑：LLM-065 的参数量公式与合并等价性

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
| 作者 | Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen |
| 年份 | 2022 |
| 来源/出版方 | ICLR |
| 原文链接 | https://arxiv.org/abs/2106.09685 |
| 标签 | lora、peft、微调 |
| 最后核对 | 2026-11-09 |

- **关联知识点**：[LLM-048 微调范式：全参与参数高效](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/05%20%E9%A2%84%E8%AE%AD%E7%BB%83%E4%B8%8E%E5%BE%AE%E8%B0%83/LLM-048%20%E5%BE%AE%E8%B0%83%E8%8C%83%E5%BC%8F%EF%BC%9A%E5%85%A8%E5%8F%82%E4%B8%8E%E5%8F%82%E6%95%B0%E9%AB%98%E6%95%88.md) · [LLM-065 LoRA 与参数高效微调](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/05%20%E9%A2%84%E8%AE%AD%E7%BB%83%E4%B8%8E%E5%BE%AE%E8%B0%83/LLM-065%20LoRA%20%E4%B8%8E%E5%8F%82%E6%95%B0%E9%AB%98%E6%95%88%E5%BE%AE%E8%B0%83.md) · [LLM-075 PEFT 的取舍与选型](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/05%20%E9%A2%84%E8%AE%AD%E7%BB%83%E4%B8%8E%E5%BE%AE%E8%B0%83/LLM-075%20PEFT%20%E7%9A%84%E5%8F%96%E8%88%8D%E4%B8%8E%E9%80%89%E5%9E%8B.md)
- **关联项目**：[PROJ-022 用 LoRA 微调 gpt2 做文本分类](../../07_%E5%AE%9E%E6%88%98%E9%A1%B9%E7%9B%AE/05%20%E5%A4%A7%E6%A8%A1%E5%9E%8B%E9%A1%B9%E7%9B%AE/PROJ-022%20%E7%94%A8%20LoRA%20%E5%BE%AE%E8%B0%83%20gpt2%20%E5%81%9A%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB.md)
- **关联课程模块**：[M-LLM-07 大语言模型微调与 PEFT](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-LLM-07%20%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%BE%AE%E8%B0%83%E4%B8%8E%20PEFT.md)
- **关联资料**：[REF-0001 Attention Is All You Need](REF-0001%20Attention%20Is%20All%20You%20Need.md)

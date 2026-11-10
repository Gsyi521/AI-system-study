---
id: LLM-044
title: T5 与 Span Corruption
en_title: T5 and Span Corruption
domain: LLM
module: M-LLM-05
type: Concept
difficulty: L2 应用
status: planned
prerequisites: [LLM-043]
related_math: []
related_python: []
related_ml: []
related_dl: []
related_llm: [LLM-042, LLM-045]
notion_page_id: 412663e8-0333-8122-9a2b-ca9d8b60063d
notion_url: https://app.notion.com/p/T5-Span-Corruption-412663e8033381229a2bca9d8b60063d
created: 2026-11-08
updated: 2026-11-08
notion_db: 知识点
学科: 大语言模型
学习状态: 未开始
难度: L2 应用
知识类型: Concept
所属模块: [M-LLM-05]
前置知识: [LLM-043]
后继知识: [LLM-045]
tags: [大语言模型, Concept]
---

# T5 与 Span Corruption

## 1. 本节学习目标

1. 说清 T5 的预训练目标 **Span Corruption**（遮盖连续片段，用哨兵 token 恢复）。
2. 用实测看它的输入/目标构造（本课遮了 **4/13** 个词，产生 2 个 span）。
3. 理解 T5 的"**一切皆文本到文本**"统一框架。
4. 知道它与 MLM/CLM 的差异与取舍。

## 2. 为什么需要这个知识

因为 **T5 是"统一框架"的代表**：分类、翻译、摘要全变成"输入文本 → 输出文本"。

而且 Span Corruption 是"**生成式填空**"，介于 BERT 的 MLM 与 GPT 的 CLM 之间，理解它有助于看清三类目标的本质。

## 3. 直观解释

```text
原文：the cat sat on the mat and the dog sat on the log

遮盖连续片段后输入：the <X> on the <Y> the dog sat on the log
目标（按顺序恢复片段）：<X> cat sat <Y> mat and <Z>
```

**与 MLM 的区别**：MLM 一次填一个词；Span Corruption **一次填一段**，且用**生成**方式输出。

## 4. 正式定义

**Span Corruption 目标**

$$
L=-\sum_{k}\log P_\theta\big(\text{span}_k\mid \text{corrupted input},\ \text{span}_{<k}\big)
$$

**本课实测**

```text
原文: the cat sat on the mat and the dog sat on the log
输入: the <extra_id_0> on the <extra_id_1> the dog sat on the log
目标: <extra_id_0> cat sat <extra_id_1> mat and <extra_id_2>
遮盖词数: 4 / 13
```

**读法**：

1. 两个遮盖段 `[1,3)`（cat sat）与 `[5,7)`（mat and）被替换成哨兵 `<extra_id_0>`、`<extra_id_1>`。
2. **目标里按顺序列出各段内容**，末尾再接下一个哨兵 ⇒ 生成式恢复。
3. **遮盖 4/13 ≈ 31%**（T5 原论文用 15%，但按 **span 长度** 遮盖，实际比例更高）。

## 5. 历史背景或技术背景

| 年份 | 模型 | 目标 |
|---|---|---|
| 2018 | BERT | MLM（单词级） |
| 2019 | **T5** | **Span Corruption**（片段级生成） |
| 2019 | BART | 文本去噪（多种噪声） |
| 2020 | mT5 | 多语言版 |

**T5 的关键贡献**：把所有任务**统一成 text-to-text**，并系统性比较了架构/目标/数据的取舍（"T5 论文是一本工程手册"）。

## 6. 核心原理

**为什么要遮"连续片段"**

- 单词级遮盖：模型可以用相邻词猜（局部性强，任务偏简单）；
- **片段级遮盖**：需要理解更长的上下文与句法 ⇒ 任务更难、学到更多结构。

**为什么要"生成"而不是"分类"**

目标端用哨兵 token 组织多个片段 ⇒ 一次前向恢复**多个** span ⇒ 训练信号更密集，且天然支持"输出比输入短"（去噪/摘要）。

**本课对应**：输入 13 词中挖掉 4 词，目标只有 8 个 token（含哨兵）⇒ 输出比输入短。

## 7. 数学表达式

**均匀遮盖比例（T5 配置）**

$$
p_{\text{mask}}=0.15,\qquad
\text{span length}\sim \text{Geometric}(p=1/3)\ \text{（均值约 3）}
$$

**本课**：2 个 span、平均长度 2、覆盖率 $4/13\approx31\%$（小样本波动）。

**损失（教师强制）**

$$
L=-\sum_{t}\log P_\theta(y_t\mid y_{<t},x)
$$

## 8. 详细数学推导

**推导：为什么目标端要"按顺序"输出**

若目标写成"任意顺序"，模型无法用**自回归**一次生成（自回归只能从左到右）。按原文顺序排列 span ⇒ 可用标准 CLM 解码。

**推导：哨兵 token 的作用**

$$
\text{target}=\langle s_0\rangle\ \text{span}_0\ \langle s_1\rangle\ \text{span}_1\ \langle s_2\rangle
$$

哨兵既**分隔**各段，又**告知段数**；末尾的 $\langle s_k\rangle$ 表示"结束"。

## 9. 图解

```mermaid
flowchart LR
    A["原文"] --> B["随机选片段遮盖<br/>替换为哨兵"]
    B --> C["编码器（双向）"]
    C --> D["解码器（因果）"]
    D --> E["按顺序生成各片段"]
```

## 10. 具体数值示例

```text
原文 13 词
遮盖 2 段：cat sat（2 词）、mat and（2 词）
输入：the <extra_id_0> on the <extra_id_1> the dog sat on the log
目标：<extra_id_0> cat sat <extra_id_1> mat and <extra_id_2>
遮盖比例：4/13 ≈ 0.31
目标长度：8 token（含 3 个哨兵）
```

## 11. NumPy 实现

```python
words = "the cat sat on the mat and the dog sat on the log".split()
spans = [(1, 3), (5, 7)]

inp, tgt, sid, i = [], [], 0, 0
while i < len(words):
    span = next((s for s in spans if s[0] == i), None)
    if span:
        s, e = span
        inp.append(f"<extra_id_{sid}>")
        tgt.append(f"<extra_id_{sid}> " + " ".join(words[s:e]))
        sid += 1
        i = e
    else:
        inp.append(words[i])
        i += 1
tgt.append(f"<extra_id_{sid}>")

print("输入:", " ".join(inp))
print("目标:", " ".join(tgt))
print("遮盖比例:", sum(e - s for s, e in spans), "/", len(words))
```

## 12. PyTorch 实现

```python
import torch
import torch.nn as nn

# 用 nn.Transformer 演示 encoder-decoder（T5 的骨架）
model = nn.Transformer(d_model=32, nhead=4, num_encoder_layers=2,
                       num_decoder_layers=2, batch_first=True)
src = torch.randn(2, 10, 32)     # 损坏后的输入
tgt = torch.randn(2, 6, 32)     # 目标片段（已移位）
out = model(src, tgt)
print("编码输入:", tuple(src.shape), " 解码目标:", tuple(tgt.shape))
print("输出:", tuple(out.shape), " ⇒ 逐位置预测目标 token")
print("参数:", sum(p.numel() for p in model.parameters()))
```

## 13. Hugging Face 实践

```python
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoConfig

# T5 未缓存，故不加载；这里只演示如何判断架构类型
cfg = AutoConfig.from_pretrained("gpt2")
print("gpt2 is_encoder_decoder:", cfg.is_encoder_decoder, "（Decoder-only）")
print("T5 类模型的 config 里 is_encoder_decoder=True")
print("（本机未缓存 T5，不展示其真实配置值）")
```

## 14. 代码逐行解释

| 代码 | 作用 |
|---|---|
| `spans` | 要遮盖的连续区间 |
| `while` + `next` | 按位置扫描并插入哨兵 |
| `tgt.append(f"<extra_id_{sid}>")` | 末尾哨兵（结束标志） |
| `nn.Transformer` | Encoder-Decoder 骨架 |
| `cfg.is_encoder_decoder` | 判断架构 |

## 15. 实际运行结果

```text
原文: the cat sat on the mat and the dog sat on the log
输入: the <extra_id_0> on the <extra_id_1> the dog sat on the log
目标: <extra_id_0> cat sat <extra_id_1> mat and <extra_id_2>
遮盖词数: 4 / 13
```

## 16. 应用场景

| 场景 | 说明 |
|---|---|
| 翻译/摘要 | T5 的原生场景 |
| 文本去噪 | 恢复被破坏的文本 |
| 统一多任务 | 全部转成 text-to-text |
| 填空式数据增强 | 造"输入-输出"对 |
| 多语言 | mT5 |

## 17. 优点与局限

| 优点 | 局限 |
|---|---|
| 片段级遮盖，任务更难更有效 | 需要 Encoder-Decoder（参数多 33%） |
| 统一框架，任务迁移简单 | 推理两段式，比 Decoder-only 慢 |
| 输出可短于输入（去噪） | 哨兵 token 增加词表复杂度 |
| 训练信号密集（一次多段） | 需要配对数据做下游 |

## 18. 常见错误

| 错误 | 现象 | 纠正 |
|---|---|---|
| 把 Span Corruption 当 MLM | 目标端结构错 | 它是**生成式**恢复 |
| 目标不按原文顺序 | 自回归无法生成 | 必须顺序排列 |
| 忘末尾哨兵 | 不知何时结束 | 加 `<extra_id_k>` |
| 按 15% 单词算遮盖 | 实际比例更高 | 按 span 长度算 |
| 用 Decoder-only 复现 T5 | 结构不符 | 需要 Enc-Dec |

## 19. 本节总结

- T5 用 **Span Corruption**：遮盖**连续片段**，用**哨兵 token** 按序生成恢复。
- 本例 13 词遮 2 段共 **4 词（31%）**，目标 8 token（含 3 哨兵）。
- 与 MLM 的区别：**片段级 + 生成式**；与 CLM 的区别：**有编码器、双向可见**。

## 20. 前置与后续知识链接

- 前置：[[LLM-043 GPT 系列的演进]]、[[LLM-042 BERT 的预训练任务]]
- 后续：[[LLM-045 预训练目标对比：MLM、CLM 与 Span Corruption]]
- 相关：[[LLM-027 Encoder-Only、Decoder-Only 与 Encoder-Decoder]]

---

## 练习与参考答案

**基础 1**：Span Corruption 与 MLM 的核心区别是什么？

**基础 2**：本课遮了几个词？占多少比例？目标有几个 token？

**进阶 1**：为什么目标端必须按原文顺序排列片段，且末尾要加哨兵？

**综合**：你要做"会议纪要生成"（长文本 → 短摘要）。请说明用 T5 类还是 GPT 类，并给出理由。

> [!success]- 参考答案

> **基础 1**
> - **MLM**：遮**单个词**，用**分类**方式预测（每个位置一个词表分布）；
> - **Span Corruption**：遮**连续片段**，用**生成**方式按序输出各段内容（带哨兵 token）。
> 后者任务更难、输出可比输入短。

> **基础 2**
> 遮 **4/13** 个词（2 段：cat sat、mat and），比例约 **0.31**；目标 **8** 个 token（含 3 个哨兵）。

> **进阶 1**
> **按顺序**：解码器是自回归的，只能从左到右生成，乱序无法一次生成。
> **末尾哨兵**：告诉模型"片段列举结束"，提供明确的停止信号；同时哨兵起到分隔各段的作用。

> **综合（参考）**
> **选择：T5 类（Encoder-Decoder）**，理由：
> 1. 摘要是"输入长、输出短"的**序列转换**，Enc-Dec 结构原生匹配（编码器读全文，解码器只生成摘要）；
> 2. 编码器双向可见，长输入理解更充分；Decoder-only 需把全文塞进 prompt，长文本成本高；
> 3. T5 的 Span Corruption 预训练本身就含"去噪/缩短"能力。
> **替代**：若已有强 Decoder-only 模型且上下文够长，也可直接用（工程更简单）。
> **评估**：ROUGE-L + 人工可读性；注意摘要任务容易"抄原文"，需检查抽象度。

## 学习记录

| 日期 | 状态 | 用时 | 问题 | 复盘 |
|---|---|---|---|---|
| 2026-11-08 | planned | — | — | — |

---

## 🔗 Notion 同步信息

> 以下内容由 Notion「AI Knowledge System · 知识点」数据库按知识 ID 合并而来（导出于 2026-11-09）。

| 属性 | 值 |
|---|---|
| 学科 | 大语言模型 |
| 学习状态 | 未开始 |
| 难度 | L2 应用 |
| 知识类型 | Concept |

- **所属模块**：[M-LLM-05 GPT、BERT 与现代语言模型架构](../%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-LLM-05%20GPT%E3%80%81BERT%20%E4%B8%8E%E7%8E%B0%E4%BB%A3%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84.md)
- **前置知识**：[LLM-043 GPT 系列的演进](LLM-043%20GPT%20%E7%B3%BB%E5%88%97%E7%9A%84%E6%BC%94%E8%BF%9B.md)
- **后继知识**：[LLM-045 预训练目标对比：MLM、CLM 与 Span Corruption](LLM-045%20%E9%A2%84%E8%AE%AD%E7%BB%83%E7%9B%AE%E6%A0%87%E5%AF%B9%E6%AF%94%EF%BC%9AMLM%E3%80%81CLM%20%E4%B8%8E%20Span%20Corruption.md)

---
id: LLM-036
title: Embedding 与向量表示
en_title: Embeddings and Vector Representations
domain: LLM
module: M-LLM-04
type: Concept
difficulty: L2 应用
status: planned
prerequisites: [LLM-035]
related_math: [MATH-216]
related_python: []
related_ml: []
related_dl: [DL-012]
related_llm: [LLM-037, LLM-039]
notion_page_id: 412663e8-0333-81a9-8ff1-fcb81150da22
notion_url: https://app.notion.com/p/Embedding-412663e8033381a98ff1fcb81150da22
created: 2026-11-08
updated: 2026-11-08
notion_db: 知识点
学科: 大语言模型
学习状态: 未开始
难度: L2 应用
知识类型: Concept
所属模块: [M-LLM-04]
前置知识: [LLM-035]
后继知识: [LLM-037]
关联数学: [MATH-216]
对应项目: [PROJ-022]
tags: [大语言模型, Concept]
---

# Embedding 与向量表示

## 1. 本节学习目标

1. 说清 Embedding 的本质：**一个可学习的查表矩阵**。
2. 用实测验证"查表 = one-hot × W"（本课差 **0.0**）。
3. 算清嵌入层的参数占比（本课 $V{=}50257,d{=}768$ ⇒ **38,597,376**）。
4. 理解为什么"离散 ID → 连续向量"是深度学习处理文本的第一步。

## 2. 为什么需要这个知识

因为 token ID 本身**没有语义**：`3` 和 `7` 的距离不代表语义距离。

Embedding 把 ID 映射到 $\mathbb{R}^d$，让"相似词距离近"成为**可学习、可计算**的性质——这是后面一切（Attention、RAG 检索）的基础。

## 3. 直观解释

```text
token id 42  →  查表第 42 行  →  [0.12, -0.33, ..., 0.07]（d 维向量）
```

**类比**：一张"词 → 坐标"的字典，坐标不是人写的，而是**训练学出来的**。

## 4. 正式定义

**嵌入矩阵**

$$
E\in\mathbb{R}^{V\times d}
$$

**查表**

$$
\mathbf{e}_i=E[i]\in\mathbb{R}^{d}
$$

**与 one-hot 等价**

$$
E^\top\mathbf{1}_i=E[i]
$$

**本课实测**

```text
词表 100 × 维度 64；参数: 6400
ids: [3, 7, 42] → 输出形状: (3, 64)
查表 vs one-hot@W 最大差: 0.0

V=50257  d=768  : 嵌入参数   38,597,376
V=32000  d=4096 : 嵌入参数  131,072,000
V=128000 d=4096 : 嵌入参数  524,288,000
```

**读法**：

1. **查表与 one-hot 乘法完全等价**（差 **0.0**）⇒ Embedding 层不是一个特殊操作，只是一个"稀疏矩阵乘"的高效实现。
2. **参数量 = $V\times d$**：$50257\times768\approx38.6$M（GPT-2 里约占 31%）。
3. **大词表代价高**：$V{=}128000,d{=}4096$ ⇒ **5.24 亿**参数，比很多小模型整体还大。

## 5. 历史背景或技术背景

| 年份 | 进展 |
|---|---|
| 2003 | Bengio 神经语言模型：第一次用可学习词向量 |
| 2013 | word2vec：静态词向量普及（CBOW/Skip-gram） |
| 2014 | GloVe：基于共现矩阵分解 |
| 2018+ | 上下文向量（ELMo/BERT/GPT）：同一词不同语境不同向量 |
| 2020+ | 句向量 / Embedding 模型（检索、RAG） |

**关键演进**：**静态向量**（一个词一个向量）→ **上下文向量**（同一词在不同句子不同向量），见 LLM-038。

## 6. 核心原理

**为什么"查表"能训练**

Embedding 的第 $i$ 行只在 token $i$ 出现时被用到 ⇒ 反向传播时只有该行收到梯度：

$$
\frac{\partial L}{\partial E[i]}=\sum_{t:\,x_t=i}\frac{\partial L}{\partial \mathbf{e}_{t}}
$$

⇒ 每个 token 的向量都由"它出现的所有上下文"共同塑造（这已经隐含了语义）。

**参数效率**

$$
\#=V\cdot d
$$

本课 GPT-2 规模下：$\frac{38.6}{124.4}\approx31\%$ 的参数在嵌入层。

## 7. 数学表达式

**与输出头的关系（weight tying）**

$$
\text{logits}=E\,\mathbf{h}\quad(\text{共享 }E)
$$

本课实测：绑定后 $6400$ 参数代替 $12800$，**省 50%**（LLM-039 详述）。

**梯度稀疏性**

$$
\frac{\partial L}{\partial E}\ \text{只有被选中的行非零}
$$

## 8. 详细数学推导

**推导：one-hot × W = 查表**

设 $\mathbf{1}_i\in\{0,1\}^V$ 为第 $i$ 个 one-hot，$E\in\mathbb{R}^{V\times d}$：

$$
(\mathbf{1}_i^\top E)_j=\sum_k (\mathbf{1}_i)_k E_{kj}=E_{ij}
$$

⇒ 恰好取第 $i$ 行 ✅（本课数值验证差 0.0）。

**推导：为什么大词表用"分块/共享"**

$V$ 很大时 $Vd$ 会主导参数。缓解手段：
- **weight tying**（输出头共享嵌入）；
- **小词表 + 字节级 BPE**（$V$ 只需 3–13 万）；
- **低秩分解**（$E=AB$，$A\in\mathbb{R}^{V\times r}$）。

## 9. 图解

```mermaid
flowchart LR
    A["token id: 42"] --> B["Embedding 矩阵 E<br/>V×d"]
    B --> C["第 42 行 → 向量 (d)"]
    C --> D["后续 Transformer 层"]
```

## 10. 具体数值示例

```text
V=100, d=64 ⇒ 参数 6,400
y = E[3]        （查表）
y = onehot(3) @ E （等价，差 0.0）

GPT-2: 50257 × 768 = 38,597,376（约 38.6M，占总参数 31%）
```

## 11. NumPy 实现

```python
import numpy as np

V, d = 100, 64
rng = np.random.default_rng(0)
E = rng.normal(size=(V, d))
ids = np.array([3, 7, 42])

lookup = E[ids]
onehot = np.eye(V)[ids]
matmul = onehot @ E
print("查表形状:", lookup.shape)
print("两者最大差:", float(np.abs(lookup - matmul).max()))
print("参数量:", E.size)
```

## 12. PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

emb = nn.Embedding(100, 64)
ids = torch.tensor([3, 7, 42])
out = emb(ids)

onehot = F.one_hot(ids, 100).float()
manual = onehot @ emb.weight
print("输出:", tuple(out.shape), " 差别:", float((out - manual).abs().max()))
print("参数量:", emb.weight.numel())
print("梯度是否稀疏:（只有被选中的行有梯度）")
out.sum().backward()
grad = emb.weight.grad
print("  非零行数:", int((grad.abs().sum(-1) > 0).sum()), "/", 100)
```

## 13. Hugging Face 实践

```python
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
m = AutoModelForCausalLM.from_pretrained("gpt2")
emb = m.transformer.wte
print("词表 × 维度:", tuple(emb.weight.shape), " 参数量:", emb.weight.numel())
ids = tok("cat")["input_ids"]
print("'cat' 的 id:", ids, " 向量前 5 维:", [round(float(v), 3) for v in emb.weight[ids[0]][:5]])
print("输出头与嵌入是否共享:", m.lm_head.weight is emb.weight)
```

## 14. 代码逐行解释

| 代码 | 作用 |
|---|---|
| `nn.Embedding(V, d)` | 声明 $V\times d$ 可学习表 |
| `emb(ids)` | 查表（不是矩阵乘，但等价） |
| `F.one_hot` | 用于验证等价性 |
| `emb.weight.grad` | 只有被选中行非零 |
| `m.transformer.wte` | 真实模型的输入嵌入 |
| `lm_head.weight is wte.weight` | 检查是否 weight tying |

## 15. 实际运行结果

```text
词表 100 × 维度 64；参数: 6400
ids: [3, 7, 42] → 输出形状: (3, 64)
查表 vs one-hot@W 最大差: 0.0

V=50257  d=768  : 嵌入参数   38,597,376
V=32000  d=4096 : 嵌入参数  131,072,000
V=128000 d=4096 : 嵌入参数  524,288,000
```

## 16. 应用场景

| 场景 | 用途 |
|---|---|
| 模型输入 | token → 向量 |
| 输出头 | 向量 → 词表分布（可共享） |
| 检索/RAG | 句向量（embedding 模型） |
| 推荐 | 用户/物品 ID → 向量 |
| 类别特征 | 高基数类别编码 |

## 17. 优点与局限

| 优点 | 局限 |
|---|---|
| 连续、可微、可学习 | 参数量 $Vd$（本课 5.24 亿的最坏情况） |
| 相似词自动靠近 | 静态向量无法区分多义（LLM-038） |
| 实现极简、高效 | 冷门 token 训练不足 |

## 18. 常见错误

| 错误 | 现象 | 纠正 |
|---|---|---|
| 把 embedding 当"固定词向量" | 无法理解上下文向量 | 它也是可训练参数 |
| 忽略 $Vd$ 参数量 | 显存估算错 | 加上嵌入 |
| 以为查表与乘法不同 | 实现困惑 | 二者等价（差 0.0） |
| 换 tokenizer 不换嵌入 | ID 语义错位 | 词表与权重绑定 |
| 忘 `padding_idx` | pad 也学梯度 | 设置 `padding_idx` |

## 19. 本节总结

- Embedding = **$V\times d$ 可学习查表矩阵**；与 one-hot 乘法**完全等价**（实测差 **0.0**）。
- 参数量 $Vd$：本课 $50257\times768=\mathbf{38{,}597{,}376}$（GPT-2 约 31%）。
- 大词表代价巨大（$128000\times4096$ ⇒ **5.24 亿**）⇒ 需 weight tying / 小词表 / 低秩。

## 20. 前置与后续知识链接

- 前置：[[LLM-035 Token 数量、上下文窗口与成本]]
- 后续：[[LLM-037 语义空间与相似度]]、[[LLM-038 从静态词向量到上下文向量]]、[[LLM-039 Embedding 与模型参数]]
- 相关：[[MATH-216 矩阵乘法]]、[[DL-012 线性变换的矩阵形式]]

---

## 练习与参考答案

**基础 1**：Embedding 层的参数量怎么算？为什么它与 one-hot 乘法等价？

**基础 2**：本课 $V{=}50257,d{=}768$ 的嵌入参数量是多少？

**进阶 1**：为什么说"Embedding 的梯度是稀疏的"？这带来什么工程影响？

**综合**：$V{=}128000$、$d{=}4096$ 的模型，嵌入层有多大？如果显存吃紧，请给出两种压缩方案。

> [!success]- 参考答案

> **基础 1**
> 参数量 $=V\times d$（$V$ 词表、$d$ 维度）。
> 等价性：$\mathbf{1}_i^\top E$ 恰好取第 $i$ 行，所以"查表"就是"one-hot 与 $E$ 相乘"，只是实现上跳过了零元素。

> **基础 2**
> $50257\times768=\mathbf{38{,}597{,}376}$（约 38.6M）。

> **进阶 1**
> 一次前向只用到被选中 token 对应的行，其他行梯度为 0（本课实测 3 个 id、100 行里只有 3 行非零）。
> **影响**：① 可用稀疏优化器（如 SparseAdam）加速；② 分布式训练时嵌入层常需专门分片（否则某卡负载不均）；③ 冷门 token 更新次数少，容易欠训练。

> **综合（参考）**
> **大小**：$128000\times4096=\mathbf{524{,}288{,}000}$（约 5.24 亿参数）；fp16 约 **1.0 GB**，bf16 同量级。
> **两种压缩**：
> 1. **Weight tying**：输出头与嵌入共享 ⇒ 省一份同样大小的参数（若原本不共享）；
> 2. **低秩分解**：$E=AB$，$A\in\mathbb{R}^{V\times r}$、$B\in\mathbb{R}^{r\times d}$，参数量从 $Vd$ 降到 $r(V+d)$（$r\ll d$）；
> 3. 补充：**缩小词表**（字节级 BPE 只需 3–13 万）、**量化嵌入**（INT8/INT4）。

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

- **所属模块**：[M-LLM-04 Tokenizer、Embedding 与文本表示](../%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-LLM-04%20Tokenizer%E3%80%81Embedding%20%E4%B8%8E%E6%96%87%E6%9C%AC%E8%A1%A8%E7%A4%BA.md)
- **前置知识**：[LLM-035 Token 数量、上下文窗口与成本](LLM-035%20Token%20%E6%95%B0%E9%87%8F%E3%80%81%E4%B8%8A%E4%B8%8B%E6%96%87%E7%AA%97%E5%8F%A3%E4%B8%8E%E6%88%90%E6%9C%AC.md)
- **后继知识**：[LLM-037 语义空间与相似度](LLM-037%20%E8%AF%AD%E4%B9%89%E7%A9%BA%E9%97%B4%E4%B8%8E%E7%9B%B8%E4%BC%BC%E5%BA%A6.md)
- **关联数学**：[MATH-216 矩阵乘法](../../06_%E6%95%B0%E5%AD%A6%E5%9F%BA%E7%A1%80/02%20%E7%BA%BF%E6%80%A7%E4%BB%A3%E6%95%B0/MATH-216%20%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95.md)
- **对应项目**：[PROJ-022 用 LoRA 微调 gpt2 做文本分类](../../07_%E5%AE%9E%E6%88%98%E9%A1%B9%E7%9B%AE/05%20%E5%A4%A7%E6%A8%A1%E5%9E%8B%E9%A1%B9%E7%9B%AE/PROJ-022%20%E7%94%A8%20LoRA%20%E5%BE%AE%E8%B0%83%20gpt2%20%E5%81%9A%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB.md)

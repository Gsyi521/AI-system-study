---
id: PROJ-021
title: 从零实现一个 Transformer 语言模型
domain: PROJ
module: [M-LLM-05, M-LLM-06]
type: project
difficulty: L3 推导
status: planned
tech_stack: [Python, PyTorch, NumPy]
prerequisites: [LLM-021, LLM-025, LLM-030]
related_math: [MATH-216]
related: [LLM-013, LLM-017, LLM-024, LLM-025, LLM-030, PROJ-022]
notion_page_id: 412663e8-0333-8137-ad8f-cd94a32dd551
notion_url: https://app.notion.com/p/Transformer-412663e803338137ad8fcd94a32dd551
created: 2026-11-08
updated: 2026-11-08
notion_db: 实战项目
所属学科: 机器学习
项目状态: 未开始
项目阶段: 立项
难度: L3 推导
项目目标: 从零实现一个 Transformer 语言模型
技术栈: [Python, PyTorch, NumPy]
项目仓库: ai-knowledge-system/projects/PROJ-021
对应课程模块: [M-LLM-05, M-LLM-06]
前置知识: [LLM-021, LLM-025, LLM-030]
对应知识点: [LLM-013, LLM-017, LLM-024, LLM-025, LLM-030]
tags: [机器学习]
---

# PROJ-021 从零实现一个 Transformer 语言模型

## 1. 项目背景

调用 `nn.TransformerEncoderLayer` 谁都会，但**自己写一遍**才会暴露三件事：

1. 因果掩码到底加在 softmax 之前还是之后、加在哪一维；
2. Pre-LN 与 Post-LN 的残差加法顺序差在哪；
3. **位置编码到底是不是必需的**——把 `pos_emb` 删掉，模型还能不能学？

第 3 点是本项目真正的实验。理论答案很清楚：没有位置编码时自注意力对序列是**置换等变**的，模型只能看到"前文里有哪些字符"，看不到"它们的顺序"。但"理论上更差"和"实测更差"是两件事——本项目**实测发现反过来了**：

> 去掉位置编码后，训练 loss 从 1.4480 掉到 0.0703（更好看）；
> 但在从未训练过的留出文本上，loss 从 1.9692 升到 2.9546（更差）。
>
> 即：**无位置编码的模型在背训练集**，含位置编码的模型在学规律。

这个反直觉结果来自一个具体的机制（字符多重集泄漏位置），第 8 节展开。

## 2. 项目目标

| 目标 | 验收方式 |
|---|---|
| 自己实现多头自注意力（含因果掩码） | 不用 `nn.MultiheadAttention`，手写 QKV + mask + softmax |
| 自己实现 FFN / Pre-LN / 位置编码 / 输出头 | 四个模块都在脚本里，可单独 import |
| 从零训练出能续写的字符级 LM | 给定 `"the small robot "` 续写 32 token |
| 记录可验证指标 | 初始/最终 loss、PPL、参数量、耗时、生成样例 |
| **对比：有位置编码 vs 无位置编码** | 两条 loss 曲线的终值 + 置换不变性探针 |
| 不许美化 | 留出集比训练集差就照实写；探针没证明的也照实写 |

## 3. 技术栈与环境

| 项 | 值 |
|---|---|
| Python | 3.13（`py -3.13`，默认 `python` 是 3.15 无 torch） |
| PyTorch | 2.15.0+cu132 |
| GPU | NVIDIA GeForce RTX 4070 Ti，12.88 GB，sm_89 |
| 第三方依赖 | 无（本项目的 Transformer 全部手写，不 import transformers） |
| 模型规模 | d_model=128，4 头，2 层，d_ff=512，block=64 |
| 训练 | AdamW lr=2e-3，betas=(0.9,0.99)，wd=0.01，warmup 50 + 余弦衰减，800 步，batch 32，梯度裁剪 1.0 |
| 显存/耗时 | 单次实验约 7–9 s（GPU 与其他进程共享，绝对耗时有波动） |

## 4. 数据集

**自建**：脚本内写死一段手写英文短文，主题是"小机器人去看海"。

```text
语料：《The small robot》手写短文       1340 字符
词表：34 个字符（大小写字母 + 空格 + 标点 + 换行）
切分：前 1072 字符训练 / 后 268 字符留出（留出段从不参与训练）
```

为什么用英文而不是中文：字符级中文词表会膨胀到几千字，1340 字的语料下每个字符样本太少，训练不出东西。英文 34 个字符，`ln(34)=3.5264` 就是"瞎猜"的初始 loss，便于判断模型是否真的学了东西。

留出段的 268 字符是**最后 5 句话**，模型一次都没见过——这是后面"背 vs 学"判断的唯一依据。

## 5. 方案设计

```text
token id (B,T)
  → tok_emb (34 × 128)
  → [+ 正弦位置编码 (T,128)]        ← 实验 B 删掉这一步
  → Block × 2:
        x = x + CausalSelfAttention(LayerNorm(x))   ← Pre-LN
        x = x + FFN(LayerNorm(x))
  → LayerNorm
  → head (128 → 34)                 ← 与 tok_emb 权重绑定
  → logits (B,T,34) → cross_entropy(shift)
```

**参数量手算校验**（必须和脚本打印的 401,152 对上）：

| 部分 | 计算 | 参数 |
|---|---|---|
| token embedding | 34 × 128 | 4,352 |
| c_attn（QKV 合一） | 128×384 + 384 | 49,536 |
| c_proj | 128×128 + 128 | 16,512 |
| LayerNorm ×2 | 2 × 2 × 128 | 512 |
| FFN（fc1+fc2） | (128×512+512) + (512×128+128) | 131,712 |
| **单层合计** | | **198,272** |
| 2 层 | 198,272 × 2 | 396,544 |
| ln_f | 2 × 128 | 256 |
| **总计** | 396,544 + 256 + 4,352 | **401,152** |

输出头不额外占参数：`self.head.weight = self.tok_emb.weight`（GPT-2 的权重绑定做法），手算与实测一致。

**对照实验的唯一变量**：`use_pos=True/False`。其余（种子、数据、步数、lr、初始化）完全相同。

## 6. 核心代码

### 6.1 多头自注意力 + 因果掩码（自包含可运行）

```python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """多头自注意力 + 因果掩码（下三角），从零实现。"""

    def __init__(self, d_model=128, n_head=4, block_size=64, dropout=0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.c_attn = nn.Linear(d_model, 3 * d_model)   # QKV 一次算完
        self.c_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
            persistent=False,
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))  # 因果掩码
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


if __name__ == "__main__":
    attn = CausalSelfAttention()
    x = torch.randn(2, 8, 128)
    out = attn(x)
    print("out:", tuple(out.shape), "参数量:", sum(p.numel() for p in attn.parameters()))

    # 因果性自检：改第 5 个 token，前 5 个位置的输出必须不变
    x2 = x.clone()
    x2[:, 5] += 1.0
    print("前 5 位最大变化:", (out[:, :5] - attn(x2)[:, :5]).abs().max().item())
    print("第 5 位之后最大变化:", (out[:, 5:] - attn(x2)[:, 5:]).abs().max().item())
```

实测输出（真的跑了这段，不是手算）：

```text
out: (2, 8, 128) 参数量: 66048
前 5 位最大变化: 0.0
第 5 位之后最大变化: 0.196755051612854
```

前 5 位变化 **严格 0.0**、第 5 位及之后变化 0.197 —— 说明扰动第 5 个 token 只影响它自己和后面的位置，因果掩码是对的。

### 6.2 正弦位置编码（自包含）

```python
import math

import torch


def sinusoidal_position_encoding(block_size: int, d_model: int) -> torch.Tensor:
    """原论文式正弦位置编码，不可学习。返回 (block_size, d_model)。"""
    pe = torch.zeros(block_size, d_model)
    pos = torch.arange(0, block_size, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                    * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


if __name__ == "__main__":
    pe = sinusoidal_position_encoding(64, 128)
    print("shape:", tuple(pe.shape), "dtype:", pe.dtype)
    print("第 0 行前 4 维:", pe[0, :4].tolist())
    print("第 1 行前 4 维:", pe[1, :4].tolist())
```

实测输出（第 0 行偶数列全是 `sin(0)=0`、奇数列全是 `cos(0)=1`，符合预期）：

```text
shape: (64, 128) dtype: torch.float32
第 0 行前 4 维: [0.0, 1.0, 0.0, 1.0]
第 1 行前 4 维: [0.8414709568023682, 0.5403023362159729, 0.7617204189300537, 0.6479058265686035]
```

### 6.3 Pre-LN 残差块（`D_FF` 是脚本内常量，本片段依赖它）

```text
class Block(nn.Module):
    """Pre-LN 残差块：x = x + Attn(LN(x)); x = x + FFN(LN(x))"""
    def __init__(self, d_model, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, block_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, D_FF)   # D_FF=512，脚本内常量

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # ← 先归一化再进子层 = Pre-LN
        x = x + self.ffn(self.ln2(x))
        return x
```

Pre-LN 与 Post-LN 的差别只在这里的括号位置：Post-LN 是 `LN(x + Attn(x))`。Pre-LN 的主干上是一条**无归一化的残差直通路**，梯度回传更稳，所以 2 层小模型也敢用 lr=2e-3。

## 7. 运行结果

完整控制台输出（真实运行，未删改）：

```text
==================================================================================
PROJ-021 从零实现 Transformer 语言模型（字符级）
==================================================================================
device=cuda  torch=2.15.0+cu132
gpu=NVIDIA GeForce RTX 4070 Ti  vram=12.88 GB
语料字符数=1340  词表大小=34  理论初始 loss=ln(34)=3.5264
切分: 训练 1072 字符 / 留出 268 字符（留出段从不参与训练）
配置: block=64 d_model=128 n_head=4 n_layer=2 d_ff=512 batch=32 steps=800 lr=0.002 dropout=0.0

--- 实验 A：有正弦位置编码 ---
  step    1 | batch 3.5243 | train 3.5102 | heldout 3.5205 | lr 0.00008
  step  100 | batch 2.9314 | train 2.9013 | heldout 2.9448 | lr 0.00198
  step  200 | batch 2.7957 | train 2.5090 | heldout 2.6971 | lr 0.00181
  step  300 | batch 2.3423 | train 2.3477 | heldout 2.5735 | lr 0.00150
  step  400 | batch 2.0429 | train 1.8869 | heldout 2.1941 | lr 0.00110
  step  500 | batch 1.8268 | train 1.7128 | heldout 2.0831 | lr 0.00069
  step  600 | batch 1.6377 | train 1.5554 | heldout 2.0090 | lr 0.00033
  step  700 | batch 1.5172 | train 1.4720 | heldout 1.9862 | lr 0.00009
  step  800 | batch 1.4612 | train 1.4480 | heldout 1.9692 | lr 0.00000

--- 实验 B：无位置编码（去掉 pos_emb，其余完全相同） ---
  step    1 | batch 3.5919 | train 3.5382 | heldout 3.5415 | lr 0.00008
  step  100 | batch 1.7571 | train 1.0104 | heldout 2.0420 | lr 0.00198
  step  200 | batch 0.6468 | train 0.3436 | heldout 2.3276 | lr 0.00181
  step  300 | batch 0.2427 | train 0.1713 | heldout 2.6791 | lr 0.00150
  step  400 | batch 0.1422 | train 0.1244 | heldout 2.7082 | lr 0.00110
  step  500 | batch 0.1036 | train 0.0926 | heldout 2.9170 | lr 0.00069
  step  600 | batch 0.0854 | train 0.0771 | heldout 2.8925 | lr 0.00033
  step  700 | batch 0.0766 | train 0.0719 | heldout 2.9216 | lr 0.00009
  step  800 | batch 0.0719 | train 0.0703 | heldout 2.9546 | lr 0.00000

==================================================================================
结果汇总
==================================================================================
配置              参数量    初始loss    训练loss    留出loss    训练PPL    留出PPL    耗时(s)
有位置编码       401,152    3.5298    1.4480    1.9692     4.25     7.16     9.15
无位置编码       401,152    3.5925    0.0703    2.9546     1.07    19.19     8.80

--- 生成样例（32 token，temperature=0.8, top_k=40, 同一随机种子）---
[有位置编码] 'the small robot thea and nad bo the the sooy sev'
[无位置编码] 'the small robot cleaned the floor and sorted the'

--- 置换不变性探针（同一字符多重集、不同词序）---
  前缀 A = 'small robot the '
  前缀 B = 'robot small the '
  有位置编码: 第1层 max|Δp|=0.00889185  整模型 max|Δp|=0.06632301
      top1: 第1层 A='b'/B='b'  整模型 A='s'/B='s'
  无位置编码: 第1层 max|Δp|=0.00000000  整模型 max|Δp|=0.11934590
      top1: 第1层 A='w'/B='w'  整模型 A='a'/B='a'

指标已写入 ...\proj021_metrics.json
Δ留出 loss(无位置-有位置) = +0.9854   Δ留出 PPL = +12.03
Δ训练 loss(无位置-有位置) = -1.3777
```

## 8. 结果分析

### 8.1 三条曲线讲了三件事

| 读数 | 有位置编码 | 无位置编码 | 说明 |
|---|---|---|---|
| 训练 loss | 1.4480 | **0.0703** | 无 PE 把训练段背下来了 |
| 留出 loss | **1.9692** | 2.9546 | 无 PE 在没见过的文本上崩了 |
| 留出 PPL | **7.16** | 19.19 | 差 2.7 倍 |
| 初始 loss | 3.5298 | 3.5925 | 都≈ln(34)=3.5264，初始化正常 |

**为什么无 PE 的训练 loss 能低到 0.0703（PPL 1.07）？**

没有位置编码时，位置 $i$ 的输出只依赖两样东西：当前 token $x_i$，以及前缀的**字符多重集** $\{x_1..x_i\}$（注意力是对集合求和，与顺序无关）。而本语料是**一段连续文本**，字符计数随位置单调增长——"目前为止出现过多少个 a、多少个空格"几乎唯一确定了"我现在在第几个字符"。于是无 PE 模型等价于一张 `字符直方图 → 下一个字符` 的查找表，它把 1072 个字符背了下来，训练 loss 自然趋近 0。

一旦换到留出段，直方图的取值区间整个跑出了训练分布，查找表失效 → 2.9546（PPL 19.19），比"瞎猜"的 3.5264 只好一点。**这就是"背"与"学"的量化差别。**

### 8.2 置换不变性探针：我自己写的断言只对了一半

探针输入两段**字符多重集完全相同、词序不同**的前缀：`'small robot the '` 与 `'robot small the '`。

- 无位置编码：**第 1 层**输出 `max|Δp| = 0.00000000`（严格相等），完全符合"置换等变"的理论预期；
- 有位置编码：第 1 层 `max|Δp| = 0.0089`，顺序信息确实被编码进去了；
- 但**整模型**层面，无 PE 也出现了 `0.1193` 的差异。

原因：第 2 层看到的 value 集合是"各位置的第 1 层输出"，而每个位置的第 1 层输出又依赖**它自己的前缀多重集**——堆到第 2 层，顺序信息从"每个位置各自的前缀"里漏了回来。所以"无位置编码 = 完全看不见顺序"只在**单层**成立，2 层就不是严格成立了。

这一条是我最初写探针时**预期错了**（我预期整模型也是 0）。如实记录：探针只证明了单层的严格置换不变性，整模型的不变性被深度击穿了。真正可靠的证据是 8.1 的训练/留出 gap，不是这个探针。

### 8.3 生成样例：好看的那个反而是背的

```text
[无位置编码] 'the small robot cleaned the floor and sorted the'   ← 语料里的原句，一字不差
[有位置编码] 'the small robot thea and nad bo the the sooy sev'   ← 拼错的词
```

无 PE 的输出是训练集里的原句片段（"cleaned the floor and sorted the"），因为它在按直方图查表。有 PE 的输出反而更"烂"，但它是**在生成**（有 `thea`、`sooy` 这种拼错但符合字符级统计的新词），不是在背诵。

**结论不能只看生成样例是否通顺**——通顺可能是背的，拼错可能是泛化的代价。

## 9. 数据来源与差异说明（必读）

1. **语料是我手写的合成文本**，不是任何真实语料库，没有版权与隐私问题，但也**没有任何真实语言分布的复杂性**。
2. 全文 1340 字符，词表 34，属于"玩具规模"。真实字符级 LM 通常用几 MB 到几 GB 文本，PPL 量级完全不同，**本项目的 PPL 数字不能与任何论文数字对比**。
3. **"无位置编码更差"这个结论只在本设定下成立**。如果语料换成"每行独立、顺序随机"的多句文本，字符直方图不再单调定位，无 PE 的背诵优势会消失，两条曲线可能重合。我没有做这个对照，所以**不能外推**成"任何情况下位置编码都提升 2.7 倍 PPL"。
4. 训练/留出只切了一刀（前 80% / 后 20%），没有交叉验证，单次运行的 loss 有 ±0.02 量级波动（我重跑过三次，终值稳定在 1.96–1.97 / 2.95–2.96）。
5. dropout=0、800 步、单卡单种子：结论是"这个配置下的一次观测"，不是统计显著的 A/B。
6. 探针实验的**预期是错的**（见 8.2），文档保留了错误预期的说明，没有事后改写成"我早就知道"。

## 10. 复现步骤

```text
1. 确认解释器（默认 python 3.15 没有 torch）：
   py -3.13 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   期望：2.15.0+cu132 True

2. 设置编码（中文输出不乱码）：
   $env:PYTHONIOENCODING='utf-8'

3. 运行（脚本在 09-llm/projects 下，约 20 s 跑完两个实验）：
   cd <USER_HOME>\Documents\deepseek-harness\default-workspace\09-llm\projects
   py -3.13 proj021_transformer_from_scratch.py

4. 校验：
   - 参数量必须是 401,152（两行相同）
   - 有 PE 留出 loss ≈ 1.9692 / PPL ≈ 7.16
   - 无 PE 留出 loss ≈ 2.9546 / PPL ≈ 19.19
   - 同目录生成 proj021_metrics.json

5. 偏差排查：
   - 参数量不对 → 检查是否漏了权重绑定、d_ff 是否为 512
   - 留出 loss 明显不同 → 检查随机种子（SEED=42）与切分点（int(1340*0.8)=1072）
   - 无 PE 的留出 loss 没有上升 → 检查 pos_emb 是否真的被跳过（use_pos=False）
```

## 11. 完整训练脚本

脚本位置与运行命令：

```text
路径：<USER_HOME>\Documents\deepseek-harness\default-workspace\09-llm\projects\proj021_transformer_from_scratch.py
大小：19,940 字节
命令：py -3.13 proj021_transformer_from_scratch.py
产出：proj021_metrics.json（同目录）
```

脚本内固定配置（`__main__` 前一段）：

```python
BLOCK = 64
D_MODEL = 128
N_HEAD = 4
N_LAYER = 2
D_FF = 512
DROPOUT = 0.0
BATCH = 32
STEPS = 800
LR = 2e-3
WARMUP = 50
EVAL_EVERY = 100
TRAIN_FRAC = 0.8
SEED = 42
```

脚本结构（按行号）：

```text
1-30     文件头、环境变量（HF_ENDPOINT / HF_HUB_OFFLINE）、import
31-62    CORPUS 手写语料（1340 字符）
63-90    超参数常量 + set_seed
91-160   CausalSelfAttention / FeedForward / Block
161-190  sinusoidal_position_encoding
191-250  TinyTransformerLM（含 generate 采样）
251-300  build_data / get_batch / eval_loss / order_probe
301-380  train_one（训练 + 固定滑窗评测 + 生成 + 探针）
381-470  __main__：两个实验 + 结果汇总 + 写 JSON
```

运行环境要求：`py -3.13` + torch 2.15.0，**不需要 transformers**，不联网。

## 12. 验收清单

| 验收项 | 是否达成 | 证据 |
|---|---|---|
| 自己实现多头自注意力（含因果掩码） | ✅ | 第 6.1 节 `CausalSelfAttention`，6.1 自检显示前 5 位变化 0.0 |
| 自己实现 FFN | ✅ | 脚本 `FeedForward`：Linear→GELU→Linear，d_ff=512 |
| 自己实现 Pre-LN | ✅ | 第 6.3 节 `Block`：`x = x + Attn(LN(x))` |
| 自己实现位置编码 | ✅ | 第 6.2 节正弦编码，无参数、注册为 buffer |
| 自己实现输出头 | ✅ | `head = nn.Linear(128,34,bias=False)`，与 `tok_emb` 权重绑定 |
| 字符级 + 自建小语料 | ✅ | 1340 字符，词表 34，脚本内写死 |
| 训练 300–800 步 | ✅ | `STEPS = 800` |
| 记录初始 loss | ✅ | 3.5298（有 PE）/ 3.5925（无 PE） |
| 记录最终 loss + PPL | ✅ | 1.4480 / PPL 4.25（训练），1.9692 / PPL 7.16（留出） |
| 记录参数量 | ✅ | 401,152（手算与实测一致，见第 5 节） |
| 记录训练耗时 | ✅ | 9.15 s / 8.80 s |
| 生成样例 20–40 token | ✅ | 32 token，两条样例见第 7 节 |
| **有 PE vs 无 PE 两条曲线终值** | ✅ | 第 7 节两条曲线各 9 个点，终值 1.4480 vs 0.0703（训练）、1.9692 vs 2.9546（留出） |
| 指标写 JSON | ✅ | `proj021_metrics.json`（4,221 字节） |
| 可重复运行（固定种子） | ✅ | `SEED=42`，`set_seed()` 同时设 random/torch/cuda；重跑 3 次终值稳定 |
| 数据来源诚实说明 | ✅ | 第 9 节 6 条，含"预期错了"的说明 |

## 13. 结论与局限

**结论**

1. 手写 401,152 参数的 2 层 Transformer，800 步、9.15 s，把 1340 字符语料的留出 loss 从 3.5298 降到 **1.9692（PPL 7.16）**。
2. **位置编码的价值要看留出集**：训练集上无 PE 更好（0.0703 vs 1.4480），留出集上有 PE 更好（1.9692 vs 2.9546）。只报训练 loss 会得到完全相反的结论。
3. 置换不变性在**单层**严格成立（`max|Δp|=0.0`），2 层就不成立（0.1193）——深度会把顺序信息漏回来。
4. "生成通顺"不代表"学会了"：无 PE 的通顺输出是训练集原句。

**局限**（不修饰）

1. 语料 1340 字符是玩具规模，PPL 数值不能与任何真实结果对比。
2. 只切了一刀训练/留出，没做交叉验证；单种子，±0.02 的波动足以改变"谁更好"的排序。
3. 没有 dropout、没有早停；有 PE 的模型在 step 100–150 卡在 2.9 附近约 100 步才突破，说明这个配置对它并不友好（可能是 warmup 后 lr 过高的短暂平台）。
4. 探针只证明了单层不变性，没有设计出能证明"多层也看不见顺序"的实验——这条留作待办。
5. 没做"多句独立语料"的对照，所以第 9 节第 3 条的边界条件只是推断，没有实测。

## 14. 与后续项目的关系

- 本项目产出的 `CausalSelfAttention` / `Pre-LN Block` 是后面所有 LLM 项目的地基：[[LLM-021 Transformer 总体架构]]、[[LLM-017 Multi-Head Attention]]、[[LLM-025 残差、LayerNorm 与 Pre-LN、Post-LN]] 的理论在这里第一次变成能跑的代码。
- 位置编码的讨论接 [[LLM-024 位置编码]]；如果要做 RoPE 对照，属于 `LLM-028 RoPE 与长上下文外推`（待建）。
- **PROJ-022**（LoRA 微调 gpt2）是"不再从零训"的下一步：同一个 Transformer，换成 124M 预训练权重 + 低秩适配。见 [[PROJ-022 用 LoRA 微调 gpt2 做文本分类]]。
- 想继续深挖本项目的负结果：把语料换成"20 句独立短句、句序打乱"，重跑有/无 PE 对照，验证第 9 节第 3 条的边界条件。

## 学习记录

| 日期 | 状态 | 用时 | 问题 | 复盘 |
|---|---|---|---|---|
| 2026-11-08 | planned | — | — | — |

---

## 🔗 Notion 同步信息

> 以下内容由 Notion「AI Knowledge System · 实战项目」数据库按知识 ID 合并而来（导出于 2026-11-09）。

| 属性 | 值 |
|---|---|
| 所属学科 | 机器学习 |
| 项目状态 | 未开始 |
| 项目阶段 | 立项 |
| 难度 | L3 推导 |
| 技术栈 | Python、PyTorch、NumPy |
| 项目仓库 | ai-knowledge-system/projects/PROJ-021 |

- **对应课程模块**：[M-LLM-05 GPT、BERT 与现代语言模型架构](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-LLM-05%20GPT%E3%80%81BERT%20%E4%B8%8E%E7%8E%B0%E4%BB%A3%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84.md) · [M-LLM-06 大语言模型训练与对齐](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-LLM-06%20%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E4%B8%8E%E5%AF%B9%E9%BD%90.md)
- **前置知识**：[LLM-021 Transformer 总体架构](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/04%20Transformer%E6%9E%B6%E6%9E%84/LLM-021%20Transformer%20%E6%80%BB%E4%BD%93%E6%9E%B6%E6%9E%84.md) · [LLM-025 残差、LayerNorm 与 Pre-LN/Post-LN](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/04%20Transformer%E6%9E%B6%E6%9E%84/LLM-025%20%E6%AE%8B%E5%B7%AE%E3%80%81LayerNorm%20%E4%B8%8E%20Pre-LN%E3%80%81Post-LN.md) · [LLM-030 Transformer 参数量与计算量估算](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/04%20Transformer%E6%9E%B6%E6%9E%84/LLM-030%20Transformer%20%E5%8F%82%E6%95%B0%E9%87%8F%E4%B8%8E%E8%AE%A1%E7%AE%97%E9%87%8F%E4%BC%B0%E7%AE%97.md)
- **对应知识点**：[LLM-013 点积注意力与 Scaled Dot-Product Attention](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/03%20Attention%E6%9C%BA%E5%88%B6/LLM-013%20%E7%82%B9%E7%A7%AF%E6%B3%A8%E6%84%8F%E5%8A%9B%E4%B8%8E%20Scaled%20Dot-Product%20Attention.md) · [LLM-017 Multi-Head Attention](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/03%20Attention%E6%9C%BA%E5%88%B6/LLM-017%20Multi-Head%20Attention.md) · [LLM-024 位置编码](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/04%20Transformer%E6%9E%B6%E6%9E%84/LLM-024%20%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81.md) · [LLM-025 残差、LayerNorm 与 Pre-LN/Post-LN](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/04%20Transformer%E6%9E%B6%E6%9E%84/LLM-025%20%E6%AE%8B%E5%B7%AE%E3%80%81LayerNorm%20%E4%B8%8E%20Pre-LN%E3%80%81Post-LN.md) · [LLM-030 Transformer 参数量与计算量估算](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/04%20Transformer%E6%9E%B6%E6%9E%84/LLM-030%20Transformer%20%E5%8F%82%E6%95%B0%E9%87%8F%E4%B8%8E%E8%AE%A1%E7%AE%97%E9%87%8F%E4%BC%B0%E7%AE%97.md)

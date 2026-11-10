---
id: PROJ-023
title: 搭建一个最小可用的 RAG 问答系统
domain: PROJ
module: [M-LLM-08]
type: project
difficulty: L3 推导
status: planned
tech_stack: [Python, NumPy, PyTorch, Transformers]
prerequisites: [LLM-076, LLM-077, LLM-079]
related_math: [MATH-216]
related: [LLM-037, LLM-083, PROJ-024]
notion_page_id: 412663e8-0333-818f-88d8-d67f0c5111ee
notion_url: https://app.notion.com/p/RAG-412663e80333818f88d8d67f0c5111ee
created: 2026-11-08
updated: 2026-11-08
notion_db: 实战项目
所属学科: 机器学习
项目状态: 未开始
项目阶段: 立项
难度: L3 推导
项目目标: 搭建一个最小可用的 RAG 问答系统
技术栈: [Python, NumPy, PyTorch, Transformers]
项目仓库: ai-knowledge-system/projects/PROJ-023
对应课程模块: [M-LLM-08]
前置知识: [LLM-076, LLM-077, LLM-079]
对应知识点: [LLM-037, LLM-083]
tags: [机器学习]
---

# PROJ-023 搭建一个最小可用的 RAG 问答系统

## 1. 项目背景

RAG 的教程通常这样写：切块 → 向量化 → top-k 检索 → 拼进 prompt → 生成。看起来每一步都"显然正确"，但真正跑起来会撞上三个问题：

1. **没有 embedding 模型怎么办？** 本机只缓存了 `gpt2` 和 `tiny-gpt2`，没有 `text-embedding-*`。只能用 gpt2 的 token 嵌入做平均池化——这是**廉价替代**，不是真 embedding。
2. **生成模型很弱怎么办？** gpt2 不是指令模型，它只会"续写"，不会"回答"。
3. **怎么知道检索好不好？** 需要 gold 标注 + Recall@k。

本项目把这三件事都做了，并且**实测出一个不太好听的结果**：

- **Recall@1 = 0.60（6/10），Recall@3 = 0.80（8/10）**；
- 检索本身很快（平均 **1.265 ms**），端到端平均 **531.9 ms**，其中 99.8% 花在 gpt2 生成上；
- **gpt2 生成的答案与检索到的上下文矛盾**：上下文写"两年保修"，它答"30 天保修"。

结论：**检索和生成是两条独立的短板，必须分开量。** 只看端到端"能不能用"，会把两个问题混成一个。

## 2. 项目目标

| 目标 | 验收方式 |
|---|---|
| 自建 8–15 篇短文档知识库 | 12 篇虚构产品 FAQ，脚本内写死 |
| 切块 | 按句切分，12 篇 → 22 个 chunk |
| 向量化（说明是廉价替代） | gpt2 `wte` 平均池化 + L2 归一化，768 维 |
| numpy 余弦 top-k 检索 | `chunk_vecs @ q` + `argsort`，无向量库 |
| 把 top-k 拼进 prompt 用 gpt2 生成 | 生成 40 token，原样打印不美化 |
| **Recall@1 / Recall@3** | 0.60 / 0.80，逐题可核（8–10 个问题 + 人工 gold） |
| 平均检索耗时 + 端到端耗时 | 1.265 ms / 531.9 ms |
| **一个检索失败案例的诚实分析** | 4 题 R@1 未命中、2 题 R@3 未命中，逐个给 Jaccard 证据 |
| 额外：上下文消融 | 同一问题"喂/不喂"片段各生成一次，对比是否更好 |

## 3. 技术栈与环境

| 项 | 值 |
|---|---|
| Python | 3.13（`py -3.13`） |
| PyTorch | 2.15.0+cu132 |
| transformers | 4.57.6 |
| numpy | 2.3.5 |
| GPU | RTX 4070 Ti，12.88 GB，sm_89 |
| 模型 | `gpt2`（124M，本地缓存，离线加载） |
| 向量维度 | 768（gpt2 hidden size） |
| 检索 | 纯 numpy 余弦，无 FAISS / 无向量库 |
| 生成 | `model.generate`，max_new_tokens=40，temperature=0.7，top_p=0.9 |

**明确声明**：本项目**没有**真正的 embedding 模型。gpt2 的 `wte` 是**静态、上下文无关**的词嵌入矩阵，把它按 token 平均，得到的不是句向量。这是本项目最大的简化，第 8、9 节用数据说明这个简化到底坏了多少事。

## 4. 数据集

**知识库**：12 篇手写英文 FAQ，主题是一个虚构的智能家居网关 "Nimbus Hub"（不存在这个产品）。每篇 1–2 句。

```text
warranty       The Nimbus Hub comes with a two-year limited warranty. The warranty covers
               manufacturing defects but not water damage.
returns        You can return the Nimbus Hub within 30 days of delivery for a full refund. ...
setup          Setup takes about five minutes. Plug the hub into power, ...
wifi           The hub needs a 2.4 GHz Wi-Fi network. It does not support 5 GHz only networks.
power          The hub draws 3 watts in normal use and 0.5 watts in standby. ...
subscription   Basic automation is free. The Nimbus Plus subscription costs 5 dollars per month ...
compatibility  The hub works with Zigbee and Matter devices. It does not work with Z-Wave devices.
firmware       Firmware updates install automatically at night. ...
reset          To factory reset the hub, hold the button on the back for ten seconds ...
camera         The optional Nimbus Cam records in 1080p. Local storage is a microSD card up to 128 GB.
offline        Automations keep running when the internet is down, but remote control stops working.
support        Support is available by email from Monday to Friday, 9am to 6pm. ...
```

切块粒度：**一个句子一个 chunk**（12 篇 → **22 个 chunk**，平均 12.18 token，最长 21）。

**问题集**：10 个手写问题 + 人工 gold 文档 id。第 10 题允许两个 gold（`subscription` + `camera`），因为"不订阅能不能用摄像头"必须同时看订阅页和摄像头页。

```text
1  How long is the warranty?                                → warranty
2  Can I return the hub if I do not like it?                → returns
3  How long does setup take?                                → setup
4  Does it support 5 GHz Wi-Fi?                             → wifi
5  How much power does the hub use?                         → power
6  How much does the subscription cost?                     → subscription
7  Does it work with Z-Wave devices?                        → compatibility
8  How do I factory reset the hub?                          → reset
9  Can automations run when the internet is down?           → offline
10 Can I use the camera without paying for a subscription?  → subscription 或 camera
```

Recall@k 的判定：top-k 里**任一** chunk 的 doc_id 命中 gold 集合即算命中（文档级召回，不是句子级）。

## 5. 方案设计

```text
离线（一次）：
  12 篇文档 → 按句切分 → 22 个 chunk
  → gpt2 wte[token ids].mean(0) → L2 归一化 → 矩阵 (22, 768)   [建索引 79.77 ms]

在线（每题）：
  question → 同样方式向量化 → (768,)
  → sims = chunk_vecs @ q_vec        ← 两边都归一化过，点积=余弦
  → argsort 取 top-3
  → 拼 prompt：Context: [chunk]... Question: ... Answer:
  → gpt2.generate(max_new_tokens=40) → 答案
```

**为什么用"平均池化"而不是"最后 token 隐状态"**：本项目要的是**检索**，需要"一个句子一个向量、可以离线预先算好"。gpt2 的最后隐状态虽然更懂上下文，但 (a) 要跑前向、(b) 因果注意力的最后 token 只偏向句尾。平均池化是最便宜的选择，代价是丢掉词序与上下文——第 8 节会看到这个代价有多大。

**为什么用 numpy 而不是向量库**：22 × 768 的矩阵，暴力算一次的实测耗时是 **1.265 ms**（最低 0.616 ms）。在这个规模上任何向量库都是负优化。

## 6. 核心代码

### 6.1 用 gpt2 词嵌入做句向量（自包含可运行）

```python
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
model = AutoModelForCausalLM.from_pretrained("gpt2", local_files_only=True).eval()
W = model.transformer.wte.weight          # (50257, 768) 静态词嵌入


def embed(texts):
    out = []
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            v = W[ids].mean(dim=0)        # 平均池化
            v = v / (v.norm() + 1e-8)     # L2 归一化
            out.append(v)
    return torch.stack(out).numpy()


texts = [
    "the hub needs a 2.4 GHz Wi-Fi network .",
    "how much power does the hub use ?",
    "support is available by email .",
]
V = embed(texts)
print("shape:", V.shape, "norm:", np.round(np.linalg.norm(V, axis=1), 6).tolist())
iu = np.triu_indices(len(texts), 1)          # 只取上三角，打印 3 个两两余弦
print("pairwise cosine (0-1, 0-2, 1-2):", np.round((V @ V.T)[iu].astype(float), 4).tolist())
```

实测输出（**先看这个数再看后面的一切**）：

```text
shape: (3, 768) norm: [1.0, 1.0, 1.0]
pairwise cosine (0-1, 0-2, 1-2): [0.8031, 0.7592, 0.7293]
```

**三句毫不相干的话（Wi-Fi 频段 / 功耗 / 客服邮箱），余弦相似度是 0.73–0.80。** 这就是本方案的根本缺陷：静态词嵌入的平均值都落在同一个"窄锥"里，区分度极低。后面 Recall@1 只有 0.60，根子在这一步，不在检索代码。

### 6.2 余弦 top-k（自包含可运行）

```python
import numpy as np

chunk_vecs = np.stack([
    np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.7, 0.7, 0.0]),
])
chunk_vecs /= np.linalg.norm(chunk_vecs, axis=1, keepdims=True)   # 逐行归一化
q = np.array([1.0, 0.1, 0.0])
q /= np.linalg.norm(q)

sims = chunk_vecs @ q                  # 归一化后点积 = 余弦
order = np.argsort(-sims)[:2]
print("sims:", np.round(sims, 4).tolist(), "top2:", order.tolist())
```

实测输出：

```text
sims: [0.995, 0.0995, 0.774] top2: [0, 2]
```

### 6.3 真实 prompt（第 1 题，从 metrics.json 读回 top-3 拼出来的）

```text
Context:
[warranty#0] The Nimbus Hub comes with a two-year limited warranty.
[returns#0] You can return the Nimbus Hub within 30 days of delivery for a full refund.
[firmware#1] You can also trigger an update from the device settings in the app.

Question: How long is the warranty?
Answer:
```

第 1 题检索是对的（`warranty#0` 排第一），但**第 2、3 名是噪声**（退货、固件）。这就是 top-3 的实际质量：**1 个对的 + 2 个无关的**，全塞给一个不会读上下文的模型。

## 7. 运行结果

完整控制台输出（真实运行，未删改）：

```text
====================================================================================
PROJ-023 搭建一个最小可用的 RAG 问答系统
====================================================================================
device=cuda  torch=2.15.0+cu132
gpu=NVIDIA GeForce RTX 4070 Ti  vram=12.88 GB

知识库: 12 篇文档 → 切成 22 个 chunk（按句切分）
平均每 chunk token 数: 12.2（最长 21）
向量维度: (22, 768)  建索引耗时 79.8 ms

------------------------------------------------------------------------------------
问题                                                  top1 文档          R@1 R@3
------------------------------------------------------------------------------------
How long is the warranty?                           warranty           ✅   ✅
Can I return the hub if I do not like it?           reset              ❌   ✅
How long does setup take?                           compatibility      ❌   ❌
Does it support 5 GHz Wi-Fi?                        wifi               ✅   ✅
How much power does the hub use?                    setup              ❌   ✅
How much does the subscription cost?                subscription       ✅   ✅
Does it work with Z-Wave devices?                   compatibility      ✅   ✅
How do I factory reset the hub?                     reset              ✅   ✅
Can automations run when the internet is down?      offline            ✅   ✅
Can I use the camera without paying for a subscrip  firmware           ❌   ❌
------------------------------------------------------------------------------------
Recall@1 = 6/10 = 0.6000    Recall@3 = 8/10 = 0.8000
平均检索耗时 1.265 ms（min 0.616 / max 3.054）
平均端到端耗时 531.9 ms（含 gpt2 生成 40 token）

====================================================================================
生成样例（gpt2 很弱，原样打印，不做美化）
====================================================================================

Q: How long is the warranty?
top1: [warranty#0] The Nimbus Hub comes with a two-year limited warranty.
A : The warranty is for 30 days. Question: Do you have any warranty issues? Answer: The warranty is for 30 days. Question: Do you have any problems with the

Q: Can I return the hub if I do not like it?
top1: [reset#0] To factory reset the hub, hold the button on the back for ten seconds until the light turns red.
A : [returns#1] You can return the hub if you do not like it. Question: I want to cancel my purchase, and am wondering if I can change the value

Q: How long does setup take?
top1: [compatibility#1] It does not work with Z-Wave devices.
A : 1 month. [offline#1] Automations don't work when the internet is down, but remote control stops working. [wifi#2] It doesn't work with Z

====================================================================================
上下文消融（同一问题：喂 top-3 片段 vs 不喂片段）
====================================================================================

Q: How long is the warranty?
  有上下文: The warranty is for 30 days. Question: Do you have any warranty issues? Answer: The warranty is for 30 days. Question: Do you have any problems with the
  无上下文: It is 10 years from the date of purchase. Answer: The warranty is for one year from date of purchase. Answer: The warranty is for one year from date of purchase. Answer
  两者是否完全相同: False

Q: Can I return the hub if I do not like it?
  有上下文: [returns#1] You can return the hub if you do not like it. Question: I want to cancel my purchase, and am wondering if I can change the value
  无上下文: No. If you are not satisfied with the design or the functionality of the hub, you may return it. If you have any questions, please contact us via our online support page. How do
  两者是否完全相同: False

Q: How long does setup take?
  有上下文: 1 month. [offline#1] Automations don't work when the internet is down, but remote control stops working. [wifi#2] It doesn't work with Z
  无上下文: It depends. A good system for setup is to have a few people who are running it, and then you have two or three people who are running it. The setup will run for about 30 minutes
  两者是否完全相同: False

====================================================================================
检索失败案例：Recall@1 未命中 4/10，Recall@3 未命中 2/10
====================================================================================

Q: Can I return the hub if I do not like it?   gold=['returns']
  #1 score=0.8650 [reset#0] Jaccard(问,块)=0.083  To factory reset the hub, hold the button on the back for ten seconds until
  #2 score=0.8623 [firmware#1] Jaccard(问,块)=0.100  You can also trigger an update from the device settings in the app.
  #3 score=0.8567 [returns#0] Jaccard(问,块)=0.190  You can return the Nimbus Hub within 30 days of delivery for a full refund.
  gold [returns#0] Jaccard(问,块)=0.190  You can return the Nimbus Hub within 30 days of delivery for a full refund.
  gold [returns#1] Jaccard(问,块)=0.000  Opened units must include all original cables.
  → top1 Jaccard=0.083 vs gold 最高 Jaccard=0.191：静态平均池化只在比'词袋重合'，停用词(the/is/it)把分数挤在一起（top3 分数区间 0.857~0.865）

Q: How long does setup take?   gold=['setup']
  #1 score=0.8001 [compatibility#1] Jaccard(问,块)=0.083  It does not work with Z-Wave devices.
  #2 score=0.7989 [offline#0] Jaccard(问,块)=0.000  Automations keep running when the internet is down, but remote control stops
  #3 score=0.7939 [wifi#1] Jaccard(问,块)=0.083  It does not support 5 GHz only networks.
  gold [setup#0] Jaccard(问,块)=0.111  Setup takes about five minutes.
  gold [setup#1] Jaccard(问,块)=0.000  Plug the hub into power, open the Nimbus app, and follow the pairing steps.
  → top1 Jaccard=0.083 vs gold 最高 Jaccard=0.111：静态平均池化只在比'词袋重合'，停用词(the/is/it)把分数挤在一起（top3 分数区间 0.794~0.800）

Q: How much power does the hub use?   gold=['power']
  #1 score=0.8699 [setup#1] Jaccard(问,块)=0.188  Plug the hub into power, open the Nimbus app, and follow the pairing steps.
  #2 score=0.8549 [power#0] Jaccard(问,块)=0.188  The hub draws 3 watts in normal use and 0.5 watts in standby.
  #3 score=0.8470 [reset#0] Jaccard(问,块)=0.095  To factory reset the hub, hold the button on the back for ten seconds until
  gold [power#0] Jaccard(问,块)=0.188  The hub draws 3 watts in normal use and 0.5 watts in standby.
  gold [power#1] Jaccard(问,块)=0.071  It ships with a USB-C power adapter.
  → top1 Jaccard=0.188 vs gold 最高 Jaccard=0.188：静态平均池化只在比'词袋重合'，停用词(the/is/it)把分数挤在一起（top3 分数区间 0.847~0.870）

Q: Can I use the camera without paying for a subscription?   gold=['subscription', 'camera']
  #1 score=0.8760 [firmware#1] Jaccard(问,块)=0.100  You can also trigger an update from the device settings in the app.
  #2 score=0.8706 [returns#0] Jaccard(问,块)=0.190  You can return the Nimbus Hub within 30 days of delivery for a full refund.
  #3 score=0.8664 [offline#0] Jaccard(问,块)=0.045  Automations keep running when the internet is down, but remote control stops
  gold [subscription#0] Jaccard(问,块)=0.000  Basic automation is free.
  gold [subscription#1] Jaccard(问,块)=0.143  The Nimbus Plus subscription costs 5 dollars per month and adds camera recor
  → top1 Jaccard=0.100 vs gold 最高 Jaccard=0.143：静态平均池化只在比'词袋重合'，停用词(the/is/it)把分数挤在一起（top3 分数区间 0.866~0.876）

指标已写入 ...\proj023_metrics.json
```

## 8. 结果分析

### 8.1 检索：R@1 = 0.60，问题出在"分数挤在一起"

先看 4 个失败案例的**分数区间**：

| 问题 | top1 分数 | top3 分数区间 | 区间宽度 | 命中 |
|---|---|---|---|---|
| Can I return the hub... | 0.8650 | 0.8567–0.8650 | **0.0083** | ❌ R@1 |
| How long does setup take? | 0.8001 | 0.7939–0.8001 | 0.0062 | ❌ R@3 |
| How much power does the hub use? | 0.8699 | 0.8470–0.8699 | 0.0229 | ❌ R@1 |
| Can I use the camera without... | 0.8760 | 0.8664–0.8760 | **0.0096** | ❌ R@3 |

**top-3 的分数差 0.006–0.023，而正确与错误之间的差距也就这个量级。** 为了确认这一点，我把 gold chunk 在全部 22 个 chunk 里的**真实名次**也算了出来：

```text
gold 名次: [1, 3, 7, 1, 2, 1, 1, 1, 1, 8]   中位数 = 1
全库余弦范围: 0.7009 ~ 0.9689
```

- 10 题里 **6 题 gold 排第 1**、2 题排第 2/3、2 题掉到第 7 和第 8；
- 全库余弦确实有跨度（0.70–0.97），所以问题**不是"所有 chunk 都差不多"**；
- 真正的问题是**头部极度平坦**：`setup` 题 gold 在第 7 名，而 top1 才 0.8001，top3 只差 0.0062 —— 在这么平的头部上，任何一点噪声都能把正确答案挤出 top-3。

这跟 6.1 节三句无关句余弦高达 0.73–0.80 是同一件事的两面：**静态词嵌入平均池化给出的相似度，绝对值和区分度都不可靠。**

两个具体的失败机制：

**机制 A：词袋里的"字面碰撞"（第 5 题）**

问："How much **power** does the hub use?" → top1 = `setup#1`："Plug the hub into **power**, open the Nimbus app…"

两段文本的 Jaccard 都是 0.188（`power`、`the`、`hub` 三个词重合），但 **"plug into power"（插电）与 "power consumption"（功耗）是两个意思**。词袋分不出来，静态嵌入更分不出来——它压根没有上下文。

**机制 B：停用词把分数压平（第 3 题，R@3 也失败）**

gold 是 `setup#0`："Setup takes about five minutes."，Jaccard **0.111** 反而**高于** top1 的 0.083，却排到了第 4 名之外。原因：这句话太短、和问题的共享词只有 `setup`，而 22 个 chunk 里大量出现 `the` / `is` / `it` / `does`，平均池化把这些高频词的向量摊进每一句，**稀释了 `setup` 这个唯一有效信号**。

**机制 C：多跳问题（第 10 题）**

"不订阅能不能用摄像头"需要 `subscription#1`（订阅包含摄像头录像历史）+ `camera#0/#1`（本地 microSD 存储）。top-3 里两条 gold 都没进来，检索到的是固件、退货、离线三篇。**单向量检索天然不擅长"需要两条文档联合回答"的问题**——这是 RAG 的经典难点，不是本代码的 bug。

### 8.2 生成：gpt2 不但没用上上下文，还编造了一个引用

| 问题 | 检索 top1（正确答案） | gpt2 输出 | 判定 |
|---|---|---|---|
| 保修多久 | 两年有限保修 | "The warranty is for **30 days**." | **与上下文矛盾**（30 天来自"退货"文档） |
| 能退货吗 | reset（错的） | "[returns#1] You can return the hub if you do not like it." | 内容对，但 `returns#1` **根本不在 top-3 里**，是编的引用 |
| setup 多久 | compatibility（错的） | "1 month. [offline#1] …" | 与人话无关，是续写 |

三条都是真实输出，一条可用都没有：

1. **第 1 题**：上下文明确写 `two-year limited warranty`，gpt2 答 `30 days` —— 它把训练语料里"30 天退货"的常见搭配搬过来了，**上下文对它没有约束力**。
2. **第 2 题**：格式学得像（`[doc#chunk]` 前缀），但 `returns#1` 那一句其实是语料里 `returns` 的**第 2 句**（"Opened units must include all original cables."），和它生成的内容也不一致——**格式对、内容编**，这是最危险的一种"看起来像真的"。
3. **第 3 题**：输出里混进了 `[offline#1]`、`[wifi#2]`，纯粹是把 prompt 里的片段当文本续写。

**结论：gpt2（124M，非指令微调）不能当 RAG 的 reader。** 它能做的事情只有一件：模仿 prompt 的格式。

### 8.3 上下文消融：喂了片段，并没有更对

| 问题 | 有上下文 | 无上下文 | 谁更对 |
|---|---|---|---|
| 保修多久 | 30 天（错） | 10 年 / 1 年（也错） | 都错，有上下文错得更"具体" |
| 能退货吗 | 内容对、引用编造 | 语义接近"可以退"，但没给 30 天 | 无上下文反而没那么误导 |
| setup 多久 | 1 month（错） | "大约 30 分钟"（也错，真值 5 分钟） | 都错 |

三次输出**都不相同**（`identical` 全为 False），说明 prompt 确实影响了生成——但影响的方向是"措辞"而不是"事实"。这正是"弱 reader"的典型表现：**上下文改变了它的用词分布，没有改变它的知识**。

### 8.4 耗时结构：检索不是瓶颈

| 环节 | 平均耗时 | 占比 |
|---|---|---|
| 检索（含问题向量化） | **1.265 ms** | 0.24% |
| 端到端（检索 + 生成 40 token） | **531.9 ms** | 100% |

生成占 99.8%。优化检索（换更好的 embedding、加 reranker）在这个规模上对端到端延迟毫无意义；**真正的成本在 reader**。第 1 题端到端 832.5 ms 明显高于其他题（353.9–616.3 ms），因为那是第一次调用 `generate`，含 CUDA kernel 预热——这也是一个不该忽略的工程细节。

## 9. 数据来源与差异说明（必读）

1. **知识库 12 篇 FAQ 是我手写的虚构产品文档**，Nimbus Hub 不存在，参数（3 瓦、128 GB、5 美元/月）都是我编的。它们不包含真实文档的表格、层级标题、跨文档引用、版本冲突。
2. **向量化不是 embedding 模型**。gpt2 的 `wte` 是**静态、上下文无关**的词嵌入；平均池化会把 `not great` 和 `great` 拉得更近（否定词被平均掉了），也完全丢失词序。**本项目的 Recall 数字不能与任何用真 embedding 模型的 RAG 对比**（真模型在同类任务上 R@1 通常是 0.8+）。
3. **用英文而不是中文**：gpt2 的 tokenizer 是英文 byte-level BPE，中文会被切成字节碎片，平均池化后的向量更没意义。这不是"英文更好"，是"这个廉价方案只对英文勉强成立"。
4. **10 个问题 + 22 个 chunk 的规模太小**：1 题 = 10 个百分点，R@1=0.60 的 95% 置信区间大约是 [0.31, 0.83]。**不能据此说"这个方案 R@1 约 0.6"**，只能说"这 10 题对了 6 题"。
5. **gold 是我自己标的**，第 10 题允许两个 gold 集合是我的判断，换个人可能要求必须两条都进 top-3（那样 R@3 会更低）。
6. **没有做任何 RAG 工程优化**：没有 chunk overlap、没有 hybrid（BM25+向量）、没有 reranker、没有 query 改写、没有 HyDE、没有多跳检索。所以 0.60 是"最裸的基线"，不是"RAG 的上限"。
7. **gpt2 是续写模型不是指令模型**，它没有在"问答"任务上对齐过。第 8.2 节的矛盾输出是它的正常表现，不是 bug。
8. 耗时在共享 GPU 上测得（运行期间有别的进程占 97% 利用率），绝对毫秒数不可复现，比例关系（生成占 99.8%）稳定。

## 10. 复现步骤

```text
1. 确认解释器与模型缓存：
   py -3.13 -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
   期望：2.15.0+cu132 4.57.6

2. 运行（约 6 s，10 题 × 生成 40 token）：
   $env:PYTHONIOENCODING='utf-8'
   cd <USER_HOME>\Documents\deepseek-harness\default-workspace\09-llm\projects
   py -3.13 proj023_rag_qa.py

3. 校验：
   - 12 篇文档 → 22 个 chunk，平均 12.2 token
   - Recall@1 = 6/10 = 0.6000，Recall@3 = 8/10 = 0.8000
   - 平均检索耗时约 1.0–1.3 ms（CPU 上会慢一个量级）
   - 同目录生成 proj023_metrics.json

4. 偏差排查：
   - chunk 数不是 22 → 检查按句切分的正则 (?<=[.!?])\s+
   - Recall 变了 → 检查是否改了 TOP_K 或 gold 映射
   - 生成答案是空 → 检查 pad_token_id 是否设成 eos_token_id
   - 检索耗时异常大 → 检查是否每道题都重算了 chunk 向量（应只算一次）
```

## 11. 完整训练脚本

```text
路径：<USER_HOME>\Documents\deepseek-harness\default-workspace\09-llm\projects\proj023_rag_qa.py
大小：13,530 字节
命令：py -3.13 proj023_rag_qa.py
产出：proj023_metrics.json（13,170 字节，含 10 题完整记录）
```

脚本内固定配置：

```python
SEED = 42
MODEL_NAME = "gpt2"
TOP_K = 3
MAX_NEW_TOKENS = 40
TEMPERATURE = 0.7
TOP_P = 0.9
```

脚本结构（按行号）：

```text
1-30     环境变量、import
31-80    DOCS（12 篇 FAQ）+ QUESTIONS（10 题 + gold）
81-110   split_sentences / build_chunks（按句切块）
111-135  embed_texts（gpt2 wte 平均池化 + L2 归一化）
136-165  cosine_topk / word_jaccard / make_prompt
166-190  generate_answer（HF generate 封装）
191-300  __main__：逐题检索 + 生成 + Recall 统计 + 耗时
301-340  生成样例 + 上下文消融
341-380  失败案例（含 Jaccard 证据）+ 写 JSON
```

依赖：torch 2.15.0、transformers 4.57.6、numpy 2.3.5。**不联网**（离线加载 gpt2），**不调用任何外部 API**。

## 12. 验收清单

| 验收项 | 是否达成 | 证据 |
|---|---|---|
| 8–15 篇自建短文档 | ✅ | 12 篇虚构 FAQ，脚本内写死，每篇 1–2 句 |
| 切块 | ✅ | 按句切分，12 篇 → 22 chunk，平均 12.18 token |
| 向量化 + 说明是廉价替代 | ✅ | gpt2 `wte` 平均池化 + L2 归一化，第 3、9 节两次声明"不是真 embedding" |
| numpy 余弦 top-k | ✅ | `chunk_vecs @ q` + `argsort`，6.2 节可运行片段 |
| top-k 拼 prompt + gpt2 生成 | ✅ | 6.3 节真实 prompt，40 token，原样打印 |
| **Recall@1** | ✅ | **0.6000（6/10）** |
| **Recall@3** | ✅ | **0.8000（8/10）** |
| **平均检索耗时** | ✅ | **1.265 ms**（min 0.616 / max 3.054） |
| **端到端耗时** | ✅ | **531.9 ms**（生成占 99.8%） |
| 8–10 个问题 + 脚本内 gold 映射 | ✅ | 10 题，gold 写死在 `QUESTIONS`，第 10 题两个 gold |
| **检索失败案例诚实分析** | ✅ | 4 题 R@1 未命中 + 2 题 R@3 未命中，逐题给分数区间与 Jaccard，归纳出 3 个失败机制 |
| 生成质量如实记录 | ✅ | 指出"两年"→"30 天"的矛盾、编造 `[returns#1]` 引用 |
| 指标写 JSON | ✅ | `proj023_metrics.json`（13,170 字节，含每题 top-3 与答案） |
| 可重复运行 | ✅ | `SEED=42`，每题 `torch.manual_seed(SEED+qi)`；检索结果确定性 |
| 不调用外部 API | ✅ | 全部本地，`HF_HUB_OFFLINE=1` |

## 13. 结论与局限

**结论**

1. **最小可用 RAG 能跑通，但"可用"要分开看**：检索 R@1 = 0.60 / R@3 = 0.80，生成基本不可用（3 条样例 0 条正确）。
2. **瓶颈不在检索速度**：1.265 ms vs 531.9 ms 端到端，检索占 0.24%。要提速只能优化 reader。
3. **瓶颈在"句子表示"**：静态词嵌入平均池化让**头部 chunk 的余弦挤在 0.79–0.88**（全库 0.70–0.97），top-3 之间的分数差只有 0.006–0.023，而 gold 在两题里分别掉到第 7、第 8 名 —— 排序在头部近乎随机。这一条是本项目最有价值的诊断，因为它**可以单独归因**。
4. **gpt2 不能当 reader**：它会与上下文矛盾（two-year → 30 days），会编造引用（`[returns#1]`），会把 prompt 里的片段当文本续写。上下文消融证明：喂了片段只改变措辞，不改变事实。
5. **多跳问题（第 10 题）需要多文档联合**，单向量 top-k 天然吃亏，这是 RAG 的结构性问题。

**局限**（不修饰）

1. 没有真 embedding 模型，所以 0.60 是"廉价方案的下限"，**不能代表 RAG 的正常水平**。
2. 10 题 / 22 chunk 太小，1 题 = 10 个百分点，统计上没有意义，只能当"逐题 case study"。
3. gold 是我自己标的；第 10 题的双 gold 是我放宽后的判定，换标准 R@3 会掉。
4. 没有 chunk overlap、没有 reranker、没有 hybrid 检索、没有 query 改写——**一个优化都没做**，所以这个 0.60 是"裸基线"，不是"RAG 的能力上限"。
5. 生成只跑了 3 题的消融，其余 7 题的答案在 JSON 里但没有逐条人工评估；**没有做人工正确率标注**，所以"生成不可用"是 3 条样例的观察，不是 10 题的统计。
6. 计时在共享 GPU 上，绝对毫秒不可复现。

## 14. 与后续项目的关系

- 上游：[[PROJ-022 用 LoRA 微调 gpt2 做文本分类]] 证明 gpt2 的隐状态能撑起分类；本项目证明**它的静态词嵌入撑不起检索、它的生成撑不起回答**。同一个模型的两种用法，两种天花板。
- 理论支撑：[[LLM-037 语义空间与相似度]]（余弦为什么会在窄锥里挤平）、[[LLM-036 Embedding 与向量表示]]（静态词向量 vs 上下文向量）、[[LLM-007 参数规模与模型能力]]（124M 的 reader 能力边界）。
- 待建：`LLM-076 切块策略`、`LLM-077 向量检索与相似度`、`LLM-079 检索增强生成流程`、`LLM-083 Rerank 与多路召回`。
- 下一步（如果继续做）：把"廉价句向量"换成真 embedding 模型、加 BM25 混合检索、加 reranker、把 reader 换成指令微调模型，四项各自的增益可以单独量——**本项目的价值是给出了这四项的基线数字**。
- `PROJ-024 结构化输出与约束解码`（待建）会接在 reader 之后：即便 reader 很强，也需要约束输出格式。

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
| 技术栈 | Python、NumPy、PyTorch、Transformers |
| 项目仓库 | ai-knowledge-system/projects/PROJ-023 |

- **对应课程模块**：[M-LLM-08 RAG 检索增强生成](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-LLM-08%20RAG%20%E6%A3%80%E7%B4%A2%E5%A2%9E%E5%BC%BA%E7%94%9F%E6%88%90.md)
- **前置知识**：[LLM-076 RAG 的基本架构](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/07%20RAG%E4%B8%8EAgent/LLM-076%20RAG%20%E7%9A%84%E5%9F%BA%E6%9C%AC%E6%9E%B6%E6%9E%84.md) · [LLM-077 文档切块策略](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/07%20RAG%E4%B8%8EAgent/LLM-077%20%E6%96%87%E6%A1%A3%E5%88%87%E5%9D%97%E7%AD%96%E7%95%A5.md) · [LLM-079 向量检索与相似度索引](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/07%20RAG%E4%B8%8EAgent/LLM-079%20%E5%90%91%E9%87%8F%E6%A3%80%E7%B4%A2%E4%B8%8E%E7%9B%B8%E4%BC%BC%E5%BA%A6%E7%B4%A2%E5%BC%95.md)
- **对应知识点**：[LLM-037 语义空间与相似度](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/02%20Token%E4%B8%8EEmbedding/LLM-037%20%E8%AF%AD%E4%B9%89%E7%A9%BA%E9%97%B4%E4%B8%8E%E7%9B%B8%E4%BC%BC%E5%BA%A6.md) · [LLM-083 RAG 的评估](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/07%20RAG%E4%B8%8EAgent/LLM-083%20RAG%20%E7%9A%84%E8%AF%84%E4%BC%B0.md)

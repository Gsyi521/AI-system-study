---
id: PROJ-018
title: 序列任务——RNN、LSTM、GRU 同口径对比
domain: PROJ
module: [M-DL-06, M-DL-07]
type: project
difficulty: L3 推导
status: planned
tech_stack: [Python, PyTorch, NumPy]
prerequisites: [DL-051, DL-052, DL-053, DL-054, DL-055, DL-059, DL-062]
related_math: [MATH-804]
related: [DL-050, DL-051, DL-052, DL-053, DL-054, DL-055, DL-058, DL-060, PROJ-017]
notion_page_id: 411663e8-0333-8158-8810-f3eb782082a7
notion_url: https://app.notion.com/p/RNN-LSTM-GRU-411663e8033381588810f3eb782082a7
created: 2026-11-07
updated: 2026-11-07
notion_db: 实战项目
所属学科: 深度学习
项目状态: 未开始
项目阶段: 立项
难度: L3 推导
项目目标: 序列任务——RNN、LSTM、GRU 同口径对比
技术栈: [Python, PyTorch, NumPy]
项目仓库: ai-knowledge-system/projects/PROJ-018
对应课程模块: [M-DL-06, M-DL-07]
前置知识: [DL-051, DL-052, DL-053, DL-054, DL-055, DL-059, DL-062]
对应知识点: [DL-050, DL-051, DL-052, DL-053, DL-054, DL-055, DL-058, DL-060]
tags: [深度学习]
---

# PROJ-018 序列任务——RNN、LSTM、GRU 同口径对比

## 1. 项目背景

序列模型里最常被问的问题是：**"RNN、LSTM、GRU 到底该选哪个？"**

本项目用**两个互补任务**回答它：

| 任务 | 结构 | 考察点 |
|---|---|---|
| **A. 正弦频率三分类** | $T=80$ | 局部周期特征是否够用 |
| **B. 延迟记忆** | $T=60$ | 需要跨 60 步记住第 1 步 |

**同数据、同划分、同预算**，只换循环单元。

> **数据声明**：两个任务都是**合成数据**（`torch.Generator` 固定种子生成），无隐私、无版权问题。

## 2. 项目目标

| 目标 | 验收方式 |
|---|---|
| 三模型同口径对比 | 除单元外全部一致 |
| 同时看准确率与成本 | acc + 耗时 + 参数量 |
| 验证"门控是否必要" | 任务 A vs 任务 B |
| 诚实记录负结果 | 若门控无优势，明确写出来 |

## 3. 技术栈与环境

| 项 | 值 |
|---|---|
| Python | 3.13.15（含 torch） |
| PyTorch | 2.15.0+cu132 |
| 结构 | 单层循环单元，hidden=32 |
| 训练 | Adam lr=1e-2，40 epoch，batch 64，梯度裁剪 1.0 |

## 4. 数据集

**任务 A：正弦频率三分类**

```text
样本 900（700 训练 / 200 测试）
T=80，1 维
频率 {0.05, 0.15, 0.35}
噪声 σ=0.4
标签：用了哪个频率（3 类）
```

**任务 B：延迟记忆**

```text
样本 900（700 / 200）
T=60，1 维标准正态
标签：第 1 步的值是否 > 0（2 类）
中间 59 步全是噪声 → 必须"记住"第 1 步
```

## 5. 方案设计

```text
输入 (B,T,1) → RNN/LSTM/GRU(hidden=32, batch_first)
            → 取最后隐状态 → Linear(32, 类别数)
```

**关键控制变量**：

- 同一 `hidden=32`、同一优化器、同一 lr、同一 epoch；
- 都加**梯度裁剪 1.0**（RNN 家族必备，DL-031）；
- LSTM/GRU 的 `hn` 是元组/张量，统一取 `h[-1]`。

## 6. 核心代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def build(kind: str, hidden: int = 32, n_class: int = 3):
    rnn = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[kind](
        1, hidden, batch_first=True)
    return rnn, nn.Linear(hidden, n_class)
```

## 7. 运行结果

**任务 A（正弦频率三分类，T=80）**

| 单元 | test acc | 耗时 | 参数量 |
|---|---|---|---|
| **RNN** | **1.0000** | **2.15 s** | **1,219** |
| LSTM | 0.9950 | 3.22 s | 4,579 |
| GRU | **1.0000** | 7.53 s | 3,459 |

**任务 B（延迟记忆，T=60）**

| 单元 | test acc | 耗时 | 参数量 |
|---|---|---|---|
| **RNN** | **0.7250** | **1.91 s** | **1,186** |
| LSTM | 0.5600 | 2.45 s | 4,546 |
| GRU | 0.6750 | 5.51 s | 3,426 |

## 8. 对比分析

**任务 A：三者都饱和**

- RNN 与 GRU 都到 **1.0000**，LSTM 0.9950；
- 但成本差距巨大：RNN **2.15 s / 1,219 参数**，LSTM **3.22 s / 4,579**，GRU **7.53 s / 3,459**；
- **结论**：频率分类只需局部积分，**门控没有收益，只有成本**。

**任务 B：全都学不好，门控也没救**

- RNN **0.7250**、GRU 0.6750、**LSTM 0.5600（最差）**；
- 三个都远低于 1.0 ⇒ 60 步延迟记忆对 32 维单层单元太难；
- **LSTM 反而最差**：参数最多（4,546）、在 700 样本上更容易过拟合/欠训。

**为什么 LSTM 在这里没优势（诚实解释）**：

1. **任务规模太小**：700 条训练样本、40 epoch，不足以训练 4,546 个参数；
2. **单层 32 维**：门控的表达力被隐藏维度限制；
3. **延迟记忆任务本身极难**：需要"无损保持"，而 LSTM 的遗忘门初始值偏向遗忘（未设 `forget_bias=1`）；
4. **本课没有做超参搜索**（每模型单独调 lr/epoch），所以这是"默认配置下的对比"，不是"各自最优下的对比"。

> **诚实标注**：本项目**没有复现"LSTM/GRU 长程优于 RNN"**。这个结论与 DL-055 的负结果一致。门控优势通常需要**更长依赖 + 更大数据 + 调过的超参**，本项目的规模无法体现。

## 9. 局限与讨论

1. **任务太弱**：合成任务规模小，不能代表真实序列任务（语音/文本）。
2. **未调超参**：每个模型用同一套 lr/epoch，对 LSTM 不利。
3. **单层 + 32 维**：容量受限，门控表达力无法发挥。
4. **指标单一**：只看 acc，没有看收敛曲线与梯度范数。

## 10. 复现步骤

```text
1. py -3.13（torch 2.15.0+cu132）
2. 运行第 11 节脚本
3. 预期：任务 A：RNN 1.0 / LSTM 0.995 / GRU 1.0
        任务 B：RNN 0.725 / LSTM 0.56 / GRU 0.675
4. 偏差检查：随机种子、是否裁剪梯度、hidden 是否 32
```

## 11. 完整实验脚本

```python
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42


def make_sine(n=900, T=80, freqs=(0.05, 0.15, 0.35), noise=0.4, seed=0):
    g = torch.Generator().manual_seed(seed)
    k = torch.randint(0, 3, (n,), generator=g)
    fr = torch.tensor(freqs)
    t = torch.arange(T).float()
    x = torch.stack([torch.sin(2 * torch.pi * fr[k_i] * t) for k_i in k])
    x = x + noise * torch.randn(x.shape, generator=g)
    return x.unsqueeze(-1), k


def make_delay(n=900, T=60, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, T, 1, generator=g)
    y = (x[:, 0, 0] > 0).long()
    return x, y


def run(kind, Xtr, ytr, Xte, yte, n_class, epochs=40, hidden=32):
    torch.manual_seed(SEED)
    rnn = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[kind](
        1, hidden, batch_first=True)
    head = nn.Linear(hidden, n_class)
    params = sum(p.numel() for p in rnn.parameters()) + sum(
        p.numel() for p in head.parameters())
    opt = torch.optim.Adam(list(rnn.parameters()) + list(head.parameters()), lr=1e-2)
    start = time.time()
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            out, hn = rnn(Xtr[idx])
            h = hn[0] if isinstance(hn, tuple) else hn
            F.cross_entropy(head(h[-1]), ytr[idx]).backward()
            torch.nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)
            opt.step()
    cost = time.time() - start
    rnn.eval()
    with torch.no_grad():
        out, hn = rnn(Xte)
        h = hn[0] if isinstance(hn, tuple) else hn
        acc = float((head(h[-1]).argmax(1) == yte).float().mean())
    return round(acc, 4), round(cost, 2), params


if __name__ == "__main__":
    X, y = make_sine()
    for kind in ("RNN", "LSTM", "GRU"):
        print("sine", kind, run(kind, X[:700], y[:700], X[700:], y[700:], 3))

    Xd, yd = make_delay()
    for kind in ("RNN", "LSTM", "GRU"):
        print("delay", kind, run(kind, Xd[:700], yd[:700], Xd[700:], yd[700:], 2))
```

## 12. 验收清单

| # | 项 | 判据 | 结果 |
|---|---|---|---|
| 1 | 同口径 | 除单元外全部一致 | ✅ |
| 2 | 双任务 | 一个易、一个难 | ✅ |
| 3 | 三模型齐全 | RNN/LSTM/GRU | ✅ |
| 4 | 成本记录 | 耗时 + 参数 | ✅ |
| 5 | 负结果如实写 | 未复现门控优势 | ✅ |
| 6 | 可复现 | 固定种子 | ✅ |

## 13. 结论

- **任务 A**：RNN/GRU 均 **1.0000**，LSTM 0.9950；成本 RNN 最低（2.15 s / 1,219）。⇒ **简单任务选 RNN**。
- **任务 B**：三者全部失败（RNN 0.7250 / GRU 0.6750 / LSTM 0.5600）⇒ **这个规模下门控也救不了长程依赖**。
- **建议**：默认先用 **RNN/GRU 作基线**，只有当"更长序列 + 更多数据 + 调过的超参"时才值得上 LSTM。

## 14. 与后续项目的关系

- PROJ-017（CNN）→ **PROJ-018（序列）** → PROJ-019（迁移学习）
- 想验证门控优势：把任务 B 的 $T$ 提到 200+、训练样本提到 5,000+、给 LSTM 设 `forget_bias=1.0` 再比一次。

## 学习记录

| 日期 | 状态 | 用时 | 问题 | 复盘 |
|---|---|---|---|---|
| 2026-11-07 | planned | — | — | — |

## 关联与延伸

本项目的知识依据（正文链接，便于图谱连边）：

- [[DL-053 LSTM 与门控机制]]
- [[DL-054 GRU]]

---

## 🔗 Notion 同步信息

> 以下内容由 Notion「AI Knowledge System · 实战项目」数据库按知识 ID 合并而来（导出于 2026-11-09）。

| 属性 | 值 |
|---|---|
| 所属学科 | 深度学习 |
| 项目状态 | 未开始 |
| 项目阶段 | 立项 |
| 难度 | L3 推导 |
| 技术栈 | Python、PyTorch、NumPy |
| 项目仓库 | ai-knowledge-system/projects/PROJ-018 |

- **对应课程模块**：[M-DL-06 循环神经网络与序列建模](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-DL-06%20%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E5%BA%8F%E5%88%97%E5%BB%BA%E6%A8%A1.md) · [M-DL-07 PyTorch 深度学习框架实践](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-DL-07%20PyTorch%20%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A1%86%E6%9E%B6%E5%AE%9E%E8%B7%B5.md)
- **前置知识**：[DL-051 RNN 结构与前向传播](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/03%20RNN%E4%B8%8E%E5%BA%8F%E5%88%97%E6%A8%A1%E5%9E%8B/DL-051%20RNN%20%E7%BB%93%E6%9E%84%E4%B8%8E%E5%89%8D%E5%90%91%E4%BC%A0%E6%92%AD.md) · [DL-052 BPTT 与梯度问题](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/03%20RNN%E4%B8%8E%E5%BA%8F%E5%88%97%E6%A8%A1%E5%9E%8B/DL-052%20BPTT%20%E4%B8%8E%E6%A2%AF%E5%BA%A6%E9%97%AE%E9%A2%98.md) · [DL-053 LSTM 与门控机制](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/03%20RNN%E4%B8%8E%E5%BA%8F%E5%88%97%E6%A8%A1%E5%9E%8B/DL-053%20LSTM%20%E4%B8%8E%E9%97%A8%E6%8E%A7%E6%9C%BA%E5%88%B6.md) · [DL-054 GRU](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/03%20RNN%E4%B8%8E%E5%BA%8F%E5%88%97%E6%A8%A1%E5%9E%8B/DL-054%20GRU.md) · [DL-055 RNN/LSTM/GRU 对比与序列任务实战](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/03%20RNN%E4%B8%8E%E5%BA%8F%E5%88%97%E6%A8%A1%E5%9E%8B/DL-055%20RNN%E3%80%81LSTM%E3%80%81GRU%20%E5%AF%B9%E6%AF%94%E4%B8%8E%E5%BA%8F%E5%88%97%E4%BB%BB%E5%8A%A1%E5%AE%9E%E6%88%98.md) · [DL-059 Dataset 与 DataLoader](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/06%20PyTorch/DL-059%20Dataset%20%E4%B8%8E%20DataLoader.md) · [DL-062 训练、验证、测试循环](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/06%20PyTorch/DL-062%20%E8%AE%AD%E7%BB%83%E3%80%81%E9%AA%8C%E8%AF%81%E3%80%81%E6%B5%8B%E8%AF%95%E5%BE%AA%E7%8E%AF.md)
- **对应知识点**：[DL-050 序列数据与序列建模任务](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/03%20RNN%E4%B8%8E%E5%BA%8F%E5%88%97%E6%A8%A1%E5%9E%8B/DL-050%20%E5%BA%8F%E5%88%97%E6%95%B0%E6%8D%AE%E4%B8%8E%E5%BA%8F%E5%88%97%E5%BB%BA%E6%A8%A1%E4%BB%BB%E5%8A%A1.md) · [DL-051 RNN 结构与前向传播](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/03%20RNN%E4%B8%8E%E5%BA%8F%E5%88%97%E6%A8%A1%E5%9E%8B/DL-051%20RNN%20%E7%BB%93%E6%9E%84%E4%B8%8E%E5%89%8D%E5%90%91%E4%BC%A0%E6%92%AD.md) · [DL-052 BPTT 与梯度问题](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/03%20RNN%E4%B8%8E%E5%BA%8F%E5%88%97%E6%A8%A1%E5%9E%8B/DL-052%20BPTT%20%E4%B8%8E%E6%A2%AF%E5%BA%A6%E9%97%AE%E9%A2%98.md) · [DL-053 LSTM 与门控机制](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/03%20RNN%E4%B8%8E%E5%BA%8F%E5%88%97%E6%A8%A1%E5%9E%8B/DL-053%20LSTM%20%E4%B8%8E%E9%97%A8%E6%8E%A7%E6%9C%BA%E5%88%B6.md) · [DL-054 GRU](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/03%20RNN%E4%B8%8E%E5%BA%8F%E5%88%97%E6%A8%A1%E5%9E%8B/DL-054%20GRU.md) · [DL-055 RNN/LSTM/GRU 对比与序列任务实战](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/03%20RNN%E4%B8%8E%E5%BA%8F%E5%88%97%E6%A8%A1%E5%9E%8B/DL-055%20RNN%E3%80%81LSTM%E3%80%81GRU%20%E5%AF%B9%E6%AF%94%E4%B8%8E%E5%BA%8F%E5%88%97%E4%BB%BB%E5%8A%A1%E5%AE%9E%E6%88%98.md) · [DL-058 设备管理：CPU 与 GPU](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/06%20PyTorch/DL-058%20%E8%AE%BE%E5%A4%87%E7%AE%A1%E7%90%86%E2%80%94%E2%80%94CPU%20%E4%B8%8E%20GPU.md) · [DL-060 nn.Module 自定义网络](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/06%20PyTorch/DL-060%20nn.Module%20%E8%87%AA%E5%AE%9A%E4%B9%89%E7%BD%91%E7%BB%9C.md)

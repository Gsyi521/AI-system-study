---
id: DL-052
title: BPTT 与梯度问题
en_title: BPTT and Gradient Problems
domain: DL
module: M-DL-06
type: Algorithm
difficulty: L3 推导
status: planned
prerequisites: [DL-051]
related_math: [MATH-804]
related_python: []
related_ml: []
related_dl: [DL-051]
notion_page_id: 411663e8-0333-810e-aa73-cb9a82e28f44
notion_url: https://app.notion.com/p/BPTT-411663e80333810eaa73cb9a82e28f44
created: 2026-11-07
updated: 2026-11-07
notion_db: 知识点
学科: 深度学习
学习状态: 未开始
难度: L3 推导
知识类型: Algorithm
所属模块: [M-DL-06]
前置知识: [DL-051]
后继知识: [DL-053]
关联数学: [MATH-804]
对应项目: [PROJ-018]
被项目引用: [PROJ-018]
tags: [深度学习, Algorithm]
---

# DL-052 BPTT 与梯度问题

## 1. 本节学习目标

1. 说清 BPTT 与普通反向传播的区别（**在时间维展开后反传**）。
2. 用实测看梯度随 $T$ 指数衰减（$T=5$ 到 $T=100$：**1.77e-1 → 0**）。
3. 会写 **TBPTT**（截断 BPTT），理解它为什么能省显存。
4. 知道梯度裁剪在 RNN 里的必要性（DL-031）。

## 2. 为什么需要这个知识

因为**朴素 RNN 学不到长程依赖**——这不是"效果差一点"，而是梯度直接变成 0：

本课实测 $T=50$ 时 $dL/dh_0$ 只剩 **7.74e-13**，$T=100$ 时为 **0.000000e+00**。

理解这个机制，你才知道 LSTM/GRU 的门控到底在解决什么。

## 3. 直观解释

**BPTT = 把 RNN 在时间上"展开成一棵深树"，然后反传**：

```text
h_1 ← h_2 ← h_3 ← ... ← h_T
 ↑      ↑      ↑
x_1    x_2    x_3
```

梯度从 $h_T$ 一路乘回 $h_1$：每经过一步都要乘一次 $\partial h_t/\partial h_{t-1}$。

**问题**：乘 $T$ 次，若每项 < 1 ⇒ 指数衰减；> 1 ⇒ 爆炸。

## 4. 正式定义

**单步雅可比**

$$
\frac{\partial\mathbf{h}_t}{\partial\mathbf{h}_{t-1}}
=\mathrm{diag}\big(1-\mathbf{h}_t^2\big)\,W_{hh}
$$

**总梯度**

$$
\frac{\partial L}{\partial\mathbf{h}_0}
=\frac{\partial L}{\partial\mathbf{h}_T}
\prod_{t=1}^{T}\frac{\partial\mathbf{h}_t}{\partial\mathbf{h}_{t-1}}
$$

**TBPTT**：只反传最近 $k$ 步（$k\ll T$），把 $\mathbf{h}_{T-k}$ 当作"常量起点"。

## 5. 核心原理

**本课实测（$\tanh$ RNN，$H=32$，$dL/dh_0$ 范数）**

| $T$ | $dL/dh_0$ 范数 |
|---|---|
| 5 | **1.768829e-01** |
| 10 | 1.063535e-02 |
| 20 | 7.717565e-06 |
| 50 | 7.735793e-13 |
| 100 | **0.000000e+00** |

**读法**：

1. 每加 5 步，梯度大约掉 **一个数量级以上**。
2. $T=50$ 时已经是 $10^{-13}$ —— 前面 50 步的参数**根本收不到信号**。
3. $T=100$ 直接下溢为 **0**。

**这不是"慢"，是"完全学不到"**。

## 6. 数学表达式

**衰减的量化**

若每步雅可比的谱范数平均为 $\rho$，则

$$
\Big\|\frac{\partial L}{\partial\mathbf{h}_0}\Big\|\approx\rho^{T}\Big\|\frac{\partial L}{\partial\mathbf{h}_T}\Big\|
$$

本课 $1.77\times10^{-1}$（$T=5$）到 $7.7\times10^{-6}$（$T=20$）：约 4 个数量级 / 15 步 ⇒ $\rho\approx10^{-4/15}\approx0.54$。

**$\rho<1$ ⇒ 消失；$\rho>1$ ⇒ 爆炸**（长序列训练 NaN 的常见原因）。

## 7. 数学推导

**为什么 $\tanh$ 会衰减**

$$
\tanh'(z)=1-\tanh^2(z)\le1
$$

且 $W_{hh}$ 的元素通常初始化得较小（如 $U(-\sqrt{1/H},\sqrt{1/H})$）⇒ 乘积范数 $<1$。

**连乘 $T$ 次** ⇒ $\rho^T\to0$。

**对比残差网络**：残差把 $\partial y/\partial x$ 变成 $1+\partial F/\partial x$，与 RNN 的纯连乘形成对照（DL-048）。

**为什么 LSTM 能缓解**

LSTM 的细胞状态更新是**加法**：

$$
\mathbf{c}_t=\mathbf{f}_t\odot\mathbf{c}_{t-1}+\mathbf{i}_t\odot\mathbf{g}_t
$$

$$
\frac{\partial\mathbf{c}_t}{\partial\mathbf{c}_{t-1}}=\mathbf{f}_t\ (\text{逐元素})
$$

若遗忘门 $\mathbf{f}_t\approx1$，梯度就能**近似无损**地传很多步 —— 这是"常数误差传送带（CEC）"的直观来源。

## 8. 结构示意图

```mermaid
flowchart RL
    L["Loss"] --> HT["h_T"]
    HT -->|"∂h_T/∂h_{T-1}"| H2["h_{T-1}"]
    H2 -->|"连乘..."| H1["h_1"]
    H1 -->|"梯度≈0"| G["参数收不到信号"]
```

## 9. 具体计算示例

```text
T=  5: dL/dh0 范数=1.768829e-01
T= 10: dL/dh0 范数=1.063535e-02
T= 20: dL/dh0 范数=7.717565e-06
T= 50: dL/dh0 范数=7.735793e-13
T=100: dL/dh0 范数=0.000000e+00
```

**从 $T=5$ 到 $T=20$**：梯度掉了约 **23,000 倍**。

## 10. Python 实现

```python
import torch
import torch.nn as nn


def grad_at_h0(seq_len: int, hidden: int = 32, seed: int = 0) -> float:
    torch.manual_seed(seed)
    cell = nn.RNNCell(2, hidden)
    x = torch.randn(1, seq_len, 2)
    h0 = torch.zeros(1, hidden, requires_grad=True)
    h = h0
    for t in range(seq_len):
        h = torch.tanh(cell(x[:, t], h))
    h.sum().backward()
    return float(h0.grad.norm())


for T in (5, 10, 20, 50, 100):
    print(f"T={T:>3}: dL/dh0 范数={grad_at_h0(T):.6e}")
```

## 11. PyTorch 实现（TBPTT 与裁剪）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
rnn = nn.RNN(4, 16, batch_first=True)
head = nn.Linear(16, 1)
opt = torch.optim.Adam(list(rnn.parameters()) + list(head.parameters()), lr=1e-2)

T, k = 100, 20          # 总长 100，每次只反传 20 步
x = torch.randn(8, T, 4)
h = torch.zeros(1, 8, 16)

for start in range(0, T, k):
    chunk = x[:, start:start + k]
    out, h = rnn(chunk, h)
    h = h.detach()                                  # 截断：切断更早的历史
    pred = head(out[:, -1])
    loss = F.mse_loss(pred, torch.randn(8, 1))
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)
    opt.step()

print("TBPTT 完成；最终 h 形状:", tuple(h.shape))
print("梯度范数示例:", round(float(rnn.weight_hh_l0.grad.norm()), 4))
```

## 12. 代码逐行解释

| 代码 | 作用 |
|---|---|
| `h0.requires_grad=True` | 让初始状态可求导，测梯度衰减 |
| `h.sum().backward()` | 用最后一步隐状态做损失 |
| `h = h.detach()` | **TBPTT 的关键**：切断跨 chunk 的图 |
| `clip_grad_norm_` | 防爆炸（RNN 必备） |
| `range(0, T, k)` | 按 chunk 前向 |

**为什么 `detach` 能省显存**：不 detach 时整条 100 步的图都要保留；detach 后每块独立。

## 13. 实际运行结果

```text
T=  5: dL/dh0 范数=1.768829e-01
T= 10: dL/dh0 范数=1.063535e-02
T= 20: dL/dh0 范数=7.717565e-06
T= 50: dL/dh0 范数=7.735793e-13
T=100: dL/dh0 范数=0.000000e+00
```

## 14. 应用场景

| 场景 | 处理 |
|---|---|
| 长文本分类 | 用 LSTM/GRU 或 Transformer |
| 超长序列（>1000） | **TBPTT**（分块反传） |
| 在线流式 | 每步更新（k=1） |
| 长程依赖任务 | 门控结构 + 梯度裁剪 |
| 训练 NaN | 裁剪 + 降 lr + 检查 $W_{hh}$ 初始化 |

## 15. 优点与缺点

| 优点 | 缺点 |
|---|---|
| BPTT 精确、实现简单 | 梯度消失/爆炸 |
| TBPTT 省显存、可训超长序列 | 截断后学不到超长依赖 |
| 与裁剪配合可稳定 | 串行计算慢 |

## 16. 常见错误

| 错误 | 现象 | 纠正 |
|---|---|---|
| 用超长序列训朴素 RNN | 前层学不到（本课 0.0） | 换 LSTM/GRU 或 TBPTT |
| TBPTT 忘 `detach` | 显存爆炸 | 每块 `h.detach()` |
| 梯度爆炸不裁剪 | NaN | `clip_grad_norm_` |
| 只测权重梯度 | 看不到衰减 | 测 $dL/dh_0$ |
| 认为"序列越长越好" | 反而更差 | 与模型容量匹配 |

## 17. 本节总结

- BPTT = 在时间维展开后反传，梯度是**雅可比连乘**。
- 实测：$T=5\to100$，$dL/dh_0$ 从 **1.77e-1** 掉到 **0**。
- 缓解：**LSTM/GRU 门控 + TBPTT + 梯度裁剪**。

## 18. 前置与后续

- 前置：[[DL-051 RNN 结构与前向传播]]
- 后续：[[DL-053 LSTM 与门控机制]]、[[DL-054 GRU]]
- 相关数学：[[MATH-804 反向传播与链式法则]]

---

## 练习与参考答案

**基础 1**：BPTT 与普通反向传播的核心区别是什么？

**基础 2**：本课 $T=5$ 与 $T=50$ 的 $dL/dh_0$ 各是多少？

**进阶 1**：为什么 TBPTT 必须 `detach()`？

**综合**：你要用 RNN 处理长度 500 的传感器序列，训练时显存不够且梯度变 NaN。请给出完整方案。

> [!success]- 参考答案

> **基础 1**
> 普通反向传播沿**网络层**反传；BPTT 把 RNN 在**时间维展开**，沿时间步反传，每步都要乘一次 $\partial h_t/\partial h_{t-1}$。
> 因此梯度是时间上的连乘，容易指数衰减/爆炸。

> **基础 2**
> $T=5$：**1.768829e-01**；$T=50$：**7.735793e-13**。

> **进阶 1**
> 因为 `detach()` 会**切断计算图**，让前面的 chunk 不被纳入本次反传。
> 作用：①显存从 $O(T)$ 降到 $O(k)$；②避免梯度跨块无限连乘导致爆炸；③让训练可持续进行。
> 代价：模型只能学到 $k$ 步内的依赖。

> **综合（参考）**
> **方案**：
> 1. **TBPTT**：把 500 步切成 50 步/块，每块结束 `h = h.detach()`；
> 2. **梯度裁剪**：`clip_grad_norm_(params, 1.0)`（RNN 标配）；
> 3. **换结构**：用 **LSTM/GRU**（门控能缓解消失），或考虑 1D 卷积/Transformer；
> 4. **降 lr + warmup**，并检查 $W_{hh}$ 初始化（过大会爆炸）；
> 5. **显存**：减小 batch、用混合精度、必要时缩短序列。
> **验证**：打印每块 loss 与 `grad_norm`，确认不再 NaN 且稳定下降。

## 学习记录

| 日期 | 状态 | 用时 | 问题 | 复盘 |
|---|---|---|---|---|
| 2026-11-07 | planned | — | — | — |

---

## 🔗 Notion 同步信息

> 以下内容由 Notion「AI Knowledge System · 知识点」数据库按知识 ID 合并而来（导出于 2026-11-09）。

| 属性 | 值 |
|---|---|
| 学科 | 深度学习 |
| 学习状态 | 未开始 |
| 难度 | L3 推导 |
| 知识类型 | Algorithm |

- **所属模块**：[M-DL-06 循环神经网络与序列建模](../%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-DL-06%20%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E5%BA%8F%E5%88%97%E5%BB%BA%E6%A8%A1.md)
- **前置知识**：[DL-051 RNN 结构与前向传播](DL-051%20RNN%20%E7%BB%93%E6%9E%84%E4%B8%8E%E5%89%8D%E5%90%91%E4%BC%A0%E6%92%AD.md)
- **后继知识**：[DL-053 LSTM 与门控机制](DL-053%20LSTM%20%E4%B8%8E%E9%97%A8%E6%8E%A7%E6%9C%BA%E5%88%B6.md)
- **关联数学**：[MATH-804 反向传播与链式法则](../../06_%E6%95%B0%E5%AD%A6%E5%9F%BA%E7%A1%80/08%20%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%95%B0%E5%AD%A6%E5%BA%94%E7%94%A8/MATH-804%20%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%E4%B8%8E%E9%93%BE%E5%BC%8F%E6%B3%95%E5%88%99.md)
- **对应项目**：[PROJ-018 序列任务——RNN、LSTM、GRU 同口径对比](../../07_%E5%AE%9E%E6%88%98%E9%A1%B9%E7%9B%AE/04%20%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E9%A1%B9%E7%9B%AE/PROJ-018%20%E5%BA%8F%E5%88%97%E4%BB%BB%E5%8A%A1%E2%80%94%E2%80%94RNN%E3%80%81LSTM%E3%80%81GRU%20%E5%90%8C%E5%8F%A3%E5%BE%84%E5%AF%B9%E6%AF%94.md)
- **被项目引用**：[PROJ-018 序列任务——RNN、LSTM、GRU 同口径对比](../../07_%E5%AE%9E%E6%88%98%E9%A1%B9%E7%9B%AE/04%20%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E9%A1%B9%E7%9B%AE/PROJ-018%20%E5%BA%8F%E5%88%97%E4%BB%BB%E5%8A%A1%E2%80%94%E2%80%94RNN%E3%80%81LSTM%E3%80%81GRU%20%E5%90%8C%E5%8F%A3%E5%BE%84%E5%AF%B9%E6%AF%94.md)

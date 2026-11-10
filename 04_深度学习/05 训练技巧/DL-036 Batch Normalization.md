---
id: DL-036
title: Batch Normalization
en_title: Batch Normalization
domain: DL
module: M-DL-04
type: Technique
difficulty: L3 推导
status: planned
prerequisites: [DL-035]
related_math: [MATH-306]
related_python: []
related_ml: []
related_dl: [DL-035]
notion_page_id: 411663e8-0333-81a9-8db3-e9066242f9c5
notion_url: https://app.notion.com/p/Batch-Normalization-411663e8033381a98db3e9066242f9c5
created: 2026-11-07
updated: 2026-11-07
notion_db: 知识点
学科: 深度学习
学习状态: 未开始
难度: L3 推导
知识类型: Technique
所属模块: [M-DL-04]
前置知识: [DL-035]
后继知识: [DL-037]
关联数学: [MATH-306]
对应项目: [PROJ-019]
tags: [深度学习, Technique]
---

# DL-036 Batch Normalization

## 1. 本节学习目标

1. 写出 BN 的归一化 + 仿射变换公式，理解 $\gamma,\beta$ 的作用。
2. 用**深 8 层实测**看清 BN 的威力（无 BN：全 **0.5**；有 BN：**0.9417**）。
3. 说清训练/推理差异（batch 统计量 vs running stats）。
4. 知道 BN 的适用条件与局限（小 batch 不稳）。

## 2. 为什么需要这个知识

因为**深层网络训不动的头号原因之一就是"内部协变量偏移"**：每层输入分布随前层参数变化而漂移。

BN 把每层输入拉回稳定分布，让深层网络能用**更大的学习率**、**更快收敛**。本课的 8 层实验就是最直接的证据：**不装 BN 完全学不动**。

## 3. 直观解释

**BN 做的事**：对每个 batch 的每个特征维度做标准化，再用可学习的 $\gamma,\beta$ 缩放平移。

```text
输入 → 减均值、除标准差 → 乘 γ、加 β → 输出
```

**为什么有效**：

- 每层输入尺度稳定 ⇒ 梯度不会因尺度漂移而爆炸/消失。
- 允许更大 lr ⇒ 收敛更快。
- 轻微正则效果（batch 统计量带来噪声）。

## 4. 正式定义

**对一个 mini-batch $B$、某维度：**

$$
\mu_B=\frac1{|B|}\sum_{i\in B}x_i,\qquad
\sigma_B^2=\frac1{|B|}\sum_{i\in B}(x_i-\mu_B)^2
$$

$$
\hat x_i=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}},\qquad
y_i=\gamma\hat x_i+\beta
$$

**推理时**：用**训练累积的 running mean/var**，不用当前 batch 的统计量。

**PyTorch**

```text
nn.BatchNorm1d(num_features)
```

## 5. 核心原理

**本课实测 1：深 8 层网络（lr=0.1，40 epoch，moons）**

| epoch | 无 BN | **有 BN** |
|---|---|---|
| 1 | 0.5000 | **0.7375** |
| 5 | 0.5000 | **0.9417** |
| 10 | 0.5000 | 0.9375 |
| 20 | 0.5000 | 0.9292 |
| 40 | **0.5000** | **0.9250** |

**这是本课最有说服力的一组数据**：8 层 MLP 不装 BN **完全学不动**（恒 0.5），装上后 5 个 epoch 就到 **0.9417**。

**本课实测 2：浅层网络（2→64→1）**

| epoch | 无 BN | 有 BN |
|---|---|---|
| 1 | 0.9083 | 0.9125 |
| 5 | **0.9333** | **0.9333** |
| 10 | 0.9083 | 0.9125 |
| 50 | 0.9250 | **0.9417** |
| 200 | 0.9250 | 0.9292 |

**读法**：浅层网络差异很小（BN 略好），**BN 的价值随深度增加才显现**。

## 6. 数学表达式

**BN 的仿射参数**

$$
\frac{\partial L}{\partial\gamma}=\sum_i \frac{\partial L}{\partial y_i}\hat x_i,\qquad
\frac{\partial L}{\partial\beta}=\sum_i\frac{\partial L}{\partial y_i}
$$

**为什么要 $\gamma,\beta$**：只有标准化会限制表示能力（强制零均值单位方差）；加仿射让网络**可以学回**需要的分布（$\gamma=1,\beta=0$ 是恒等）。

## 7. 数学推导

**训练 vs 推理的统计量**

- 训练：$\mu_B,\sigma_B^2$ 来自当前 batch，同时用滑动平均更新 running stats：

$$
\mu_{\text{run}}\leftarrow(1-m)\mu_{\text{run}}+m\mu_B
$$

- 推理：直接用 $\mu_{\text{run}},\sigma_{\text{run}}^2$。

**本课实测 running stats**（在 32 条随机数据上跑 10 次 train）：

```text
running_mean 前3: [-0.3076, -0.2551, -0.3517]
running_var  前3: [0.4508, 0.6993, 0.4441]
```

**推理一致性**：`eval` 两次输出**完全一致**（`True`），因为不再依赖当前 batch。

**train 与 eval 输出最大差 0.3735** —— 这个差异是**正常的**：训练用 batch 统计量，推理用 running stats。

## 8. 结构示意图

```mermaid
flowchart LR
    X["Linear 输出"] --> N["减均值/除标准差"]
    N --> A["×γ + β"]
    A --> R["ReLU"]
    R --> Y["下一层"]
```

**常见位置**：`Linear → BN → ReLU`（BN 在激活**前**）。

## 9. 具体计算示例

**手算一个 batch 的一维 BN**

设某维 $x=[1,2,3,4]$：

$$
\mu=2.5,\quad \sigma^2=\frac{(1.5)^2+(0.5)^2+(0.5)^2+(1.5)^2}{4}=1.25
$$

$$
\hat x=\frac{x-2.5}{\sqrt{1.25}}\approx[-1.342,\ -0.447,\ 0.447,\ 1.342]
$$

若 $\gamma=1,\beta=0$，输出即上式（均值 0、方差 1）。

## 10. Python 实现

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
bn = nn.BatchNorm1d(4)
x = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                  [2.0, 3.0, 4.0, 5.0],
                  [3.0, 4.0, 5.0, 6.0],
                  [4.0, 5.0, 6.0, 7.0]])

bn.train()
y = bn(x)
print("训练输出均值:", [round(float(v), 4) for v in y.mean(dim=0)])
print("训练输出标准差:", [round(float(v), 4) for v in y.std(dim=0, unbiased=False)])
print("running_mean:", [round(float(v), 4) for v in bn.running_mean])

bn.eval()
y1, y2 = bn(x), bn(x)
print("eval 两次一致:", bool(torch.allclose(y1, y2)))
```

## 11. PyTorch 实现（深网对比）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=800, noise=0.30, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                      random_state=42, stratify=y)
Xtr = torch.tensor(Xtr, dtype=torch.float32)
ytr = torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
Xte = torch.tensor(Xte, dtype=torch.float32)
yte = torch.tensor(yte, dtype=torch.float32).view(-1, 1)


def build(depth: int, use_bn: bool) -> nn.Sequential:
    layers, dim = [], 2
    for _ in range(depth):
        layers.append(nn.Linear(dim, 64))
        if use_bn:
            layers.append(nn.BatchNorm1d(64))
        layers += [nn.ReLU()]
        dim = 64
    layers.append(nn.Linear(dim, 1))
    return nn.Sequential(*layers)


def evaluate(m):
    m.eval()
    with torch.no_grad():
        acc = float((torch.sigmoid(m(Xte)) >= 0.5).float().eq(yte).float().mean())
    m.train()
    return acc


for use_bn in (False, True):
    torch.manual_seed(0)
    model = build(8, use_bn)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    marks = {}
    for epoch in range(1, 41):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 32):
            idx = perm[i:i + 32]
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        if epoch in (1, 5, 10, 20, 40):
            marks[epoch] = round(evaluate(model), 4)
    print(f"BN={use_bn}: " + " ".join(f"ep{k}={v}" for k, v in marks.items()))
```

## 12. 代码逐行解释

| 代码 | 作用 |
|---|---|
| `nn.BatchNorm1d(64)` | 对 64 维特征逐维归一化 |
| `bn.train()` / `bn.eval()` | 切换 batch 统计量 / running stats |
| `running_mean/var` | 训练累积的统计量（推理用） |
| `unbiased=False` | 用总体标准差（与 BN 公式一致） |
| `build(depth, use_bn)` | 控制变量法对比 |

**顺序**：`Linear → BatchNorm → ReLU` 是常见套路（BN 在激活前）。

## 13. 实际运行结果

```text
深 8 层, lr=0.1, 40 epoch
BN=False: ep1=0.5 ep5=0.5 ep10=0.5 ep20=0.5 ep40=0.5
BN=True : ep1=0.7375 ep5=0.9417 ep10=0.9375 ep20=0.9292 ep40=0.925

浅层 2→64→1
BN=False: ep1=0.9083 ep5=0.9333 ep10=0.9083 ep50=0.925 ep200=0.925
BN=True : ep1=0.9125 ep5=0.9333 ep10=0.9125 ep50=0.9417 ep200=0.9292

BN running stats 示例:
running_mean 前3: [-0.3076, -0.2551, -0.3517]
running_var  前3: [0.4508, 0.6993, 0.4441]
eval 两次输出一致: True
train 与 eval 输出最大差: 0.3735
```

## 14. 应用场景

| 场景 | 建议 |
|---|---|
| 深层 CNN（ResNet 等） | ✅ 标配 |
| 深层 MLP | ✅ 明显改善 |
| 小 batch（<8） | ⚠️ 统计量不稳，改 GroupNorm/LayerNorm |
| RNN/Transformer | 用 LayerNorm 而非 BN（DL-037） |
| 推理延迟敏感 | BN 可与卷积融合（fold） |

## 15. 优点与缺点

| 优点 | 缺点 |
|---|---|
| 允许更大 lr、收敛更快 | 小 batch 不稳定 |
| 缓解梯度消失（本课 8 层从 0.5→0.94） | 有轻微正则副作用，需调 |
| 对初始化不敏感 | 训练/推理行为不同，易踩坑 |

## 16. 常见错误

| 错误 | 现象 | 纠正 |
|---|---|---|
| 评估忘 `eval()` | 指标随 batch 组成波动 | 切 eval |
| batch 太小（1–4） | 统计量失真 | 加大 batch 或换 LN |
| BN 放在 ReLU 后 | 效果变差 | 常见是 BN 在激活前 |
| 只在输出层加 BN | 无意义 | 每层隐层加 |
| 与 Dropout 顺序混乱 | 训练不稳 | `Linear→BN→ReLU→Dropout` |
| 以为 BN 能替代初始化 | 仍需合理初始化 | 两者互补 |

## 17. 本节总结

- BN = **逐维标准化 + 可学习仿射**；训练用 batch 统计量，推理用 running stats。
- **深 8 层实测：无 BN 恒 0.5，有 BN 第 5 轮就到 0.9417**。
- 浅层网络收益小（0.925 → 0.9292），**BN 的价值随深度增加**。

## 18. 前置与后续

- 前置：[[DL-035 Dropout]]
- 后续：`DL-037 Layer Normalization 与 BN 的区别`（待建）、`DL-038 Early Stopping 与模型检查点`（待建）
- 相关数学：[[MATH-306 期望、方差与协方差]]

---

## 练习与参考答案

**基础 1**：写出 BN 的归一化与仿射公式，并说明 $\gamma,\beta$ 的作用。

**基础 2**：本课深 8 层「无 BN / 有 BN」在第 5 轮与第 40 轮各是多少？

**进阶 1**：为什么 BN 在推理时不用当前 batch 的均值方差？

**综合**：你训练一个 batch=4 的分割网络，加了 BN 后 loss 剧烈震荡。请诊断并给出 2 个修正方案。

> [!success]- 参考答案

> **基础 1**
> $$\hat x=\frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}},\qquad y=\gamma\hat x+\beta$$
> $\gamma$（缩放）与 $\beta$（平移）是**可学习参数**，让网络能恢复需要的分布——若只有标准化会强制零均值单位方差，限制表示能力。

> **基础 2**
> 第 5 轮：无 BN **0.5000**、有 BN **0.9417**；第 40 轮：无 BN **0.5000**、有 BN **0.9250**。

> **进阶 1**
> 因为推理时样本可能逐条到来（batch=1），**没有"batch 统计量"可用**；而且推理要**确定性**（同一输入必须同输出）。
> 用训练累积的 running stats 才能保证：①不依赖 batch 组成；②同一输入结果固定（实测 eval 两次完全一致）。

> **综合（参考）**
> **诊断**：batch=4 太小 ⇒ 每个 batch 的均值/方差估计**噪声极大** ⇒ 归一化后的值抖动 ⇒ loss 震荡。
> **两个修正**：
> 1. **换归一化**：用 `GroupNorm` 或 `LayerNorm`（不依赖 batch 维度）；
> 2. **增大有效 batch**：用**梯度累积**（累积 8 个 micro-batch 再 step），或降低分辨率/裁剪输入以放大 batch。
> **辅助**：训练初期用 `momentum=0.1` 的 running stats 更新不要太快；必要时冻结 BN 统计量（`track_running_stats=False` 并手动处理）。

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
| 知识类型 | Technique |

- **所属模块**：[M-DL-04 模型泛化、正则化与训练技巧](../%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-DL-04%20%E6%A8%A1%E5%9E%8B%E6%B3%9B%E5%8C%96%E3%80%81%E6%AD%A3%E5%88%99%E5%8C%96%E4%B8%8E%E8%AE%AD%E7%BB%83%E6%8A%80%E5%B7%A7.md)
- **前置知识**：[DL-035 Dropout](DL-035%20Dropout.md)
- **后继知识**：[DL-037 Layer Normalization 与 BN 的区别](DL-037%20Layer%20Normalization%20%E4%B8%8E%20BN%20%E7%9A%84%E5%8C%BA%E5%88%AB.md)
- **关联数学**：[MATH-306 期望、方差与协方差](../../06_%E6%95%B0%E5%AD%A6%E5%9F%BA%E7%A1%80/03%20%E6%A6%82%E7%8E%87%E8%AE%BA/MATH-306%20%E6%9C%9F%E6%9C%9B%E3%80%81%E6%96%B9%E5%B7%AE%E4%B8%8E%E5%8D%8F%E6%96%B9%E5%B7%AE.md)
- **对应项目**：[PROJ-019 迁移学习——小样本分类](../../07_%E5%AE%9E%E6%88%98%E9%A1%B9%E7%9B%AE/04%20%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E9%A1%B9%E7%9B%AE/PROJ-019%20%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0%E2%80%94%E2%80%94%E5%B0%8F%E6%A0%B7%E6%9C%AC%E5%88%86%E7%B1%BB.md)

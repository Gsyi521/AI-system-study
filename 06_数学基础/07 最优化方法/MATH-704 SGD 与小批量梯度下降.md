---
id: MATH-704
title: SGD 与小批量梯度下降
en_title: Stochastic and Mini-batch Gradient Descent
domain: MATH
module: M-MATH-05
type: Algorithm
difficulty: L3 推导
status: planned
prerequisites: [MATH-703]
related_math: [MATH-703]
related_python: [PY-068]
related_ml: []
related_dl: []
related_llm: []
notion_page_id: 40c663e8-0333-81aa-9fcc-d8c1a914f049
notion_url: https://app.notion.com/p/SGD-40c663e8033381aa9fccd8c1a914f049
created: 2026-11-02
updated: 2026-11-02
notion_db: 知识点
学科: 数学
学习状态: 未开始
难度: L3 推导
知识类型: Algorithm
所属模块: [M-MATH-05]
前置知识: [MATH-703]
后继知识: [MATH-705]
关联Python: [PY-068]
关联数学: [MATH-703]
被引用为数学: [MATH-705, DL-024, DL-025, DL-026, DL-027, DL-028]
tags: [数学, Algorithm]
---

# MATH-704 SGD 与小批量梯度下降

## 1. 知识概览

| 项 | 内容 |
|---|---|
| 知识 ID | `MATH-704` |
| 英文名称 | Stochastic and Mini-batch Gradient Descent |
| 所属模块 | M-MATH-05 优化方法 |
| 难度等级 | L3 推导 |
| 前置知识 | [[MATH-703 批量梯度下降]] |
| 所属知识体系 | 噪声 → 学习率衰减 → 收敛性 |
| **AI 应用场景** | DataLoader 的 shuffle、学习率衰减、SGD 泛化优势 |
| **预计学习时间** | 50 分钟 |

**为什么学它**：SGD 的**噪声**既是缺点也是优点：缺点是收敛抖动，优点是能跳出浅坑、泛化更好。理解噪声的量级（$\propto 1/\sqrt b$）与衰减策略，才能调稳训练。

## 2. 直观理解

全批量像**每次都把所有题做一遍再改错**，SGD 像**做一题就改一次**。

- 做一题就改：改得快，但可能被这道题带偏。
- 一堆题一起改：方向准，但慢。

**关键技巧：学习率随时间衰减**。前期大步快跑，后期小步微调——因为噪声始终存在，步长不减小就会在最优点附近一直跳。

## 3. 正式数学定义

**SGD 更新**（单样本或小批量）

$$
\theta_{t+1} = \theta_t - \eta_t\,g_t, \qquad \mathbb{E}[g_t] = \nabla L(\theta_t)
$$

**噪声方差**（$b$ 为批量大小，$\sigma_g^2$ 为单样本梯度方差）

$$
\operatorname{Var}(g_t) = \frac{\sigma_g^2}{b}
$$

**常用学习率衰减**

| 策略 | 公式 |
|---|---|
| 常数 | $\eta_t=\eta_0$ |
| 反比 | $\eta_t=\dfrac{\eta_0}{1+kt}$ |
| 平方根 | $\eta_t=\dfrac{\eta_0}{\sqrt{t+1}}$ |
| 阶梯 | 每 $T$ 轮乘 0.1 |
| 余弦退火 | $\eta_t=\eta_{min}+\frac12(\eta_0-\eta_{min})(1+\cos\frac{t\pi}{T})$ |

**Robbins–Monro 条件**（保证收敛）

$$
\sum_t \eta_t = \infty, \qquad \sum_t \eta_t^2 < \infty
$$

即"衰减但不能衰减太快"。$\eta_t=\eta_0/\sqrt{t}$ 满足；$\eta_t=\eta_0/t$ 也满足；常数学习率不满足。

## 4. 数学推导

### 4.1 为什么噪声方差是 $\sigma_g^2/b$

**起点**：$g_t=\frac1b\sum_{i\in B_t}\nabla\ell_i$，$B_t$ 独立同分布随机抽样。

**第 1 步**：每个 $\nabla\ell_i$ 独立，方差 $\sigma_g^2$。

**第 2 步**：求和方差

$$
\operatorname{Var}\left(\sum_{i\in B}\nabla\ell_i\right) = b\sigma_g^2
$$

**第 3 步**：乘 $\frac1b$ 后方差乘 $\frac{1}{b^2}$

$$
\operatorname{Var}(g_t) = \frac{b\sigma_g^2}{b^2} = \frac{\sigma_g^2}{b}
$$

**第 4 步**：噪声幅度（标准差）

$$
\sigma_{g_t} = \frac{\sigma_g}{\sqrt b}
$$

**结论**：**batch 越大越稳，但收益递减**（要噪声减半需 batch 增 4 倍）。这解释了为什么 batch 从 32 增到 64 有明显改善，而从 1024 增到 2048 几乎看不出差别。

### 4.2 为什么学习率必须衰减

**起点**：在最优点附近，梯度 $\approx0$，但噪声不为 0。

**第 1 步**：更新式在最优点附近

$$
\theta_{t+1} = \theta^* - \eta_t\,g_t
$$

**第 2 步**：$g_t$ 的期望为 0（已在最优点），但方差为 $\sigma_g^2/b$。

**第 3 步**：所以 $\theta$ 会在 $\theta^*$ 附近做**随机游走**，游走幅度 $\approx\eta_t\sigma_g/\sqrt b$。

**第 4 步**：若 $\eta_t$ 恒定，游走幅度不衰减 ⇒ 无法精确收敛，只能在邻域内震荡。

**结论**：**衰减学习率让游走幅度趋于 0**，从而收敛到最优点。这是 Robbins–Monro 条件的直观含义。

## 5. 几何直观与图解

```mermaid
flowchart LR
    A["全批量：平滑路径"] --> B["直接到最优点"]
    C["SGD：锯齿路径"] --> D["在最优点附近抖动"]
    D -->|"学习率衰减"| E["抖动幅度变小，收敛"]
```

```text
全批量：   ╲
             ╲__●
SGD：     ╲ ╱╲ ╱╲
            ╳  ╳  ●  ← 抖动着靠近
```

## 6. 计算示例

**例**：目标 $L(w)=\mathbb{E}[(w-X)^2]$，$X\sim N(3,4)$，最优 $w^*=3$。

在 $w=0$ 处，梯度 $g=-2\bar X$。不同批量的梯度方差（理论上 $4\cdot 4/b=16/b$）：

| batch $b$ | 实测方差 | 理论 $16/b$ |
|---|---|---|
| 1 | 15.26 | 16 |
| 8 | 2.0066 | 2 |
| 64 | 0.2526 | 0.25 |

实测与理论一致，验证了 $\sigma_g^2/b$。

**读法**：$b=1$ 时梯度方差约 16，$b=64$ 时降到 0.25 —— **噪声幅度从 4 降到 0.5**，正好 8 倍（$\sqrt{64}$）。

## 7. Python 辅助验证

**依赖**：NumPy 2.5.4。

```powershell
python math704_sgd.py
```

```python
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(3.0, 2.0, size=200000)


def grad_variance(batch, reps=2000):
    gs = []
    for _ in range(reps):
        idx = rng.integers(0, len(X), batch)
        gs.append(2 * (0.0 - X[idx]).mean())   # w=0 处的梯度
    return float(np.var(gs))


for b in (1, 8, 64):
    print(f"batch={b}: 梯度方差 {grad_variance(b):.4f}")
```

**实际运行结果**

```
batch=1: 梯度方差 15.2600
batch=8: 梯度方差 2.0066
batch=64: 梯度方差 0.2526
```

**代码解析**

- 每个 batch 独立重采样，模拟真实训练里的 shuffle。
- 方差随着 $b$ 增大近似按 $1/b$ 下降：$15.26\to2.01\to0.25$（约 1 : 8 : 64）。
- 采样有随机性，数值不会精确等于 $16/b$，但量级一致。
- 这也说明**梯度估计是无偏的**：均值都接近 0（在 $w=0$ 处 $E[g]=-6$，这里只测方差）。

## 8. 知识关联

| 类型 | 链接 |
|---|---|
| 前置数学 | [[MATH-703 批量梯度下降]]、[[MATH-702 学习率与收敛]]、[[MATH-401 大数定律与中心极限定理]] |
| 后续数学 | [[MATH-705 Momentum 与 RMSProp]]、[[MATH-706 Adam 优化器]] |
| 相关 Python | [[PY-068 随机数、视图与副本]] |
| 相关机器学习 | 待关联（SGD 的泛化优势、学习率调度） |
| 相关深度学习 | 待关联（OneCycle、余弦退火、warmup） |
| 相关大语言模型 | 待关联（大批量训练 + 线性缩放） |

## 9. 实战练习

**基础 1**：batch 从 16 增到 64，噪声幅度变成原来的几分之几？

**基础 2**：判断 $\eta_t=1/t$ 是否满足 Robbins–Monro 条件。

**进阶 1**：为什么"batch 增大 4 倍，学习率增 4 倍"是常见经验规则？

**综合 1**：训练 100 轮后 loss 一直小幅震荡不降，可能原因与解决办法？

## 10. 参考答案

> [!success]- 基础 1
> $$\sqrt{16/64} = \sqrt{1/4} = \frac12$$
>
> 变成 **1/2**。

> [!success]- 基础 2
> $\sum 1/t$ 发散（调和级数）✓；$\sum 1/t^2$ 收敛 ✓。
>
> 满足条件，理论上可收敛。但实际衰减太快，前期就走不动，工程上少用。

> [!success]- 进阶 1
> batch 增 $k$ 倍 ⇒ 梯度方差降 $k$ 倍 ⇒ 噪声幅度降 $\sqrt k$ 倍。
>
> 理论上可把学习率放大到 $k$ 倍（步子更准，可以迈更大），这被称为**线性缩放规则**；实践中常配 warmup 避免初期不稳。

> [!success]- 综合 1
> 典型原因是**学习率到了后期仍然太大**，在最优点附近随机游走。
>
> 解决办法：
> 1. 加学习率衰减（余弦退火 / ReduceLROnPlateau）；
> 2. 增大 batch 降低噪声；
> 3. 换 Adam 等自适应方法；
> 4. 检查是否已达到该模型的能力上限（loss 平台）。

## 11. 常见错误

| 类型 | 错误表现 | 原因 | 纠正 |
|---|---|---|---|
| 实践错误 | 全程固定大学习率 | 后期抖动 | 加衰减 |
| 概念错误 | 认为噪声一定有害 | 噪声帮助跳出浅坑 | 适度噪声有益 |
| 计算错误 | 以为噪声 $\propto1/b$ | 混淆方差与标准差 | 幅度 $\propto1/\sqrt b$ |
| 实践错误 | shuffle 只在第一轮做 | 顺序偏差 | 每轮都打乱 |
| 实践错误 | 衰减太快 | 前期学不动 | 用 $\sqrt t$ 或阶段衰减 |

## 12. 学习记录

| 日期 | 状态 | 掌握程度 | 复习状态 | 学习笔记 | 关联练习 |
|---|---|---|---|---|---|
| — | planned | — | — | — | 练习 1–4 |

---

## 🔗 Notion 同步信息

> 以下内容由 Notion「AI Knowledge System · 知识点」数据库按知识 ID 合并而来（导出于 2026-11-09）。

| 属性 | 值 |
|---|---|
| 学科 | 数学 |
| 学习状态 | 未开始 |
| 难度 | L3 推导 |
| 知识类型 | Algorithm |

- **所属模块**：[M-MATH-05 优化方法](../%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-MATH-05%20%E4%BC%98%E5%8C%96%E6%96%B9%E6%B3%95.md)
- **前置知识**：[MATH-703 批量梯度下降](MATH-703%20%E6%89%B9%E9%87%8F%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D.md)
- **后继知识**：[MATH-705 Momentum 与 RMSProp](MATH-705%20Momentum%20%E4%B8%8E%20RMSProp.md)
- **关联Python**：[PY-068 随机数、视图与副本](../../01_Python%E5%9F%BA%E7%A1%80/08%20NumPy/PY-068%20%E9%9A%8F%E6%9C%BA%E6%95%B0%E3%80%81%E8%A7%86%E5%9B%BE%E4%B8%8E%E5%89%AF%E6%9C%AC.md)
- **关联数学**：[MATH-703 批量梯度下降](MATH-703%20%E6%89%B9%E9%87%8F%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D.md)
- **被引用为数学**：[MATH-705 Momentum 与 RMSProp](MATH-705%20Momentum%20%E4%B8%8E%20RMSProp.md) · [DL-024 Epoch、Batch 与 Mini-batch](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/05%20%E8%AE%AD%E7%BB%83%E6%8A%80%E5%B7%A7/DL-024%20Epoch%E3%80%81Batch%20%E4%B8%8E%20Mini-batch.md) · [DL-025 Batch Size 与 Learning Rate](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/05%20%E8%AE%AD%E7%BB%83%E6%8A%80%E5%B7%A7/DL-025%20Batch%20Size%20%E4%B8%8E%20Learning%20Rate.md) · [DL-026 SGD 与 Momentum](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/05%20%E8%AE%AD%E7%BB%83%E6%8A%80%E5%B7%A7/DL-026%20SGD%20%E4%B8%8E%20Momentum.md) · [DL-027 Nesterov、AdaGrad 与 RMSprop](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/05%20%E8%AE%AD%E7%BB%83%E6%8A%80%E5%B7%A7/DL-027%20Nesterov%E3%80%81AdaGrad%20%E4%B8%8E%20RMSprop.md) · [DL-028 Adam 与 AdamW](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/05%20%E8%AE%AD%E7%BB%83%E6%8A%80%E5%B7%A7/DL-028%20Adam%20%E4%B8%8E%20AdamW.md)

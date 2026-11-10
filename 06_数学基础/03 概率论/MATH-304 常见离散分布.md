---
id: MATH-304
title: 常见离散分布
en_title: Common Discrete Distributions
domain: MATH
module: M-MATH-04
type: Concept
difficulty: L3 推导
status: planned
prerequisites: [MATH-303]
related_math: [MATH-303]
related_python: [PY-068]
related_ml: []
related_dl: []
related_llm: []
notion_page_id: 40c663e8-0333-810a-b2e5-c95dead1a205
notion_url: https://app.notion.com/p/40c663e80333810ab2e5c95dead1a205
created: 2026-11-02
updated: 2026-11-02
notion_db: 知识点
学科: 数学
学习状态: 未开始
难度: L3 推导
知识类型: Concept
所属模块: [M-MATH-04]
前置知识: [MATH-303]
后继知识: [MATH-305]
关联Python: [PY-068]
关联数学: [MATH-303]
被引用为数学: [MATH-305]
tags: [数学, Concept]
---

# MATH-304 常见离散分布

## 1. 知识概览

| 项 | 内容 |
|---|---|
| 知识 ID | `MATH-304` |
| 英文名称 | Common Discrete Distributions |
| 所属模块 | M-MATH-04 概率论与数理统计 |
| 难度等级 | L3 推导 |
| 前置知识 | [[MATH-303 随机变量与分布函数]] |
| 所属知识体系 | 伯努利 → 二项 → 泊松 |
| **AI 应用场景** | 分类任务用伯努利；采样/计数建模用二项与泊松；dropout 本质是伯努利 |
| **预计学习时间** | 55 分钟 |

**为什么学它**：模型输出常常被假设成某个分布——二分类是伯努利、多分类是类别分布、计数是泊松。**选定分布就等于选定了损失函数**（负对数似然）。

## 2. 直观理解

三种离散分布，是同一件事的三个层次：

| 分布 | 场景 | 类比 |
|---|---|---|
| 伯努利 | 做一次试验，成功/失败 | 抛一次硬币 |
| 二项 | 做 $n$ 次，成功几次 | 抛 $n$ 次硬币数正面 |
| 泊松 | 单位时间/空间内发生几次 | 一小时内来几个电话 |

**关系**：伯努利是 $n=1$ 的二项；二项在 $n$ 大、$p$ 小时近似泊松。

## 3. 正式数学定义

**伯努利分布** $X\sim\text{Bernoulli}(p)$

$$
P(X=1)=p,\quad P(X=0)=1-p \qquad \mathbb{E}=p,\ \operatorname{Var}=p(1-p)
$$

**二项分布** $X\sim B(n,p)$

$$
P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}, \quad k=0,\dots,n
$$

$$
\mathbb{E}=np,\qquad \operatorname{Var}=np(1-p)
$$

**泊松分布** $X\sim\text{Poisson}(\lambda)$

$$
P(X=k)=\frac{\lambda^k e^{-\lambda}}{k!}, \quad k=0,1,2,\dots
$$

$$
\mathbb{E}=\lambda,\qquad \operatorname{Var}=\lambda
$$

**泊松近似**：当 $n$ 大、$p$ 小、$\lambda=np$ 适中时

$$
B(n,p)\approx\text{Poisson}(np)
$$

## 4. 数学推导

### 4.1 二项分布的期望与方差

**起点**：$X=\sum_{i=1}^{n}X_i$，每个 $X_i\sim\text{Bernoulli}(p)$ 独立。

**第 1 步**：期望的线性性（无需独立）

$$
\mathbb{E}[X] = \sum_i\mathbb{E}[X_i] = np
$$

**第 2 步**：方差在独立时才可加

$$
\operatorname{Var}(X) = \sum_i\operatorname{Var}(X_i) = n\,p(1-p)
$$

**结论**：这两个公式来自"把二项拆成 $n$ 个伯努利之和"——比直接对 $\sum k\binom{n}{k}p^k q^{n-k}$ 求和省事得多。

### 4.2 泊松分布是二项分布的极限

**起点**：$X\sim B(n,p)$，令 $np=\lambda$ 固定，$n\to\infty$、$p\to0$。

**第 1 步**：写出概率并代入 $p=\lambda/n$

$$
P(X=k)=\binom{n}{k}\left(\frac{\lambda}{n}\right)^k\left(1-\frac{\lambda}{n}\right)^{n-k}
$$

**第 2 步**：展开组合数

$$
=\frac{n(n-1)\cdots(n-k+1)}{k!}\cdot\frac{\lambda^k}{n^k}\cdot\left(1-\frac{\lambda}{n}\right)^{n-k}
$$

**第 3 步**：分别取极限

$$
\frac{n(n-1)\cdots(n-k+1)}{n^k}\to1, \qquad \left(1-\frac{\lambda}{n}\right)^{n}\to e^{-\lambda}
$$

以及 $\left(1-\frac{\lambda}{n}\right)^{-k}\to1$。

**第 4 步**：合并

$$
P(X=k)\to\frac{\lambda^k e^{-\lambda}}{k!}
$$

**结论**：**稀有事件的二项分布近似泊松**。这就是"每天交通事故数""每小时呼叫数"用泊松建模的原因。

## 5. 几何直观与图解

```mermaid
flowchart LR
    A["一次试验<br/>Bernoulli(p)"] -->|"n 次独立重复"| B["二项 B(n,p)"]
    B -->|"n→∞，p→0，np=λ"| C["泊松 Poisson(λ)"]
```

```text
B(10,0.3): 柱子中间高（0.267 在 k=3），两侧衰减
Poisson(4): 右偏，峰值在 k=3~4
```

## 6. 计算示例

**例 1**：$B(10,0.3)$，求 $P(X=3)$。

$$
\binom{10}{3}(0.3)^3(0.7)^7 = 120\times0.027\times0.0823543 \approx 0.266828
$$

**例 2**：$B(10,0.3)$ 的期望与方差。

$$
\mathbb{E}=10\times0.3=3,\qquad \operatorname{Var}=3\times0.7=2.1
$$

**例 3**：$\text{Poisson}(4)$，求 $P(X=2)$。

$$
\frac{4^2 e^{-4}}{2!} = \frac{16\times0.0183156}{2} \approx 0.146525
$$

**例 4**：验证二项分布归一化。

$$
\sum_{k=0}^{10}\binom{10}{k}0.3^k0.7^{10-k} = 1 \;\checkmark
$$

## 7. Python 辅助验证

**依赖**：`math`（`comb` 组合数）。

```powershell
python math304_discrete.py
```

```python
import math
from math import comb


def binom_pmf(k, n, p):
    return comb(n, k) * p**k * (1 - p) ** (n - k)


def poisson_pmf(k, lam):
    return math.exp(-lam) * lam**k / math.factorial(k)


print("B(10,0.3) P(X=3):", round(binom_pmf(3, 10, 0.3), 6))
print("B(10,0.3) 均值:", 10 * 0.3, "方差:", round(10 * 0.3 * 0.7, 6))
print("Poisson(4) P(X=2):", round(poisson_pmf(2, 4), 6))
print("二项分布全概率:", round(sum(binom_pmf(k, 10, 0.3) for k in range(11)), 10))
```

**实际运行结果**

```
B(10,0.3) P(X=3): 0.266828
B(10,0.3) 均值: 3.0 方差: 2.1
Poisson(4) P(X=2): 0.146525
二项分布全概率: 1.0
```

**代码解析**

- `comb(n, k)` 直接给组合数，避免手写阶乘。
- 全概率求和恰好等于 1.0，验证 PMF 合法。
- 泊松用 `math.factorial`，$k$ 很大时会溢出，工程上应改用 `math.lgamma` 或对数形式。
- 均值与方差用公式直接算，与理论一致。

## 8. 知识关联

| 类型 | 链接 |
|---|---|
| 前置数学 | [[MATH-303 随机变量与分布函数]]、[[MATH-301 概率基础与条件概率]] |
| 后续数学 | [[MATH-305 常见连续分布与正态分布]]、[[MATH-306 期望、方差与协方差]]、`MATH-404 最大似然估计`（待建） |
| 相关 Python | [[PY-068 随机数、视图与副本]] |
| 相关机器学习 | 待关联（二分类的伯努利假设、dropout） |
| 相关深度学习 | 待关联（采样、计数建模） |
| 相关大语言模型 | 待关联（token 采样的类别分布） |

## 9. 实战练习

**基础 1**：$B(5,0.4)$，求 $P(X=2)$。

**基础 2**：$X\sim\text{Poisson}(3)$，求 $P(X=0)$。

**进阶 1**：某工厂次品率 0.02，抽 100 件，求次品数不超过 1 件的概率（用二项与泊松各算一次并比较）。

**综合 1**：为什么 dropout 可以看成伯努利采样？写出它的期望作用。

## 10. 参考答案

> [!success]- 基础 1
> $$\binom{5}{2}(0.4)^2(0.6)^3 = 10\times0.16\times0.216 = 0.3456$$

> [!success]- 基础 2
> $$P(X=0) = \frac{3^0e^{-3}}{0!} = e^{-3} \approx 0.049787$$

> [!success]- 进阶 1
> **二项**：$P(X\le1)=P(0)+P(1)$
>
> $$0.98^{100} + 100\times0.02\times0.98^{99} \approx 0.1326 + 0.2707 = 0.4033$$
>
> **泊松**（$\lambda=2$）：$e^{-2}(1+2) = 3e^{-2}\approx0.4060$
>
> 两者相差约 0.003，泊松近似够用。

> [!success]- 综合 1
> dropout 对每个神经元以概率 $p$ 保留、$1-p$ 置零，这正是**伯努利采样**（$\text{Bernoulli}(p)$）逐元素相乘。
>
> 若不做缩放，期望输出会变成原来的 $p$ 倍，所以推理时要乘 $p$（或训练时除以 $p$，即 inverted dropout）。

## 11. 常见错误

| 类型 | 错误表现 | 原因 | 纠正 |
|---|---|---|---|
| 概念错误 | 二项方差写成 $np$ | 与泊松混淆 | $np(1-p)$ |
| 公式错误 | 组合数写成 $n^k$ | 忽略"哪几次" | $\binom{n}{k}$ |
| 概念错误 | 泊松用于"总次数固定"的场景 | 适用条件错 | 泊松用于稀有独立事件 |
| 计算错误 | 大 $k$ 直接算阶乘 | 溢出 | 用对数或 `lgamma` |
| 概念错误 | 认为二项与泊松是同一分布 | 近似≠相等 | 仅在 $n$ 大 $p$ 小时近似 |

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
| 知识类型 | Concept |

- **所属模块**：[M-MATH-04 概率论与数理统计](../%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-MATH-04%20%E6%A6%82%E7%8E%87%E8%AE%BA%E4%B8%8E%E6%95%B0%E7%90%86%E7%BB%9F%E8%AE%A1.md)
- **前置知识**：[MATH-303 随机变量与分布函数](MATH-303%20%E9%9A%8F%E6%9C%BA%E5%8F%98%E9%87%8F%E4%B8%8E%E5%88%86%E5%B8%83%E5%87%BD%E6%95%B0.md)
- **后继知识**：[MATH-305 常见连续分布与正态分布](MATH-305%20%E5%B8%B8%E8%A7%81%E8%BF%9E%E7%BB%AD%E5%88%86%E5%B8%83%E4%B8%8E%E6%AD%A3%E6%80%81%E5%88%86%E5%B8%83.md)
- **关联Python**：[PY-068 随机数、视图与副本](../../01_Python%E5%9F%BA%E7%A1%80/08%20NumPy/PY-068%20%E9%9A%8F%E6%9C%BA%E6%95%B0%E3%80%81%E8%A7%86%E5%9B%BE%E4%B8%8E%E5%89%AF%E6%9C%AC.md)
- **关联数学**：[MATH-303 随机变量与分布函数](MATH-303%20%E9%9A%8F%E6%9C%BA%E5%8F%98%E9%87%8F%E4%B8%8E%E5%88%86%E5%B8%83%E5%87%BD%E6%95%B0.md)
- **被引用为数学**：[MATH-305 常见连续分布与正态分布](MATH-305%20%E5%B8%B8%E8%A7%81%E8%BF%9E%E7%BB%AD%E5%88%86%E5%B8%83%E4%B8%8E%E6%AD%A3%E6%80%81%E5%88%86%E5%B8%83.md)

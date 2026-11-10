---
id: MATH-404
title: 最大似然估计
en_title: Maximum Likelihood Estimation
domain: MATH
module: M-MATH-04
type: Algorithm
difficulty: L3 推导
status: planned
prerequisites: [MATH-403]
related_math: [MATH-403]
related_python: [PY-067]
related_ml: []
related_dl: []
related_llm: []
notion_page_id: 40c663e8-0333-81fb-a976-df5f3c0b3973
notion_url: https://app.notion.com/p/40c663e8033381fba976df5f3c0b3973
created: 2026-11-02
updated: 2026-11-02
notion_db: 知识点
学科: 数学
学习状态: 未开始
难度: L3 推导
知识类型: Algorithm
所属模块: [M-MATH-04]
前置知识: [MATH-403]
关联Python: [PY-067]
关联数学: [MATH-403]
tags: [数学, Algorithm]
---

# MATH-404 最大似然估计

## 1. 知识概览

| 项 | 内容 |
|---|---|
| 知识 ID | `MATH-404` |
| 英文名称 | Maximum Likelihood Estimation |
| 所属模块 | M-MATH-04 概率论与数理统计 |
| 难度等级 | L3 推导 |
| 前置知识 | [[MATH-403 假设检验基础]] |
| 所属知识体系 | 似然函数 → 对数似然 → 求导求极值 → MLE |
| **AI 应用场景** | 几乎所有分类/回归的损失函数都是负对数似然：MSE↔高斯、交叉熵↔伯努利 |
| **预计学习时间** | 60 分钟 |

**为什么学它**：**交叉熵损失不是随手定的**，它是"伯努利分布下的负对数似然"。理解 MLE，就能明白"为什么分类用交叉熵、回归用 MSE"，以及"为什么加 L2 正则等于加高斯先验"。

## 2. 直观理解

MLE 的思路：**哪个参数值最可能产生我看到的这批数据？**

- 抛 10 次硬币出 7 次正面。真实正面概率 $p$ 是多少？
- 试 $p=0.5$：出现 7 正的概率较小。
- 试 $p=0.7$：出现 7 正的概率最大。
- 所以 MLE 给出 $\hat p=0.7$。

**似然 vs 概率**：概率是"给定参数，数据出现的可能性"；似然是"给定数据，把它看成参数的函数"。**数值一样，视角相反**。

## 3. 正式数学定义

**似然函数**（$x_1,\dots,x_n$ 独立同分布）

$$
L(\theta) = \prod_{i=1}^{n} f(x_i;\theta)
$$

**对数似然**（乘积变求和，且不改变极值点）

$$
\ell(\theta) = \ln L(\theta) = \sum_{i=1}^{n}\ln f(x_i;\theta)
$$

**最大似然估计**

$$
\hat\theta_{MLE} = \arg\max_\theta \ell(\theta)
$$

**求解步骤**：求导 → 令导数为 0 → 解方程 → 验证是极大（二阶导 < 0）。

**负对数似然（NLL）**：$-\ell(\theta)$，最小化 NLL 等价于最大化似然。机器学习里通常写成 NLL 的均值。

**不变性**：若 $\hat\theta$ 是 $\theta$ 的 MLE，则 $g(\hat\theta)$ 是 $g(\theta)$ 的 MLE。

## 4. 数学推导

### 4.1 伯努利分布的 MLE 就是样本频率

**起点**：$x_i\in\{0,1\}$，$P(X=1)=p$。

**第 1 步**：似然

$$
L(p) = \prod_i p^{x_i}(1-p)^{1-x_i} = p^{k}(1-p)^{n-k}, \quad k=\sum x_i
$$

**第 2 步**：对数似然

$$
\ell(p) = k\ln p + (n-k)\ln(1-p)
$$

**第 3 步**：求导并令 0

$$
\ell'(p) = \frac{k}{p} - \frac{n-k}{1-p} = 0
$$

**第 4 步**：交叉相乘

$$
k(1-p) = (n-k)p \;\Rightarrow\; k = np \;\Rightarrow\; \hat p = \frac{k}{n}
$$

**第 5 步**：二阶导 $-\frac{k}{p^2}-\frac{n-k}{(1-p)^2}<0$，确为极大。

**结论**：$\hat p = \bar x$，**MLE 就是样本频率**。

### 4.2 正态分布 MLE 的方差为什么除以 $n$

**起点**：$x_i\sim N(\mu,\sigma^2)$。

**第 1 步**：对数似然

$$
\ell = -\frac{n}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum(x_i-\mu)^2
$$

**第 2 步**：对 $\mu$ 求导并令 0

$$
\frac{1}{\sigma^2}\sum(x_i-\mu)=0 \;\Rightarrow\; \hat\mu = \bar x
$$

**第 3 步**：对 $\sigma^2$ 求导并令 0（记 $v=\sigma^2$）

$$
-\frac{n}{2v} + \frac{1}{2v^2}\sum(x_i-\bar x)^2 = 0 \;\Rightarrow\; \hat\sigma^2 = \frac{1}{n}\sum(x_i-\bar x)^2
$$

**第 4 步**：注意这里是 $1/n$，不是 $1/(n-1)$。

**结论**：**MLE 的方差估计是有偏的**（系统性偏小）。这与 [[MATH-402 参数估计]] 的结论一致——无偏估计要除以 $n-1$，但 MLE 给的是 $n$。

## 5. 几何直观与图解

```mermaid
flowchart LR
    A["数据 x₁…xₙ"] --> B["写似然 L(θ)"]
    B --> C["取对数 ℓ(θ)"]
    C --> D["求导=0"]
    D --> E["解出 θ̂"]
    E --> F["验证二阶导<0"]
```

```text
ℓ(θ) 随 θ 变化（伯努利，7 正 3 反）
      ╱‾╲
    ╱     ╲
  ╱         ╲
─┴────┼────┴──
     0.7
   峰值在样本频率处
```

## 6. 计算示例

**例**：抛硬币 10 次，7 次正面。求 $p$ 的 MLE。

$$
\hat p = \frac{7}{10} = 0.7
$$

**验证**：比较似然值

| $p$ | $L(p)=p^7(1-p)^3$ |
|---|---|
| 0.5 | $0.000977$ |
| 0.7 | $0.002224$ |
| 0.9 | $0.000478$ |

$p=0.7$ 处最大 ✓

**例 2**：正态样本 $n=2000$，真值 $\mu=2$、$\sigma=1.5$。

实测：$\hat\mu = 2.0415$，$\hat\sigma^2 = 2.2164$（MLE，除以 $n$）。

对数似然：

| $\mu$ | $\ell$ |
|---|---|
| 1.0 | $-4123.165$ |
| 2.0 | $-3634.548$ |
| 2.0415（MLE） | $-3633.771$ |

MLE 处最大 ✓。若把方差放大 1.5 倍，$\ell$ 降到 $-3705.903$。

## 7. Python 辅助验证

**依赖**：NumPy 2.5.4 + `math`。

```powershell
python math404_mle.py
```

```python
import math

import numpy as np

rng = np.random.default_rng(3)
x = rng.normal(2.0, 1.5, size=2000)
n = len(x)

mu_hat = float(x.mean())
s2_hat = float(((x - mu_hat) ** 2).mean())
print("MLE mu:", round(mu_hat, 4), " MLE sigma²:", round(s2_hat, 4))


def loglik(mu, var):
    return float(-0.5 * n * math.log(2 * math.pi * var)
                 - ((x - mu) ** 2).sum() / (2 * var))


for mu in (1.0, 2.0, round(mu_hat, 4)):
    print(f"  mu={mu}: loglik={round(loglik(mu, s2_hat), 3)}")

print("方差放大 1.5 倍:", round(loglik(mu_hat, s2_hat * 1.5), 3))
```

**实际运行结果**

```
MLE mu: 2.0415  MLE sigma²: 2.2164
  mu=1.0: loglik=-4123.165
  mu=2.0: loglik=-3634.548
  mu=2.0415: loglik=-3633.771
方差放大 1.5 倍: -3705.903
```

**代码解析**

- `MLE` 的 $\mu$ 就是样本均值，$\sigma^2$ 用除以 $n$ 的版本。
- 三个 $\mu$ 的对数似然中，$\hat\mu$ 处最大——数值确认了 MLE 的定义。
- 方差放大后似然明显下降，说明 MLE 对 $\sigma^2$ 也是最优的。
- 真值 $\mu=2$ 与估计值 2.0415 的差来自抽样误差（$\operatorname{SE}\approx1.5/\sqrt{2000}\approx0.034$）。

## 8. 知识关联

| 类型 | 链接 |
|---|---|
| 前置数学 | [[MATH-403 假设检验基础]]、[[MATH-402 参数估计]]、[[MATH-305 常见连续分布与正态分布]] |
| 后续数学 | `MATH-502 交叉熵与 KL 散度`（待建）、`MATH-801 线性回归的数学`（待建） |
| 相关 Python | [[PY-067 聚合与统计]] |
| 相关机器学习 | 待关联（交叉熵＝伯努利 NLL，MSE＝高斯 NLL） |
| 相关深度学习 | 待关联（损失函数设计） |
| 相关大语言模型 | 待关联（语言模型的负对数似然训练目标） |

## 9. 实战练习

**基础 1**：抛 20 次硬币，12 次正面，求 $p$ 的 MLE。

**基础 2**：写出正态分布的对数似然（含 $\mu,\sigma^2$）。

**进阶 1**：证明伯努利 MLE $\hat p = k/n$ 是无偏的。

**综合 1**：为什么分类任务用交叉熵而不用 MSE？用 MLE 的语言解释。

## 10. 参考答案

> [!success]- 基础 1
> $$\hat p = \frac{12}{20} = 0.6$$

> [!success]- 基础 2
> $$\ell(\mu,\sigma^2) = -\frac{n}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i-\mu)^2$$

> [!success]- 进阶 1
> $k=\sum x_i$，每个 $x_i\sim\text{Bernoulli}(p)$，$\mathbb{E}[x_i]=p$。
>
> $$\mathbb{E}[\hat p] = \mathbb{E}\left[\frac1n\sum x_i\right] = \frac1n\cdot np = p$$
>
> 所以无偏。

> [!success]- 综合 1
> 分类的目标是预测概率，对应伯努利（或多类时类别）分布的似然。其负对数似然正是交叉熵。
>
> 而 MSE 对应"高斯噪声下的回归"——它隐含假设误差正态。用 MSE 训练分类器，等价于假设一个不合适的噪声模型，梯度在预测严重错误时反而变小（梯度消失），学习变慢。
>
> 用交叉熵则梯度与误差成正比，收敛更快。

## 11. 常见错误

| 类型 | 错误表现 | 原因 | 纠正 |
|---|---|---|---|
| 概念错误 | 把似然当"参数的概率" | 混淆似然与后验 | 似然不是概率分布 |
| 计算错误 | 直接对乘积求导 | 太复杂易错 | 先取对数 |
| 概念错误 | 认为 MLE 一定无偏 | 正态方差就是反例 | MLE 未必无偏 |
| 计算错误 | 忘记验证二阶导 | 可能求出极大/极小 | 检查 $\ell''<0$ |
| 概念错误 | 样本少时过度相信 MLE | 方差大 | 小样本可加先验（MAP） |

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

- **所属模块**：[M-MATH-04 概率论与数理统计](../%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-MATH-04%20%E6%A6%82%E7%8E%87%E8%AE%BA%E4%B8%8E%E6%95%B0%E7%90%86%E7%BB%9F%E8%AE%A1.md)
- **前置知识**：[MATH-403 假设检验基础](MATH-403%20%E5%81%87%E8%AE%BE%E6%A3%80%E9%AA%8C%E5%9F%BA%E7%A1%80.md)
- **关联Python**：[PY-067 聚合与统计](../../01_Python%E5%9F%BA%E7%A1%80/08%20NumPy/PY-067%20%E8%81%9A%E5%90%88%E4%B8%8E%E7%BB%9F%E8%AE%A1.md)
- **关联数学**：[MATH-403 假设检验基础](MATH-403%20%E5%81%87%E8%AE%BE%E6%A3%80%E9%AA%8C%E5%9F%BA%E7%A1%80.md)

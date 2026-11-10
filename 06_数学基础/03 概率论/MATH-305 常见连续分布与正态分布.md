---
id: MATH-305
title: 常见连续分布与正态分布
en_title: Common Continuous Distributions and the Normal Distribution
domain: MATH
module: M-MATH-04
type: Concept
difficulty: L3 推导
status: planned
prerequisites: [MATH-304]
related_math: [MATH-304]
related_python: [PY-068]
related_ml: []
related_dl: []
related_llm: []
notion_page_id: 40c663e8-0333-8124-9311-d7b0b93b6ce8
notion_url: https://app.notion.com/p/40c663e8033381249311d7b0b93b6ce8
created: 2026-11-02
updated: 2026-11-02
notion_db: 知识点
学科: 数学
学习状态: 未开始
难度: L3 推导
知识类型: Concept
所属模块: [M-MATH-04]
前置知识: [MATH-304]
后继知识: [MATH-306]
关联Python: [PY-068]
关联数学: [MATH-304]
被引用为数学: [MATH-306, ML-027, PY-068]
tags: [数学, Concept]
---

# MATH-305 常见连续分布与正态分布

## 1. 知识概览

| 项 | 内容 |
|---|---|
| 知识 ID | `MATH-305` |
| 英文名称 | Common Continuous Distributions and the Normal Distribution |
| 所属模块 | M-MATH-04 概率论与数理统计 |
| 难度等级 | L3 推导 |
| 前置知识 | [[MATH-304 常见离散分布]] |
| 所属知识体系 | 均匀 → 正态 → 指数 |
| **AI 应用场景** | 权重初始化（正态）、噪声建模（高斯）、等待时间（指数）、CLT 的极限分布 |
| **预计学习时间** | 55 分钟 |

**为什么学它**：**正态分布是"万物趋近的极限"**（中心极限定理）。权重初始化、变分自编码器的先验、扩散模型的噪声，全都用正态。指数分布建模"等待时间"；均匀分布是随机数生成的基础。

## 2. 直观理解

| 分布 | 直观 | 用途 |
|---|---|---|
| 均匀 $U(a,b)$ | 区间内每个点一样可能 | 随机数、采样 |
| 正态 $N(\mu,\sigma^2)$ | 钟形，中间高两头低 | 自然的误差 |
| 指数 $\text{Exp}(\lambda)$ | 从 0 开始单调下降 | 等待时间 |

**正态为什么无处不在**：大量独立小因素叠加，结果趋向正态。身高、测量误差、噪声都近似如此。

**68-95-99.7 法则**：正态数据约有 68% 落在 $\mu\pm\sigma$、95% 在 $\mu\pm2\sigma$、99.7% 在 $\mu\pm3\sigma$。

## 3. 正式数学定义

**均匀分布** $X\sim U(a,b)$

$$
f(x)=\frac{1}{b-a}\ (a\le x\le b), \qquad \mathbb{E}=\frac{a+b}{2},\ \operatorname{Var}=\frac{(b-a)^2}{12}
$$

**正态分布** $X\sim N(\mu,\sigma^2)$

$$
f(x)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

$$
\mathbb{E}=\mu,\qquad \operatorname{Var}=\sigma^2
$$

**标准正态** $Z\sim N(0,1)$，其 CDF 记 $\Phi$。标准化：

$$
Z=\frac{X-\mu}{\sigma}\sim N(0,1)
$$

**指数分布** $X\sim\text{Exp}(\lambda)$（$x\ge0$）

$$
f(x)=\lambda e^{-\lambda x}, \qquad \mathbb{E}=\frac1\lambda,\ \operatorname{Var}=\frac{1}{\lambda^2}
$$

**无记忆性**（指数分布独有）

$$
P(X>s+t\mid X>s) = P(X>t)
$$

## 4. 数学推导

### 4.1 正态密度为什么积分为 1（高斯积分）

**起点**：要证

$$
\int_{-\infty}^{+\infty}e^{-x^2/2}\,dx = \sqrt{2\pi}
$$

**第 1 步**：令 $I=\int e^{-x^2/2}dx$，则

$$
I^2 = \int\!\!\int e^{-(x^2+y^2)/2}\,dx\,dy
$$

**第 2 步**：换成极坐标 $x=r\cos\theta$、$y=r\sin\theta$，$dx\,dy = r\,dr\,d\theta$

$$
I^2 = \int_0^{2\pi}\!\!\int_0^{\infty} e^{-r^2/2}\,r\,dr\,d\theta
$$

**第 3 步**：内层积分（令 $u=r^2/2$，$du=r\,dr$）

$$
\int_0^{\infty}e^{-r^2/2}r\,dr = \left[-e^{-r^2/2}\right]_0^{\infty} = 1
$$

**第 4 步**：外层

$$
I^2 = 2\pi\cdot1 = 2\pi \;\Rightarrow\; I=\sqrt{2\pi}
$$

**结论**：乘上系数 $\frac{1}{\sqrt{2\pi}\sigma}$ 后，正态密度全积分为 1，是合法的 PDF。

### 4.2 指数分布的无记忆性

**起点**：$P(X>t)=e^{-\lambda t}$（由 $\int_t^\infty\lambda e^{-\lambda x}dx$ 算出）。

**第 1 步**：条件概率定义

$$
P(X>s+t\mid X>s) = \frac{P(X>s+t)}{P(X>s)}
$$

**第 2 步**：代入

$$
=\frac{e^{-\lambda(s+t)}}{e^{-\lambda s}} = e^{-\lambda t}
$$

**第 3 步**：这正是 $P(X>t)$。

**结论**：**已经等了 $s$ 时间，对未来等待的分布毫无影响**。这与"灯泡不老化"的假设对应；若会老化（如机械磨损），应改用 Weibull 分布。

## 5. 几何直观与图解

```mermaid
flowchart LR
    A["正态 N(μ,σ²)"] -->|"标准化"| B["标准正态 N(0,1)"]
    B --> C["Φ 表 / erf 计算"]
    C --> D["68-95-99.7 法则"]
```

```text
正态：      ╱‾╲          均匀：  ┌────┐
          ╱   ╲                │    │
        ╱       ╲              └────┘
       μ                          a    b

指数：╲
      ╲___
     0
```

## 6. 计算示例

**例 1**：$Z\sim N(0,1)$，求 $P(|Z|<1)$、$P(|Z|<2)$、$P(|Z|<3)$。

$$
\Phi(1)-\Phi(-1)\approx0.6827,\quad 0.9545,\quad 0.9973
$$

**例 2**：$X\sim N(3,4)$（即 $\sigma=2$），求 $P(X<5)$。

标准化：$Z=(5-3)/2=1$，$P(Z<1)=\Phi(1)\approx0.8413$。

**例 3**：$X\sim\text{Exp}(0.5)$，求 $\mathbb{E}$、$\operatorname{Var}$、$P(X>2)$。

$$
\mathbb{E}=\frac1{0.5}=2,\quad \operatorname{Var}=\frac1{0.25}=4,\quad P(X>2)=e^{-1}\approx0.367879
$$

**例 4**：$X\sim U(0,4)$，求 $\mathbb{E}$、$\operatorname{Var}$。

$$
\mathbb{E}=2,\qquad \operatorname{Var}=\frac{16}{12}\approx1.3333
$$

## 7. Python 辅助验证

**依赖**：`math`（正态 CDF 用 `erf`）。

```powershell
python math305_continuous.py
```

```python
import math


def normal_cdf(x, mu=0.0, sigma=1.0):
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))


print("P(|Z|<1) =", round(normal_cdf(1) - normal_cdf(-1), 4))
print("P(|Z|<2) =", round(normal_cdf(2) - normal_cdf(-2), 4))
print("P(|Z|<3) =", round(normal_cdf(3) - normal_cdf(-3), 4))

lam = 0.5
print("Exp(0.5) 均值:", 1 / lam, " 方差:", 1 / lam**2)
print("P(X>2) =", round(math.exp(-lam * 2), 6))

print("U(0,4) 均值:", (0 + 4) / 2, " 方差:", round((4 - 0) ** 2 / 12, 6))
```

**实际运行结果**

```
P(|Z|<1) = 0.6827
P(|Z|<2) = 0.9545
P(|Z|<3) = 0.9973
Exp(0.5) 均值: 2.0  方差: 4.0
P(X>2) = 0.367879
U(0,4) 均值: 2.0  方差: 1.333333
```

**代码解析**

- `math.erf` 给出标准正态 CDF，无需 SciPy。
- 三个概率正是 68-95-99.7 法则的数值。
- 指数分布均值 $1/\lambda=2$，方差 $1/\lambda^2=4$。
- 均匀分布方差公式 $(b-a)^2/12$，这里 $16/12$。

## 8. 知识关联

| 类型 | 链接 |
|---|---|
| 前置数学 | [[MATH-304 常见离散分布]]、[[MATH-303 随机变量与分布函数]]、[[MATH-128 多重积分]] |
| 后续数学 | [[MATH-306 期望、方差与协方差]]、[[MATH-401 大数定律与中心极限定理]]、[[MATH-402 参数估计]] |
| 相关 Python | [[PY-068 随机数、视图与副本]] |
| 相关机器学习 | 待关联（高斯噪声假设、权重初始化） |
| 相关深度学习 | 待关联（扩散模型的噪声、VAE 先验） |
| 相关大语言模型 | 待关联（logits 初始化） |

## 9. 实战练习

**基础 1**：$Z\sim N(0,1)$，求 $P(Z<1.96)$。

**基础 2**：$X\sim N(10,25)$，求 $\sigma$ 与 $P(X<15)$。

**进阶 1**：某设备寿命服从 $\text{Exp}(0.1)$（单位：年），求平均寿命与 3 年后仍正常的概率。

**综合 1**：$X\sim U(0,1)$，求 $P(0.2<X<0.7)$，并说明为什么均匀分布适合做随机数源。

## 10. 参考答案

> [!success]- 基础 1
> $$\Phi(1.96)\approx0.975$$
>
> 这对应"95% 分位点"，双侧 95% 置信区间的临界值。

> [!success]- 基础 2
> $\sigma^2=25 \Rightarrow \sigma=5$。
>
> $Z=(15-10)/5=1$，$P(X<15)=\Phi(1)\approx0.8413$。

> [!success]- 进阶 1
> 平均寿命 $=1/0.1=10$ 年。
>
> $$P(X>3)=e^{-0.1\times3}=e^{-0.3}\approx0.740818$$
>
> **注意**：无记忆性意味着"已用 3 年"不改变未来寿命分布。

> [!success]- 综合 1
> $$P(0.2<X<0.7)=0.7-0.2=0.5$$
>
> 均匀分布的 CDF 是线性函数，区间概率等于区间长度；加上易于通过线性变换 $a+(b-a)U$ 生成任意区间均匀随机数，因此适合作为随机数源。

## 11. 常见错误

| 类型 | 错误表现 | 原因 | 纠正 |
|---|---|---|---|
| 概念错误 | 正态参数写成 $N(\mu,\sigma)$ | 第二参数是方差 | $N(\mu,\sigma^2)$ |
| 计算错误 | 不标准化就查表 | 用错表 | 先 $Z=(x-\mu)/\sigma$ |
| 概念错误 | 指数分布有记忆性 | 混淆 | 指数是无记忆的 |
| 计算错误 | 均匀方差写成 $(b-a)/12$ | 漏平方 | $(b-a)^2/12$ |
| 概念错误 | 认为正态只能取有限范围 | 定义域是全实数 | 只是概率集中在 $\mu$ 附近 |

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
- **前置知识**：[MATH-304 常见离散分布](MATH-304%20%E5%B8%B8%E8%A7%81%E7%A6%BB%E6%95%A3%E5%88%86%E5%B8%83.md)
- **后继知识**：[MATH-306 期望、方差与协方差](MATH-306%20%E6%9C%9F%E6%9C%9B%E3%80%81%E6%96%B9%E5%B7%AE%E4%B8%8E%E5%8D%8F%E6%96%B9%E5%B7%AE.md)
- **关联Python**：[PY-068 随机数、视图与副本](../../01_Python%E5%9F%BA%E7%A1%80/08%20NumPy/PY-068%20%E9%9A%8F%E6%9C%BA%E6%95%B0%E3%80%81%E8%A7%86%E5%9B%BE%E4%B8%8E%E5%89%AF%E6%9C%AC.md)
- **关联数学**：[MATH-304 常见离散分布](MATH-304%20%E5%B8%B8%E8%A7%81%E7%A6%BB%E6%95%A3%E5%88%86%E5%B8%83.md)
- **被引用为数学**：[MATH-306 期望、方差与协方差](MATH-306%20%E6%9C%9F%E6%9C%9B%E3%80%81%E6%96%B9%E5%B7%AE%E4%B8%8E%E5%8D%8F%E6%96%B9%E5%B7%AE.md) · [ML-027 朴素贝叶斯](../../03_%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/02%20%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0/ML-027%20%E6%9C%B4%E7%B4%A0%E8%B4%9D%E5%8F%B6%E6%96%AF.md) · [PY-068 随机数、视图与副本](../../01_Python%E5%9F%BA%E7%A1%80/08%20NumPy/PY-068%20%E9%9A%8F%E6%9C%BA%E6%95%B0%E3%80%81%E8%A7%86%E5%9B%BE%E4%B8%8E%E5%89%AF%E6%9C%AC.md)

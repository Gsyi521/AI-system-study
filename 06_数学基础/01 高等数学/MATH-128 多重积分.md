---
id: MATH-128
title: 多重积分
en_title: Multiple Integrals
domain: MATH
module: M-MATH-03
type: Technique
difficulty: L3 推导
status: planned
prerequisites: [MATH-127]
related_math: [MATH-127]
related_python: [PY-066]
related_ml: []
related_dl: []
related_llm: []
notion_page_id: 40c663e8-0333-8102-a6c2-c74f969a7a36
notion_url: https://app.notion.com/p/40c663e803338102a6c2c74f969a7a36
created: 2026-11-02
updated: 2026-11-02
notion_db: 知识点
学科: 数学
学习状态: 未开始
难度: L3 推导
知识类型: Technique
所属模块: [M-MATH-03]
前置知识: [MATH-127]
关联Python: [PY-066]
关联数学: [MATH-127]
tags: [数学, Technique]
---

# MATH-128 多重积分

## 1. 知识概览

| 项 | 内容 |
|---|---|
| 知识 ID | `MATH-128` |
| 英文名称 | Multiple Integrals |
| 所属模块 | M-MATH-03 微积分 |
| 难度等级 | L3 推导 |
| 前置知识 | [[MATH-127 Hessian 与多元展开]] |
| 所属知识体系 | 二重积分 → 累次积分 → 换序 → 三重积分 |
| **AI 应用场景** | 二维/高维概率分布的归一化、边缘分布、联合密度积分、期望 |
| **预计学习时间** | 50 分钟 |

**为什么学它**：连续型多维分布的概率是**多重积分**：$P((X,Y)\in D)=\iint_D f(x,y)\,dx\,dy$。要求边缘分布，就得对另一个变量"积掉"——这就是多重积分的直接用途。

## 2. 直观理解

一重积分求**面积**，二重积分求**体积**——曲面 $z=f(x,y)$ 与 $xy$ 平面之间那块立体的体积。

**怎么算**：切成小方块，每块高度近似 $f$，体积 $\approx f\cdot\Delta x\Delta y$，加起来取极限。

**实际做法（累次积分）**：先固定 $x$，对 $y$ 积一次（得到一个关于 $x$ 的函数），再对 $x$ 积一次。**像切面包：先切条，再切块。**

## 3. 正式数学定义

**二重积分**：在区域 $D$ 上

$$
\iint_D f(x,y)\,dA = \lim_{\|\Delta\|\to0}\sum_{i,j} f(\xi_i,\eta_j)\,\Delta x_i\Delta y_j
$$

**累次积分**（矩形区域 $[a,b]\times[c,d]$）

$$
\iint_D f\,dA = \int_a^b\left(\int_c^d f(x,y)\,dy\right)dx
$$

**Fubini 定理**：若 $f$ 在矩形上连续，则两种积分顺序结果相同

$$
\int_a^b\int_c^d f\,dy\,dx = \int_c^d\int_a^b f\,dx\,dy
$$

**三重积分**

$$
\iiint_V f\,dV = \int\!\!\int\!\!\int_V f(x,y,z)\,dx\,dy\,dz
$$

**概率上的应用**：联合密度 $f(x,y)$ 的边缘密度

$$
f_X(x) = \int_{-\infty}^{+\infty} f(x,y)\,dy
$$

## 4. 数学推导

### 4.1 为什么累次积分等于二重积分（矩形情形）

**起点**：把 $[a,b]$ 分成 $m$ 份、$[c,d]$ 分成 $n$ 份，形成 $mn$ 个小矩形。

**第 1 步**：二重积分的黎曼和

$$
S = \sum_{i=1}^{m}\sum_{j=1}^{n} f(\xi_i,\eta_j)\,\Delta x_i\,\Delta y_j
$$

**第 2 步**：把内层求和提出来

$$
S = \sum_{i=1}^{m}\left(\sum_{j=1}^{n} f(\xi_i,\eta_j)\,\Delta y_j\right)\Delta x_i
$$

**第 3 步**：括号里正是"固定 $x=\xi_i$ 时对 $y$ 的黎曼和"，当 $\Delta y\to0$ 时趋于

$$
g(\xi_i) = \int_c^d f(\xi_i,y)\,dy
$$

**第 4 步**：再对 $x$ 取极限

$$
\iint_D f\,dA = \int_a^b g(x)\,dx = \int_a^b\left(\int_c^d f\,dy\right)dx
$$

**结论**：二重积分可以拆成两次一重积分，顺序可换（Fubini）。

### 4.2 边缘密度为什么是"积掉另一个变量"

**起点**：联合密度 $f(x,y)$，要求 $X$ 的密度 $f_X$。

**第 1 步**：$P(X\le x)$ 是"$X\le x$、$Y$ 任意"的概率

$$
F_X(x) = \int_{-\infty}^{x}\int_{-\infty}^{+\infty} f(u,y)\,dy\,du
$$

**第 2 步**：两边对 $x$ 求导（用变上限积分求导）

$$
f_X(x) = \frac{d}{dx}F_X(x) = \int_{-\infty}^{+\infty} f(x,y)\,dy
$$

**结论**：**"求边缘" = "对不要的变量积分消掉"**。这条在贝叶斯与生成模型里反复出现。

## 5. 几何直观与图解

```mermaid
flowchart LR
    A["区域 D"] --> B["切成小方块 Δx·Δy"]
    B --> C["每块体积 ≈ f·Δx·Δy"]
    C --> D["求和取极限 = 二重积分"]
    D --> E["=∑ 先对 y 积，再对 x 积"]
```

```text
     z
     │   ╱▔▔╲
     │  ╱    ╲   曲面
     │ ╱______╲
     └──────────── y
    ╱
   x    投影区域 D 上方的体积
```

## 6. 计算示例

**例 1**：$\iint_D xy\,dA$，$D=[0,1]\times[0,1]$。

$$
\int_0^1\left(\int_0^1 xy\,dy\right)dx = \int_0^1 x\cdot\frac12\,dx = \frac12\cdot\frac12 = \frac14
$$

**例 2**：$\iint_D (x+y)\,dA$，$D=[0,1]\times[0,2]$。

$$
\int_0^1\left(\int_0^2 (x+y)\,dy\right)dx = \int_0^1 \left[xy+\frac{y^2}{2}\right]_0^2 dx = \int_0^1 (2x+2)\,dx = 1+2 = 3
$$

**例 3**：$\iint_D (x^2+y^2)\,dA$，$D=[0,1]^2$。

$$
\int_0^1\left(\frac13 + y^2\right)dy = \frac13 + \frac13 = \frac23
$$

## 7. Python 辅助验证

**依赖**：NumPy 2.5.4（网格平均法近似二重积分）。

```powershell
python math128_multiple.py
```

```python
import numpy as np


def double_integral(fn, ax, bx, ay, by, n=400):
    x = np.linspace(ax, bx, n + 1)
    y = np.linspace(ay, by, n + 1)
    X, Y = np.meshgrid(x, y)
    Z = fn(X, Y)
    return Z.mean() * (bx - ax) * (by - ay)


print("∬ x·y on [0,1]²:", round(float(double_integral(lambda X, Y: X * Y, 0, 1, 0, 1)), 6))
print("∬ x+y on [0,1]×[0,2]:",
      round(float(double_integral(lambda X, Y: X + Y, 0, 1, 0, 2)), 6))
print("∬ x²+y² on [0,1]²:", round(float(double_integral(lambda X, Y: X**2 + Y**2, 0, 1, 0, 1)), 6),
      "解析 2/3 =", round(2 / 3, 6))
```

**实际运行结果**

```
∬ x·y on [0,1]²: 0.25
∬ x+y on [0,1]×[0,2]: 3.0
∬ x²+y² on [0,1]²: 0.6675
解析 2/3 = 0.666667
```

**代码解析**

- 网格平均法：`Z.mean()` 乘上区域面积，等价于**中点黎曼和**。
- 前两个结果精确到小数点后 6 位，因为被积函数是低阶多项式。
- 第三个 0.6675 vs 0.666667 差约 $8\times10^{-4}$——这是**数值方法本身的误差**，不是算错。误差随 $n$ 增大而减小（$O(1/n)$ 量级）。
- 要更高精度：增大 $n$、改用 Simpson 法，或直接解析积分。

## 8. 知识关联

| 类型 | 链接 |
|---|---|
| 前置数学 | [[MATH-127 Hessian 与多元展开]]、[[MATH-125 定积分与微积分基本定理]] |
| 后续数学 | `MATH-303 随机变量与分布函数`、`MATH-306 期望、方差与协方差`（待建） |
| 相关 Python | [[PY-066 矩阵运算]]、[[PY-067 聚合与统计]] |
| 相关机器学习 | 待关联（二维密度的归一化、边缘化） |
| 相关深度学习 | 待关联（高维积分的近似，如变分推断） |
| 相关大语言模型 | 待关联（配分函数） |

## 9. 实战练习

**基础 1**：求 $\iint_D 1\,dA$，$D=[0,2]\times[0,3]$。

**基础 2**：求 $\iint_D x\,dA$，$D=[0,1]\times[0,1]$。

**进阶 1**：求 $\iint_D xy\,dA$，$D=[0,2]\times[0,1]$。

**综合 1**：联合密度 $f(x,y)=4xy$（$0\le x,y\le1$），求边缘密度 $f_X(x)$，并验证它积分等于 1。

## 10. 参考答案

> [!success]- 基础 1
> $$\iint_D 1\,dA = \text{面积} = 2\times3 = 6$$

> [!success]- 基础 2
> $$\int_0^1\!\!\int_0^1 x\,dy\,dx = \int_0^1 x\,dx = \frac12$$

> [!success]- 进阶 1
> $$\int_0^2\!\!\int_0^1 xy\,dy\,dx = \int_0^2 x\cdot\frac12\,dx = \frac12\cdot\frac{4}{2} = 1$$

> [!success]- 综合 1
> $$f_X(x) = \int_0^1 4xy\,dy = 4x\cdot\frac12 = 2x \quad (0\le x\le1)$$
>
> 验证：$\int_0^1 2x\,dx = 1$ ✓
>
> **要点**：边缘密度就是把另一个变量从 $0$ 到 $1$ 积掉；$\int_0^1 4xy\,dy$ 里 $x$ 当常数提出来。

## 11. 常见错误

| 类型 | 错误表现 | 原因 | 纠正 |
|---|---|---|---|
| 计算错误 | 内层积分把外层变量也积掉 | 变量混淆 | 对 $y$ 积时 $x$ 是常数 |
| 概念错误 | 认为积分顺序不能换 | 不了解 Fubini | 连续函数可换序 |
| 计算错误 | 积分限写错 | 未看清区域 | 先画区域再定限 |
| 概念错误 | 把二重积分当"两次独立积分相乘" | 只有可分离函数才行 | $f=g(x)h(y)$ 时才拆 |
| 数值错误 | 网格太粗就下结论 | 截断误差 | 增大 $n$ 或换方法 |

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
| 知识类型 | Technique |

- **所属模块**：[M-MATH-03 微积分](../%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-MATH-03%20%E5%BE%AE%E7%A7%AF%E5%88%86.md)
- **前置知识**：[MATH-127 Hessian 与多元展开](MATH-127%20Hessian%20%E4%B8%8E%E5%A4%9A%E5%85%83%E5%B1%95%E5%BC%80.md)
- **关联Python**：[PY-066 矩阵运算](../../01_Python%E5%9F%BA%E7%A1%80/08%20NumPy/PY-066%20%E7%9F%A9%E9%98%B5%E8%BF%90%E7%AE%97.md)
- **关联数学**：[MATH-127 Hessian 与多元展开](MATH-127%20Hessian%20%E4%B8%8E%E5%A4%9A%E5%85%83%E5%B1%95%E5%BC%80.md)

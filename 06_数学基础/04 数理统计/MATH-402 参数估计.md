---
id: MATH-402
title: 参数估计
en_title: Parameter Estimation
domain: MATH
module: M-MATH-04
type: Technique
difficulty: L3 推导
status: planned
prerequisites: [MATH-401]
related_math: [MATH-401]
related_python: [PY-067]
related_ml: []
related_dl: []
related_llm: []
notion_page_id: 40c663e8-0333-817a-ac4e-d627e4b99002
notion_url: https://app.notion.com/p/40c663e80333817aac4ed627e4b99002
created: 2026-11-02
updated: 2026-11-02
notion_db: 知识点
学科: 数学
学习状态: 未开始
难度: L3 推导
知识类型: Technique
所属模块: [M-MATH-04]
前置知识: [MATH-401]
后继知识: [MATH-403]
关联Python: [PY-067]
关联数学: [MATH-401]
被引用为数学: [MATH-403]
tags: [数学, Technique]
---

# MATH-402 参数估计

## 1. 知识概览

| 项 | 内容 |
|---|---|
| 知识 ID | `MATH-402` |
| 英文名称 | Parameter Estimation |
| 所属模块 | M-MATH-04 概率论与数理统计 |
| 难度等级 | L3 推导 |
| 前置知识 | [[MATH-401 大数定律与中心极限定理]] |
| 所属知识体系 | 点估计 → 无偏性 → 标准误 → 区间估计 |
| **AI 应用场景** | 用训练集估计模型参数、评估指标的不确定性、置信区间报告 |
| **预计学习时间** | 55 分钟 |

**为什么学它**：训练模型就是**参数估计**——用数据估计 $\theta$。但估计值本身有随机性，所以要看：是否有偏（bias）、方差多大、给出区间而不是一个点。

## 2. 直观理解

**点估计**：给一个数，如"平均身高 172 cm"。

**区间估计**：给一个范围，如"平均身高 172 ± 1.5 cm（95% 置信）"。

只报一个数的问题是：**换个样本，数就变了**。区间把这种不确定性说出来。

**无偏**：反复抽样时，估计值的平均正好等于真值。
**有偏**：系统性地偏高或偏低（比如样本方差除以 $n$ 就偏低）。

## 3. 正式数学定义

**点估计量**：$\hat\theta = \hat\theta(X_1,\dots,X_n)$，是样本的函数（随机变量）。

**无偏性**

$$
\mathbb{E}[\hat\theta] = \theta
$$

**样本均值与样本方差**

$$
\bar X = \frac1n\sum X_i,\qquad S^2 = \frac{1}{n-1}\sum (X_i-\bar X)^2
$$

$$
\mathbb{E}[\bar X]=\mu,\qquad \mathbb{E}[S^2]=\sigma^2
$$

**标准误**：$\operatorname{SE}(\bar X)=\dfrac{\sigma}{\sqrt n}$，实践中用 $S/\sqrt n$ 代替。

**置信区间（$\sigma$ 已知）**

$$
\bar X \pm z_{1-\alpha/2}\cdot\frac{\sigma}{\sqrt n}
$$

95% 时 $z_{0.975}=1.96$。

**评价指标**：均方误差

$$
\operatorname{MSE}(\hat\theta) = \operatorname{Bias}^2 + \operatorname{Var}
$$

## 4. 数学推导

### 4.1 为什么除以 $n$ 的方差是有偏的

**起点**：$\hat\sigma^2=\frac1n\sum(X_i-\bar X)^2$。

**第 1 步**：展开离差平方和

$$
\sum(X_i-\bar X)^2 = \sum(X_i-\mu)^2 - n(\bar X-\mu)^2
$$

（把 $\bar X$ 当作参照点展开，交叉项为零）

**第 2 步**：取期望

$$
\mathbb{E}\left[\sum(X_i-\bar X)^2\right] = n\sigma^2 - n\cdot\frac{\sigma^2}{n} = (n-1)\sigma^2
$$

这里用了 $\operatorname{Var}(\bar X)=\sigma^2/n$。

**第 3 步**：除以 $n$

$$
\mathbb{E}[\hat\sigma^2] = \frac{n-1}{n}\sigma^2 < \sigma^2
$$

**第 4 步**：除以 $n-1$ 修正

$$
\mathbb{E}[S^2] = \sigma^2
$$

**结论**：**除以 $n$ 会系统性低估方差**，因为用 $\bar X$ 代替 $\mu$ 已经"用掉"了一个自由度。

### 4.2 置信区间为什么是 $\bar X\pm1.96\sigma/\sqrt n$

**起点**：由 CLT，$\dfrac{\bar X-\mu}{\sigma/\sqrt n}\approx N(0,1)$。

**第 1 步**：取 95% 的临界值

$$
P\left(-1.96<\frac{\bar X-\mu}{\sigma/\sqrt n}<1.96\right)\approx0.95
$$

**第 2 步**：解不等式得

$$
P\left(\bar X-1.96\frac{\sigma}{\sqrt n}<\mu<\bar X+1.96\frac{\sigma}{\sqrt n}\right)\approx0.95
$$

**结论**：区间随样本变化而移动，**95% 的区间会覆盖真值**。注意这不是"$\mu$ 有 95% 概率落在某个固定区间"——$\mu$ 是常数，随机的是区间。

## 5. 几何直观与图解

```mermaid
flowchart LR
    A["样本"] --> B["点估计 θ̂"]
    B --> C["算标准误 SE"]
    C --> D["区间 θ̂ ± 1.96·SE"]
    D --> E["报告：估计值 + 不确定性"]
```

```text
重复抽样，每次画一个区间：
  ─────●─────    覆盖 μ ✓
    ─────●─────  覆盖 μ ✓
  ──●──────────  未覆盖 ✗（约 5%）
               μ
```

## 6. 计算示例

**例**：正态数据，真值 $\mu=5$、$\sigma=2$，抽 $n=10000$。

$$
\bar x \approx 5.0298,\quad \hat\sigma^2_{n} \approx 4.0367,\quad S^2 \approx 4.0371
$$

- 用 $S^2$ 估计 $\sigma^2=4$：偏差 $0.0371$（在抽样误差内）。
- 标准误 $S/\sqrt n = 0.0201$，理论 $2/100=0.02$。
- 95% CI：$5.0298 \pm 1.96\times0.0201 \approx [4.990, 5.069]$，包含真值 5 ✓

**对比有偏与无偏**：$n$ 大时两者差别很小（差因子 $\frac{n}{n-1}$）；但 $n=10$ 时相差 10%，不容忽略。

## 7. Python 辅助验证

**依赖**：NumPy 2.5.4。

```powershell
python math402_estimation.py
```

```python
import numpy as np

rng = np.random.default_rng(2)

d = rng.binomial(1, 0.3, size=1000)
print("伯努利 p 的 MLE（样本频率）:", round(float(d.mean()), 4), "（真值 0.3）")

x = rng.normal(5, 2, size=10000)
print("均值估计:", round(float(x.mean()), 4), "（真值 5）")
print("方差（除以 n）:", round(float(x.var()), 4))
print("方差（除以 n-1）:", round(float(x.var(ddof=1)), 4), "（真值 4）")
print("标准误:", round(float(x.std(ddof=1) / np.sqrt(len(x))), 4), "（理论 0.02）")
```

**实际运行结果**

```
伯努利 p 的 MLE（样本频率）: 0.293 （真值 0.3）
均值估计: 5.0298 （真值 5）
方差（除以 n）: 4.0367
方差（除以 n-1）: 4.0371 （真值 4）
标准误: 0.0201 （理论 0.02）
```

**代码解析**

- 伯努利参数的 MLE 就是**样本频率** $\bar x$。
- `x.var()` 是除以 $n$ 的有偏估计，`x.var(ddof=1)` 是除以 $n-1$ 的无偏估计。
- 两者差 $0.0004$，因为 $n=10000$ 时 $\frac{n}{n-1}\approx1.0001$。
- 标准误 $0.0201$ 与理论 $0.02$ 吻合，验证 $\operatorname{SE}=\sigma/\sqrt n$。

## 8. 知识关联

| 类型 | 链接 |
|---|---|
| 前置数学 | [[MATH-401 大数定律与中心极限定理]]、[[MATH-306 期望、方差与协方差]] |
| 后续数学 | `MATH-403 假设检验基础`（待建）、`MATH-404 最大似然估计`（待建）、[[MATH-451 最小二乘与回归统计]] |
| 相关 Python | [[PY-067 聚合与统计]] |
| 相关机器学习 | 待关联（训练集/验证集、指标置信区间） |
| 相关深度学习 | 待关联（多次随机种子的方差报告） |
| 相关大语言模型 | 待关联（评测结果的不确定性） |

## 9. 实战练习

**基础 1**：样本 $2,4,6,8$，求均值与无偏方差。

**基础 2**：$\bar x=100$、$S=10$、$n=25$，求 95% 置信区间。

**进阶 1**：解释为什么 $n$ 很小时"除以 $n$"的方差估计不能接受。

**综合 1**：某模型在测试集上准确率 0.85，测试集 1000 条。求准确率的 95% 置信区间（提示：伯努利方差 $p(1-p)$）。

## 10. 参考答案

> [!success]- 基础 1
> $$\bar x = \frac{2+4+6+8}{4} = 5$$
>
> 离差平方和 $= 9+1+1+9 = 20$，所以
>
> $$S^2 = \frac{20}{4-1} = \frac{20}{3} \approx 6.6667$$

> [!success]- 基础 2
> $$\operatorname{SE} = \frac{10}{\sqrt{25}} = 2$$
>
> $$100 \pm 1.96\times2 = [96.08,\ 103.92]$$

> [!success]- 进阶 1
> 有偏因子是 $\frac{n-1}{n}$：$n=10$ 时低估 10%，$n=5$ 时低估 20%。
>
> 小样本下这个偏差与抽样误差同量级，会导致系统性低估波动，进而让置信区间偏窄、检验过于乐观。

> [!success]- 综合 1
> $\hat p=0.85$，$\operatorname{SE}=\sqrt{\hat p(1-\hat p)/n}=\sqrt{0.85\times0.15/1000}=\sqrt{0.0001275}\approx0.0113$。
>
> $$0.85 \pm 1.96\times0.0113 \approx [0.828,\ 0.872]$$
>
> 即约 **82.8%–87.2%**。报告准确率时给区间比给单点更可信。

## 11. 常见错误

| 类型 | 错误表现 | 原因 | 纠正 |
|---|---|---|---|
| 概念错误 | 认为置信区间是"$\mu$ 的概率" | 混淆随机对象 | 随机的是区间 |
| 计算错误 | 方差除以 $n$ 当无偏 | 记混 | 无偏用 $n-1$ |
| 概念错误 | 认为 $n$ 大就无偏 | 无偏是估计量性质 | 与 $n$ 无关（渐近无偏是另一回事） |
| 概念错误 | 只看点估计不报不确定性 | 丢失信息 | 必须给 SE 或区间 |
| 计算错误 | 标准误用 $\sigma$ 不用 $\sigma/\sqrt n$ | 漏根号 | 除以 $\sqrt n$ |

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

- **所属模块**：[M-MATH-04 概率论与数理统计](../%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-MATH-04%20%E6%A6%82%E7%8E%87%E8%AE%BA%E4%B8%8E%E6%95%B0%E7%90%86%E7%BB%9F%E8%AE%A1.md)
- **前置知识**：[MATH-401 大数定律与中心极限定理](MATH-401%20%E5%A4%A7%E6%95%B0%E5%AE%9A%E5%BE%8B%E4%B8%8E%E4%B8%AD%E5%BF%83%E6%9E%81%E9%99%90%E5%AE%9A%E7%90%86.md)
- **后继知识**：[MATH-403 假设检验基础](MATH-403%20%E5%81%87%E8%AE%BE%E6%A3%80%E9%AA%8C%E5%9F%BA%E7%A1%80.md)
- **关联Python**：[PY-067 聚合与统计](../../01_Python%E5%9F%BA%E7%A1%80/08%20NumPy/PY-067%20%E8%81%9A%E5%90%88%E4%B8%8E%E7%BB%9F%E8%AE%A1.md)
- **关联数学**：[MATH-401 大数定律与中心极限定理](MATH-401%20%E5%A4%A7%E6%95%B0%E5%AE%9A%E5%BE%8B%E4%B8%8E%E4%B8%AD%E5%BF%83%E6%9E%81%E9%99%90%E5%AE%9A%E7%90%86.md)
- **被引用为数学**：[MATH-403 假设检验基础](MATH-403%20%E5%81%87%E8%AE%BE%E6%A3%80%E9%AA%8C%E5%9F%BA%E7%A1%80.md)

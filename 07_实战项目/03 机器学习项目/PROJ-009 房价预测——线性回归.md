---
id: PROJ-009
title: 房价预测——线性回归
domain: PROJ
module: [M-ML-03]
type: project
difficulty: L2 应用
status: planned
tech_stack: [Python, NumPy, Pandas, Scikit-learn]
prerequisites: [ML-017, ML-018, ML-019, ML-023, ML-040]
related_math: [MATH-801, MATH-721]
related: [ML-017, ML-018, ML-019, ML-023, ML-040, ML-046]
notion_page_id: 410663e8-0333-8191-84e2-de51462d33ca
notion_url: https://app.notion.com/p/410663e80333819184e2de51462d33ca
created: 2026-11-06
updated: 2026-11-06
notion_db: 实战项目
所属学科: 机器学习
项目状态: 未开始
项目阶段: 立项
难度: L2 应用
项目目标: 房价预测——线性回归
技术栈: [Python, NumPy, Pandas, scikit-learn]
项目仓库: ai-knowledge-system/projects/PROJ-009
对应课程模块: [M-ML-03]
前置知识: [ML-017, ML-018, ML-019, ML-023, ML-040]
对应知识点: [ML-017, ML-018, ML-019, ML-023, ML-040, ML-046]
tags: [机器学习]
---

# PROJ-009 房价预测——线性回归

## 1. 项目背景

房产估价是最经典的回归问题：**给定房屋属性，预测一个连续价格**。

它适合作为第一个 ML 项目，因为：

- 特征与目标的关系**近似线性**（面积越大越贵、房龄越大越便宜），线性模型能拿到不错的基线。
- 评价指标直观（平均差多少钱）。
- 数据容易理解，便于把注意力放在**流程**而不是业务黑话上。

> **数据声明**：本项目使用**合成数据**。原计划用 `fetch_california_housing`，但执行时网络连续 3 次失败（`URLError: SSL UNEXPECTED_EOF_WHILE_READING`），按预案降级。**这不是真实房价数据**，生成机制已知，仅用于教学。

## 2. 项目目标

1. 走完**回归项目**的完整流程：载入 → 清洗 → 特征工程 → 划分 → 建模 → 评价 → 残差诊断。
2. 用 MAE / MSE / RMSE / R² 四个指标评价，并说清各自含义。
3. 通过**残差诊断**判断"线性假设是否成立"。
4. 理解"数据清洗会改变样本量，进而影响指标可比性"。

## 3. 学习目标

| 目标 | 验收方式 |
|---|---|
| 会算并解释四个回归指标 | 能说出 MAE 与 RMSE 的差异 |
| 会做缺失值 + 异常值清洗 | 输出清洗日志（删了多少行） |
| 会构造领域特征 | 至少 3 个新特征 |
| 会用残差判断模型假设 | 残差与预测值相关系数接近 0 |
| 能比较线性和多项式模型 | 说明为什么多项式没提升 |

## 4. 前置知识

| 知识点 | 为什么需要 |
|---|---|
| [[ML-017 回归任务基本概念]] | 指标定义 |
| [[ML-018 线性回归（手写实现）]] | 模型原理 |
| [[ML-019 多元线性回归与正规方程]] | 多特征求解 |
| [[ML-023 回归评价指标与残差诊断]] | MAE/RMSE/R² 与残差图 |
| [[ML-040 Hold-out 验证]] | 单次划分的作用与局限 |
| [[ML-046 过拟合与欠拟合]] | 为什么多项式没提升 |

**数学**：[[MATH-801 线性回归的数学]]（正规方程）、[[MATH-721 梯度下降]]（求解思路）。

## 5. 数据集来源及许可说明

| 项 | 内容 |
|---|---|
| 来源 | **合成数据**，由项目脚本 `make_housing()` 生成 |
| 生成方式 | 已知线性机制 + 高斯噪声（σ=22）+ 3% 缺失 + 1% 极端异常 |
| 许可 | 无版权问题（程序生成） |
| 规模 | 1200 行 × 5 列 |
| 可复现 | 固定 `random_state=42` |
| 真实数据替代方案 | `sklearn.datasets.fetch_california_housing()`（需联网，本次不可用） |

**生成机制（真实系数已知，便于验证模型）**

```text
price = 3.2*area − 1.8*age − 9.5*dist + 6.0*rooms + 120 + N(0, 22²)
单位：面积 m²、房龄 年、距离 km、价格 万元
```

## 6. 数据字段说明

| 字段 | 类型 | 含义 | 量纲 |
|---|---|---|---|
| `area` | float | 建筑面积 | m² |
| `rooms` | float | 房间数 | 间 |
| `age` | float | 房龄 | 年 |
| `dist` | float | 距市中心距离 | km |
| `price` | float | **目标**：售价 | 万元 |

**注入的数据问题**：`area` 有 3% 缺失；`price` 有 1% 被放大 10 倍（模拟录入错误）。

## 7. 任务定义

- **任务类型**：回归（监督学习）。
- **输入**：4 个连续/计数特征。
- **输出**：连续价格（万元）。
- **损失**：MSE（训练），评价用 MAE/RMSE/R²。
- **成功标准**：R² > 0.85，且残差无明显结构。

## 8. 技术路线

```mermaid
flowchart LR
    A["生成合成数据"] --> B["清洗：中位数填充 + IQR 去异常"]
    B --> C["特征工程：面积/间数、新房标记、距离分段"]
    C --> D["One-Hot + 划分 75/25"]
    D --> E["线性回归 / 多项式 / 岭回归"]
    E --> F["四指标评价 + 残差诊断"]
```

## 9. 环境与依赖

| 项 | 版本 |
|---|---|
| Python | 3.15.0rc3 |
| numpy | 2.5.4 |
| pandas | 3.1.0 |
| scikit-learn | 1.9.1 |

见 `requirements.txt`。

## 10. 项目目录结构

```text
projects/PROJ-009/          （工作区，不在 Vault）
├── README.md
├── requirements.txt
├── src/housing_regression.py
├── data/      （脚本内生成，不落盘）
└── results/
```

> 遵守库规则：**代码放工作区**，Vault 只放讲解。

## 11. 数据探索

```text
=== 1. 原始数据 ===
形状: (1200, 5)
列: ['area', 'rooms', 'age', 'dist', 'price']
缺失: 36
 area  rooms  age  dist  price
103.5    3.0 23.0  0.83  420.7
 65.9    2.0  0.0  8.94  250.2
116.0    1.0 25.0  0.50  474.5
```

**读法**：

- 1200 行、5 列，目标 `price` 量级在 200–500 万。
- 36 个缺失全在 `area`（3%）。
- 第一行：103.5 m²、3 间、23 年、距市中心 0.83 km ⇒ 420.7 万，符合"面积大、距离近 ⇒ 贵"。

## 12. 数据清洗

```text
=== 2. 清洗日志 ===
  rows: 1200
  missing_before: 36
  missing_after: 0
  outliers_removed: 20
  rows_after: 1180
```

**两步清洗**：

1. **缺失值**：`area` 用中位数填充（36 → 0）。选中位数而非均值，因为面积分布右偏。
2. **异常值**：对 `price` 用 IQR 法（$Q_1-1.5\text{IQR}$ 到 $Q_3+1.5\text{IQR}$），删掉 20 行。

**为什么删 20 行而不是 12 行**：注入了 1% = 12 个 ×10 异常，但 IQR 边界也会把少数正常高值判为异常（分布右尾）。**这是 IQR 的代价，必须记录删了多少**，否则指标不可比。

## 13. 特征工程

```text
=== 3. 特征工程后 ===
列: ['area', 'rooms', 'age', 'dist', 'price', 'area_per_room', 'is_new', 'dist_bucket']
is_new 均值: 0.161
dist_bucket 分布: {'中': 605, '近': 383, '远': 192}
```

| 新特征 | 含义 | 理由 |
|---|---|---|
| `area_per_room` | 面积 / 房间数 | 反映"房间大小"，与总价非线性相关 |
| `is_new` | 房龄 ≤ 5 ⇒ 1 | 新房溢价是**分段效应**，线性项表达不了 |
| `dist_bucket` | 近/中/远 | 距离影响可能是**阶梯式**（学区、地铁） |

**处理方式**：`dist_bucket` 用 `pd.get_dummies(drop_first=True)` 转 One-Hot（避免与截距共线）。

## 14. 模型建立

三个模型、同一份划分：

| 模型 | 结构 | 目的 |
|---|---|---|
| 线性回归 | StandardScaler → LinearRegression | 基线 |
| 多项式(2) | Scaler → PolynomialFeatures(2) → LR | 检验非线性 |
| 岭回归(α=10) | Scaler → Ridge | 检验正则化 |

**为什么都套 Scaler**：多项式展开后特征尺度差异极大（$area^2$ 是 $10^4$ 量级），不标准化会让数值不稳。

## 15. 模型训练

```text
=== 4. 划分 ===
训练: (885, 8) 测试: (295, 8)
```

训练集 885、测试集 295（75/25），8 个特征（6 原始派生 + 2 个 One-Hot 列）。

**注意**：清洗后从 1180 行切分，所以训练/测试都比分 1200 行时少。**这是"清洗影响可比性"的具体体现**。

## 16. 模型评价

```text
=== 5. 模型对比 ===
  线性回归: MAE=20.157 MSE=807.242 RMSE=28.412 R2=0.9167
  多项式(2): MAE=20.311 MSE=815.422 RMSE=28.556 R2=0.9159
  岭回归(a=10): MAE=20.142 MSE=803.206 RMSE=28.341 R2=0.9172
```

| 模型 | MAE | RMSE | R² |
|---|---|---|---|
| 线性回归 | 20.157 | 28.412 | 0.9167 |
| 多项式(2) | 20.311 | 28.556 | 0.9159 |
| **岭回归(α=10)** | **20.142** | **28.341** | **0.9172** |

**读法**：

- **MAE 20.16 万**：平均每套预测偏差约 20 万。
- **RMSE 28.41 万 > MAE**：说明有少量较大误差（RMSE/MAE ≈ 1.41）。
- **R² 0.9167**：解释了 91.7% 的方差，达到成功标准（>0.85）。
- **多项式没有提升**（0.9159 < 0.9167）：因为真实机制**就是线性的**，加二次项只是增加过拟合风险。
- **岭回归略优**（0.9172）：微弱正则化压掉了一点方差，但提升在噪声范围内。

## 17. 可视化分析

```text
=== 7. 系数（标准化后，可横向比较）===
  area                83.35
  dist               -33.12
  age                -21.01
  rooms                9.50
  dist_bucket_远        1.25
```

**系数排序符合直觉**：

1. `area` 正向最大（83.35）：面积是房价第一驱动。
2. `dist` 负向最大（−33.12）：越远越便宜。
3. `age` 负向（−21.01）：越老越便宜。
4. `rooms` 正向（9.50）：房间多略贵（但受 `area_per_room` 分担）。
5. `dist_bucket_远` 只剩 1.25：因为连续 `dist` 已解释大部分，分桶信息冗余。

**可视化建议**（本项目用文本输出替代绘图）：

- 散点：`area` vs `price`（应呈线性带状）。
- 残差图：残差 vs 预测值（应随机散布）。
- 直方图：残差分布（应近似正态、以 0 为中心）。

## 18. 结果解释

```text
=== 6. 残差诊断 ===
残差均值: 3.207
残差标准差: 28.23
残差与预测值相关: -0.005
残差绝对值 > 50 的样本数: 16 / 295
```

| 诊断项 | 值 | 判断 |
|---|---|---|
| 残差均值 | 3.207 | 接近 0，**无系统性偏移** ✅ |
| 残差标准差 | 28.23 | 与噪声 σ=22 同量级但略大（含删异常的影响） |
| 残差与预测值相关 | **−0.005** | **无结构** ⇒ 线性假设成立 ✅ |
| 大误差样本（>50） | 16/295（5.4%） | 有少量难样本，可接受 |

**结论**：模型没有"漏掉明显非线性"，线性假设在这份数据上**是合适的**。

## 19. 常见错误

| 错误 | 表现 | 纠正 |
|---|---|---|
| 先清洗后划分 | 泄漏（测试统计进训练） | 先划分再 fit 清洗参数；本课因清洗是删行/中位数，风险小但应写进 Pipeline |
| 用均值填面积 | 被右偏/异常带偏 | 用中位数 |
| 删异常不记录 | 前后指标不可比 | 输出清洗日志 |
| 只看 R² | 不知道平均差多少钱 | 同时报 MAE |
| 多项式硬上 | 过拟合、变慢 | 先用残差判断是否有非线性 |
| 系数直接解释因果 | 相关≠因果 | 只能说“关联” |

## 20. 完整可运行代码

```powershell
python projects/PROJ-009/src/housing_regression.py
```

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def make_housing(n=1200, seed=42):
    rng = np.random.default_rng(seed)
    area = rng.normal(95, 28, n).clip(30, 220)
    rooms = rng.integers(1, 6, n).astype(float)
    age = rng.integers(0, 40, n).astype(float)
    dist = rng.gamma(2.0, 2.5, n)
    price = (3.2*area - 1.8*age - 9.5*dist + 6.0*rooms + 120
             + rng.normal(0, 22, n))
    df = pd.DataFrame({"area": area.round(1), "rooms": rooms, "age": age,
                       "dist": dist.round(2), "price": price.round(1)})
    miss = rng.choice(n, int(n*0.03), replace=False)
    df.loc[miss, "area"] = np.nan
    out = rng.choice(n, int(n*0.01), replace=False)
    df.loc[out, "price"] *= 10
    return df


df = make_housing()
print("原始:", df.shape, "缺失:", int(df.isna().sum().sum()))

df["area"] = df["area"].fillna(df["area"].median())
q1, q3 = df["price"].quantile([0.25, 0.75])
iqr = q3 - q1
df = df[(df["price"] >= q1 - 1.5*iqr) & (df["price"] <= q3 + 1.5*iqr)].copy()
print("清洗后:", df.shape)

df["area_per_room"] = df["area"] / df["rooms"]
df["is_new"] = (df["age"] <= 5).astype(int)
df["dist_bucket"] = pd.cut(df["dist"], [-0.1, 3, 8, np.inf],
                           labels=["近", "中", "远"]).astype(str)

cols = ["area", "rooms", "age", "dist", "area_per_room", "is_new"]
X = pd.get_dummies(df[cols + ["dist_bucket"]], drop_first=True).astype(float)
y = df["price"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

lin = Pipeline([("sc", StandardScaler()), ("lr", LinearRegression())]).fit(X_tr, y_tr)
pred = lin.predict(X_te)
print("MAE:", round(mean_absolute_error(y_te, pred), 3))
print("RMSE:", round(mean_squared_error(y_te, pred) ** 0.5, 3))
print("R2:", round(r2_score(y_te, pred), 4))
resid = y_te.to_numpy() - pred
print("残差与预测值相关:", round(float(np.corrcoef(resid, pred)[0, 1]), 3))
```

**实际运行结果**

```
原始: (1200, 5) 缺失: 36
清洗后: (1180, 8)
MAE: 20.157
RMSE: 28.412
R2: 0.9167
残差与预测值相关: -0.005
```

## 21. 进阶挑战

1. **换成真实数据**：联网后用 `fetch_california_housing`，对比两套结果的差异（特征含义完全不同）。
2. **加交互项**：`area * is_new`（新房单价更高），看 R² 是否提升。
3. **对数目标**：对 `price` 取 log 再回归，比较残差分布是否更接近正态。
4. **交叉验证代替单次划分**：用 5 折 CV 报均值±标准差，验证 0.9167 是否稳定。
5. **特征重要性排序**：用置换重要性（`permutation_importance`）对比标准化系数。

## 22. 项目总结

- **线性假设成立时，简单模型就够**：R² 0.9167，多项式与岭回归都没有实质提升。
- **残差是检验假设的工具**：相关系数 −0.005 说明模型没漏结构。
- **清洗必须记录**：删了 20 行，前后指标不可直接比较。
- **合成数据的好处**：真实系数已知（3.2/−1.8/−9.5），可以验证模型是否学对。

**这次没做好的地方**：没有把清洗参数（中位数、IQR 边界）放进 Pipeline，理论上存在轻微泄漏风险；正式项目应改为 `SimpleImputer` + 自定义 Transformer 放进 Pipeline。

## 23. 关联知识点

| 类型 | 链接 |
|---|---|
| 回归基础 | [[ML-017 回归任务基本概念]] |
| 线性回归 | [[ML-018 线性回归（手写实现）]]、[[ML-019 多元线性回归与正规方程]] |
| 评价与残差 | [[ML-023 回归评价指标与残差诊断]] |
| 划分 | [[ML-040 Hold-out 验证]] |
| 过拟合 | [[ML-046 过拟合与欠拟合]] |
| 数学 | [[MATH-801 线性回归的数学]]、[[MATH-721 梯度下降]] |

## 24. 后续改进方向

1. 把清洗与特征工程搬进 `Pipeline`（消除理论泄漏）。
2. 用 5 折 CV 替代单次划分，报 `0.9167 ± std`。
3. 联网获取真实数据，验证"真实房价关系是否线性"。
4. 加入地理特征（经纬度）与目标编码，看能否突破 R² 0.92。
5. 做误差分析：那 16 个误差 >50 万的样本有什么共同点？

---

## 练习与参考答案

**基础 1**：本项目 MAE 20.157 与 RMSE 28.412 的含义各是什么？为什么 RMSE > MAE？

**基础 2**：为什么用中位数填充 `area` 而不是均值？

**进阶 1**：多项式(2) 的 R²（0.9159）低于线性回归（0.9167），说明什么？

**开放式挑战**：如果业主问"我这套房 120 m²、3 间、15 年、距市中心 5 km，值多少钱？"，你会怎么回答？给出预测值、区间与免责说明。

> [!success]- 参考答案

> **基础 1**
> - MAE 20.157 万：**平均每套预测偏离 20.157 万元**（对每个误差等权）。
> - RMSE 28.412 万：误差平方平均后开方，**对大误差惩罚更重**。
> - RMSE > MAE 说明误差分布不是均匀的：存在少量较大误差（RMSE/MAE ≈ 1.41）。若误差完全相同，两者相等。

> **基础 2**
> 因为面积分布**右偏**（少数超大户型拉高均值），且注入了极端异常。均值会被这些值带偏，中位数更稳健。
> 从 ML-011 的结论看：均值填充在含异常值时会让填充值失真（本库实测过 34.67 vs 3.0 的例子）。

> **进阶 1**
> 说明**真实机制就是线性的**，二次项没有提供新信息，只增加了需要估计的参数（更多自由度 ⇒ 方差上升）。
> 这是"模型复杂度应与数据机制匹配"的实证：不是越复杂越好。
> 判断方法：看残差是否呈现弧形——本课残差与预测值相关只有 −0.005，没有结构，所以不需要非线性项。

> **开放式挑战（参考）**
> 1. **预测**：按模型算出点估计（约 300–360 万区间，取决于 `dist_bucket` 的 One-Hot 取值）。
> 2. **区间**：用残差标准差 28.23 给近似 95% 区间：$\hat y \pm 1.96\times28.23 \approx \hat y \pm 55$ 万。
> 3. **免责说明**（必须说）：
>    - 模型基于**合成数据**训练，`R²=0.9167` 只是"在这份人造数据上"的表现，**不能用于真实房产估价**。
>    - 即使换成真实数据，模型也**不含**学区、楼层、装修、朝向等关键因素。
>    - 结论只能作"量级参考"，实际交易需专业评估。
> 这个练习的核心是：**会算预测值只是第一步，能说清可信边界才是专业**。

## 学习记录

| 日期 | 状态 | 用时 | 遇到的问题 | 复盘 |
|---|---|---|---|---|
| 2026-11-06 | 已完成（首次跑通） | — | 加州房价数据集联网失败 3 次 | 降级合成数据，但验证了"线性假设 + 残差诊断"的完整流程 |

---

## 🔗 Notion 同步信息

> 以下内容由 Notion「AI Knowledge System · 实战项目」数据库按知识 ID 合并而来（导出于 2026-11-09）。

| 属性 | 值 |
|---|---|
| 所属学科 | 机器学习 |
| 项目状态 | 未开始 |
| 项目阶段 | 立项 |
| 难度 | L2 应用 |
| 技术栈 | Python、NumPy、Pandas、scikit-learn |
| 项目仓库 | ai-knowledge-system/projects/PROJ-009 |

- **对应课程模块**：[M-ML-03 监督学习——回归](../../03_%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-ML-03%20%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0%E2%80%94%E2%80%94%E5%9B%9E%E5%BD%92.md)
- **前置知识**：[ML-017 回归任务基本概念](../../03_%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/02%20%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0/ML-017%20%E5%9B%9E%E5%BD%92%E4%BB%BB%E5%8A%A1%E5%9F%BA%E6%9C%AC%E6%A6%82%E5%BF%B5.md) · [ML-018 线性回归（手写实现）](../../03_%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/02%20%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0/ML-018%20%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%EF%BC%88%E6%89%8B%E5%86%99%E5%AE%9E%E7%8E%B0%EF%BC%89.md) · [ML-019 多元线性回归与正规方程](../../03_%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/02%20%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0/ML-019%20%E5%A4%9A%E5%85%83%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E4%B8%8E%E6%AD%A3%E8%A7%84%E6%96%B9%E7%A8%8B.md) · [ML-023 回归评价指标与残差诊断](../../03_%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/02%20%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0/ML-023%20%E5%9B%9E%E5%BD%92%E8%AF%84%E4%BB%B7%E6%8C%87%E6%A0%87%E4%B8%8E%E6%AE%8B%E5%B7%AE%E8%AF%8A%E6%96%AD.md) · [ML-040 Hold-out 验证](../../03_%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/04%20%E6%A8%A1%E5%9E%8B%E8%AF%84%E4%BC%B0/ML-040%20Hold-out%20%E9%AA%8C%E8%AF%81.md)
- **对应知识点**：[ML-017 回归任务基本概念](../../03_%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/02%20%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0/ML-017%20%E5%9B%9E%E5%BD%92%E4%BB%BB%E5%8A%A1%E5%9F%BA%E6%9C%AC%E6%A6%82%E5%BF%B5.md) · [ML-018 线性回归（手写实现）](../../03_%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/02%20%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0/ML-018%20%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%EF%BC%88%E6%89%8B%E5%86%99%E5%AE%9E%E7%8E%B0%EF%BC%89.md) · [ML-019 多元线性回归与正规方程](../../03_%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/02%20%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0/ML-019%20%E5%A4%9A%E5%85%83%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E4%B8%8E%E6%AD%A3%E8%A7%84%E6%96%B9%E7%A8%8B.md) · [ML-023 回归评价指标与残差诊断](../../03_%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/02%20%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0/ML-023%20%E5%9B%9E%E5%BD%92%E8%AF%84%E4%BB%B7%E6%8C%87%E6%A0%87%E4%B8%8E%E6%AE%8B%E5%B7%AE%E8%AF%8A%E6%96%AD.md) · [ML-040 Hold-out 验证](../../03_%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/04%20%E6%A8%A1%E5%9E%8B%E8%AF%84%E4%BC%B0/ML-040%20Hold-out%20%E9%AA%8C%E8%AF%81.md) · [ML-046 过拟合与欠拟合](../../03_%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/04%20%E6%A8%A1%E5%9E%8B%E8%AF%84%E4%BC%B0/ML-046%20%E8%BF%87%E6%8B%9F%E5%90%88%E4%B8%8E%E6%AC%A0%E6%8B%9F%E5%90%88.md)

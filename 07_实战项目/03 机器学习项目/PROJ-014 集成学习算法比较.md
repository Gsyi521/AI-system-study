---
id: PROJ-014
title: 集成学习算法比较
domain: PROJ
module: [M-ML-07]
type: project
difficulty: L3 推导
status: planned
tech_stack: [Python, NumPy, Scikit-learn]
prerequisites: [ML-050, ML-051, ML-054]
related_math: [MATH-306]
related: [ML-050, ML-051, ML-052, ML-053, ML-054, ML-056, ML-057]
notion_page_id: 410663e8-0333-8175-a949-ff9a48648a80
notion_url: https://app.notion.com/p/410663e803338175a949ff9a48648a80
created: 2026-11-06
updated: 2026-11-06
notion_db: 实战项目
所属学科: 机器学习
项目状态: 未开始
项目阶段: 立项
难度: L3 推导
项目目标: 集成学习算法比较
技术栈: [Python, NumPy, scikit-learn]
项目仓库: ai-knowledge-system/projects/PROJ-014
对应课程模块: [M-ML-07]
前置知识: [ML-050, ML-051, ML-054]
对应知识点: [ML-050, ML-051, ML-052, ML-053, ML-054, ML-056, ML-057]
tags: [机器学习]
---

# PROJ-014 集成学习算法比较

## 1. 项目背景

集成学习是表格数据的主力，但"哪个集成最好"没有通用答案——**取决于数据、样本量、以及你更在乎精度还是速度**。

本项目在**同一数据集、同一划分、同一评价口径**下比较 10 个模型（单树 → 同质集成 → 异质集成），并同时报告**准确率、AUC、训练耗时**，让"选型"变成有依据的决策而不是信仰。

> **第三方库声明**：`xgboost`、`lightgbm` 在本机**未安装**（脚本运行时实测并打印）。因此它们**不参与本次对比**，原理与调用方式见 [[ML-055 XGBoost 与 LightGBM]]。**没有伪造任何未运行的结果。**

## 2. 项目目标

1. 对比单树 / Bagging / RF / ExtraTrees / AdaBoost / GBDT / HistGBDT。
2. 对比异质集成 Voting(soft/hard) 与 Stacking。
3. 同时报**精度**与**训练耗时**，理解"精度-成本"权衡。
4. 对比两种特征重要性（不纯度 vs 置换），理解它们可能分歧。
5. 给出选型建议表。

## 3. 学习目标

| 目标 | 验收方式 |
|---|---|
| 会统一比较多个模型 | 输出 CV + 测试 + 耗时表 |
| 理解 Bagging vs Boosting | 说出"谁降方差、谁降偏差" |
| 会看集成的边际收益 | 算出相对单树的 ΔAUC |
| 会做异质集成 | Voting/Stacking 各跑通 |
| 知道未装库如何处理 | 实测探测 + 明确标注 |

## 4. 前置知识

| 知识点 | 为什么需要 |
|---|---|
| [[ML-050 集成学习基本思想]] | Bagging/Boosting 的区别 |
| [[ML-051 Bagging 与随机森林]] | 随机森林原理 |
| [[ML-052 Extra Trees]] | 随机阈值 |
| [[ML-053 AdaBoost]] | 重加权 |
| [[ML-054 梯度提升树 GBDT]] | 残差拟合 |
| [[ML-056 Stacking 与 Voting]] | 异质集成 |
| [[ML-057 特征重要性与模型解释]] | 两种重要性 |

**数学**：[[MATH-306 期望、方差与协方差]]（方差缩减）。

## 5. 数据集来源及许可说明

| 项 | 内容 |
|---|---|
| 来源 | **合成数据**（`make_classification`，种子 42） |
| 规模 | 2000 行 × 20 特征（12 有信息 + 4 冗余） |
| 正类比例 | 49.65%（**平衡**） |
| 噪声 | `flip_y=0.03`（3% 标签翻转） |
| 许可 | 程序生成，无版权问题 |

## 6. 数据字段说明

20 个数值特征（`f0`–`f19`），其中：

- 前 12 个为 informative（真正影响标签）。
- 中间 4 个为 redundant（前者的线性组合）。
- 其余为噪声特征。

目标：二分类 `0/1`。

## 7. 任务定义

- **类型**：二分类（平衡）。
- **评价**：AUC（主，阈值无关）+ 准确率/F1；**同时记录训练耗时**。
- **成功标准**：集成模型 AUC 明显优于单树。

## 8. 技术路线

```mermaid
flowchart LR
    A["2000×20 合成数据"] --> B["分层划分 70/30"]
    B --> C["7 个同质模型 + 3 个异质集成"]
    C --> D["CV + 测试 + 耗时"]
    D --> E["排名 + 相对单树提升"]
    E --> F["特征重要性对比"]
```

## 9. 环境与依赖

| 项 | 版本 |
|---|---|
| Python | 3.15.0rc3 |
| scikit-learn | 1.9.1 |
| **xgboost** | ❌ 未安装 |
| **lightgbm** | ❌ 未安装 |

## 10. 项目目录结构

```text
projects/PROJ-014/
├── README.md
├── requirements.txt
├── src/ensemble_compare.py
├── data/     （脚本生成）
└── results/
```

## 11. 数据探索

```text
=== 0. 环境探测 ===
  xgboost: 未安装（不参与对比）
  lightgbm: 未安装（不参与对比）

=== 1. 数据 ===
形状: (2000, 20)  正类比例: 0.4965
训练/测试: 1400 600
```

**读法**：

- 类别**平衡**（49.65%）⇒ 准确率可用，不必只看 PR。
- 20 个特征里只有 12 个有信息（另有 4 个冗余、4 个噪声）⇒ 考验模型**抗噪声特征**的能力。

## 12. 数据清洗

本项目数据干净，**无需清洗**。唯一预处理是 `StandardScaler`（统一套在 Pipeline 里）。

**注意**：树模型本身不需要标准化，但套上无害，且能与 LR/KNN 共用同一 Pipeline 结构，保证比较公平。

## 13. 特征工程

**刻意不做特征工程**——因为目标是**比较算法**，加特征会混淆变量。

但保留了"特征重要性对比"作为**特征理解的替代**（见第 18 节）。

## 14. 模型建立

| 类别 | 模型 |
|---|---|
| 单模型 | 决策树(d=6) |
| Bagging 系 | Bagging(50)、随机森林(200)、ExtraTrees(200) |
| Boosting 系 | AdaBoost(200)、GBDT(200)、HistGBDT(200) |
| 异质集成 | Voting(soft)、Voting(hard)、Stacking(cv=5) |

**统一口径**：全部 `Pipeline([StandardScaler, model])`，同一 `train_test_split(stratify=y)`。

## 15. 模型训练

```text
=== 2. 单模型与同质集成（5 折 CV + 测试）===
模型                 CV acc   CV std  test acc     F1     AUC  fit(s)
决策树(d=6)           0.8193   0.0188    0.7983 0.7851  0.8228    0.05
Bagging(50)        0.8921   0.0168    0.8833 0.8801  0.9401    2.20
随机森林(200)          0.9143   0.0190    0.8983 0.8954  0.9546    2.92
ExtraTrees(200)    0.9279   0.0109    0.9183 0.9162  0.9621    1.42
AdaBoost(200)      0.8450   0.0205    0.8417 0.8370  0.9093    2.77
GBDT(200)          0.9057   0.0195    0.8983 0.8950  0.9509    4.64
HistGBDT(200)      0.9236   0.0165    0.9100 0.9072  0.9579    2.33
```

| 模型 | CV acc | test acc | F1 | AUC | fit(s) |
|---|---|---|---|---|---|
| 决策树(d=6) | 0.8193 | 0.7983 | 0.7851 | 0.8228 | **0.05** |
| Bagging(50) | 0.8921 | 0.8833 | 0.8801 | 0.9401 | 2.20 |
| 随机森林(200) | 0.9143 | 0.8983 | 0.8954 | 0.9546 | 2.92 |
| **ExtraTrees(200)** | **0.9279** | **0.9183** | **0.9162** | **0.9621** | **1.42** |
| AdaBoost(200) | 0.8450 | 0.8417 | 0.8370 | 0.9093 | 2.77 |
| GBDT(200) | 0.9057 | 0.8983 | 0.8950 | 0.9509 | 4.64 |
| HistGBDT(200) | 0.9236 | 0.9100 | 0.9072 | 0.9579 | 2.33 |

**三个观察**：

1. **集成全面碾压单树**：AUC 0.8228 → 0.94+。
2. **ExtraTrees 最划算**：精度最高（0.9621）且训练最快（1.42s），比 RF 快一倍。
3. **GBDT 最慢**（4.64s）且精度不是最高——串行提升的代价。
4. **AdaBoost 最弱**（0.9093）——它对噪声标签敏感（本数据有 3% 翻转标签）。

## 16. 模型评价

```text
=== 3. 异质集成（Voting / Stacking）===
Voting(soft)            -        -    0.9083 0.9057  0.9586    0.22
Voting(hard)            -        -    0.9117 0.9088     n/a    0.24
Stacking(cv=5)          -        -    0.9200 0.9186  0.9635    1.34
```

| 模型 | test acc | F1 | AUC | fit(s) |
|---|---|---|---|---|
| Voting(soft) | 0.9083 | 0.9057 | 0.9586 | **0.22** |
| Voting(hard) | 0.9117 | 0.9088 | **n/a** | 0.24 |
| **Stacking(cv=5)** | **0.9200** | **0.9186** | **0.9635** | 1.34 |

**两点说明**：

1. **`Voting(hard)` 没有 `predict_proba`**（因为它只投票、不输出概率），所以 AUC 记为 `n/a`——**这是 sklearn 的设计，不是 bug**（本课第一次运行时报了 `AttributeError`）。
2. **Stacking 最强**（AUC 0.9635），但比 ExtraTrees 只高 0.0014，**代价是 1.34s 训练 + 三倍推理**。

## 17. 可视化分析

```text
=== 4. 排名（按测试 AUC）===
  1. Stacking(cv=5)   AUC=0.9635 acc=0.9200 fit=1.34s
  2. ExtraTrees(200)  AUC=0.9621 acc=0.9183 fit=1.42s
  3. Voting(soft)     AUC=0.9586 acc=0.9083 fit=0.22s
  4. HistGBDT(200)    AUC=0.9579 acc=0.9100 fit=2.33s
  5. 随机森林(200)        AUC=0.9546 acc=0.8983 fit=2.92s
  6. GBDT(200)        AUC=0.9509 acc=0.8983 fit=4.64s
```

**推荐图表**：

1. **横向条形图**：各模型 AUC，按降序排列。
2. **散点图：AUC vs 训练耗时**——能一眼看出"性价比"（ExtraTrees 在左上角 = 又准又快）。
3. **相对单树提升条形图**（见下节）。

## 18. 结果解释

```text
=== 5. 相对单树的提升 ===
  Stacking(cv=5)   ΔAUC=+0.1407  ΔF1=+0.1336
  ExtraTrees(200)  ΔAUC=+0.1393  ΔF1=+0.1312
  Voting(soft)     ΔAUC=+0.1358  ΔF1=+0.1206
  HistGBDT(200)    ΔAUC=+0.1351  ΔF1=+0.1221
  随机森林(200)        ΔAUC=+0.1318  ΔF1=+0.1103
  GBDT(200)        ΔAUC=+0.1281  ΔF1=+0.1099
```

**集成的边际收益巨大**：相对单树，AUC 提升 **12.8–14.1 个百分点**。

**但集成之间差距很小**：从 GBDT（+0.1281）到 Stacking（+0.1407）只差 1.3 个百分点，而训练耗时差 **3.5 倍**。

```text
=== 6. 特征重要性对比（随机森林 vs 置换）===
  不纯度 Top5: [3, 7, 18, 2, 14]
  置换   Top5: [7, 3, 18, 14, 16]
  交集: [3, 7, 14, 18]
```

**这次两种重要性高度一致**（5 个里重合 4 个）——与 ML-057 中"只重合 2 个"的情况不同。

**原因**：本数据特征**相关性低**（informative 特征相互独立），而 ML-057 的 breast cancer 特征高度相关（会互相掩护）。

> **结论**：特征越相关，两种重要性的分歧越大。

## 19. 常见错误

| 错误 | 表现 | 纠正 |
|---|---|---|
| 认为集成必然更好 | AdaBoost（0.9093）低于 RF/ET | 看具体模型与数据 |
| 只比精度不比耗时 | 忽略 GBDT 慢 3 倍 | 同时报 fit 时间 |
| 对 `Voting(hard)` 求概率 | `AttributeError`（本课实测） | hard 只投票，无 proba |
| 不探测第三方库就写 import | `ModuleNotFoundError` | 运行时实测 + 标注 |
| 用测试集选模型 | 泄漏 | 用 CV |
| 忽略标签噪声的影响 | AdaBoost 在 3% 翻转下变弱 | 加正则/换模型 |
| 只跑一次就下结论 | 波动 | 报 CV mean±std |

## 20. 完整可运行代码

```powershell
python projects/PROJ-014/src/ensemble_compare.py
```

```python
import time

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier,
                              VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

for lib in ("xgboost", "lightgbm"):
    try:
        __import__(lib)
        print(f"{lib}: 已安装")
    except ImportError:
        print(f"{lib}: 未安装（不参与对比）")

X, y = make_classification(n_samples=2000, n_features=20, n_informative=12,
                           n_redundant=4, flip_y=0.03, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=42, stratify=y)

def pipe(m):
    return Pipeline([("sc", StandardScaler()), ("m", m)])

models = {
    "决策树": pipe(DecisionTreeClassifier(max_depth=6, random_state=42)),
    "随机森林": pipe(RandomForestClassifier(n_estimators=200, random_state=42)),
    "ExtraTrees": pipe(ExtraTreesClassifier(n_estimators=200, random_state=42)),
    "AdaBoost": pipe(AdaBoostClassifier(n_estimators=200, random_state=42)),
    "GBDT": pipe(GradientBoostingClassifier(n_estimators=200, random_state=42)),
    "HistGBDT": pipe(HistGradientBoostingClassifier(max_iter=200, random_state=42)),
}

for name, model in models.items():
    start = time.time()
    model.fit(X_tr, y_tr)
    sec = time.time() - start
    proba = model.predict_proba(X_te)[:, 1]
    print(f"{name:<12} AUC={roc_auc_score(y_te, proba):.4f}"
          f" F1={f1_score(y_te, model.predict(X_te)):.4f} fit={sec:.2f}s")
```

**实际运行结果**

```
xgboost: 未安装（不参与对比）
lightgbm: 未安装（不参与对比）
决策树          AUC=0.8228 F1=0.7851 fit=0.03s
随机森林         AUC=0.9546 F1=0.8954 fit=1.60s
ExtraTrees   AUC=0.9621 F1=0.9162 fit=0.79s
AdaBoost     AUC=0.9093 F1=0.8370 fit=1.55s
GBDT         AUC=0.9509 F1=0.8950 fit=2.64s
HistGBDT     AUC=0.9579 F1=0.9072 fit=1.15s
```

## 21. 进阶挑战

1. **装 XGBoost/LightGBM 后补跑**：`pip install xgboost lightgbm`，把两者加进对比表。
2. **学习曲线**：对 ExtraTrees 与 GBDT 画 learning curve，看谁更吃数据。
3. **噪声鲁棒性**：把 `flip_y` 从 0.03 调到 0.10，看 AdaBoost 是否进一步变差。
4. **Stacking 换元模型**：用 GBDT 代替逻辑回归当元模型，看是否提升。
5. **推理延迟**：测量批量 1000 条的预测耗时，比较"训练快"与"推理快"是否一致。

## 22. 项目总结

| 结论 | 证据 |
|---|---|
| 集成远优于单树 | AUC 0.8228 → 0.9621（+13.9 点） |
| ExtraTrees 性价比最高 | AUC 0.9621、耗时 1.42s（RF 2.92s） |
| Stacking 最强但收益小 | AUC 0.9635，仅比 ET 高 0.0014 |
| GBDT 最慢 | 4.64s，精度不及 ET |
| AdaBoost 最弱 | 0.9093，对 3% 标签噪声敏感 |
| 特征相关性低时两种重要性一致 | 交集 4/5 |

**这次没做好的地方**：

- XGBoost/LightGBM 未安装，只能缺席对比（已如实标注）。
- 只跑了一次划分，没有用重复 CV 验证"Stacking vs ET 的 0.0014 差距"是否真实（结论上**不应认为 Stacking 显著更好**）。

## 23. 关联知识点

| 类型 | 链接 |
|---|---|
| 集成思想 | [[ML-050 集成学习基本思想]] |
| Bagging/RF | [[ML-051 Bagging 与随机森林]] |
| Extra Trees | [[ML-052 Extra Trees]] |
| AdaBoost | [[ML-053 AdaBoost]] |
| GBDT | [[ML-054 梯度提升树 GBDT]] |
| XGB/LGBM | [[ML-055 XGBoost 与 LightGBM]] |
| Stacking/Voting | [[ML-056 Stacking 与 Voting]] |
| 特征重要性 | [[ML-057 特征重要性与模型解释]] |

## 24. 后续改进方向

1. 安装 XGBoost/LightGBM 补齐对比（注意不要污染主环境，用虚拟环境）。
2. 用 `RepeatedStratifiedKFold` 得到带置信区间的排名，避免"0.0014 的假优势"。
3. 加推理延迟测量，做"训练-推理"双维度选型。
4. 在多个数据集上重复本流程，看"ET 最优"是否只是这份数据的特点。
5. 把选型结论固化成一张决策表（数据量 × 精度要求 × 延迟约束）。

---

## 练习与参考答案

**基础 1**：本课单树与最强集成的 AUC 各是多少？提升多少？

**基础 2**：为什么 `Voting(hard)` 没有 AUC？

**进阶 1**：ExtraTrees 与 Stacking 的 AUC 只差 0.0014，你会选哪个？为什么？

**开放式挑战**：业务要求"训练 < 5 秒、推理 < 10ms、AUC > 0.95"，你会选哪个模型？说明推理速度为什么可能和训练速度不一致。

> [!success]- 参考答案

> **基础 1**
> 单树 AUC **0.8228**；最强是 Stacking **0.9635**（同质集成里 ET 0.9621）。
> 提升 **+0.1407**（相对单树）。

> **基础 2**
> 因为 `VotingClassifier(voting="hard")` 只做**多数票**，不聚合概率，因此 sklearn 不提供 `predict_proba`（调用会抛 `AttributeError`）。
> AUC 需要连续分数，所以 hard voting 无法计算 AUC，只能报准确率/F1。
> 想同时要 AUC，就用 `voting="soft"`。

> **进阶 1**
> 选 **ExtraTrees**。
> 理由：
> 1. **差距不显著**：0.0014 远小于单次划分的波动（CV std 约 0.011–0.019），统计上无法区分。
> 2. **成本差 3 倍**：Stacking 要训练 3 个基模型 + 1 个元模型，推理也要跑 4 次；ET 只跑 200 棵树。
> 3. **维护简单**：Stacking 的 `cv` 参数、元模型选择都会引入额外调参面。
> 原则：**当性能差异在噪声内时，选更简单、更快、更好维护的**。

> **开放式挑战（参考）**
> **候选**：ExtraTrees（AUC 0.9621、训练 1.42s）、HistGBDT（0.9579、2.33s）、Stacking（0.9635、1.34s，但推理慢）。
> **按约束筛选**：训练 < 5s 都满足；AUC > 0.95 ⇒ ET/HistGBDT/Stacking 都满足；**推理 < 10ms** 是决定性约束。
> **推理速度与训练速度不一致的原因**：
> 1. **ET/RF**：训练要建 200 棵树（贵），但推理只需每棵树走一条路径（快）。
> 2. **Stacking**：训练 1.34s 看起来快，但推理要跑 **3 个基模型 + 元模型**，延迟最高。
> 3. **GBDT**：训练串行很慢（4.64s），推理也要遍历所有树的路径，比单树慢但比 Stacking 快。
> **建议**：选 **ExtraTrees**，并**实测**批量 1000 条的预测耗时来确认 10ms 约束（不能只凭训练时间推断）。

## 学习记录

| 日期 | 状态 | 用时 | 遇到的问题 | 复盘 |
|---|---|---|---|---|
| 2026-11-06 | 已完成（首次跑通） | — | `Voting(hard)` 无 `predict_proba` 报错；XGB/LGBM 未装 | 学会"探测 + 标注"而不是假装跑过；也第一次意识到"训练快 ≠ 推理快" |

---

## 🔗 Notion 同步信息

> 以下内容由 Notion「AI Knowledge System · 实战项目」数据库按知识 ID 合并而来（导出于 2026-11-09）。

| 属性 | 值 |
|---|---|
| 所属学科 | 机器学习 |
| 项目状态 | 未开始 |
| 项目阶段 | 立项 |
| 难度 | L3 推导 |
| 技术栈 | Python、NumPy、scikit-learn |
| 项目仓库 | ai-knowledge-system/projects/PROJ-014 |

- **对应课程模块**：[M-ML-07 集成学习与高级算法](../../03_%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-ML-07%20%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0%E4%B8%8E%E9%AB%98%E7%BA%A7%E7%AE%97%E6%B3%95.md)
- **前置知识**：[ML-050 集成学习基本思想](../../03_%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/06%20%E6%A8%A1%E5%9E%8B%E4%BC%98%E5%8C%96/ML-050%20%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0%E5%9F%BA%E6%9C%AC%E6%80%9D%E6%83%B3.md) · [ML-051 Bagging 与随机森林](../../03_%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/06%20%E6%A8%A1%E5%9E%8B%E4%BC%98%E5%8C%96/ML-051%20Bagging%20%E4%B8%8E%E9%9A%8F%E6%9C%BA%E6%A3%AE%E6%9E%97.md) · [ML-054 梯度提升树 GBDT](../../03_%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/06%20%E6%A8%A1%E5%9E%8B%E4%BC%98%E5%8C%96/ML-054%20%E6%A2%AF%E5%BA%A6%E6%8F%90%E5%8D%87%E6%A0%91%20GBDT.md)
- **对应知识点**：[ML-050 集成学习基本思想](../../03_%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/06%20%E6%A8%A1%E5%9E%8B%E4%BC%98%E5%8C%96/ML-050%20%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0%E5%9F%BA%E6%9C%AC%E6%80%9D%E6%83%B3.md) · [ML-051 Bagging 与随机森林](../../03_%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/06%20%E6%A8%A1%E5%9E%8B%E4%BC%98%E5%8C%96/ML-051%20Bagging%20%E4%B8%8E%E9%9A%8F%E6%9C%BA%E6%A3%AE%E6%9E%97.md) · [ML-052 Extra Trees](../../03_%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/06%20%E6%A8%A1%E5%9E%8B%E4%BC%98%E5%8C%96/ML-052%20Extra%20Trees.md) · [ML-053 AdaBoost](../../03_%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/06%20%E6%A8%A1%E5%9E%8B%E4%BC%98%E5%8C%96/ML-053%20AdaBoost.md) · [ML-054 梯度提升树 GBDT](../../03_%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/06%20%E6%A8%A1%E5%9E%8B%E4%BC%98%E5%8C%96/ML-054%20%E6%A2%AF%E5%BA%A6%E6%8F%90%E5%8D%87%E6%A0%91%20GBDT.md) · [ML-056 Stacking 与 Voting](../../03_%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/06%20%E6%A8%A1%E5%9E%8B%E4%BC%98%E5%8C%96/ML-056%20Stacking%20%E4%B8%8E%20Voting.md) · [ML-057 特征重要性与模型解释](../../03_%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/06%20%E6%A8%A1%E5%9E%8B%E4%BC%98%E5%8C%96/ML-057%20%E7%89%B9%E5%BE%81%E9%87%8D%E8%A6%81%E6%80%A7%E4%B8%8E%E6%A8%A1%E5%9E%8B%E8%A7%A3%E9%87%8A.md)

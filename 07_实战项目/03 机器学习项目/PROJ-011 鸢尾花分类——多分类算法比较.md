---
id: PROJ-011
title: 鸢尾花分类——多分类算法比较
domain: PROJ
module: [M-ML-04]
type: project
difficulty: L2 应用
status: planned
tech_stack: [Python, NumPy, Scikit-learn]
prerequisites: [ML-025, ML-026, ML-028, ML-029]
related_math: []
related: [ML-025, ML-026, ML-028, ML-029, ML-031, ML-041]
notion_page_id: 410663e8-0333-81bf-a0e5-e1b8ff381649
notion_url: https://app.notion.com/p/410663e8033381bfa0e5e1b8ff381649
created: 2026-11-06
updated: 2026-11-06
notion_db: 实战项目
所属学科: 机器学习
项目状态: 未开始
项目阶段: 立项
难度: L2 应用
项目目标: 鸢尾花分类——多分类算法比较
技术栈: [Python, NumPy, scikit-learn]
项目仓库: ai-knowledge-system/projects/PROJ-011
对应课程模块: [M-ML-04]
前置知识: [ML-025, ML-026, ML-028, ML-029]
对应知识点: [ML-025, ML-026, ML-028, ML-029, ML-031, ML-041]
tags: [机器学习]
---

# PROJ-011 鸢尾花分类——多分类算法比较

## 1. 项目背景

鸢尾花是机器学习最经典的数据集：150 个样本、4 个特征、3 个类别，**干净、平衡、可分**。

正因为它"太容易"，它最适合做一件别的事：**在完全相同的条件下比较四种分类器**，从而看清"算法差异"到底有多大——以及当数据本身简单时，**准确率根本无法区分模型优劣**。

## 2. 项目目标

1. 在**同一划分、同一预处理**下比较 KNN / 逻辑回归 / 决策树 / SVM。
2. 同时报 5 折 CV 与单次测试分数，说明两者的差异。
3. 用混淆矩阵定位"哪两类最容易混"。
4. 只用 2 个特征（花瓣长/宽）重做一遍，观察**降维对决策边界的影响**。
5. 得出"该按什么标准选模型"的结论。

## 3. 学习目标

| 目标 | 验收方式 |
|---|---|
| 会统一比较多个模型 | 输出 CV acc/std + test acc + macro F1 |
| 会读混淆矩阵 | 指出 versicolor ↔ virginica 的混淆 |
| 会做 CV | 报 `mean ± std` |
| 会做误差分析 | 列出错分样本的真实/预测/特征 |
| 理解"数据简单时指标无区分力" | 能说出 4 个模型差距仅几个百分点 |

## 4. 前置知识

| 知识点 | 为什么需要 |
|---|---|
| [[ML-025 逻辑回归]] | 线性多分类基线 |
| [[ML-026 K 近邻]] | 距离型分类器 |
| [[ML-028 决策树]] | 非线性、可解释 |
| [[ML-029 支持向量机]] | 间隔 + 核方法 |
| [[ML-031 多分类策略]] | OvR / multinomial |
| [[ML-041 K 折交叉验证]] | 稳定评估 |

## 5. 数据集来源及许可说明

| 项 | 内容 |
|---|---|
| 来源 | `sklearn.datasets.load_iris` |
| 性质 | **真实数据**（Fisher 1936 经典数据集） |
| 许可 | sklearn 内置，BSD 许可，可自由使用 |
| 规模 | 150 行 × 4 特征 × 3 类（每类 50） |
| 可复现 | 完全确定，无随机 |

## 6. 数据字段说明

| 字段 | 单位 | 含义 |
|---|---|---|
| sepal length | cm | 花萼长度 |
| sepal width | cm | 花萼宽度 |
| **petal length** | cm | 花瓣长度（判别力最强） |
| **petal width** | cm | 花瓣宽度（判别力次强） |
| target | — | 0=setosa / 1=versicolor / 2=virginica |

## 7. 任务定义

- **类型**：三分类（监督学习）。
- **评价**：准确率（主）+ macro F1（辅助，三类等权）。
- **成功标准**：测试准确率 > 0.90。

## 8. 技术路线

```mermaid
flowchart LR
    A["iris 150×4"] --> B["分层划分 70/30"]
    B --> C["4 个分类器 + 统一 StandardScaler"]
    C --> D["5 折 CV + 单次测试"]
    D --> E["混淆矩阵 + 误差分析"]
    E --> F["只用花瓣 2 维重做"]
```

## 9. 环境与依赖

| 项 | 版本 |
|---|---|
| Python | 3.15.0rc3 |
| scikit-learn | 1.9.1 |
| numpy | 2.5.4 |

## 10. 项目目录结构

```text
projects/PROJ-011/
├── README.md
├── requirements.txt
├── src/multiclass_compare.py
├── data/     （用内置数据集，不落盘）
└── results/
```

## 11. 数据探索

```text
=== 1. 数据 ===
形状: (150, 4)
特征: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
类别: ['setosa', 'versicolor', 'virginica']
分布: [50, 50, 50]
```

**读法**：

- 150 样本、4 特征、类别**完全平衡**（50/50/50）。
- 平衡意味着准确率可用（不像 PROJ-010 那样需要看 P/R）。
- 后续分析会看到：**花瓣两个特征几乎就能分开三类**，萼片特征贡献很小。

## 12. 数据清洗

```text
=== 2. 划分 ===
训练: (105, 4)  测试: (45, 4)
训练类别分布: [35, 35, 35]
测试类别分布: [15, 15, 15]
```

**本项目无需清洗**：iris 无缺失、无异常、量纲统一（cm）。这是它作为"教学数据集"的原因，也是它**不能代表真实项目**的原因。

**唯一处理**：分层划分（`stratify=y`），保证三类在训练/测试中都恰好是 35/15。

## 13. 特征工程

**本项目刻意不做特征工程**，原因：

1. 目的是**比较算法**，不是提升分数；加特征会混淆变量。
2. 后面第 5 节会做一次"只用 2 个特征"的对照，那就是最小的特征选择实验。

**唯一预处理**：`StandardScaler`。理由：KNN 与 SVM 依赖距离/内积，必须标准化；LR 带正则也需要；树不需要但无害（统一口径便于比较）。

## 14. 模型建立

| 模型 | 关键参数 | 为什么选它 |
|---|---|---|
| KNN(k=5) | `n_neighbors=5` | 距离型、无参数、非线性 |
| 逻辑回归 | `max_iter=2000` | 线性基线、输出概率 |
| 决策树(d=3) | `max_depth=3` | 非线性、可解释 |
| SVM(rbf) | `kernel="rbf"` | 间隔最大、核方法 |

**统一口径**：全部包在 `Pipeline([("sc", StandardScaler()), ("clf", model)])` 里，保证 CV 时每折重新标准化（不泄漏）。

## 15. 模型训练

```text
=== 3. 模型对比（5 折 CV + 单次测试）===
模型               CV acc   CV std  test acc  macro F1
KNN(k=5)         0.9619   0.0356    0.9111    0.9095
逻辑回归             0.9810   0.0233    0.9111    0.9107
决策树(d=3)         0.9524   0.0301    0.9778    0.9778
SVM(rbf)         0.9714   0.0233    0.9333    0.9333
```

| 模型 | CV acc | CV std | test acc | macro F1 |
|---|---|---|---|---|
| KNN(k=5) | 0.9619 | 0.0356 | 0.9111 | 0.9095 |
| 逻辑回归 | **0.9810** | 0.0233 | 0.9111 | 0.9107 |
| **决策树(d=3)** | 0.9524 | 0.0301 | **0.9778** | **0.9778** |
| SVM(rbf) | 0.9714 | 0.0233 | 0.9333 | 0.9333 |

**这里出现了一个必须解释的矛盾**：

- **CV 最好的是逻辑回归**（0.9810），
- **单次测试最好的是决策树**（0.9778），
- 而逻辑回归的测试只有 0.9111。

**原因**：测试集只有 45 条，1 条错 = 2.2 个百分点。决策树在 CV 上只有 0.9524（最差），却在这次测试上恰好对了 44/45。**这是抽样波动，不是"树更好"。**

**正确做法**：以 CV 为准（逻辑回归 0.9810），或用重复 CV 进一步稳定。

## 16. 模型评价

```text
=== 4. 混淆矩阵（决策树(d=3)）===
[[15  0  0]
 [ 0 15  0]
 [ 0  1 14]]
              precision    recall  f1-score   support

      setosa      1.000     1.000     1.000        15
  versicolor      0.938     1.000     0.968        15
   virginica      1.000     0.933     0.966        15

    accuracy                          0.978        45
   macro avg      0.979     0.978     0.978        45
```

**逐类解读**：

| 类 | P | R | F1 | 说明 |
|---|---|---|---|---|
| setosa | 1.000 | 1.000 | 1.000 | **完美可分**（与另两类线性可分） |
| versicolor | 0.938 | 1.000 | 0.968 | 全召回，但混进 1 个 virginica |
| virginica | 1.000 | 0.933 | 0.966 | 漏了 1 个（被判成 versicolor） |

**结论**：错分**只发生在 versicolor 与 virginica 之间**（这两类在特征空间有重叠），setosa 从未出错。

## 17. 可视化分析

```text
=== 5. 决策边界分析（只用花瓣长/宽 2 维）===
模型(2D)          test acc
KNN(k=5)          0.9111
逻辑回归              0.9111
决策树(d=3)          0.9333
SVM(rbf)          0.9111
```

**降维后（4 维 → 2 维）**：

- 三个模型（KNN/LR/SVM）都是 0.9111，与 4 维时**几乎相同**。
- 说明**花瓣两个特征已经承载了绝大部分判别信息**。

**可视化建议**（本项目用文本输出替代绘图）：

1. **花瓣长 vs 花瓣宽的散点图**，按类别着色 ⇒ 能看到 setosa 完全分离，另两类有一条缝。
2. **决策边界填充图**：在 2D 网格上预测，用 `contourf` 画区域 ⇒ 直观看到：
   - 逻辑回归：**直线边界**（三条线把平面切成三块）
   - 决策树：**轴对齐的阶梯边界**
   - KNN：**不规则、跟随样本**
   - SVM(rbf)：**平滑曲线**

## 18. 结果解释

```text
=== 7. 错误分析（4D，全部错分样本）===
错分样本数: 3 / 150
  真实=versicolor 预测=virginica  特征=[5.9, 3.2, 4.8, 1.8]
  真实=virginica  预测=versicolor  特征=[4.9, 2.5, 4.5, 1.7]
  真实=virginica  预测=versicolor  特征=[7.2, 3.0, 5.8, 1.6]
```

**三个错分样本的共同点**：花瓣长度都在 **4.5–5.8 cm**（versicolor 通常 3–5，virginica 通常 4.5–7）。这是两类的**过渡带**，标注本身就有模糊性。

**业务含义**：如果这是真实植物分类任务，这 3 个样本可能需要**专家复核**，而不是怪模型。

## 19. 常见错误

| 错误 | 表现 | 纠正 |
|---|---|---|
| 只看单次测试分 | 本课树 0.9778 > 逻辑回归 0.9111 | 以 CV 为准（0.9810） |
| 用测试集选模型 | 泄漏 | 用 CV 选 |
| 不标准化就上 SVM/KNN | 效果略降、不稳定 | Pipeline 内标准化 |
| 三分类用二分类平均错 | P/R 计算错 | 指定 `average="macro"` |
| 认为 setosa 难分 | 实际 100% 正确 | 看混淆矩阵定位难点 |
| 结论写成"树最好" | 被单次波动误导 | 写"CV 上逻辑回归最好" |

## 20. 完整可运行代码

```powershell
python projects/PROJ-011/src/multiclass_compare.py
```

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

models = {
    "KNN(k=5)": KNeighborsClassifier(n_neighbors=5),
    "逻辑回归": LogisticRegression(max_iter=2000),
    "决策树(d=3)": DecisionTreeClassifier(max_depth=3, random_state=42),
    "SVM(rbf)": SVC(kernel="rbf"),
}

print(f"{'模型':<14}{'CV acc':>9}{'test acc':>10}{'macro F1':>10}")
for name, clf in models.items():
    pipe = Pipeline([("sc", StandardScaler()), ("clf", clf)])
    cv = cross_val_score(pipe, X_tr, y_tr, cv=5)
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    print(f"{name:<14}{cv.mean():>9.4f}{accuracy_score(y_te, pred):>10.4f}"
          f"{f1_score(y_te, pred, average='macro'):>10.4f}")
```

**实际运行结果**

```
模型               CV acc  test acc  macro F1
KNN(k=5)         0.9619    0.9111    0.9095
逻辑回归             0.9810    0.9111    0.9107
决策树(d=3)         0.9524    0.9778    0.9778
SVM(rbf)         0.9714    0.9333    0.9333
```

## 21. 进阶挑战

1. **重复 5 折 CV（10 次）**：看决策树与逻辑回归的差异是否稳定存在。
2. **调 SVM 超参**：对 `C`、`gamma` 做网格搜索，看能否超过 0.98。
3. **只用萼片特征**：`X[:, :2]`，看准确率掉多少 ⇒ 量化花瓣特征的价值。
4. **画决策边界**：用 `np.meshgrid` + `contourf` 画出四个模型在 2D 上的分界。
5. **换多分类策略**：对比 `OneVsRestClassifier` 与默认 multinomial（参考 ML-031）。

## 22. 项目总结

- **数据简单时，四个模型差距 < 3 个百分点**，此时"选哪个"不该看准确率。
- **CV 与单次测试会打架**（逻辑回归 CV 最好、树测试最好），因为测试集只有 45 条。**以 CV 为准**。
- **混淆矩阵给出了真正的信息**：难点只在 versicolor ↔ virginica。
- **降维到 2 个花瓣特征几乎不掉分**（0.9111），说明特征冗余度高。

**选型建议（按场景）**：

| 需求 | 推荐 |
|---|---|
| 需要概率 | 逻辑回归 |
| 需要规则可读 | 决策树(d=3) |
| 需要最高精度、数据中等 | SVM(rbf) |
| 需要极简实现 | KNN |

## 23. 关联知识点

| 类型 | 链接 |
|---|---|
| 逻辑回归 | [[ML-025 逻辑回归]] |
| KNN | [[ML-026 K 近邻]] |
| 决策树 | [[ML-028 决策树]] |
| SVM | [[ML-029 支持向量机]] |
| 多分类策略 | [[ML-031 多分类策略]] |
| 交叉验证 | [[ML-041 K 折交叉验证]] |

## 24. 后续改进方向

1. 用 `RepeatedStratifiedKFold` 消除单次划分的波动，得到稳定排名。
2. 加入学习曲线，判断 150 条样本是否已成为瓶颈。
3. 画决策边界图，把"四种算法的边界形状差异"变成可看的证据。
4. 换一个更难的数据集（如 wine 或 digits），看模型差异是否被放大。

---

## 练习与参考答案

**基础 1**：本课四个模型的 CV 准确率各是多少？最好的是哪个？

**基础 2**：混淆矩阵显示哪个类从未出错？哪两类互相混淆？

**进阶 1**：为什么"CV 最优"与"测试最优"是不同的模型？该信哪个？

**开放式挑战**：业务方说"我们要一个能解释规则的模型"，你会推荐哪个？为什么不是准确率最高的那个？

> [!success]- 参考答案

> **基础 1**
> KNN 0.9619、逻辑回归 **0.9810**（最好）、决策树 0.9524、SVM 0.9714。
> 注意：**CV 最好的是逻辑回归**，不是测试分数最高的决策树。

> **基础 2**
> - **setosa 从未出错**（P/R/F1 全 1.000，混淆矩阵第一行/列全对）。
> - **versicolor 与 virginica 互相混淆**：1 个 virginica 被判成 versicolor，导致 versicolor 的 P=0.938、virginica 的 R=0.933。

> **进阶 1**
> 因为**测试集只有 45 条**：1 条样本 = 2.2 个百分点。决策树这次恰好对了 44 条（0.9778），而它在 CV 上只有 0.9524（四个模型里最差）。
> **应信 CV**（逻辑回归 0.9810），因为：
> 1. CV 用了 5 次评估、覆盖全部训练数据；
> 2. 单次划分的测试分数波动可达数个点，不足以区分模型；
> 3. 用测试集选模型本身就是泄漏。

> **开放式挑战（参考）**
> 推荐 **决策树（`max_depth=3`）**。
> 理由：
> 1. **可解释性**是这次的第一诉求，而树能直接打印成 `if petal_length <= 2.45 then setosa` 这样的规则。
> 2. 它的 CV（0.9524）虽然低于逻辑回归（0.9810），但差距在 std（0.0301）范围内，**统计上不显著**。
> 3. 逻辑回归的"规则"是系数加权和，业务方需要理解 sigmoid 和权重，门槛更高。
> 4. SVM 最不可解释（核空间中的间隔），KNN 只能说"看最近的 5 个邻居"。
> **结论**：当准确率差异在噪声范围内时，**选最匹配业务约束的模型**，而不是分数最高的。

## 学习记录

| 日期 | 状态 | 用时 | 遇到的问题 | 复盘 |
|---|---|---|---|---|
| 2026-11-06 | 已完成（首次跑通） | — | CV 与单次测试结论不一致 | 认识到 45 条测试集无法区分模型；学会用"差异是否在 std 内"判断 |

---

## 🔗 Notion 同步信息

> 以下内容由 Notion「AI Knowledge System · 实战项目」数据库按知识 ID 合并而来（导出于 2026-11-09）。

| 属性 | 值 |
|---|---|
| 所属学科 | 机器学习 |
| 项目状态 | 未开始 |
| 项目阶段 | 立项 |
| 难度 | L2 应用 |
| 技术栈 | Python、NumPy、scikit-learn |
| 项目仓库 | ai-knowledge-system/projects/PROJ-011 |

- **对应课程模块**：[M-ML-04 监督学习——分类](../../03_%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-ML-04%20%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0%E2%80%94%E2%80%94%E5%88%86%E7%B1%BB.md)
- **前置知识**：[ML-025 逻辑回归](../../03_%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/02%20%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0/ML-025%20%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92.md) · [ML-026 K 近邻](../../03_%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/02%20%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0/ML-026%20K%20%E8%BF%91%E9%82%BB.md) · [ML-028 决策树](../../03_%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/02%20%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0/ML-028%20%E5%86%B3%E7%AD%96%E6%A0%91.md) · [ML-029 支持向量机](../../03_%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/02%20%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0/ML-029%20%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA.md)
- **对应知识点**：[ML-025 逻辑回归](../../03_%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/02%20%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0/ML-025%20%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92.md) · [ML-026 K 近邻](../../03_%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/02%20%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0/ML-026%20K%20%E8%BF%91%E9%82%BB.md) · [ML-028 决策树](../../03_%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/02%20%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0/ML-028%20%E5%86%B3%E7%AD%96%E6%A0%91.md) · [ML-029 支持向量机](../../03_%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/02%20%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0/ML-029%20%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA.md) · [ML-031 多分类策略](../../03_%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/02%20%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0/ML-031%20%E5%A4%9A%E5%88%86%E7%B1%BB%E7%AD%96%E7%95%A5.md) · [ML-041 K 折交叉验证](../../03_%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/04%20%E6%A8%A1%E5%9E%8B%E8%AF%84%E4%BC%B0/ML-041%20K%20%E6%8A%98%E4%BA%A4%E5%8F%89%E9%AA%8C%E8%AF%81.md)

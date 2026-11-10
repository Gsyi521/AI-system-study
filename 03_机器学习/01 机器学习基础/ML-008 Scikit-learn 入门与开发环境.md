---
id: ML-008
title: Scikit-learn 入门与开发环境
en_title: Getting Started with Scikit-learn
domain: ML
module: M-ML-01
type: Technique
difficulty: L2 应用
status: planned
prerequisites: [ML-006]
related_math: []
related_python: [PY-049, PY-050]
related_ml: [ML-003, ML-006]
notion_page_id: 40d663e8-0333-81f8-b3c8-c23e298b21e6
notion_url: https://app.notion.com/p/Scikit-learn-40d663e8033381f8b3c8c23e298b21e6
created: 2026-11-03
updated: 2026-11-03
notion_db: 知识点
学科: 机器学习
学习状态: 未开始
难度: L2 应用
知识类型: Technique
所属模块: [M-ML-01]
前置知识: [ML-006]
关联Python: [PY-049, PY-050]
tags: [机器学习, Technique]
---

# ML-008 Scikit-learn 入门与开发环境

## 1. 本节学习目标

1. 能列出 sklearn 的三件套 API：估计器、转换器、Pipeline。
2. 会写"导入 → 划分 → 建模 → fit → score"的最小流程。
3. 知道本机实际版本，并能用 `sklearn.__version__` 自查。
4. 理解为什么所有模型接口一致——这是 Pipeline 与 CV 能通用的前提。

## 2. 为什么需要这个知识

因为后面 50+ 篇笔记**全部**用同一套接口。接口不熟，每篇都要重新查文档；接口熟了，换模型只是换一行 import。

更重要的是：**接口一致 ⇒ 可以写循环比较模型、可以整体塞进 Pipeline、可以统一做交叉验证**。这是 sklearn 最大的设计价值。

## 3. 直观解释

sklearn 把"机器学习"抽象成三个动作：

| 动作 | 方法 | 谁有 |
|---|---|---|
| 学习 | `fit(X, y)` | 所有估计器 |
| 预测 | `predict(X)` | 有监督模型 |
| 打分 | `score(X, y)` | 有监督模型 |
| 变换 | `transform(X)` | 预处理器（Scaler/Encoder） |
| 联合 | `fit_transform(X)` | 预处理器 |

一句话：**有 `fit` 就是估计器；有 `transform` 就是转换器；两个都有就能进 Pipeline**。

## 4. 正式定义

**估计器（Estimator）**：实现 `fit` 的对象，训练后把学到的量存为**带下划线**的属性（`coef_`、`mean_`）。

**转换器（Transformer）**：实现 `fit` + `transform`，把 $X$ 映射成新的 $X'$。

**预测器（Predictor）**：实现 `predict`，可选 `predict_proba`、`decision_function`。

**Pipeline**：把若干转换器 + 一个估计器串成单个估计器，满足同一接口。

```text
pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
pipe.fit(X_train, y_train)      # 内部：scaler.fit_transform → clf.fit
pipe.predict(X_test)            # 内部：scaler.transform → clf.predict
```

## 5. 核心原理

**为什么接口统一能带来复利**：

1. 一行换模型：`KNeighborsClassifier` → `DecisionTreeClassifier`。
2. 统一做网格搜索：`GridSearchCV(estimator, param_grid)`，参数名用 `步骤名__参数名`。
3. 统一做交叉验证：`cross_val_score(estimator, X, y)`。
4. 统一保存：`joblib.dump(model)`。

**约定优于配置**：sklearn 用命名约定（`coef_` 表示学习到的系数、`_` 结尾表示"fit 后才存在"）替代了大量文档查询。

## 6. 数学表达式

**统一的经验风险最小化**

$$
\hat\theta = \arg\min_\theta \frac{1}{n}\sum_{i=1}^{n} L\big(f_\theta(\mathbf{x}_i), y_i\big) + \Omega(\theta)
$$

不同模型只是 $f_\theta$ 与 $\Omega$ 不同，**外层流程完全一样**——这就是接口能统一的原因。

**Pipeline 的复合映射**

$$
\hat y = f_{\theta_2}\big(T_{\theta_1}(\mathbf{x})\big)
$$

## 7. 数学库链接

| 用到的数学 | 对应篇目 | 用在哪 |
|---|---|---|
| 距离/内积 | [[MATH-201 向量与空间基础]] | KNN、线性模型的几何解释 |
| 矩阵乘法 | [[MATH-216 矩阵乘法]] | `X @ coef` 的计算 |

## 8. 算法执行流程

```mermaid
flowchart LR
    A["import"] --> B["load 数据"]
    B --> C["train_test_split"]
    C --> D["实例化模型（超参）"]
    D --> E["fit"]
    E --> F["predict / score"]
    F --> G["（可选）存模型"]
```

## 9. 可视化解释

```text
         ┌──────────── Pipeline ────────────┐
X_raw ──►│ Scaler.fit_transform ──► clf.fit │──► 模型
         └──────────────────────────────────┘
X_new ──►│ Scaler.transform ──────► clf.predict │──► ŷ
         └──────────────────────────────────┘
          同一个对象，两条路径，接口不变
```

## 10. Python 实现

```powershell
python ml008_sklearn_intro.py
```

```python
import sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

print("sklearn 版本:", sklearn.__version__)

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print("数据形状:", X.shape, y.shape, " 训练/测试:", X_train.shape[0], X_test.shape[0])

models = {
    "KNN(k=5)": KNeighborsClassifier(n_neighbors=5),
    "LogReg": LogisticRegression(max_iter=1000),
    "Tree(depth=3)": DecisionTreeClassifier(max_depth=3, random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)          # 同一个接口
    print(f"{name:<14} 测试准确率={model.score(X_test, y_test):.4f}")

print("都有 fit/predict:", all(hasattr(m, "fit") and hasattr(m, "predict")
                               for m in models.values()))
```

**实际运行结果**

```
sklearn 版本: 1.9.1
数据形状: (150, 4) (150,)  训练/测试: 105 45
KNN(k=5)       测试准确率=0.9778
LogReg         测试准确率=0.9333
Tree(depth=3)  测试准确率=0.9778
都有 fit/predict: True
```

## 11. 代码逐行解释

| 代码 | 作用 |
|---|---|
| `sklearn.__version__` | 自查版本，避免照抄旧教程的废弃 API |
| `load_iris(return_X_y=True)` | 直接返回 `(X, y)` 元组，比 `.data/.target` 简洁 |
| 字典存模型 | 便于循环比较（接口统一的好处） |
| `model.fit` / `model.score` | 完全一样的调用方式 |
| `hasattr(m, "fit")` | 演示"估计器"的判据是方法而不是类型 |

**版本提示**：本机 1.9.1 相对老教程有变化（如 `normalize=` 已移除、`squared=` 改成 `root_mean_squared_error`），照抄旧代码常报 `TypeError`——**先查版本再抄代码**。

## 12. 实际运行结果

见第 10 节。要点：

- 三个模型只换了 import 与超参，流程一字未改。
- 本例 KNN 与 Tree 并列 0.9778，LogReg 0.9333——**没有"最好模型"，只有"在这份数据上更好"**。
- 45 条测试集，1 条错 = 2.2 个百分点，0.9778 与 0.9333 的差距（2 条）在噪声范围内。

## 13. 适用场景

| 需求 | 用哪个 API |
|---|---|
| 快速基线 | `DummyClassifier` / `LogisticRegression` |
| 多模型比较 | 循环 + 统一 `score` |
| 预处理 + 模型 | `Pipeline` |
| 调参 | `GridSearchCV` / `RandomizedSearchCV` |
| 评估 | `cross_val_score` / `cross_validate` |
| 持久化 | `joblib.dump` / `joblib.load` |

## 14. 优点与缺点

| 优点 | 缺点 |
|---|---|
| 接口统一，学习成本低 | 不支持深度学习/GPU 训练 |
| 文档与示例极多 | 大数据（>百万行）性能有限 |
| 默认参数通常合理 | 高阶定制（自定义损失）较麻烦 |
| 与 pandas/numpy 无缝 | 类别特征需先编码 |

## 15. 常见错误

| 错误 | 表现 | 纠正 |
|---|---|---|
| 抄旧版 API | `TypeError: unexpected keyword` | 先看 `sklearn.__version__` |
| 忘记 `fit` 就 `predict` | `NotFittedError` | 先 fit |
| 用 `predict` 拿概率 | 得到 0/1 | 用 `predict_proba` |
| 把 `random_state` 设成不同值比较模型 | 差异来自随机性 | 固定种子 |
| 在 Pipeline 外做预处理 | 泄漏 | 全放进 Pipeline |
| 用训练集 `score` 报结果 | 虚高 | 用测试集 |

## 16. 本节总结

- sklearn 的三件套：**估计器（fit）、转换器（transform）、Pipeline**。
- 接口统一带来复利：换模型、调参、交叉验证、保存都是一行。
- 先查版本再抄代码，1.9 已移除若干旧参数。

## 17. 前置与后续

- 前置：[[ML-006 模型、参数与超参数]]、[[ML-003 机器学习工作流程]]
- 后续：`ML-009 结构化数据读取`、`ML-010 数据清洗`、`ML-016 预处理 Pipeline 与 ColumnTransformer`（待建）、`ML-059 Pipeline 完整实践`（待建）
- 相关 Python：[[PY-049 包与项目结构]]、[[PY-050 pip 与虚拟环境]]

## 18. 练习与参考答案

**基础 1**：列出 sklearn 里"估计器"与"转换器"的判据（看哪个方法）。

**基础 2**：用 `make_pipeline` 组合 `StandardScaler` 与 `LogisticRegression`，报告测试准确率。

**进阶 1**：`fit_transform` 与 `transform` 有什么区别？为什么测试集只能用 `transform`？

**综合 1**：用统一循环比较两个模型并找出更好者。

> [!success]- 参考答案

> **基础 1**
> - 估计器：有 `fit`。
> - 转换器：有 `fit` + `transform`（典型还有 `fit_transform`）。
> - 预测器：有 `predict`（分类器通常还有 `predict_proba`）。

> **基础 2**
> ```python
> from sklearn.datasets import load_iris
> from sklearn.linear_model import LogisticRegression
> from sklearn.model_selection import train_test_split
> from sklearn.pipeline import make_pipeline
> from sklearn.preprocessing import StandardScaler
>
> X, y = load_iris(return_X_y=True)
> Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
> pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
> print("测试准确率:", round(pipe.score(Xte, yte), 4))
> ```
> 实跑输出：
> ```
> 测试准确率: 0.9111
> ```

> **进阶 1**
> - `fit_transform`：先 `fit`（学参数，如均值方差）再 `transform`（应用）。**只能对训练集用**。
> - `transform`：只用已学到的参数做变换。
> 测试集若用 `fit_transform`，就会用测试集自己的均值方差 ⇒ 测试集信息进入预处理 ⇒ 数据泄漏。

> **综合 1**
> ```python
> res = {}
> for name, m in {"LogReg": LogisticRegression(max_iter=1000),
>                 "Tree": DecisionTreeClassifier(max_depth=3, random_state=42)}.items():
>     m.fit(X_train, y_train)
>     res[name] = round(m.score(X_test, y_test), 4)
> print(res, "最好:", max(res, key=res.get))
> ```
> 实跑输出：
> ```
> {'LogReg': 0.9333, 'Tree': 0.9778} 最好: Tree
> ```
> **易错点**：0.9778 与 0.9333 只差 2 条样本，不能据此断言"树永远更好"；要换种子/交叉验证再看（见 `ML-041 K 折交叉验证`）。

---

## 🔗 Notion 同步信息

> 以下内容由 Notion「AI Knowledge System · 知识点」数据库按知识 ID 合并而来（导出于 2026-11-09）。

| 属性 | 值 |
|---|---|
| 学科 | 机器学习 |
| 学习状态 | 未开始 |
| 难度 | L2 应用 |
| 知识类型 | Technique |

- **所属模块**：[M-ML-01 机器学习基础](../%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-ML-01%20%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80.md)
- **前置知识**：[ML-006 模型、参数与超参数](ML-006%20%E6%A8%A1%E5%9E%8B%E3%80%81%E5%8F%82%E6%95%B0%E4%B8%8E%E8%B6%85%E5%8F%82%E6%95%B0.md)
- **关联Python**：[PY-049 包与项目结构](../../01_Python%E5%9F%BA%E7%A1%80/06%20%E6%A8%A1%E5%9D%97%E4%B8%8E%E5%8C%85/PY-049%20%E5%8C%85%E4%B8%8E%E9%A1%B9%E7%9B%AE%E7%BB%93%E6%9E%84.md) · [PY-050 pip 与虚拟环境](../../01_Python%E5%9F%BA%E7%A1%80/06%20%E6%A8%A1%E5%9D%97%E4%B8%8E%E5%8C%85/PY-050%20pip%20%E4%B8%8E%E8%99%9A%E6%8B%9F%E7%8E%AF%E5%A2%83.md)

---
id: PROJ-008
title: 数据可视化项目
domain: PROJ
module: []
type: project
difficulty: L3 推导
tech_stack: [Python, Matplotlib]
status: planned
prerequisites: [PY-076, PY-077, PY-073]
related_math: []
related: []
notion_page_id: 40b663e8-0333-8135-9bf7-c209d8bf20e3
notion_url: https://app.notion.com/p/40b663e8033381359bf7c209d8bf20e3
created: 2026-11-01
updated: 2026-11-01
notion_db: 实战项目
所属学科: 科学计算
项目状态: 未开始
项目阶段: 立项
难度: L3 推导
项目目标: 一页四图看板：日趋势、月汇总、分布、滞后关系。
技术栈: [Pandas, Matplotlib]
对应课程模块: [M-PY-08]
前置知识: [PY-076, PY-077, PY-073]
tags: [科学计算]
---

# PROJ-008 数据可视化项目

## 1. 项目目标

把同一份数据做成**一页四图看板**：日趋势、月汇总、分布、滞后关系，每张图只回答一个问题。

- 完成后应能证明：你会选图型、会排子图、能判断"图有没有把结论说清"。
- 工作量：约 3 小时。
- 难度：L3（中级）。

## 2. 前置知识

| 知识点 | 为什么需要 |
|---|---|
| [[PY-076 Matplotlib 基础]] | figure/axes 与保存 |
| [[PY-077 图表设计与可读性]] | 图型选择与排版 |
| [[PY-073 分组聚合]] | 月度汇总 |
| [[PY-075 数据转换与时间序列]] | 日期处理 |

## 3. 需求说明

| 编号 | 需求 | 验收 |
|---|---|---|
| F1 | 四张子图（2×2） | `axes.size == 4` |
| F2 | 每张图有标题 | 标题不为空 |
| F3 | 日趋势用折线，月汇总用柱状 | 图型正确 |
| F4 | 分布用直方图 | bins 合理（≤20） |
| F5 | 保存为 PNG 且文件非空 | 大小 > 0 |
| F6 | 无界面环境可运行 | 用 `Agg` 后端 |

## 4. 项目目录结构

```text
viz_dashboard/
├── dashboard.py   # 绘图脚本
├── output/        # 生成的图片（运行后）
└── README.md
```

## 5. 实现步骤

1. 造 90 天数据（固定种子，保证可复现）。
2. 派生 `month` 列并做月汇总。
3. `subplots(2, 2)` 建四格。
4. 分别画：折线（日）、柱状（月）、直方图（分布）、散点（今日 vs 昨日）。
5. `tight_layout()` 后保存，验证文件存在且非空，再删除演示文件。

## 6. 完整参考代码

```powershell
python dashboard.py
```

```python
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rng = np.random.default_rng(1)
dates = pd.date_range("2026-01-01", periods=90, freq="D")
sales = rng.integers(50, 200, size=90).astype(float)

df = pd.DataFrame({"date": dates, "sales": sales})
df["month"] = df["date"].dt.to_period("M").astype(str)
monthly = df.groupby("month")["sales"].sum()

fig, axes = plt.subplots(2, 2, figsize=(9, 6))

axes[0, 0].plot(df["date"], df["sales"], linewidth=1)
axes[0, 0].set_title("Daily Sales")

axes[0, 1].bar(monthly.index, monthly.values, color="#4C78A8")
axes[0, 1].set_title("Monthly Total")

axes[1, 0].hist(df["sales"], bins=10, color="#F2A93B")
axes[1, 0].set_title("Distribution")

axes[1, 1].scatter(df["sales"].shift(1), df["sales"], s=8)
axes[1, 1].set_title("Lag Scatter")

fig.tight_layout()
p = Path("dashboard.png")
fig.savefig(p, dpi=90)
plt.close(fig)

print("面板数:", axes.size)
print("文件生成:", p.exists(), p.stat().st_size > 0)
p.unlink()
```

## 7. 运行说明

```powershell
cd viz_dashboard
python dashboard.py
```

**预期输出**

```
面板数: 4
文件生成: True True
```

## 8. 扩展挑战

1. **加基准线**：在月汇总图加一条"月均线"，让高低一眼可见。
2. **改横向条形**：把月汇总改成 `barh` 并按金额排序。
3. **双 y 轴**：日趋势 + 7 日滑动平均（用 `rolling(7).mean()`），共用一个 x 轴。
4. **配色与可读性**：统一主色、去掉多余网格线、给关键点加注释（`ax.annotate`）。
5. **生成多页报告**：按月份循环出图，保存成 `2026-01.png` 等。

## 9. 复盘要点

- 四张图分别回答什么问题？如果把"分布"换成折线，会误导什么？
- `shift(1)` 散点想说明什么关系？它等于因果吗？
- 为什么图内标签用英文？（中文需额外配置字体，否则出豆腐块）

## 10. 学习记录

| 日期 | 状态 | 用时 | 遇到的问题 | 复盘 |
|---|---|---|---|---|
| — | planned | — | — | — |

---

## 🔗 Notion 同步信息

> 以下内容由 Notion「AI Knowledge System · 实战项目」数据库按知识 ID 合并而来（导出于 2026-11-09）。

| 属性 | 值 |
|---|---|
| 所属学科 | 科学计算 |
| 项目状态 | 未开始 |
| 项目阶段 | 立项 |
| 难度 | L3 推导 |
| 项目目标 | 一页四图看板：日趋势、月汇总、分布、滞后关系。 |
| 技术栈 | Pandas、Matplotlib |

- **对应课程模块**：[M-PY-08 Pandas 与数据可视化](../../01_Python%E5%9F%BA%E7%A1%80/%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-PY-08%20Pandas%20%E4%B8%8E%E6%95%B0%E6%8D%AE%E5%8F%AF%E8%A7%86%E5%8C%96.md)
- **前置知识**：[PY-076 Matplotlib 基础](../../01_Python%E5%9F%BA%E7%A1%80/09%20Pandas%E4%B8%8E%E5%8F%AF%E8%A7%86%E5%8C%96/PY-076%20Matplotlib%20%E5%9F%BA%E7%A1%80.md) · [PY-077 图表设计与可读性](../../01_Python%E5%9F%BA%E7%A1%80/09%20Pandas%E4%B8%8E%E5%8F%AF%E8%A7%86%E5%8C%96/PY-077%20%E5%9B%BE%E8%A1%A8%E8%AE%BE%E8%AE%A1%E4%B8%8E%E5%8F%AF%E8%AF%BB%E6%80%A7.md) · [PY-073 分组聚合](../../01_Python%E5%9F%BA%E7%A1%80/09%20Pandas%E4%B8%8E%E5%8F%AF%E8%A7%86%E5%8C%96/PY-073%20%E5%88%86%E7%BB%84%E8%81%9A%E5%90%88.md)

---
id: PROJ-007
title: Pandas 数据分析项目
domain: PROJ
module: []
type: project
difficulty: L3 推导
tech_stack: [Python, Pandas]
status: planned
prerequisites: [PY-072, PY-073, PY-074]
related_math: []
related: []
notion_page_id: 40b663e8-0333-815f-9fed-fe59197ee0de
notion_url: https://app.notion.com/p/Pandas-40b663e80333815f9fedfe59197ee0de
created: 2026-11-01
updated: 2026-11-01
notion_db: 实战项目
所属学科: 科学计算
项目状态: 未开始
项目阶段: 立项
难度: L3 推导
项目目标: 造数→清洗→分组→关联→结论 的完整分析流水线。
技术栈: [Pandas, NumPy]
对应课程模块: [M-PY-08]
前置知识: [PY-072, PY-073, PY-074]
tags: [科学计算]
---

# PROJ-007 Pandas 数据分析项目

## 1. 项目目标

从"原始脏数据"走到"可读结论"：**造数据 → 检查 → 清洗 → 分组 → 关联 → 输出洞察**，全程可复现。

- 完成后应能证明：你会做完整分析流水线，并且每一步都能说清为什么。
- 工作量：约 4 小时。
- 难度：L3（中级）。

## 2. 前置知识

| 知识点 | 为什么需要 |
|---|---|
| [[PY-070 数据读取与写出]] | 读写 CSV |
| [[PY-072 缺失值处理]] | 清洗 |
| [[PY-073 分组聚合]] | 汇总 |
| [[PY-074 merge 与 concat]] | 表关联 |
| [[PY-075 数据转换与时间序列]] | 日期处理 |

## 3. 需求说明

| 编号 | 需求 | 验收 |
|---|---|---|
| F1 | 生成 120 天、含 5 个缺失的销售数据 | 行数正确 |
| F2 | 清洗并**记录删了几行** | 打印删除数量 |
| F3 | 按月、按周各汇总一次 | 两个结果形状不同 |
| F4 | 关联"城市"维表，按城市汇总 | 无匹配的行要说明 |
| F5 | 输出 3 条文字结论 | 每条都能从数据验证 |

## 4. 项目目录结构

```text
sales_analysis/
├── analyze.py    # 主流程
├── data/         # 运行时生成的中间文件（可删）
└── report.md     # 结论输出
```

## 5. 实现步骤

1. 用 `default_rng(seed)` 造日期与销售额，故意插入 NaN。
2. 打印 `shape` 与缺失数——**先看再洗**。
3. `dropna(subset=["sales"])` 后打印删了多少行。
4. 用 `to_period("M")` / `to_period("W")` 分组汇总。
5. `merge` 城市维表，统计各城市销售额。
6. 打印结论：最高月份、城市数、是否有未匹配。

## 6. 完整参考代码

```powershell
python analyze.py
```

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
dates = pd.date_range("2026-01-01", periods=120, freq="D")
sales = rng.integers(50, 200, size=120).astype(float)
sales[[1, 20, 40, 60, 80]] = np.nan

df = pd.DataFrame({"date": dates, "sales": sales})
print("原始:", df.shape, "缺失:", int(df["sales"].isna().sum()))

before = len(df)
df = df.dropna(subset=["sales"]).copy()
print("清洗后:", len(df), "删除:", before - len(df))

df["month"] = df["date"].dt.to_period("M").astype(str)
df["week"] = df["date"].dt.to_period("W").astype(str)

monthly = df.groupby("month")["sales"].agg(天数="count", 总额="sum")
weekly = df.groupby("week")["sales"].sum()
print("按月:", monthly.shape, "按周:", weekly.shape)

cities = pd.DataFrame({
    "month": ["2026-01", "2026-02", "2026-03", "2026-04"],
    "city": ["北京", "北京", "上海", "上海"],
})
merged = monthly.reset_index().merge(cities, on="month", how="left")
print("未匹配城市数:", int(merged["city"].isna().sum()))

by_city = merged.dropna(subset=["city"]).groupby("city")["总额"].sum()
print("城市销售额:", by_city.round(1).to_dict())

print("结论1 最高月份:", monthly["总额"].idxmax())
print("结论2 有销售记录的天数:", int(len(df)))
print("结论3 月总额方差大于 0:", bool(monthly["总额"].var() > 0))
```

## 7. 运行说明

```powershell
cd sales_analysis
python analyze.py
```

**预期输出**（数值随种子固定，但具体金额不在文档写死）

```
原始: (120, 2) 缺失: 5
清洗后: 115 删除: 5
按月: (4, 2) 按周: (18,)
未匹配城市数: 0
城市销售额: {'上海': ..., '北京': ...}
结论1 最高月份: 2026-0?
结论2 有销售记录的天数: 115
结论3 月总额方差大于 0: True
```

> 只写结构（形状、计数、布尔），不写具体金额——金额会随 NumPy 版本与算法变化，写死等于伪造。

## 8. 扩展挑战

1. **按城市×月份透视**：用 `pivot_table` 生成二维汇总表。
2. **同比环比**：算月环比增长率（`pct_change`）。
3. **异常检测**：找出日销售额超过均值 2 倍标准差的日子。
4. **导出报告**：把三条结论写进 `report.md`。
5. **换数据源**：把造数据换成读 CSV，其余流程不变（体会"分析逻辑与数据源解耦"）。

## 9. 复盘要点

- 为什么先打印缺失再清洗，而不是直接洗？
- `to_period("W")` 的"周"从周几开始？你的业务口径一致吗？
- merge 出现未匹配时，用 `how="left"` 还是 `"inner"`，结论会差多少？

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
| 项目目标 | 造数→清洗→分组→关联→结论 的完整分析流水线。 |
| 技术栈 | Pandas、NumPy |

- **对应课程模块**：[M-PY-08 Pandas 与数据可视化](../../01_Python%E5%9F%BA%E7%A1%80/%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-PY-08%20Pandas%20%E4%B8%8E%E6%95%B0%E6%8D%AE%E5%8F%AF%E8%A7%86%E5%8C%96.md)
- **前置知识**：[PY-072 缺失值处理](../../01_Python%E5%9F%BA%E7%A1%80/09%20Pandas%E4%B8%8E%E5%8F%AF%E8%A7%86%E5%8C%96/PY-072%20%E7%BC%BA%E5%A4%B1%E5%80%BC%E5%A4%84%E7%90%86.md) · [PY-073 分组聚合](../../01_Python%E5%9F%BA%E7%A1%80/09%20Pandas%E4%B8%8E%E5%8F%AF%E8%A7%86%E5%8C%96/PY-073%20%E5%88%86%E7%BB%84%E8%81%9A%E5%90%88.md) · [PY-074 merge 与 concat](../../01_Python%E5%9F%BA%E7%A1%80/09%20Pandas%E4%B8%8E%E5%8F%AF%E8%A7%86%E5%8C%96/PY-074%20merge%20%E4%B8%8E%20concat.md)

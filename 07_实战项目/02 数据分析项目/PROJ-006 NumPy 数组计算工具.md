---
id: PROJ-006
title: NumPy 数组计算工具
domain: PROJ
module: []
type: project
difficulty: L2 应用
tech_stack: [Python, NumPy]
status: planned
prerequisites: [PY-063, PY-065]
related_math: [MATH-201]
related: []
notion_page_id: 40b663e8-0333-818a-bc16-d9b0ffd2bd12
notion_url: https://app.notion.com/p/NumPy-40b663e80333818abc16d9b0ffd2bd12
created: 2026-11-01
updated: 2026-11-01
notion_db: 实战项目
所属学科: 科学计算
项目状态: 未开始
项目阶段: 立项
难度: L2 应用
项目目标: 滑动平均/归一化/标准化/统计摘要，全程向量化。
技术栈: [NumPy]
对应课程模块: [M-PY-07]
前置知识: [PY-063, PY-065]
tags: [科学计算]
---

# PROJ-006 NumPy 数组计算工具

## 1. 项目目标

写一组可复用的数组工具函数：**滑动平均、归一化、标准化、统计摘要**，全部向量化实现，不写 Python 循环。

- 完成后应能证明：你会把"循环思维"换成"数组思维"。
- 工作量：约 2–3 小时。
- 难度：L2（初级）。

## 2. 前置知识

| 知识点 | 为什么需要 |
|---|---|
| [[PY-059 NumPy 与列表的区别]] | 理解数组 |
| [[PY-063 数组运算]] | 逐元素运算 |
| [[PY-065 向量化与性能]] | 去掉循环 |
| [[PY-067 聚合与统计]] | 均值与标准差 |

## 3. 需求说明

| 编号 | 需求 | 验收 |
|---|---|---|
| F1 | `moving_average(x, w)` | 窗口非法时抛 `ValueError` |
| F2 | `normalize(x)` | 结果落在 [0,1]；常数序列返回全 0 |
| F3 | `standardize(x)` | 均值 0、标准差 1 |
| F4 | `summary(x)` | 返回 count/mean/std/min/max |
| F5 | 全程不使用 Python for 循环 | 代码里没有 for |

## 4. 项目目录结构

```text
nptools/
├── nptools.py   # 工具函数
├── demo.py      # 演示脚本
└── README.md
```

## 5. 实现步骤

1. 用 `np.convolve` 实现滑动平均（`mode="valid"` 保证长度 = n-w+1）。
2. 归一化：减最小值、除极差；极差为 0 时返回全 0（避免除零）。
3. 标准化：减均值、除标准差；标准差为 0 时同样保护。
4. 摘要用 `np.mean/std/min/max` 一次算齐。
5. 演示：对一组数据跑四个函数并打印。

## 6. 完整参考代码

```powershell
python demo.py
```

```python
import numpy as np


def moving_average(x, w=3):
    x = np.asarray(x, dtype=float)
    if w <= 0 or w > len(x):
        raise ValueError("窗口必须满足 1 <= w <= 序列长度")
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="valid").round(3)


def normalize(x):
    x = np.asarray(x, dtype=float)
    lo, hi = float(x.min()), float(x.max())
    if hi == lo:
        return np.zeros_like(x)
    return ((x - lo) / (hi - lo)).round(3)


def standardize(x):
    x = np.asarray(x, dtype=float)
    std = float(x.std())
    if std == 0:
        return np.zeros_like(x)
    return ((x - x.mean()) / std).round(3)


def summary(x):
    x = np.asarray(x, dtype=float)
    return {
        "count": int(x.size),
        "mean": round(float(x.mean()), 3),
        "std": round(float(x.std()), 3),
        "min": round(float(x.min()), 3),
        "max": round(float(x.max()), 3),
    }


data = [10, 12, 14, 13, 15, 18]
print("滑动平均:", moving_average(data, 3).tolist())
print("归一化:", normalize(data).tolist())
print("标准化:", standardize(data).tolist())
print("摘要:", summary(data))

try:
    moving_average(data, 10)
except ValueError as e:
    print("拒绝:", e)
```

## 7. 运行说明

```powershell
cd nptools
python demo.py
```

**预期输出**

```
滑动平均: [12.0, 13.0, 14.0, 15.333]
归一化: [0.0, 0.25, 0.5, 0.375, 0.625, 1.0]
标准化: [-1.47, -0.668, 0.134, -0.267, 0.535, 1.737]
摘要: {'count': 6, 'mean': 13.667, 'std': 2.494, 'min': 10.0, 'max': 18.0}
拒绝: 窗口必须满足 1 <= w <= 序列长度
```

## 8. 扩展挑战

1. **加权滑动平均**：支持传入权重数组（提示：`np.convolve` 的 kernel 直接换掉）。
2. **二维支持**：对 `(n, m)` 矩阵按列做滑动平均。
3. **离群值裁剪**：实现 `clip_outliers(x, k=1.5)`，用 IQR 规则截断。
4. **性能对比**：用列表推导实现同样的滑动平均，比较结果是否一致（不写耗时）。
5. **接口封装**：把四个函数收进 `class ArrayTool`，支持链式调用。

## 9. 复盘要点

- `np.convolve(mode="valid")` 为什么长度是 `n-w+1`？
- 归一化与标准化分别在什么场景用？
- 常数序列为什么要特判？不判会发生什么？

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
| 难度 | L2 应用 |
| 项目目标 | 滑动平均/归一化/标准化/统计摘要，全程向量化。 |
| 技术栈 | NumPy |

- **对应课程模块**：[M-PY-07 NumPy 科学计算](../../01_Python%E5%9F%BA%E7%A1%80/%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-PY-07%20NumPy%20%E7%A7%91%E5%AD%A6%E8%AE%A1%E7%AE%97.md)
- **前置知识**：[PY-063 数组运算](../../01_Python%E5%9F%BA%E7%A1%80/08%20NumPy/PY-063%20%E6%95%B0%E7%BB%84%E8%BF%90%E7%AE%97.md) · [PY-065 向量化与性能](../../01_Python%E5%9F%BA%E7%A1%80/08%20NumPy/PY-065%20%E5%90%91%E9%87%8F%E5%8C%96%E4%B8%8E%E6%80%A7%E8%83%BD.md)

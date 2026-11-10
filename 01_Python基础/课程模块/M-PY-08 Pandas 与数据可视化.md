---
id: M-PY-08
title: Pandas 与数据可视化
type: module
notion_page_id: 40b663e8-0333-8106-a1e0-d76b5e38fa26
notion_url: https://app.notion.com/p/40b663e803338106a1e0d76b5e38fa26
notion_db: 课程模块
学科: Python基础
学习阶段: 基础
排序: 8
当前状态: 未开始
学习目标: 掌握 Series/DataFrame、读写、清洗、分组、合并、时间序列与绘图。
所属路线: [RM-001]
知识点: [PY-069, PY-070, PY-071, PY-072, PY-073, PY-074, PY-075, PY-076, PY-077, PY-078]
实战项目: [PROJ-007, PROJ-008]
tags: [Python基础]
---

# M-PY-08 Pandas 与数据可视化

> 本笔记由 Notion「课程模块」数据库导出生成（Obsidian 侧原无对应笔记）。

## 学习目标

掌握 Series/DataFrame、读写、清洗、分组、合并、时间序列与绘图。

---

## 🔗 关联

> 以下内容由 Notion「AI Knowledge System · 课程模块」数据库按知识 ID 合并而来（导出于 2026-11-09）。

| 属性 | 值 |
|---|---|
| 学科 | Python基础 |
| 学习阶段 | 基础 |
| 排序 | 8 |
| 当前状态 | 未开始 |

- **所属路线**：[RM-001 Python 基础路线](../../00_%E7%9F%A5%E8%AF%86%E9%A6%96%E9%A1%B5/%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF/RM-001%20Python%20%E5%9F%BA%E7%A1%80%E8%B7%AF%E7%BA%BF.md)
- **知识点**：[PY-069 Series 与 DataFrame](../09%20Pandas%E4%B8%8E%E5%8F%AF%E8%A7%86%E5%8C%96/PY-069%20Series%20%E4%B8%8E%20DataFrame.md) · [PY-070 数据读取与写出](../09%20Pandas%E4%B8%8E%E5%8F%AF%E8%A7%86%E5%8C%96/PY-070%20%E6%95%B0%E6%8D%AE%E8%AF%BB%E5%8F%96%E4%B8%8E%E5%86%99%E5%87%BA.md) · [PY-071 筛选与排序](../09%20Pandas%E4%B8%8E%E5%8F%AF%E8%A7%86%E5%8C%96/PY-071%20%E7%AD%9B%E9%80%89%E4%B8%8E%E6%8E%92%E5%BA%8F.md) · [PY-072 缺失值处理](../09%20Pandas%E4%B8%8E%E5%8F%AF%E8%A7%86%E5%8C%96/PY-072%20%E7%BC%BA%E5%A4%B1%E5%80%BC%E5%A4%84%E7%90%86.md) · [PY-073 分组聚合](../09%20Pandas%E4%B8%8E%E5%8F%AF%E8%A7%86%E5%8C%96/PY-073%20%E5%88%86%E7%BB%84%E8%81%9A%E5%90%88.md) · [PY-074 merge 与 concat](../09%20Pandas%E4%B8%8E%E5%8F%AF%E8%A7%86%E5%8C%96/PY-074%20merge%20%E4%B8%8E%20concat.md) · [PY-075 数据转换与时间序列](../09%20Pandas%E4%B8%8E%E5%8F%AF%E8%A7%86%E5%8C%96/PY-075%20%E6%95%B0%E6%8D%AE%E8%BD%AC%E6%8D%A2%E4%B8%8E%E6%97%B6%E9%97%B4%E5%BA%8F%E5%88%97.md) · [PY-076 Matplotlib 基础](../09%20Pandas%E4%B8%8E%E5%8F%AF%E8%A7%86%E5%8C%96/PY-076%20Matplotlib%20%E5%9F%BA%E7%A1%80.md) · [PY-077 图表设计与可读性](../09%20Pandas%E4%B8%8E%E5%8F%AF%E8%A7%86%E5%8C%96/PY-077%20%E5%9B%BE%E8%A1%A8%E8%AE%BE%E8%AE%A1%E4%B8%8E%E5%8F%AF%E8%AF%BB%E6%80%A7.md) · [PY-078 数据分析综合实践](../09%20Pandas%E4%B8%8E%E5%8F%AF%E8%A7%86%E5%8C%96/PY-078%20%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%E7%BB%BC%E5%90%88%E5%AE%9E%E8%B7%B5.md)
- **实战项目**：[PROJ-007 Pandas 数据分析项目](../../07_%E5%AE%9E%E6%88%98%E9%A1%B9%E7%9B%AE/02%20%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%E9%A1%B9%E7%9B%AE/PROJ-007%20Pandas%20%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%E9%A1%B9%E7%9B%AE.md) · [PROJ-008 数据可视化项目](../../07_%E5%AE%9E%E6%88%98%E9%A1%B9%E7%9B%AE/02%20%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%E9%A1%B9%E7%9B%AE/PROJ-008%20%E6%95%B0%E6%8D%AE%E5%8F%AF%E8%A7%86%E5%8C%96%E9%A1%B9%E7%9B%AE.md)

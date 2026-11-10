---
id: M-PY-05
title: 文件、异常与数据持久化
type: module
notion_page_id: 40b663e8-0333-81f3-82ae-f9145c99bc19
notion_url: https://app.notion.com/p/40b663e8033381f382aef9145c99bc19
notion_db: 课程模块
学科: Python基础
学习阶段: 基础
排序: 5
当前状态: 未开始
学习目标: 掌握文件读写、路径处理、CSV/JSON、异常、上下文管理器、日志与 SQLite。
知识点: [PY-041, PY-042, PY-043, PY-044, PY-045, PY-046, PY-047, PY-048]
实战项目: [PROJ-003, PROJ-004, PROJ-005]
tags: [Python基础]
---

# M-PY-05 文件、异常与数据持久化

> 本笔记由 Notion「课程模块」数据库导出生成（Obsidian 侧原无对应笔记）。

## 学习目标

掌握文件读写、路径处理、CSV/JSON、异常、上下文管理器、日志与 SQLite。

---

## 🔗 关联

> 以下内容由 Notion「AI Knowledge System · 课程模块」数据库按知识 ID 合并而来（导出于 2026-11-09）。

| 属性 | 值 |
|---|---|
| 学科 | Python基础 |
| 学习阶段 | 基础 |
| 排序 | 5 |
| 当前状态 | 未开始 |

- **知识点**：[PY-041 文件读写](../05%20%E5%BC%82%E5%B8%B8%E4%B8%8E%E6%96%87%E4%BB%B6/PY-041%20%E6%96%87%E4%BB%B6%E8%AF%BB%E5%86%99.md) · [PY-042 路径与 pathlib](../05%20%E5%BC%82%E5%B8%B8%E4%B8%8E%E6%96%87%E4%BB%B6/PY-042%20%E8%B7%AF%E5%BE%84%E4%B8%8E%20pathlib.md) · [PY-043 CSV 与 JSON](../05%20%E5%BC%82%E5%B8%B8%E4%B8%8E%E6%96%87%E4%BB%B6/PY-043%20CSV%20%E4%B8%8E%20JSON.md) · [PY-044 异常处理](../05%20%E5%BC%82%E5%B8%B8%E4%B8%8E%E6%96%87%E4%BB%B6/PY-044%20%E5%BC%82%E5%B8%B8%E5%A4%84%E7%90%86.md) · [PY-045 自定义异常与上下文管理器](../05%20%E5%BC%82%E5%B8%B8%E4%B8%8E%E6%96%87%E4%BB%B6/PY-045%20%E8%87%AA%E5%AE%9A%E4%B9%89%E5%BC%82%E5%B8%B8%E4%B8%8E%E4%B8%8A%E4%B8%8B%E6%96%87%E7%AE%A1%E7%90%86%E5%99%A8.md) · [PY-046 logging 日志](../05%20%E5%BC%82%E5%B8%B8%E4%B8%8E%E6%96%87%E4%BB%B6/PY-046%20logging%20%E6%97%A5%E5%BF%97.md) · [PY-047 SQLite 基础](../05%20%E5%BC%82%E5%B8%B8%E4%B8%8E%E6%96%87%E4%BB%B6/PY-047%20SQLite%20%E5%9F%BA%E7%A1%80.md) · [PY-048 数据持久化方案](../05%20%E5%BC%82%E5%B8%B8%E4%B8%8E%E6%96%87%E4%BB%B6/PY-048%20%E6%95%B0%E6%8D%AE%E6%8C%81%E4%B9%85%E5%8C%96%E6%96%B9%E6%A1%88.md)
- **实战项目**：[PROJ-003 文件批量处理工具](../../07_%E5%AE%9E%E6%88%98%E9%A1%B9%E7%9B%AE/01%20Python%E9%A1%B9%E7%9B%AE/PROJ-003%20%E6%96%87%E4%BB%B6%E6%89%B9%E9%87%8F%E5%A4%84%E7%90%86%E5%B7%A5%E5%85%B7.md) · [PROJ-004 JSON 数据管理工具](../../07_%E5%AE%9E%E6%88%98%E9%A1%B9%E7%9B%AE/01%20Python%E9%A1%B9%E7%9B%AE/PROJ-004%20JSON%20%E6%95%B0%E6%8D%AE%E7%AE%A1%E7%90%86%E5%B7%A5%E5%85%B7.md) · [PROJ-005 简易 SQLite 数据库应用](../../07_%E5%AE%9E%E6%88%98%E9%A1%B9%E7%9B%AE/01%20Python%E9%A1%B9%E7%9B%AE/PROJ-005%20%E7%AE%80%E6%98%93%20SQLite%20%E6%95%B0%E6%8D%AE%E5%BA%93%E5%BA%94%E7%94%A8.md)

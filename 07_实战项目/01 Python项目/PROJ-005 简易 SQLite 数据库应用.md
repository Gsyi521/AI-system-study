---
id: PROJ-005
title: 简易 SQLite 数据库应用
domain: PROJ
module: []
type: project
difficulty: L3 推导
tech_stack: [Python, SQLite]
status: planned
prerequisites: [PY-047, PY-044]
related_math: []
related: []
notion_page_id: 40b663e8-0333-8138-8131-cbc734dccd1d
notion_url: https://app.notion.com/p/SQLite-40b663e8033381388131cbc734dccd1d
created: 2026-11-01
updated: 2026-11-01
notion_db: 实战项目
所属学科: Python基础
项目状态: 未开始
项目阶段: 立项
难度: L3 推导
项目目标: 通讯录 CRUD，唯一约束防重复，参数化防注入。
技术栈: [Python, SQLite]
对应课程模块: [M-PY-05]
前置知识: [PY-047, PY-044]
tags: [Python基础]
---

# PROJ-005 简易 SQLite 数据库应用

## 1. 项目目标

做一个通讯录：**建表、增、查、统计、防重复**，全部用参数化查询，并用唯一约束处理重复录入。

- 完成后应能证明：你会用 `sqlite3` 做真实 CRUD，知道"约束"比"先查再插"更可靠。
- 工作量：约 3 小时。
- 难度：L3（中级）。

## 2. 前置知识

| 知识点 | 为什么需要 |
|---|---|
| [[PY-047 SQLite 基础]] | 连接、执行、提交 |
| [[PY-044 异常处理]] | 捕获 `IntegrityError` |
| [[PY-048 数据持久化方案]] | 判断何时该用数据库 |
| [[PY-041 文件读写]] | 理解文件型数据库 |

## 3. 需求说明

| 编号 | 需求 | 验收 |
|---|---|---|
| F1 | 建 `contact` 表（姓名唯一） | 重复插入失败 |
| F2 | 添加联系人 | 成功返回 True，重复返回 False |
| F3 | 按姓名查电话 | 不存在返回 None |
| F4 | 统计总人数 | 数字正确 |
| F5 | 参数化查询 | 传入 `' OR '1'='1` 查不到数据 |

## 4. 项目目录结构

```text
contacts/
├── contacts.py    # 数据库操作
├── demo.py        # 演示脚本
└── contacts.db    # 运行时生成
```

## 5. 实现步骤

1. 连接数据库，建表（`name TEXT UNIQUE`）。
2. `add()`：用 `try/except sqlite3.IntegrityError` 处理重复。
3. `find()`：参数化查询，无结果返回 `None`。
4. 统计：`SELECT COUNT(*)`。
5. 演示：插入两个 → 查一个 → 重复插入 → 新增一个 → 统计 → 关闭并删除库文件。

## 6. 完整参考代码

```powershell
python demo.py
```

```python
import sqlite3
from pathlib import Path

db = Path("contacts.db")
conn = sqlite3.connect(db)
conn.execute(
    "CREATE TABLE contact (id INTEGER PRIMARY KEY, name TEXT UNIQUE, phone TEXT)"
)
conn.executemany(
    "INSERT INTO contact (name, phone) VALUES (?, ?)",
    [("小明", "13800000001"), ("小红", "13800000002")],
)
conn.commit()


def find(name):
    row = conn.execute("SELECT phone FROM contact WHERE name = ?", (name,)).fetchone()
    return row[0] if row else None


def add(name, phone):
    try:
        conn.execute("INSERT INTO contact (name, phone) VALUES (?, ?)", (name, phone))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


print(find("小明"))
print("重复添加:", add("小明", "1"))
print("新增:", add("小刚", "13800000003"))
print("总人数:", conn.execute("SELECT COUNT(*) FROM contact").fetchone()[0])
print("注入尝试:", find("' OR '1'='1"))

conn.close()
db.unlink()
```

## 7. 运行说明

```powershell
cd contacts
python demo.py
```

**预期输出**

```
13800000001
重复添加: False
新增: True
总人数: 3
注入尝试: None
```

> 最后一行是关键：`' OR '1'='1` 被当成普通字符串，没有查出任何数据——参数化避免了注入。

## 8. 扩展挑战

1. **更新与删除**：加 `update(name, phone)`、`remove(name)`，返回是否影响行数。
2. **模糊查询**：支持按姓名片段查（`LIKE ?`，记得加 `%`）。
3. **分页**：每页 10 条，用 `LIMIT ? OFFSET ?`。
4. **事务**：批量导入 1000 条，用一次 `commit` 而不是每条都提交，比较耗时差异。
5. **迁移**：给表加一列 `email`，写一个"检测列是否存在，不存在则 ALTER"的函数。

## 9. 复盘要点

- 为什么用 `UNIQUE` 约束，而不是"先 SELECT 再 INSERT"？
- `fetchone()` 与 `fetchall()` 分别在什么场景用？
- 忘记 `commit()` 会怎样？数据库文件里会有数据吗？

## 10. 学习记录

| 日期 | 状态 | 用时 | 遇到的问题 | 复盘 |
|---|---|---|---|---|
| — | planned | — | — | — |

---

## 🔗 Notion 同步信息

> 以下内容由 Notion「AI Knowledge System · 实战项目」数据库按知识 ID 合并而来（导出于 2026-11-09）。

| 属性 | 值 |
|---|---|
| 所属学科 | Python基础 |
| 项目状态 | 未开始 |
| 项目阶段 | 立项 |
| 难度 | L3 推导 |
| 项目目标 | 通讯录 CRUD，唯一约束防重复，参数化防注入。 |
| 技术栈 | Python、SQLite |

- **对应课程模块**：[M-PY-05 文件、异常与数据持久化](../../01_Python%E5%9F%BA%E7%A1%80/%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-PY-05%20%E6%96%87%E4%BB%B6%E3%80%81%E5%BC%82%E5%B8%B8%E4%B8%8E%E6%95%B0%E6%8D%AE%E6%8C%81%E4%B9%85%E5%8C%96.md)
- **前置知识**：[PY-047 SQLite 基础](../../01_Python%E5%9F%BA%E7%A1%80/05%20%E5%BC%82%E5%B8%B8%E4%B8%8E%E6%96%87%E4%BB%B6/PY-047%20SQLite%20%E5%9F%BA%E7%A1%80.md) · [PY-044 异常处理](../../01_Python%E5%9F%BA%E7%A1%80/05%20%E5%BC%82%E5%B8%B8%E4%B8%8E%E6%96%87%E4%BB%B6/PY-044%20%E5%BC%82%E5%B8%B8%E5%A4%84%E7%90%86.md)

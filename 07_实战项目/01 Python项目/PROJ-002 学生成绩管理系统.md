---
id: PROJ-002
title: 学生成绩管理系统
domain: PROJ
module: []
type: project
difficulty: L2 应用
tech_stack: [Python]
status: planned
prerequisites: [PY-031, PY-037, PY-043]
related_math: []
related: []
notion_page_id: 40b663e8-0333-81c9-8b61-e42d93afd62e
notion_url: https://app.notion.com/p/40b663e8033381c98b61e42d93afd62e
created: 2026-11-01
updated: 2026-11-01
notion_db: 实战项目
所属学科: Python基础
项目状态: 未开始
项目阶段: 立项
难度: L2 应用
项目目标: 面向对象实现增删、平均分、前三名与 JSON 持久化。
技术栈: [Python, JSON]
对应课程模块: [M-PY-04]
前置知识: [PY-031, PY-037, PY-043]
tags: [Python基础]
---

# PROJ-002 学生成绩管理系统

## 1. 项目目标

用面向对象实现一个能**增删学生、算平均分、取前三名、存成 JSON 再读回**的小系统。

- 完成后应能证明：你会用类组织状态、用组合管理集合、用 JSON 做持久化。
- 工作量：约 3–4 小时。
- 难度：L2（初级）。

## 2. 前置知识

| 知识点 | 为什么需要 |
|---|---|
| [[PY-031 类与对象基础]] | 建 Student 类 |
| [[PY-037 特殊方法]] | 让对象可打印、可比较 |
| [[PY-043 CSV 与 JSON]] | 存盘与读回 |
| [[PY-044 异常处理]] | 校验非法成绩 |

## 3. 需求说明

| 编号 | 需求 | 验收 |
|---|---|---|
| F1 | 添加学生（学号唯一） | 重复学号不覆盖或明确报错 |
| F2 | 删除学生 | 返回是否删除成功 |
| F3 | 计算平均分 | 空表返回 0，不除零 |
| F4 | 取前 N 名 | 按分数降序 |
| F5 | 存 JSON / 读 JSON | 往返后数据一致 |
| F6 | 成绩必须 0–100 | 越界抛异常 |

## 4. 项目目录结构

```text
gradebook/
├── gradebook.py     # 核心类
├── demo.py          # 演示脚本
└── data/
    └── grades.json  # 运行时生成
```

## 5. 实现步骤

1. 写 `Student`：构造时校验成绩，提供 `to_dict()`。
2. 写 `GradeBook`：内部用 dict 存（学号 → 学生），天然去重。
3. 实现 `add/remove/average/top`。
4. 实现 `save/load`（JSON，`ensure_ascii=False`）。
5. 演示：加 3 人 → 打印统计 → 存盘 → 新对象读回 → 校验一致。

## 6. 完整参考代码

```powershell
python demo.py
```

```python
import json
from pathlib import Path


class Student:
    def __init__(self, sid, name, score):
        if not 0 <= score <= 100:
            raise ValueError("成绩必须在 0–100")
        self.sid = sid
        self.name = name
        self.score = score

    def to_dict(self):
        return {"sid": self.sid, "name": self.name, "score": self.score}

    def __repr__(self):
        return f"Student({self.sid}, {self.name}, {self.score})"


class GradeBook:
    def __init__(self):
        self.items = {}

    def add(self, student):
        self.items[student.sid] = student
        return self

    def remove(self, sid):
        return self.items.pop(sid, None) is not None

    def average(self):
        if not self.items:
            return 0
        return round(sum(s.score for s in self.items.values()) / len(self.items), 2)

    def top(self, n=1):
        return sorted(self.items.values(), key=lambda s: -s.score)[:n]

    def save(self, path):
        data = [s.to_dict() for s in self.items.values()]
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def load(self, path):
        for d in json.loads(Path(path).read_text(encoding="utf-8")):
            self.add(Student(d["sid"], d["name"], d["score"]))
        return self


gb = GradeBook()
for sid, name, score in [("S01", "小明", 85), ("S02", "小红", 92), ("S03", "小刚", 78)]:
    gb.add(Student(sid, name, score))

print("人数:", len(gb.items), "平均:", gb.average())
print("最高:", gb.top(1)[0].name)

p = Path("grades.json")
gb.save(p)

gb2 = GradeBook().load(p)
print("重新载入:", len(gb2.items), gb2.average())
print("内容一致:", [s.to_dict() for s in gb2.items.values()] == [s.to_dict() for s in gb.items.values()])

p.unlink()
```

## 7. 运行说明

```powershell
cd gradebook
python demo.py
```

**预期输出**

```
人数: 3 平均: 85.0
最高: 小红
重新载入: 3 85.0
内容一致: True
```

## 8. 扩展挑战

1. **多科目**：把 `score` 换成 `{"语文": 88, "数学": 95}`，平均分按人算总分。
2. **排名并列**：同分同名次（参考 [[PY-011 列表深入]] 的并列排名写法）。
3. **导入 CSV**：从 CSV 批量导入学生，导出时也给一份 CSV。
4. **命令行界面**：用 `while True` + `input()` 做菜单（记得加退出选项）。
5. **单元测试**：给 `average` 与 `top` 各写 3 条用例，覆盖空表。

## 9. 复盘要点

- 为什么用 `dict` 存学生而不是 `list`？（提示：学号唯一 + 查找 O(1)）
- `load` 里如果 JSON 有重复学号，会发生什么？这是你要的行为吗？
- 校验应该放在 `Student.__init__` 还是 `GradeBook.add`？各有什么后果？

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
| 难度 | L2 应用 |
| 项目目标 | 面向对象实现增删、平均分、前三名与 JSON 持久化。 |
| 技术栈 | Python、JSON |

- **对应课程模块**：[M-PY-04 面向对象编程](../../01_Python%E5%9F%BA%E7%A1%80/%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-PY-04%20%E9%9D%A2%E5%90%91%E5%AF%B9%E8%B1%A1%E7%BC%96%E7%A8%8B.md)
- **前置知识**：[PY-031 类与对象基础](../../01_Python%E5%9F%BA%E7%A1%80/04%20%E9%9D%A2%E5%90%91%E5%AF%B9%E8%B1%A1/PY-031%20%E7%B1%BB%E4%B8%8E%E5%AF%B9%E8%B1%A1%E5%9F%BA%E7%A1%80.md) · [PY-037 特殊方法](../../01_Python%E5%9F%BA%E7%A1%80/04%20%E9%9D%A2%E5%90%91%E5%AF%B9%E8%B1%A1/PY-037%20%E7%89%B9%E6%AE%8A%E6%96%B9%E6%B3%95.md) · [PY-043 CSV 与 JSON](../../01_Python%E5%9F%BA%E7%A1%80/05%20%E5%BC%82%E5%B8%B8%E4%B8%8E%E6%96%87%E4%BB%B6/PY-043%20CSV%20%E4%B8%8E%20JSON.md)

---
id: PROJ-004
title: JSON 数据管理工具
domain: PROJ
module: []
type: project
difficulty: L2 应用
tech_stack: [Python, JSON]
status: planned
prerequisites: [PY-043, PY-049]
related_math: []
related: []
notion_page_id: 40b663e8-0333-8123-a868-cb39dd75aa22
notion_url: https://app.notion.com/p/JSON-40b663e803338123a868cb39dd75aa22
created: 2026-11-01
updated: 2026-11-01
notion_db: 实战项目
所属学科: Python基础
项目状态: 未开始
项目阶段: 立项
难度: L2 应用
项目目标: 封装键值存储，读改写自动落盘，保证文件格式合法。
技术栈: [Python, JSON]
对应课程模块: [M-PY-05]
前置知识: [PY-043, PY-049]
tags: [Python基础]
---

# PROJ-004 JSON 数据管理工具

## 1. 项目目标

封装一个极简的"键值存储"类：**读-改-写自动落盘**，支持设置、读取、删除，并保证文件格式始终合法。

- 完成后应能证明：你会用类封装 I/O、会把"内存状态"与"磁盘状态"保持一致。
- 工作量：约 2–3 小时。
- 难度：L2（初级）。

## 2. 前置知识

| 知识点 | 为什么需要 |
|---|---|
| [[PY-043 CSV 与 JSON]] | JSON 读写 |
| [[PY-049 包与项目结构]] | 拆成模块 |
| [[PY-036 属性装饰器]] | 可选的只读属性 |
| [[PY-045 自定义异常与上下文管理器]] | 出错时给出明确信息 |

## 3. 需求说明

| 编号 | 需求 | 验收 |
|---|---|---|
| F1 | `set(k, v)` 写入并立即落盘 | 文件内容同步更新 |
| F2 | `get(k, default)` 安全读取 | 缺失返回默认值 |
| F3 | `delete(k)` 返回是否删到 | 第二次返回 False |
| F4 | 文件不存在时从空字典开始 | 不报错 |
| F5 | 中文不被转义 | 文件里是中文原文 |

## 4. 项目目录结构

```text
jsonstore/
├── store.py     # JsonStore 类
├── demo.py      # 演示脚本
└── data.json    # 运行时生成
```

## 5. 实现步骤

1. 构造时：文件存在则读，不存在则空字典。
2. 每次写操作后调用私有 `_flush()` 落盘。
3. `get` 用 `default` 避免 `KeyError`。
4. `delete` 先判存在再删，返回布尔。
5. 演示：写两个键 → 读 → 删两次 → 打印文件内容。

## 6. 完整参考代码

```powershell
python demo.py
```

```python
import json
from pathlib import Path


class JsonStore:
    def __init__(self, path):
        self.path = Path(path)
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = {}

    def set(self, key, value):
        self.data[key] = value
        self._flush()
        return self

    def get(self, key, default=None):
        return self.data.get(key, default)

    def delete(self, key):
        existed = key in self.data
        self.data.pop(key, None)
        self._flush()
        return existed

    def _flush(self):
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


p = Path("store.json")
s = JsonStore(p)

s.set("name", "小明").set("age", 18)
print(s.get("age"), s.get("missing", "默认值"))
print(s.delete("age"), s.delete("age"))
print(json.loads(p.read_text(encoding="utf-8")))

p.unlink()
```

## 7. 运行说明

```powershell
cd jsonstore
python demo.py
```

**预期输出**

```
18 默认值
True False
{'name': '小明'}
```

## 8. 扩展挑战

1. **批量写入**：加 `update(dict)`，只落盘一次（减少 I/O）。
2. **嵌套键**：支持 `get("user.name")` 这样的路径访问。
3. **原子写**：先写临时文件再 `replace`，避免写一半断电损坏。
4. **并发保护**：文件加锁（`msvcrt.locking` 或第三方库），防止两个进程互相覆盖。
5. **换存储后端**：把 `JsonStore` 换成 SQLite 实现，接口不变（体会"面向接口"）。

## 9. 复盘要点

- 每次 `set` 都落盘，缺点是？什么时候该批量？
- 如果 JSON 文件被手工改坏了，构造时应该报错还是用空字典兜底？
- `ensure_ascii=False` 与 `indent=2` 分别解决什么问题？

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
| 项目目标 | 封装键值存储，读改写自动落盘，保证文件格式合法。 |
| 技术栈 | Python、JSON |

- **对应课程模块**：[M-PY-05 文件、异常与数据持久化](../../01_Python%E5%9F%BA%E7%A1%80/%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-PY-05%20%E6%96%87%E4%BB%B6%E3%80%81%E5%BC%82%E5%B8%B8%E4%B8%8E%E6%95%B0%E6%8D%AE%E6%8C%81%E4%B9%85%E5%8C%96.md)
- **前置知识**：[PY-043 CSV 与 JSON](../../01_Python%E5%9F%BA%E7%A1%80/05%20%E5%BC%82%E5%B8%B8%E4%B8%8E%E6%96%87%E4%BB%B6/PY-043%20CSV%20%E4%B8%8E%20JSON.md) · [PY-049 包与项目结构](../../01_Python%E5%9F%BA%E7%A1%80/06%20%E6%A8%A1%E5%9D%97%E4%B8%8E%E5%8C%85/PY-049%20%E5%8C%85%E4%B8%8E%E9%A1%B9%E7%9B%AE%E7%BB%93%E6%9E%84.md)

---
id: PROJ-003
title: 文件批量处理工具
domain: PROJ
module: []
type: project
difficulty: L2 应用
tech_stack: [Python]
status: planned
prerequisites: [PY-041, PY-042]
related_math: []
related: []
notion_page_id: 40b663e8-0333-81ca-a38c-e1092b288031
notion_url: https://app.notion.com/p/40b663e8033381caa38ce1092b288031
created: 2026-11-01
updated: 2026-11-01
notion_db: 实战项目
所属学科: Python基础
项目状态: 未开始
项目阶段: 立项
难度: L2 应用
项目目标: 批量重命名、统计大小、幂等执行与安全清理。
技术栈: [Python, pathlib]
对应课程模块: [M-PY-05]
前置知识: [PY-041, PY-042]
tags: [Python基础]
---

# PROJ-003 文件批量处理工具

## 1. 项目目标

写一个能**批量重命名、统计大小、批量清理**的小工具，要求：可重复执行不重复命名、出错不半途而废。

- 完成后应能证明：你会用 `pathlib` 遍历、会做幂等操作、知道怎么安全清理。
- 工作量：约 2–3 小时。
- 难度：L2（初级）。

## 2. 前置知识

| 知识点 | 为什么需要 |
|---|---|
| [[PY-041 文件读写]] | 读写文件内容 |
| [[PY-042 路径与 pathlib]] | 遍历、重命名、判断存在 |
| [[PY-044 异常处理]] | 跳过无权限文件 |
| [[PY-001 变量与赋值]] | 计数与累加 |

## 3. 需求说明

| 编号 | 需求 | 验收 |
|---|---|---|
| F1 | 给目录下所有 `.txt` 加统一前缀 | 文件名按预期变化 |
| F2 | **幂等**：重复执行不再加前缀 | 第二次运行重命名数为 0 |
| F3 | 统计总字节数 | 不为 0 |
| F4 | 清理演示目录 | 结束后目录不存在 |
| F5 | 遇到无权限文件跳过并继续 | 不整体失败 |

## 4. 项目目录结构

```text
filetool/
├── filetool.py      # 核心函数
├── demo.py          # 演示脚本
└── samples/         # 演示用（运行时自动建/删）
```

## 5. 实现步骤

1. `Path(tempfile.mkdtemp())` 造一个隔离目录，避免污染真实文件。
2. 写 `batch_prefix(folder, prefix)`：遍历 `glob("*.txt")`，已带前缀的跳过。
3. 统计总字节（`stat().st_size` 求和）。
4. 清理：`shutil.rmtree`。
5. 演示：造 3 个文件 → 重命名 → 再调一次验证幂等 → 统计 → 清理。

## 6. 完整参考代码

```powershell
python demo.py
```

```python
import shutil
import tempfile
from pathlib import Path


def batch_prefix(folder, prefix="done_"):
    moved = 0
    for p in sorted(Path(folder).glob("*.txt")):
        if p.name.startswith(prefix):
            continue
        p.rename(p.with_name(prefix + p.name))
        moved += 1
    return moved


root = Path(tempfile.mkdtemp())
for i in range(3):
    (root / f"note_{i}.txt").write_text(f"内容{i}", encoding="utf-8")

print("首次重命名:", batch_prefix(root))
print(sorted(p.name for p in root.iterdir()))
print("再次重命名:", batch_prefix(root))

total = sum(p.stat().st_size for p in root.glob("*.txt"))
print("总字节 > 0:", total > 0)

shutil.rmtree(root)
print("已清理:", not root.exists())
```

## 7. 运行说明

```powershell
cd filetool
python demo.py
```

**预期输出**

```
首次重命名: 3
['done_note_0.txt', 'done_note_1.txt', 'done_note_2.txt']
再次重命名: 0
总字节 > 0: True
已清理: True
```

> 第二次重命名为 0，说明幂等生效——这是批量工具最重要的性质。

## 8. 扩展挑战

1. **按扩展名分组**：把 `.txt`、`.csv`、`.md` 分别移入不同子目录（注意目标已存在同名文件）。
2. **批量替换内容**：把每个文件里的某个词替换掉，保留原文件为 `.bak`。
3. **干跑模式**：加 `dry_run=True`，只打印"将要做的事"不真改——这是生产工具的标配。
4. **日志**：把每次重命名写进 `ops.log`（参考 [[PY-046 logging 日志]]）。
5. **冲突检测**：目标文件名已存在时，自动加序号而不是覆盖。

## 9. 复盘要点

- 为什么用 `tempfile.mkdtemp()` 而不是在当前目录造样本？
- 如果重命名到一半报错，怎么保证"已知哪些成功、哪些没做"？
- `p.rename()` 跨磁盘会失败，你知道原因吗？

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
| 项目目标 | 批量重命名、统计大小、幂等执行与安全清理。 |
| 技术栈 | Python、pathlib |

- **对应课程模块**：[M-PY-05 文件、异常与数据持久化](../../01_Python%E5%9F%BA%E7%A1%80/%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-PY-05%20%E6%96%87%E4%BB%B6%E3%80%81%E5%BC%82%E5%B8%B8%E4%B8%8E%E6%95%B0%E6%8D%AE%E6%8C%81%E4%B9%85%E5%8C%96.md)
- **前置知识**：[PY-041 文件读写](../../01_Python%E5%9F%BA%E7%A1%80/05%20%E5%BC%82%E5%B8%B8%E4%B8%8E%E6%96%87%E4%BB%B6/PY-041%20%E6%96%87%E4%BB%B6%E8%AF%BB%E5%86%99.md) · [PY-042 路径与 pathlib](../../01_Python%E5%9F%BA%E7%A1%80/05%20%E5%BC%82%E5%B8%B8%E4%B8%8E%E6%96%87%E4%BB%B6/PY-042%20%E8%B7%AF%E5%BE%84%E4%B8%8E%20pathlib.md)

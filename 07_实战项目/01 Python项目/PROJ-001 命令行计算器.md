---
id: PROJ-001
title: 命令行计算器
domain: PROJ
module: []
type: project
difficulty: L1 概念
tech_stack: [Python]
status: planned
prerequisites: [PY-001, PY-003, PY-005]
related_math: []
related: []
notion_page_id: 40b663e8-0333-81d8-9de8-c925d04fd0da
notion_url: https://app.notion.com/p/40b663e8033381d89de8c925d04fd0da
created: 2026-11-01
updated: 2026-11-01
notion_db: 实战项目
所属学科: Python基础
项目状态: 未开始
项目阶段: 立项
难度: L1 概念
项目目标: 用纯 Python 实现四则运算计算器，正确处理非法运算符与除零。
技术栈: [Python]
对应课程模块: [M-PY-01]
前置知识: [PY-001, PY-003, PY-005]
tags: [Python基础]
---

# PROJ-001 命令行计算器

## 1. 项目目标

用纯 Python 实现一个四则运算计算器，**能正确处理非法运算符与除零**，并把错误变成清晰提示而不是崩栈。

- 完成后应能证明：你会用字典做分派、用异常做错误处理。
- 工作量：约 1–2 小时。
- 难度：L1（入门）。

## 2. 前置知识

| 知识点 | 为什么需要 |
|---|---|
| [[PY-001 变量与赋值]] | 存输入与结果 |
| [[PY-003 运算符]] | 四则运算与优先级 |
| [[PY-005 条件判断]] | 分支处理 |
| [[PY-044 异常处理]] | 捕获除零与非法输入 |

## 3. 需求说明

**功能需求**

| 编号 | 需求 | 验收 |
|---|---|---|
| F1 | 支持 `+ - * /` | 四个用例结果正确 |
| F2 | 除以 0 给出提示 | 不崩溃，返回明确信息 |
| F3 | 不支持的运算符给出提示 | 同上 |
| F4 | 结果格式友好 | `8 / 2 = 4.0` 这类可读输出 |

**非功能需求**

- 不引入第三方库。
- 函数可被单独导入测试。

## 4. 项目目录结构

```text
calc/
├── calculator.py     # 核心逻辑
├── demo.py           # 演示脚本（本笔记的代码）
└── README.md         # 运行说明
```

## 5. 实现步骤

1. 定义运算符 → 函数的映射表（避免一长串 `if`）。
2. 写 `calc(a, op, b)`：先查表、再判除零、最后计算。
3. 写演示脚本，跑 4 个正常用例 + 2 个错误用例。
4. 加类型注解，跑 `ruff check`。

## 6. 完整参考代码

```powershell
python demo.py
```

```python
import operator


OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
}


def calc(a, op, b):
    if op not in OPS:
        raise ValueError(f"不支持的运算符: {op}")
    if op == "/" and b == 0:
        raise ZeroDivisionError("除数不能为 0")
    return OPS[op](a, b)


cases = [(2, "+", 3), (10, "-", 4), (6, "*", 7), (8, "/", 2)]
for a, op, b in cases:
    print(f"{a} {op} {b} = {calc(a, op, b)}")

try:
    calc(1, "/", 0)
except ZeroDivisionError as e:
    print("错误:", e)

try:
    calc(1, "%", 2)
except ValueError as e:
    print("错误:", e)
```

## 7. 运行说明

```powershell
cd calc
python demo.py
```

**预期输出**

```
2 + 3 = 5
10 - 4 = 6
6 * 7 = 42
8 / 2 = 4.0
错误: 除数不能为 0
错误: 不支持的运算符: %
```

## 8. 扩展挑战

1. **加括号支持**：实现 `1 + 2 * 3` 的优先级（提示：调度场算法或递归下降）。
2. **支持连续输入**：从命令行循环读取，输入 `q` 退出（注意不要写成死循环）。
3. **加记忆功能**：把最近 10 次计算写入 JSON，启动时载入。
4. **单元测试**：给 `calc` 写 6 条 `unittest` 用例（正常 4 条、异常 2 条）。

## 9. 复盘要点（做完再填）

- 用映射表代替 `if/elif` 后，加新运算符要改几行？
- 异常应该抛给调用方，还是在这里就打印？为什么？
- 如果用户输入 `"abc"` 当数字，应该在哪一层拦截？

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
| 难度 | L1 概念 |
| 项目目标 | 用纯 Python 实现四则运算计算器，正确处理非法运算符与除零。 |
| 技术栈 | Python |

- **对应课程模块**：[M-PY-01 Python 编程基础](../../01_Python%E5%9F%BA%E7%A1%80/%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-PY-01%20Python%20%E7%BC%96%E7%A8%8B%E5%9F%BA%E7%A1%80.md)
- **前置知识**：[PY-001 变量与赋值](../../01_Python%E5%9F%BA%E7%A1%80/01%20%E8%AF%AD%E6%B3%95%E4%B8%8E%E7%B1%BB%E5%9E%8B/PY-001%20%E5%8F%98%E9%87%8F%E4%B8%8E%E8%B5%8B%E5%80%BC.md) · [PY-003 运算符](../../01_Python%E5%9F%BA%E7%A1%80/01%20%E8%AF%AD%E6%B3%95%E4%B8%8E%E7%B1%BB%E5%9E%8B/PY-003%20%E8%BF%90%E7%AE%97%E7%AC%A6.md) · [PY-005 条件判断](../../01_Python%E5%9F%BA%E7%A1%80/01%20%E8%AF%AD%E6%B3%95%E4%B8%8E%E7%B1%BB%E5%9E%8B/PY-005%20%E6%9D%A1%E4%BB%B6%E5%88%A4%E6%96%AD.md)

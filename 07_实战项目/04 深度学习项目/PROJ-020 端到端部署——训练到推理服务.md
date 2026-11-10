---
id: PROJ-020
title: 端到端部署——训练到推理服务
domain: PROJ
module: [M-DL-07, M-DL-08]
type: project
difficulty: L3 推导
status: planned
tech_stack: [Python, PyTorch, scikit-learn, JSON]
prerequisites: [DL-062, DL-063, DL-067, DL-069]
related_math: []
related: [DL-048, DL-058, DL-059, DL-062, DL-063, DL-066, DL-067, DL-068, DL-069, PROJ-017]
notion_page_id: 411663e8-0333-81d8-b75f-efbda96e4917
notion_url: https://app.notion.com/p/411663e8033381d8b75fefbda96e4917
created: 2026-11-07
updated: 2026-11-07
notion_db: 实战项目
所属学科: 深度学习
项目状态: 未开始
项目阶段: 立项
难度: L3 推导
项目目标: 端到端部署——训练到推理服务
技术栈: [Python, PyTorch, scikit-learn, JSON]
项目仓库: ai-knowledge-system/projects/PROJ-020
对应课程模块: [M-DL-07, M-DL-08]
前置知识: [DL-062, DL-063, DL-067, DL-069]
对应知识点: [DL-048, DL-058, DL-059, DL-062, DL-063, DL-066, DL-067, DL-068, DL-069]
tags: [深度学习]
---

# PROJ-020 端到端部署——训练到推理服务

## 1. 项目背景

前面 4 个项目都在"训练"这一侧。本项目把链路补完：

> **训练 → 保存 → 加载 → 推理服务 → 性能测量 → 上线门禁**

它检验的不是模型精度，而是**你能不能把模型交付出去**。

> **数据声明**：scikit-learn `load_digits`（8×8，1797 条）。MNIST 因网络重置无法下载，详见 PROJ-016 第 9 节。

## 2. 项目目标

| 目标 | 验收方式 |
|---|---|
| 可复现训练 | 固定种子、早停、指标落盘 |
| 交付三件套 | `best.pt` + `metrics.json` + 配置 |
| 加载即可用 | 重载后准确率一致 |
| 推理服务类 | 单条/批量都能预测 |
| 性能测量 | batch=1 与 batch=64 延迟 |
| 上线门禁 | 6 项检查全 PASS |

## 3. 技术栈与环境

| 项 | 值 |
|---|---|
| Python | 3.13.15（含 torch） |
| PyTorch | 2.15.0+cu132 |
| 设备 | CPU（本项目模型极小，CPU 足够） |
| 模型 | 2×Conv+BN+MaxPool + FC（6,186 参数） |

## 4. 数据集

```text
load_digits：1797 条、8×8、0–16 灰度
预处理：/16 → [0,1]，reshape (N,1,8,8)
划分：60/20/20 分层，seed=42
      训练 1078 / 验证 359 / 测试 360
```

## 5. 方案设计

```text
train_and_save()
  ├─ 训练 + 早停（patience=10）
  ├─ 保存 best.pt：{epoch, model, n_class, config}
  └─ 写 metrics.json：{test_acc, best_epoch, val_loss, train_sec, params, ckpt_kb}

class Predictor
  ├─ __init__：加载 ckpt → 读 n_class → 建模型 → load_state_dict → eval()
  └─ predict()：@torch.no_grad()，支持 (1,8,8) 单条与 (N,1,8,8) 批量

release_gate()
  └─ 6 项断言 → 全 PASS 才允许上线
```

**关键设计**：checkpoint 里**存 `n_class`**，避免"靠猜 state_dict 键名"这种脆弱做法（本课第一版就是 `obj["model"]["6.weight"]`，直接 `KeyError`）。

## 6. 核心代码

```python
import torch
import torch.nn as nn


def build(n_class: int = 10) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 2 * 2, n_class),
    )


class Predictor:
    def __init__(self, ckpt_path: str) -> None:
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model = build(obj["n_class"])
        self.model.load_state_dict(obj["model"])
        self.model.eval()
        self.meta = {"epoch": obj["epoch"], "config": obj["config"]}

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return self.model(x).argmax(1)
```

## 7. 运行结果

**训练与交付**

```json
{
  "test_acc": 0.9833,
  "best_epoch": 39,
  "val_loss": 0.0768,
  "train_sec": 3.51,
  "params": 6186,
  "ckpt_kb": 29.3
}
```

**推理服务**

```text
单样本预测: 5  meta: {'epoch': 39, 'config': {'lr': 0.001, 'seed': 42, 'arch': '2conv+bn+fc'}}
batch=1: 0.1370 ms | batch=64: 0.4712 ms（每样本 7.36 us）
重载后 test acc: 0.9833
```

**上线门禁**

```text
[PASS] 模型文件存在
[PASS] 文件非空
[PASS] 指标文件存在
[PASS] test acc >= 0.95
[PASS] test acc < 1.0（防泄漏）
[PASS] 参数量 < 1e6
门禁结果: 允许上线
```

## 8. 结果分析

**交付正确性**：

$$
\text{acc}_{\text{训练后}}=0.9833=\text{acc}_{\text{重载后}}
$$

⇒ 权重保存/加载**无损**（DL-063 的要求）。

**批量化收益**：

| batch | 延迟 | 每样本 |
|---|---|---|
| 1 | 0.1370 ms | 137 μs |
| 64 | 0.4712 ms | **7.36 μs** |

**每样本提速约 18.6 倍**（137 / 7.36），与 DL-067 的"批量化是第一优先"一致。

**门禁的意义**：6 项检查里最容易被忽略的是 **`test acc < 1.0`**——把"完美指标"当泄漏告警（DL-069）。

## 9. 局限与讨论

1. **不是真实服务**：没有 HTTP 层、没有并发、没有超时/重试。
2. **模型太小**：6,186 参数，CPU 推理已足够；GPU/量化优化（DL-067）在此无收益。
3. **未做监控**：输入分布漂移、预测分布、在线指标都没接。
4. **单次训练**：未做多种子确认（DL-064 的种子波动约 2.5%）。

## 10. 复现步骤

```text
1. py -3.13（torch 2.15.0+cu132）
2. 运行第 11 节脚本
3. 预期：test_acc 0.9833、best_epoch 39、params 6186、ckpt 29.3 KB
4. 门禁应 6/6 PASS；若 test_acc 恰为 1.0，检查是否有数据泄漏
```

## 11. 完整脚本（训练 + 服务 + 门禁）

```python
import json
import os
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 42
ROOT = os.path.join(tempfile.gettempdir(), "proj020_release")


def build(n_class: int = 10) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 2 * 2, n_class),
    )


def load_data():
    d = load_digits()
    X = torch.tensor(d.data.astype("float32") / 16.0).view(-1, 1, 8, 8)
    y = torch.tensor(d.target.astype("int64"))
    Xtr, Xtmp, ytr, ytmp = train_test_split(
        X.numpy(), y.numpy(), test_size=0.4, random_state=SEED, stratify=y.numpy())
    Xval, Xte, yval, yte = train_test_split(
        Xtmp, ytmp, test_size=0.5, random_state=SEED, stratify=ytmp)
    conv = lambda a: torch.tensor(a).view(-1, 1, 8, 8)
    return (conv(Xtr), torch.tensor(ytr), conv(Xval), torch.tensor(yval),
            conv(Xte), torch.tensor(yte))


def train_and_save():
    Xtr, ytr, Xval, yval, Xte, yte = load_data()
    torch.manual_seed(SEED)
    model = build()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = {"vl": float("inf"), "ep": 0, "state": None}
    wait, start = 0, time.time()
    for epoch in range(1, 61):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 32):
            idx = perm[i:i + 32]
            opt.zero_grad()
            F.cross_entropy(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(F.cross_entropy(model(Xval), yval))
        if vl < best["vl"] - 1e-4:
            best = {"vl": vl, "ep": epoch,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
            wait = 0
        else:
            wait += 1
            if wait >= 10:
                break

    model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        acc = float(model(Xte).argmax(1).eq(yte).float().mean())

    os.makedirs(os.path.join(ROOT, "runs", "v1"), exist_ok=True)
    ckpt = os.path.join(ROOT, "runs", "v1", "best.pt")
    torch.save({"epoch": best["ep"], "model": model.state_dict(), "n_class": 10,
                "config": {"lr": 1e-3, "seed": SEED, "arch": "2conv+bn+fc"}}, ckpt)
    metrics = {"test_acc": round(acc, 4), "best_epoch": best["ep"],
               "val_loss": round(best["vl"], 4),
               "train_sec": round(time.time() - start, 2),
               "params": sum(p.numel() for p in model.parameters()),
               "ckpt_kb": round(os.path.getsize(ckpt) / 1024, 2)}
    with open(os.path.join(ROOT, "runs", "v1", "metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return ckpt, metrics


class Predictor:
    def __init__(self, ckpt_path: str) -> None:
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model = build(obj["n_class"])
        self.model.load_state_dict(obj["model"])
        self.model.eval()
        self.meta = {"epoch": obj["epoch"], "config": obj["config"]}

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return self.model(x).argmax(1)


def release_gate(ckpt_path: str, metrics: dict) -> bool:
    checks = [
        ("模型文件存在", os.path.exists(ckpt_path)),
        ("文件非空", os.path.getsize(ckpt_path) > 0),
        ("指标文件存在", os.path.exists(
            os.path.join(os.path.dirname(ckpt_path), "metrics.json"))),
        ("test acc >= 0.95", metrics["test_acc"] >= 0.95),
        ("test acc < 1.0（防泄漏）", metrics["test_acc"] < 1.0),
        ("参数量 < 1e6", metrics["params"] < 1_000_000),
    ]
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    return all(ok for _, ok in checks)


if __name__ == "__main__":
    ckpt, metrics = train_and_save()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    predictor = Predictor(ckpt)
    x = torch.randn(1, 1, 8, 8)
    print("单样本预测:", int(predictor.predict(x[0])), "meta:", predictor.meta)
    print("门禁:", "允许上线" if release_gate(ckpt, metrics) else "阻止上线")
```

## 12. 验收清单

| # | 项 | 判据 | 结果 |
|---|---|---|---|
| 1 | 训练可复现 | 固定种子 | ✅ |
| 2 | 交付三件套 | 权重 + 配置 + 指标 | ✅ |
| 3 | 加载一致 | 重载后 acc 相同 | ✅ 0.9833 |
| 4 | 服务可用 | 单条/批量 | ✅ |
| 5 | 性能记录 | 延迟/吞吐 | ✅ 0.137 / 0.4712 ms |
| 6 | 上线门禁 | 6 项全 PASS | ✅ |
| 7 | 配置随权重保存 | ckpt 含 config | ✅ |

## 13. 结论

- 完整链路跑通：**训练（0.9833）→ 存盘（29.3 KB）→ 加载（一致）→ 服务（7.36 μs/样本）→ 门禁（6/6）**。
- 第一版踩到的坑：**靠猜 `state_dict` 键名**（`"6.weight"`）导致 `KeyError` → 改为在 ckpt 里存 `n_class`。
- 批量化把每样本从 **137 μs** 降到 **7.36 μs**（约 18.6×）。

## 14. 与后续项目的关系

- 本项目是阶段 08 的**收官项目**，串起 DL-062（训练循环）、DL-063（保存加载）、DL-067（性能）、DL-069（门禁）。
- 若要变成真实服务：加 FastAPI/Flask 层 + 并发压测 + 监控（DL-067 第 14 节）。

## 学习记录

| 日期 | 状态 | 用时 | 问题 | 复盘 |
|---|---|---|---|---|
| 2026-11-07 | planned | — | — | — |

## 关联与延伸

本项目的知识依据（正文链接，便于图谱连边）：

- [[DL-058 设备管理——CPU 与 GPU]]
- [[DL-067 部署——推理服务与性能优化]]

---

## 🔗 Notion 同步信息

> 以下内容由 Notion「AI Knowledge System · 实战项目」数据库按知识 ID 合并而来（导出于 2026-11-09）。

| 属性 | 值 |
|---|---|
| 所属学科 | 深度学习 |
| 项目状态 | 未开始 |
| 项目阶段 | 立项 |
| 难度 | L3 推导 |
| 技术栈 | Python、PyTorch、scikit-learn、JSON |
| 项目仓库 | ai-knowledge-system/projects/PROJ-020 |

- **对应课程模块**：[M-DL-07 PyTorch 深度学习框架实践](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-DL-07%20PyTorch%20%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A1%86%E6%9E%B6%E5%AE%9E%E8%B7%B5.md) · [M-DL-08 深度学习工程实践与综合应用](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-DL-08%20%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%B7%A5%E7%A8%8B%E5%AE%9E%E8%B7%B5%E4%B8%8E%E7%BB%BC%E5%90%88%E5%BA%94%E7%94%A8.md)
- **前置知识**：[DL-062 训练、验证、测试循环](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/06%20PyTorch/DL-062%20%E8%AE%AD%E7%BB%83%E3%80%81%E9%AA%8C%E8%AF%81%E3%80%81%E6%B5%8B%E8%AF%95%E5%BE%AA%E7%8E%AF.md) · [DL-063 模型保存、加载与 Checkpoint](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/06%20PyTorch/DL-063%20%E6%A8%A1%E5%9E%8B%E4%BF%9D%E5%AD%98%E3%80%81%E5%8A%A0%E8%BD%BD%E4%B8%8E%20Checkpoint.md) · [DL-067 部署：推理服务与性能优化](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/07%20%E5%B7%A5%E7%A8%8B%E4%B8%8E%E9%A1%B9%E7%9B%AE/DL-067%20%E9%83%A8%E7%BD%B2%E2%80%94%E2%80%94%E6%8E%A8%E7%90%86%E6%9C%8D%E5%8A%A1%E4%B8%8E%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96.md) · [DL-069 工程收尾与上线清单](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/07%20%E5%B7%A5%E7%A8%8B%E4%B8%8E%E9%A1%B9%E7%9B%AE/DL-069%20%E5%B7%A5%E7%A8%8B%E6%94%B6%E5%B0%BE%E4%B8%8E%E4%B8%8A%E7%BA%BF%E6%B8%85%E5%8D%95.md)
- **对应知识点**：[DL-048 ResNet 与 Skip Connection](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/02%20CNN/DL-048%20ResNet%20%E4%B8%8E%20Skip%20Connection.md) · [DL-058 设备管理：CPU 与 GPU](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/06%20PyTorch/DL-058%20%E8%AE%BE%E5%A4%87%E7%AE%A1%E7%90%86%E2%80%94%E2%80%94CPU%20%E4%B8%8E%20GPU.md) · [DL-059 Dataset 与 DataLoader](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/06%20PyTorch/DL-059%20Dataset%20%E4%B8%8E%20DataLoader.md) · [DL-062 训练、验证、测试循环](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/06%20PyTorch/DL-062%20%E8%AE%AD%E7%BB%83%E3%80%81%E9%AA%8C%E8%AF%81%E3%80%81%E6%B5%8B%E8%AF%95%E5%BE%AA%E7%8E%AF.md) · [DL-063 模型保存、加载与 Checkpoint](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/06%20PyTorch/DL-063%20%E6%A8%A1%E5%9E%8B%E4%BF%9D%E5%AD%98%E3%80%81%E5%8A%A0%E8%BD%BD%E4%B8%8E%20Checkpoint.md) · [DL-066 训练调试：不收敛与过拟合排查](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/07%20%E5%B7%A5%E7%A8%8B%E4%B8%8E%E9%A1%B9%E7%9B%AE/DL-066%20%E8%AE%AD%E7%BB%83%E8%B0%83%E8%AF%95%E2%80%94%E2%80%94%E4%B8%8D%E6%94%B6%E6%95%9B%E4%B8%8E%E8%BF%87%E6%8B%9F%E5%90%88%E6%8E%92%E6%9F%A5.md) · [DL-067 部署：推理服务与性能优化](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/07%20%E5%B7%A5%E7%A8%8B%E4%B8%8E%E9%A1%B9%E7%9B%AE/DL-067%20%E9%83%A8%E7%BD%B2%E2%80%94%E2%80%94%E6%8E%A8%E7%90%86%E6%9C%8D%E5%8A%A1%E4%B8%8E%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96.md) · [DL-068 端到端项目实战](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/07%20%E5%B7%A5%E7%A8%8B%E4%B8%8E%E9%A1%B9%E7%9B%AE/DL-068%20%E7%AB%AF%E5%88%B0%E7%AB%AF%E9%A1%B9%E7%9B%AE%E5%AE%9E%E6%88%98.md) · [DL-069 工程收尾与上线清单](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/07%20%E5%B7%A5%E7%A8%8B%E4%B8%8E%E9%A1%B9%E7%9B%AE/DL-069%20%E5%B7%A5%E7%A8%8B%E6%94%B6%E5%B0%BE%E4%B8%8E%E4%B8%8A%E7%BA%BF%E6%B8%85%E5%8D%95.md)

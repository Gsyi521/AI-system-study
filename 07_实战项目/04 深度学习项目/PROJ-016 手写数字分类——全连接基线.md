---
id: PROJ-016
title: 手写数字分类——全连接基线
domain: PROJ
module: [M-DL-05, M-DL-07]
type: project
difficulty: L2 应用
status: planned
tech_stack: [Python, PyTorch, scikit-learn, NumPy]
prerequisites: [DL-008, DL-017, DL-060, DL-062]
related_math: [MATH-805]
related: [DL-008, DL-011, DL-017, DL-041, DL-056, DL-059, DL-060, DL-062, DL-063, PROJ-017]
notion_page_id: 411663e8-0333-8103-910c-e658a3ff4423
notion_url: https://app.notion.com/p/411663e803338103910ce658a3ff4423
created: 2026-11-07
updated: 2026-11-07
notion_db: 实战项目
所属学科: 深度学习
项目状态: 未开始
项目阶段: 立项
难度: L2 应用
项目目标: 手写数字分类——全连接基线
技术栈: [Python, PyTorch, scikit-learn, NumPy]
项目仓库: ai-knowledge-system/projects/PROJ-016
对应课程模块: [M-DL-05, M-DL-07]
前置知识: [DL-008, DL-017, DL-060, DL-062]
对应知识点: [DL-008, DL-011, DL-017, DL-041, DL-056, DL-059, DL-060, DL-062, DL-063]
tags: [深度学习]
---

# PROJ-016 手写数字分类——全连接基线

## 1. 项目背景

这是深度学习项目的**第一块基石**：用最朴素的**全连接网络**做 10 类手写数字分类。

它不追求最好成绩，而是给你一个**可复现、可解释、有数字**的基线——后面所有改进（CNN、残差、增强）都要和它比。

> **数据声明**：使用 **scikit-learn `load_digits`**（8×8 灰度、1797 条、10 类）。
> 原始计划是 MNIST（28×28、7 万条），但本机下载在 **55% 处被网络重置**（`ConnectionResetError WinError 10054`），因此改用**离线自带的同类数据**。数据来源与差异已在第 9 节说明。

## 2. 项目目标

| 目标 | 验收方式 |
|---|---|
| 跑通 10 类分类全流程 | 训练/验证/测试都有输出 |
| 有可靠基线指标 | test acc ≥ 0.95（本课实际 0.9722） |
| 看清每类表现 | 输出每类 recall |
| 可复现 | 同种子两次结果一致 |
| 能交付 | 存 checkpoint + 指标 JSON |

## 3. 技术栈与环境

| 项 | 值 |
|---|---|
| Python | 3.13.15（含 torch） |
| PyTorch | 2.15.0+cu132 |
| 设备 | RTX 4070 Ti（本项目矩阵极小，实际 CPU 更快） |
| 数据 | sklearn digits，8×8×1 |

## 4. 数据集

```text
样本数：1797
特征：64（8×8 灰度，取值 0–16）
类别：10（0–9，近似均衡）
预处理：除以 16 归一化到 [0,1]
划分：60% train / 20% val / 20% test（分层抽样，seed=42）
       实际 1078 / 359 / 360
```

## 5. 方案设计

```text
Flatten(64) → Linear(64,64) → ReLU → Linear(64,10)
损失：CrossEntropyLoss
优化器：Adam lr=1e-3
batch=32，最多 80 epoch，早停 patience=12
```

**为什么先做全连接**：它把每个像素当作独立特征，**忽略空间结构**——正好用来量化"空间结构的价值"（PROJ-017 会对比）。

## 6. 核心代码

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 42


def load_data():
    d = load_digits()
    X = d.data.astype(np.float32) / 16.0
    y = d.target.astype(np.int64)
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4,
                                            random_state=SEED, stratify=y)
    Xval, Xte, yval, yte = train_test_split(Xtmp, ytmp, test_size=0.5,
                                            random_state=SEED, stratify=ytmp)
    as_img = lambda a: torch.tensor(a).view(-1, 1, 8, 8)
    return (as_img(Xtr), torch.tensor(ytr), as_img(Xval), torch.tensor(yval),
            as_img(Xte), torch.tensor(yte))


def build_fc() -> nn.Sequential:
    return nn.Sequential(
        nn.Flatten(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10)
    )
```

完整训练循环（早停 + 每类 recall + checkpoint）见第 11 节。

## 7. 运行结果

```text
FC: params=4810 best_ep=76 val_loss=0.1039 train=0.9991 val=0.9694 test=0.9722 sec=1.39

每类 recall:
0: 0.9714  1: 0.9459  2: 1.0     3: 1.0     4: 0.9722
5: 1.0     6: 0.9722  7: 1.0     8: 0.9429  9: 0.9167
```

**读法**：

1. **train 0.9991 但 test 0.9722** ⇒ 轻微过拟合（4,810 参数 vs 1,078 训练样本）。
2. **最差三类：9（0.9167）、8（0.9429）、1（0.9459）**——手写数字里 8/9 与 9/4 本就易混。
3. 训练只 **1.39 s**，说明这个规模下 CPU 足够。

## 8. 结果分析

**过拟合的量化**：

$$
\text{gap}=0.9991-0.9722=0.0269
$$

**诊断**：全连接把 64 个像素当独立特征，参数 4,810 对 1,078 样本偏多 ⇒ 记住训练集细节。

**可改进方向**（按性价比排序）：

| 手段 | 预期 | 备注 |
|---|---|---|
| **换 CNN** | 显著（PROJ-017 实测 +1.39 点） | 利用 2D 空间结构 |
| Weight Decay | 小 | 压 gap |
| Dropout | 小～中 | 本数据规模小 |
| 数据增强 | 中（平移/轻微旋转） | 8×8 空间有限 |

## 9. 数据来源与差异说明（必读）

| 项 | MNIST（计划） | digits（实际） |
|---|---|---|
| 尺寸 | 28×28 | **8×8** |
| 样本 | 70,000 | **1,797** |
| 灰度 | 0–255 | 0–16 |
| 下载 | 需联网 | **本地自带** |

**为什么换**：本机执行 `MNIST(download=True)` 时在 **9.91 MB 的 55%** 处被重置：
```text
MNIST FAIL: ConnectionResetError [WinError 10054] 远程主机强迫关闭了一个现有的连接。
```

**影响**：8×8 的图更小、类间差异更弱，**绝对指标不能与 MNIST 论文数字直接比较**；但"同数据上 FC vs CNN"的对比仍然有效（PROJ-017）。

**如何切回 MNIST**：把 `load_data()` 换成 `torchvision.datasets.MNIST` + `DataLoader`（代码结构不变），或先手动下载 `MNIST/raw/*.gz` 放到缓存目录。

## 10. 复现步骤

```text
1. 准备环境：py -3.13（torch 2.15.0+cu132）
2. 运行训练脚本（第 11 节）
3. 预期输出：test acc 0.9722，params 4810
4. 同种子重跑：结果应完全一致（DL-064 已验）
```

## 11. 完整训练脚本

```python
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 42


def load_data():
    d = load_digits()
    X = d.data.astype(np.float32) / 16.0
    y = d.target.astype(np.int64)
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4,
                                            random_state=SEED, stratify=y)
    Xval, Xte, yval, yte = train_test_split(Xtmp, ytmp, test_size=0.5,
                                            random_state=SEED, stratify=ytmp)
    as_img = lambda a: torch.tensor(a).view(-1, 1, 8, 8)
    return (as_img(Xtr), torch.tensor(ytr), as_img(Xval), torch.tensor(yval),
            as_img(Xte), torch.tensor(yte))


Xtr, ytr, Xval, yval, Xte, yte = load_data()

torch.manual_seed(SEED)
model = nn.Sequential(nn.Flatten(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

best = {"vl": float("inf"), "ep": 0, "state": None}
wait, patience = 0, 12
start = time.time()

for epoch in range(1, 81):
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
        if wait >= patience:
            break

model.load_state_dict(best["state"])
model.eval()
with torch.no_grad():
    pred = model(Xte).argmax(1)
    acc = float(pred.eq(yte).float().mean())
    cm = torch.zeros(10, 10, dtype=torch.int64)
    for t, p in zip(yte.tolist(), pred.tolist()):
        cm[t, p] += 1

print(f"best_ep={best['ep']} val_loss={best['vl']:.4f} "
      f"test_acc={acc:.4f} sec={time.time() - start:.2f}")
print("params:", sum(p.numel() for p in model.parameters()))
for c in range(10):
    sup = int(cm[c].sum())
    print(f"  {c}: recall={int(cm[c, c]) / sup:.4f} (n={sup})")
```

## 12. 验收清单

| # | 项 | 判据 | 结果 |
|---|---|---|---|
| 1 | 数据划分 | 训练/验证/测试三段、分层 | ✅ 1078/359/360 |
| 2 | 基线指标 | test acc ≥ 0.95 | ✅ **0.9722** |
| 3 | 每类表现 | 10 类 recall 都输出 | ✅ 最低 9: 0.9167 |
| 4 | 早停 | 用 val_loss、可复现最佳轮 | ✅ ep 76 |
| 5 | 可复现 | 同种子一致 | ✅ |
| 6 | 可交付 | checkpoint + 指标 | ✅ |
| 7 | 算力记录 | 参数量与耗时 | ✅ 4,810 / 1.39 s |

## 13. 结论与局限

**结论**：

- 全连接基线在 digits 上达到 **test acc 0.9722**（4,810 参数、1.39 s）。
- 存在轻微过拟合（train 0.9991 vs test 0.9722，gap **0.0269**）。
- 最弱类为 **9（0.9167）**，是后续改进的重点观察对象。

**局限**：

1. 数据是 **8×8**，比 MNIST 小得多，**绝对指标不可外推**；
2. 全连接忽略空间结构 ⇒ 自然会输给 CNN（PROJ-017 验证）；
3. 只做了单次划分，未做多种子/交叉验证；
4. 未做数据增强与正则化对比。

## 14. 与后续项目的关系

- 本项目 = **基线**（PROJ-016）
- 下一项目 = **CNN 改进**（PROJ-017）：同数据、同划分、同评估，只换模型结构
- 对比口径：**test acc + 参数量 + 训练耗时 + 每类 recall**

## 学习记录

| 日期 | 状态 | 用时 | 问题 | 复盘 |
|---|---|---|---|---|
| 2026-11-07 | planned | — | — | — |

## 关联与延伸

本项目的知识依据（正文链接，便于图谱连边）：

- [[DL-011 神经网络前向传播]]
- [[DL-060 nn.Module 自定义网络]]

---

## 🔗 Notion 同步信息

> 以下内容由 Notion「AI Knowledge System · 实战项目」数据库按知识 ID 合并而来（导出于 2026-11-09）。

| 属性 | 值 |
|---|---|
| 所属学科 | 深度学习 |
| 项目状态 | 未开始 |
| 项目阶段 | 立项 |
| 难度 | L2 应用 |
| 技术栈 | Python、PyTorch、scikit-learn、NumPy |
| 项目仓库 | ai-knowledge-system/projects/PROJ-016 |

- **对应课程模块**：[M-DL-05 卷积神经网络 CNN](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-DL-05%20%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%20CNN.md) · [M-DL-07 PyTorch 深度学习框架实践](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-DL-07%20PyTorch%20%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A1%86%E6%9E%B6%E5%AE%9E%E8%B7%B5.md)
- **前置知识**：[DL-008 多层感知机 MLP](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/01%20%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E7%A1%80/DL-008%20%E5%A4%9A%E5%B1%82%E6%84%9F%E7%9F%A5%E6%9C%BA%20MLP.md) · [DL-017 多分类交叉熵与 Softmax](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/01%20%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E7%A1%80/DL-017%20%E5%A4%9A%E5%88%86%E7%B1%BB%E4%BA%A4%E5%8F%89%E7%86%B5%E4%B8%8E%20Softmax.md) · [DL-060 nn.Module 自定义网络](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/06%20PyTorch/DL-060%20nn.Module%20%E8%87%AA%E5%AE%9A%E4%B9%89%E7%BD%91%E7%BB%9C.md) · [DL-062 训练、验证、测试循环](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/06%20PyTorch/DL-062%20%E8%AE%AD%E7%BB%83%E3%80%81%E9%AA%8C%E8%AF%81%E3%80%81%E6%B5%8B%E8%AF%95%E5%BE%AA%E7%8E%AF.md)
- **对应知识点**：[DL-008 多层感知机 MLP](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/01%20%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E7%A1%80/DL-008%20%E5%A4%9A%E5%B1%82%E6%84%9F%E7%9F%A5%E6%9C%BA%20MLP.md) · [DL-011 神经网络前向传播](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/01%20%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E7%A1%80/DL-011%20%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%89%8D%E5%90%91%E4%BC%A0%E6%92%AD.md) · [DL-017 多分类交叉熵与 Softmax](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/01%20%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E7%A1%80/DL-017%20%E5%A4%9A%E5%88%86%E7%B1%BB%E4%BA%A4%E5%8F%89%E7%86%B5%E4%B8%8E%20Softmax.md) · [DL-041 图像张量、通道与批次维度](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/02%20CNN/DL-041%20%E5%9B%BE%E5%83%8F%E5%BC%A0%E9%87%8F%E3%80%81%E9%80%9A%E9%81%93%E4%B8%8E%E6%89%B9%E6%AC%A1%E7%BB%B4%E5%BA%A6.md) · [DL-056 PyTorch 与 Tensor 基础](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/06%20PyTorch/DL-056%20PyTorch%20%E4%B8%8E%20Tensor%20%E5%9F%BA%E7%A1%80.md) · [DL-059 Dataset 与 DataLoader](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/06%20PyTorch/DL-059%20Dataset%20%E4%B8%8E%20DataLoader.md) · [DL-060 nn.Module 自定义网络](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/06%20PyTorch/DL-060%20nn.Module%20%E8%87%AA%E5%AE%9A%E4%B9%89%E7%BD%91%E7%BB%9C.md) · [DL-062 训练、验证、测试循环](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/06%20PyTorch/DL-062%20%E8%AE%AD%E7%BB%83%E3%80%81%E9%AA%8C%E8%AF%81%E3%80%81%E6%B5%8B%E8%AF%95%E5%BE%AA%E7%8E%AF.md) · [DL-063 模型保存、加载与 Checkpoint](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/06%20PyTorch/DL-063%20%E6%A8%A1%E5%9E%8B%E4%BF%9D%E5%AD%98%E3%80%81%E5%8A%A0%E8%BD%BD%E4%B8%8E%20Checkpoint.md)

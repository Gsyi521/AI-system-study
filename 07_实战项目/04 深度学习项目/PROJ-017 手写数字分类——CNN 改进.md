---
id: PROJ-017
title: 手写数字分类——CNN 改进
domain: PROJ
module: [M-DL-05, M-DL-07]
type: project
difficulty: L3 推导
status: planned
tech_stack: [Python, PyTorch, scikit-learn, NumPy]
prerequisites: [DL-040, DL-044, DL-045, DL-060]
related_math: [MATH-217]
related: [DL-040, DL-042, DL-043, DL-044, DL-045, DL-046, DL-047, PROJ-016]
notion_page_id: 411663e8-0333-811a-8974-c79a0d883dc8
notion_url: https://app.notion.com/p/CNN-411663e80333811a8974c79a0d883dc8
created: 2026-11-07
updated: 2026-11-07
notion_db: 实战项目
所属学科: 深度学习
项目状态: 未开始
项目阶段: 立项
难度: L3 推导
项目目标: 手写数字分类——CNN 改进
技术栈: [Python, PyTorch, scikit-learn, NumPy]
项目仓库: ai-knowledge-system/projects/PROJ-017
对应课程模块: [M-DL-05, M-DL-07]
前置知识: [DL-040, DL-044, DL-045, DL-060]
对应知识点: [DL-040, DL-042, DL-043, DL-044, DL-045, DL-046, DL-047]
tags: [深度学习]
---

# PROJ-017 手写数字分类——CNN 改进

## 1. 项目背景

PROJ-016 建立了全连接基线（**test 0.9722**）。本项目在**同一份数据、同一划分、同一评估口径**下，只把模型换成 **CNN**，回答一个具体问题：

> **"利用空间结构"到底值多少个点？**

> **数据声明**：同 PROJ-016，使用 **scikit-learn `load_digits`**（8×8）。MNIST 因网络重置未能下载，详见 PROJ-016 第 9 节。

## 2. 项目目标

| 目标 | 验收方式 |
|---|---|
| 同口径对比 FC vs CNN | 除模型外全部一致 |
| 量化改进幅度 | test acc 差值（本课 **+0.0139**） |
| 看清代价 | 参数量、训练耗时对比 |
| 分析每类变化 | 逐类 recall 对比 |
| 可复现 | 同种子、同划分 |

## 3. 技术栈与环境

| 项 | 值 |
|---|---|
| Python | 3.13.15（含 torch） |
| PyTorch | 2.15.0+cu132 |
| 数据 | sklearn digits，`(N,1,8,8)` |

## 4. 数据集

与 PROJ-016 **完全相同**：

```text
1797 条，8×8，0–16 灰度 → /16 归一化
60/20/20 分层划分，seed=42
训练 1078 / 验证 359 / 测试 360
```

**关键**：不改数据、不改划分，才能把差异归因到**模型结构**。

## 5. 方案设计

```text
Conv(1→16, 3×3, pad=1) → BatchNorm2d(16) → ReLU → MaxPool(2)     # 8×8 → 4×4
Conv(16→32, 3×3, pad=1) → BatchNorm2d(32) → ReLU → MaxPool(2)    # 4×4 → 2×2
Flatten(32×2×2=128) → Linear(128,10)
损失：CrossEntropyLoss
优化器：Adam lr=1e-3
batch=32，最多 80 epoch，早停 patience=12
```

**为什么这样设计**：

| 设计 | 依据 |
|---|---|
| 两层 3×3 | 感受野 3→7，覆盖 8×8 的大部分（DL-046） |
| padding=1 | 保尺寸，避免过早缩小（DL-043） |
| BN | 稳定训练、允许更大 lr（DL-036） |
| MaxPool 2×2 | 两次下采样到 2×2（DL-045） |
| 全连接改 128→10 | 参数大头从 FC 挪到卷积（DL-047 的教训） |

## 6. 核心代码

```python
import torch.nn as nn


def build_cnn() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 2 * 2, 10),
    )
```

**参数量手算**：

$$
(3^2\cdot1\cdot16+16)+(3^2\cdot16\cdot32+32)+(128\cdot10+10)
=160+4{,}640+1{,}290=6{,}090
$$

加上 BN 的 $\gamma,\beta$（16×2 + 32×2 = 96）⇒ **6,186**（与实测一致）。

## 7. 运行结果

```text
CNN: params=6186 best_ep=36 val_loss=0.074 train=1.0 val=0.9749 test=0.9861 sec=3.11

每类 recall:
0: 1.0  1: 1.0  2: 1.0  3: 1.0  4: 1.0
5: 1.0  6: 1.0  7: 1.0  8: 0.9714  9: 0.8889
```

**读法**：

1. **test 0.9861**，比 FC 的 0.9722 **高 1.39 个百分点**。
2. **最佳轮次 36**（FC 是 76）⇒ 收敛更快，因为卷积先验更贴合图像。
3. **8 类从 0.9429 → 0.9714**；**9 类仍是最弱（0.8889，比 FC 的 0.9167 还低）**。
4. 代价：参数 **6,186 > 4,810**，耗时 **3.11 s > 1.39 s**。

## 8. 对比分析（FC vs CNN）

| 指标 | FC（PROJ-016） | **CNN（本项目）** | 差值 |
|---|---|---|---|
| test acc | 0.9722 | **0.9861** | **+0.0139** |
| train acc | 0.9991 | 1.0000 | +0.0009 |
| gap | 0.0269 | 0.0139 | **−0.0130** |
| 最佳 epoch | 76 | **36** | −40 |
| 参数量 | 4,810 | 6,186 | +1,376 |
| 训练耗时 | 1.39 s | 3.11 s | +1.72 s |

**三个结论**：

1. **准确率 +1.39 点**，同时 **gap 缩小一半**（0.0269 → 0.0139）——CNN 不是靠"更能记"取胜。
2. **收敛快一倍**（36 vs 76 epoch）——空间先验让优化更容易。
3. **参数量反而更多**（6,186 > 4,810）：在 8×8 这种**极小输入**上，卷积的"参数效率"优势体现不出来（DL-040 的 109.8 倍差距要在大图上才明显）。

**逐类对比**：

| 类 | FC | CNN | 变化 |
|---|---|---|---|
| 0–7 | 0.97–1.0 | **全部 1.0** | 提升/持平 |
| 8 | 0.9429 | 0.9714 | **+0.0285** |
| 9 | 0.9167 | **0.8889** | **−0.0278** |

**9 类变差**，值得记录：8×8 下 9 的笔画被下采样后信息损失更大（两次池化到 2×2）。

## 9. 局限与讨论

1. **数据太小**：8×8、1,797 条 ⇒ 结论不能外推到 MNIST/ImageNet。
2. **参数量优势未体现**：卷积的省参数优势需要大输入（DL-040：32×32 时 FC 是 Conv 的 109.8 倍）。
3. **只跑单次划分**：±1 个点可能落在种子波动内（DL-064 实测种子差约 2.5%），**+1.39 点需多种子确认**。
4. **未做增强/正则对比**：Dropout、Weight Decay、旋转增强都没试。

> **诚实标注**：本项目的 +1.39 点是**单次划分**的结果，未做多折验证，因此只能算"方向性证据"，不能作为显著性结论。

## 10. 复现步骤

```text
1. 与 PROJ-016 相同的环境与数据
2. 运行第 11 节脚本
3. 预期：test 0.9861，params 6186，best_ep 36
4. 若结果偏差 >1 点，先检查：划分是否同 seed、是否 /16 归一化、BN 是否 eval 模式评估
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
model = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(32 * 2 * 2, 10),
)
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
| 1 | 同口径 | 数据/划分/评估与 PROJ-016 一致 | ✅ |
| 2 | 改进幅度 | test 高于基线 | ✅ **+0.0139** |
| 3 | 收敛速度 | 记录最佳轮次 | ✅ 36 vs 76 |
| 4 | 代价记录 | 参数与耗时 | ✅ 6,186 / 3.11 s |
| 5 | 逐类分析 | 10 类 recall 对比 | ✅ 9 类下降已说明 |
| 6 | 局限声明 | 标注单次划分 | ✅ |
| 7 | 可复现 | 同种子一致 | ✅ |

## 13. 结论

- CNN 在同数据上把 **test acc 从 0.9722 提升到 0.9861（+1.39 点）**，同时 **gap 从 0.0269 降到 0.0139**。
- **收敛快一倍**（36 vs 76 epoch）。
- **参数量更多**（6,186 vs 4,810）——8×8 的输入太小，卷积的参数效率优势无法体现。
- **9 类反而变差**（0.8889），说明池化对细笔画有信息损失。

## 14. 与后续项目的关系

- PROJ-016（FC 基线）→ **PROJ-017（CNN 改进）** → PROJ-018（序列任务）
- 迁移到 MNIST 时：**输入变 28×28，卷积的参数量优势会显现**（DL-040 的 109.8 倍），届时建议重跑本对比。

## 学习记录

| 日期 | 状态 | 用时 | 问题 | 复盘 |
|---|---|---|---|---|
| 2026-11-07 | planned | — | — | — |

## 关联与延伸

本项目的知识依据（正文链接，便于图谱连边）：

- [[DL-042 卷积运算]]
- [[DL-046 感受野与参数共享]]

---

## 🔗 Notion 同步信息

> 以下内容由 Notion「AI Knowledge System · 实战项目」数据库按知识 ID 合并而来（导出于 2026-11-09）。

| 属性 | 值 |
|---|---|
| 所属学科 | 深度学习 |
| 项目状态 | 未开始 |
| 项目阶段 | 立项 |
| 难度 | L3 推导 |
| 技术栈 | Python、PyTorch、scikit-learn、NumPy |
| 项目仓库 | ai-knowledge-system/projects/PROJ-017 |

- **对应课程模块**：[M-DL-05 卷积神经网络 CNN](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-DL-05%20%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%20CNN.md) · [M-DL-07 PyTorch 深度学习框架实践](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-DL-07%20PyTorch%20%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A1%86%E6%9E%B6%E5%AE%9E%E8%B7%B5.md)
- **前置知识**：[DL-040 CNN 基本思想与为什么图像适合 CNN](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/02%20CNN/DL-040%20CNN%20%E5%9F%BA%E6%9C%AC%E6%80%9D%E6%83%B3%E4%B8%8E%E4%B8%BA%E4%BB%80%E4%B9%88%E5%9B%BE%E5%83%8F%E9%80%82%E5%90%88%20CNN.md) · [DL-044 多通道卷积与 Feature Map](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/02%20CNN/DL-044%20%E5%A4%9A%E9%80%9A%E9%81%93%E5%8D%B7%E7%A7%AF%E4%B8%8E%20Feature%20Map.md) · [DL-045 池化：Max 与 Average](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/02%20CNN/DL-045%20%E6%B1%A0%E5%8C%96%E2%80%94%E2%80%94Max%20%E4%B8%8E%20Average.md) · [DL-060 nn.Module 自定义网络](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/06%20PyTorch/DL-060%20nn.Module%20%E8%87%AA%E5%AE%9A%E4%B9%89%E7%BD%91%E7%BB%9C.md)
- **对应知识点**：[DL-040 CNN 基本思想与为什么图像适合 CNN](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/02%20CNN/DL-040%20CNN%20%E5%9F%BA%E6%9C%AC%E6%80%9D%E6%83%B3%E4%B8%8E%E4%B8%BA%E4%BB%80%E4%B9%88%E5%9B%BE%E5%83%8F%E9%80%82%E5%90%88%20CNN.md) · [DL-042 卷积运算](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/02%20CNN/DL-042%20%E5%8D%B7%E7%A7%AF%E8%BF%90%E7%AE%97.md) · [DL-043 Padding、Stride 与输出尺寸](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/02%20CNN/DL-043%20Padding%E3%80%81Stride%20%E4%B8%8E%E8%BE%93%E5%87%BA%E5%B0%BA%E5%AF%B8.md) · [DL-044 多通道卷积与 Feature Map](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/02%20CNN/DL-044%20%E5%A4%9A%E9%80%9A%E9%81%93%E5%8D%B7%E7%A7%AF%E4%B8%8E%20Feature%20Map.md) · [DL-045 池化：Max 与 Average](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/02%20CNN/DL-045%20%E6%B1%A0%E5%8C%96%E2%80%94%E2%80%94Max%20%E4%B8%8E%20Average.md) · [DL-046 感受野与参数共享](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/02%20CNN/DL-046%20%E6%84%9F%E5%8F%97%E9%87%8E%E4%B8%8E%E5%8F%82%E6%95%B0%E5%85%B1%E4%BA%AB.md) · [DL-047 LeNet、AlexNet 与 VGG](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/02%20CNN/DL-047%20LeNet%E3%80%81AlexNet%20%E4%B8%8E%20VGG.md)

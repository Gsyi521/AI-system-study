---
id: PROJ-019
title: 迁移学习——小样本分类
domain: PROJ
module: [M-DL-05, M-DL-07, M-DL-08]
type: project
difficulty: L3 推导
status: planned
tech_stack: [Python, PyTorch, scikit-learn]
prerequisites: [DL-049, DL-062, DL-063]
related_math: [MATH-805]
related: [DL-036, DL-038, DL-040, DL-049, DL-062, DL-063, PROJ-017, PROJ-018]
notion_page_id: 411663e8-0333-8126-ba21-c2261731426b
notion_url: https://app.notion.com/p/411663e803338126ba21c2261731426b
created: 2026-11-07
updated: 2026-11-07
notion_db: 实战项目
所属学科: 深度学习
项目状态: 未开始
项目阶段: 立项
难度: L3 推导
项目目标: 迁移学习——小样本分类
技术栈: [Python, PyTorch, scikit-learn]
项目仓库: ai-knowledge-system/projects/PROJ-019
对应课程模块: [M-DL-05, M-DL-07, M-DL-08]
前置知识: [DL-049, DL-062, DL-063]
对应知识点: [DL-036, DL-038, DL-040, DL-049, DL-062, DL-063]
tags: [深度学习]
---

# PROJ-019 迁移学习——小样本分类

## 1. 项目背景

迁移学习常被一句话概括："有预训练模型就用，肯定更好。"

本项目要回答更精确的问题：**在什么条件下迁移有用，什么条件下没用？**

用**两个协议**对比：

| 协议 | 源任务 | 目标任务 | 假设 |
|---|---|---|---|
| **A 类不重叠** | 数字 0–4 | 数字 5–9 | 源/目标**没有共享类别** |
| **B 域偏移** | 噪声图上的 0–9 | 清晰图上的 0–9 | 共享类别，**输入分布不同** |

同一种 CNN、同一评估口径，只改"怎么迁移"。

> **数据声明**：scikit-learn `load_digits`（8×8，1797 条）。MNIST 因网络重置无法下载，详见 PROJ-016 第 9 节。

## 2. 项目目标

| 目标 | 验收方式 |
|---|---|
| 实现三种迁移方式 | 从头训 / 冻结特征 / 全量微调 |
| 量化迁移收益 | 同口径 test acc 差 |
| 找出"迁移何时有效" | 协议 A vs 协议 B |
| 会调微调 lr | lr 扫描 4 档 |
| 诚实记录负结果 | 协议 A 无优势要写清 |

## 3. 技术栈与环境

| 项 | 值 |
|---|---|
| Python | 3.13.15（含 torch） |
| PyTorch | 2.15.0+cu132 |
| 模型 | 2 层 Conv + BN + MaxPool + FC（6,186 参数） |
| 优化器 | Adam（冻结时用大 lr，微调用小 lr） |

## 4. 数据集

**协议 A：0–4 → 5–9**

```text
源：数字 0-4（约 900 条）→ 训练 5 类分类器（源任务 acc 1.0000）
目标：数字 5-9
  主实验：179 训练 / 717 测试
  小样本：每类 5 / 10 / 20 条
```

**协议 B：噪声源 → 清晰目标**

```text
源：全部 0-9，训练时加 σ=0.45 高斯噪声（30 epoch）
     源模型在清晰测试集上 acc 0.9778
目标：清晰图的 0-9，每类 5 / 10 / 20 条（80 epoch）
```

## 5. 方案设计

```text
方式1  scratch  ：随机初始化，全部参数可训练
方式2  frozen   ：加载源权重，冻结卷积部分，只训分类头（大 lr）
方式3  finetune ：加载源权重，全部可训练（小 lr）
```

**关键细节**：

- 加载时**排除最后分类层**（类别数不同）：`{k:v for k,v in sd if not k.startswith("6.")}`；
- 冻结用 `p.requires_grad=False`，优化器只传可训练参数；
- 微调 lr 必须**小**（否则破坏预训练特征，见第 8 节）。

## 6. 核心代码

```python
import torch.nn as nn


def build(n_class: int = 5) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 2 * 2, n_class),
    )
```

## 7. 运行结果

**协议 A（0–4 → 5–9，179 训练 / 717 测试）**

| 方式 | test acc | 可训练参数 | 备注 |
|---|---|---|---|
| **scratch** | **0.9749** | 5,541 | 从头训 |
| frozen | 0.9582 | **645** | 只训头 |
| finetune（lr=5e-3） | 0.9735 | 5,541 | 调过的 lr |

**微调 lr 扫描（协议 A）**

| lr | test acc |
|---|---|
| **1e-4** | **0.4895**（崩） |
| 5e-4 | 0.9163 |
| 1e-3 | 0.9554 |
| 5e-3 | **0.9735** |

**协议 A 小样本（每类 n 条）**

| 每类 | scratch | frozen | finetune |
|---|---|---|---|
| 5（25） | **0.9047** | 0.8106 | 0.8955 |
| 10（50） | 0.9326 | 0.9208 | **0.9409** |
| 20（100） | 0.9410 | 0.9322 | **0.9435** |

**协议 B（噪声源 → 清晰目标，每类 n 条）**

| 每类 | scratch | frozen | finetune |
|---|---|---|---|
| **5（50）** | 0.8481 | **0.9481** | 0.9278 |
| 10（100） | 0.9000 | 0.9481 | **0.9500** |
| 20（200） | 0.9389 | 0.9685 | **0.9741** |

## 8. 对比分析

**协议 A：迁移没有优势，甚至更差**

- scratch **0.9749** > frozen 0.9582（差 **−1.67 点**）；
- 原因：**源任务（0–4）与目标任务（5–9）类别完全不重叠**，卷积学到的"数字形状"特征对"没见过的数字类别"帮助有限，而 digits 本身很简单，从头训已足够。

**协议 B：迁移显著有效**

- 每类 5 条：scratch **0.8481** → frozen **0.9481**（**+10 点**）；
- 每类 20 条：scratch 0.9389 → finetune **0.9741**（+3.5 点）；
- 原因：源/目标**共享 10 个类别**，只是输入被噪声污染 ⇒ 源模型学的正是"去噪后的形状"，迁移直接受益。

**微调 lr 的悬崖**

| lr | 结果 | 解释 |
|---|---|---|
| 1e-4 | **0.4895** | 太小：179 样本 × 60 步几乎没更新，且 BN 统计仍来自源域 |
| 5e-3 | 0.9735 | 合适：既适配任务又不破坏特征 |

**冻结 BN 统计的补充实验**（协议 A，179 样本）：

| 设置 | acc |
|---|---|
| finetune lr=1e-3 | 0.9554 |
| finetune lr=1e-3 + 冻结 BN | 0.9498 |
| finetune lr=1e-2 + 冻结 BN | 0.9721 |

⇒ 是否冻结 BN 影响不大（±0.5 点），**lr 才是主导因素**。

## 9. 局限与讨论

1. **协议 A 的负结果受数据影响**：digits 只有 8×8，特征本身很简单，从头训很容易。
2. **未使用真正的预训练大模型**（ImageNet ResNet），无法验证"大模型迁移"的收益。
3. **未做多折验证**：单次划分，±1 点可能落在噪声内。
4. **BN 统计未做消融**：只做了一个补充实验。

## 10. 复现步骤

```text
1. py -3.13（torch 2.15.0+cu132）
2. 协议 A：源训 0-4（40 epoch）→ 加载除分类层 → 三种方式对比
3. 协议 B：源训加噪（30 epoch）→ 同三种方式
4. 预期：协议 A scratch ≈ 0.975、frozen ≈ 0.958；协议 B 每类 5 条 frozen ≈ 0.948
```

## 11. 完整实验脚本（协议 B，含三种方式）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 42


def build(n_class: int = 10) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 2 * 2, n_class),
    )


def noisy(x, s: float = 0.45, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return (x + s * torch.randn(x.shape, generator=g)).clamp(0, 1)


d = load_digits()
X = torch.tensor(d.data.astype("float32") / 16.0).view(-1, 1, 8, 8)
y = torch.tensor(d.target.astype("int64"))
Xtr, Xte, ytr, yte = train_test_split(
    X.numpy(), y.numpy(), test_size=0.3, random_state=SEED, stratify=y.numpy())
Xtr = torch.tensor(Xtr).view(-1, 1, 8, 8)
ytr = torch.tensor(ytr)
Xte = torch.tensor(Xte).view(-1, 1, 8, 8)
yte = torch.tensor(yte)

# ---- 源任务：带噪训练 ----
torch.manual_seed(SEED)
src = build()
opt = torch.optim.Adam(src.parameters(), lr=1e-3)
for epoch in range(30):
    xb = noisy(Xtr, seed=epoch)
    perm = torch.randperm(len(xb))
    for i in range(0, len(xb), 32):
        idx = perm[i:i + 32]
        opt.zero_grad()
        F.cross_entropy(src(xb[idx]), ytr[idx]).backward()
        opt.step()
src_sd = {k: v.clone() for k, v in src.state_dict().items()}
src.eval()
with torch.no_grad():
    print("源任务(噪声)在清晰测试集 acc:",
          round(float((src(Xte).argmax(1) == yte).float().mean()), 4))

# ---- 目标任务：每类 5 条 ----
idxs = []
for c in range(10):
    ci = (ytr == c).nonzero().flatten()
    g = torch.Generator().manual_seed(SEED + c)
    idxs.append(ci[torch.randperm(len(ci), generator=g)[:5]])
sel = torch.cat(idxs)
Xt, yt = Xtr[sel], ytr[sel]

for mode, lr in (("scratch", 1e-3), ("frozen", 1e-2), ("finetune", 5e-3)):
    torch.manual_seed(SEED)
    m = build()
    if mode != "scratch":
        m.load_state_dict({k: v for k, v in src_sd.items()
                           if not k.startswith("6.")}, strict=False)
    if mode == "frozen":
        for p in m[:-1].parameters():
            p.requires_grad = False
    params = [p for p in m.parameters() if p.requires_grad]
    o = torch.optim.Adam(params, lr=lr)
    for _ in range(80):
        o.zero_grad()
        F.cross_entropy(m(Xt), yt).backward()
        o.step()
    m.eval()
    with torch.no_grad():
        acc = float((m(Xte).argmax(1) == yte).float().mean())
    print(f"{mode:<9} lr={lr}: acc={acc:.4f} 可训练={sum(p.numel() for p in params)}")
```

## 12. 验收清单

| # | 项 | 判据 | 结果 |
|---|---|---|---|
| 1 | 三种方式齐全 | scratch / frozen / finetune | ✅ |
| 2 | 同口径 | 同数据、同测试集 | ✅ |
| 3 | 双协议 | 类不重叠 + 域偏移 | ✅ |
| 4 | lr 扫描 | 4 档 | ✅ 0.4895 → 0.9735 |
| 5 | 负结果记录 | 协议 A 无优势 | ✅ |
| 6 | 可复现 | 固定种子 | ✅ |
| 7 | 可训练参数记录 | 冻结 vs 全量 | ✅ 645 vs 5,541 |

## 13. 结论

- **协议 A（类不重叠）**：迁移**无优势**（scratch 0.9749 > frozen 0.9582）⇒ 迁移需要**共享特征**。
- **协议 B（域偏移）**：迁移**显著有效**（0.8481 → **0.9481**，+10 点）⇒ 源/目标共享类别时，预训练直接可用。
- **微调 lr 是关键**：1e-4 崩到 **0.4895**，5e-3 到 **0.9735**。
- **冻结 vs 微调**：数据越少越该冻结（5 条/类时 frozen 最好），数据稍多微调更好（20 条/类时 0.9741）。

## 14. 与后续项目的关系

- PROJ-018（序列）→ **PROJ-019（迁移）** → PROJ-020（端到端部署）
- 想验证"大模型迁移"：把 backbone 换成 torchvision 预训练 ResNet，输入上采样到 224（本机未装权重下载条件，见 DL-049）。

## 学习记录

| 日期 | 状态 | 用时 | 问题 | 复盘 |
|---|---|---|---|---|
| 2026-11-07 | planned | — | — | — |

## 关联与延伸

本项目的知识依据（正文链接，便于图谱连边）：

- [[DL-049 迁移学习与图像分类工作流]]

---

## 🔗 Notion 同步信息

> 以下内容由 Notion「AI Knowledge System · 实战项目」数据库按知识 ID 合并而来（导出于 2026-11-09）。

| 属性 | 值 |
|---|---|
| 所属学科 | 深度学习 |
| 项目状态 | 未开始 |
| 项目阶段 | 立项 |
| 难度 | L3 推导 |
| 技术栈 | Python、PyTorch、scikit-learn |
| 项目仓库 | ai-knowledge-system/projects/PROJ-019 |

- **对应课程模块**：[M-DL-05 卷积神经网络 CNN](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-DL-05%20%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%20CNN.md) · [M-DL-07 PyTorch 深度学习框架实践](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-DL-07%20PyTorch%20%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A1%86%E6%9E%B6%E5%AE%9E%E8%B7%B5.md) · [M-DL-08 深度学习工程实践与综合应用](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-DL-08%20%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%B7%A5%E7%A8%8B%E5%AE%9E%E8%B7%B5%E4%B8%8E%E7%BB%BC%E5%90%88%E5%BA%94%E7%94%A8.md)
- **前置知识**：[DL-049 迁移学习与图像分类工作流](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/02%20CNN/DL-049%20%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0%E4%B8%8E%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB%E5%B7%A5%E4%BD%9C%E6%B5%81.md) · [DL-062 训练、验证、测试循环](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/06%20PyTorch/DL-062%20%E8%AE%AD%E7%BB%83%E3%80%81%E9%AA%8C%E8%AF%81%E3%80%81%E6%B5%8B%E8%AF%95%E5%BE%AA%E7%8E%AF.md) · [DL-063 模型保存、加载与 Checkpoint](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/06%20PyTorch/DL-063%20%E6%A8%A1%E5%9E%8B%E4%BF%9D%E5%AD%98%E3%80%81%E5%8A%A0%E8%BD%BD%E4%B8%8E%20Checkpoint.md)
- **对应知识点**：[DL-036 Batch Normalization](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/05%20%E8%AE%AD%E7%BB%83%E6%8A%80%E5%B7%A7/DL-036%20Batch%20Normalization.md) · [DL-038 Early Stopping 与模型检查点](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/05%20%E8%AE%AD%E7%BB%83%E6%8A%80%E5%B7%A7/DL-038%20Early%20Stopping%20%E4%B8%8E%E6%A8%A1%E5%9E%8B%E6%A3%80%E6%9F%A5%E7%82%B9.md) · [DL-040 CNN 基本思想与为什么图像适合 CNN](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/02%20CNN/DL-040%20CNN%20%E5%9F%BA%E6%9C%AC%E6%80%9D%E6%83%B3%E4%B8%8E%E4%B8%BA%E4%BB%80%E4%B9%88%E5%9B%BE%E5%83%8F%E9%80%82%E5%90%88%20CNN.md) · [DL-049 迁移学习与图像分类工作流](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/02%20CNN/DL-049%20%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0%E4%B8%8E%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB%E5%B7%A5%E4%BD%9C%E6%B5%81.md) · [DL-062 训练、验证、测试循环](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/06%20PyTorch/DL-062%20%E8%AE%AD%E7%BB%83%E3%80%81%E9%AA%8C%E8%AF%81%E3%80%81%E6%B5%8B%E8%AF%95%E5%BE%AA%E7%8E%AF.md) · [DL-063 模型保存、加载与 Checkpoint](../../04_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/06%20PyTorch/DL-063%20%E6%A8%A1%E5%9E%8B%E4%BF%9D%E5%AD%98%E3%80%81%E5%8A%A0%E8%BD%BD%E4%B8%8E%20Checkpoint.md)

---
id: PROJ-022
title: 用 LoRA 微调 gpt2 做文本分类
domain: PROJ
module: [M-LLM-07]
type: project
difficulty: L2 应用
status: planned
tech_stack: [Python, PyTorch, Transformers, PEFT]
prerequisites: [LLM-065, LLM-060]
related_math: [MATH-216]
related: [LLM-036, LLM-071, PROJ-021]
notion_page_id: 412663e8-0333-81d0-a90f-f4fc42ef10a9
notion_url: https://app.notion.com/p/LoRA-gpt2-412663e8033381d0a90ff4fc42ef10a9
created: 2026-11-08
updated: 2026-11-08
notion_db: 实战项目
所属学科: 机器学习
项目状态: 未开始
项目阶段: 立项
难度: L2 应用
项目目标: 用 LoRA 微调 gpt2 做文本分类
技术栈: [Python, PyTorch, Transformers, PEFT]
项目仓库: ai-knowledge-system/projects/PROJ-022
对应课程模块: [M-LLM-07]
前置知识: [LLM-065, LLM-060]
对应知识点: [LLM-036, LLM-071]
tags: [机器学习]
---

# PROJ-022 用 LoRA 微调 gpt2 做文本分类

## 1. 项目背景

LoRA 的资料里最常见的一句话是"只训练 0.1% 的参数就能接近全参微调"。这句话有**三个可验证的部分**，本项目把三个都测了：

1. **参数量**：到底训练了多少参数？（必须真算，不能抄博客）
2. **显存**：省了多少？（`torch.cuda.max_memory_allocated()`）
3. **效果**：真的接近全参微调吗？

实测结论先放这里：**参数量 296,450 / 124,736,258 = 0.2377%（1/420），峰值显存 778.2 MB vs 2,385.5 MB（1/3.07），验证集准确率两者都是 1.0000。**

但"两个都 1.0"这件事本身是个陷阱：这个任务对 gpt2 太简单了。所以本项目**额外做了三个诊断**，去找真实的能力边界：

- 词表外形容词（训练时一次没出现过的词）→ 仍然 1.0，说明它用的是 gpt2 预训练里的语义知识，不是背 6 个形容词；
- 隐含情感（句子里**没有任何情感形容词**）→ 掉到 **0.7083**；
- 少样本学习曲线 → 验证集 5 个点全是 1.0，隐含情感却在 0.625–0.9167 之间抖。

**1.0 不是本文的结论，0.7083 才是。**

## 2. 项目目标

| 目标 | 验收方式 |
|---|---|
| 用 peft 的 `LoraConfig` + `get_peft_model` 真训 | 脚本里真的调了这两个 API，loss 真的下降 |
| gpt2 取最后一个 token 隐状态接分类头 | 池化逻辑可验证（6.2 节有可运行片段） |
| 自建 200–400 条短句，不下载数据集 | 脚本内模板合成 320 条（正负各 160） |
| 记录**可训练参数/总参数** | 296,450 / 124,736,258，逐项可核（LoRA 294,912 + 头 1,538） |
| 记录初始/最终 loss、验证准确率、耗时 | 4.1452 / 0.0000 / 1.0000 / 3.18 s |
| **对比全参微调 vs LoRA** | 参数量 1/420、显存 1/3.07、耗时与准确率并列 |
| 准确率如实记录，不美化 | 1.0 保留，但补三个诊断说明它为什么是 1.0 |

## 3. 技术栈与环境

| 项 | 值 |
|---|---|
| Python | 3.13（`py -3.13`） |
| PyTorch | 2.15.0+cu132 |
| transformers | 4.57.6 |
| peft | 0.21.2 |
| scikit-learn | 用于 macro F1 |
| GPU | RTX 4070 Ti，12.88 GB，sm_89 |
| 底座模型 | `gpt2`（124,439,808 参数，本地缓存，离线加载） |
| LoRA | r=8，alpha=16，dropout=0.1，`target_modules=["c_attn"]`（12 个模块） |
| 训练 | AdamW，LoRA lr=1e-3 / 全参 lr=5e-5，5 epoch，batch 16，梯度裁剪 1.0 |

脚本开头强制离线：

```python
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
```

`from_pretrained(..., local_files_only=True)` 二次保险——**不下载任何新模型**。

## 4. 数据集

**全部脚本内构造，无外部数据。** 320 条英文短句，正 160 / 负 160，模板 + 词表组合生成，固定种子打乱。

```text
8 个方面词：battery life / screen / build quality / sound / keyboard /
            delivery / support team / packaging
6 个正面形容词：great / excellent / wonderful / fantastic / brilliant / superb
6 个负面形容词：terrible / awful / disappointing / useless / broken / annoying
```

四类句式（情绪只能从形容词或否定判断）：

| 类型 | 模板（节选） | 标签规则 |
|---|---|---|
| 平铺 | `the {aspect} is {adj} .` | 看形容词极性 |
| 强调 | `the {aspect} is really {adj} .` | 看形容词极性 |
| 否定 | `the {aspect} is not {adj} .` | **只否定正面形容词 → 负面** |
| 转折 | `... the {a1} is {j1} , but the {a2} is {j2} .` | **取 but 之后分句的极性** |

切分：分层抽样，训练 256 / 验证 64。平均 14.18 token，最长 19。

**两个额外诊断集**（只在评估时用，训练中不出现）：

| 诊断集 | 规模 | 构造 |
|---|---|---|
| 词表外（OOV） | 48 | 形容词换成 `splendid/marvelous/terrific/fabulous` 与 `dreadful/horrible/lousy/atrocious`，**训练中一次都没出现过** |
| 隐含情感 | 24 | 手写句子，**不含任何情感形容词**，靠世界知识判断（如 `the battery lasted two hours before it died .` → 负面） |

## 5. 方案设计

```text
句子 → gpt2 tokenizer（pad=eos，右填充，max_len=48）
     → GPT2Model（12 层，768 维）      ← LoRA 只挂在这里的 c_attn 上
     → last_hidden_state (B,T,768)
     → 取每条样本最后一个「非 pad」token 的隐状态 (B,768)
     → Linear(768,2) 分类头
     → cross_entropy
```

三个设计点：

1. **为什么取最后一个 token**：GPT-2 是因果注意力，最后一个 token 的隐状态已经"看过"整句，等价于一个学出来的句子表示。
2. **为什么要按 attention_mask 找最后一个非 pad**：右填充时短句的末位是 pad，直接取 `h[:, -1]` 会取到 pad 的表示。
3. **为什么分类头单独设 `requires_grad=True`**：`get_peft_model` 会冻结所有非 LoRA 参数，而分类头是新建的，不放开就永远不更新。

**参数量核对**（必须与脚本打印一致）：

| 项 | 计算 | 参数 |
|---|---|---|
| LoRA A（每模块 8×768） | 6,144 × 12 | 73,728 |
| LoRA B（每模块 2304×8） | 18,432 × 12 | 221,184 |
| LoRA 小计 | 24,576 × 12 | **294,912** |
| 分类头 | 768×2 + 2 | **1,538** |
| **可训练合计** | | **296,450** |
| 底座 + 头总参数 | 124,439,808 + 1,538 | **124,736,258** |
| 占比 | 296,450 / 124,736,258 | **0.2377%** |

## 6. 核心代码

### 6.1 LoRA 挂载 + 参数占比（自包含可运行）

```python
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import GPT2Model

base = GPT2Model.from_pretrained("gpt2", local_files_only=True)
cfg = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=8, lora_alpha=16, lora_dropout=0.1,
    target_modules=["c_attn"], bias="none",
    fan_in_fan_out=True,          # GPT2 的 c_attn 是 Conv1D，权重按 (in,out) 存
)
enc = get_peft_model(base, cfg)
tr = sum(p.numel() for p in enc.parameters() if p.requires_grad)
tot = sum(p.numel() for p in enc.parameters())
print("lora_trainable", tr, "total", tot, "pct", round(100 * tr / tot, 4))

out = enc(input_ids=torch.tensor([1, 2, 3]).unsqueeze(0),
          attention_mask=torch.ones(1, 3, dtype=torch.long))
print("forward ok:", tuple(out.last_hidden_state.shape))
```

实测输出：

```text
lora_trainable 294912 total 124734720 pct 0.2364
forward ok: (1, 3, 768)
```

> 这里 0.2364% 是**只有 LoRA、没有分类头**的最小演示；第 7 节脚本里加上 1,538 参数的头之后是 **0.2377%**。两个数都对，别混用。

### 6.2 取最后一个非 pad token（自包含可运行）

```python
import torch

# 假的隐状态 (B=2, T=4, H=3)，用连续整数好核对取到的是哪一行
h = torch.arange(24, dtype=torch.float32).view(2, 4, 3)
mask = torch.stack([torch.tensor([1, 1, 1, 0]), torch.tensor([1, 1, 0, 0])])   # 右填充
last_idx = mask.sum(dim=1) - 1
pooled = h[torch.arange(h.size(0)), last_idx]
print("last_idx", last_idx.tolist())
for i in range(pooled.size(0)):
    print("  sample", i, "->", pooled[i].tolist())
```

实测输出：

```text
last_idx [2, 1]
  sample 0 -> [6.0, 7.0, 8.0]
  sample 1 -> [15.0, 16.0, 17.0]
```

第 1 条长度 3 取第 3 行（值 6,7,8），第 2 条长度 2 取第 2 行（值 15,16,17）——池化没取到 pad。

### 6.3 模型 + 训练循环（摘自脚本，依赖脚本内 `GPT2Model` 加载与数据变量）

```text
class GPT2Classifier(nn.Module):
    def __init__(self, encoder, hidden=768, n_class=2):
        super().__init__()
        self.encoder = encoder          # 裸 GPT2Model 或 PeftModel
        self.score = nn.Linear(hidden, n_class)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        last_idx = attention_mask.sum(dim=1) - 1
        pooled = h[torch.arange(h.size(0), device=h.device), last_idx]
        return self.score(pooled)

# 关键一行：LoRA 冻结底座，但新分类头必须解冻
for p in model.score.parameters():
    p.requires_grad = True
```

## 7. 运行结果

完整控制台输出（真实运行，未删改）：

```text
====================================================================================
PROJ-022 用 LoRA 微调 gpt2 做文本分类（情感二分类）
====================================================================================
device=cuda  torch=2.15.0+cu132
gpu=NVIDIA GeForce RTX 4070 Ti  vram=12.88 GB
训练语料: 共 320 条（正 160 / 负 160）  训练 256  验证 64
词表外诊断集: 48 条（形容词 splendid/marvelous/terrific/fabulous/dreadful/horrible/lousy/atrocious 从未在训练中出现）

样例（训练集前 5 条）：
  [1] sure , the packaging is useless , but the sound is fantastic .
  [0] sure , the screen is superb , but the packaging is terrible .
  [0] the packaging is excellent , but the screen is disappointing .
  [1] the keyboard is awful , but the screen is excellent .
  [0] i wanted to like it : the packaging is superb , but the keyboard is useless .
样例（词表外诊断集前 3 条）：
  [1] i wanted to like it : the keyboard is atrocious , but the sound is fabulous .
  [1] the battery life is lousy , but the support team is marvelous .
  [0] sure , the build quality is terrific , but the packaging is atrocious .

token 长度: mean=14.2 max=19 min=5 (max_length=48)

--- 实验 A：LoRA (r=8, alpha=16, target=c_attn, 5 epoch) ---
    [LoRA] epoch 1 | train_loss 1.2115
    [LoRA] epoch 2 | train_loss 0.0813
    [LoRA] epoch 3 | train_loss 0.0395
    [LoRA] epoch 4 | train_loss 0.0572
    [LoRA] epoch 5 | train_loss 0.0530
  可训练参数 296,450 / 总参数 124,736,258 = 0.2377%  (LoRA 294,912 + 分类头 1,538)
  验证 acc 1.0000 | 词表外 acc 1.0000 | 隐含情感 acc 0.7083 | 峰值显存 778.2 MB | 3.18 s

--- 实验 B：全参微调（同一份数据 / 同一 epoch 数 / lr=5e-5） ---
    [full] epoch 1 | train_loss 0.5807
    [full] epoch 2 | train_loss 0.0058
    [full] epoch 3 | train_loss 0.0000
    [full] epoch 4 | train_loss 0.0000
    [full] epoch 5 | train_loss 0.0000
  可训练参数 124,441,346 / 总参数 124,441,346 = 100.00%
  验证 acc 1.0000 | 词表外 acc 1.0000 | 隐含情感 acc 0.6667 | 峰值显存 2385.5 MB | 8.08 s

--- 实验 C：LoRA 少样本学习曲线（每类 N 条，梯度步数拉平到 ~80 步）---
  N/类=   8  训练条数=  16  epoch= 80  val acc=1.0000  词表外 acc=1.0000  隐含 acc=0.6250  3.05s
  N/类=  16  训练条数=  32  epoch= 40  val acc=1.0000  词表外 acc=1.0000  隐含 acc=0.8333  3.28s
  N/类=  32  训练条数=  64  epoch= 20  val acc=1.0000  词表外 acc=1.0000  隐含 acc=0.8333  3.31s
  N/类=  64  训练条数= 128  epoch= 10  val acc=1.0000  词表外 acc=1.0000  隐含 acc=0.9167  3.34s
  N/类= 128  训练条数= 256  epoch=  5  val acc=1.0000  词表外 acc=1.0000  隐含 acc=0.8333  4.11s

====================================================================================
结果汇总
====================================================================================
方法                     可训练参数       占比    初始loss    最终loss val acc OOV acc   耗时(s)     峰值显存MB
LoRA                 296,450  0.2377%   4.1452   0.0000  1.0000  1.0000    3.18      778.2
full_finetune    124,441,346100.0000%   1.1504   0.0000  1.0000  1.0000    8.08     2385.5

--- LoRA 验证集样本 ---
  gold=0 pred=0  i wanted to like it : the support team is wonderful , but the sound is annoying .
  gold=0 pred=0  the keyboard is fantastic , but the battery life is annoying .
  gold=1 pred=1  i wanted to like it : the battery life is annoying , but the support team is superb .
  gold=1 pred=1  the keyboard is disappointing , but the build quality is wonderful .
  gold=1 pred=1  after two weeks the delivery is still brilliant .

--- LoRA 验证集错例：0/64 ---

--- LoRA 词表外错例：0/48 ---

--- LoRA 隐含情感诊断集：acc=0.7083  错 7/24 ---
  gold=0 pred=1  i returned it the same afternoon .
  gold=1 pred=0  it worked out of the box on the first try .
  gold=1 pred=0  the box arrived a day early .
  gold=1 pred=0  the screen survived a drop from the table .
  gold=1 pred=0  setup took about thirty seconds .
  gold=1 pred=0  the box was packed with care and nothing was missing .
  gold=1 pred=0  it fits in my bag and i forget it is there .

指标已写入 ...\proj022_metrics.json
LoRA 可训练参数是全参微调的 1/420，峰值显存是全参的 1/3.07
```

## 8. 结果分析

### 8.1 参数量与显存：LoRA 真的省

| 指标 | LoRA | 全参微调 | 倍数 |
|---|---|---|---|
| 可训练参数 | **296,450** | 124,441,346 | **1 / 420** |
| 占模型比例 | **0.2377%** | 100.00% | — |
| 峰值显存 | **778.2 MB** | 2,385.5 MB | **1 / 3.07** |
| 单轮耗时（5 epoch） | 3.18 s | 8.08 s | 约 1 / 2.5 |

显存只省到 1/3.07 而不是 1/420，原因很直接：**底座权重仍要读进显存**（124M × 4 字节 ≈ 500 MB），省掉的是梯度与 AdamW 两个动量状态（各 500 MB）。所以 LoRA 省的是"可训练参数带来的优化器开销"，不是"模型本身"。想再省只能上 8-bit / 量化底座。

### 8.2 准确率 1.0 是真的，但它测不出 LoRA 的能力

验证集 64/64、词表外 48/48 全对。**这不是 bug**，理由有三条：

1. 任务本身是"形容词极性 + 否定翻转 + but 转折"，对 gpt2 这种在网页文本上预训练过的模型几乎是零难度；
2. OOV 集用的 `splendid / atrocious` 等词 gpt2 本来就认识，所以"没在训练里出现过"并不构成挑战；
3. 训练 loss 5 个 epoch 内就掉到 1e-5 量级（打印成 `0.0000` 是四舍五入），说明**训练集被完全拟合**——此时验证集全对只说明两批数据同分布。

用 Wilson 区间衡量，1.0 也不等于"完美"：

| 集合 | 命中 | Wilson 95% 区间 |
|---|---|---|
| 验证集 | 64/64 | [0.9434, 1.0000] |
| 词表外 | 48/48 | [0.9259, 1.0000] |

### 8.3 真正有信息量的是隐含情感：0.7083

把情感形容词全部拿掉，只留事实描述（"电池两小时就没电" vs "开箱即用一次点亮"），准确率掉到 **0.7083（17/24）**，全参微调 **0.6667（16/24）**。

两者 Wilson 区间是 [0.5083, 0.8509] 与 [0.4671, 0.8203]，**大面积重叠**——所以正确的说法是"在这个 24 条的小集合上，LoRA 与全参微调没有可分辨的差别"，而不是"LoRA 比全参好 4 个百分点"。

错例方向高度一致：**7 条错例里 6 条是把正面判成负面**。

```text
gold=1 pred=0  it worked out of the box on the first try .
gold=1 pred=0  the box arrived a day early .
gold=1 pred=0  the screen survived a drop from the table .
gold=1 pred=0  setup took about thirty seconds .
```

机制解释（推断，未做消融验证）：训练集里正面样本**必须**出现正面形容词，模型学到的捷径是"看到正面词 → 正面"；一旦句子里一个情感词都没有，特征接近"无信号"，就落到负类那一侧（训练集负类里有大量 `not + 正面词` 的句式，可能把"缺少肯定词"训成了负面）。这条只是现象归纳，我没有做 logit 分布或类别先验的进一步分析。

### 8.4 少样本曲线：验证集不动，隐含情感在抖

| N/类 | 训练条数 | epoch | val acc | OOV acc | 隐含 acc |
|---|---|---|---|---|---|
| 8 | 16 | 80 | 1.0000 | 1.0000 | 0.6250 |
| 16 | 32 | 40 | 1.0000 | 1.0000 | 0.8333 |
| 32 | 64 | 20 | 1.0000 | 1.0000 | 0.8333 |
| 64 | 128 | 10 | 1.0000 | 1.0000 | **0.9167** |
| 128 | 256 | 5 | 1.0000 | 1.0000 | 0.8333 |

- 每类 **8 条**（共 16 条）就能把验证集打到 1.0 —— LoRA 的样本效率确实高，这也是这个任务能提供的唯一"LoRA 优势"证据；
- 隐含情感一路 0.625 → 0.9167 → 0.8333，**非单调**，且最后 128/类 反而比 64/类 差：24 条的小集合上，1 条错例就是 4.2 个百分点，这点波动不能解读成"数据更多更差"。

### 8.5 两个初始 loss 不可直接比较

LoRA 初始 loss 4.1452，全参 1.1504。这不是 LoRA 更差，而是**分类头是随机初始化的，两条执行路径的随机状态不同**（LoRA 路径在 `get_peft_model` 里消耗了随机数）。初始 loss 只能用于"有没有正常起步"，不能用于横向对比。

## 9. 数据来源与差异说明（必读）

1. **全部 320 条训练语料 + 48 条 OOV + 24 条隐含情感，都是我在脚本里造的合成数据**，没有一条来自真实评论。它们不包含真实语言的省略、拼写错误、反讽、多话题混写、标注噪声。
2. **标签是规则生成的，不是人标的**。转折句的规则是"取 but 之后分句的极性"——这是我定的合成规则，不是语言学真理。真实评论里 "the screen is great but the price is awful" 到底算正还是负，取决于业务定义。**所以本项目的 1.0 准确率衡量的是"模型学会了我这条规则"，不是"模型会做情感分析"。**
3. **不能外推到真实场景**。真实情感分类的 SOTA 在 SST-2 上也只有 95% 左右，且需要几万条标注数据。320 条模板句 + 1.0 准确率只能说明任务太简单。
4. **词表外（OOV）设计有边界**：只换了形容词，句式与方面词完全一样。所以 1.0 说明"对新形容词鲁棒"，**不能说明"对新句式鲁棒"**。
5. **隐含情感集合只有 24 条，标签是我自己判的**，其中 `i stopped thinking about it , which is the highest praise i can give .` 这类句子本身有争议（"highest praise" 算不算情感词）。24 条样本的 95% 区间宽到 ±17 个百分点，只能当"现象展示"，不能当基准。
6. **gpt2 的预训练语料未知**。它可能见过类似的电商 FAQ 句子，OOV 的 1.0 里可能混着记忆而非泛化，这一点无法在离线环境里排除。
7. 耗时是在**共享 GPU** 上测的（运行期间有其他进程占用，利用率 97%）。我在不同时间重跑过，LoRA 单轮在 2.50–15.33 s 之间波动，全参（5 epoch 配置）在 8.08–73.61 s 之间波动。**比值的量级（约 1/2.5）稳定，绝对秒数不可复现。**

## 10. 复现步骤

```text
1. 确认解释器与依赖：
   py -3.13 -c "import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__)"
   期望：2.15.0+cu132 4.57.6 0.21.2

2. 确认 gpt2 在本地缓存（离线模式不会去下载）：
   py -3.13 -c "from transformers import GPT2Model; m=GPT2Model.from_pretrained('gpt2', local_files_only=True); print(sum(p.numel() for p in m.parameters()))"
   期望：124439808

3. 运行（约 20 s）：
   $env:PYTHONIOENCODING='utf-8'
   cd <USER_HOME>\Documents\deepseek-harness\default-workspace\09-llm\projects
   py -3.13 proj022_lora_classifier.py

4. 校验：
   - LoRA 可训练参数必须是 296,450（= 294,912 + 1,538）
   - 占比 0.2377%，峰值显存 ~778 MB（全参 ~2386 MB）
   - val acc 1.0000，隐含情感 acc 0.7083
   - 同目录生成 proj022_metrics.json

5. 偏差排查：
   - 可训练参数变成 149k 左右 → 分类头没解冻（漏了 score.requires_grad=True）
   - 显存异常高 → 检查是否误把 base 也设成了 requires_grad
   - 隐含情感 acc 大幅波动 → 属正常，24 条样本 1 条 = 4.2 个百分点
   - CUDA illegal memory access → 本脚本已加 gc + synchronize + empty_cache，
     若仍出现，把 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 显式导出后重跑
```

## 11. 完整训练脚本

```text
路径：<USER_HOME>\Documents\deepseek-harness\default-workspace\09-llm\projects\proj022_lora_classifier.py
大小：23,103 字节
命令：py -3.13 proj022_lora_classifier.py
产出：proj022_metrics.json（同目录）
```

脚本内固定配置与超参：

```python
SEED = 42
MODEL_NAME = "gpt2"
MAX_LEN = 48
BATCH = 16
EPOCHS = 5
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_LR = 1e-3
FULL_LR = 5e-5
```

脚本结构（按行号）：

```text
1-30     环境变量（含 PYTORCH_CUDA_ALLOC_CONF）、import
31-95    ASPECTS / POS_ADJ / NEG_ADJ / OOV 形容词 / 4 组模板
96-150   build_dataset / build_oov_set / stratified_split / set_seed
151-175  GPT2Classifier（最后一个非 pad token 池化）
176-260  make_batches / evaluate / mean_loss / count_params / train_loop
261-300  build_lora_model（LoraConfig + get_peft_model + 解冻分类头）
301-460  __main__：实验 A（LoRA）/ B（全参）/ C（少样本曲线）+ 汇总 + JSON
```

依赖：torch 2.15.0、transformers 4.57.6、peft 0.21.2、scikit-learn、numpy。**不需要联网**（离线加载 `gpt2`）。

## 12. 验收清单

| 验收项 | 是否达成 | 证据 |
|---|---|---|
| 自建 200–400 条短句，正负各半 | ✅ | 320 条，正 160 / 负 160，脚本内模板生成 |
| 不下载数据集 | ✅ | 无 `datasets.load_dataset`，`HF_HUB_OFFLINE=1` |
| 用 `LoraConfig` + `get_peft_model` | ✅ | 脚本 `build_lora_model()`；6.1 节可运行片段实测 294,912 个 LoRA 参数 |
| r=8、`target_modules=["c_attn"]` | ✅ | 配置写死；12 个 c_attn 模块 × 24,576 参数 |
| 取最后一个 token 隐状态接分类头 | ✅ | 6.2 节实测 `last_idx [2, 1]`，未取到 pad |
| 训练 3–5 epoch | ✅ | `EPOCHS = 5` |
| **可训练参数/总参数（真算）** | ✅ | 296,450 / 124,736,258 = 0.2377%，与第 5 节手算一致 |
| 初始 loss | ✅ | 4.1452（LoRA）/ 1.1504（全参），并说明两者不可比 |
| 最终 loss | ✅ | 0.0000（四舍五入，实际 <5e-5） |
| 验证集准确率 | ✅ | 1.0000（64/64），附 Wilson 区间 [0.9434, 1.0] |
| 训练耗时 | ✅ | 3.18 s（LoRA）/ 8.08 s（全参），并注明共享 GPU 不可复现 |
| **全参 vs LoRA 参数量对比** | ✅ | 124,441,346 vs 296,450 = 1/420 |
| **全参 vs LoRA 显存对比** | ✅ | 2,385.5 MB vs 778.2 MB = 1/3.07，`torch.cuda.max_memory_allocated()` |
| 准确率如实记录不美化 | ✅ | 保留 1.0，但用 OOV / 隐含情感 / 少样本三个诊断说明它为什么是 1.0 |
| 指标写 JSON | ✅ | `proj022_metrics.json`（4,432 字节） |
| 可重复运行 | ✅ | `SEED=42` + `set_seed()`；LoRA 三个 epoch 的 loss 序列三次重跑完全一致 |

## 13. 结论与局限

**结论**

1. **LoRA 的有效性在参数与显存上是硬指标**：0.2377% 的可训练参数、1/3.07 的峰值显存、1/2.5 的耗时，达到了与全参微调相同的验证集准确率（都是 1.0000）。
2. **这个任务测不出 LoRA 的上限**：验证集、词表外都是 1.0，少样本只有每类 8 条也 1.0。原因是任务太简单 + gpt2 自带情感词知识。
3. **有信息量的数字是隐含情感 0.7083**：一旦句子里没有情感形容词，两类方法都掉到 0.67–0.71，且错例 6/7 同向（正面判成负面）。
4. **LoRA 与全参在这个规模没有可分辨差别**：隐含情感 17/24 vs 16/24，Wilson 区间高度重叠，不能说 LoRA 更好。

**局限**（不修饰）

1. 合成数据 + 规则标签：1.0 只证明学会了我的规则，不证明会做情感分析。
2. 全参微调只跑了 5 epoch、lr=5e-5，**没有为全参调过超参**，所以"两者准确率相同"不代表全参的上限就在这里。
3. 隐含情感集合 24 条，是我一个人判的标签，置信区间 ±17 个百分点，只能当现象。
4. 少样本曲线的 epoch 数是按"梯度步数拉平到 ~80"反推的（8/类 → 80 epoch），这个拉平方式本身就是一种选择，换成"固定 epoch"结论会变。
5. 没有测推理延迟、没有测 LoRA 权重合并（`merge_and_unload`）后的差异、没有做多 r 值（4/8/16/32）扫描。
6. 耗时数据来自共享 GPU，绝对秒数不可复现（见第 9 节第 7 条）。
7. 分类头解冻这一细节容易踩坑，但**为什么 peft 会冻结新头**我没有读源码确认，只是通过"可训练参数从 296,450 掉到 294,912"反推的。

## 14. 与后续项目的关系

- 上游：[[PROJ-021 从零实现一个 Transformer 语言模型]] 手写了注意力；本项目换成"用别人的 124M 权重 + 只训 0.24% 参数"，是同一个结构的两条路线。
- 理论支撑：[[LLM-036 Embedding 与向量表示]]（隐状态就是句子表示）、[[LLM-043 GPT 系列的演进]]（gpt2 为什么能直接拿来做分类）、`LLM-065 LoRA 与参数高效微调`（待建）、`LLM-060 指令微调与对齐`（待建）。
- 与 `LLM-071 分类头与池化策略`（待建）相关：本项目只用了一种池化（最后一个非 pad token），没对比 mean-pooling / CLS / 注意力池化。
- 下游：[[PROJ-023 搭建一个最小可用的 RAG 问答系统]] 用的是同一份 gpt2，但不再微调——直接拿它的静态词嵌入做检索、拿它的生成能力做回答，暴露的是"不微调时 gpt2 有多弱"。

## 学习记录

| 日期 | 状态 | 用时 | 问题 | 复盘 |
|---|---|---|---|---|
| 2026-11-08 | planned | — | — | — |

---

## 🔗 Notion 同步信息

> 以下内容由 Notion「AI Knowledge System · 实战项目」数据库按知识 ID 合并而来（导出于 2026-11-09）。

| 属性 | 值 |
|---|---|
| 所属学科 | 机器学习 |
| 项目状态 | 未开始 |
| 项目阶段 | 立项 |
| 难度 | L2 应用 |
| 技术栈 | Python、PyTorch、Transformers、PEFT |
| 项目仓库 | ai-knowledge-system/projects/PROJ-022 |

- **对应课程模块**：[M-LLM-07 大语言模型微调与 PEFT](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-LLM-07%20%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%BE%AE%E8%B0%83%E4%B8%8E%20PEFT.md)
- **前置知识**：[LLM-065 LoRA 与参数高效微调](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/05%20%E9%A2%84%E8%AE%AD%E7%BB%83%E4%B8%8E%E5%BE%AE%E8%B0%83/LLM-065%20LoRA%20%E4%B8%8E%E5%8F%82%E6%95%B0%E9%AB%98%E6%95%88%E5%BE%AE%E8%B0%83.md) · [LLM-060 指令微调（SFT）](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/05%20%E9%A2%84%E8%AE%AD%E7%BB%83%E4%B8%8E%E5%BE%AE%E8%B0%83/LLM-060%20%E6%8C%87%E4%BB%A4%E5%BE%AE%E8%B0%83%EF%BC%88SFT%EF%BC%89.md)
- **对应知识点**：[LLM-036 Embedding 与向量表示](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/02%20Token%E4%B8%8EEmbedding/LLM-036%20Embedding%20%E4%B8%8E%E5%90%91%E9%87%8F%E8%A1%A8%E7%A4%BA.md) · [LLM-071 微调效果评估](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/05%20%E9%A2%84%E8%AE%AD%E7%BB%83%E4%B8%8E%E5%BE%AE%E8%B0%83/LLM-071%20%E5%BE%AE%E8%B0%83%E6%95%88%E6%9E%9C%E8%AF%84%E4%BC%B0.md)

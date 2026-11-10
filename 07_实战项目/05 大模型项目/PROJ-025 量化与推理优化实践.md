---
id: PROJ-025
title: 量化与推理优化实践
domain: PROJ
module: [M-LLM-09]
type: project
difficulty: L3 推导
status: planned
tech_stack: [Python, PyTorch, transformers]
prerequisites: [LLM-094, LLM-090]
related_math: []
related: [LLM-020, LLM-096, PROJ-026]
notion_page_id: 412663e8-0333-81cf-8ba0-c8ef6ff985f2
notion_url: https://app.notion.com/p/412663e8033381cf8ba0c8ef6ff985f2
created: 2026-11-08
updated: 2026-11-08
notion_db: 实战项目
所属学科: 机器学习
项目状态: 未开始
项目阶段: 立项
难度: L3 推导
项目目标: 量化与推理优化实践
技术栈: [Python, PyTorch, Transformers]
项目仓库: ai-knowledge-system/projects/PROJ-025
对应课程模块: [M-LLM-09]
前置知识: [LLM-094, LLM-090]
对应知识点: [LLM-020, LLM-096]
tags: [机器学习]
---

# PROJ-025 量化与推理优化实践

## 1. 项目背景

一个 124M 的 gpt2 用 fp32 存权重要 324 MB，7B 模型要 28 GB。**量化**就是把权重从 4 字节压到
1 字节（INT8）或半字节（INT4），换来显存和带宽的节省——代价是数值误差。

另一条优化线是 **KV Cache**：自回归解码时，第 t 步的注意力要看到前 t-1 个 token 的 K/V，
不缓存就要每步把整段重算一遍。缓存把"每步 O(t)"变成"每步 O(1)"，代价是显存随长度线性增长。

这两个东西在教科书上都是"显然更快/更省"，但**真的量一遍**才会发现：
本机上 KV Cache 的端到端收益接近 0（0.99×），因为瓶颈根本不在算力，在每步的启动开销。

本项目给出四组真实数字：INT8/INT4 的权重重建误差与 logits 相似度、KV Cache 的 MB（公式 vs 实测）、
use_cache 的速度对比，以及一个"为什么没有加速"的归因实验。

相关理论见 [[LLM-020 Attention 的计算复杂度与局限]]（O(n²) 从哪来）、
[[LLM-030 Transformer 参数量与计算量估算]]（算 KV 显存的公式）。

## 2. 项目目标

| 目标 | 验收方式 | 实测 |
|---|---|---|
| INT8 对称量化（per-tensor / per-channel） | 权重 MSE、最大相对误差 | 1.90e-04 / 2.16e-06 |
| INT4 量化 | 同上 + 误差放大倍数 | 1.01e-02 / 5.82e-04，放大 53× / 270× |
| 输出 logits 余弦相似度 | 与 fp32 基线对比 | 0.9991 / 0.99999 / 0.9688 / 0.9970 |
| top-1 一致率 | 比余弦更接近"功能是否等价" | INT4 per-tensor 只剩 7.7% |
| KV Cache 元素数与显存 | 公式 vs `past_key_values` 实测 vs `memory_allocated` | 72.0 KB/token，三者一致 |
| use_cache 速度对比 | warmup + `synchronize` + 交替 A/B | 0.99×（几乎无收益），归因见第 8 节 |

## 3. 技术栈与环境

| 项 | 值 |
|---|---|
| Python | 3.13（`py -3.13`） |
| PyTorch | 2.15.0+cu132 |
| transformers | 4.57.6（`GPT2LMHeadModel`，注意力实现 sdpa） |
| GPU | RTX 4070 Ti，12.88 GB |
| 模型 | `gpt2`，124.4M 参数，49 个线性层（48 个 `Conv1D` + 1 个 `lm_head`） |
| 基线精度 | fp32（本项目不涉及 fp16/bf16 对照） |
| 随机性 | seed=42 |

**一个必须知道的实现细节**：gpt2 的注意力与 FFN 用的是 transformers 的 `Conv1D`，
权重形状是 `(in, out)`，跟 `nn.Linear` 的 `(out, in)` **转置**。
第一版只匹配 `nn.Linear`，结果只找到 `lm_head` 一层，量化 0 层、直接除零崩掉。
所以脚本里统一用 `get_w()/set_w()` 转成 `(out, in)` 视角，per-channel 的轴才对齐。

## 4. 数据集

量化不需要训练数据，但需要**校准/评估输入**（脚本内写死两段英文）：

```text
calib_text (48 token)：客服退款场景的一段话，用于测 logits 余弦（第二个口径）
eval_texts (2 条 × 26 token)：量化技术说明 + KV cache 说明
```

- 权重统计覆盖 **48 个线性层、124.4M 参数中的 324 MB**（`lm_head` 不量化）；
- logits 对比口径：`(2, 26, 50257)`，`ref_absmean=109.6993`；
- **没有真实下游任务**（如 GLUE / 生成质量），所以第 7 节只有"数值保真度"，没有"任务精度"。
  这是本项目最大的局限，第 9 节会再强调一次。

## 5. 方案设计

```text
① 量化（weight-only，模拟）
   w ──absmax──► scale ──round/clamp──► q(int8) ──dequant──► w'  (fp32)
   per-tensor : scale = max|w| / 127                  （整层一个）
   per-channel: scale_i = max|w_i| / 127  (dim=0 输出通道) （每行一个）
   INT4 把 127 换成 7

② 误差度量
   MSE / 相对 MSE = Σ(w-w')² / Σw²
   max_rel = max|w-w'| / max|w|            （逐层算再取最大，避免小权重放大）
   logits 余弦相似度 + top-1 一致率 + 最大绝对差

③ KV Cache
   公式：elems = 2 × n_layer × n_head × head_dim × L × batch
   实测：sum(t.numel() for layer in past_key_values for t in layer)
   显存：cache_bytes（精确）+ memory_allocated 增量（含 logits 与碎片）

④ 速度
   generate 32 token，use_cache=True/False，交替 A/B × 6 轮，取 median
   手工解码循环（隔离 generate 调度开销）
   单次前向 1 / 48 / 80 token（判断瓶颈是算力还是启动）
```

**为什么用"反量化回 fp32"来模拟**：PyTorch 没有 int4 张量，真正的 INT4 推理要靠
`bitsandbytes` / `GPTQ` / `llama.cpp` 的内核。本项目只验证"量化误差"这一半，
所以 `w' = q * scale` 存回 fp32——**真实显存并不会变**。第 7 节的 `packed_MB` 是理论打包值，
标题里写清楚，不冒充实测。

## 6. 核心代码

```python
def quantize_per_tensor(w, bits):
    qmax = 2 ** (bits - 1) - 1                      # INT8:127, INT4:7
    amax = w.abs().max().clamp(min=1e-12)
    scale = amax / qmax
    q = torch.clamp(torch.round(w / scale), -qmax, qmax)
    return q.to(torch.int8), scale, q

def quantize_per_channel(w, bits):
    qmax = 2 ** (bits - 1) - 1
    amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)   # 每个输出通道一个 scale
    scale = amax / qmax
    q = torch.clamp(torch.round(w / scale), -qmax, qmax)
    return q.to(torch.int8), scale, q

def apply_quant(model, bits, mode, skip_lm_head=True):
    backups, stats = {}, {...}
    fn = quantize_per_tensor if mode == "per_tensor" else quantize_per_channel
    for name, m in linear_layers(model):                     # Conv1D 与 Linear 都覆盖
        if skip_lm_head and name == "lm_head":
            continue
        w = get_w(m)                                          # 统一成 (out, in)
        backups[name] = w.clone()
        q, scale, _ = fn(w, bits)
        set_w(m, dequantize(q, scale))                        # 反量化写回 fp32
        err = get_w(m) - w
        stats["sum_sq_err"] += err.pow(2).sum().item()
        stats["sum_sq_w"]   += w.pow(2).sum().item()
        stats["max_rel"] = max(stats["max_rel"],
                               (err.abs().max() / w.abs().max().clamp(min=1e-12)).item())
        stats["packed_bytes"] += w.numel() * bits / 8 + scale.numel() * 4
    return backups, stats

def restore(model, backups):
    for name, m in linear_layers(model):
        if name in backups:
            set_w(m, backups[name])
```

KV Cache 实测（`past_key_values` 是 `DynamicCache`，可迭代出每层的 `(k, v)`）：

```python
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
cfg = model.config
n_layer, n_head, d = cfg.n_layer, cfg.n_head, cfg.n_embd
head_dim = d // n_head
batch = 1

for L in (16, 64, 256):
    ids = torch.randint(0, 1000, (batch, L))
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
    elems = sum(t.numel() for layer in out.past_key_values for t in layer)
    mib = sum(t.numel() * t.element_size()
              for layer in out.past_key_values for t in layer) / 1024 ** 2
    formula = 2 * n_layer * n_head * head_dim * L * batch
    print(f"L={L:>3}: 实测元素 {elems:>9,} | 公式 {formula:>9,} | 一致 {elems == formula}"
          f" | {mib:.3f} MiB（fp32）")
```

## 7. 运行结果

真实终端输出（`py -3.13 proj025_quantization_bench.py`）：

```text
device=cuda torch=2.15.0+cu132
model=gpt2 params=124.4M linear_layers=49
eval logits shape=(2, 26, 50257) ref_absmean=109.6993
[int8_per_tensor ] w_mse=1.897e-04 rel_mse=1.111e-02 max_rel=0.0039 cos=0.999104 agree=0.731 324.0MB->81.0MB
[int8_per_channel] w_mse=2.156e-06 rel_mse=1.263e-04 max_rel=0.0039 cos=0.999993 agree=0.962 324.0MB->81.3MB
[int4_per_tensor ] w_mse=1.006e-02 rel_mse=5.890e-01 max_rel=0.0714 cos=0.968819 agree=0.077 324.0MB->40.5MB
[int4_per_channel] w_mse=5.821e-04 rel_mse=3.410e-02 max_rel=0.0714 cos=0.996983 agree=0.615 324.0MB->40.8MB
[amp] per_tensor : mse x53.01,  cos drop int8->int4 0.030285
[amp] per_channel: mse x270.07, cos drop int8->int4 0.003010

[KV cache]
  L=  16 cache=  294912 elems (  1.125 MB) formula=  294912 (  1.125 MB) alloc_delta=  4.193 MB (logits  3.067) per_tok=None KB
  L=  32 cache=  589824 elems (  2.250 MB) formula=  589824 (  2.250 MB) alloc_delta=  9.000 MB (logits  6.135) per_tok=72.0 KB
  L=  64 cache= 1179648 elems (  4.500 MB) formula= 1179648 (  4.500 MB) alloc_delta= 17.750 MB (logits 12.270) per_tok=72.0 KB
  L= 128 cache= 2359296 elems (  9.000 MB) formula= 2359296 (  9.000 MB) alloc_delta= 33.540 MB (logits 24.540) per_tok=72.0 KB
  L= 256 cache= 4718592 elems ( 18.000 MB) formula= 4718592 ( 18.000 MB) alloc_delta= 68.000 MB (logits 49.079) per_tok=72.0 KB

[speed] generate 32 tokens, use_cache True vs False (interleaved A/B)
  generate use_cache_True : median=321.9 ms mean=319.3 ms std=26.5 ms (10.06 ms/token)
  generate use_cache_False: median=317.5 ms mean=318.5 ms std=14.5 ms ( 9.92 ms/token)

[speed] manual decode loop (isolates cache effect)
  cached        : median=251.8 ms ( 7.87 ms/token)
  full_recompute: median=403.0 ms (12.59 ms/token)
  forward_1tok : median=8.90 ms
  forward_48tok: median=10.35 ms
  forward_one_shot_80tok: median=9.31 ms
  token-forward passes: cached=79 full=2032 ratio=25.7x
  gpu_state_at_end: 75 %, 2057 MiB

================ SUMMARY ================
int8_per_tensor  mse=1.897e-04 cos=0.999104 top1=0.731 packed=81.0MB comp=4.00x
int8_per_channel mse=2.156e-06 cos=0.999993 top1=0.962 packed=81.3MB comp=3.98x
int4_per_tensor  mse=1.006e-02 cos=0.968819 top1=0.077 packed=40.5MB comp=8.00x
int4_per_channel mse=5.821e-04 cos=0.996983 top1=0.615 packed=40.8MB comp=7.94x
kv per-token measured=[72.0, 72.0, 72.0, 72.0] KB formula=72.0 KB
speed cache=321.9ms no_cache=317.5ms speedup=0.99x | manual loop 252ms vs 403ms
peak_gpu=906.3 MB | metrics -> proj025_metrics.json
```

误差放大倍数（同口径 INT4/INT8）：

| 模式 | INT8 MSE | INT4 MSE | 放大 | INT8 cos | INT4 cos | top-1 一致率 |
|---|---|---|---|---|---|---|
| per-tensor | 1.897e-04 | 1.006e-02 | **53.0×** | 0.999104 | 0.968819 | 0.731 → **0.077** |
| per-channel | 2.156e-06 | 5.821e-04 | **270.1×** | 0.999993 | 0.996983 | 0.962 → 0.615 |

## 8. 结果分析

**① 误差从哪来：scale 是"整层一个"还是"每行一个"。**
对称量化的最大舍入误差是 `scale/2 = max|w| / (2·qmax)`。
per-tensor 用整层 absmax 当 scale，一个异常大的权重会把整层（甚至整模型的）步长拉大，
于是大部分正常权重被压进同一个桶里。per-channel 每行单独算 absmax，
MSE 从 1.90e-04 掉到 2.16e-06（**88 倍**），代价只是多存 48×两个向量（81.0 → 81.3 MB）。

**② "余弦相似度"会骗人。**
INT4 per-tensor 的 logits 余弦还有 0.9688，看起来"基本一样"；
但 top-1 一致率只剩 **7.7%**——几乎每步都选错下一个 token。
余弦是全局相关，被少数大 logit 主导；采样只关心"第一名是不是同一个"。
**工程上应盯 top-1 一致率 / 困惑度，而不是余弦。** 这条是本项目最重要的教训。

**③ INT4 的误差放大不是线性的，而是"per-channel 更痛"。**
per-tensor 放大 53×（接近理论 4²=16 的量级再叠加舍入），per-channel 放大 270×。
原因：per-channel 的 INT8 已经把误差压到 2e-06，分母太小，
换成 4 bit 后每行步长是 INT8 的 18 倍，误差跳变更剧烈。
绝对值上看，INT4 per-channel（5.82e-04）仍比 INT8 per-tensor（1.90e-04）大 3 倍——
**"4 bit + per-channel" 不等于 "8 bit + per-tensor" 的精度**，选型要按任务敏感度来。

**④ KV Cache 公式与实测完全对上。**
`2 × 12 层 × 12 头 × 64 维 × L × batch`：每 token 增量 18,432 元素 × 4 字节 = **72.0 KB**，
与 `past_key_values` 实测增长率**逐点相等**（72.0 KB/token，L=32→256 五次采样全中）。
`memory_allocated` 的增量比 cache 大：L=256 时 68.0 MB 里有 49.1 MB 是 logits（`256×50257×4`），
剩下 18.0 MB 才是 cache。**做显存预算时，logits 这一项经常被忘掉**——
对 50k 词表的模型，它是 O(L × vocab) 的大头。

**⑤ 速度：KV Cache 在本机几乎不省时间（0.99×），但这不是"cache 没用"。**
- 手工解码循环里 cache 是有效的：251.8 ms vs 403.0 ms = **1.60×**；
- 但 `generate` 层面被调度开销吃掉了：321.9 ms vs 317.5 ms = **0.99×**。
- 归因实验：单次前向 **1 token 8.90 ms**、**48 token 10.35 ms**、**80 token 一次性 9.31 ms**。
  48 个 token 只比 1 个 token 贵 16%——说明每次前向的墙钟时间几乎与 token 数无关，
  被**内核启动/调度**（Windows + WDDM + fp32 + 小模型，每层几十个小 kernel）主导。
- 计算量对比：cached 只推进 79 个 token-pass，full 要 2032 个（**25.7×**），
  但墙钟只差 1.6× → **算力占比不到 5%**。
- 结论：**KV Cache 的价值要在"每步算力足够大"（大模型 / 长上下文 / batch>1）时才显形**。
  124M 模型 + 单请求 + fp32，瓶颈是启动开销，缓存省下的 FLOPs 换不成时间。
  真正要提速应该先做 kernel fusion / CUDA Graph / 提高 batch，量化反而其次。

**⑥ 量化在这里没有测速。** `w' = q·scale` 存回 fp32 后，前向仍是 fp32 矩阵乘，
速度不会变。要测"INT8 真的快多少"必须用 `bitsandbytes` / `torch.ao.quantization` 的整数内核，
本机没有装、也不允许下载新依赖。**本项目只测了"量化误差"，没测"量化加速"**，别混。

## 9. 数据来源与差异说明（必读）

- **模型是 gpt2（124M），不是 7B/70B。** 结论里的"KV Cache 收益 1.6×/0.99×"强依赖模型大小与
  硬件；大模型上 cache 的收益通常是数倍。本项目的数字**只能**用来说明"如何量"和"瓶颈在哪"。
- **量化是模拟的，不是真内核。** `packed_MB`（81.0 / 40.5 MB）是按 `numel × bits/8 + scale`
  算的理论打包值；真实显存因为反量化回 fp32 **没有变化**（峰值 906.3 MB）。
  真正的 INT8/INT4 推理要用整数 GEMM，误差与速度都会与本项目不同（尤其激活量化）。
- **只做 weight-only，没有激活量化。** 工业界的 INT8（如 SmoothQuant）会同时量化激活，
  误差更大、加速更明显；本项目只碰权重这一半。
- **评估只有 logits 相似度，没有下游任务。** 没有跑困惑度、没有跑生成质量、没有跑 GLUE。
  "top-1 一致率 7.7%"是**功能退化的强烈信号**，但它不等于"BLEU 掉多少"。
- **GPU 是共享的。** 测速期间同机还有别的进程（`nvidia-smi` 末尾采样到 75% 利用率、
  2057 MiB 占用）。脚本因此改成**交替 A/B + median + 记录 std**（std 14.5-26.5 ms），
  绝对值会有漂移，**比值**才可引用。早期在 GPU 打满（100%、10.7 GB）时测到
  `forward_1tok=101 ms` 这种离谱数字，已被丢弃重测。
- **`lm_head` 不量化。** 它占 38.6M 参数（50257×768），量化它会显著影响输出分布；
  真机上 lm_head 常保持 fp16。本项目刻意跳过并在代码里显式标注。

## 10. 复现步骤

```powershell
$env:PYTHONIOENCODING='utf-8'
cd <USER_HOME>\Documents\deepseek-harness\default-workspace\09-llm\projects
py -3.13 proj025_quantization_bench.py
```

产物：`proj025_metrics.json`（四组量化指标、KV 五行原始数据、速度原始 6 次采样）。

要点：

1. 必须 `py -3.13`（有 torch）；默认 `python` 是 3.15，import 就失败；
2. 只用缓存里的 `gpt2`；`HF_HUB_OFFLINE=1` 下不会触发下载；
3. 跑之前看一眼 `nvidia-smi`：显存 > 8 GB 被占时，速度那一节的绝对值不可信，建议重跑；
4. 峰值显存约 906 MB，不吃满卡；**不要在 GPU 上做越界/断言实验**，会污染 CUDA 上下文，
   本项目的误差注入全部走"数值"路径（clamp + round），不触碰非法访存。

## 11. 完整训练脚本

本项目**不训练**，交付物是基准脚本 `proj025_quantization_bench.py`（约 330 行）：

```text
proj025_quantization_bench.py
├─ quantize_per_tensor / quantize_per_channel / dequantize
├─ linear_layers / get_w / set_w      Conv1D 与 Linear 的 (in,out)/(out,in) 统一
├─ apply_quant / restore              就地改权重，用完恢复
├─ logits_of / cos_sim
├─ main()
│   ├─ 基线 logits（eval 2×26 + calib 1×48）
│   ├─ 四种量化：int8/int4 × per-tensor/per-channel → MSE/cos/top-1/打包体积
│   ├─ KV Cache：L ∈ {16,32,64,128,256}，元素数/公式/alloc 增量/每 token 斜率
│   ├─ 速度：generate 交替 A/B ×6、手工循环 cached vs full、单次前向 1/48/80 token
│   └─ 写 proj025_metrics.json
```

## 12. 验收清单

| 验收项 | 是否达成 | 证据 |
|---|---|---|
| INT8 对称量化 per-tensor | ✅ | `w_mse=1.897e-04`，`packed 81.0MB` |
| INT8 对称量化 per-channel | ✅ | `w_mse=2.156e-06`，比 per-tensor 小 88× |
| 权重 MSE 与最大相对误差 | ✅ | `max_rel=0.0039`（两种模式同值） |
| logits 余弦相似度 | ✅ | 0.999104 / 0.999993 |
| INT4 量化 + 误差放大倍数 | ✅ | 1.006e-02 / 5.821e-04，放大 53.0× / 270.1× |
| KV Cache 元素数（公式 vs 实测） | ✅ | 五行全等，L=256 时 4,718,592 |
| KV Cache MB（公式 vs 实测） | ✅ | 72.0 KB/token 逐点相等；alloc 增量扣除 logits 后吻合 |
| 速度对比（warmup + synchronize） | ✅ | 0.99× generate、1.60× 手工循环 |
| 打印清晰指标 | ✅ | 逐项 4 行 + KV 5 行 + SUMMARY |
| 写出 `proj025_metrics.json` | ✅ | 同目录，含原始采样 |
| 固定种子 | ✅ | `SEED=42` |

## 13. 结论与局限

**结论**

1. **per-channel 是几乎免费的精度提升**：显存只多一点（+0.3 MB），MSE 降 88 倍。
   weight-only INT8 应该默认 per-channel。
2. **别用余弦相似度判断量化是否可用**：INT4 per-tensor 余弦 0.9688 但 top-1 一致率 7.7%。
   要测就测 top-1 / 困惑度 / 任务指标。
3. **KV Cache 的显存公式完全可预测**（72.0 KB/token，两次口径一致），
   但它的**速度收益高度依赖"每步算力是否占主导"**。本机 124M + fp32 下只有 1.6×（generate 0.99×），
   因为一次前向的固定开销 8.9 ms 已经盖过 48 token 的增量。
4. 做显存预算时别忘 **logits**：L=256 时它占 68 MB 里的 49 MB。

**局限**

- 量化是 fp32 模拟，没有真实整数内核，**没有加速比可谈**；
- 没做激活量化、没做 KV Cache 量化（INT8 KV 能把 72 KB/token 砍到 18 KB）；
- 没有下游任务指标，无法回答"这个误差业务上能不能接受"；
- 单条请求、batch=1，没有测吞吐（tokens/s）随 batch 的变化；
- 共享 GPU 上测速，绝对值有漂移。

## 14. 与后续项目的关系

- **PROJ-026 端到端 LLM 应用：客服问答**：那边每一问都要跑一次生成，
  本项目的"单步 8.9 ms 固定开销"正是它延迟的主要来源，
  也是它最终选择缓存 + 降级的量化依据。
- `LLM-094 量化`（待建）：INT8/INT4、GPTQ/AWQ、per-channel 的理论篇。
- `LLM-096 KV Cache 与推理加速`（待建）：DynamicCache / PagedAttention 的工程篇。
- [[LLM-020 Attention 的计算复杂度与局限]]：为什么解码是 O(n²) 累计、缓存能救多少。
- [[LLM-030 Transformer 参数量与计算量估算]]：324 MB / 72 KB 这些数字的推导来源。

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
| 难度 | L3 推导 |
| 技术栈 | Python、PyTorch、Transformers |
| 项目仓库 | ai-knowledge-system/projects/PROJ-025 |

- **对应课程模块**：[M-LLM-09 大语言模型推理、部署与优化](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-LLM-09%20%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86%E3%80%81%E9%83%A8%E7%BD%B2%E4%B8%8E%E4%BC%98%E5%8C%96.md)
- **前置知识**：[LLM-094 量化：INT8、INT4 与 GPTQ](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/06%20%E6%8E%A8%E7%90%86%E4%B8%8E%E9%83%A8%E7%BD%B2/LLM-094%20%E9%87%8F%E5%8C%96%EF%BC%9AINT8%E3%80%81INT4%20%E4%B8%8E%20GPTQ.md) · [LLM-090 KV Cache 的原理与显存](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/06%20%E6%8E%A8%E7%90%86%E4%B8%8E%E9%83%A8%E7%BD%B2/LLM-090%20KV%20Cache%20%E7%9A%84%E5%8E%9F%E7%90%86%E4%B8%8E%E6%98%BE%E5%AD%98.md)
- **对应知识点**：[LLM-020 Attention 的计算复杂度与局限](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/03%20Attention%E6%9C%BA%E5%88%B6/LLM-020%20Attention%20%E7%9A%84%E8%AE%A1%E7%AE%97%E5%A4%8D%E6%9D%82%E5%BA%A6%E4%B8%8E%E5%B1%80%E9%99%90.md) · [LLM-096 推理延迟与吞吐的权衡](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/06%20%E6%8E%A8%E7%90%86%E4%B8%8E%E9%83%A8%E7%BD%B2/LLM-096%20%E6%8E%A8%E7%90%86%E5%BB%B6%E8%BF%9F%E4%B8%8E%E5%90%9E%E5%90%90%E7%9A%84%E6%9D%83%E8%A1%A1.md)

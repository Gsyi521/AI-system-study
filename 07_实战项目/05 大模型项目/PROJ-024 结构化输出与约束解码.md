---
id: PROJ-024
title: 结构化输出与约束解码
domain: PROJ
module: [M-LLM-10]
type: project
difficulty: L3 推导
status: planned
tech_stack: [Python, PyTorch, transformers]
prerequisites: [LLM-101, LLM-092]
related_math: []
related: [LLM-034, LLM-099, PROJ-025]
notion_page_id: 412663e8-0333-8155-9b29-feabede2cbca
notion_url: https://app.notion.com/p/412663e8033381559b29feabede2cbca
created: 2026-11-08
updated: 2026-11-08
notion_db: 实战项目
所属学科: 机器学习
项目状态: 未开始
项目阶段: 立项
难度: L3 推导
项目目标: 结构化输出与约束解码
技术栈: [Python, PyTorch, Transformers]
项目仓库: ai-knowledge-system/projects/PROJ-024
对应课程模块: [M-LLM-10]
前置知识: [LLM-101, LLM-092]
对应知识点: [LLM-034, LLM-099]
tags: [机器学习]
---

# PROJ-024 结构化输出与约束解码

## 1. 项目背景

让模型"说人话"很容易，让模型"说程序能读的话"很难。下游要的是 `{"intent": "refund", ...}`，
而语言模型输出的是一串条件概率采样出来的字符流：可能少一个引号、可能把 `intent` 写成 `id`、
可能干脆开始讲故事。

工业界解决这件事有三层手段，成本依次上升：

1. **生成后解析 + 校验**（parse & validate）：最便宜，失败就丢弃；
2. **失败重试**（retry with repair）：多花几次调用，能救"偶然手滑"，救不了"根本不会"；
3. **约束解码**（constrained / grammar decoding）：在每一步把非法 token 的概率置零，
   从生成过程上保证格式合法。

本项目用 **gpt2（124M，未做任何指令微调）** 当实验体。它是"最坏情况"的模型：
不会遵循格式、不会输出 JSON、还会重复 prompt 里的片段。正因为差，三种手段的差距才看得清楚：

> 自由生成 schema 合法率 **0%** → 重试 3 次仍然 **0%** → 约束解码 **100%**（intent 正确率 40%）。

相关概念见 [[LLM-034 特殊 Token 与 Attention Mask]]（token 边界与解码控制）、
[[LLM-031 Tokenizer 基础与子词切分]]（"合法前缀"为什么要在 token 层判断）。

## 2. 项目目标

| 目标 | 验收方式 | 实测 |
|---|---|---|
| 定义并实现 JSON schema | 自写 `validate()`，不依赖 jsonschema | 6 类错误可检出 |
| 生成后解析 + 校验 | 10 条用例逐条打印 | 首轮 parse 20%、schema 0% |
| 失败重试 | 最多 3 次，记录尝试次数分布 | 分布 `{3: 10}`，0% 成功 |
| 简化约束解码 | 自写 `prefix_allowed_tokens_fn` | schema 100%、intent 40% |
| 量化对比 | 两种模式的合法率同口径统计 | 0% vs 100% |

## 3. 技术栈与环境

| 项 | 值 |
|---|---|
| Python | 3.13（`py -3.13`，默认 `python` 是 3.15 且无 torch） |
| PyTorch | 2.15.0+cu132 |
| transformers | 4.57.6 |
| 设备 | CUDA，RTX 4070 Ti 12.88 GB |
| 模型 | `gpt2`（124.4M 参数，vocab 50257），**未微调** |
| 随机性 | seed=42（`random` / `numpy` / `torch` 三处都固定） |
| 网络 | `HF_HUB_OFFLINE=1` + `HF_ENDPOINT=https://hf-mirror.com`，只用本地缓存 |

约束解码是纯 CPU 逻辑（自动机 + token 文本匹配），只有前向在 GPU 上。

## 4. 数据集

没有公开数据集可对齐"客服意图抽取 + 槽位"这个 schema，所以 **10 条用例在脚本里写死**（合成数据）：

```text
id  文本                                                          期望 intent
00  I want a refund for order 88213, the laptop arrived broken.   refund
01  Where is my order 40127? I ordered headphones last week.      query_order
02  My monitor 77310 stopped working after two days, ...          refund
03  Can you tell me the warranty policy for the phone?            faq
04  Order 12904 is late again, this is the third delay, ...       complaint
05  Please cancel order 88213 and refund me.                      refund
06  What is the shipping time to Berlin for a laptop?             faq
07  Do you have the monitor in stock? My order 40127 never ...    query_order
08  The headphones I got are the wrong color, I want to return... complaint
09  How do I reset the password of my account?                    faq
```

目标 schema（三个必填字段，不允许多余字段）：

```json
{"intent": "<query_order|refund|faq|complaint>",
 "slots": {"order_id": "<4-8 位数字>", "product": "<非空字符串>"},
 "confidence": <0.0-1.0 的浮点>}
```

**数据局限**：10 条样本、4 个意图、4 个订单号，只够验证"链路是否成立"，不足以训练或评估抽取精度。
`product` 未做枚举约束（自由生成时可以随便写），只有 `intent` 参与语义正确率统计。

## 5. 方案设计

```text
                 ┌─────────────────────────────────────────┐
prompt ─────────►│ ① 自由生成 (greedy, 64 tok)             │──► 解析 ──► 校验 ──► 统计
                 └─────────────────────────────────────────┘
                              │ 失败
                              ▼
                 ┌─────────────────────────────────────────┐
                 │ ② 重试 (do_sample, T=0.9, 最多 3 次)     │──► 有成功即停
                 └─────────────────────────────────────────┘
                              │
                              ▼
                 ┌─────────────────────────────────────────┐
                 │ ③ 约束解码：每步只保留合法前缀 token     │──► 结构必合法
                 │    prefix_allowed_tokens_fn(batch, ids)  │
                 └─────────────────────────────────────────┘
```

**关键设计一：校验分四级**，而不是一个布尔值。
`parsed`（能截出 `{...}` 且 `json.loads` 成功）→ `schema_ok`（类型/枚举/范围/多余字段）→
`intent_ok`（语义是否等于期望）→ `order_ok`（槽位是否抽对）。
只报一个"合法率"会掩盖"格式对了但内容全错"，本项目实测正是这种情况。

**关键设计二：约束解码不碰模型权重**。
在 logits 上加 mask（非法 token 置 `-inf`），用 `LogitsProcessor` 包住
`prefix_allowed_tokens_fn`。模型仍然是原来的 gpt2，只是"说不出口的字"被掐掉。

**关键设计三：语法用"字面量段 + 枚举段"拼接**，而不是完整 JSON-Schema 编译器。
值域被收窄成有限集合（4 意图 / 4 订单号 / 4 商品 / 5 个置信度），这样自动机可以在
30 行内写完；真正的生产方案（outlines / xgrammar）是 schema → 正则/CFG → token 前缀表，
思想相同但工程量大两个数量级。

## 6. 核心代码

校验函数（自写，不依赖第三方 schema 库）：

```python
def validate(obj):
    errs = []
    if not isinstance(obj, dict):
        return False, ["not_a_dict"]
    keys = set(obj)
    required = {"intent", "slots", "confidence"}
    if required - keys:
        errs.append(f"missing:{sorted(required - keys)}")
    if keys - required:
        errs.append(f"extra:{sorted(keys - required)}")
    intent = obj.get("intent")
    if not isinstance(intent, str):
        errs.append("intent_not_str")
    elif intent not in INTENTS:
        errs.append("intent_not_in_enum")
    slots = obj.get("slots")
    if not isinstance(slots, dict):
        errs.append("slots_not_dict")
    else:
        oid = slots.get("order_id")
        if not isinstance(oid, str) or not re.match(r"^\d{4,8}$", oid or ""):
            errs.append("order_id_bad")
        prod = slots.get("product")
        if not isinstance(prod, str) or not prod.strip():
            errs.append("product_bad")
    conf = obj.get("confidence")
    if isinstance(conf, bool) or not isinstance(conf, (int, float)):
        errs.append("confidence_not_number")
    elif not (0.0 <= float(conf) <= 1.0):
        errs.append("confidence_out_of_range")
    return len(errs) == 0, errs
```

约束解码的核心：自动机 + 前缀判定 + token 过滤。

```python
# 与脚本中同名常量一致（脚本内是完整取值集，此处给最小可用子集）
INTENTS = ["退货", "查询订单", "退换货政策"]
ORDER_IDS = ["A1001", "B2002"]
PRODUCTS = ["耳机", "键盘"]
CONFS = ["0.9", "0.8"]

GRAMMAR = ["{", '"intent": "', INTENTS, '", "slots": {"order_id": "', ORDER_IDS,
           '", "product": "', PRODUCTS, '"}, "confidence": ', CONFS, "}"]

def allowed_next(state):
    """state = 已生成前缀；返回 (允许的下一字符集合, 是否已完整)"""
    i, o = 0, 0
    while i < len(GRAMMAR):
        opts = [GRAMMAR[i]] if isinstance(GRAMMAR[i], str) else list(GRAMMAR[i])
        rem = state[o:]
        fits = [opt for opt in opts if opt.startswith(rem)]
        if fits:
            longer = [opt for opt in fits if len(opt) > len(rem)]
            if longer:
                return {opt[len(rem)] for opt in longer}, False
            o += len(rem); i += 1; continue          # 本段恰好吃完
        consumed = [opt for opt in opts if rem.startswith(opt)]
        if not consumed:
            raise ValueError(f"invalid prefix: {state!r}")
        o += len(consumed[0]); i += 1
    return set(), True

def can_consume(state, text):
    s = state
    for ch in text:
        allowed, complete = allowed_next(s)
        if complete or ch not in allowed:
            return False
        s += ch
    return True

class ConstrainedDecoder:
    def prefix_allowed_tokens_fn(self, batch_id, input_ids):
        state = self.tok.decode(input_ids[self.prompt_len:].tolist())
        if state in self._cache:
            return self._cache[state]
        allowed_chars, complete = allowed_next(state)
        out = [self.eos_id] if complete else []
        for ch in allowed_chars:                       # 只测首字符合法的 token
            for tid in self.by_char.get(ch, ()):
                if can_consume(state, self.texts[tid]):
                    out.append(tid)
        self._cache[state] = out or [self.eos_id]
        return self._cache[state]
```

接入 `generate`（非法 token 的 logit 置 `-inf`，等价于"只从合法集合里采样"）：

```python
import torch
from transformers import LogitsProcessor


class AllowedTokensProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        for b in range(input_ids.shape[0]):
            ids = self.fn(b, input_ids[b])
            mask = torch.full((self.vocab_size,), float("-inf"),
                              device=scores.device, dtype=scores.dtype)
            mask[torch.as_tensor(ids, device=scores.device, dtype=torch.long)] = 0.0
            scores[b] = scores[b] + mask
        return scores
```

## 7. 运行结果

真实终端输出（`py -3.13 proj024_structured_output.py`，seed=42）：

```text
device=cuda torch=2.15.0+cu132
model=gpt2 params=124.4M vocab=50257

[1] free generation + parse + schema validate (greedy, first pass)
  case00 schema_ok=False intent_ok=False errs=['no_json_found'] raw=' { "id": "88213", "name": "I want a refund for order 88213",'
    retried -> attempts=3 schema_ok=False raw='\nTo find a ticket, use the schema: ticket object.\nSchema: re'
  case02 schema_ok=False errs=["missing:['confidence', 'intent', 'slots']", "extra:['description', 'id', 'name']"] raw=' { "id": "77310", "name": "Monitor 77310", "description": "M'
  case06 schema_ok=False intent_ok=False errs=['no_json_found'] raw=' "I am not sure what the shipping time is for a laptop. I am'
  ...

[3] constrained decoding (prefix_allowed_tokens_fn)
  case00 schema_ok=True intent_ok=True 1301ms raw='{"intent": "refund", "slots": {"order_id": "88213", "product": "laptop"}, "confidence": 0.5}'
  case01 schema_ok=True intent_ok=False 1337ms raw='{"intent": "complaint", "slots": {"order_id": "40127", "product": "phone"}, "confidence": 0.5}'
  case02 schema_ok=True intent_ok=True 1692ms raw='{"intent": "refund", "slots": {"order_id": "77310", "product": "phone"}, "confidence": 0.5}'
  case05 schema_ok=True intent_ok=True 1515ms raw='{"intent": "refund", "slots": {"order_id": "88213", "product": "phone"}, "confidence": 0.5}'
  case06 schema_ok=True intent_ok=True 1981ms raw='{"intent": "faq", "slots": {"order_id": "12904", "product": "monitor"}, "confidence": 0.5}'
  ...

================ SUMMARY ================
free first-pass  parse=20.0% schema=0.0% intent=0.0%
after retry      schema=0.0% avg_attempts=3.00 dist={'3': 10}
constrained      schema=100.0% intent=40.0% order=40.0%
latency/case     free+retry=1.535s  constrained=1.510s
```

约束解码的搜索开销（`proj024_metrics.json`）：
`decode_calls=428` 次回调、`cache_states=370` 个不同前缀、平均生成 33.1 token。

## 8. 结果分析

**① 自由生成 schema 合法率 0%，不是"差一点"，是"完全不是这个任务"。**
gpt2 把 prompt 当成了"续写文本"，它学到的最强模式是 `{"id": ..., "name": ...}` 这种网页/日志片段。
10 条里有 2 条（20%）能截出可 `json.loads` 的对象，但那 2 条键名是 `id / name / description`，
schema 一条都不合格；剩下 8 条连 `{` 都没配平。**parse 20% 是"看起来有戏"的假信号**，
必须用 schema 校验才能拆穿。

**② 重试 3 次 0% 成功，且 10 条全部用满 3 次（`dist={'3': 10}`）。**
重试的前提是"失败是随机的"。这里失败是**系统性**的：模型压根不知道 schema 长什么样，
换一个采样种子只会换一段无关文本。工程结论：**重试只能修格式抖动，修不了能力缺失**。
要真正提升，应该微调（见 `LLM-101 指令微调`（待建））或改约束解码——本项目走后者。

**③ 约束解码把 schema 合法率从 0% 拉到 100%，但 intent 只有 40%。**
这是本项目最重要的一条数据：**结构合法 ≠ 语义正确**。
强制它只能输出 `{"intent": "..."}` 之后，模型仍在 4 个枚举里"瞎猜"——
gpt2 没有指令微调，"refund" 与 "money back" 的对应关系它没学过。
10 条里 4 条 intent 对（case00/02/05/06），order_id 也恰好 4 条正确（同样集中在这些 case）：
都是 prompt 里字面出现 `refund` / 命中固定模板的那几条。

**④ 延迟：约 3 倍生成次数换 100% 合法率。**
自由生成 + 重试 1.535 s/条（≈3 次生成），约束解码 1.510 s/条（1 次生成）。
单次生成对比：约束解码每步要跑一次自动机 + token 过滤（428 次回调 / 370 个状态），
单条 ~1.5 s，比自由生成的单次（~0.5 s）慢约 3 倍。
**但如果把重试算进来，"约束解码 1 次"反而比"自由生成 3 次"便宜**——
这就是约束解码在工业界赢的原因：它把不确定性从"事后补救"变成"事前排除"。

**⑤ 一个真实踩坑（已修）**：自动机第一版在"某段恰好被前缀吃完"时返回空集合，
导致第一步生成 `{` 之后就无 token 可选，`generate` 立刻 EOS，10 条全部只输出 `{`，
耗时 24 ms。修复是在"恰好吃完"时继续推进到下一段。这条写进第 9 节：**约束解码的状态机
最容易被"边界相等"打穿，必须写单测**。

## 9. 数据来源与差异说明（必读）

- **数据是合成的**。10 条客服文本是我按 4 个意图手写的，不是真实工单；4 个订单号、4 个商品名
  也是构造的。因此本项目的"40% intent 正确率"**不能**外推为"gpt2 的意图识别能力"——
  样本量太小，且模板化严重。
- **模型未微调**。这是刻意的：要让"约束解码 vs 自由生成"的差距显形。
  在真实业务里，做法是"微调后的 7B/70B 模型 + 约束解码"，格式合法率会接近 100%，
  intent 正确率取决于标注质量。本项目的 40% 是**下界**，不是上限。
- **语法是简化版**。值域被枚举化（4 个订单号 / 4 个商品 / 5 个置信度），
  所以自动机能在 30 行内写完。真实 schema 有 `oneOf`、嵌套数组、可选字段、联合类型，
  要交给 outlines / xgrammar / guidance 这类工具，把 JSON-Schema 编译成
  正则或 CFG，再算 token 前缀表。
- **token 层约束有 UTF-8 陷阱**。gpt2 是 byte-level BPE，一个汉字跨多个 token，
  逐字符判定会把"半个汉字"当成非法前缀。本项目语料是 ASCII 所以没触发；
  中文场景必须在 **byte 层**做自动机，或在 token 级缓存"可接受 token 集合"。
- **`product` 字段没有语义约束**。约束解码输出的 `product` 是枚举之一（laptop/phone/...），
  但真实抽取应该来自用户原话。本项目的 `order_ok=40%` 已说明"枚举里猜"有多不可靠。
- **延迟是共享 GPU 上测的，绝对值会漂**。同脚本在 GPU 更空闲时重跑过一次：
  `free+retry=1.175 s`、`constrained=1.268 s`（记录值是 1.535 / 1.510）。
  **比率稳定（约束解码 ≈ 1 次生成、自由生成 + 重试 ≈ 3 次生成），绝对秒数不稳**；
  四个合法率（0% / 0% / 100% / 40%）两次完全一致，因为它们由解码逻辑决定，与负载无关。

## 10. 复现步骤

```powershell
$env:PYTHONIOENCODING='utf-8'
cd <USER_HOME>\Documents\deepseek-harness\default-workspace\09-llm\projects
py -3.13 proj024_structured_output.py
```

产物：同目录 `proj024_metrics.json`（含每条用例的 raw 输出、错误列表、重试记录）。

环境要点：

1. 必须用 `py -3.13`——默认 `python` 是 3.15，没有 torch；
2. `HF_HUB_OFFLINE=1` 只能命中本地缓存的 `gpt2` / `sshleifer/tiny-gpt2`，不许下载新模型；
3. 约束解码是纯 CPU 逻辑，GPU 只跑前向；显存占用约 0.9 GB，与 PROJ-025 的基准不冲突；
4. 结果确定：greedy 解码 + 固定 seed，重跑数字一致（重试那一步用采样，分布可能 ±1 条）。

## 11. 完整训练脚本

本项目**不训练**——训练一个能稳定输出 schema 的模型需要指令微调数据与多卡，
不在本机约束内。交付物是推理侧脚本 `proj024_structured_output.py`（约 440 行），
结构如下：

```text
proj024_structured_output.py
├─ 常量区：INTENTS / ORDER_IDS / PRODUCTS / CONFS / CASES(10 条)
├─ build_prompt()               零样本指令拼接
├─ extract_first_json()         括号配平 + 字符串转义扫描，截第一个 {...}
├─ validate()                   6 类 schema 错误
├─ semantic_ok()                intent / order_id 是否等于期望
├─ GRAMMAR + allowed_next()     字面量/枚举段拼接的前缀自动机
├─ can_consume()                逐字符推进，判断 token 是否合法
├─ build_token_index()          首字符倒排索引（50k token，过滤 U+FFFD）
├─ ConstrainedDecoder            prefix_allowed_tokens_fn + 状态缓存
├─ AllowedTokensProcessor       logits mask（-inf）
├─ gen_free() / gen_constrained()
└─ main()                       三段实验 + 写 proj024_metrics.json
```

跑通判据（脚本末尾打印）：`constrained schema=100.0%`。

## 12. 验收清单

| 验收项 | 是否达成 | 证据 |
|---|---|---|
| 定义 JSON schema（intent/slots/confidence） | ✅ | 第 4 节 schema 文档 + `SCHEMA_DOC` |
| 自写 schema 校验函数 | ✅ | `validate()`，6 类错误码 |
| 生成后解析 + 校验并记录 | ✅ | 第 7 节 case00-case09 逐条 `errs` |
| 失败重试 ≤3 次 | ✅ | `MAX_RETRY=3`，`dist={'3': 10}` |
| 记录重试次数分布 | ✅ | `metrics.retry.attempt_distribution` |
| 实现 `prefix_allowed_tokens_fn` | ✅ | `ConstrainedDecoder.prefix_allowed_tokens_fn` |
| 约束解码 vs 自由生成合法率对比 | ✅ | 0.0% vs 100.0%（同 10 条用例） |
| 打印清晰指标 | ✅ | SUMMARY 4 行 + 逐用例行 |
| 写出 `proj024_metrics.json` | ✅ | 同目录，8.7 KB |
| 固定随机种子 | ✅ | `SEED=42`，三处都设 |

## 13. 结论与局限

**结论**

1. 自由生成 + 事后校验在弱模型上是**无效兜底**：schema 合法率 0%，重试 3 次仍 0%。
2. 约束解码能把**结构合法率**做到 100%，代价是单次生成慢约 3 倍（自动机 + token 过滤），
   但省掉了重试，总成本反而更低。
3. **结构合法率不是业务指标**。同一批样本上 intent 正确率只有 40%——
   格式由解码器保证，语义仍由模型能力决定，这两件事必须分开度量。

**局限**

- 样本 10 条、意图 4 类，统计意义有限，只能证明链路成立；
- 语法枚举化了值域，没有覆盖嵌套对象、数组、可选字段；
- ASCII 语料掩盖了 UTF-8 多字节前缀问题；
- 没做"约束 + 微调"的对照，40% 只是 gpt2 的下界。

## 14. 与后续项目的关系

- **PROJ-025 量化与推理优化实践**：本项目每次生成要跑 428 次自动机回调，
  延迟敏感；量化能否在不破坏 logits 排序的前提下提速，是下一个问题。
- `LLM-099 结构化输出与约束解码`（待建）：本项目的理论篇，
  会展开 JSON-Schema → CFG → token 前缀表的完整编译流程。
- `LLM-101 指令微调`（待建）：本项目证明了"不微调的模型 + 约束解码"只能解决格式，
  想解决语义必须微调。
- [[LLM-034 特殊 Token 与 Attention Mask]]：约束解码的边界条件（EOS 何时允许）
  本质就是特殊 token 与 mask 的问题。

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
| 项目仓库 | ai-knowledge-system/projects/PROJ-024 |

- **对应课程模块**：[M-LLM-10 大模型应用开发与工程实践](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-LLM-10%20%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%BA%94%E7%94%A8%E5%BC%80%E5%8F%91%E4%B8%8E%E5%B7%A5%E7%A8%8B%E5%AE%9E%E8%B7%B5.md)
- **前置知识**：[LLM-101 结构化输出与约束解码](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/08%20%E5%BA%94%E7%94%A8%E5%B7%A5%E7%A8%8B/LLM-101%20%E7%BB%93%E6%9E%84%E5%8C%96%E8%BE%93%E5%87%BA%E4%B8%8E%E7%BA%A6%E6%9D%9F%E8%A7%A3%E7%A0%81.md) · [LLM-092 采样策略：温度、Top-K、Top-P](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/06%20%E6%8E%A8%E7%90%86%E4%B8%8E%E9%83%A8%E7%BD%B2/LLM-092%20%E9%87%87%E6%A0%B7%E7%AD%96%E7%95%A5%EF%BC%9A%E6%B8%A9%E5%BA%A6%E3%80%81Top-K%E3%80%81Top-P.md)
- **对应知识点**：[LLM-034 特殊 Token 与 Attention Mask](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/02%20Token%E4%B8%8EEmbedding/LLM-034%20%E7%89%B9%E6%AE%8A%20Token%20%E4%B8%8E%20Attention%20Mask.md) · [LLM-099 推理成本与优化清单](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/06%20%E6%8E%A8%E7%90%86%E4%B8%8E%E9%83%A8%E7%BD%B2/LLM-099%20%E6%8E%A8%E7%90%86%E6%88%90%E6%9C%AC%E4%B8%8E%E4%BC%98%E5%8C%96%E6%B8%85%E5%8D%95.md)

---
id: PROJ-026
title: 端到端 LLM 应用：客服问答
domain: PROJ
module: [M-LLM-08, M-LLM-10]
type: project
difficulty: L3 推导
status: planned
tech_stack: [Python, PyTorch, transformers, numpy]
prerequisites: [LLM-076, LLM-107]
related_math: []
related: [LLM-083, LLM-106, PROJ-023]
notion_page_id: 412663e8-0333-81e2-92d5-fadf917c7364
notion_url: https://app.notion.com/p/LLM-412663e8033381e292d5fadf917c7364
created: 2026-11-08
updated: 2026-11-08
notion_db: 实战项目
所属学科: 机器学习
项目状态: 未开始
项目阶段: 立项
难度: L3 推导
项目目标: 端到端 LLM 应用：客服问答
技术栈: [Python, PyTorch, Transformers, NumPy]
项目仓库: ai-knowledge-system/projects/PROJ-026
对应课程模块: [M-LLM-08, M-LLM-10]
前置知识: [LLM-076, LLM-107]
对应知识点: [LLM-083, LLM-106]
tags: [机器学习]
---

# PROJ-026 端到端 LLM 应用：客服问答

## 1. 项目背景

模型能力 ≠ 产品能力。一个能上线的客服问答，除了"会说话"，还必须具备：

- **检索**：答案来自知识库，而不是模型记忆（可追溯、可更新）；
- **缓存**：同一个问题问 5 次，不该算 5 次 GPU 时间；
- **降级**：检索不到就转人工，而不是编一个答案；
- **指标**：命中率、延迟、Recall@K、降级次数——没有数字就无法上线。

本项目把这些零件串成一条真实链路，样本只有 12 条 FAQ、12 个问题，但每一步都是可运行的代码：

```text
问题 → 归一化 & hash → 缓存命中？ ─是→ 直接返回（0.04 ms）
                        └否→ 嵌入检索(numpy 余弦) → 阈值判断 ─低于→ 转人工话术
                                                          └高于→ gpt2 生成 → 回填缓存
```

**最重要的结论写在前面**：gpt2 生成的回答是**不可用的**（第 8.6 节有原样输出）。
本项目的价值在**工程链路**——检索、缓存、降级、度量都能独立验证；
回答质量这一环，需要换成本项目的经济能力之外的模型。

## 2. 项目目标

| 目标 | 验收方式 | 实测 |
|---|---|---|
| 自建 FAQ 知识库 | 12 条，脚本内写死 | 12 条 |
| 用户问题集 | 8-12 个，含重复与域外 | 12 个（2 重复 + 1 域外） |
| 检索（numpy 余弦，gpt2 token 嵌入平均池化） | Recall@1 / Recall@3 | 0.636 / **0.909** |
| 缓存（问题 hash → 答案） | 命中率 + 命中/未命中延迟 | 2/12 = **16.7%**，0.04 ms vs 762 ms |
| 降级（分数 < 阈值 → 转人工） | 触发次数 + 域外是否被拦 | 1 次，域外**被拦**（0.8198 < 0.8399） |
| 生成（gpt2） | 如实记录质量 | **不可用**，见 8.6 |
| 端到端成功回答 | 检索命中 + 未降级 + 非空 | 7/12 = 0.583 |

## 3. 技术栈与环境

| 项 | 值 |
|---|---|
| Python | 3.13（`py -3.13`） |
| PyTorch | 2.15.0+cu132 |
| transformers | 4.57.6 |
| 模型 | `gpt2`（124.4M），权重冻结，只做前向 |
| 设备 | **CPU**（可用 `DSH_PROJ026_DEVICE=cuda` 切 GPU） |
| 向量检索 | `numpy` 矩阵乘法（12×768，无需 faiss） |
| 随机性 | seed=42 |

**为什么默认 CPU**：gpt2 在 CPU 上答一句约 0.8-1.2 s，属于这个 demo 可接受的量级，
而且延迟数字可复现。本机 GPU 被其他任务共享，`nvidia-smi` 采样到 44%-100% 波动，
在 GPU 上测出来的"平均延迟"是噪声。这是一个工程判断：**demo 用 CPU，报告里有意义的比值**。

## 4. 数据集

**12 条 FAQ**（标题 + 答案，全部写在脚本里，英文——原因见第 9 节）：

```text
 0 Return policy      7 天内可退，保留包装/配件/发票，生鲜与定制除外
 1 Refund timing      审核通过后 3-5 个工作日原路退回，钱包当天
 2 Shipping time      现货 24 小时内发出，预售按商品页
 3 Shipping cost      满 99 免运费，否则 10，偏远另计
 4 Invoice download   订单页申请，下载 PDF 或发邮箱
 5 Warranty period    整机一年，电池/线材三个月，人为损坏不保
 6 Change address     未发货可自助改，已发货联系拦截
 7 Order tracking     「我的订单」筛选，或短信链接查物流
 8 Payment methods    微信/支付宝/银行卡，企业可对公
 9 Loyalty points     1 元 1 分，100 分抵 1 元，一年有效
10 Exchange process   7 天内质量问题免费换，需上传照片
11 Support hours      每天 9:00-21:00，其余留言次日 10 点前回
```

**12 个用户问题**（10 个不同意图 + 2 个完全重复 + 1 个域外）：

| # | 问题 | gold FAQ | 设计意图 |
|---|---|---|---|
| 1 | How do I return an item I bought last week? | 0 | 措辞与 FAQ 不同（return vs Return policy） |
| 2 | How long does a refund take to arrive? | 1 | 近义改写 |
| 3 | When will my order be shipped? | 2 | 近义改写 |
| 4 | How much is shipping and when is it free? | 3 | 两个条件问句 |
| 5 | Where can I download the invoice PDF? | 4 | 关键词命中 |
| 6 | What is the warranty period for the battery? | 5 | 关键词命中 |
| 7 | How do I return an item I bought last week? | 0 | **完全重复 → 缓存命中** |
| 8 | Can I change the delivery address after ordering? | 6 | 近义改写 |
| 9 | How long does a refund take to arrive? | 1 | **完全重复 → 缓存命中** |
| 10 | What are your support hours? | 11 | 关键词命中 |
| 11 | Which payment methods do you accept? | 8 | 关键词命中 |
| 12 | What is your company stock ticker? | — | **域外 → 应降级** |

**数据局限**：知识库是我手写的，问题也是我按"应该能检索到"的思路设计的，
所以 Recall 数字偏乐观；真实的客服问题有错别字、口语、指代（"那个东西"）、多意图混合，
本数据集一个都没有。**这 12 条只能证明链路通，不能证明效果。**

## 5. 方案设计

**① 检索：廉价嵌入**

```text
faq_vec[i] = mean( wte[tokenize(faq_i)] )  ，再 L2 归一化
sim(q)     = faq_mat @ q_vec                （numpy，12×768 一次矩阵乘）
top3       = argsort(-sim)[:3]
```

`wte` 是 gpt2 的 `transformer.wte`（50257×768），冻结不训练。
**这不是句向量模型**：没有上下文、没有对比学习、没有池化训练。它只是"手边唯一的向量"。

**② 缓存：问题 hash**

```text
key = sha1(normalize(question))[:16]      normalize = 去标点/小写/去空白
命中 → 直接返回上次的 answer（含当时的检索结果）
未命中 → 走完整链路，结束后回填
```

**③ 降级：绝对阈值**

```text
if top1_score < threshold: 返回转人工话术，不生成
```

阈值的标定：4 条域内 dev + 2 条域外 dev，取"域内最低 top1"与"域外最高 top1"的中点。
**若不可分（域内最低 < 域外最高）**，退化为"域内最低分"，即**召回优先**：
宁可放过域外，也不误伤正常问题。本数据集正好不可分，走的是这条规则（第 7 节有证据）。

**④ 生成：把 FAQ 塞进 prompt**

```text
FAQ knowledge: {title}. {answer}
Customer question: {question}
Reply to the customer in one short sentence:
```

greedy、`max_new_tokens=24`。

## 6. 核心代码

```text
class CheapEmbedder:
    """gpt2 token embedding 平均池化：便宜、无上下文、效果弱"""
    def __init__(self, model, tok):
        self.wte = model.transformer.wte.weight.detach()      # (50257, 768)
        self.tok = tok

    @torch.no_grad()
    def encode(self, texts):
        out = []
        for t in texts:
            ids = self.tok(t, return_tensors="pt", truncation=True,
                           max_length=64)["input_ids"][0]
            vec = self.wte[ids].mean(dim=0).float().cpu().numpy()
            out.append(vec / (np.linalg.norm(vec) + 1e-12))
        return np.stack(out)

def cosine_topk(qv, mat, k=3):
    sims = mat @ qv                       # 归一化后内积 = 余弦
    return [(int(i), float(sims[i])) for i in np.argsort(-sims)[:k]]

class SupportBot:
    def answer(self, question, use_cache=True):
        key = qhash(question)                              # sha1(normalize(q))
        if use_cache and key in self.cache:
            self.stats["cache_hits"] += 1
            rec = dict(self.cache[key])
            rec["cache_hit"] = True
            rec["latency_s"] = time.perf_counter() - t0     # 只剩字典查找
            return rec
        hits, t_ret = self.retrieve(question)               # 嵌入 + 余弦
        top1_idx, top1_score = hits[0]
        if top1_score < self.threshold:                     # 降级
            self.stats["degraded"] += 1
            rec = {"answer": FALLBACK, "faq_idx": -1, "degraded": True, ...}
        else:
            text, t_gen = self.generate(question, top1_idx) # gpt2 生成
            rec = {"answer": text, "faq_idx": top1_idx, "degraded": False, ...}
        self.cache[key] = {k: v for k, v in rec.items()
                           if k not in ("cache_hit", "latency_s")}
        return rec
```

> 以上为脚本 `proj026_llm_app_e2e.py` 的类结构摘录（依赖脚本内 `self.cache`、`qhash`、`FALLBACK` 等变量），**不能独立运行**，故以 `text` 呈现。

阈值标定（含"不可分"分支）：

```python
# 示例分数（真实分数见 proj026_metrics.json；此处仅为让阈值逻辑可独立运行）
in_scores = [0.912, 0.887, 0.864, 0.840]
ood_scores = [0.855, 0.820]

lo_in, hi_ood = min(in_scores), max(ood_scores)
separable = lo_in > hi_ood
if separable:
    thr, rule = (lo_in + hi_ood) / 2.0, "midpoint"
else:
    thr, rule = lo_in, "in_domain_floor_recall_first"   # 召回优先
```

## 7. 运行结果

真实终端输出（`py -3.13 proj026_llm_app_e2e.py`，CPU，seed=42）：

```text
device=cpu torch=2.15.0+cu132 seed=42
model=gpt2 params=124.4M faq=12 queries=12
[calib] min_in=0.8399 max_ood=0.8549 thr=0.8399 rule=in_domain_floor_recall_first separable=False ood_leak=True

[pass1] cold (cache disabled)
  How do I return an item I bought last week?    top1= 1(0.8788) gold=0  deg=0 gen= 1241ms tot= 1242ms
  How long does a refund take to arrive?         top1= 1(0.8963) gold=1  deg=0 gen= 1134ms tot= 1135ms
  When will my order be shipped?                 top1= 6(0.8744) gold=2  deg=0 gen=  970ms tot=  972ms
  How much is shipping and when is it free?      top1=10(0.8842) gold=3  deg=0 gen=  973ms tot=  974ms
  Where can I download the invoice PDF?          top1= 4(0.8969) gold=4  deg=0 gen=  951ms tot=  952ms
  What is the warranty period for the battery?   top1= 5(0.8873) gold=5  deg=0 gen= 1070ms tot= 1071ms
  How do I return an item I bought last week?    top1= 1(0.8788) gold=0  deg=0 gen= 1121ms tot= 1123ms
  Can I change the delivery address after orde   top1= 6(0.9089) gold=6  deg=0 gen= 1077ms tot= 1078ms
  How long does a refund take to arrive?         top1= 1(0.8963) gold=1  deg=0 gen=  992ms tot=  995ms
  What are your support hours?                   top1=11(0.8604) gold=11 deg=0 gen=  938ms tot=  939ms
  Which payment methods do you accept?           top1= 8(0.8447) gold=8  deg=0 gen=  818ms tot=  819ms
  What is your company stock ticker?             top1=-1(0.8198) gold=None deg=1 gen=    0ms tot=    0ms

[pass2] warm (cache enabled, cleared after pass1)
  How do I return an item I bought last week?    hit=0 top1= 1 tot=  995.42ms
  ...
  How do I return an item I bought last week?    hit=1 top1= 1 tot=    0.04ms      ← 缓存命中
  How long does a refund take to arrive?         hit=1 top1= 1 tot=    0.04ms      ← 缓存命中
  What is your company stock ticker?             hit=0 top1=-1 tot=    0.79ms      ← 降级，不生成

================ SUMMARY ================
Recall@1=0.636 Recall@3=0.909 (n_eval=11)
cache hit=2/12=16.7% hit=0.042ms miss=761.6ms speedup=18133x
degraded=1 ood_triggered=True ood_score=0.8198 thr=0.8399
cold_mean=942ms warm_mean=634.7ms
e2e_success=7/12 answered=11
sample answer: '"I bought this item from you and it is now gone. I will not return it to you for any reason'
```

检索命中细节（Recall@3 = 10/11 的原因在最后一列）：

| # | 问题 | gold | top-1 | top-1 分数 | 是否命中@3 |
|---|---|---|---|---|---|
| 1 | return an item last week | 0 | 1 | 0.8788 | ✅ |
| 2 | refund take to arrive | 1 | 1 | 0.8963 | ✅ |
| 3 | order be shipped | 2 | 6 | 0.8744 | ✅ |
| 4 | shipping, free | 3 | 10 | 0.8842 | ✅ |
| 5 | invoice PDF | 4 | 4 | 0.8969 | ✅ top-1 |
| 6 | warranty for battery | 5 | 5 | 0.8873 | ✅ top-1 |
| 7 | （重复 #1） | 0 | 1 | 0.8788 | ✅ |
| 8 | change delivery address | 6 | 6 | 0.9089 | ✅ top-1 |
| 9 | （重复 #2） | 1 | 1 | 0.8963 | ✅ |
| 10 | support hours | 11 | 11 | 0.8604 | ✅ top-1 |
| 11 | payment methods | 8 | 8 | 0.8447 | ✅ top-1 |
| 12 | stock ticker（域外） | — | -1 | 0.8198 | 降级（正确） |

## 8. 结果分析

**① 检索：Recall@3=0.909，Recall@1=0.636。**
平均池化嵌入能抓"话题"，抓不住"意图粒度"：
问题 1（return an item）被排在 1（Refund timing）而不是 0（Return policy）——
`return` 和 `refund` 在 gpt2 的嵌入空间里几乎同义；
问题 3（shipped）配到了 6（Change address），问题 4（shipping cost）配到了 10（Exchange process）。
**如果下游只允许 top-1（不允许并列看 3 条），这个检索器只有 64% 可用**——
这正是"召回@K ≠ 排序质量"的经典差别。

**② 缓存：命中率 16.7%，命中 0.042 ms vs 未命中 761.6 ms（≈1.8×10⁴ 倍）。**
这个倍数看着夸张，但含义很简单：**命中路径只做一次 sha1 + 字典查找，一次前向都没跑**。
更有意义的数字是"省下了多少次生成"：2 次命中 × 约 1.0 s ≈ **2.05 s** 生成时间。
命中率低是数据集决定的（12 条里只有 2 条重复）；真实客服场景重复率通常 20%-40%，
**这个 16.7% 是下限，不是收益上限**。
另一个设计点：缓存键用的是**原问题 hash**，所以"改一个字就不命中"，
生产上应该用"检索结果 + 语义 hash"或"FAQ id + 参数"作键，命中率会高很多（第 9 节）。

**③ 降级：触发 1 次，域外问题被正确拦下（0.8198 < 0.8399）。**
但标定集显示这套嵌入**分不开域内外**：域内最低 0.8399，域外最高 0.8549（`ood_leak=True`）。
也就是说，`What is the weather` 这类问题会**漏过阈值**去调生成。
这是绝对阈值方案的固有缺陷——**分数尺度不可比**。
生产方案要么用"相对阈值 + 二分类判别器"，要么用真正训练过的 embedding + 校准
（`LLM-083 RAG`（待建）会讲）。

**④ 延迟：冷启动 942 ms，带缓存 634.7 ms。**
其中生成占 0.8-1.2 s，检索（嵌入 + 余弦）只占 ~2-5 ms。
**瓶颈是生成**，这跟 PROJ-025 的结论一致：gpt2 单步固定开销 ~9 ms，24 token ≈ 0.9 s。
如果这里换成"FAQ 直接返回答案、不让模型改写"，延迟能掉到 10 ms 级——
这也说明"LLM 应用"里最贵的一步往往是可以被砍掉的那一步。

**⑤ 端到端成功 7/12（0.583）。**
判定口径很宽（检索 top-1 命中 gold + 未降级 + 答案非空），仍然只有 58%：
5 条没成功——**4 条检索 top-1 不是 gold**（问题 1 与它的那条重复问法、问题 3、问题 4），
1 条域外被正确降级。把"重复问法"折成 2 个不同问题来数，
**真正答对的是 7/12，其中 2 条是缓存命中的重复问法**——独立域内问题 9 个里答对 6 个（6/9）。
**这个数字已经比"回答是否真的有用"乐观得多。**

### 8.6 诚实说明：gpt2 的回答质量不足，本项目价值在工程链路

下面是**原样输出**，不是挑的坏例子，是第一条就长这样：

```text
Q: How do I return an item I bought last week?
gpt2: "I bought this item from you and it is now gone. I will not return it to you for any reason
```

再看两条：

```text
Q: How long does a refund take to arrive?
gpt2: "I have not received the refund yet, but I have been told that it will be processed within
Q: What are your support hours?
gpt2: "I am not sure what the support hours are, but I will try to find out.
```

结论很清楚：

1. **它没有回答问题**，只是在续写"客服语料风格"的句子；FAQ 内容对输出的影响微乎其微；
2. 带引号、 abruptly 截断、第一人称乱入——**连格式都不可用**；
3. 因此本项目的**回答质量指标不成立**：无法计算准确率、无法做人工评分（样本也太小）。

**那这个项目在交付什么？** 交付"链路的可验证性"：

| 组件 | 是否可独立验证 | 证据 |
|---|---|---|
| 检索 | ✅ 与模型能力无关 | Recall@1 0.636 / Recall@3 0.909 |
| 缓存 | ✅ 纯工程 | 2/12 命中、0.042 ms、省 2.05 s |
| 降级 | ✅ 纯工程 | 域外被拦、漏过条件已量化 |
| 生成 | ❌ 模型能力不足 | 上面三段原样输出 |

**把 gpt2 换成任何一个指令微调模型（或直接返回 FAQ 原文），这条链路一行不用改就能上线。**
这就是"工程链路"与"模型能力"解耦的意义，也是本项目真正的学习点。

**⑥ 一个被砍掉的版本（真实记录）。**
第一版知识库是**中文**的，用同一套 gpt2 嵌入：任意两问余弦落在 **0.9607-0.9827**，
标定出的阈值 0.9644 把正常问题「保修期是多久」判成域外，而域外问题「股票代码」拿到 0.9733 直接通过；
Recall@1 只有 0.5。原因是 gpt2 对中文按**字节**切分，嵌入被高频字节 token 主导，
向量近似共线。这个失败版本的数字留在 `proj026_metrics.json` 的 `chinese_variant` 字段里，
没有删——它比成功版本更能说明"廉价嵌入"的边界。

## 9. 数据来源与差异说明（必读）

- **知识库与问题都是合成的**，12 条 FAQ 是我按常见电商客服场景手写的英文条目，
  问题也是按"能被检索到"设计的。**Recall@3=0.909 是乐观值**：真实用户会说
  "我那个上周买的东西能退吗"，本数据集里一句这样的都没有。
- **语料是英文的**，这是刻意的取舍：gpt2 是英文模型，中文下 token embedding 平均池化
  退化成近乎常量向量（见 8.6 ⑥ 的真实数字）。换英文是为了让"检索 + 降级"这一段
  工程链路**可被验证**；代价是文档里的例子和答案都是英文。
  中文生产环境必须换 `bge-m3` / `text-embedding-3` 这类句向量模型，本项目的方法论不变。
- **嵌入是"廉价替代"**：token embedding 平均池化 ≠ 句向量。它没有上下文、
  没有池化训练、对词序不敏感（"狗咬人"和"人咬狗"向量相同）。第 8 节的检索错误
  （return/refund 混淆、shipped 配到 change address）都是这个缺陷的直接后果。
- **缓存键是原问题 hash**：改一个字就不命中。生产上应该用"FAQ id + 槽位"或
  语义 hash（embedding 量化后做 key），命中率能高一个量级。
- **降级用绝对阈值**：分数尺度不可比（`ood_leak=True` 已量化），
  换了知识库就要重标定。生产上要用判别器或相对阈值。
- **延迟是 CPU 数字**：与 PROJ-025 的 GPU 数字不可直接比较。
  CPU 上 gpt2 生成 24 token ≈ 0.9 s，比 GPU 慢，但**比值**（命中 vs 未命中）与硬件无关。
- **没有并发**：单线程串行 12 条查询，没有测 QPS、没有测缓存淘汰（LRU/TTL）、
  没有多轮对话（"那运费呢？"这种指代完全没处理）。

## 10. 复现步骤

```powershell
$env:PYTHONIOENCODING='utf-8'
cd <USER_HOME>\Documents\deepseek-harness\default-workspace\09-llm\projects
py -3.13 proj026_llm_app_e2e.py
```

产物：`proj026_metrics.json`（检索/缓存/降级/延迟/逐条样本 + 中文失败版本的记录）。

可选：

```powershell
# 想在 GPU 上复现（本机共享 GPU，数值会漂）
$env:DSH_PROJ026_DEVICE='cuda'; py -3.13 proj026_llm_app_e2e.py
```

要点：

1. 必须 `py -3.13`；
2. 只用缓存里的 `gpt2`，`HF_HUB_OFFLINE=1` 不会联网；
3. 全程可复现（greedy + seed=42），但缓存命中那两条的 `latency_s` 是 0.03-0.05 ms 级，
   受系统调度影响会有 ±0.01 ms 抖动；
4. 不需要 GPU，笔记本 CPU 也能跑完（12 条 ≈ 10 s）。

## 11. 完整训练脚本

本项目**不训练**。交付物是应用脚本 `proj026_llm_app_e2e.py`（约 380 行）：

```text
proj026_llm_app_e2e.py
├─ FAQS(12) / QUERIES(12) / CALIB_IN(4) / CALIB_OOD(2)   全部写死在脚本里
├─ normalize() / qhash()                                 缓存键：去标点 + sha1 前 16 位
├─ CheapEmbedder.encode()                                gpt2 wte 平均池化 + L2 归一化
├─ cosine_topk()                                         numpy 内积取 top-k
├─ SupportBot
│   ├─ retrieve()    嵌入 + 余弦，返回 top3 与耗时
│   ├─ generate()    拼 prompt → gpt2 greedy 24 token
│   └─ answer()      hash 查缓存 → 检索 → 阈值降级 → 生成 → 回填
├─ calibrate()       域内最低 / 域外最高 → 阈值（含"不可分"分支）
└─ main()            pass1 冷启动 / pass2 带缓存 → 指标 → proj026_metrics.json
```

## 12. 验收清单

| 验收项 | 是否达成 | 证据 |
|---|---|---|
| FAQ 知识库 10-15 条 | ✅ | 12 条，`FAQS` |
| 用户问题 8-12 个 | ✅ | 12 个（含 2 重复 + 1 域外） |
| numpy 余弦检索 | ✅ | `faq_mat @ q_vec`，无需 faiss |
| 用 gpt2 token 嵌入平均池化 | ✅ | `CheapEmbedder`，并注明是廉价替代 |
| 缓存：hash → 答案 | ✅ | `qhash()` = sha1(normalize)[:16] |
| 测命中率与省下的耗时 | ✅ | 16.7%、0.042 ms vs 761.6 ms、省 2.05 s |
| 降级：低分转人工 | ✅ | 阈值 0.8399，域外 0.8198 被拦 |
| Recall@3 | ✅ | 0.909（top-1 0.636） |
| 端到端成功回答数 | ✅ | 7/12 = 0.583 |
| 诚实说明生成质量不足 | ✅ | 第 8.6 节，含 3 段原样输出 |
| 写出 `proj026_metrics.json` | ✅ | 同目录，含逐条样本 + 中文失败记录 |
| 固定种子 | ✅ | `SEED=42` |

## 13. 结论与局限

**结论**

1. 一条**可上线的客服链路**由四段组成：检索、缓存、降级、生成。
   本项目四段都跑通了，而且**前三段与模型能力无关**——这才是工程价值的落点。
2. **缓存是性价比最高的一段**：0.04 ms vs 762 ms，改 10 行代码。
   但它的收益完全取决于重复率，而重复率取决于业务，不取决于模型。
3. **绝对阈值降级不可靠**：这套廉价嵌入分不开域内外（域内 0.8399 / 域外 0.8549），
   只能"召回优先"。真要做，得训判别器或换 embedding。
4. **Recall@3 0.909 掩盖了排序问题**：top-1 只有 0.636。只允许看一条的下游
   （比如自动回答）实际可用率是 64%，不是 91%。
5. **gpt2 的回答不可用**（第 8.6 节），这不需要掩饰：本项目的验收口径里，
   "回答质量"这一项是**不成立**的，能量化的只有链路。

**局限**

- 12 条 FAQ / 12 条问题，全部手写，Recall 偏乐观；
- 嵌入是平均池化，词序无关，检索错误有明确的结构性原因；
- 缓存键是原问题 hash，命中率被系统性低估；
- 单轮、串行、无并发、无 LRU/TTL、无多轮指代；
- 降级阈值在小样本上标定，换知识库就要重来。

## 14. 与后续项目的关系

- **PROJ-025 量化与推理优化实践**：本项目 942 ms 的冷启动里，约 0.9 s 是 gpt2 的
  24 步生成，每步 ~9 ms 的固定开销——这正是 PROJ-025 量出来的瓶颈。
  想让这条链路真正变快，顺序是"减少生成步数 → 缓存 → 量化"，不是反过来。
- `PROJ-023 RAG 问答系统`（待建）：同属"应用工程"分支，本项目是它的下游
  （RAG 的检索引擎 + 生成），区别是那边的知识库与评测更完整。
- `LLM-083 RAG`（待建）：检索增强生成的理论篇，本项目只做了最朴素的"FAQ 塞 prompt"。
- `LLM-106 缓存与降级策略`（待建）：语义缓存、TTL、熔断、兜底话术的工程篇。
- `LLM-107 端到端评测`（待建）：为什么"回答质量"需要 LLM-as-judge / 人工标注，
  以及为什么本项目只能报链路指标。
- [[LLM-035 Token 数量、上下文窗口与成本]]：prompt 长度直接决定延迟与成本。
- [[LLM-037 语义空间与相似度]]：为什么"余弦相似"不等于"语义相同"，本项目 8.6 ⑥ 是活例子。

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
| 技术栈 | Python、PyTorch、Transformers、NumPy |
| 项目仓库 | ai-knowledge-system/projects/PROJ-026 |

- **对应课程模块**：[M-LLM-08 RAG 检索增强生成](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-LLM-08%20RAG%20%E6%A3%80%E7%B4%A2%E5%A2%9E%E5%BC%BA%E7%94%9F%E6%88%90.md) · [M-LLM-10 大模型应用开发与工程实践](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/%E8%AF%BE%E7%A8%8B%E6%A8%A1%E5%9D%97/M-LLM-10%20%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%BA%94%E7%94%A8%E5%BC%80%E5%8F%91%E4%B8%8E%E5%B7%A5%E7%A8%8B%E5%AE%9E%E8%B7%B5.md)
- **前置知识**：[LLM-076 RAG 的基本架构](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/07%20RAG%E4%B8%8EAgent/LLM-076%20RAG%20%E7%9A%84%E5%9F%BA%E6%9C%AC%E6%9E%B6%E6%9E%84.md) · [LLM-107 缓存、重试与降级策略](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/08%20%E5%BA%94%E7%94%A8%E5%B7%A5%E7%A8%8B/LLM-107%20%E7%BC%93%E5%AD%98%E3%80%81%E9%87%8D%E8%AF%95%E4%B8%8E%E9%99%8D%E7%BA%A7%E7%AD%96%E7%95%A5.md)
- **对应知识点**：[LLM-083 RAG 的评估](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/07%20RAG%E4%B8%8EAgent/LLM-083%20RAG%20%E7%9A%84%E8%AF%84%E4%BC%B0.md) · [LLM-106 成本、延迟与质量的三角权衡](../../05_%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/08%20%E5%BA%94%E7%94%A8%E5%B7%A5%E7%A8%8B/LLM-106%20%E6%88%90%E6%9C%AC%E3%80%81%E5%BB%B6%E8%BF%9F%E4%B8%8E%E8%B4%A8%E9%87%8F%E7%9A%84%E4%B8%89%E8%A7%92%E6%9D%83%E8%A1%A1.md)

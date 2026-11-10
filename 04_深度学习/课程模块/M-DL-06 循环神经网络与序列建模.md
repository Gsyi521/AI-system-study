---
id: M-DL-06
title: 循环神经网络与序列建模
type: module
notion_page_id: 410663e8-0333-8167-9ce6-d187a507d831
notion_url: https://app.notion.com/p/410663e8033381679ce6d187a507d831
notion_db: 课程模块
学科: 深度学习
排序: 6
当前状态: 未开始
学习目标: RNN 的循环结构与 BPTT，LSTM/GRU 如何缓解长程依赖问题。
所属路线: [RM-005]
知识点: [DL-050, DL-051, DL-052, DL-053, DL-054, DL-055]
实战项目: [PROJ-018]
tags: [深度学习]
---

# M-DL-06 循环神经网络与序列建模

> 本笔记由 Notion「课程模块」数据库导出生成（Obsidian 侧原无对应笔记）。

## 学习目标

RNN 的循环结构与 BPTT，LSTM/GRU 如何缓解长程依赖问题。

---

## 🔗 关联

> 以下内容由 Notion「AI Knowledge System · 课程模块」数据库按知识 ID 合并而来（导出于 2026-11-09）。

| 属性 | 值 |
|---|---|
| 学科 | 深度学习 |
| 排序 | 6 |
| 当前状态 | 未开始 |

- **所属路线**：[RM-005 深度学习路线](../../00_%E7%9F%A5%E8%AF%86%E9%A6%96%E9%A1%B5/%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF/RM-005%20%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF.md)
- **知识点**：[DL-050 序列数据与序列建模任务](../03%20RNN%E4%B8%8E%E5%BA%8F%E5%88%97%E6%A8%A1%E5%9E%8B/DL-050%20%E5%BA%8F%E5%88%97%E6%95%B0%E6%8D%AE%E4%B8%8E%E5%BA%8F%E5%88%97%E5%BB%BA%E6%A8%A1%E4%BB%BB%E5%8A%A1.md) · [DL-051 RNN 结构与前向传播](../03%20RNN%E4%B8%8E%E5%BA%8F%E5%88%97%E6%A8%A1%E5%9E%8B/DL-051%20RNN%20%E7%BB%93%E6%9E%84%E4%B8%8E%E5%89%8D%E5%90%91%E4%BC%A0%E6%92%AD.md) · [DL-052 BPTT 与梯度问题](../03%20RNN%E4%B8%8E%E5%BA%8F%E5%88%97%E6%A8%A1%E5%9E%8B/DL-052%20BPTT%20%E4%B8%8E%E6%A2%AF%E5%BA%A6%E9%97%AE%E9%A2%98.md) · [DL-053 LSTM 与门控机制](../03%20RNN%E4%B8%8E%E5%BA%8F%E5%88%97%E6%A8%A1%E5%9E%8B/DL-053%20LSTM%20%E4%B8%8E%E9%97%A8%E6%8E%A7%E6%9C%BA%E5%88%B6.md) · [DL-054 GRU](../03%20RNN%E4%B8%8E%E5%BA%8F%E5%88%97%E6%A8%A1%E5%9E%8B/DL-054%20GRU.md) · [DL-055 RNN/LSTM/GRU 对比与序列任务实战](../03%20RNN%E4%B8%8E%E5%BA%8F%E5%88%97%E6%A8%A1%E5%9E%8B/DL-055%20RNN%E3%80%81LSTM%E3%80%81GRU%20%E5%AF%B9%E6%AF%94%E4%B8%8E%E5%BA%8F%E5%88%97%E4%BB%BB%E5%8A%A1%E5%AE%9E%E6%88%98.md)
- **实战项目**：[PROJ-018 序列任务——RNN、LSTM、GRU 同口径对比](../../07_%E5%AE%9E%E6%88%98%E9%A1%B9%E7%9B%AE/04%20%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E9%A1%B9%E7%9B%AE/PROJ-018%20%E5%BA%8F%E5%88%97%E4%BB%BB%E5%8A%A1%E2%80%94%E2%80%94RNN%E3%80%81LSTM%E3%80%81GRU%20%E5%90%8C%E5%8F%A3%E5%BE%84%E5%AF%B9%E6%AF%94.md)

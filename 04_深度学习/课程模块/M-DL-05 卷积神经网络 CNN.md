---
id: M-DL-05
title: 卷积神经网络 CNN
type: module
notion_page_id: 410663e8-0333-8198-80f2-e1a78711de37
notion_url: https://app.notion.com/p/410663e80333819880f2e1a78711de37
notion_db: 课程模块
学科: 深度学习
排序: 5
当前状态: 未开始
学习目标: 卷积/池化/感受野/残差连接，以及迁移学习的实际用法。
所属路线: [RM-005]
知识点: [DL-040, DL-041, DL-042, DL-043, DL-044, DL-045, DL-046, DL-047, DL-048, DL-049]
实战项目: [PROJ-016, PROJ-017, PROJ-019]
tags: [深度学习]
---

# M-DL-05 卷积神经网络 CNN

> 本笔记由 Notion「课程模块」数据库导出生成（Obsidian 侧原无对应笔记）。

## 学习目标

卷积/池化/感受野/残差连接，以及迁移学习的实际用法。

---

## 🔗 关联

> 以下内容由 Notion「AI Knowledge System · 课程模块」数据库按知识 ID 合并而来（导出于 2026-11-09）。

| 属性 | 值 |
|---|---|
| 学科 | 深度学习 |
| 排序 | 5 |
| 当前状态 | 未开始 |

- **所属路线**：[RM-005 深度学习路线](../../00_%E7%9F%A5%E8%AF%86%E9%A6%96%E9%A1%B5/%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF/RM-005%20%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF.md)
- **知识点**：[DL-040 CNN 基本思想与为什么图像适合 CNN](../02%20CNN/DL-040%20CNN%20%E5%9F%BA%E6%9C%AC%E6%80%9D%E6%83%B3%E4%B8%8E%E4%B8%BA%E4%BB%80%E4%B9%88%E5%9B%BE%E5%83%8F%E9%80%82%E5%90%88%20CNN.md) · [DL-041 图像张量、通道与批次维度](../02%20CNN/DL-041%20%E5%9B%BE%E5%83%8F%E5%BC%A0%E9%87%8F%E3%80%81%E9%80%9A%E9%81%93%E4%B8%8E%E6%89%B9%E6%AC%A1%E7%BB%B4%E5%BA%A6.md) · [DL-042 卷积运算](../02%20CNN/DL-042%20%E5%8D%B7%E7%A7%AF%E8%BF%90%E7%AE%97.md) · [DL-043 Padding、Stride 与输出尺寸](../02%20CNN/DL-043%20Padding%E3%80%81Stride%20%E4%B8%8E%E8%BE%93%E5%87%BA%E5%B0%BA%E5%AF%B8.md) · [DL-044 多通道卷积与 Feature Map](../02%20CNN/DL-044%20%E5%A4%9A%E9%80%9A%E9%81%93%E5%8D%B7%E7%A7%AF%E4%B8%8E%20Feature%20Map.md) · [DL-045 池化：Max 与 Average](../02%20CNN/DL-045%20%E6%B1%A0%E5%8C%96%E2%80%94%E2%80%94Max%20%E4%B8%8E%20Average.md) · [DL-046 感受野与参数共享](../02%20CNN/DL-046%20%E6%84%9F%E5%8F%97%E9%87%8E%E4%B8%8E%E5%8F%82%E6%95%B0%E5%85%B1%E4%BA%AB.md) · [DL-047 LeNet、AlexNet 与 VGG](../02%20CNN/DL-047%20LeNet%E3%80%81AlexNet%20%E4%B8%8E%20VGG.md) · [DL-048 ResNet 与 Skip Connection](../02%20CNN/DL-048%20ResNet%20%E4%B8%8E%20Skip%20Connection.md) · [DL-049 迁移学习与图像分类工作流](../02%20CNN/DL-049%20%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0%E4%B8%8E%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB%E5%B7%A5%E4%BD%9C%E6%B5%81.md)
- **实战项目**：[PROJ-016 手写数字分类——全连接基线](../../07_%E5%AE%9E%E6%88%98%E9%A1%B9%E7%9B%AE/04%20%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E9%A1%B9%E7%9B%AE/PROJ-016%20%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E5%88%86%E7%B1%BB%E2%80%94%E2%80%94%E5%85%A8%E8%BF%9E%E6%8E%A5%E5%9F%BA%E7%BA%BF.md) · [PROJ-017 手写数字分类——CNN 改进](../../07_%E5%AE%9E%E6%88%98%E9%A1%B9%E7%9B%AE/04%20%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E9%A1%B9%E7%9B%AE/PROJ-017%20%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E5%88%86%E7%B1%BB%E2%80%94%E2%80%94CNN%20%E6%94%B9%E8%BF%9B.md) · [PROJ-019 迁移学习——小样本分类](../../07_%E5%AE%9E%E6%88%98%E9%A1%B9%E7%9B%AE/04%20%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E9%A1%B9%E7%9B%AE/PROJ-019%20%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0%E2%80%94%E2%80%94%E5%B0%8F%E6%A0%B7%E6%9C%AC%E5%88%86%E7%B1%BB.md)

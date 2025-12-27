---
title: "论文梗概：Bi-Directional Attention Flow for Machine Comprehension"
date: 2019-11-19T16:47:58+08:00
tags:
  - MRC
  - bidaf
  - attention
---

## Abstract 要点

- 需要对 context 和 query 之间的复杂交互进行建模
- 使用 attention 关注 context 的小部分并用固定大小的向量进行概括
- 多阶段分层过程：在不同粒度级别表示 context，使用双向 attention flow 机制获取 query-aware 的 context 表示，避免过早概括

## Introduction 要点

### Bi-directional Attention Flow 的三个创新点

**第一：不用 attention 概括 context 为固定大小向量**

Attention 在每个时间步计算，attended vector 连同之前层的表示一起流向后续的 modeling layer（这与 self-attention 类似），避免了过早概括造成的信息损失。

**第二：Memory-less Attention 机制**

虽然我们迭代地计算 attention，但每个时间步的 attention 只是当前 query 和 context 的函数，不直接依赖于前一时间步的 attention。

这迫使 attention layer 专注于学习 query 和 context 之间的 attention，使 modeling layer 专注于学习 query-aware context 表示内部的交互。同时避免了前一时间步错误 attention 对当前时间步的影响。

**关键对比：**

| 类型 | 描述 |
|------|------|
| **Dynamic Attention** | 当前时间步的 attention 权重是前一时间步 attended vector 的函数 |
| **Memory-less Attention** | 当前时间步的 attention 只依赖于当前的 query 和 context |

> 作者声称 memory-less attention 比 dynamic attention 有明显优势。

**第三：双向信息互补**

双向的 attention 提供了互补的信息。

## BiDAF 网络架构

前三层在不同粒度级别计算 query 和 context 的特征，类似于 CV 中 CNN 的多阶段特征计算。

### 1. Character-level Embedding

字符级别的嵌入表示。

### 2. Word-level Embedding

词级别的嵌入表示。

### 3. Contextual Embedding

利用上下文线索来精炼词的嵌入：

> 在前面层提供的 embeddings 之上使用 LSTM 来建模词之间的时序交互。双向放置 LSTM 并拼接两个方向的输出。

### 4. Attention Flow Layer

耦合 query 和 context 向量，为 context 中的每个词生成 query-aware 的特征向量。

```
α(h, u) = W[h; u; h ⊙ u]
```

**两种 Attention 方向：**

- **Context-to-Query (C2Q)**：表示每个 context 词最相关的 query 词
- **Query-to-Context (Q2C)**：表示哪些 context 词与某个 query 词最相似，对回答问题最关键

### 5. Modeling Layer

使用 RNN 扫描 context，捕获长距离依赖。

### 6. Output Layer

提供问题的答案（面向任务的具体实现）。

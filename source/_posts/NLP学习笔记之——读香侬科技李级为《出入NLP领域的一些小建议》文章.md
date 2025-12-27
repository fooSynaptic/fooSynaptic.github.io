---
title: "NLP 学习笔记：读香侬科技李纪为《初入 NLP 领域的一些小建议》"
date: 2019-10-31T17:31:53+08:00
tags:
  - NLP
  - machine learning
  - deep learning
---

## 推荐书籍和学习材料

- *Speech and Language Processing*
- *Introduction to Information Retrieval*
- 吴恩达的机器学习课程
- *Pattern Recognition and Machine Learning*

## 1. 了解 NLP 的最基本知识

### N-gram 模型

N-gram 模型蕴含了概率语言模型的假设，同时也是一个 n-1 阶的马尔可夫假设——一个词出现的概率只和它前面 n-1 个词相关：

```
p(w_k | w_1...w_{k-1}) = count(w_{k-n+1}...w_k) / count(w_{k-n+1}...w_{k-1})
```

### BERT 中的随机替换

斯坦福吴恩达组的 Ziang Xie 在 *Data Noising as Smoothing in Neural Network Language Models* 中首次提出此方法并给出理论解释：这种随机替换本质上属于语言模型中基于 interpolation 的平滑方式。

## 2. 了解经典的 NLP 模型

### IBM 翻译模型

对于每个词有一个 alignment list，目标函数为 P(a|src, tgt)，表示以平行翻译语料为条件，找到对齐的概率。

参数估计使用 **EM 算法**：
1. **E 步**：利用对隐藏变量的现有估计值，计算最大似然估计值
2. **M 步**：通过最大似然目标估计参数

IBM Model 中的隐变量就是句子中词语的对齐信息 a。

### Attention 机制

诞生于神经网络机器翻译的 attention，本质上是 IBM 模型的神经网络版本。

### 对话系统的无聊回复处理

用反向概率做 reranking 可以进一步提高检索结果。在早期的统计机器翻译（phrase-based MT）中，需要对 N-best list 用 MERT 做 reranking，反向概率（given target, the p of source）是 reranking 中重要的 feature。

## 3. 了解机器学习基本模型

### EM 算法

EM 算法通过概率模型寻找最大似然估计或最大后验估计。当模型中有无法观测的隐变量时，EM 算法经过两个步骤交替计算，不断迭代。

### CRF（条件随机场）

参考我的文章：[条件随机场的原理以及从零实现](https://zhuanlan.zhihu.com/p/89553094)

### Dropout

一种 bagging 模型集成策略。

### 优化器

- SGD
- Momentum
- AdaBoost
- AdaGrad
- Adam

## 4. 多看 NLP 其他子领域的论文

- Machine Translation
- Information Extraction
- Parsing
- Tagging
- Sentiment Analysis
- Machine Reading Comprehension

## 5. 了解 CV 和 Data Mining 领域的基本重大进展

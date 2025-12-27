---
title: "NLP学习笔记之——读香侬科技李级为《出入NLP领域的一些小建议》文章"
date: 2019-10-31T17:31:53+08:00
tags:
  - bayesian network
  - machine learning
  - causality infer
---

### 涉及到的书籍和学习材料

  * Speech and Language Processing
  * Introduction to infromation retrieval
  * 吴恩达的机器学习
  * Pattern recognition and Machine learning



## 1.了解NLP的最基本的知识

### Ngram

ngram模型中蕴含了概率语言模型的假设，同时也是一个n-1阶的马尔科夫假设，也就是一个词出现的概率只和它前面n-1个词相关  
p(w_k|w1…k-1) = count(w(k-n+1…k)) / count(w(k-n+1…k-1))

### Bert 里面训练LM的随机替换能够使得训练结果变得更好地原因

斯坦福的吴恩达组的ziang Xie的Data Noising as Smoothing in Neurala Network Language Models就首次提出了此方法，并且给了理论解释，这种random的替换其实本质上属于language model里面基于interpolation的平滑方式。

## 了解早点经典的NLP模型以及论文

### 机器翻译中的IBM模型大概是干嘛的

对于每个词有一个alignment的list，目标函数为P(a|src, tgt)，表示以平行翻译语料作为条件，找到它对齐的概率。  
参数估计的方法就是EM算法，通过概率模型中寻找最大似然估计或者最大后验估计的算法。概率模型中有无法观测的隐形变量，em算法经过两个步骤交替进行计算，第一步是计算期望，利用对隐藏变量的现有估计值（起始位随机初始化），计算其最大似然估计值；第二步是最大后验概率，最大后验概率通过求得最大似然的目标来估计参数，然后新股寄出来的参数用于下一个计算期望的步骤，不断迭代进行。  
IBMmodel中的隐变量就是句子中词语的对齐信息a；  
所以  
P(a|src, tgt)  
= P(a, src|tgt) / P(src|tgt)  
P(src|tgt) 重点  
= P(src, a|tgt)

### 神经机器翻译中正向翻译和反向翻译预测的target要一致对齐，这个是通过双向attention的约束项来实现的。

### 处理对话系统的无聊回复

用反向概率做rerank可以进一步提高检索的结果。在早期的统计机器翻译中（phrase-base MT）需要对一个大的N-best list用MERT做reranking，反向概率（given target， the p of source）是reranking中feature的重要标志。

### 诞生于神经网络机器翻译的attention，其实就是IBM模型的神经网络版本。

## 了解机器学习的基本模型

### EM算法是什么

参数估计的方法就是EM算法，通过概率模型中寻找最大似然估计或者最大后验估计的算法。概率模型中有无法观测的隐形变量，em算法经过两个步骤交替进行计算，第一步是计算期望，利用对隐藏变量的现有估计值（起始位随机初始化），计算其最大似然估计值；第二步是最大后验概率，最大后验概率通过求得最大似然的目标来估计参数，然后新股寄出来的参数用于下一个计算期望的步骤，不断迭代进行。

### 什么是variational inference

### 如何理解CRF

参考我的文章[ref](https://zhuanlan.zhihu.com/p/89553094)

### Dropout

一种bagging模型集成策略

### SGD, momentum, adaboost, adagrad, adam

## 多看NLP其他子领域的论文

  * MT, information extracition, parsing, ragging, sentiment analysis, machine reading comprehension.



## 了解CV和data mining领域的基本重大进展

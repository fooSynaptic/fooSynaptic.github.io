---
title: "神经网络机器阅读理解：从 Attention 到 LLM"
date: 2019-11-22T13:31:56+08:00
tags:
  - MRC
  - KBQA
  - Deep learning
  - LLM
---

本文综述神经网络在机器阅读理解和对话系统中的发展历程，从早期的注意力机制到现代大语言模型。

## 发展时间线

```
2015-2016: 注意力机制兴起
    └── Attentive Reader, Impatient Reader, BiDAF

2017-2018: 深度交互与预训练
    └── R-Net, QANet, BERT

2019-2020: 大规模预训练
    └── RoBERTa, ALBERT, T5

2021-2023: 大语言模型时代
    └── GPT-3, ChatGPT, GPT-4, LLaMA

2024-: 检索增强与多模态
    └── RAG, Vision-Language Models
```

## 核心技术演进

### 阶段一：注意力机制 (2015-2017)

**问题**：如何让模型"关注"与问题相关的上下文？

$$
\alpha_i = \text{softmax}(s(h_i, q))
$$

$$
c = \sum_i \alpha_i h_i
$$

**代表模型**：Attentive Reader, BiDAF

### 阶段二：深度交互 (2017-2018)

**问题**：如何建模问题和上下文的复杂交互？

**技术**：多轮注意力、自注意力、门控机制

```python
# 多轮推理 (R-Net 风格)
for layer in range(num_layers):
    # 自注意力
    context = self_attention(context, context)
    # 交叉注意力
    context = cross_attention(context, question)
```

### 阶段三：预训练语言模型 (2018-2020)

**范式转变**：从 task-specific 到 pretrain-finetune

$$
\theta^* = \arg\min_\theta \mathcal{L}_{task}(\text{PLM}_\theta(x), y)
$$

**代表模型**：BERT, RoBERTa, ALBERT

```python
from transformers import AutoModelForQuestionAnswering

model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
# Fine-tune on SQuAD
```

### 阶段四：大语言模型 (2020-至今)

**范式转变**：从 fine-tuning 到 prompting

```python
# Few-shot prompting
prompt = """
Context: The Eiffel Tower was built in 1889.
Question: When was the Eiffel Tower built?
Answer: 1889

Context: {context}
Question: {question}
Answer:"""
```

## 架构对比

| 模型 | 参数量 | 训练范式 | SQuAD 2.0 F1 |
|------|--------|----------|--------------|
| BiDAF | ~2M | 从零训练 | 77.3 |
| BERT-base | 110M | 预训练+微调 | 88.5 |
| BERT-large | 340M | 预训练+微调 | 90.9 |
| RoBERTa-large | 355M | 预训练+微调 | 91.4 |
| GPT-3 | 175B | Few-shot | ~88 |
| GPT-4 | ~1.8T | Zero-shot | ~95 |

## 现代 MRC 系统设计

### RAG 架构

```python
class ModernMRC:
    def __init__(self, retriever, reader):
        self.retriever = retriever  # Dense retriever
        self.reader = reader        # LLM
    
    def answer(self, question: str, knowledge_base: str = None):
        # 1. 检索
        if knowledge_base:
            docs = self.retriever.retrieve(question, knowledge_base)
            context = "\n\n".join([d.text for d in docs])
        else:
            context = ""
        
        # 2. 阅读理解/生成
        prompt = self._build_prompt(question, context)
        answer = self.reader.generate(prompt)
        
        # 3. 后处理（可选：验证、引用）
        return self._postprocess(answer, docs)
    
    def _build_prompt(self, question, context):
        if context:
            return f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}
Answer:"""
        else:
            return f"Question: {question}\nAnswer:"
```

### 多跳推理

```python
class MultiHopReasoner:
    def __init__(self, retriever, llm, max_hops=3):
        self.retriever = retriever
        self.llm = llm
        self.max_hops = max_hops
    
    def reason(self, question):
        reasoning_chain = []
        current_query = question
        
        for hop in range(self.max_hops):
            # 检索
            docs = self.retriever.retrieve(current_query)
            
            # 生成中间推理
            intermediate = self.llm.generate(
                f"Based on: {docs}\nQuestion: {current_query}\n"
                f"Provide intermediate reasoning or the final answer:"
            )
            
            reasoning_chain.append({
                'query': current_query,
                'docs': docs,
                'reasoning': intermediate
            })
            
            # 检查是否已得到答案
            if self._is_final_answer(intermediate):
                break
            
            # 生成下一跳查询
            current_query = self._generate_next_query(question, reasoning_chain)
        
        return self._synthesize_answer(question, reasoning_chain)
```

## 对话系统中的 MRC

### 对话式问答

```python
class ConversationalQA:
    def __init__(self, mrc_model, history_length=5):
        self.mrc_model = mrc_model
        self.history = []
        self.history_length = history_length
    
    def ask(self, question, context=None):
        # 将对话历史纳入问题
        contextualized_question = self._contextualize(question)
        
        # 获取答案
        answer = self.mrc_model.answer(contextualized_question, context)
        
        # 更新历史
        self.history.append({'q': question, 'a': answer})
        if len(self.history) > self.history_length:
            self.history.pop(0)
        
        return answer
    
    def _contextualize(self, question):
        if not self.history:
            return question
        
        history_text = "\n".join([
            f"Q: {turn['q']}\nA: {turn['a']}"
            for turn in self.history
        ])
        
        return f"Conversation history:\n{history_text}\n\nCurrent question: {question}"
```

## 评估体系

### 传统指标

| 指标 | 定义 | 适用场景 |
|------|------|----------|
| EM | 精确匹配 | 抽取式 QA |
| F1 | Token 重叠 | 抽取式 QA |
| BLEU | N-gram 重叠 | 生成式 QA |
| ROUGE | 召回导向重叠 | 摘要、长答案 |

### LLM 时代指标

```python
# LLM-as-Judge
def llm_evaluate(question, reference, prediction):
    prompt = f"""Evaluate the answer quality on a scale of 1-5:

Question: {question}
Reference Answer: {reference}
Model Answer: {prediction}

Criteria:
- Correctness: Is the information accurate?
- Completeness: Does it fully answer the question?
- Conciseness: Is it appropriately brief?

Score (1-5):"""
    
    return llm.generate(prompt)
```

## 延伸阅读

- [Reading Wikipedia to Answer Open-Domain Questions](https://arxiv.org/abs/1704.00051)
- [RAG Paper](https://arxiv.org/abs/2005.11401)
- [HotpotQA: Multi-hop Reasoning](https://arxiv.org/abs/1809.09600)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)

---

> 转载请注明出处

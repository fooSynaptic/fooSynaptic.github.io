---
title: "NLP 学习路线：从基础到大语言模型"
date: 2019-10-31T17:31:53+08:00
tags:
  - NLP
  - machine learning
  - LLM
---

本文整理了 NLP 领域的学习路线，结合经典理论与现代大语言模型技术。

## 推荐学习资源

### 经典教材

| 书籍 | 内容 | 难度 |
|------|------|------|
| *Speech and Language Processing* (Jurafsky) | NLP 全面综述 | ⭐⭐ |
| *Introduction to Information Retrieval* | 信息检索基础 | ⭐⭐ |
| *Pattern Recognition and Machine Learning* | 机器学习理论 | ⭐⭐⭐⭐ |
| *Deep Learning* (Goodfellow) | 深度学习基础 | ⭐⭐⭐ |

### 现代资源

- [Stanford CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/)
- [Hugging Face Course](https://huggingface.co/course)
- [LLM University by Cohere](https://docs.cohere.com/docs/llmu)

## 阶段一：NLP 基础

### 语言模型基础

**N-gram 模型**：N-1 阶马尔可夫假设

$$
P(w_k | w_1, ..., w_{k-1}) \approx P(w_k | w_{k-n+1}, ..., w_{k-1})
$$

```python
from collections import defaultdict
import numpy as np

class NGramLM:
    def __init__(self, n=3):
        self.n = n
        self.counts = defaultdict(lambda: defaultdict(int))
        self.totals = defaultdict(int)
    
    def train(self, corpus):
        for sentence in corpus:
            tokens = ['<s>'] * (self.n - 1) + sentence + ['</s>']
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i+self.n-1])
                word = tokens[i+self.n-1]
                self.counts[context][word] += 1
                self.totals[context] += 1
    
    def probability(self, word, context):
        context = tuple(context[-(self.n-1):])
        return self.counts[context][word] / max(self.totals[context], 1)
```

### 词向量

从 One-hot 到 Dense Embedding 的演进：

| 方法 | 年份 | 特点 |
|------|------|------|
| One-hot | - | 稀疏，无语义 |
| Word2Vec | 2013 | 分布式表示 |
| GloVe | 2014 | 全局统计 |
| FastText | 2016 | 子词信息 |
| ELMo | 2018 | 上下文相关 |
| BERT | 2018 | 双向上下文 |

## 阶段二：深度学习 NLP

### Transformer 架构

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # Linear projections
        Q = self.W_q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        
        # Concatenate and project
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        return self.W_o(output)
```

### 注意力机制的数学表达

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

## 阶段三：大语言模型

### LLM 架构演进

```
GPT-1 (2018) → GPT-2 → GPT-3 → ChatGPT → GPT-4
     ↓
BERT → RoBERTa → DeBERTa
     ↓
T5 → Flan-T5 → UL2
     ↓
LLaMA → LLaMA 2 → Mistral → Mixtral
```

### Prompt Engineering

```python
# 1. Zero-shot
prompt = "Translate to French: Hello, how are you?"

# 2. Few-shot
prompt = """Translate to French:
Hello -> Bonjour
Goodbye -> Au revoir
How are you? ->"""

# 3. Chain-of-Thought
prompt = """Q: If I have 3 apples and buy 5 more, how many do I have?
A: Let's think step by step.
1. I start with 3 apples.
2. I buy 5 more apples.
3. Total = 3 + 5 = 8 apples.
The answer is 8.

Q: If I have 7 oranges and eat 2, how many remain?
A: Let's think step by step."""
```

### Fine-tuning 技术

| 方法 | 可训练参数 | 适用场景 |
|------|-----------|----------|
| **Full Fine-tuning** | 100% | 大量数据，充足算力 |
| **LoRA** | 0.1-1% | 资源受限 |
| **QLoRA** | 0.1% | 消费级 GPU |
| **Prefix Tuning** | 0.1% | 多任务 |
| **Prompt Tuning** | <0.01% | 极端资源受限 |

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
)

model = get_peft_model(base_model, lora_config)
print(f"Trainable params: {model.print_trainable_parameters()}")
```

## 阶段四：高级主题

### 检索增强生成 (RAG)

```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 构建向量库
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
vectorstore = Chroma.from_documents(documents, embeddings)

# 创建 RAG 链
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)
```

### 模型评估

```python
# 困惑度 (Perplexity)
def perplexity(model, tokenizer, text):
    encodings = tokenizer(text, return_tensors='pt')
    max_length = model.config.n_positions
    
    nlls = []
    for i in range(0, encodings.input_ids.size(1), max_length):
        begin_loc = max(i - max_length, 0)
        end_loc = i + max_length
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-1] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss)
    
    return torch.exp(torch.stack(nlls).mean())
```

## 实践项目建议

1. **入门**：情感分析、文本分类
2. **进阶**：命名实体识别、机器翻译
3. **高级**：问答系统、RAG 应用
4. **专家**：LLM 预训练、RLHF

## 延伸阅读

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [LLaMA Paper](https://arxiv.org/abs/2302.13971)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

---

> 转载请注明出处

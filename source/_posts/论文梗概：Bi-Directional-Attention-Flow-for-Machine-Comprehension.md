---
title: "BiDAF 论文解读：双向注意力流机制"
date: 2019-11-19T16:47:58+08:00
tags:
  - MRC
  - attention
  - deep learning
---

BiDAF (Bi-Directional Attention Flow) 是机器阅读理解领域的经典模型，其双向注意力机制对后续 Transformer 架构产生了深远影响。

## 核心创新

### 1. Memory-less Attention

传统动态注意力 vs BiDAF 的无记忆注意力：

| 特性 | Dynamic Attention | Memory-less Attention |
|------|------------------|----------------------|
| 依赖 | 前一时间步的 attended vector | 仅当前 query 和 context |
| 优势 | 可建模时序依赖 | 避免错误累积 |
| 缺点 | 错误会传播 | 无法建模长程依赖 |

### 2. 双向注意力

同时计算：
- **Context-to-Query (C2Q)**：每个 context 词最相关的 query 词
- **Query-to-Context (Q2C)**：对回答问题最关键的 context 词

## 模型架构

```
Input → Embedding → Encoding → Attention → Modeling → Output
  │         │           │          │           │         │
 词向量    字符CNN     BiLSTM    双向注意力   BiLSTM   Span预测
```

### 数学表达

**相似度矩阵**：

$$
S_{ij} = \alpha(H_i, U_j) = w^T[H_i; U_j; H_i \odot U_j]
$$

其中 $H \in \mathbb{R}^{T \times d}$ 是 context 表示，$U \in \mathbb{R}^{J \times d}$ 是 query 表示。

**C2Q Attention**：

$$
\tilde{U}_i = \sum_j a_{ij} U_j, \quad a_i = \text{softmax}(S_i)
$$

**Q2C Attention**：

$$
\tilde{H} = \sum_i b_i H_i, \quad b = \text{softmax}(\max_j S_{:j})
$$

**融合表示**：

$$
G_i = [H_i; \tilde{U}_i; H_i \odot \tilde{U}_i; H_i \odot \tilde{H}]
$$

## PyTorch 实现

```python
import torch
import torch.nn as nn

class BiDAFAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size * 3, 1, bias=False)
    
    def forward(self, context, query, c_mask, q_mask):
        """
        Args:
            context: (batch, c_len, hidden)
            query: (batch, q_len, hidden)
            c_mask: (batch, c_len)
            q_mask: (batch, q_len)
        """
        batch, c_len, hidden = context.size()
        q_len = query.size(1)
        
        # 扩展维度以计算所有 (i, j) 对
        c_expand = context.unsqueeze(2).expand(-1, -1, q_len, -1)
        q_expand = query.unsqueeze(1).expand(-1, c_len, -1, -1)
        
        # 计算相似度矩阵 S
        cq = torch.cat([c_expand, q_expand, c_expand * q_expand], dim=-1)
        S = self.W(cq).squeeze(-1)  # (batch, c_len, q_len)
        
        # Mask
        q_mask_expand = q_mask.unsqueeze(1).expand(-1, c_len, -1)
        S = S.masked_fill(~q_mask_expand, -1e9)
        
        # C2Q attention
        a = torch.softmax(S, dim=-1)
        c2q = torch.bmm(a, query)  # (batch, c_len, hidden)
        
        # Q2C attention
        b = torch.softmax(S.max(dim=-1)[0], dim=-1)
        q2c = torch.bmm(b.unsqueeze(1), context)  # (batch, 1, hidden)
        q2c = q2c.expand(-1, c_len, -1)
        
        # 融合
        G = torch.cat([context, c2q, context * c2q, context * q2c], dim=-1)
        
        return G
```

## 与 Transformer 的对比

| 特性 | BiDAF | Transformer |
|------|-------|-------------|
| 注意力方向 | 双向（C2Q, Q2C） | 全方向自注意力 |
| 位置编码 | BiLSTM 隐式编码 | 显式位置编码 |
| 并行化 | 受限于 RNN | 完全并行 |
| 长距离依赖 | 受限 | 理论上无限 |
| 参数量 | 较少 | 较多 |

## 现代演进

BiDAF 的思想在现代模型中的体现：

### 1. Cross-Attention in Transformer

```python
class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads)
    
    def forward(self, query, key_value):
        # query 来自一个序列，key/value 来自另一个序列
        return self.mha(query, key_value, key_value)
```

### 2. FiD (Fusion-in-Decoder)

用于 RAG 的架构，类似 BiDAF 的融合思想：

```python
class FiD(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, question, passages):
        # 独立编码每个 passage
        encoded = []
        for passage in passages:
            enc = self.encoder(question + passage)
            encoded.append(enc)
        
        # 融合解码
        fused = torch.cat(encoded, dim=1)
        return self.decoder(fused)
```

## 实验结果（原论文）

在 SQuAD 1.1 上的表现：

| 模型 | EM | F1 |
|------|----|----|
| BiDAF | 67.7 | 77.3 |
| BiDAF + Self Attention | 72.1 | 81.1 |
| BERT-base | 80.8 | 88.5 |
| GPT-4 (few-shot) | ~90 | ~95 |

## 延伸阅读

- [BiDAF Paper](https://arxiv.org/abs/1611.01603)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [BERT for QA](https://arxiv.org/abs/1810.04805)

---

> 转载请注明出处

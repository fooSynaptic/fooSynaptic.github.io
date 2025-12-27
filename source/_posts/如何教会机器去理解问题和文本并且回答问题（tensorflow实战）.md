---
title: "MRC 模型实现：从 TensorFlow 到 PyTorch"
date: 2019-11-19T13:16:27+08:00
tags:
  - MRC
  - Deep learning
  - PyTorch
---

本文介绍机器阅读理解模型的完整实现，涵盖经典架构和现代最佳实践。

## 问题定义

**输入**：
- 问题 $Q = (q_1, q_2, ..., q_m)$
- 文档 $P = (p_1, p_2, ..., p_n)$

**输出**：
- 答案起始位置 $start \in [1, n]$
- 答案结束位置 $end \in [start, n]$

## 经典架构

```
Input → Embedding → Encoding → Matching → Fusion → Decoding
```

### 各层详解

| 层 | 功能 | 现代替代 |
|---|------|----------|
| Embedding | Token → Vector | Subword Tokenization |
| Encoding | 序列编码 | Transformer Encoder |
| Matching | Q-P 交互 | Cross-Attention |
| Fusion | 信息融合 | Self-Attention |
| Decoding | Span 预测 | Linear + Softmax |

## PyTorch 实现

### 完整模型

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class MRCModel(nn.Module):
    """基于 Transformer 的 MRC 模型"""
    
    def __init__(
        self, 
        model_name: str = "bert-base-chinese",
        dropout: float = 0.1,
        max_answer_length: int = 30
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.max_answer_length = max_answer_length
        
        self.dropout = nn.Dropout(dropout)
        self.start_fc = nn.Linear(hidden_size, 1)
        self.end_fc = nn.Linear(hidden_size, 1)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        start_positions: torch.Tensor = None,
        end_positions: torch.Tensor = None,
    ):
        # 编码
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = self.dropout(outputs.last_hidden_state)
        
        # 预测 start/end
        start_logits = self.start_fc(sequence_output).squeeze(-1)
        end_logits = self.end_fc(sequence_output).squeeze(-1)
        
        # Mask padding
        mask = attention_mask.bool()
        start_logits = start_logits.masked_fill(~mask, float('-inf'))
        end_logits = end_logits.masked_fill(~mask, float('-inf'))
        
        # 计算损失
        loss = None
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2
        
        return {
            'loss': loss,
            'start_logits': start_logits,
            'end_logits': end_logits,
        }
    
    def decode(
        self,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """解码最佳答案 span"""
        batch_size, seq_len = start_logits.shape
        
        # 计算所有有效 (start, end) 对的分数
        start_probs = F.softmax(start_logits, dim=-1)
        end_probs = F.softmax(end_logits, dim=-1)
        
        results = []
        for b in range(batch_size):
            best_score = float('-inf')
            best_start, best_end = 0, 0
            
            for start in range(seq_len):
                if not attention_mask[b, start]:
                    continue
                for end in range(start, min(start + self.max_answer_length, seq_len)):
                    if not attention_mask[b, end]:
                        continue
                    score = start_probs[b, start] * end_probs[b, end]
                    if score > best_score:
                        best_score = score
                        best_start, best_end = start, end
            
            results.append((best_start, best_end))
        
        return results
```

### 数据处理

```python
from dataclasses import dataclass
from typing import List, Optional
import json

@dataclass
class MRCExample:
    qid: str
    question: str
    context: str
    answer: Optional[str] = None
    start_position: Optional[int] = None

@dataclass
class MRCFeature:
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]
    start_position: int
    end_position: int
    offset_mapping: List[tuple]

class MRCProcessor:
    def __init__(self, model_name: str, max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
    
    def process(self, example: MRCExample) -> MRCFeature:
        encoding = self.tokenizer(
            example.question,
            example.context,
            max_length=self.max_length,
            truncation='only_second',
            return_offsets_mapping=True,
            padding='max_length',
        )
        
        # 定位答案位置
        start_token, end_token = 0, 0
        if example.start_position is not None:
            offset = encoding['offset_mapping']
            for idx, (start, end) in enumerate(offset):
                if start <= example.start_position < end:
                    start_token = idx
                if start < example.start_position + len(example.answer) <= end:
                    end_token = idx
                    break
        
        return MRCFeature(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask'],
            token_type_ids=encoding.get('token_type_ids', [0] * len(encoding['input_ids'])),
            start_position=start_token,
            end_position=end_token,
            offset_mapping=encoding['offset_mapping'],
        )
```

### 训练循环

```python
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(**batch)
        loss = outputs['loss']
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                token_type_ids=batch.get('token_type_ids'),
            )
            
            spans = model.decode(
                outputs['start_logits'],
                outputs['end_logits'],
                batch['attention_mask'],
            )
            predictions.extend(spans)
    
    return predictions

# 主训练流程
def main():
    # 配置
    model_name = "bert-base-chinese"
    batch_size = 16
    learning_rate = 3e-5
    num_epochs = 3
    
    # 初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MRCModel(model_name).to(device)
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=num_epochs * len(train_dataloader),
    )
    
    # 训练
    for epoch in range(num_epochs):
        loss = train_epoch(model, train_dataloader, optimizer, scheduler, device)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
        
        # 验证
        predictions = evaluate(model, val_dataloader, device)
        f1 = compute_f1(predictions, val_labels)
        print(f"Validation F1: {f1:.4f}")
```

## 评估指标

```python
import re
import string
from collections import Counter

def normalize_answer(s: str) -> str:
    """标准化答案文本"""
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    s = ' '.join(s.split())
    return s

def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    
    if not pred_tokens or not gold_tokens:
        return int(pred_tokens == gold_tokens)
    
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    
    if precision + recall == 0:
        return 0
    
    return 2 * precision * recall / (precision + recall)

def compute_em(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))
```

## 与现代方法对比

| 方面 | 经典 MRC (BiDAF) | BERT-based | LLM-based |
|------|-----------------|------------|-----------|
| 参数量 | ~2M | 110M-340M | 7B-70B+ |
| 训练数据 | Task-specific | 预训练+微调 | 大规模预训练 |
| 推理方式 | Span extraction | Span extraction | Generation |
| 长文档 | 需要切分 | 需要切分 | 更大上下文窗口 |
| 多跳推理 | 困难 | 有限 | 较好 |

## 生产环境优化

### 量化推理

```python
import torch.quantization as quant

# 动态量化
model_int8 = quant.quantize_dynamic(
    model.cpu(),
    {nn.Linear},
    dtype=torch.qint8
)
```

### ONNX 导出

```python
import torch.onnx

dummy_input = {
    'input_ids': torch.ones(1, 512, dtype=torch.long),
    'attention_mask': torch.ones(1, 512, dtype=torch.long),
    'token_type_ids': torch.zeros(1, 512, dtype=torch.long),
}

torch.onnx.export(
    model,
    (dummy_input['input_ids'], dummy_input['attention_mask'], dummy_input['token_type_ids']),
    "mrc_model.onnx",
    input_names=['input_ids', 'attention_mask', 'token_type_ids'],
    output_names=['start_logits', 'end_logits'],
    dynamic_axes={
        'input_ids': {0: 'batch', 1: 'seq'},
        'attention_mask': {0: 'batch', 1: 'seq'},
        'token_type_ids': {0: 'batch', 1: 'seq'},
    }
)
```

## 延伸阅读

- [Hugging Face Question Answering](https://huggingface.co/docs/transformers/task_summary#question-answering)
- [SQuAD Leaderboard](https://rajpurkar.github.io/SQuAD-explorer/)
- [CMRC 2018 中文阅读理解](https://github.com/ymcui/cmrc2018)

---

> 转载请注明出处

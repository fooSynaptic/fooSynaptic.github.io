---
title: "如何教会机器理解问题和文本并回答问题（TensorFlow 实战）"
date: 2019-11-19T13:16:27+08:00
tags:
  - Machine reading comprehension
  - tensorflow
---

这篇文章主要阐述机器阅读理解的实现。

## 输入

- 问题 q
- 与问题 q 相关的文档集合（利用 symbolic matching 做初步召回）
- 标注好的文档中的起始和结束位点

## 需要解决的问题

1. 如何找到一个映射，将文档和问题从各自的空间映射到二维整数空间？（神经网络）
2. 如何在向量空间中表示问题和文档？
3. 什么样的损失函数是好的损失？
4. 如果答案存在于不连续的区间，如何解决？
5. 如果需要生成答案而不是直接抽取，如何解决？

## 网络架构

1. Embedding Layer
2. Matching Layer
3. Fusion Layer
4. Decoding Layer

## Embedding and Encoding Layers

Embedding 和 Encoding 层输入 token 序列，输出向量序列。使用预训练的 embedding matrix 和双向 GRU：

```python
embed_shape = [vocab_size, vocab_embed_dim]
embed_placeholder = tf.placeholder(tf.float32, embed_shape)
word_embed = tf.get_variable("word_embeddings", embed_shape, trainable=False)

embed_init_op = word_embed.assign(embed_placeholder)

# 从 numpy 数组加载预计算的 embedding
with tf.Session() as sess:
    sess.run(embed_init_op, feed_dict={embed_placeholder: pre_embed})
```

**编码问题和文档：**

```python
q_emb = tf.nn.embedding_lookup(word_embed, q)

with tf.variable_scope("Question_Encoder"):
    cell_fw = GRUCell(num_units=hidden_size)
    cell_bw = GRUCell(num_units=hidden_size)
    output, _ = tf.nn.bidirectional_dynamic_rnn(
        cell_fw, cell_bw, q_emb, sequence_length=q_len)
    
    # 拼接前向和后向编码信息
    q_encodes = tf.concat(output, 2)

# 对 passage 做同样处理得到 p_encodes
```

## Matching Layer

```python
p_mask = tf.sequence_mask(p_len, tf.shape(p)[1], dtype=tf.float32, name="passage_mask")
q_mask = tf.sequence_mask(q_len, tf.shape(q)[1], dtype=tf.float32, name="question_mask")

sim_matrix = tf.matmul(p_encodes, q_encodes, transpose_b=True)
sim_mask = tf.matmul(tf.expand_dims(p_mask, -1), 
                     tf.expand_dims(q_mask, -1), transpose_b=True)

# 用极小数掩盖零值
sim_matrix -= (1 - sim_mask) * 1E30

passage2question_attn = tf.matmul(tf.nn.softmax(sim_matrix, -1), q_encodes)
b = tf.nn.softmax(tf.expand_dims(tf.reduce_max(sim_matrix, 2), 1), -1)

question2passage_attn = tf.tile(tf.matmul(b, p_encodes), 
                                [1, tf.shape(p_encodes)[1], 1])

p_mask = tf.expand_dims(p_mask, -1)
passage2question_attn *= p_mask
question2passage_attn *= p_mask

match_out = tf.concat([
    p_encodes,
    p_encodes * passage2question_attn,
    p_encodes * question2passage_attn
], -1)
```

## Fusion Layer

Fusion Layer 的目的是：
1. 获取 match_out 中的长程依赖
2. 获取尽可能多的信息为 decoding 阶段做准备

**方法：**
- 将 match_out 作为双向 RNN 的输入
- 使用 CNN，用多个 conv1d 与 match_out 进行交叉相关

```python
# 使用 CNN
out_dim = 64
window_len = 10

conv_match = tf.layers.conv1d(match_out, out_dim, window_len, strides=window_len)
conv_match_up = tf.squeeze(tf.image.resize_images(
    tf.expand_dims(conv_match, axis=-1),
    [tf.shape(match_out)[1], out_dim],
    method=ResizeMethod.NEAREST_NEIGHBOR), axis=-1)

fuse_out = tf.concat([p_encodes, match_out, conv_match_up], axis=-1)
```

> 上采样步骤是为了将卷积特征与 match_out 和 p_encodes 拼接。fuse_out 的大小是 [B, L, D]，其中 B 是 batch size，L 是 passage 长度，D 是 fusion layer 中卷积 filter 控制的深度。

## Decoding Layer & Loss Function

将 fuse_out 解码为答案片段。简单方法是用 dense layer 将 fuse_out 的最后一维降到 1，然后用 softmax：

```python
start_logit = tf.layers.dense(fuse_out, 1)
end_logit = tf.layers.dense(fuse_out, 1)

# 在 softmax 之前掩盖 padding 符号
start_logit -= (1 - p_mask) * 1E30
end_logit -= (1 - p_mask) * 1E30

# 计算损失
start_loss = tf.losses.sparse_softmax_cross_entropy(
    labels=start_label, logits=start_logit)
end_loss = tf.losses.sparse_softmax_cross_entropy(
    labels=end_label, logits=end_logit)
loss = (start_loss + end_loss) / 2
```

## 生成最终答案

```python
max_answ_len = 50

start_prob = tf.nn.softmax(start_logit, axis=1)
end_prob = tf.nn.softmax(end_logit, axis=1)

# 计算外积
outer = tf.matmul(tf.expand_dims(start_prob, axis=2),
                  tf.expand_dims(end_prob, axis=1))
outer = tf.matrix_band_part(outer, 0, max_answ_len)

start_pos = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
end_pos = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)

# 从原始 passage 中提取答案
final_answer = passage_tokens[start_pos: end_pos + 1]
```

## 参考

- [Teach Machine to Comprehend Text and Answer Question with TensorFlow](http://hanxiao.io/2018/04/21/Teach-Machine-to-Comprehend-Text-and-Answer-Question-with-Tensorflow/)

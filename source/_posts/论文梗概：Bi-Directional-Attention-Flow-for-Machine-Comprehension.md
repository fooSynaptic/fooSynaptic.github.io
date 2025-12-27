---
title: "论文梗概：Bi-Directional Attention Flow for Machine Comprehension"
date: 2019-11-19T16:47:58+08:00
tags:
  - MRC
  - bidaf
  - dynamic attention vs memory-less attention
---

# Abstract insights:

  * requires modeling complex interactions between the context and the query.
  * use attention to focus on a small portion of the context and summarize it with a fixed-size of vector
  * muliti-stage hierachical process that represents the context at different levels of granularity and uses bi-directional attention flow mechanism to obtain a query-aware context representation without early summarization.



# Introduction insights:

## Bi-directional attention flow:

  * First: the attention layer Is not used to summarize the context paragraph into a fixed-size vector. Instead, the attention is computed for every time step, and the attended vector at each time step, along with the representations form previous layers, is allowed to flow through to the subsequent modeling layer.(how similar to self-attention), prevent the information loss by early summary.
  * Second, we use a memory-less attention mechanism. Thai is while we iteratively compute attention through time, the attention at each time step is a function of only the query and the context paragraph at the current time step and does not directly depend on the attention at the previous time step.  
Second mechanism forces the attention layer to focus on learning the attention between the query and the context, and enables the modeling layer to focus on learning the interaction within the query-aware context representation( the output of the attention layer). It also allows the attention at each time step to be unaffected from incorrect attendances at previous time steps.



### keynotes:

Conventional dynamic attention: the attention weights at the current time step are a function of the attended vector at the previous time step.  
bidaf: the attention is a computed for every time step, and the attended vector at each time step. The memeory-less attention: the attention at each time step is a function of only the query and the context paragraph at the current time step and doses not directly depend on the attention at the previous time step.

_The author claim the memory-less attention gives a clear advanatge over dynamic attention._

  * Third: the bi-direction provide complimentary information to each other.



# BiDAF network Architecture:

First three layers, computing features from the query and context at different levels of granularity, akin to the multi-stage feature computation of convolutional NN in the CV field.

  * character-level
  * word-level
  * contextual embedding： utilizes contextual cues from surrounding words to refine the embedding of the words.  
::=>We use a LSTM on top of the embeddings provided by the previous layers to model the temporal interactions between words. We place an LSTM in both directions and concatenate the outputs of the two LSTMs,


  * Attention Flow Layer: couples the query and context vectors and produces a set of query-aware feature vectors for each word in the context.  
α(h, u) = W[h;u;h@u]  
Context-to-query attention. C2q attenton signifies which query words are most relevant to each context word.  
Query-to-context(Q2C) attention signifies wich context words have the closest similarity to one of the query words and are hence critical for answering the query.


  * Modeling Layer: employs a Recurrent Neural Network to scan the context.
  * output Layer: provides an answer to the query(task oriented realization).



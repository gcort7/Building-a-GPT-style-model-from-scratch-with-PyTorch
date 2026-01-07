# Minimal GPT-Style Transformer From Scratch (PyTorch)

This repository contains a **from-scratch implementation of a decoder-only Transformer**, inspired by GPT-style language models.

The goal of this project is **educational clarity**, not maximal performance or architectural novelty. Every major component of a modern autoregressive Transformer is implemented explicitly using PyTorch.

---

## ✨ What This Project Is

- A **decoder-only Transformer** (GPT-style)
- Trained with **next-token prediction**
- Uses **causal self-attention**
- Trained on a **continuous stream of tokens**
- Built end-to-end:
  - Custom WordPiece tokenizer
  - Token + positional embeddings
  - Multi-head self-attention
  - Feed-forward networks
  - Autoregressive text generation

The model is trained on the **Tiny Shakespeare** dataset.

---

## 🧠 Model Architecture Overview

```
Token IDs
   ↓
Token Embedding + Positional Encoding
   ↓
[N × Decoder Blocks]
   ↓
Linear Projection (Language Modeling Head)
   ↓
Softmax → Next-Token Probabilities
```

Each decoder block contains:
1. Masked multi-head self-attention
2. Residual connection + LayerNorm
3. Position-wise feed-forward network
4. Residual connection + LayerNorm

---

## 🔹 GPT-Style Training (Token Streams)

The model is trained on **continuous token streams**, not sentences.

- No sentence boundaries
- Random sliding windows over a long token sequence
- Input:  `[t₀, t₁, ..., tₙ₋₁]`
- Target: `[t₁, t₂, ..., tₙ]`

This matches how GPT models are trained and how text is generated at inference time.

---

## 🔹 Token Embeddings (Not Word2Vec)

Token embeddings are implemented as a simple lookup table:

```python
nn.Embedding(vocab_size, d_model)
```

Key points:
- Embeddings are **randomly initialized**
- Each row corresponds to a token ID
- There is **no explicit semantic objective**
- Semantic structure emerges **indirectly** through next-token prediction

Tokens with similar meanings may end up close in embedding space after training, but this is an emergent property—not something directly supervised.

---

## 🔹 No Sentence Awareness

The model has no explicit concept of:
- Sentences
- Grammar
- Syntax rules

It learns only from statistical regularities in token sequences.

---

## 🔹 Positional Encoding

Sinusoidal positional encodings are added to token embeddings to inject order information:

```python
x = token_embeddings + positional_encodings
```

---

## 🔹 Multi-Head Self-Attention

Each attention layer:
- Projects inputs into Q, K, V
- Splits into multiple heads
- Applies scaled dot-product attention
- Uses a **causal mask** to prevent looking ahead
- Concatenates heads and projects back to `d_model`

All tokens in a sequence are processed **in parallel during training**.

---

## 🔹 Parallelism in Transformers

Training is highly parallelizable because:
- All tokens in a sequence are processed simultaneously
- Causality is enforced with masking, not sequential computation
- Feed-forward layers operate independently per token

Sequential processing only occurs during **autoregressive generation**, one token at a time.

---

## 🚀 Text Generation

The model generates text autoregressively:
1. Predict next-token logits
2. Apply temperature and optional top-k sampling
3. Sample the next token
4. Append it to the sequence
5. Repeat

---

## 🧪 Dataset

- **Tiny Shakespeare**
- Loaded via Hugging Face `datasets`
- Tokenized using a custom WordPiece tokenizer

---

## 🎯 Purpose

This repository is meant to:
- Demystify GPT-style Transformers
- Serve as a learning reference
- Provide a clean, readable implementation

Not intended for production or benchmark performance.

---
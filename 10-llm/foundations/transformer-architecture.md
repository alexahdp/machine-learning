# Transformer Architecture

## Table of Contents

1. [Motivation: Why Replace RNNs?](#motivation-why-replace-rnns)
2. [High-Level Architecture](#high-level-architecture)
3. [Encoder Block](#encoder-block)
4. [Decoder Block](#decoder-block)
5. [Layer Normalization: Pre-LN vs Post-LN](#layer-normalization-pre-ln-vs-post-ln)
6. [Feed-Forward Network](#feed-forward-network)
7. [Residual Connections](#residual-connections)
8. [Full Encoder-Decoder Flow](#full-encoder-decoder-flow)
9. [Decoder-Only (GPT-Style) Variant](#decoder-only-gpt-style-variant)
10. [Parameter Count Analysis](#parameter-count-analysis)
11. [Why Transformers Parallelize Well](#why-transformers-parallelize-well)
12. [Common Hyperparameters](#common-hyperparameters)
13. [References](#references)

---

## Motivation: Why Replace RNNs?

By 2017, RNNs and LSTMs were the dominant architecture for sequence modeling. They had three fundamental problems:

**1. Sequential dependency prevents parallelization.** To compute $\mathbf{h}_t$, you must first compute $\mathbf{h}_{t-1}$. The entire forward pass is a sequential chain of length $n$. Even with GPUs, you cannot exploit parallelism within a sequence.

**2. Long-range dependencies are hard to learn.** Even LSTMs with their gating mechanisms struggle when relevant context is hundreds of tokens away. Gradient signal traversing many recurrent steps either vanishes or explodes.

**3. Inductive bias toward locality.** RNNs naturally weight recent tokens more heavily (the hidden state summarizes history with compression loss). Attention allows *direct* access to any position with equal path length.

The Transformer (Vaswani et al., 2017) replaced all recurrence with attention, making the maximum path length between any two positions $O(1)$ (versus $O(n)$ for RNNs) and enabling full parallelization across the sequence dimension.

---

## High-Level Architecture

The original Transformer is an **encoder-decoder** model:

- **Encoder**: Maps input tokens to a sequence of continuous representations.
- **Decoder**: Autoregressively generates output tokens, conditioned on the encoder output and previously generated tokens.

Both encoder and decoder are stacks of identical blocks (the original paper used $N = 6$ each).

---

## Encoder Block

Each encoder block applies two sublayers in sequence, each wrapped with a residual connection and layer normalization.

**Sublayer 1: Multi-Head Self-Attention (MHSA)**

$$\mathbf{Z} = \text{LayerNorm}(\mathbf{X} + \text{MultiHead}(\mathbf{X}, \mathbf{X}, \mathbf{X}))$$

All positions attend to all other positions in the same sequence. There is no causal mask — the encoder is bidirectional.

**Sublayer 2: Position-wise Feed-Forward Network (FFN)**

$$\mathbf{Y} = \text{LayerNorm}(\mathbf{Z} + \text{FFN}(\mathbf{Z}))$$

The same FFN is applied independently to each position. This is the "position-wise" qualifier — weights are shared across positions but not across layers.

**Single encoder block:**

```
Input X [n × d_model]
    |
    +──────────────────┐
    |                  |
[MHSA]          (residual)
    |                  |
    └──── Add ─────────┘
              |
         [LayerNorm]
              |
    ┌─────────┴────────────┐
    |                      |
  [FFN]             (residual)
    |                      |
    └──── Add ─────────────┘
              |
         [LayerNorm]
              |
    Output [n × d_model]
```

---

## Decoder Block

The decoder has three sublayers per block:

**Sublayer 1: Masked Multi-Head Self-Attention**

$$\mathbf{Z}_1 = \text{LayerNorm}(\mathbf{Y} + \text{MaskedMultiHead}(\mathbf{Y}, \mathbf{Y}, \mathbf{Y}))$$

Causal masking ensures each position only attends to positions $\leq i$. This maintains the autoregressive property: position $i$ cannot see future tokens.

**Sublayer 2: Cross-Attention (Encoder-Decoder Attention)**

$$\mathbf{Z}_2 = \text{LayerNorm}(\mathbf{Z}_1 + \text{MultiHead}(\mathbf{Z}_1, \mathbf{H}_{\text{enc}}, \mathbf{H}_{\text{enc}}))$$

Queries come from the decoder; Keys and Values come from the encoder's final output $\mathbf{H}_{\text{enc}}$. This is the bridge through which the decoder reads the encoded source.

**Sublayer 3: Feed-Forward Network**

$$\mathbf{Z}_3 = \text{LayerNorm}(\mathbf{Z}_2 + \text{FFN}(\mathbf{Z}_2))$$

**Single decoder block:**

```
Decoder Input Y [m × d_model]   Encoder Output H_enc [n × d_model]
      |                                    |
      +──────────┐                         |
      |          |                         |
[Masked MHSA]  (res)                       |
      |          |                         |
      └── Add ───┘                         |
          |                               |
      [LayerNorm]                         |
          |                               |
      +───┴──────────────────────┐        |
      |                          |        |
[Cross-Attn(Q=·, K=H, V=H)] ←──────────┘
      |                          |
      └──── Add ─────────────────┘
                |
           [LayerNorm]
                |
      +─────────┴──────────────┐
      |                        |
    [FFN]                   (res)
      |                        |
      └───── Add ──────────────┘
                  |
             [LayerNorm]
                  |
        Output [m × d_model]
```

---

## Layer Normalization: Pre-LN vs Post-LN

### What Layer Norm Does

Layer normalization normalizes across the feature dimension (not the batch dimension, unlike BatchNorm):

$$\text{LN}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sigma + \epsilon} + \beta$$

where $\mu = \frac{1}{d}\sum_i x_i$, $\sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2$, and $\gamma, \beta \in \mathbb{R}^d$ are learned scale and shift parameters.

### Post-LN (Original Paper)

$$\mathbf{x}_{l+1} = \text{LN}(\mathbf{x}_l + \text{Sublayer}(\mathbf{x}_l))$$

Normalization occurs *after* the residual addition. This is what Vaswani et al. (2017) used.

**Problem**: The gradient at layer $l$ passes through all subsequent LN operations unmodified *only* via the residual stream. The LN on the summed output can cause the residual stream to have highly variable scale across layers early in training, contributing to instability. Post-LN models often require **learning rate warmup** to avoid divergence.

### Pre-LN

$$\mathbf{x}_{l+1} = \mathbf{x}_l + \text{Sublayer}(\text{LN}(\mathbf{x}_l))$$

Normalization occurs *before* the sublayer. The residual path is now clean — the gradient can flow through the residual stream without passing through any normalization.

**Benefits**:
- Training is significantly more stable; learning rate warmup is often unnecessary.
- Gradients are better conditioned at initialization.

**Drawback**: The final output representation is not normalized before the output projection, requiring a final LN after the last layer.

Pre-LN is now the dominant choice in practice (GPT-2, LLaMA, Mistral, etc.) despite not being used in the original Transformer.

### RMSNorm

A further simplification: Root Mean Square Layer Normalization (Zhang & Sennrich, 2019) drops the mean subtraction (centering) and uses only the RMS scaling:

$$\text{RMSNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}}$$

Computationally cheaper and empirically equivalent or better. Used in LLaMA, Mistral, and many modern models.

---

## Feed-Forward Network

The FFN applied to each position independently is a two-layer MLP with an intermediate expansion:

$$\text{FFN}(\mathbf{x}) = f(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$$

where $\mathbf{W}_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$, $\mathbf{W}_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$.

**Dimensions**: The original paper used $d_{\text{ff}} = 4 \times d_{\text{model}}$ (e.g., 512 → 2048 → 512). Modern models follow this ratio.

**Activation function**: The original used ReLU. Modern models predominantly use:
- **GeLU** (Gaussian Error Linear Unit): $\text{GeLU}(x) = x \cdot \Phi(x)$ where $\Phi$ is the standard normal CDF. Smooth, allows some negative values. Used in BERT, GPT-2.
- **SwiGLU** (Shazeer, 2020): Uses a gating mechanism, $\text{SwiGLU}(\mathbf{x}) = \text{Swish}(\mathbf{x}\mathbf{W}_1) \odot \mathbf{x}\mathbf{V}$. Requires a third weight matrix but delivers consistent gains. Used in LLaMA, PaLM.

**Role of the FFN**: The MHSA mixes information *across positions*. The FFN then processes each position *independently*, giving the model capacity to apply nonlinear transformations to the mixed representations. It is widely hypothesized that FFN layers serve as key-value memories (Geva et al., 2021), with each neuron pattern-matching to specific input features and contributing to factual recall.

---

## Residual Connections

Every sublayer is wrapped in:

$$\text{output} = \mathbf{x} + \text{Sublayer}(\mathbf{x})$$

This is the residual (skip) connection from He et al. (2016), adapted from ResNets.

### Why They Help

**Gradient flow**: Without residuals, gradients must pass through every layer's Jacobian in sequence. For $L$ layers, the gradient norm is the product of $L$ Jacobians, which drives gradients to zero (vanishing) or infinity (exploding). With residual connections, the gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{x}_l}$ includes a direct path via the identity:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \cdot \prod_{k=l}^{L-1}\left(1 + \frac{\partial \text{Sublayer}_k}{\partial \mathbf{x}_k}\right)$$

The $+1$ ensures the gradient contains a term that does not vanish even when sublayer Jacobians are small.

**Representation refinement**: The network learns *residuals* (corrections to the input) rather than full transformations from scratch. This is an easier optimization landscape.

**Identity initialization**: At initialization (with weights near zero), each sublayer computes approximately zero, so the stack approximates an identity function. Training starts from a meaningful point rather than a degenerate one.

---

## Full Encoder-Decoder Flow

```
Source tokens                 Target tokens (shifted right)
["Die", "Katze"]              ["<s>", "The", "cat"]
     |                                |
[Token Embedding]            [Token Embedding]
     |                                |
[+ Positional Encoding]      [+ Positional Encoding]
     |                                |
     |    ┌───────────────────────────┘
     ↓    ↓
┌─────────────────────────┐    ┌─────────────────────────┐
│   Encoder Block 1       │    │   Decoder Block 1       │
│  ┌──────────────────┐   │    │  ┌──────────────────┐   │
│  │ MHSA (no mask)   │   │    │  │ Masked MHSA      │   │
│  │ Add & LayerNorm  │   │    │  │ Add & LayerNorm  │   │
│  │ FFN              │   │    │  │ Cross-Attn ←─────────── encoder output
│  │ Add & LayerNorm  │   │    │  │ Add & LayerNorm  │   │
│  └──────────────────┘   │    │  │ FFN              │   │
│   ...                   │    │  │ Add & LayerNorm  │   │
│   Encoder Block N       │    │  └──────────────────┘   │
└───────────┬─────────────┘    │   ...                   │
            │                  │   Decoder Block N       │
            │                  └───────────┬─────────────┘
            │                              |
            │                       Linear projection
            │                       (d_model → vocab_size)
            │                              |
            │                          Softmax
            │                              |
            │                       Next token logits
            └──────────────────────────────┘
                (cross-attention keys/values)
```

At **training time**: Both encoder and decoder process the full sequences in parallel. The decoder uses teacher forcing — it receives the ground-truth target tokens (shifted right, so position $i$ receives token $i-1$) and predicts all output positions simultaneously.

At **inference time**: The decoder generates one token at a time. Each new token is appended to the target, and the full sequence is processed again (or KV cache is used for efficiency).

---

## Decoder-Only (GPT-Style) Variant

Modern large language models (GPT, LLaMA, Mistral, Falcon, etc.) drop the encoder entirely and use only the decoder stack with **causal self-attention only** (no cross-attention sublayer):

```
Input: ["The", "cat", "sat"]

[Token Embedding]
       |
[+ Positional Encoding]
       |
┌──────────────────────────┐
│  Decoder-Only Block 1    │
│  ┌───────────────────┐   │
│  │ Causal MHSA       │   │
│  │ Add & LayerNorm   │   │
│  │ FFN               │   │
│  │ Add & LayerNorm   │   │
│  └───────────────────┘   │
│   ... × N layers         │
└──────────────────────────┘
       |
  [Final LayerNorm]   ← Pre-LN models add this
       |
  Linear (d_model → vocab_size)
       |
  Softmax → P(next token | context)
```

The model is trained with the **causal language modeling objective**: predict each token from its left context only. The causal mask enforces this at training time without any architectural changes. At generation, the model autoregressively produces tokens one by one.

**Why decoder-only dominates**: Encoder-decoder models require paired input-output training data. A decoder-only model trained on raw text with CLM can learn from *any* document, making massive pretraining corpora accessible. Additionally, encoder-decoder models require re-encoding the source for every generation step (without caching), while decoder-only models need only a single forward pass direction.

---

## Parameter Count Analysis

Given:
- $d$ = `d_model` (embedding dimension)
- $h$ = number of attention heads
- $d_{\text{ff}} = 4d$ (FFN intermediate size)
- $L$ = number of layers
- $V$ = vocabulary size

### Per Transformer Layer

| Component | Parameters |
|-----------|-----------|
| Multi-head attention ($\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V, \mathbf{W}^O$) | $4d^2$ |
| FFN ($\mathbf{W}_1, \mathbf{W}_2$) | $2 \cdot d \cdot d_{\text{ff}} = 8d^2$ |
| LayerNorm $\gamma, \beta$ (×2 sublayers) | $4d$ |
| **Per layer total** | $\approx 12d^2$ |

### Total Model

$$N_{\text{params}} \approx 12Ld^2 + Vd$$

where $Vd$ is the token embedding matrix (shared with the output projection in most models — this is called **weight tying**).

**Examples:**

| Model | $L$ | $d$ | $d_{\text{ff}}$ | $V$ | Approx Params |
|-------|-----|-----|---------|-----|--------------|
| BERT-Base | 12 | 768 | 3072 | 30522 | 110M |
| GPT-2 | 12 | 768 | 3072 | 50257 | 117M |
| GPT-3 | 96 | 12288 | 49152 | 50257 | 175B |
| LLaMA-7B | 32 | 4096 | 11008 | 32000 | 7B |

**Quick sanity check for GPT-3**:
$$12 \times 96 \times 12288^2 \approx 12 \times 96 \times 1.51 \times 10^8 \approx 174\text{B}$$

---

## Why Transformers Parallelize Well

RNN computation at step $t$ requires $\mathbf{h}_{t-1}$ — a strict sequential dependency. An $n$-step sequence has a critical path of $n$ sequential operations, regardless of hardware parallelism.

Transformers have **no sequential dependency within a layer**:

1. The attention score matrix $\mathbf{Q}\mathbf{K}^\top$ is a single matrix multiplication — fully parallelizable.
2. The FFN at each position is independent of other positions.
3. Different layers are sequential, but all $n$ positions within a layer compute simultaneously.

This maps naturally to GPU SIMD architecture: matrix multiplications are the dominant operation, and GPUs can execute large matrix ops with near-peak throughput.

The tradeoff: $O(n^2)$ memory and compute in attention limits sequence length. RNNs scale linearly in sequence length but quadratically in hidden size per step ($O(d^2)$ per step). For practical sequence lengths ($n < 10{,}000$) and modern embedding sizes ($d \geq 1024$), Transformers win on both speed and quality.

---

## Common Hyperparameters

| Hyperparameter | Symbol | Typical Values | Notes |
|---------------|--------|----------------|-------|
| Model dimension | $d_{\text{model}}$ | 512, 768, 1024, 2048, 4096, 8192 | Core scaling parameter |
| Number of layers | $L$ | 6–96 | Scales approximately with $d$ |
| Number of heads | $h$ | 8–96 | Usually $d/h = 64$ or $128$ |
| Head dimension | $d_k = d/h$ | 64, 128 | 64 is standard |
| FFN width | $d_{\text{ff}}$ | $4d$ (or $8d/3$ for SwiGLU) | SwiGLU with $8d/3$ ≈ same FLOPs as $4d$ ReLU |
| Dropout | — | 0.1 (small models), 0.0 (large models) | Large models often trained without dropout |
| Vocabulary size | $V$ | 30k–100k+ | See tokenization.md |
| Attention dropout | — | 0.1 or 0.0 | Applied to attention weights |

The ratio $d_{\text{model}} / n_{\text{layers}}$ matters: too wide and shallow → poor feature hierarchy; too narrow and deep → slow to train, diminishing returns. Common practice: increase $d$ and $L$ proportionally. Kaplan et al. (2020) found that for a fixed compute budget, depth and width should scale roughly equally.

---

## References

- Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS 2017.
- He, K., et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016.
- Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). *Layer Normalization.* arXiv:1607.06450.
- Zhang, B., & Sennrich, R. (2019). *Root Mean Square Layer Normalization.* NeurIPS 2019.
- Xiong, R., et al. (2020). *On Layer Normalization in the Transformer Architecture.* ICML 2020.
- Shazeer, N. (2020). *GLU Variants Improve Transformer.* arXiv:2002.05202.
- Geva, M., et al. (2021). *Transformer Feed-Forward Layers Are Key-Value Memories.* EMNLP 2021.
- Kaplan, J., et al. (2020). *Scaling Laws for Neural Language Models.* arXiv:2001.08361.
- Touvron, H., et al. (2023). *LLaMA: Open and Efficient Foundation Language Models.* arXiv:2302.13971.

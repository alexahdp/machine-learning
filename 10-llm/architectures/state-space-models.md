# State Space Models (SSMs)

## Table of Contents

1. [Motivation: The O(n²) Bottleneck](#1-motivation-the-on-bottleneck)
2. [Classical SSM Formulation](#2-classical-ssm-formulation)
3. [Discretization](#3-discretization)
4. [S4: Structured State Space Sequence Model](#4-s4-structured-state-space-sequence-model)
5. [Mamba: Selective State Space Models](#5-mamba-selective-state-space-models)
6. [Mamba Architecture](#6-mamba-architecture)
7. [Linear Attention: Connection to SSMs](#7-linear-attention-connection-to-ssms)
8. [Hybrid Models: Jamba and Others](#8-hybrid-models-jamba-and-others)
9. [Computational Complexity](#9-computational-complexity)
10. [Practical Limitations vs Transformers](#10-practical-limitations-vs-transformers)
11. [Current Status (2024–2025)](#11-current-status-20242025)
12. [References](#references)

---

## 1. Motivation: The O(n²) Bottleneck

Standard self-attention has $O(n^2)$ computational complexity and $O(n^2)$ memory complexity in sequence length $n$ (the full attention matrix must be computed and stored). For most LLM tasks (sequence lengths in the hundreds to a few thousand), this is manageable. But for:

- **Genomics**: DNA sequences of millions of base pairs
- **Long documents**: books, codebases, full scientific papers
- **Audio**: raw waveforms at 16kHz for 60 seconds = 960K timesteps
- **Video**: raw frame sequences

the $O(n^2)$ cost becomes prohibitive.

Workarounds (sparse attention, sliding window, linear attention approximations) sacrifice expressiveness or are complex to implement efficiently. SSMs offer a fundamentally different computational paradigm: they model sequences as **linear dynamical systems** with $O(n)$ complexity in both time and memory.

The key question: can a model without explicit attention mechanisms match the quality of Transformers on language modeling tasks?

---

## 2. Classical SSM Formulation

A State Space Model describes a system that maps a 1D input signal $u(t) \in \mathbb{R}$ to a 1D output $y(t) \in \mathbb{R}$ through a latent state $h(t) \in \mathbb{R}^N$:

$$\dot{h}(t) = \mathbf{A} h(t) + \mathbf{B} u(t)$$
$$y(t) = \mathbf{C} h(t) + \mathbf{D} u(t)$$

where:
- $\mathbf{A} \in \mathbb{R}^{N \times N}$: state transition matrix (how the state evolves)
- $\mathbf{B} \in \mathbb{R}^{N \times 1}$: input projection (how input affects state)
- $\mathbf{C} \in \mathbb{R}^{1 \times N}$: output projection (how state produces output)
- $\mathbf{D} \in \mathbb{R}$: direct feedthrough (skip connection, usually $\mathbf{D} = 0$ or treated as a residual)

This is the **continuous-time** formulation from control theory. The state $h(t)$ acts as a compressed memory of the entire input history up to time $t$.

For sequences (discrete inputs), we need to discretize this system.

### The Problem with Naive SSMs

Naively, $\mathbf{A}$ is an $N \times N$ matrix that must be learned. For a model to have long-range memory, $N$ must be large, making $\mathbf{A}$ expensive to compute with and hard to train (vanishing/exploding gradients, analogous to the RNN problem).

---

## 3. Discretization

To apply an SSM to a discrete sequence $(u_1, u_2, \ldots, u_n)$, we discretize using a step size $\Delta$ (the timescale of each input):

Using the **Zero-Order Hold (ZOH)** method:

$$\bar{\mathbf{A}} = e^{\mathbf{A}\Delta}$$
$$\bar{\mathbf{B}} = (\mathbf{A})^{-1}(e^{\mathbf{A}\Delta} - \mathbf{I})\mathbf{B} \approx \Delta \mathbf{B}$$

The discrete recurrence:

$$h_t = \bar{\mathbf{A}} h_{t-1} + \bar{\mathbf{B}} u_t$$
$$y_t = \mathbf{C} h_t$$

This is a **linear recurrence** — analogous to an LSTM but without the nonlinear gates. For the full sequence:

$$y_t = \mathbf{C} \bar{\mathbf{B}} u_t + \mathbf{C} \bar{\mathbf{A}} \bar{\mathbf{B}} u_{t-1} + \mathbf{C} \bar{\mathbf{A}}^2 \bar{\mathbf{B}} u_{t-2} + \ldots$$

This is a **convolution** of the input with the kernel $\mathbf{k} = (\mathbf{C}\bar{\mathbf{B}}, \mathbf{C}\bar{\mathbf{A}}\bar{\mathbf{B}}, \mathbf{C}\bar{\mathbf{A}}^2\bar{\mathbf{B}}, \ldots)$:

$$y = \mathbf{k} * u, \quad \mathbf{k}_j = \mathbf{C}\bar{\mathbf{A}}^j\bar{\mathbf{B}}$$

**This dual view is crucial**:
- **Recurrent mode**: $O(n \cdot N)$ per sequence, $O(N)$ memory (useful for inference — sequential generation)
- **Convolutional mode**: compute $\mathbf{k}$ once, then convolve with FFT in $O(n \log n)$ (useful for training — full parallelism)

SSMs can therefore be trained like CNNs (with global convolution) and deployed like RNNs (with constant-memory recurrence). This is their central practical advantage.

---

## 4. S4: Structured State Space Sequence Model

S4 (Gu et al., 2021) is the model that brought SSMs to deep learning for long-range sequence modeling. It solves two key problems:

### Problem 1: Efficient $\bar{\mathbf{A}}^j$ Computation

Computing $\mathbf{k}_j = \mathbf{C}\bar{\mathbf{A}}^j\bar{\mathbf{B}}$ for all $j = 0, \ldots, n-1$ naively costs $O(n N^2)$ (matrix-vector product at each step). S4 makes $\mathbf{A}$ **diagonal** (or near-diagonal), so powers $\bar{\mathbf{A}}^j$ are trivially computed: $(\bar{A}_{ii})^j$.

### Problem 2: What Values Should $\mathbf{A}$ Take?

Random initialization of $\mathbf{A}$ leads to unstable training — too large eigenvalues cause exponential growth; too small cause rapid forgetting of distant inputs. S4 addresses this with **HiPPO initialization** (High-order Polynomial Projection Operator).

HiPPO theory constructs $\mathbf{A}$ such that the state $h(t)$ optimally approximates the history of the input $u(t)$ via Legendre polynomial projections:

$$(\text{HiPPO})_{nk} = \begin{cases} -(2n+1)^{1/2}(2k+1)^{1/2} & \text{if } n > k \\ n+1 & \text{if } n = k \\ 0 & \text{if } n < k \end{cases}$$

The HiPPO matrix is dense, but S4 approximates it with a **diagonal-plus-low-rank (DPLR)** structure, enabling efficient computation. Specifically, $\mathbf{A} = \Lambda - PQ^*$ where $\Lambda$ is diagonal and $P, Q \in \mathbb{C}^{N \times 1}$.

### S4 for Language Modeling

S4 handles multi-dimensional inputs (token embeddings of dimension $d$) by applying $d$ independent SSMs in parallel, one per dimension. The state matrix $\mathbf{A}$ is shared across all dimensions (tied parameters), reducing the parameter count.

S4 demonstrates competitive perplexity with Transformers on certain long-range tasks (Long Range Arena benchmark) and outperforms Transformers on tasks requiring very long context (sequence lengths 1K-16K). However, it lags on language modeling benchmarks at the same scale.

### S4's Core Limitation: No Content-Based Reasoning

The parameters $\mathbf{A}, \mathbf{B}, \mathbf{C}, \Delta$ in S4 are **fixed** (input-independent). The same filter is applied to every token in every sequence. This means the model cannot:
- Recall specific content from the context (the copy task: output a specific earlier token)
- Selectively focus on or ignore parts of the input based on what those tokens contain

Transformers can do this trivially via attention: a query directly matches against keys. S4 applies a learned but fixed convolution kernel to every sequence.

---

## 5. Mamba: Selective State Space Models

Mamba (Gu & Dao, 2023) is the key innovation that addresses S4's limitation: **selective SSMs**, where the SSM parameters depend on the input.

### Selective Parameterization

In Mamba, $\mathbf{B}$, $\mathbf{C}$, and $\Delta$ are **functions of the input** $u_t$:

$$\mathbf{B}_t = \text{Linear}(u_t), \quad \mathbf{C}_t = \text{Linear}(u_t), \quad \Delta_t = \text{softplus}(\text{Linear}(u_t))$$

The recurrence becomes input-dependent:

$$h_t = \bar{\mathbf{A}}_t h_{t-1} + \bar{\mathbf{B}}_t u_t$$
$$y_t = \mathbf{C}_t h_t$$

where $\bar{\mathbf{A}}_t = e^{\mathbf{A} \Delta_t}$ and $\bar{\mathbf{B}}_t \approx \Delta_t \mathbf{B}_t$.

**Why this enables content-based reasoning**: $\Delta_t$ controls how much the state updates based on token $t$. If $\Delta_t$ is large, the new input replaces the state (focus on current token). If $\Delta_t \approx 0$, the state is unchanged (forget current token, preserve memory). The model learns to set $\Delta_t$ based on the content of $u_t$.

This is analogous to a soft, differentiable attention mechanism but implemented via selective state update rather than a query-key dot product.

### The Training Problem: Parallelism

The original SSM recurrence is parallelizable during training (via convolution). But input-dependent parameters break this: $\bar{\mathbf{A}}_t$ now changes at each step, so you cannot precompute a fixed kernel $\mathbf{k}$.

Mamba solves this with a **hardware-aware parallel scan** algorithm (also called parallel prefix scan or associative scan). Given the recurrence:

$$h_t = \bar{\mathbf{A}}_t h_{t-1} + \bar{\mathbf{B}}_t u_t$$

this can be computed as a parallel prefix operation (like parallel prefix sum) in $O(\log n)$ depth, which maps efficiently to GPU tensor cores. The implementation avoids materializing the full $h_t$ sequence in HBM (GPU memory), using kernel fusion to keep intermediate states in SRAM.

This hardware-aware implementation is what makes Mamba practical — the mathematical formulation is not new, but efficient GPU implementation was the bottleneck.

---

## 6. Mamba Architecture

### The Mamba Block

Mamba replaces the Transformer block (Attention + FFN) with a single SSM block:

```
Input (d)
  ↓
Linear projection (d → 2·d_expand)  [split into two paths]
  ├── Path 1: Linear → SSM → ──────────────┐
  └── Path 2: Linear → SiLU (gate) ────────┤
                                           × (element-wise multiply)
  ↓
Linear projection (d_expand → d)
  ↓
Output (d)
```

The gate (Path 2) multiplied with the SSM output (Path 1) is a **selective output gate**: the model learns which parts of the SSM output to pass through based on the current input.

### Comparison to Transformer Block

```
Transformer block:
  x → LayerNorm → MultiHeadAttention → + x
    → LayerNorm → FFN → + x

Mamba block:
  x → LayerNorm → Mamba (SSM + gating) → + x
```

Mamba replaces both attention and FFN with a single module. No separate FFN sublayer (the expand/contract projections serve this role).

### Parameter Count

For hidden dim $d$, expand factor $E$ (typically $E=2$), state dim $N$ (typically $N=16$):

- Linear expand: $d \times 2Ed$
- SSM parameters: $N$ (for $\mathbf{A}$, diagonal) + $Ed \times N$ (for $\mathbf{B}$, $\mathbf{C}$, projected per-token) + $Ed$ (for $\Delta$)
- Linear contract: $Ed \times d$

SSM parameters are a small fraction of total — most parameters are in the linear projections, similar to how most Transformer parameters are in the FFN/attention matrices.

### HuggingFace Loading

```python
# Requires: pip install mamba-ssm causal-conv1d
from transformers import MambaForCausalLM, AutoTokenizer
import torch

model = MambaForCausalLM.from_pretrained(
    "state-spaces/mamba-2.8b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")

inputs = tokenizer("Mamba is a state space model that", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 7. Linear Attention: Connection to SSMs

### Standard vs Linear Attention

Standard attention:

$$\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

The softmax makes this $O(n^2)$: the full $n \times n$ attention matrix must be formed.

Linear attention replaces softmax with a kernel function $\phi$:

$$\text{LinAttn}(\mathbf{Q}, \mathbf{K}, \mathbf{V})_t = \frac{\phi(\mathbf{q}_t)^T \sum_{s \leq t} \phi(\mathbf{k}_s) \mathbf{v}_s^T}{\phi(\mathbf{q}_t)^T \sum_{s \leq t} \phi(\mathbf{k}_s)}$$

By the associative property of matrix multiplication, define $\mathbf{S}_t = \sum_{s \leq t} \phi(\mathbf{k}_s) \mathbf{v}_s^T$ (an outer product accumulator), then:

$$\mathbf{S}_t = \mathbf{S}_{t-1} + \phi(\mathbf{k}_t) \mathbf{v}_t^T$$
$$\text{output}_t = \frac{\phi(\mathbf{q}_t)^T \mathbf{S}_t}{\phi(\mathbf{q}_t)^T \mathbf{z}_t}$$

This is exactly an SSM recurrence: $\mathbf{S}_t$ is the hidden state, updated linearly at each step. Linear attention is therefore a special case of SSM with state dim $N = d_k \times d_v$ and specific parameter structure.

### The Kernel Function $\phi$

Common choices:
- **ELU + 1**: $\phi(x) = \text{ELU}(x) + 1$ (must be non-negative for $\mathbf{S}$ to be interpretable as an attention distribution)
- **Feature map from random Fourier features** (Performer: Choromanski et al., 2020)
- **Taylor expansion of softmax** (approximation with controllable error)

### Why Linear Attention Underperforms

Linear attention loses the sharp, sparse attention patterns that softmax provides. With softmax, the model can concentrate nearly all attention weight on a single position. With a linear kernel, the attention is a diffuse weighted average. Tasks requiring **retrieval** (find the specific relevant token) degrade significantly with linear attention.

This same limitation affects pure SSMs: without input-dependent selection, they cannot perform precise retrieval. Mamba's input-dependent $\mathbf{B}_t, \mathbf{C}_t$ partially recover this ability.

---

## 8. Hybrid Models: Jamba and Others

Given that pure SSMs underperform Transformers on some tasks and pure Transformers are $O(n^2)$, a natural approach is **hybrid models** that interleave SSM and attention layers.

### Jamba (AI21 Labs, 2024)

Jamba interleaves Mamba layers and Transformer (attention) layers:

```
Layer 1:  Mamba
Layer 2:  Mamba
Layer 3:  Attention
Layer 4:  Mamba
...
```

The attention layers handle tasks requiring precise retrieval; the Mamba layers handle long-range context integration efficiently. Additional MoE layers are incorporated for capacity.

Jamba-1.5 (52B total, 12B active) achieves near-GPT-4 performance on many benchmarks while supporting very long contexts (256K tokens) at lower memory than pure-Transformer models of similar quality.

The ratio of Mamba to Attention layers is a hyperparameter: more Mamba layers = longer context capability, better efficiency; more Attention layers = better retrieval, better in-context learning.

### Other Hybrid Approaches

**MambaByte** (Yu et al., 2023): Mamba applied at the byte level (no tokenization). With $O(n)$ complexity, processing raw bytes becomes feasible.

**Zamba** (Zyphra, 2024): a hybrid that shares attention parameters across layers (a single attention layer reused multiple times) combined with Mamba layers. Extremely parameter-efficient.

**RWKV** (Peng et al., 2023): a different SSM-style architecture ("linear transformer") with a custom attention mechanism that is recurrent. RWKV-v5/6 achieves competitive performance with Transformers at similar scale.

---

## 9. Computational Complexity

### Time Complexity

| Model | Training | Inference (per token) |
|-------|----------|----------------------|
| Dense Transformer | $O(n^2 d)$ | $O(n d)$ (with KV cache) |
| SSM (S4, Mamba) | $O(n d N)$ via parallel scan | $O(d N)$ constant |
| Linear Attention | $O(n d^2)$ | $O(d^2)$ constant |
| Hybrid (Mamba+Attn) | $O(n^2 d \cdot r + n d N \cdot (1-r))$ | Between the two |

where $N$ is the SSM state dimension, $n$ is sequence length, $d$ is hidden dim, $r$ is fraction of attention layers.

For Mamba with $N = 16$ and typical $d = 2048$: SSM operations are $16 \times$ fewer than full attention dimensions but attention has quadratic sequence cost. The crossover point where SSM is faster than attention is approximately:

$$n > N \cdot d / d = N$$

i.e., once the sequence is longer than the state dimension $N$, SSMs are more compute-efficient. But $N=16$ while $d$ can be thousands, so in practice SSMs become competitive at much longer sequences.

### Memory Complexity

| Model | KV Cache Memory (inference) |
|-------|---------------------------|
| Transformer | $O(n \cdot d)$ growing with context |
| Mamba | $O(N \cdot d)$ constant (state dim × hidden dim) |

For a 2.8B Mamba model with $d=2560$, $N=16$: the "context memory" is $16 \times 2560 \times 4$ bytes $\approx 160KB$ per layer, regardless of sequence length. A 1M-token context requires the same memory as a 10-token context (if no attention layers are present).

This is the fundamental advantage for extremely long sequences: Mamba's memory footprint is **constant in sequence length**.

---

## 10. Practical Limitations vs Transformers

### In-Context Learning

Transformers learn in-context at GPT-3 scale: a few examples in the prompt dramatically improve performance without weight updates. SSMs and linear-attention models show significantly weaker ICL, because ICL requires the model to:

1. Retrieve the pattern from the few-shot examples (retrieval task)
2. Apply it to the new instance

Both require content-based attention to specific positions — precisely what SSMs struggle with.

### Retrieval Tasks

"Copy what I told you at the beginning of the conversation" requires pointing to a specific earlier position. Attention does this exactly (via a sharp attention distribution). SSMs compress all past context into a fixed-size state; specific earlier tokens are not directly recoverable.

Evaluation on RULER (a suite of long-context retrieval tasks) shows that even Mamba models trained on 1M context struggle with tasks that require precise retrieval from earlier in the sequence, while Transformer models handle these well.

### Training Parallelism

Transformers train with embarrassingly parallel attention (compute the full attention matrix at once). SSMs use parallel scan, which is also parallel but with more complex dependencies (sequential prefix computation). In practice, SSMs are somewhat harder to optimize on modern GPU hardware, though implementations like Mamba's CUDA kernels close much of this gap.

### Recency Bias

Because SSM state is updated at every token, recent tokens tend to have disproportionate influence on the state. Early tokens' information decays exponentially (governed by $\bar{\mathbf{A}}^{n-t}$). Transformers with full attention treat all positions equally in principle (though positional bias and attention patterns introduce their own recency effects).

---

## 11. Current Status (2024–2025)

### Where SSMs Stand

SSMs are a **research direction that has produced real models** (Mamba, Jamba, RWKV) but has not displaced Transformers as the dominant architecture for LLMs. The current state:

**What SSMs have demonstrated**:
- Competitive perplexity with Transformers at the 1B-7B scale when trained on similar data
- Dramatically better inference efficiency for very long sequences ($n > 10K$)
- Constant memory consumption regardless of sequence length
- Viable alternative for specialized applications (audio, genomics, time series)

**Where SSMs still lag**:
- In-context learning at few-shot tasks
- Precise retrieval tasks ("what was the value I told you 10K tokens ago?")
- Benchmark performance vs best Transformers of similar total training compute at large scale
- Ecosystem maturity (libraries, tooling, hardware optimization)

**Hybrid models (Mamba + Attention) are the practical near-term direction**: combining Mamba's $O(n)$ efficiency for bulk processing with sparse attention for retrieval tasks. Jamba, Zamba, and similar hybrid models show that you can get most of the efficiency of SSMs while retaining most of the quality of Transformers.

### Open Research Questions

- Can Mamba-style selective SSMs close the ICL gap with Transformers at scale?
- What is the optimal Mamba:Attention layer ratio for hybrid models?
- Can hardware evolve to make SSMs comparably efficient to Transformers on training?
- Does Mamba-2 (state space duality: SSMs formulated as structured masked attention) unlock better scaling?

**Mamba-2** (Dao & Gu, 2024) reformulates the SSM recurrence as a form of structured masked attention, proving a theoretical equivalence between certain SSMs and linear attention. This "state space duality" may allow more efficient training and better GPU utilization.

---

## References

- Gu, A., Goel, K., Re, C. (2021). *Efficiently Modeling Long Sequences with Structured State Spaces*. ICLR 2022. [arXiv:2111.00396](https://arxiv.org/abs/2111.00396)
- Gu, A., Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- Dao, T., Gu, A. (2024). *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*. ICML 2024. [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)
- Lieber, O., et al. (2024). *Jamba: A Hybrid Transformer-Mamba Language Model*. [arXiv:2403.19887](https://arxiv.org/abs/2403.19887)
- Katharopoulos, A., et al. (2020). *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*. ICML 2020. [arXiv:2006.16236](https://arxiv.org/abs/2006.16236)
- Peng, B., et al. (2023). *RWKV: Reinventing RNNs for the Transformer Era*. EMNLP 2023. [arXiv:2305.13048](https://arxiv.org/abs/2305.13048)
- Gu, A., et al. (2020). *HiPPO: Recurrent Memory with Optimal Polynomial Projections*. NeurIPS 2020. [arXiv:2008.07669](https://arxiv.org/abs/2008.07669)
- Hsieh, C.P., et al. (2024). *RULER: What's the Real Context Size of Your Long Context Language Models?* [arXiv:2404.06654](https://arxiv.org/abs/2404.06654)

---

*[← Mixture of Experts](./mixture-of-experts.md) | [← Back to Architectures](./README.md)*

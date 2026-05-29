# Positional Encoding

## Table of Contents

1. [Why Position Information Is Needed](#why-position-information-is-needed)
2. [Absolute vs. Relative Position Encoding](#absolute-vs-relative-position-encoding)
3. [Sinusoidal Positional Encoding](#sinusoidal-positional-encoding)
4. [Learned Absolute Positional Encoding](#learned-absolute-positional-encoding)
5. [Relative Positional Encoding (Shaw et al.)](#relative-positional-encoding-shaw-et-al)
6. [Rotary Position Embedding (RoPE)](#rotary-position-embedding-rope)
7. [ALiBi (Attention with Linear Biases)](#alibi-attention-with-linear-biases)
8. [Comparison Table](#comparison-table)
9. [Long-Context Extension and Why PE Matters](#long-context-extension-and-why-pe-matters)
10. [References](#references)

---

## Why Position Information Is Needed

The Transformer's self-attention operation computes, for each pair of positions $(i, j)$:

$$e_{ij} = \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_k}}$$

This score depends only on the *content* of positions $i$ and $j$ — there is no notion of distance or order. If you permute the input sequence, the attention scores between any two tokens are unchanged (their relative positions change, but the Q/K dot products do not).

Formally, self-attention is **permutation equivariant**: for any permutation $\sigma$,

$$\text{Attention}(\sigma(\mathbf{X})) = \sigma(\text{Attention}(\mathbf{X}))$$

This is catastrophic for language: "The cat ate the mouse" and "The mouse ate the cat" would produce identical attention outputs for every token.

**Positional encoding** breaks this symmetry by injecting position information into the representations before (or during) attention computation.

---

## Absolute vs. Relative Position Encoding

**Absolute PE**: Each position $i$ is assigned a vector $\mathbf{p}_i$ that encodes its absolute index. Position $i=0$ always gets the same encoding regardless of context or sequence length.

**Relative PE**: The representation of a position encodes its *distance to other positions* rather than its absolute index. Position $i$ in a sequence of length 10 and position $i$ in a sequence of length 1000 may get different treatment when attending to tokens at distance 5.

Relative PE is better motivated for language: what typically matters is the distance between words ("adjacent," "within the same clause"), not their absolute position in the document. Absolute PE must learn these relative relationships implicitly.

---

## Sinusoidal Positional Encoding

Vaswani et al. (2017) proposed fixed (non-learned) positional encodings using sine and cosine functions at different frequencies:

$$\text{PE}(pos, 2i) = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$\text{PE}(pos, 2i+1) = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

where $pos$ is the position index ($0, 1, \ldots, n-1$) and $i$ is the dimension index ($0, 1, \ldots, d_{\text{model}}/2 - 1$).

The encoding is added element-wise to the token embeddings:

$$\mathbf{x}'_{pos} = \mathbf{x}_{pos} + \mathbf{PE}_{pos}$$

### Intuition: Different Frequencies

Each pair of dimensions $(2i, 2i+1)$ corresponds to a sinusoidal wave of period:

$$T_i = 2\pi \cdot 10000^{2i/d_{\text{model}}}$$

- Dimension 0/1: Period $2\pi \approx 6.28$ — completes a full cycle in ~6 positions.
- Dimension 512/513 (for $d=1024$): Period $\approx 62{,}832$ — barely changes across sequences of realistic length.

This gives the encoding a **multi-scale** structure analogous to a binary representation (where the least significant bit alternates every step, the next bit every two steps, etc.). The model can learn to extract position information at the granularity it needs for a given task.

### Why Sinusoidal?

The key property: for any fixed offset $k$, $\text{PE}_{pos+k}$ can be expressed as a linear function of $\text{PE}_{pos}$:

$$\text{PE}(pos+k, 2i) = \cos(\omega_i k)\,\text{PE}(pos, 2i) + \sin(\omega_i k)\,\text{PE}(pos, 2i+1)$$

$$\text{PE}(pos+k, 2i+1) = -\sin(\omega_i k)\,\text{PE}(pos, 2i) + \cos(\omega_i k)\,\text{PE}(pos, 2i+1)$$

where $\omega_i = 10000^{-2i/d_{\text{model}}}$. This rotation matrix is independent of $pos$ — it depends only on the offset $k$. The model can theoretically learn to extract relative position information by learning linear combinations of the absolute position encodings.

### Extrapolation

Because sinusoidal PE is a fixed function, it can be evaluated at any integer position, including positions beyond those seen in training. However, the embedding addition means that the *combined* (token + PE) representation at large positions may be out-of-distribution — the model's attention and FFN layers may not have seen those patterns. Empirically, sinusoidal PE extrapolates poorly beyond ~1.5× the training length.

---

## Learned Absolute Positional Encoding

Instead of a fixed formula, each position $i$ learns a separate embedding vector $\mathbf{p}_i$ trained jointly with the rest of the model:

$$\mathbf{x}'_i = \mathbf{x}_i + \mathbf{p}_i, \quad \mathbf{p}_i \in \mathbb{R}^{d_{\text{model}}}$$

The position embedding matrix $\mathbf{P} \in \mathbb{R}^{n_{\text{max}} \times d_{\text{model}}}$ is a learned parameter.

**Used in**: BERT, GPT-2 (both use learned absolute PE with $n_{\text{max}} = 512$ and $1024$ respectively).

### Advantages

- No assumption about the functional form of position information.
- Can learn complex position-dependent patterns that sinusoidal encoding cannot.

### Disadvantages

**Hard sequence length limit**: Positions beyond $n_{\text{max}}$ have no embedding — the model simply cannot process sequences longer than its maximum training length.

**Poor extrapolation**: Even at positions near $n_{\text{max}}$, if training data rarely contains sequences that long, those position embeddings are poorly trained. Performance typically degrades near the maximum.

**Parameter cost**: For $n_{\text{max}} = 512$, $d = 768$: adds $512 \times 768 = 393{,}216$ parameters — negligible for large models but worth noting.

---

## Relative Positional Encoding (Shaw et al.)

Shaw et al. (2018) modify the attention score computation to incorporate learned relative position biases:

$$e_{ij} = \frac{(\mathbf{x}_i\mathbf{W}^Q)(\mathbf{x}_j\mathbf{W}^K + \mathbf{a}^K_{ij})^\top}{\sqrt{d_k}}$$

$$\mathbf{z}_i = \sum_j \alpha_{ij}(\mathbf{x}_j\mathbf{W}^V + \mathbf{a}^V_{ij})$$

where $\mathbf{a}^K_{ij}$ and $\mathbf{a}^V_{ij}$ are learned vectors indexed by the clipped relative distance $\text{clip}(j - i, -k, k)$ for a maximum relative distance $k$.

**Clipping**: All positions more than $k$ steps away share the same learned vector. This limits parameters to $2k$ (not $n^2$) and forces a form of generalization.

**Problem**: Modifying the value aggregation with $\mathbf{a}^V_{ij}$ breaks the efficient matrix-product form of attention, making the implementation more complex. Later work (T5's relative PE) drops the value bias and uses a simpler bias added to the logits.

### T5's Simplified Relative PE

Raffel et al. (2020) use a shared learned scalar bias $b_{ij}$ added to attention logits:

$$e_{ij} = \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_k}} + b_{|i-j|}$$

The bias is indexed by relative distance (bucketed logarithmically), shared across all layers and heads. Zero parameters per layer (biases are in a small lookup table). T5 trains well up to ~512 tokens and extrapolates somewhat beyond training length.

---

## Rotary Position Embedding (RoPE)

RoPE (Su et al., 2021) is the most important positional encoding innovation for modern LLMs. It encodes *relative* positions via rotation matrices applied to query and key vectors — not to the token embeddings.

### Core Idea

For a 2D case, RoPE rotates a query/key vector at position $m$ by angle $m\theta$:

$$f(\mathbf{x}, m) = \begin{bmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{bmatrix} \mathbf{x}$$

The dot product between a rotated query at position $m$ and a rotated key at position $n$ then depends only on the *relative position* $m - n$:

$$f(\mathbf{q}, m) \cdot f(\mathbf{k}, n) = \mathbf{q}^\top \mathbf{R}_{m-n} \mathbf{k}$$

where $\mathbf{R}_{m-n}$ is a rotation matrix depending only on the offset $m - n$. RoPE is thus truly relative — the attention scores encode relative positions without explicitly providing them.

### General Form (High-Dimensional)

For $d$-dimensional queries and keys, the space is divided into $d/2$ independent 2D rotation planes, each with a different angular frequency $\theta_i = 10000^{-2i/d}$ (same base frequencies as sinusoidal PE):

$$\text{RoPE}(\mathbf{q}, m)_{2i:2i+2} = \begin{bmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{bmatrix} \mathbf{q}_{2i:2i+2}$$

In practice, this is efficiently computed without materializing the full rotation matrix:

$$\text{RoPE}(\mathbf{q}, m) = \mathbf{q} \odot \cos(m\boldsymbol{\theta}) + \mathbf{q}^{\perp} \odot \sin(m\boldsymbol{\theta})$$

where $\mathbf{q}^{\perp}$ is obtained by rotating pairs: $(-q_1, q_0, -q_3, q_2, \ldots)$.

### Properties and Advantages

**True relative encoding**: The attention score $\mathbf{q}_m \cdot \mathbf{k}_n$ depends only on $m - n$ (and the content of $\mathbf{q}$ and $\mathbf{k}$).

**No additional parameters**: RoPE is applied to existing Q and K projections; no new parameters.

**Sequence length flexibility**: Because RoPE applies rotations based on position indices, there is no hard maximum length at inference time. You can apply the same rotation formula to positions 0, 1, ..., 100,000 even if training only went to 4096.

**Effective extrapolation (with modifications)**: Base RoPE extrapolates reasonably to ~2× training length. For longer extrapolation, techniques like YaRN (Yet another RoPE extensioN, Peng et al., 2023) adjust the base frequency:

$$\theta'_i = 10000^{-2i/d} \cdot s$$

where $s > 1$ is a scaling factor that stretches the wavelengths, effectively "remapping" longer contexts to match the training distribution.

**Used in**: LLaMA 1/2/3, Mistral, Falcon, GPT-NeoX, PaLM 2, Gemini, and essentially all major open-weight models released after 2023.

---

## ALiBi (Attention with Linear Biases)

ALiBi (Press et al., 2022) takes a different philosophical approach: instead of modifying the input representations, add a simple position-dependent bias *directly to attention logits*:

$$e_{ij} = \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_k}} - m_h \cdot |i - j|$$

The penalty $m_h \cdot |i - j|$ increases linearly with the distance between positions. Each head $h$ uses a different slope $m_h$ (a fixed geometric sequence, e.g., $m_h \in \{1, 1/2, 1/4, \ldots, 2^{-h}\}$ for $h$ heads).

```
Attention logit matrix (head h, slope m_h = 0.5):

Position:   0     1     2     3     4
Token 0: [  0  -0.5  -1.0  -1.5  -2.0 ]
Token 1: [-0.5   0   -0.5  -1.0  -1.5 ]
Token 2: [-1.0 -0.5    0   -0.5  -1.0 ]
Token 3: [-1.5 -1.0  -0.5    0   -0.5 ]
Token 4: [-2.0 -1.5  -1.0  -0.5    0  ]
```

These biases are added to the raw Q·K scores before softmax, after dividing by $\sqrt{d_k}$.

### Why It Extrapolates

ALiBi tokens at positions beyond the training length receive logit penalties that follow the same linear pattern. The model has seen "attend to tokens 100 steps away" during training (via the bias shape), and the same bias at distance 200 is simply a stronger version of the same pattern. This is a much smoother distributional shift than trying to evaluate a learned embedding at an unseen index.

**Empirical result**: A model trained on 1024 tokens with ALiBi extrapolates to 2048+ tokens with minimal perplexity degradation. By comparison, sinusoidal and learned PE models show sharp degradation above their training length.

### Advantages

- **Zero parameters**: Slopes $m_h$ are fixed, not learned.
- **No modifications to embeddings**: Simpler to implement than RoPE.
- **Strong extrapolation**: Best extrapolation behavior of any pre-2023 method.

### Disadvantages

- **Linear bias is a strong inductive bias**: The assumption that attention strength decays linearly with distance is not always correct (e.g., coreference, global attention patterns).
- **Performance below RoPE on standard benchmarks**: At training length, models with RoPE tend to match or outperform ALiBi.
- **Head slopes must be chosen carefully**: The geometric sequence works well but is not theoretically derived.

**Used in**: BLOOM, MPT, and several other models. Less common than RoPE in 2024+ models.

---

## Comparison Table

| Method | Params | Extrapolation | Relative | Notes |
|--------|--------|--------------|----------|-------|
| Sinusoidal (Vaswani 2017) | 0 | Poor (~1.5×) | No | Original Transformer; fast to compute |
| Learned Absolute (BERT, GPT-2) | $n_{\text{max}} \times d$ | None | No | Hard limit; common in early models |
| Shaw et al. Relative PE | $2k \times d_k$ (small) | Moderate | Yes | Complex implementation; used in T5 variant |
| T5 Relative Biases | ~128 scalars | Moderate | Yes | Bucketed log-scale; simple and effective |
| RoPE (LLaMA, Mistral) | 0 | Good (~2×, >10× with YaRN) | Yes | Dominant choice 2023+ |
| ALiBi (BLOOM, MPT) | 0 | Excellent (>4×) | Yes | Simple; slightly below RoPE in quality |

---

## Long-Context Extension and Why PE Matters

Context window extension (from 4K → 128K tokens) is among the most active research areas. PE is the central obstacle.

### The Distribution Shift Problem

At training, positions $0, 1, \ldots, n_{\text{train}}$ are observed. At inference beyond $n_{\text{train}}$:
- **Learned PE**: No embedding exists for position $> n_{\text{max}}$.
- **Sinusoidal/RoPE**: Mathematically well-defined, but the network was never trained on inputs where those PE values appeared in Q/K dot products.

The attention patterns at long distances are "out-of-distribution" for the FFN and attention weights trained on shorter sequences.

### YaRN: RoPE Context Extension

YaRN (Peng et al., 2023) extends RoPE-based models to 128K+ context by:

1. **Rescaling**: Change the RoPE base frequency so that position 128K maps to where position 4K was during training (in RoPE rotation space).
2. **NTK-aware interpolation**: Mix different scaling strategies for low- and high-frequency dimensions (high-frequency dimensions handle short-range syntax and don't need stretching; low-frequency dimensions handle long-range dependencies and benefit from stretching).
3. **Fine-tuning**: Continue training for a small number of steps on longer-context documents.

The intuition: the model already learned to handle the full dynamic range of RoPE angles during training (all angles $0$ to $2\pi$). By rescaling, you map long-context inputs into the range of angles the model saw at training time.

### Practical Implications

**Choosing a PE scheme**: For new models, RoPE is the default choice. ALiBi if zero-parameter and simplicity is critical. Learned PE only for short-context specialized models.

**Evaluating long-context ability**: A model may appear to support long contexts but "forget" relevant information introduced far back in the context ("lost in the middle," Liu et al., 2023). This is a failure of attention patterns across long distances, partly PE-related and partly a training distribution issue.

---

## References

- Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS 2017.
- Shaw, P., Uszkoreit, J., & Vaswani, A. (2018). *Self-Attention with Relative Position Representations.* NAACL 2018.
- Raffel, C., et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.* JMLR 2020. (T5)
- Su, J., et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding.* arXiv:2104.09864.
- Press, O., Smith, N. A., & Lewis, M. (2022). *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation.* ICLR 2022.
- Peng, B., et al. (2023). *YaRN: Efficient Context Window Extension of Large Language Models.* arXiv:2309.00071.
- Liu, N. F., et al. (2023). *Lost in the Middle: How Language Models Use Long Contexts.* TACL 2023.
- Chen, S., et al. (2023). *Extending Context Window of Large Language Models via Positional Interpolation.* arXiv:2306.15595.

# Attention Mechanism

## Table of Contents

1. [Why Attention?](#why-attention)
2. [Bahdanau (Additive) Attention](#bahdanau-additive-attention)
3. [Luong (Multiplicative) Attention](#luong-multiplicative-attention)
4. [Self-Attention: Queries, Keys, and Values](#self-attention-queries-keys-and-values)
5. [Scaled Dot-Product Attention](#scaled-dot-product-attention)
6. [Multi-Head Attention](#multi-head-attention)
7. [Cross-Attention](#cross-attention)
8. [Attention Masks](#attention-masks)
9. [Computational Complexity](#computational-complexity)
10. [Visualizing Attention Patterns](#visualizing-attention-patterns)
11. [References](#references)

---

## Why Attention?

### The Bottleneck Problem in Seq2Seq

Before attention, sequence-to-sequence models (Sutskever et al., 2014) encoded the entire input sequence into a single fixed-size vector — the final hidden state of an RNN encoder — and forced the decoder to reconstruct the output from that alone.

```
Input:  "The cat sat on the mat"
            |
        [Encoder RNN]
            |
       [ context c ]   ← single vector, fixed size
            |
        [Decoder RNN]
            |
Output: "Le chat était assis sur le tapis"
```

This is a severe information bottleneck: a 50-token sentence must be compressed into a 512-dimensional vector, discarding most positional and relational structure. Empirically, RNN Seq2Seq models deteriorated sharply as sentence length increased.

**The core insight of attention**: Instead of one fixed context vector, let the decoder look back at *every encoder hidden state* and selectively weight which ones are relevant at each decoding step. The context vector becomes dynamic — a weighted sum over all encoder states, recomputed at each output position.

---

## Bahdanau (Additive) Attention

Bahdanau et al. (2015) introduced the first practical attention mechanism for neural machine translation. At each decoder time step $t$, the context vector $\mathbf{c}_t$ is:

$$\mathbf{c}_t = \sum_{j=1}^{T_x} \alpha_{tj} \mathbf{h}_j$$

where $\mathbf{h}_j$ is the $j$-th encoder hidden state and $\alpha_{tj}$ are attention weights summing to 1.

The weights come from a softmax over alignment scores:

$$\alpha_{tj} = \frac{\exp(e_{tj})}{\sum_{k=1}^{T_x} \exp(e_{tk})}$$

The alignment score $e_{tj}$ measures compatibility between the decoder state $\mathbf{s}_{t-1}$ and encoder state $\mathbf{h}_j$:

$$e_{tj} = \mathbf{v}_a^\top \tanh(\mathbf{W}_a \mathbf{s}_{t-1} + \mathbf{U}_a \mathbf{h}_j)$$

This is called **additive** (or concat) attention because the decoder and encoder states are combined via addition inside a tanh. The trainable parameters are $\mathbf{W}_a \in \mathbb{R}^{d_a \times d_s}$, $\mathbf{U}_a \in \mathbb{R}^{d_a \times d_h}$, and $\mathbf{v}_a \in \mathbb{R}^{d_a}$.

**Computational note**: Each alignment score requires a forward pass through a small MLP. For a sequence of length $n$, this is $O(n)$ per decoder step, $O(n^2)$ total — already quadratic, but with a large constant.

---

## Luong (Multiplicative) Attention

Luong et al. (2015) proposed simpler alignment functions. The most common is the **dot-product** variant:

$$e_{tj} = \mathbf{s}_t^\top \mathbf{h}_j$$

or the **general** variant with a learned weight matrix:

$$e_{tj} = \mathbf{s}_t^\top \mathbf{W}_a \mathbf{h}_j$$

Multiplicative attention is faster than additive because it avoids the tanh MLP — it reduces to a matrix multiplication. However, as $d$ grows large, dot products grow in magnitude, which Bahdanau-style attention is less susceptible to (the tanh bounds it). Vaswani et al. addressed this by introducing scaling.

---

## Self-Attention: Queries, Keys, and Values

The attention mechanisms above compare decoder states to encoder states (cross-attention). **Self-attention** allows each position in a sequence to attend to every other position *in the same sequence*. This is the core operation of the Transformer.

### Where Q, K, V Come From

For an input sequence of token representations $\mathbf{X} \in \mathbb{R}^{n \times d_{\text{model}}}$, three separate linear projections produce:

$$\mathbf{Q} = \mathbf{X}\mathbf{W}^Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}^K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}^V$$

where $\mathbf{W}^Q, \mathbf{W}^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ and $\mathbf{W}^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$.

### What They Represent

The query-key-value abstraction is borrowed from information retrieval:

- **Query** ($\mathbf{Q}$): What information am I looking for? The current position's "question."
- **Key** ($\mathbf{K}$): What information do I advertise? Each position's "index entry."
- **Value** ($\mathbf{V}$): What information do I actually contain? What gets aggregated once I'm selected.

The attention score between position $i$ (query) and position $j$ (key) measures how much position $i$ should attend to position $j$. The output at position $i$ is then a weighted sum of all values, weighted by these scores.

Crucially, $\mathbf{W}^Q$, $\mathbf{W}^K$, $\mathbf{W}^V$ are different learned projections. The separation allows the model to learn *asymmetric* relationships: what a position "asks" for can differ from what it "offers." Without separate projections, self-attention collapses to a symmetric similarity function over the raw embeddings.

---

## Scaled Dot-Product Attention

The full operation is:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

### Why the $\sqrt{d_k}$ Scaling?

Without scaling, the dot products $\mathbf{q}_i \cdot \mathbf{k}_j$ grow in magnitude as $d_k$ increases. Formally, if the components of $\mathbf{q}$ and $\mathbf{k}$ are independently drawn from $\mathcal{N}(0, 1)$, then:

$$\mathbf{q} \cdot \mathbf{k} = \sum_{i=1}^{d_k} q_i k_i \sim \mathcal{N}(0, d_k)$$

The variance scales linearly with $d_k$, so the standard deviation scales as $\sqrt{d_k}$. Large-magnitude dot products drive softmax into its saturation region, where:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}} \approx \begin{cases} 1 & z_i \gg 0 \\ 0 & \text{otherwise} \end{cases}$$

In this saturated regime, gradients of softmax become near-zero (the derivative of softmax at extreme values approaches 0), making learning very slow. Dividing by $\sqrt{d_k}$ normalizes the dot products back to unit variance, keeping softmax in a well-conditioned region.

### Step-by-Step Computation

For a sequence of length $n$, with $d_k = d_v = d$:

1. Compute score matrix: $\mathbf{S} = \mathbf{Q}\mathbf{K}^\top \in \mathbb{R}^{n \times n}$
2. Scale: $\mathbf{S} \leftarrow \mathbf{S} / \sqrt{d_k}$
3. Apply mask (if any): set masked positions to $-\infty$
4. Softmax row-wise: $\mathbf{A} = \text{softmax}(\mathbf{S}) \in \mathbb{R}^{n \times n}$
5. Aggregate values: $\text{Output} = \mathbf{A}\mathbf{V} \in \mathbb{R}^{n \times d_v}$

```
Q [n×d_k]  K [n×d_k]  V [n×d_v]
    |           |          |
    └────── Q·Kᵀ ──────────┘
              |
          ÷ √d_k
              |
          + mask
              |
          softmax
              |
          · V
              |
        Output [n×d_v]
```

---

## Multi-Head Attention

### Motivation

A single attention head computes one weighted aggregation of values. But natural language has multiple simultaneous relational structures: syntactic dependencies, coreference, semantic similarity, positional proximity. A single head must trade off between these. **Multiple heads** allow the model to attend to different aspects of the input simultaneously, in different representation subspaces.

### Mechanism

Run $h$ attention heads in parallel, each with its own projections:

$$\text{head}_i = \text{Attention}(\mathbf{X}\mathbf{W}_i^Q,\; \mathbf{X}\mathbf{W}_i^K,\; \mathbf{X}\mathbf{W}_i^V)$$

where $\mathbf{W}_i^Q, \mathbf{W}_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ and $\mathbf{W}_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$.

In practice, $d_k = d_v = d_{\text{model}} / h$, so the total computation stays constant regardless of the number of heads.

Concatenate all heads and apply a final linear projection:

$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\,\mathbf{W}^O$$

where $\mathbf{W}^O \in \mathbb{R}^{h \cdot d_v \times d_{\text{model}}}$.

### Parameter Count

For one multi-head attention layer with $d_{\text{model}}$ and $h$ heads (where $d_k = d_v = d_{\text{model}}/h$):

| Parameter | Shape | Count |
|-----------|-------|-------|
| $\mathbf{W}^Q$ (all heads combined) | $d_{\text{model}} \times d_{\text{model}}$ | $d_{\text{model}}^2$ |
| $\mathbf{W}^K$ (all heads combined) | $d_{\text{model}} \times d_{\text{model}}$ | $d_{\text{model}}^2$ |
| $\mathbf{W}^V$ (all heads combined) | $d_{\text{model}} \times d_{\text{model}}$ | $d_{\text{model}}^2$ |
| $\mathbf{W}^O$ | $d_{\text{model}} \times d_{\text{model}}$ | $d_{\text{model}}^2$ |
| **Total** | | $4 d_{\text{model}}^2$ |

Note: biases add $4 d_{\text{model}}$ parameters if included.

### Head Splitting in Practice

The $h$ heads are computed efficiently by reshaping:

```python
# X: [batch, seq_len, d_model]
Q = X @ W_Q  # [batch, seq_len, d_model]
Q = Q.reshape(batch, seq_len, h, d_k).transpose(1, 2)  # [batch, h, seq_len, d_k]
# Same for K, V
scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)  # [batch, h, seq_len, seq_len]
attn = softmax(scores, dim=-1)
out = attn @ V  # [batch, h, seq_len, d_v]
out = out.transpose(1, 2).reshape(batch, seq_len, d_model)
out = out @ W_O
```

---

## Cross-Attention

In encoder-decoder models (e.g., the original Transformer for translation), the decoder must incorporate information from the encoder. **Cross-attention** uses decoder states as queries and encoder states as keys and values:

$$\text{CrossAttn}(\mathbf{Q}_{\text{dec}}, \mathbf{K}_{\text{enc}}, \mathbf{V}_{\text{enc}}) = \text{softmax}\!\left(\frac{\mathbf{Q}_{\text{dec}}\mathbf{K}_{\text{enc}}^\top}{\sqrt{d_k}}\right)\mathbf{V}_{\text{enc}}$$

- $\mathbf{Q}$ comes from the decoder's previous sublayer output (shape: $[n_{\text{dec}} \times d_k]$)
- $\mathbf{K}$, $\mathbf{V}$ come from the encoder's final hidden states (shape: $[n_{\text{enc}} \times d_k]$, $[n_{\text{enc}} \times d_v]$)

The resulting score matrix is $[n_{\text{dec}} \times n_{\text{enc}}]$ — each decoder position attends over all encoder positions. This is how the translation task "looks up" relevant source tokens at each generation step.

---

## Attention Masks

Two important mask types are applied to the score matrix before softmax:

### Padding Mask

Sequences in a batch have different lengths. Shorter sequences are padded with a special `[PAD]` token to enable batched computation. These padding positions should not contribute to attention output, so their scores are set to $-\infty$ before softmax:

$$S_{ij} \leftarrow \begin{cases} S_{ij} & \text{if position } j \text{ is not padding} \\ -\infty & \text{if position } j \text{ is padding} \end{cases}$$

After softmax, $e^{-\infty} = 0$, so padding positions receive zero attention weight.

### Causal (Look-Ahead) Mask

For autoregressive generation, the model should only attend to positions at or before the current position — not future tokens it hasn't generated yet. The upper triangle of the attention score matrix is masked:

$$S_{ij} \leftarrow \begin{cases} S_{ij} & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

The resulting mask is:

```
Position:  1    2    3    4    5
Token 1: [ 0   -∞   -∞   -∞   -∞ ]
Token 2: [ 0    0   -∞   -∞   -∞ ]
Token 3: [ 0    0    0   -∞   -∞ ]
Token 4: [ 0    0    0    0   -∞ ]
Token 5: [ 0    0    0    0    0 ]
```

This mask is fixed and does not require learning. In practice both masks are applied simultaneously for decoder self-attention in training (causal mask AND padding mask), while encoder self-attention uses only the padding mask.

---

## Computational Complexity

For a sequence of length $n$, model dimension $d_{\text{model}}$:

| Operation | Time Complexity | Memory |
|-----------|----------------|--------|
| Score matrix $\mathbf{Q}\mathbf{K}^\top$ | $O(n^2 d)$ | $O(n^2)$ |
| Softmax | $O(n^2)$ | $O(n^2)$ |
| Value aggregation $\mathbf{A}\mathbf{V}$ | $O(n^2 d)$ | $O(nd)$ |
| **Total** | $O(n^2 d)$ | $O(n^2)$ |

The $O(n^2)$ memory cost for storing the attention matrix is the dominant bottleneck for long sequences. For $n = 100{,}000$ tokens (100K context), the attention matrix alone requires:

$$100{,}000^2 \times 4 \text{ bytes} \approx 40 \text{ GB}$$

per head, per layer — clearly infeasible at full precision. This is why long-context modeling requires algorithmic innovations:

- **Flash Attention** (Dao et al., 2022): Recomputes attention in tiles to avoid materializing the full $n \times n$ matrix in HBM, achieving $O(n^2 / M)$ memory where $M$ is SRAM size.
- **Sparse Attention**: Only attend to a subset of positions (local window + global tokens).
- **Linear Attention**: Kernel approximations that rewrite attention as $\mathbf{Q}(\mathbf{K}^\top\mathbf{V})$, reducing to $O(n)$ by exploiting associativity.

For comparison, an RNN processes each step in $O(d^2)$ time, $O(d)$ memory, for $O(nd^2)$ total — linear in $n$, but sequential (cannot be parallelized).

---

## Visualizing Attention Patterns

Attention weights $\mathbf{A} \in [0,1]^{n \times n}$ can be visualized as heat maps. Empirically, different heads learn qualitatively different patterns:

**Diagonal / local attention**: Head attends mostly to nearby tokens. Common in lower layers. Corresponds to local context sensitivity.

```
       it  is  a  cat
it  [ .8  .1  .1  0 ]
is  [ .2  .6  .1  .1]
a   [ .0  .1  .7  .2]
cat [ .0  .1  .2  .7]
```

**Coreference resolution**: A pronoun attends strongly to the noun it refers to, regardless of distance.

**Syntactic attention**: Determiners attend to their nouns; verbs attend to their subjects.

**Attention sink**: Some heads develop an "attention sink" token (often the `[BOS]` or first token) that absorbs probability mass from positions with no strong signal. This is not meaningful attention but rather a near-uniform fallback. Understanding this phenomenon motivated positional biasing in ALiBi and sink-token-aware architectures.

**Caution on interpretation**: Attention weights are not a faithful explanation of model behavior. A position can receive high attention weight yet have low impact on the output if its value vector is small in magnitude (or if attention weight and gradient signal disagree). Attention is a mechanism, not a post-hoc explanation.

---

## References

- Bahdanau, D., Cho, K., & Bengio, Y. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR 2015.
- Luong, T., Pham, H., & Manning, C. D. (2015). *Effective Approaches to Attention-based Neural Machine Translation.* EMNLP 2015.
- Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS 2017.
- Dao, T., et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022.
- Jain, S., & Wallace, B. C. (2019). *Attention is not Explanation.* NAACL 2019.
- Wiegreffe, S., & Pinter, Y. (2019). *Attention is not not Explanation.* EMNLP 2019.

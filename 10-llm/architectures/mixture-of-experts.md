# Mixture of Experts (MoE)

## Table of Contents

1. [Core Idea: Sparse Activation](#1-core-idea-sparse-activation)
2. [Expert Layers: Architecture](#2-expert-layers-architecture)
3. [Gating and Routing](#3-gating-and-routing)
4. [Load Balancing](#4-load-balancing)
5. [Switch Transformer](#5-switch-transformer)
6. [Mixtral 8x7B](#6-mixtral-8x7b)
7. [GLaM, Gemini, and Other MoE Variants](#7-glam-gemini-and-other-moe-variants)
8. [Parameter Count Analysis](#8-parameter-count-analysis)
9. [Expert Specialization](#9-expert-specialization)
10. [Trade-offs and Deployment Challenges](#10-trade-offs-and-deployment-challenges)
11. [References](#references)

---

## 1. Core Idea: Sparse Activation

A standard dense transformer layer applies the same FFN to every token. Mixture of Experts (MoE) replaces this with a set of $E$ independent FFN networks (the "experts"), but activates only $k$ of them per token.

The result: **total parameters scale with $E$, compute scales with $k$**. A model with 8 experts and top-2 routing has 8× the FFN parameters of a dense model but only 2× the FFN FLOPs per token. This enables substantially larger models for the same training and inference compute budget.

```
Dense FFN:
  token → [single FFN] → output
  (all parameters active every time)

MoE FFN:
  token → [Router] → selects experts {2, 5}
  token → [Expert 2] + [Expert 5] → weighted sum → output
  (only 2/8 experts active per token)
```

The key claim (supported empirically): sparse models with more total parameters train to lower loss than dense models with equivalent FLOPs. The extra parameters act as additional capacity that gets specialized — different experts learn different patterns — without proportionally increasing compute.

---

## 2. Expert Layers: Architecture

### Where MoE Replaces Dense Layers

In a standard transformer, each block contains:
1. Self-attention sublayer
2. FFN sublayer

In an MoE transformer, every $m$-th FFN sublayer is replaced with an MoE layer (e.g., every other layer in Mixtral):

```
Block 1:  Self-Attn → FFN (dense)
Block 2:  Self-Attn → MoE (sparse)
Block 3:  Self-Attn → FFN (dense)
Block 4:  Self-Attn → MoE (sparse)
...
```

Alternating dense and MoE layers keeps the model stable (attention layers remain dense) while allowing sparse FFNs.

### Expert FFN Structure

Each expert $i$ is a standard FFN:

$$\text{Expert}_i(\mathbf{x}) = \text{Act}(\mathbf{x} W_{1,i}) W_{2,i}$$

where $W_{1,i} \in \mathbb{R}^{d \times d_{\text{ff}}}$ and $W_{2,i} \in \mathbb{R}^{d_{\text{ff}} \times d}$. Experts have identical architecture but independent parameters.

All experts share the same input $\mathbf{x}$ (the token representation), but each applies its own learned transformation. The router decides which experts' outputs to use.

---

## 3. Gating and Routing

### Token-Choice Routing (Top-k)

The router is a learned linear projection followed by softmax:

$$\mathbf{g}(\mathbf{x}) = \text{softmax}(\mathbf{x} W_g), \quad W_g \in \mathbb{R}^{d \times E}$$

This produces a probability distribution over $E$ experts. Top-$k$ routing selects the $k$ experts with highest probability:

$$\text{TopK}(\mathbf{g}(\mathbf{x}), k) = \{i : g_i(\mathbf{x}) \in \text{top-}k \text{ values of } \mathbf{g}(\mathbf{x})\}$$

The MoE output is a weighted combination of selected expert outputs:

$$\text{MoE}(\mathbf{x}) = \sum_{i \in \text{TopK}(\mathbf{g}(\mathbf{x}), k)} \frac{g_i(\mathbf{x})}{\sum_{j \in \text{TopK}} g_j(\mathbf{x})} \cdot \text{Expert}_i(\mathbf{x})$$

The weights are the renormalized softmax probabilities of the selected experts.

### Why Top-k (Usually k=2)?

- **k=1 (top-1)**: simplest, lowest compute. Used by Switch Transformer. Risk: the model cannot interpolate between experts.
- **k=2 (top-2)**: used by Mixtral, most open-source MoE models. The combination of two experts allows smoother interpolation and is empirically more stable.
- **k>2**: diminishing returns; the routing cost and load balancing difficulty increase.

### Router Noise (Jitter)

During training, standard practice adds noise to router logits before softmax:

$$\mathbf{g}_{\text{noisy}}(\mathbf{x}) = \mathbf{x} W_g + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

This encourages exploration of different experts early in training, preventing premature routing collapse (where one expert always "wins").

---

## 4. Load Balancing

### The Problem

Without any balancing constraint, routing quickly collapses: one or two experts receive most tokens (they become "popular" and improve, which makes them more popular — a positive feedback loop). Most experts receive too few tokens to train effectively.

Worse, in distributed training, experts are typically sharded across devices. Uneven routing causes **device hotspots**: some devices process many tokens, others sit idle, creating a throughput bottleneck.

### Expert Capacity

Each expert is assigned a **capacity** $C$ — the maximum number of tokens it can process per batch:

$$C = \text{capacity\_factor} \times \frac{T}{E}$$

where $T$ is the total number of tokens in the batch and $E$ is the number of experts. The **capacity factor** (typically 1.0–2.0) determines how much overflow is tolerated. Tokens that exceed the capacity of their top-$k$ expert are **dropped** (bypassed with a zero output). Capacity factor > 1.0 allows some overflow before dropping.

### Auxiliary Load Balancing Loss

Zoph et al. (Switch Transformer) introduced an auxiliary loss that encourages uniform routing. Let $f_i$ be the fraction of tokens routed to expert $i$, and $p_i$ be the mean routing probability for expert $i$:

$$f_i = \frac{1}{T} \sum_{t=1}^{T} \mathbf{1}[\text{token } t \text{ assigned to expert } i]$$

$$p_i = \frac{1}{T} \sum_{t=1}^{T} g_i(\mathbf{x}_t)$$

The auxiliary loss:

$$\mathcal{L}_{\text{balance}} = \alpha E \sum_{i=1}^{E} f_i \cdot p_i$$

where $\alpha$ is a small scalar (e.g., $10^{-2}$). This is minimized when $f_i = p_i = 1/E$ (uniform distribution). The product $f_i \cdot p_i$ allows gradients to flow through $p_i$ (which is differentiable) while the indicator-based $f_i$ guides the direction.

The $E$ multiplier ensures the loss scale is independent of the number of experts.

**Why multiply $f_i$ and $p_i$ rather than just penalizing $f_i$?** Because $f_i$ is not differentiable (it involves argmax). Multiplying by $p_i$ provides a smooth gradient signal: if expert $i$ is overloaded ($f_i$ large), penalizing $p_i$ (which is differentiable) discourages routing tokens to it.

### Z-Loss (Router Regularization)

Zoph et al. (2022) introduced **z-loss** to stabilize training by penalizing large router logits:

$$\mathcal{L}_z = \frac{1}{T} \sum_{t=1}^{T} \left(\log \sum_{i=1}^{E} e^{x_{t,i}}\right)^2$$

Large logits cause numerical instability and aggressive routing. Z-loss keeps logits moderate.

---

## 5. Switch Transformer

### Design

Switch Transformer (Fedus et al., 2021) demonstrated that MoE can scale language models to trillion-parameter counts efficiently. Key design choice: **top-1 routing** (each token routed to exactly one expert, $k=1$).

Motivations for top-1:
- Halves the routing computation vs top-2
- Simpler load balancing analysis
- Empirically competitive with top-2 at large scale

### Capacity Factor Trade-off

With top-1 routing and uniform random assignment, the expected load on each expert is $T/E$ tokens. The capacity factor determines how much headroom exists:

| Capacity Factor | Effect |
|----------------|--------|
| 1.0 | Tight; any imbalance causes drops |
| 1.25 | 25% buffer; some imbalance tolerated |
| 2.0 | Generous; no drops in typical training |

Lower capacity factor = lower memory overhead but more token drops. Higher = safer but more waste.

### Sparse-Dense Comparison

Switch-Base (7.4B total, ~250M active) consistently outperforms T5-Base (220M dense) at matched FLOPs. At equal parameter count, dense models still outperform Switch at small scales — the MoE advantage emerges at scale where the extra experts provide meaningful additional capacity.

### Architecture Details

- Based on T5 encoder-decoder
- Every other FFN layer replaced with MoE
- 128 experts in many configurations
- Expert parallelism: each device holds a subset of experts

---

## 6. Mixtral 8x7B

Mixtral 8x7B (Mistral AI, 2024) is the most widely deployed open-weight MoE model and established MoE as a practical choice for open-source LLMs.

### Architecture

- **8 experts** per MoE layer
- **Top-2 routing**: each token activates 2 of 8 experts
- **32 MoE layers** (all FFN layers are MoE, unlike Switch which alternates)
- Same attention architecture as Mistral-7B: GQA, sliding window attention, RoPE, RMSNorm, SwiGLU

| Property | Value |
|----------|-------|
| Total parameters | 46.7B |
| Active parameters per token | 12.9B |
| FFN experts per layer | 8 |
| Top-k routing | k=2 |
| Context length | 32K |
| Vocabulary | 32K |

### Why 46.7B Total but 12.9B Active?

- Each expert FFN has the same shape as Mistral-7B's FFN
- With 8 experts, FFN parameters = $8 \times$ single-expert params
- Attention parameters are shared (not replicated per expert)
- Active = attention params + 2 expert FFN params (top-2)

The "8x7B" label is colloquial: the model is not 8 copies of a 7B model. It is a 7B-architecture transformer where the FFN layers are replaced with 8-expert MoE, giving ~46.7B total and ~12.9B active.

### Performance

Mixtral-8x7B matches or exceeds LLaMA-2-70B on most benchmarks at less than 1/3 the active compute (12.9B vs 70B active parameters). This is the strongest empirical demonstration of the "more total params, same compute" MoE advantage.

### Router in Mixtral

Mixtral uses top-2 routing with no router noise and no explicit load balancing loss. Load balancing is enforced via the capacity mechanism. The router logits are computed as a linear projection of the token hidden state.

```python
# Conceptual Mixtral MoE forward pass (simplified)
def moe_layer(x, W_gate, experts, k=2):
    # x: (batch, seq_len, d_model)
    # Router
    gate_logits = x @ W_gate  # (batch, seq_len, num_experts)
    weights, selected = torch.topk(gate_logits.softmax(-1), k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)  # renormalize

    # Dispatch and combine
    output = torch.zeros_like(x)
    for expert_idx in range(num_experts):
        mask = (selected == expert_idx).any(dim=-1)
        if mask.any():
            expert_weight = weights[selected == expert_idx]
            output[mask] += expert_weight.unsqueeze(-1) * experts[expert_idx](x[mask])
    return output
```

Loading Mixtral with HuggingFace:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    torch_dtype=torch.bfloat16,
    device_map="auto",  # requires ~100GB VRAM for bfloat16; use quantization otherwise
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
```

---

## 7. GLaM, Gemini, and Other MoE Variants

### GLaM (Google, 2021)

GLaM (Generalist Language Model) was the first trillion-parameter MoE model trained at scale. Key results:
- 1.2T total parameters, ~96B active (top-2 of 64 experts)
- Outperforms GPT-3 (175B dense) on zero/one-shot tasks while using 1/3 the energy per inference

GLaM demonstrated MoE scaling beyond what Switch Transformer showed: the quality-per-FLOP advantage is robust and increases with scale.

### Gemini (Google DeepMind, 2024)

Gemini Ultra and later Gemini 1.5 are reported to use MoE architectures. Gemini 1.5 Pro achieves a 1M-token context window, combining MoE with efficient attention mechanisms (linear attention / SSM components). Architectural details are not fully disclosed.

### Grok-1 (xAI, 2024)

Grok-1 (314B total, top-2 of 8 experts, ~86B active) was released as open-weights — the largest open-weight MoE model at the time of release. Architecture: similar to Mixtral but at much larger scale.

### Deepseek-MoE (DeepSeek, 2024)

DeepSeek introduces **fine-grained expert segmentation**: instead of 8 large experts, use 64 small experts with top-6 routing. More fine-grained routing allows better specialization. DeepSeek-V2 achieves strong performance with very low active parameter count (~21B active out of 236B total).

---

## 8. Parameter Count Analysis

### Dense vs MoE Parameter Budget

For a transformer with $L$ layers, hidden size $d$, FFN multiplier $r$:

**Dense model**:
$$N_{\text{dense}} = L \left( 4d^2_{\text{attn}} + 2rd^2_{\text{ffn}} \right)$$

(attention QKV + output + FFN up + down, simplified)

**MoE model** (all layers are MoE, $E$ experts):
$$N_{\text{moe}} = L \left( 4d^2_{\text{attn}} + E \cdot 2rd^2_{\text{ffn}} \right)$$

**Active parameters per token** (top-$k$):
$$N_{\text{active}} = L \left( 4d^2_{\text{attn}} + k \cdot 2rd^2_{\text{ffn}} \right)$$

For Mixtral 8x7B: $L=32$, $d=4096$, $r \approx 3.5$ (SwiGLU), $E=8$, $k=2$:
- Attention per layer: $\approx 4 \times 4096^2 \approx 67M$
- FFN per layer per expert: $\approx 2 \times 14336 \times 4096 \approx 117M$ (with SwiGLU: 3 matrices)
- Total: attention ($32 \times 67M = 2.1B$) + FFN ($32 \times 8 \times 117M = 30B$) + embeddings ($\approx 0.5B$) ≈ 46.7B
- Active: attention ($2.1B$) + FFN active ($32 \times 2 \times 117M = 7.5B$) ≈ 12.9B

### The Fundamental Trade-off

```
Fixed compute budget (FLOPs):

Dense 12.9B:
  ↓
  12.9B params always active
  ↓
  Fixed model capacity

MoE 46.7B / 12.9B active:
  ↓
  46.7B params total, 12.9B per token
  ↓
  More total capacity, same compute
  ↓
  Better quality per FLOP (if routing is good)
```

But: serving 46.7B requires loading all 46.7B into memory, even though only 12.9B are used per inference step. The memory cost is the total model size.

---

## 9. Expert Specialization

Empirical analysis of routing patterns shows that experts do specialize, though not always in human-interpretable ways:

**Domain specialization**: In multilingual models, different experts tend to activate for different languages. In code-trained models, some experts specialize in code patterns.

**Syntactic specialization**: Some experts specialize in function words and determiners; others in content words; others in punctuation.

**Position specialization**: Certain experts are activated more for tokens at the beginning or end of sentences.

**Consistency**: A given input token activates roughly the same set of experts regardless of its context (the routing is content-based but relatively stable). This supports the idea that experts learn genuine specializations.

However, specialization is not perfectly clean — experts overlap significantly and the routing is continuous, not discrete. The specialization is a soft statistical tendency, not a hard partition.

---

## 10. Trade-offs and Deployment Challenges

### Memory vs Compute Trade-off

| Model | Total Params | Active Params | VRAM (bfloat16) |
|-------|-------------|--------------|-----------------|
| LLaMA-2-13B dense | 13B | 13B | ~26GB |
| Mixtral-8x7B MoE | 46.7B | 12.9B | ~93GB |
| LLaMA-2-70B dense | 70B | 70B | ~140GB |

Mixtral-8x7B achieves LLaMA-2-70B quality at LLaMA-2-13B compute but LLaMA-2-70B memory. This is the core hardware trade-off.

### Expert Parallelism

For distributed inference, MoE requires **expert parallelism**: different experts live on different devices. For a token batch:

1. Compute router → determine expert assignments
2. **All-to-all communication**: send tokens to the devices that hold their assigned experts
3. Each device runs its local experts
4. **All-to-all communication**: gather results back to original devices
5. Combine expert outputs

The all-to-all communication overhead is the primary bottleneck at inference time. For small batch sizes (single user), the communication overhead relative to compute is high.

### Latency vs Throughput

- **Throughput**: MoE models are efficient at high batch sizes — the expert parallelism overhead amortizes over many tokens. Large-scale serving of MoE models can be efficient.
- **Latency**: At batch size 1 (interactive use), MoE models may be slower than equivalently-accurate dense models due to routing overhead and lower compute intensity per expert.

### Load Imbalance at Inference

During training, load balancing losses enforce roughly uniform routing. At inference, there is no loss — the router produces whatever distribution it learned. If the inference input distribution differs from training (domain shift), routing can become imbalanced, causing some experts to be saturated and others idle.

### Quantization Complexity

Quantizing MoE models requires quantizing each expert independently. With 8 experts × 32 layers = 256 expert FFNs, the quantization process is more complex and memory layout is less contiguous than dense models.

---

## References

- Shazeer, N., et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. ICLR 2017. [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)
- Fedus, W., Zoph, B., Shazeer, N. (2021). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. JMLR 2022. [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)
- Zoph, B., et al. (2022). *ST-MoE: Designing Stable and Transferable Sparse Expert Models*. [arXiv:2202.08906](https://arxiv.org/abs/2202.08906)
- Du, N., et al. (2021). *GLaM: Efficient Scaling of Language Models with Mixture-of-Experts*. ICML 2022. [arXiv:2112.06905](https://arxiv.org/abs/2112.06905)
- Jiang, A.Q., et al. (2024). *Mixtral of Experts*. [arXiv:2401.04088](https://arxiv.org/abs/2401.04088)
- Dai, D., et al. (2024). *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*. [arXiv:2401.06066](https://arxiv.org/abs/2401.06066)

---

*[← Encoder-Decoder Models](./encoder-decoder.md) | [← Back to Architectures](./README.md) | [→ State Space Models](./state-space-models.md)*

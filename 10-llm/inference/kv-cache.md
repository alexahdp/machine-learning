# KV Cache

## Table of Contents

1. [The Recomputation Problem](#1-the-recomputation-problem)
2. [KV Cache Mechanism](#2-kv-cache-mechanism)
3. [Memory Formula and Sizing](#3-memory-formula-and-sizing)
4. [Memory Growth and the Long-Context Bottleneck](#4-memory-growth-and-the-long-context-bottleneck)
5. [Multi-Query Attention (MQA)](#5-multi-query-attention-mqa)
6. [Grouped Query Attention (GQA)](#6-grouped-query-attention-gqa)
7. [PagedAttention](#7-pagedattention)
8. [Prefix Caching and Prompt Caching](#8-prefix-caching-and-prompt-caching)
9. [KV Cache Quantization](#9-kv-cache-quantization)
10. [Streaming Generation](#10-streaming-generation)
11. [References](#references)

---

## 1. The Recomputation Problem

Transformer attention computes queries, keys, and values for each token at each layer:

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

During **training**, all tokens are processed in parallel — this is the key efficiency advantage of the Transformer over RNNs. During **autoregressive inference**, each new token must attend to all previous tokens. Without optimization:

- Step 1: generate token $x_1$ from prompt $x_{1:\text{prompt}}$ — compute K, V for entire prompt
- Step 2: generate token $x_2$ — compute K, V for prompt + $x_1$ **again**, then append $x_2$'s K, V
- Step $t$: recompute K, V for all $t-1$ previous tokens

This is $O(T^2)$ compute to generate a sequence of length $T$ — entirely wasted, since the K and V for previous tokens **do not change** in a decoder-only model (causal masking means past tokens' representations are independent of future tokens).

---

## 2. KV Cache Mechanism

The KV cache stores the key and value tensors for all previous tokens, reusing them at each new step:

```
Prompt: "The capital of France is"

Step 1: Process full prompt
  → Compute K, V for all 6 prompt tokens at each layer
  → Cache: {layer_0: (K_0[0:6], V_0[0:6]), layer_1: ..., ...}
  → Generate token: "Paris"

Step 2: Process only "Paris"
  → Compute Q, K, V for the single new token
  → Retrieve cached K, V for positions 0-5 from cache
  → Concatenate: K_full = [K_cached | K_new]
  → Run attention with Q_new over K_full, V_full
  → Generate token: "."

Step 3: Process only "."
  → Compute Q, K, V for "."
  → Retrieve cached K, V for positions 0-6
  → ... and so on
```

**Compute savings**: Without KV cache, generating a 1024-token response to a 1024-token prompt requires computing attention over sequences of average length ~1536 for each of 1024 steps. With KV cache, only one token's Q, K, V is computed per step — O(T) compute instead of $O(T^2)$.

**Memory trade-off**: KV cache trades compute for memory. The cache must be allocated upfront or grown dynamically and held in GPU VRAM for the entire generation.

---

## 3. Memory Formula and Sizing

The total KV cache memory for a single request is:

$$\text{KV cache size} = 2 \times L \times H_{kv} \times d_h \times S \times P$$

where:
- $2$: keys and values (two tensors)
- $L$: number of transformer layers
- $H_{kv}$: number of KV heads (= number of query heads in standard MHA)
- $d_h$: dimension per head ($= d_{\text{model}} / H$)
- $S$: sequence length (prompt + generated tokens)
- $P$: bytes per element (2 for FP16/BF16, 1 for INT8)

**Example: LLaMA-2 70B**

Architecture: $L = 80$ layers, $H = 64$ query heads, $H_{kv} = 8$ KV heads (GQA), $d_h = 128$, standard BF16 ($P = 2$):

$$\text{Per-token KV} = 2 \times 80 \times 8 \times 128 \times 2 = 327{,}680 \text{ bytes} \approx 320 \text{ KB/token}$$

At 4096 context length:
$$4096 \times 320 \text{ KB} \approx 1.28 \text{ GB per request}$$

At 32K context length:
$$32768 \times 320 \text{ KB} \approx 10 \text{ GB per request}$$

**LLaMA-2 7B** (standard MHA: $H_{kv} = 32$, $L = 32$):
$$2 \times 32 \times 32 \times 128 \times 2 = 524{,}288 \text{ bytes} \approx 512 \text{ KB/token}$$

At 4096 tokens: ~2 GB per request.

**Critical implication**: On an 80 GB A100, after loading LLaMA-2 70B weights (~140 GB FP16 — requires 2 A100s), the remaining memory for KV cache with 2× A100s at 160 GB total is:
$$160 - 140 = 20 \text{ GB for KV cache} \approx 62 \text{ concurrent 4K-context requests}$$

This is the fundamental constraint driving batching and memory management design.

---

## 4. Memory Growth and the Long-Context Bottleneck

KV cache grows linearly with sequence length. This has two consequences:

**1. Maximum concurrent requests drops with context length**: At 128K context, a single request on LLaMA-2 70B (with GQA) requires ~40 GB of KV cache — the model might only serve 1-2 concurrent requests.

**2. Memory fragmentation**: Naively allocating KV cache as a contiguous block wastes memory. If a batch has requests of varying lengths (some at 100 tokens, some at 4000), allocating each to the maximum length wastes up to 4000/100 = 40x memory per short request if pre-allocated. Dynamic allocation solves this but introduces fragmentation.

**Sequence length scaling**:

```
Context    | KV per request (LLaMA-2 70B, GQA, BF16) | # requests @ 160GB (after weights)
4K         | 1.3 GB                                   | ~15
32K        | 10 GB                                    | ~2
128K       | 40 GB                                    | ~0.5 (not feasible)
```

This is why long-context models require dedicated memory management (PagedAttention) and KV compression techniques (GQA, quantization).

---

## 5. Multi-Query Attention (MQA)

Proposed by Shazeer (2019). Instead of $H$ separate KV heads (one per query head), use a **single** shared KV head for all queries:

$$Q_i = XW_{Q_i} \quad \text{for } i = 1, \ldots, H$$
$$K = XW_K, \quad V = XW_V \quad \text{(shared across all heads)}$$

**KV cache reduction**: $H \times$ reduction. For LLaMA architecture with $H = 32$ heads: from 32 KV heads to 1 KV head → 32× smaller KV cache.

**Quality impact**: MQA trades a small quality drop for dramatic memory savings. The quality gap is typically < 1% on most benchmarks but can affect tasks requiring precise per-head distinctions.

**Models using MQA**: PaLM (540B), Falcon (7B/40B), Gemini.

**Memory bandwidth advantage**: At inference, reading KV cache is the memory bandwidth bottleneck. With MQA, all query heads share one K and V — fewer bytes read from memory per step → faster decoding.

---

## 6. Grouped Query Attention (GQA)

GQA (Ainslie et al., 2023) generalizes between MHA (full KV heads) and MQA (single KV head) by using $G$ KV groups, where $1 \leq G \leq H$:

- Each group of $H/G$ query heads shares one pair of KV heads
- $G = H$: standard MHA
- $G = 1$: MQA
- $G = 8$ with $H = 64$: each group of 8 query heads shares one KV head → 8× KV cache reduction

**Formulation**: Divide query heads into $G$ groups. For group $g$, all $H/G$ query heads in that group attend to the same $K_g, V_g$:

$$\text{head}_{g,i}(X) = \text{Attn}(XW_{Q_{g,i}}, XW_{K_g}, XW_{V_g})$$

**KV cache size with GQA**:
$$\text{KV cache} = 2 \times L \times G \times d_h \times S \times P$$

**Models using GQA**:
- LLaMA-2 70B: $H = 64$, $G = 8$ (8× reduction vs MHA)
- Mistral 7B: $H = 32$, $G = 8$
- LLaMA-3 (all sizes): GQA
- Gemma, Qwen-2, Phi-3

**Training from MHA checkpoint**: Shazeer and Ainslie propose converting MHA → GQA by mean-pooling $H/G$ KV heads within each group, followed by fine-tuning. This allows upgrading existing MHA models to GQA without full retraining.

**Comparison table**:

| Method | KV heads | Cache reduction | Quality |
|--------|----------|-----------------|---------|
| MHA | $H$ | 1× (baseline) | Best |
| GQA-8 | $H/8$ | 8× | ~MHA quality |
| MQA | 1 | $H$× | Small drop |

---

## 7. PagedAttention

**Problem**: Standard KV cache allocation pre-allocates a contiguous block of memory for the maximum possible sequence length. For a batch of requests with unknown output lengths, this causes:
- **Internal fragmentation**: a request that generates 100 tokens wastes 3996 slots if max length is 4096
- **External fragmentation**: freed blocks may not be contiguous enough for a new long request
- **Low utilization**: vLLM's analysis found that 60–80% of KV cache memory is wasted under naive allocation

**PagedAttention** (Kwon et al., 2023) treats the KV cache like virtual memory in an operating system:

```
Physical KV Memory (GPU VRAM)
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│Blk 0│Blk 1│Blk 2│Blk 3│Blk 4│Blk 5│Blk 6│Blk 7│
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

Request A's logical sequence → Physical block mapping:
  Logical [0,1,2,3] → Blk 2
  Logical [4,5,6,7] → Blk 0
  Logical [8,9,10,11] → Blk 5

Request B's logical sequence:
  Logical [0,1,2,3] → Blk 1
  Logical [4,5,6,7] → Blk 3
```

Each **block** holds KV vectors for a fixed number of tokens (e.g., 16 tokens per block). A **block table** maps logical positions to physical block locations.

**Key properties**:
- Non-contiguous physical allocation: no requirement for contiguous VRAM
- Blocks allocated on demand as sequences grow
- Near-zero memory waste (< 4% internal fragmentation, only the last block per sequence)
- Enables **copy-on-write** for parallel sampling (beam search, best-of-N): shared prefixes use the same physical blocks; blocks are only copied when a request diverges

**Attention computation with PagedAttention**: During attention, the query at the new position must attend to all previous tokens. PagedAttention gathers KV data from non-contiguous blocks using a custom CUDA kernel. The block table lookup adds minimal overhead.

**Fragmentation reduction**: vLLM reports KV cache utilization > 96% with PagedAttention vs ~50% with static allocation.

---

## 8. Prefix Caching and Prompt Caching

Many production applications repeat the same prefix across requests:
- System prompt (e.g., "You are a helpful assistant...") prepended to every request
- Few-shot examples prepended to all requests in a RAG system
- Document content repeated across many Q&A queries about the same document

**Prefix caching**: Cache the KV tensors for a common prefix. When a new request arrives with the same prefix, reuse the cached KV without recomputing.

**Requirements**: The prefix must be **token-for-token identical** (even a single different token invalidates the cache at that position and all subsequent positions). System prompts must be fixed; any dynamic content must come after the cached prefix.

**Savings**: For a 2000-token system prompt prepended to 100 requests, prefix caching saves $2000 \times 100 = 200{,}000$ token-equivalents of KV computation.

**Implementations**:
- **vLLM**: Automatic prefix caching via hash-based block sharing in PagedAttention. Blocks whose content (token IDs) matches cached blocks are reused.
- **Claude API** (Anthropic): Explicit prompt caching via `cache_control` markers. Cached prefixes are stored for 5 minutes (short TTL) or longer. Reduces input processing cost by up to 90% for repeated prefixes.
- **OpenAI API**: Automatic prompt caching for contexts > 1024 tokens (as of 2024). No explicit user control.
- **TGI**: Prefix caching in continuous batching mode.

**Radix tree for prefix caching**: vLLM uses a radix tree data structure to efficiently find the longest cached prefix matching any incoming request. This enables partial prefix matches — if the first 1000 tokens match but the next 100 don't, the first 1000 tokens' KV can still be reused.

---

## 9. KV Cache Quantization

Quantizing the KV cache from FP16 to INT8 halves KV memory at the cost of some quality:

**Per-tensor INT8 quantization**:
$$K_{\text{quant}} = \text{round}\!\left(\frac{K}{\Delta}\right), \quad \Delta = \frac{\max(|K|)}{127}$$

**Per-token quantization** (better quality): compute scale factor per token vector rather than per tensor.

**Quality impact**: KV cache quantization is less sensitive than weight quantization. Keys and values have smoother distributions than weights. INT8 KV cache typically causes < 0.5 perplexity point increase for most models.

**Memory savings**: INT8 KV cache halves memory, enabling 2× more concurrent requests or 2× longer contexts at the same memory budget.

**INT4 KV cache**: More aggressive. Quality degrades more noticeably (~1-2 perplexity points). Requires careful per-token or per-head scaling. Active research area (KVQuant, 2024).

**Attention sinks**: LLMs assign disproportionately large attention to early tokens (especially BOS), causing large-magnitude KV values. INT8 quantization handles this well with per-token scaling; per-tensor scaling is more susceptible to outlier damage.

---

## 10. Streaming Generation

KV cache is what makes streaming (token-by-token output) efficient. Without it, each streaming step would require a full $O(T)$ forward pass over the entire context. With KV cache:

1. Process prompt → compute and cache all K, V → output first token
2. Process first generated token → append to KV cache → output second token
3. Repeat: each step is $O(1)$ in terms of attention compute (one new token attends to cached context)

The model sends each token to the client as soon as it is sampled, before generating the next. This is why streaming in chat interfaces like ChatGPT shows output appearing incrementally.

**Streaming stop criteria**: The KV cache grows with each generated token. Stopping criteria (max_tokens, stop sequences, EOS token) terminate generation, and the KV cache for that request is freed.

**Memory pressure during long generation**: A request generating 10,000 tokens accumulates 10,000 entries in the KV cache. In PagedAttention systems, blocks are allocated dynamically and freed when the request ends. In systems without dynamic allocation, a maximum sequence length must be pre-allocated.

---

## References

- Shazeer, N. (2019). "Fast Transformer Decoding: One Write-Head Is All You Need." — Multi-query attention.
- Ainslie, J. et al. (2023). "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." *EMNLP 2023*. — Grouped query attention.
- Kwon, W. et al. (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention." *SOSP 2023*. — PagedAttention and vLLM.
- Hooper, C. et al. (2024). "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization." — INT4 KV cache quantization.
- Xiao, G. et al. (2023). "Efficient Streaming Language Models with Attention Sinks." — Attention sink phenomenon in KV cache.
- Anthropic. (2024). "Prompt Caching." [docs.anthropic.com](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)

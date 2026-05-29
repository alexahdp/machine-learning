# Context Window Management

## Table of Contents

1. [Context Window Evolution](#context-window-evolution)
2. [The Lost-in-the-Middle Problem](#the-lost-in-the-middle-problem)
3. [Memory Hierarchy for LLMs](#memory-hierarchy-for-llms)
4. [Chunking for Long Documents](#chunking-for-long-documents)
5. [Context Compression](#context-compression)
6. [Long-Context Model Architectures](#long-context-model-architectures)
7. [RoPE Extension Methods](#rope-extension-methods)
8. [Multi-Document QA Strategies](#multi-document-qa-strategies)
9. [KV Cache and Prompt Caching](#kv-cache-and-prompt-caching)
10. [Conversation Management](#conversation-management)
11. [References](#references)

---

## Context Window Evolution

The context window is the maximum number of tokens a model can attend to simultaneously during a forward pass. It is the model's "working memory" — everything outside the window is invisible.

| Model | Context Window | Year |
|-------|---------------|------|
| GPT-2 | 1,024 tokens | 2019 |
| GPT-3 | 2,048 tokens | 2020 |
| ChatGPT (gpt-3.5-turbo) | 4,096 tokens | 2022 |
| GPT-4 (8k) | 8,192 tokens | 2023 |
| Claude 2.1 | 200,000 tokens | 2023 |
| GPT-4 Turbo | 128,000 tokens | 2023 |
| Gemini 1.5 Pro | 1,000,000 tokens | 2024 |
| Claude 3.5 Sonnet | 200,000 tokens | 2024 |
| GPT-4o | 128,000 tokens | 2024 |

The jump from 4K to 128K–1M tokens is not merely quantitative — it is qualitative. At 4K tokens, you could process a short essay. At 128K, you can process an entire book. At 1M, you can process a software repository.

**Why context windows grew**: The bottleneck was quadratic attention complexity — standard self-attention is $O(n^2)$ in sequence length. Overcoming this required algorithmic innovations (FlashAttention, ring attention, sliding window attention) and hardware advances (more HBM bandwidth on H100s, multi-node inference).

**Cost scaling**: API pricing is typically per-token. Processing a 1M-token context can cost ~$5–15 per request (at 2024 pricing) — orders of magnitude more than a 4K-token request. Context length is a resource to manage carefully.

---

## The Lost-in-the-Middle Problem

Liu et al. (2023) empirically demonstrated a striking failure mode: **LLMs attend disproportionately to the beginning and end of their context window, while content in the middle is often ignored**.

### Evidence

Across multi-document QA tasks, model accuracy as a function of where the relevant document was placed in the context:

```
Accuracy
  100%  ┤  ■                                                    ■
   80%  ┤    ■                                              ■
   60%  ┤       ■                                      ■
   40%  ┤          ■                              ■
   20%  ┤             ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
        └─────────────────────────────────────────────
         1st    5th   10th  15th  20th  ← Document Position
```

When the relevant document was first or last, accuracy was high. When it was in the middle of 20 documents, accuracy dropped by 20–40 percentage points depending on the model.

### Why It Happens

Two mechanisms:

1. **Primary recency and primacy effects from attention patterns**: The model's attention scores are highest for tokens near the current position (recency) and near the beginning of the sequence (due to training distribution effects — documents in training often have important information early).

2. **Positional confusion at scale**: At very long contexts, absolute position embeddings lose resolution, and models trained on shorter sequences generalize imperfectly to long contexts.

### Implications for Prompt Design

```
# Bad: Put relevant content in the middle
[50 irrelevant documents...]
[THE RELEVANT DOCUMENT ← model will likely ignore this]
[50 more irrelevant documents...]

# Better: Put relevant content first or last
[THE RELEVANT DOCUMENT]
[Other documents...]

# Best for multi-document tasks: rank and order before passing to LLM
[Most relevant document] ← first
[Second most relevant]
...
[Least relevant of top-k] ← last (still visible)
```

For RAG systems, **place the highest-ranked retrieved chunks first** (not at an arbitrary position) and **place a summary or the query restatement at the very end** to anchor the generation.

---

## Memory Hierarchy for LLMs

LLMs have multiple levels of "memory", each with different capacity, speed, cost, and persistence:

```
┌─────────────────────────────────────────────────────────────────┐
│ IN-CONTEXT MEMORY (Working Memory)                              │
│ Capacity: 4K–1M tokens                                          │
│ Recall: Perfect (if in window)                                  │
│ Cost: Expensive — every token charged per request               │
│ Persistence: None — cleared after each conversation            │
│ Analogy: RAM                                                    │
├─────────────────────────────────────────────────────────────────┤
│ EXTERNAL MEMORY (RAG / Vector DB)                               │
│ Capacity: Unlimited                                             │
│ Recall: Imperfect — depends on retrieval quality                │
│ Cost: Retrieval query + embedding + storage                     │
│ Persistence: As long as the DB is maintained                    │
│ Analogy: SSD / hard drive                                       │
├─────────────────────────────────────────────────────────────────┤
│ PARAMETRIC MEMORY (Model Weights)                               │
│ Capacity: Effectively unlimited (encoded in billions of params) │
│ Recall: Imperfect — statistical, blurry, can hallucinate        │
│ Cost: Baked in at training time — zero marginal cost            │
│ Persistence: Permanent (until fine-tuned or replaced)           │
│ Analogy: Long-term memory / intuition                           │
└─────────────────────────────────────────────────────────────────┘
```

**Design principle**: Use parametric memory for general world knowledge, external memory for specific facts and private data, and in-context memory for task instructions and recent conversational context.

The optimal system uses all three layers:
- Parametric: handles common reasoning, language understanding.
- External (RAG): provides fresh, private, or high-precision factual grounding.
- In-context: holds the current task specification, conversation history, and retrieved context.

---

## Chunking for Long Documents

When a document exceeds what can fit in the context window, you must decide what to include.

### Overlap and Sliding Window

For documents processed sequentially (e.g., summarizing a long report chapter by chapter):

```python
def sliding_window_chunks(
    tokens: list[int],
    window_size: int,
    step_size: int
) -> list[list[int]]:
    """Yield overlapping windows of tokens."""
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + window_size, len(tokens))
        chunks.append(tokens[start:end])
        if end == len(tokens):
            break
        start += step_size  # step_size < window_size creates overlap
    return chunks
```

Overlap (window_size - step_size) should be large enough to preserve context across chunk boundaries. For 2K windows, 200–400 token overlap is typical.

### Hierarchical Processing

For very long documents, use a hierarchical approach:

```
Level 1: Summarize each page/section independently
Level 2: Summarize the level-1 summaries (summaries of summaries)
Level 3: Final synthesis from level-2 summaries

Example for a 500-page document:
- 500 pages × ~1000 tokens each = 500K tokens (too long for context)
- Summarize each page to 100 tokens: 500 × 100 = 50K tokens (still too long)
- Group pages into 50-page sections, summarize each: 10 × ~500 = 5K tokens ✓
- Synthesize final answer from 10 section summaries
```

```python
def hierarchical_summarize(pages: list[str], llm) -> str:
    # Level 1: page summaries
    page_summaries = [
        llm.complete(f"Summarize in 2-3 sentences:\n\n{page}")
        for page in pages
    ]
    
    # Level 2: group and summarize
    groups = [page_summaries[i:i+20] for i in range(0, len(page_summaries), 20)]
    group_summaries = [
        llm.complete(f"Synthesize these page summaries:\n" + "\n".join(group))
        for group in groups
    ]
    
    # Level 3: final synthesis
    return llm.complete(
        f"Provide a comprehensive summary from these section summaries:\n"
        + "\n".join(group_summaries)
    )
```

**Information loss**: Hierarchical summarization loses detail at each level. Critical details mentioned once in the middle of a document may be dropped. For tasks requiring precise fact retrieval, RAG is preferable to hierarchical summarization.

---

## Context Compression

Rather than removing content entirely, context compression techniques condense it while preserving key information.

### Selective Context

Anthropic et al. (2023) — use a small compression LLM to score each sentence's relevance to the query, then drop low-relevance sentences:

```python
def selective_compress(context: str, query: str, target_ratio: float = 0.5) -> str:
    sentences = sent_tokenize(context)
    
    # Score each sentence's relevance to the query
    scores = []
    for sent in sentences:
        score_prompt = (
            f"Query: {query}\nSentence: {sent}\n"
            "Rate this sentence's relevance to the query from 0.0 to 1.0. "
            "Output only the number."
        )
        score = float(small_llm.complete(score_prompt))
        scores.append(score)
    
    # Keep top target_ratio fraction by relevance score
    threshold = sorted(scores, reverse=True)[int(len(scores) * target_ratio)]
    compressed = " ".join(s for s, sc in zip(sentences, scores) if sc >= threshold)
    return compressed
```

### LLMLingua

Jiang et al. (2023) — **LLMLingua** uses a small LM (LLaMA-7B) to compute the perplexity of each token given its context. Low-perplexity tokens (easily predictable) are more compressible. The algorithm:

1. Compute token-level perplexity with the compression model.
2. Use iterative search to find a compressed prompt that preserves the overall perplexity distribution.
3. Achieve 3–20× compression ratio with minimal task performance loss.

LLMLingua-2 improves on this by training a dedicated compression model.

```python
# Using LLMLingua (pip install llmlingua)
from llmlingua import PromptCompressor

compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    use_llmlingua2=True
)

compressed = compressor.compress_prompt(
    prompt,
    rate=0.33,  # Compress to 33% of original token count
    force_tokens=["\n", ".", "?"],  # Never compress these tokens
)
# compressed["compressed_prompt"] is 3× shorter
```

### Conversation History Summarization

In long multi-turn conversations, compressing old turns preserves context while freeing space for new content:

```python
def compress_history(
    messages: list[dict],
    keep_recent_n: int = 4,
    llm=None
) -> list[dict]:
    if len(messages) <= keep_recent_n:
        return messages
    
    to_compress = messages[:-keep_recent_n]
    recent = messages[-keep_recent_n:]
    
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in to_compress
    )
    
    summary = llm.complete(
        f"Summarize this conversation history concisely, preserving "
        f"key facts, decisions, and context needed to continue:\n\n{history_text}"
    )
    
    return [
        {"role": "system", "content": f"[Previous conversation summary]: {summary}"}
    ] + recent
```

---

## Long-Context Model Architectures

Extending the context window beyond training length requires architectural changes to handle the computational and memory costs of long-range attention.

### FlashAttention

Dao et al. (2022) — FlashAttention does not change the mathematical result of attention but dramatically improves hardware efficiency by restructuring the computation to minimize HBM (GPU memory) reads and writes.

Standard attention reads $Q$, $K$, $V$ from HBM repeatedly. FlashAttention tiles the computation so that each tile stays in SRAM (fast GPU on-chip memory), reducing HBM accesses from $O(n^2 d)$ to $O(nd)$.

**FlashAttention-2** and **FlashAttention-3** (2024) improve further by better parallelism over the sequence dimension and support for H100's FP8.

### Ring Attention

Liu et al. (2023) — Ring Attention distributes the attention computation across multiple devices by arranging them in a ring. Each device holds a shard of $K$ and $V$. In each ring step, each device:
1. Computes attention for its local $Q$ against the current $K, V$ shard.
2. Passes the $K, V$ shard to the next device in the ring.

This allows $O(n/D)$ memory per device (where $D$ = number of devices) and enables context lengths proportional to the number of devices. Context lengths of 1M+ tokens become feasible with ring attention on multi-node clusters.

### Sliding Window Attention

Beltagy et al. (2020) — Instead of attending to all positions, each token attends only to a local window of $w$ positions. Global tokens (special CLS-like tokens) attend to all positions. This reduces attention complexity from $O(n^2)$ to $O(n \cdot w)$.

Mistral 7B uses grouped query attention with sliding window attention (window = 4,096 tokens with 8K context). For many tasks, local attention is sufficient because most relevant context is nearby.

---

## RoPE Extension Methods

Most modern LLMs use **Rotary Position Embedding (RoPE)**, which encodes position by rotating query and key vectors. The rotation angle is a function of position and dimension:

$$\mathbf{R}_\theta(x) = x \cdot \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix}$$

When inference context length exceeds the training length, position indices go beyond the training distribution, and the model's position-sensitive behavior degrades. Extension methods adapt RoPE to work beyond training length without retraining.

### NTK-Aware Scaling

Press et al. (2021) / bloc97 (2023) — Scale the base of RoPE (analogous to changing the frequency basis) so that the effective "resolution" of position encoding is stretched to cover longer sequences.

```python
# Standard RoPE base = 10000
# Extend to 4× longer context:
new_base = old_base * (context_extension_factor ** (dim / (dim - 2)))
# For dim=128, 4× extension: new_base = 10000 * (4 ** (128/126)) ≈ 41,176
```

### YaRN (Yet Another RoPE Extensibility Method)

Peng et al. (2023) — YaRN combines NTK-aware scaling with a dynamic temperature scaling that adjusts attention entropy at long contexts. It divides RoPE dimensions into groups and applies different scaling factors to each group based on their wavelength. YaRN is the method used to extend LLaMA 2 from 4K to 32K+ context.

```
Dimensions with short wavelengths (high frequency) → interpolate (scale position)
Dimensions with long wavelengths (low frequency) → extrapolate (no change)
```

YaRN requires fine-tuning for 100–400 steps on long-context data after scaling to recover quality.

---

## Multi-Document QA Strategies

Answering questions that require synthesizing information across many documents is one of the hardest long-context tasks.

### Map-Reduce

Process each document independently, then combine results:

```python
def map_reduce_qa(query: str, documents: list[str], llm) -> str:
    # Map: extract relevant information from each document
    extractions = []
    for doc in documents:
        extraction = llm.complete(
            f"Question: {query}\n\nDocument:\n{doc}\n\n"
            "Extract any information relevant to this question. "
            "If nothing is relevant, output 'No relevant information.'"
        )
        if "No relevant information" not in extraction:
            extractions.append(extraction)
    
    # Reduce: synthesize across documents
    combined = "\n\n---\n\n".join(extractions)
    return llm.complete(
        f"Question: {query}\n\nEvidence from multiple documents:\n{combined}\n\n"
        "Synthesize a comprehensive answer based on the evidence above."
    )
```

**Advantage**: Each map call is independent (parallelizable). No single call needs to fit all documents.
**Disadvantage**: Information that requires synthesis across documents (e.g., "what was the trend across all quarters?") may be lost in individual map calls.

### Document Reranking Before LLM

Use a reranker or BM25 to find the top-k most relevant document chunks, then pass only those to the LLM:

```python
# Don't: pass all 50 documents to the LLM
# Do: retrieve the top 5 most relevant, pass those
relevant_chunks = rerank(query, all_document_chunks, top_k=5)
answer = llm_with_context(query, relevant_chunks)
```

### Iterative Refinement

For very complex multi-hop questions (Q1 depends on the answer to Q0):

```python
def iterative_refinement_qa(query: str, documents: list[str], llm) -> str:
    answer = "I don't know yet."
    
    for doc in documents:  # or sorted by relevance
        answer = llm.complete(
            f"Question: {query}\n\n"
            f"Current best answer: {answer}\n\n"
            f"New document:\n{doc}\n\n"
            "Update the answer if this document provides relevant information. "
            "Otherwise keep the current answer."
        )
    
    return answer
```

---

## KV Cache and Prompt Caching

The KV cache stores the key and value tensors for all previous tokens during autoregressive generation, avoiding recomputation. For a 70B model with 80 layers and GQA, a single 1K-token prefix costs roughly 0.5–2 GB of KV cache memory.

**Prompt caching** (offered by Anthropic and OpenAI) extends KV caching to cross-request reuse: if a large prefix (system prompt, document, example set) is reused across requests, the provider caches the computed KV state and charges ~10% of the normal token price for cache hits.

### Anthropic Prompt Caching

```python
import anthropic

client = anthropic.Anthropic()

# Mark the large document as cacheable
response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "You are a helpful assistant that answers questions about this document."
        },
        {
            "type": "text",
            "text": large_document_text,  # Could be 50K tokens
            "cache_control": {"type": "ephemeral"}  # Cache this block
        }
    ],
    messages=[{"role": "user", "content": user_question}]
)

# First call: full price for all tokens (cache miss)
# Subsequent calls with same document: ~10% price for cached tokens
print(response.usage)
# Usage(cache_creation_input_tokens=50000, cache_read_input_tokens=0, ...)
# After first call:
# Usage(cache_creation_input_tokens=0, cache_read_input_tokens=50000, ...)
```

**Cache lifetime**: 5 minutes on Anthropic (ephemeral). Placing stable content (system prompt, reference documents) at the beginning of the context and dynamic content (user query) at the end maximizes cache hit rates.

### OpenAI Prompt Caching

OpenAI caches automatically for inputs longer than 1,024 tokens. The first 1,024 tokens must be identical across requests for the cache to hit. No explicit API changes are needed — just ensure your system prompt is stable and placed first.

**Cache hit conditions**:
- Same model
- Same token prefix (byte-for-byte identical)
- Cache must not have expired (typically several hours)

**Cost savings**: For applications with long, stable system prompts (e.g., RAG systems where the same retrieved document is queried multiple times, or chatbots with extensive persona definitions), prompt caching can reduce costs by 50–90%.

---

## Conversation Management

Multi-turn conversations accumulate context. Without management, conversations eventually hit the context limit, at which point older turns must be dropped.

### Truncation Strategies

**Sliding window (naïve)**: Keep only the most recent $N$ tokens. Simple but loses all early context.

```python
def truncate_to_budget(messages: list[dict], max_tokens: int, tokenizer) -> list[dict]:
    token_counts = [len(tokenizer.encode(m["content"])) for m in messages]
    total = sum(token_counts)
    
    if total <= max_tokens:
        return messages
    
    # Remove oldest messages (except system) until within budget
    system = [m for m in messages if m["role"] == "system"]
    non_system = [m for m in messages if m["role"] != "system"]
    
    while sum(len(tokenizer.encode(m["content"])) for m in non_system) > max_tokens:
        non_system.pop(0)  # Remove oldest non-system message
    
    return system + non_system
```

**Summarize-then-truncate**: Summarize old turns, prepend summary as a system message, continue.

```python
SUMMARY_THRESHOLD = 0.8  # When context is 80% full, summarize

def manage_conversation(
    messages: list[dict],
    max_tokens: int,
    llm,
    tokenizer
) -> list[dict]:
    current_tokens = sum(len(tokenizer.encode(m["content"])) for m in messages)
    
    if current_tokens < max_tokens * SUMMARY_THRESHOLD:
        return messages
    
    # Summarize everything except the last 4 messages
    to_summarize = messages[:-4]
    recent = messages[-4:]
    
    summary = llm.complete(
        "Summarize the key facts, decisions, and context from this conversation:\n"
        + "\n".join(f"{m['role']}: {m['content']}" for m in to_summarize)
    )
    
    return [
        {"role": "system", "content": f"[Conversation history]: {summary}"}
    ] + recent
```

**Importance-based retention**: Score each message by importance (recency, contains facts/decisions, user explicitly marked important) and drop low-importance messages first.

### Conversation Memory Architecture

Production conversational systems typically use a layered approach:

```
Working Memory (last 4–8 turns)
    ↓ compresses into
Episodic Summary (compressed history of this session)
    ↓ stores in
Long-Term Memory (vector DB of past sessions)
    ↑ retrieved when relevant
```

At the start of each session, retrieve relevant past session summaries from the long-term vector store (based on the current query) and inject them into the system prompt.

---

## References

- Liu, N. et al. (2023). "Lost in the Middle: How Language Models Use Long Contexts." TACL. [Lost-in-the-middle empirical study]
- Dao, T. et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS. [FlashAttention]
- Liu, H. et al. (2023). "Ring Attention with Blockwise Transformers for Near-Infinite Context." ICLR 2024. [Ring Attention]
- Beltagy, I. et al. (2020). "Longformer: The Long-Document Transformer." arXiv:2004.05150. [Sliding window attention]
- Peng, B. et al. (2023). "YaRN: Efficient Context Window Extension of Large Language Models." ICLR 2024. [YaRN RoPE extension]
- Jiang, H. et al. (2023). "LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models." EMNLP 2023. [LLMLingua]
- Li, Y. et al. (2023). "Compressing Context to Enhance Inference Efficiency of Large Language Models." EMNLP 2023. [Selective context]
- Anthropic. (2024). "Prompt Caching." docs.anthropic.com/en/docs/prompt-caching
- OpenAI. (2024). "Prompt Caching." platform.openai.com/docs/guides/prompt-caching

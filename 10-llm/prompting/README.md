# Prompting

Prompting is the primary interface to LLM capabilities — no weight updates needed. By crafting the right input, you can steer a frozen model to classify, reason, extract, generate code, follow complex instructions, and retrieve grounded facts. Understanding prompting is not just engineering craft; it is a window into how LLMs represent and manipulate knowledge.

## Contents

| File | Topics |
|------|--------|
| [prompt-engineering.md](./prompt-engineering.md) | System prompts, role-play, delimiters, few-shot, zero-shot, prompt injection, templates |
| [chain-of-thought.md](./chain-of-thought.md) | CoT, zero-shot CoT, self-consistency, Tree of Thoughts, Program of Thoughts, Least-to-Most |
| [retrieval-augmented-generation.md](./retrieval-augmented-generation.md) | RAG pipeline, chunking, vector stores, hybrid search, re-ranking, GraphRAG |
| [structured-output.md](./structured-output.md) | JSON mode, function calling, constrained decoding, Pydantic parsing, extraction tasks |
| [context-window-management.md](./context-window-management.md) | Long context, lost-in-the-middle, compression, KV cache, conversation management |

## Why Prompting Matters

LLMs are **compressed world models** — they have internalized vast knowledge and capabilities during pre-training. Prompting is the act of querying that model precisely. A poorly specified prompt activates a different distribution of that knowledge than a well-specified one:

- Zero-shot: "What is the capital of France?" works because the fact is strongly encoded.
- Few-shot: Providing examples shifts the in-context distribution toward the desired output format.
- Chain-of-thought: Forcing intermediate steps activates latent reasoning circuits.
- RAG: Supplying retrieved context overrides parametric knowledge with fresh, grounded facts.

The hierarchy from least to most control:

```
Zero-shot → Few-shot → Chain-of-Thought → RAG → Fine-tuning → Pre-training
    ↑ cheaper, faster, more flexible           ↑ more reliable, more expensive
```

Prompting occupies the left side: maximum flexibility, zero compute overhead, but limited reliability. Fine-tuning and pre-training bake behaviors in. Prompting elicits them.

## Navigation

- [← Back to 10-LLM](../README.md)
- [← Previous: Inference](../inference/README.md)
- [→ Next: Alignment](../alignment/README.md)

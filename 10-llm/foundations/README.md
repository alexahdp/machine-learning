# LLM Foundations

The Transformer and its surrounding machinery — tokenization, positional encoding, embeddings, and attention — are not incidental implementation details. They are the design decisions that determine what LLMs can represent, how they process information, and where their fundamental limits lie. Understanding these foundations rigorously is prerequisite to reasoning about architectural tradeoffs, scaling behavior, and failure modes in every subsequent section.

## Contents

| File | Description |
|------|-------------|
| [attention-mechanism.md](./attention-mechanism.md) | Self-attention, scaled dot-product, multi-head attention, cross-attention, masks, and computational complexity |
| [transformer-architecture.md](./transformer-architecture.md) | Full encoder/decoder architecture, layer norm placement, FFN, residuals, decoder-only variant, parameter counting |
| [tokenization.md](./tokenization.md) | BPE, WordPiece, SentencePiece, vocabulary size effects, special tokens, tokenization artifacts |
| [positional-encoding.md](./positional-encoding.md) | Sinusoidal PE, learned PE, relative PE, RoPE, ALiBi, and long-context extrapolation |
| [embeddings.md](./embeddings.md) | Token embedding matrices, static vs contextual embeddings, embedding geometry, sentence embeddings, retrieval |

## Why These Foundations Matter

Every LLM architectural innovation — sparse attention, mixture of experts, state-space alternatives, context extension — is best understood as a response to a specific limitation of the baseline Transformer. The files here establish exactly what that baseline is and why each design choice was made. Without this grounding, later discussions of efficiency, fine-tuning, and alignment remain superficial.

## Navigation

- [← Back to LLM Overview](../README.md)
- [→ Next: Architectures](../architectures/README.md)

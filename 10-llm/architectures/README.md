# LLM Architectures

The original Transformer (Vaswani et al., 2017) proposed an encoder-decoder design for machine translation. Within five years, the field had split into three dominant families — encoder-only, decoder-only, and encoder-decoder — plus emerging alternatives that challenge the attention mechanism itself. This divergence was not arbitrary: each family reflects a different answer to the question *what is the primary task this model must perform?*

Understanding the families architecturally — not just as named checkpoints — is what lets you choose the right model class, reason about fine-tuning strategies, and anticipate scaling behavior.

## Why Architectural Families Diverged

The original full Transformer is a general sequence transducer. For pretraining at scale, the full architecture is often unnecessary overhead:

- **Encoder-only** models dropped the decoder. If you only need rich contextual representations — not generation — the decoder adds parameters and training complexity without benefit. Bidirectional attention over the full input is strictly more expressive than left-to-right attention for understanding tasks.
- **Decoder-only** models dropped the encoder. Autoregressive generation on a single sequence does not need a separate encoding stack. A unified stack with causal masking handles both pretraining (next-token prediction) and generation with the same forward pass. This simplicity scales extremely well.
- **Encoder-decoder** retained the full split for tasks where input and output are structurally distinct (translation, summarization) and where the encoder can attend bidirectionally while the decoder generates autoregressively.
- **Mixture of Experts** is not a separate family but an orthogonal architectural dimension: it replaces dense FFN layers with sparse expert routing, applicable to any of the above families.
- **State Space Models** are a fundamentally different computation graph — replacing quadratic-complexity attention with linear-time recurrences — aiming to escape the O(n²) bottleneck that limits Transformer context length.

## Contents

| File | Description |
|------|-------------|
| [encoder-only.md](./encoder-only.md) | BERT, RoBERTa, ALBERT, DistilBERT, DeBERTa — masked language models for understanding tasks |
| [decoder-only.md](./decoder-only.md) | GPT series, LLaMA, Mistral, Falcon — causal language models and why they dominate |
| [encoder-decoder.md](./encoder-decoder.md) | T5, BART, mT5, UL2 — sequence-to-sequence architectures for structured generation |
| [mixture-of-experts.md](./mixture-of-experts.md) | Sparse MoE layers, routing, load balancing, Switch Transformer, Mixtral |
| [state-space-models.md](./state-space-models.md) | S4, Mamba, linear-time sequence modeling — alternatives and complements to attention |

## Quick Comparison

| Family | Attention Pattern | Primary Training Objective | Dominant Use Case |
|--------|------------------|---------------------------|-------------------|
| Encoder-only | Bidirectional (full) | Masked Language Modeling | Classification, NER, extractive QA |
| Decoder-only | Causal (left-to-right) | Next-token prediction | Text generation, chat, reasoning |
| Encoder-decoder | Bidirectional enc + causal dec | Span corruption / denoising | Translation, summarization, structured gen |
| MoE (modifier) | Any of the above | Same as base family | Scale without proportional compute |
| SSM | Recurrent / linear | Next-token prediction | Very long sequences, streaming inference |

## Navigation

- [← Back to LLM Overview](../README.md)
- [← Previous: Foundations](../foundations/README.md)
- [→ Next: Pre-training](../pretraining/README.md)

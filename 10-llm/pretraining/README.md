# LLM Pre-training

Pre-training is the expensive, foundational phase of large language model development: a model is initialized with random weights and trained on hundreds of billions to trillions of tokens of raw text, consuming thousands of GPU-hours and producing a general-purpose representation of language and world knowledge. Everything that happens afterward — fine-tuning, alignment, prompting — inherits from and is constrained by the quality of this initial training run.

Understanding pre-training is critical for three reasons. First, it determines what the model knows and how robustly it generalizes. Second, design decisions made during pre-training (objective, data mixture, scale, architecture) are largely irreversible — you cannot cheaply fix a bad data mixture after spending $10M on compute. Third, as scaling laws have shown, the relationship between compute, data, and model quality is quantitative and predictable, making pre-training an engineering discipline as much as a research one.

## Why Pre-training is Expensive

A single forward/backward pass for a 70B-parameter model requires roughly $2 \times 70\text{B} \times 4$ bytes ≈ 560GB of memory for parameters and gradients alone, far exceeding any single GPU. Training on 1.4T tokens at batch size 4M tokens requires ~350,000 optimizer steps. The Chinchilla-optimal training run for a 70B model costs on the order of $1–5M in GPU compute at 2024 prices.

## Contents

| File | Description |
|------|-------------|
| [language-modeling-objectives.md](./language-modeling-objectives.md) | CLM, MLM, PLM, span corruption, prefix LM, UL2 — the training signals that shape what LLMs learn |
| [scaling-laws.md](./scaling-laws.md) | Kaplan and Chinchilla laws, compute-optimal allocation, emergent abilities, test-time compute scaling |
| [data-curation.md](./data-curation.md) | Web data, filtering pipelines, deduplication, mixing ratios, synthetic data, contamination |
| [distributed-training.md](./distributed-training.md) | Data/tensor/pipeline parallelism, ZeRO, FSDP, 3D parallelism, Flash Attention, interconnects |
| [training-stability.md](./training-stability.md) | Loss spikes, mixed precision, BF16, gradient clipping, warmup schedules, failure modes, monitoring |

## Key Themes

**Compute budget drives all decisions.** Before training begins, you must decide how many parameters to train and how many tokens to use, given a fixed compute envelope. Chinchilla scaling laws provide a quantitative answer; violating them leads to suboptimal models.

**Data quality is a multiplier on compute.** A well-curated 1T-token dataset produces better models than a poorly filtered 10T-token dataset. Filtering, deduplication, and mixing are as important as architectural choices.

**Distributed systems are not optional.** Training frontier models requires coordinating hundreds to thousands of GPUs across multiple parallelism strategies. Understanding the memory and communication trade-offs is essential for practical implementation.

**Stability is an engineering discipline.** Large-scale training fails in ways that small-scale experiments do not predict: numerical instabilities, hardware failures, data poisoning, and subtle hyperparameter interactions can waste months of compute.

## Navigation

- [← Back to LLM Section](../README.md)
- [← Previous: Architectures](../architectures/README.md)
- [→ Next: Fine-tuning](../fine-tuning/README.md)

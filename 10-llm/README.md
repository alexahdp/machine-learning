# Large Language Models (LLMs)

Large Language Models are neural networks trained on massive text corpora that have demonstrated remarkable capabilities in language understanding, generation, reasoning, and generalization. This section provides comprehensive coverage from the Transformer foundations that power LLMs to practical deployment and alignment considerations.

## Why LLMs Deserve Their Own Section

LLMs represent a paradigm shift in AI: they exhibit **emergent capabilities** (abilities that appear suddenly at scale), generalize across tasks without task-specific training, and have redefined what is possible in natural language processing. Covering them deeply requires understanding both their theoretical foundations and the vast practical ecosystem around training, fine-tuning, evaluating, and deploying them responsibly.

## Prerequisites

Before diving into this section, ensure familiarity with:
- [Neural Network Fundamentals](../05-neural-networks/fundamentals/) — backpropagation, activation functions, optimizers
- [RNN/LSTM architectures](../05-neural-networks/architectures/rnn-lstm.md) — sequential modeling context
- [Transfer Learning](../05-neural-networks/training-techniques/transfer-learning.md) — the paradigm LLMs build on
- Linear algebra (matrix operations, dot products, softmax)

## Learning Path

```
foundations/ → architectures/ → pretraining/ → fine-tuning/
     → evaluation/ → inference/ → prompting/
          → alignment/ → agents/ → multimodal/
```

## Contents

### 1. [Foundations](./foundations/README.md)
Core mathematical and conceptual building blocks that all LLMs are built upon.

| File | Topics |
|------|--------|
| [attention-mechanism.md](./foundations/attention-mechanism.md) | Self-attention, multi-head attention, scaled dot-product, cross-attention |
| [transformer-architecture.md](./foundations/transformer-architecture.md) | Encoder/decoder blocks, layer norm, residual connections, full architecture |
| [tokenization.md](./foundations/tokenization.md) | BPE, WordPiece, SentencePiece, vocabulary design, special tokens |
| [positional-encoding.md](./foundations/positional-encoding.md) | Sinusoidal PE, learned PE, RoPE, ALiBi, length generalization |
| [embeddings.md](./foundations/embeddings.md) | Token embeddings, contextual vs static embeddings, embedding geometry |

### 2. [Architectures](./architectures/README.md)
The major LLM architectural families and their design philosophies.

| File | Topics |
|------|--------|
| [encoder-only.md](./architectures/encoder-only.md) | BERT, RoBERTa, ALBERT, DistilBERT — masked language models |
| [decoder-only.md](./architectures/decoder-only.md) | GPT series, LLaMA, Mistral — causal language models |
| [encoder-decoder.md](./architectures/encoder-decoder.md) | T5, BART, mT5 — sequence-to-sequence models |
| [mixture-of-experts.md](./architectures/mixture-of-experts.md) | MoE layers, routing, Mixtral, Switch Transformer |
| [state-space-models.md](./architectures/state-space-models.md) | S4, Mamba, linear-time alternatives to attention |

### 3. [Pre-training](./pretraining/README.md)
How LLMs are trained from scratch on large corpora.

| File | Topics |
|------|--------|
| [language-modeling-objectives.md](./pretraining/language-modeling-objectives.md) | CLM, MLM, PLM, span masking, denoising objectives |
| [scaling-laws.md](./pretraining/scaling-laws.md) | Kaplan/Chinchilla laws, compute-optimal training, emergent abilities |
| [data-curation.md](./pretraining/data-curation.md) | Web data, filtering, deduplication, data mixing ratios |
| [distributed-training.md](./pretraining/distributed-training.md) | Data/model/pipeline parallelism, ZeRO, FSDP |
| [training-stability.md](./pretraining/training-stability.md) | Loss spikes, gradient norms, mixed precision, checkpointing |

### 4. [Fine-tuning](./fine-tuning/README.md)
Adapting pretrained LLMs to specific tasks and behaviors.

| File | Topics |
|------|--------|
| [supervised-fine-tuning.md](./fine-tuning/supervised-fine-tuning.md) | Full fine-tuning, catastrophic forgetting, learning rate selection |
| [instruction-tuning.md](./fine-tuning/instruction-tuning.md) | Instruction datasets, FLAN, Alpaca, chat templates, format design |
| [parameter-efficient-fine-tuning.md](./fine-tuning/parameter-efficient-fine-tuning.md) | LoRA, QLoRA, adapters, prefix tuning, prompt tuning |
| [rlhf.md](./fine-tuning/rlhf.md) | RLHF pipeline, reward modeling, PPO for LLMs, RLAIF |
| [direct-preference-optimization.md](./fine-tuning/direct-preference-optimization.md) | DPO, IPO, SimPO — reward-model-free alignment |

### 5. [Evaluation](./evaluation/README.md)
Measuring what LLMs know and how well they perform.

| File | Topics |
|------|--------|
| [benchmarks.md](./evaluation/benchmarks.md) | MMLU, HellaSwag, HumanEval, GSM8K, BIG-Bench, LMSYS |
| [generation-metrics.md](./evaluation/generation-metrics.md) | Perplexity, BLEU, ROUGE, BERTScore, BLEURT |
| [llm-as-judge.md](./evaluation/llm-as-judge.md) | LLM-based evaluation, MT-Bench, Chatbot Arena, biases |
| [safety-evaluation.md](./evaluation/safety-evaluation.md) | TruthfulQA, bias benchmarks, red-teaming evaluations |

### 6. [Inference](./inference/README.md)
Making LLMs fast, cheap, and production-ready.

| File | Topics |
|------|--------|
| [decoding-strategies.md](./inference/decoding-strategies.md) | Greedy, beam search, top-k/p, temperature, repetition penalty |
| [kv-cache.md](./inference/kv-cache.md) | KV cache mechanism, memory layout, multi-query attention |
| [quantization.md](./inference/quantization.md) | GPTQ, AWQ, INT8/INT4, bitsandbytes, quality trade-offs |
| [speculative-decoding.md](./inference/speculative-decoding.md) | Draft-target model paradigm, acceptance rates, speedups |
| [batching-and-serving.md](./inference/batching-and-serving.md) | Continuous batching, PagedAttention, vLLM, TRT-LLM |

### 7. [Prompting](./prompting/README.md)
Eliciting the best behavior from LLMs without changing weights.

| File | Topics |
|------|--------|
| [prompt-engineering.md](./prompting/prompt-engineering.md) | System prompts, role-play, delimiters, few-shot, common patterns |
| [chain-of-thought.md](./prompting/chain-of-thought.md) | CoT, zero-shot CoT, tree of thoughts, self-consistency |
| [retrieval-augmented-generation.md](./prompting/retrieval-augmented-generation.md) | RAG pipeline, chunking, vector stores, hybrid search, re-ranking |
| [structured-output.md](./prompting/structured-output.md) | JSON mode, function calling, constrained generation, tool schemas |
| [context-window-management.md](./prompting/context-window-management.md) | Long context, chunking strategies, context compression, lost-in-the-middle |

### 8. [Alignment](./alignment/README.md)
Making LLMs safe, honest, and helpful.

| File | Topics |
|------|--------|
| [alignment-fundamentals.md](./alignment/alignment-fundamentals.md) | Value alignment, reward hacking, Goodhart's law, RLHF motivation |
| [hallucination.md](./alignment/hallucination.md) | Types, causes, detection, mitigation, faithfulness vs factuality |
| [safety-techniques.md](./alignment/safety-techniques.md) | Constitutional AI, red-teaming, output filtering, refusal training |
| [bias-and-fairness.md](./alignment/bias-and-fairness.md) | Dataset bias, model bias, fairness metrics, debiasing approaches |

### 9. [Agents](./agents/README.md)
LLMs as autonomous reasoning and action systems.

| File | Topics |
|------|--------|
| [tool-use.md](./agents/tool-use.md) | Function calling, tool schemas, tool selection, error recovery |
| [agentic-frameworks.md](./agents/agentic-frameworks.md) | ReAct, Reflexion, plan-and-execute, scratchpad reasoning |
| [memory-systems.md](./agents/memory-systems.md) | In-context, external, episodic/semantic memory, retrieval |
| [multi-agent-systems.md](./agents/multi-agent-systems.md) | Agent collaboration, debate, role specialization, orchestration |

### 10. [Multimodal](./multimodal/README.md)
LLMs extended to images, audio, and beyond text.

| File | Topics |
|------|--------|
| [vision-language-models.md](./multimodal/vision-language-models.md) | CLIP, BLIP, LLaVA, GPT-4V, Flamingo — architectures and training |
| [image-generation.md](./multimodal/image-generation.md) | Diffusion models, DALL-E, Stable Diffusion, text-to-image overview |
| [audio-language-models.md](./multimodal/audio-language-models.md) | Whisper, audio tokenization, speech LLMs, voice interfaces |

## Key Themes Across the Section

**Scale is qualitatively different**: LLMs exhibit emergent capabilities that smaller models lack — abilities that appear suddenly once a compute/parameter threshold is crossed.

**The alignment tax**: More capable models are not automatically safer or more aligned. Safety and capability must be co-optimized.

**Inference is the bottleneck**: Training happens once; inference happens billions of times. Efficiency at inference time has enormous practical impact.

**Evaluation is unsolved**: Benchmarks get saturated quickly. The field constantly develops new evaluation paradigms (LLM-as-judge, human preference, contamination-aware evals).

## Navigation

- [← Back to Main](../README.md)
- [← Previous: Resources](../09-resources/README.md)

### Other Sections
- [Neural Networks](../05-neural-networks/README.md)
- [Famous Architectures](../06-famous-architectures/README.md)
- [Advanced Topics](../07-advanced-topics/README.md)
- [Practical Guides](../08-practical-guides/README.md)

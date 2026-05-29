# LLM Inference

Training an LLM is a one-time cost; inference runs continuously in production, serving every user request. At scale, inference costs dwarf pre-training costs by orders of magnitude — a model trained for $10M might generate $100M in annual inference compute. Understanding inference efficiency is therefore not an academic concern but the central engineering problem of deploying LLMs.

The core challenge: a 70B parameter model in FP16 requires 140 GB of memory just to store weights. A single forward pass for one token reads those 140 GB from GPU memory. Memory bandwidth, not arithmetic throughput, is the bottleneck. Every technique in this section — KV caching, quantization, speculative decoding, continuous batching — exists to better utilize memory bandwidth and reduce the effective bytes-per-token generated.

## The Inference Stack

```
User Request
     │
     ▼
┌─────────────────────────────────┐
│   Serving Framework             │  (vLLM, TGI, TRT-LLM, Ollama)
│   - Request scheduling          │
│   - Continuous batching         │
│   - PagedAttention              │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│   Model Execution               │
│   - Quantized weights           │
│   - KV cache management         │
│   - Attention computation       │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│   Decoding                      │
│   - Sampling strategy           │
│   - Token selection             │
│   - Stopping criteria           │
└──────────────┬──────────────────┘
               │
               ▼
         Token Stream
```

## Contents

| File | Description |
|------|-------------|
| [decoding-strategies.md](./decoding-strategies.md) | How tokens are selected: greedy, beam search, temperature, top-k/p, nucleus sampling, repetition penalties, constrained decoding |
| [kv-cache.md](./kv-cache.md) | Key-value cache mechanics, memory layout, MQA/GQA attention variants, PagedAttention, prefix caching |
| [quantization.md](./quantization.md) | Reducing model precision: INT8/INT4, GPTQ, AWQ, NF4, GGUF formats, bitsandbytes, quality trade-offs |
| [speculative-decoding.md](./speculative-decoding.md) | Draft-verify paradigm for 2–3x speedup: acceptance-rejection sampling, Medusa, EAGLE, tree drafting |
| [batching-and-serving.md](./batching-and-serving.md) | Serving infrastructure: continuous batching, throughput/latency trade-offs, vLLM, TensorRT-LLM, TGI, scheduling |

## Key Metrics

| Metric | Definition | Typical Target |
|--------|-----------|----------------|
| **TTFT** | Time to First Token — latency from request to first output token | < 500 ms for interactive |
| **TBT** | Time Between Tokens — inter-token latency during streaming | < 50 ms for smooth UX |
| **Throughput** | Tokens generated per second across all requests | Maximize for batch jobs |
| **GPU utilization** | Fraction of available compute used | > 80% for good efficiency |
| **Memory utilization** | KV cache + weights / total VRAM | > 85% for good packing |

## Why Inference Efficiency Matters

**Cost structure**: At steady-state deployment, inference is the dominant cost. OpenAI reportedly spends ~$700,000/day on inference for ChatGPT. Reducing token generation cost by 2x doubles your effective capacity.

**Latency requirements**: Interactive applications require < 50 ms inter-token latency for a fluid streaming experience. Meeting this constraint with large models requires hardware-aware optimization.

**Hardware constraints**: Most production deployments use expensive A100/H100 GPUs. Running a 70B model on a single node requires 4–8 × 80GB GPUs without quantization. Quantization enables running on 1–2 GPUs.

**Long context**: Serving 128K token contexts makes memory management the primary constraint, not computation. KV cache at 128K for a 70B model can exceed 100 GB.

## Navigation

- [← Back to 10-LLM](../README.md)
- [← Previous: Evaluation](../evaluation/README.md)
- [→ Next: Prompting](../prompting/README.md)

# Batching and Serving

## Table of Contents

1. [Static Batching](#1-static-batching)
2. [Dynamic Batching](#2-dynamic-batching)
3. [Continuous Batching](#3-continuous-batching)
4. [PagedAttention: Scheduling Perspective](#4-pagedattention-scheduling-perspective)
5. [Serving Frameworks](#5-serving-frameworks)
6. [Throughput vs Latency Trade-off](#6-throughput-vs-latency-trade-off)
7. [Key Metrics](#7-key-metrics)
8. [Tensor Parallelism at Inference](#8-tensor-parallelism-at-inference)
9. [Request Scheduling](#9-request-scheduling)
10. [Prompt Caching in Serving Systems](#10-prompt-caching-in-serving-systems)
11. [Practical: vLLM Server](#11-practical-vllm-server)
12. [References](#references)

---

## 1. Static Batching

The naive approach: collect requests into a fixed-size batch, pad all sequences to the maximum length in the batch, run the batch through the model, wait for all sequences to finish before accepting new requests.

```
Batch of 4 requests (max length = 512):
┌─────────────────────────────────────────────────────┐
│ Request A: [tokens: 50 ] [PAD x 462               ] │
│ Request B: [tokens: 512                             ] │
│ Request C: [tokens: 100] [PAD x 412               ] │
│ Request D: [tokens: 30 ] [PAD x 482               ] │
└─────────────────────────────────────────────────────┘
                     ↓
            Run 512 attention steps
                     ↓
         A finishes at step 50, but must wait
         D finishes at step 30, but must wait
         All wait for B to finish at step 512
```

**Problems**:
- **Padding waste**: if request A has 50 tokens and request B has 512, request A's slots are padded for 462 steps — compute is wasted on padding tokens
- **Blocking**: new requests cannot enter until the entire batch finishes; requests A, C, D are done early but hold their GPU slots until B finishes
- **Unpredictable latency**: a single long request in the batch delays all other requests in the batch
- **Low GPU utilization**: padding and idle waiting mean effective utilization often 30–50%

**Where it's still used**: Offline batch processing (generating embeddings, evaluating a test set) where latency doesn't matter and all requests have similar known lengths.

---

## 2. Dynamic Batching

Group requests with similar lengths to reduce padding waste. Group requests in time windows, releasing batches when the window closes or when the batch reaches maximum size.

**Length bucketing**: assign requests to buckets by input length (e.g., 0–128, 128–256, 256–512, 512+). Requests within a bucket are batched together, minimizing padding.

**Time-window batching**: hold requests for a short window (e.g., 20 ms), then release the batch. Longer windows increase throughput (more requests per batch) but increase TTFT (requests wait longer before processing starts).

**Limitations of dynamic batching**: Still inherits the core problem of static batching — once a batch starts, it runs to completion before new requests enter. A burst of new requests must wait for the current batch to finish.

---

## 3. Continuous Batching

**Continuous batching** (also called "iteration-level scheduling" or "in-flight batching") was proposed in Orca (Yu et al., 2022). It is the foundational scheduling technique enabling high-throughput LLM serving.

**Core insight**: Instead of batching at the request level (batch = set of requests that run together start-to-finish), batch at the **iteration level** (batch = set of requests executing their current generation step).

**How it works**:

```
Time →     Step 1    Step 2    Step 3    Step 4    Step 5    Step 6
           ─────────────────────────────────────────────────────────
Request A: [run     ][run     ][run     ][DONE    ][        ][        ]
Request B: [run     ][run     ][run     ][run     ][run     ][DONE    ]
Request C:           [run     ][run     ][run     ][DONE    ][        ]
Request D:                               [run     ][run     ][run     ]

Batch:      {A,B}    {A,B,C}  {A,B,C}  {A,B,C,D} {B,C,D}  {B,D}
```

When request A finishes at step 4, a new request D immediately takes its slot in the batch at step 4. The GPU never waits for a slow request to clear before accepting new work.

**GPU slot**: In practical implementation, each "slot" corresponds to preallocated KV cache memory. When a request finishes, its KV cache memory is freed and the slot is available for a new request.

**Implementation requirements**:
1. Iterate at the token level (the scheduler runs after every generation step)
2. Non-padding batching: requests at different positions must be handled without uniform padding
3. KV cache management: different requests have different KV cache lengths within the same batch

**Throughput improvement**: Orca reports 36.9× throughput improvement over the Triton Inference Server (which used static batching) on 67B GPT models.

**Why this works mathematically**: GPU utilization with static batching is bounded by $\text{length}_{\text{min}} / \text{length}_{\text{max}}$ (due to idle slots for short requests). With continuous batching, slot utilization approaches the theoretical maximum because slots are immediately reassigned.

---

## 4. PagedAttention: Scheduling Perspective

(Full mechanics in [kv-cache.md](./kv-cache.md); this section focuses on scheduling implications.)

Without PagedAttention, KV cache must be allocated contiguously per request and pre-allocated at the start. This prevents the scheduler from knowing in advance how much memory a request will need (output length is unknown).

**PagedAttention enables**:

**1. On-demand KV allocation**: KV blocks are allocated one block at a time as the sequence grows. No over-allocation.

**2. Preemption**: If the scheduler runs out of KV memory (too many concurrent requests), it can preempt (pause) low-priority requests by swapping their KV blocks to CPU memory and reassigning those GPU blocks to new requests. The preempted request resumes later by swapping blocks back.

```
GPU KV blocks (8 blocks total):
Before: [A₁][A₂][B₁][B₂][B₃][C₁][FREE][FREE]
         ←── req A ──→  ←──── req B ─────→ ←C→

New high-priority request D arrives, needs 3 blocks:
Preempt C (low priority): swap [C₁] to CPU
Now: [A₁][A₂][B₁][B₂][B₃][D₁][D₂][D₃]
```

**3. Copy-on-write for beam search**: Multiple beams of the same request share KV blocks for the common prefix. Only when beams diverge are blocks copied. For 4-beam search, this can reduce KV memory by 2–3× vs allocating separate blocks per beam.

**4. Prefix sharing across requests**: Two requests with the same system prompt share the physical KV blocks for that prefix. Block reference counting tracks shared blocks; they are freed only when no request references them.

---

## 5. Serving Frameworks

### vLLM

**Architecture**: PagedAttention + continuous batching + OpenAI-compatible REST API.

**Key features**:
- PagedAttention with block manager and scheduler
- Automatic prefix caching (radix tree for longest prefix matching)
- Speculative decoding support
- Tensor parallelism (multi-GPU)
- Multi-model serving
- Async engine for high-throughput concurrent requests

**Strengths**: Best-in-class scheduling; high GPU utilization; well-maintained open-source ecosystem.

**Quantization support**: AWQ, GPTQ, bitsandbytes INT8/INT4, FP8.

**Throughput**: On A100 80GB with LLaMA-2 70B (INT4 AWQ), vLLM achieves ~500–800 tokens/sec at batch sizes of 32–64.

### TensorRT-LLM (NVIDIA)

**Architecture**: NVIDIA's inference optimization library. Compiles models to TensorRT engines with custom CUDA kernels.

**Key features**:
- Highly optimized CUDA kernels for attention, MLP (fused operations)
- In-flight batching (continuous batching)
- INT8/FP8 quantization with calibration
- Speculative decoding
- Mixture of Experts (MoE) optimization
- Deployment via Triton Inference Server

**Strengths**: Highest raw throughput on NVIDIA hardware; best FP8 support (H100); kernel fusion reduces memory bandwidth pressure.

**Weaknesses**: NVIDIA-only; more complex deployment; closed-source custom kernels.

**Throughput comparison**: TensorRT-LLM typically achieves 20–40% higher throughput than vLLM on the same hardware for standard configurations.

### TGI (HuggingFace Text Generation Inference)

**Architecture**: Rust server + Python model executor; designed for HuggingFace Hub models.

**Key features**:
- Continuous batching
- FlashAttention integration
- Tensor parallelism
- Quantization (GPTQ, AWQ, bitsandbytes)
- OpenAI-compatible API
- Streaming via Server-Sent Events

**Strengths**: Easy HuggingFace model integration; production-tested (powers HuggingFace.co inference endpoints); good streaming support.

**Weaknesses**: Less aggressive memory optimization than vLLM; smaller community.

### Ollama

**Architecture**: Wrapper around llama.cpp; designed for developer-friendly local deployment.

**Key features**:
- GGUF model support (CPU + GPU inference)
- REST API
- Model management (pull, run, list)
- macOS/Linux/Windows native support
- Automatic GPU offloading

**Strengths**: Easiest setup for local development; works on consumer hardware (MacBook, gaming PC); M-series Mac GPU support via Metal.

**Weaknesses**: Not designed for high-throughput serving; limited batching; single-user optimization.

**Throughput**: ~15–30 tok/s for 7B Q4 on M3 Max, ~50–100 tok/s on RTX 4090.

### LMDeploy

**Architecture**: OpenMMLab project; TurboMind backend (custom inference engine) + PyTorch backend.

**Key features**:
- TurboMind: optimized CUDA engine, continuous batching
- Blocked KV cache (similar to PagedAttention)
- INT4/INT8 quantization (AWQ, GPTQ)
- Strong support for Qwen, InternLM, LLaMA families

**Strengths**: Competitive throughput with vLLM; good Qwen/InternLM model support; active development.

### Framework Comparison Summary

| Framework | Throughput | TTFT | Ease of Use | CPU support | Best for |
|-----------|-----------|------|-------------|-------------|---------- |
| vLLM | High | Low | Medium | No | Production GPU serving |
| TensorRT-LLM | Highest | Lowest | Complex | No | Max NVIDIA throughput |
| TGI | High | Low | Easy | No | HuggingFace integration |
| Ollama | Low-Med | Medium | Easiest | Yes | Local development |
| LMDeploy | High | Low | Medium | No | Qwen/InternLM |
| llama.cpp | Low | Medium | Easy | Yes | CPU inference, GGUF |

---

## 6. Throughput vs Latency Trade-off

Batching is the central mechanism controlling the throughput-latency trade-off. Larger batches increase throughput; smaller batches (or batch size 1) minimize latency.

**Throughput** (tokens/sec across all requests):
$$\text{Throughput} = \frac{B \times \bar{L}}{T_{\text{batch}}}$$

where $B$ is batch size, $\bar{L}$ is mean output length, $T_{\text{batch}}$ is time to process the batch.

As $B$ increases, $T_{\text{batch}}$ grows sub-linearly (GPU compute is better utilized) → throughput increases. But individual request latency also increases because each request waits for the full batch.

**TTFT vs throughput**:

```
Batch size   TTFT         Throughput
1            ~50 ms       ~50 tok/s
8            ~100 ms      ~300 tok/s
32           ~300 ms      ~800 tok/s
128          ~1200 ms     ~1500 tok/s
```

(Approximate numbers for 70B model on 8×A100; actual values depend on model size, hardware, and request mix.)

**SLA-driven batching**: Production systems set a maximum TTFT SLA (e.g., 500 ms for interactive chat). The scheduler increases batch size until TTFT approaches the SLA ceiling, then stops adding requests to the batch.

**Prefill vs decode bottleneck**:
- **Prefill** (processing the input prompt): compute-intensive, benefits from large batches
- **Decode** (generating output tokens): memory-bandwidth-intensive, less benefit from large batches

Some systems use **chunked prefill**: split long prompts into chunks processed over multiple steps, interleaving prefill with decode to maintain steady throughput and bounded TTFT.

---

## 7. Key Metrics

### TTFT: Time to First Token

$$\text{TTFT} = t_{\text{first token output}} - t_{\text{request received}}$$

Includes: network transmission, queuing time, prompt processing (prefill). The prefill step is proportional to prompt length: longer prompts have higher TTFT.

**Target**: < 500 ms for interactive applications; < 2 s for tolerable async.

**Dominant factor for long prompts**: A 32K-token prompt at 70B requires significant compute for the prefill step — TTFT can reach 10+ seconds without optimization.

### TBT: Time Between Tokens

$$\text{TBT} = t_{x_{t+1}} - t_{x_t}$$

The inter-token latency during decode phase. Determines streaming smoothness.

**Target**: < 50 ms for smooth streaming; < 100 ms is tolerable.

**GPU-determined**: TBT is approximately constant during a session (each decode step is the same computation). It increases as batch size increases (more requests share GPU resources).

### Throughput

$$\text{Throughput} = \frac{\text{total tokens generated}}{\text{wall clock time}}$$

Measured in tokens/second for the entire server (across all concurrent users). The primary metric for batch processing and cost efficiency.

**Relationship**:
$$\text{Throughput} \approx \frac{\text{Batch Size} \times \text{Concurrency}}{\text{TBT}}$$

### GPU Utilization

Fraction of time the GPU is doing useful work vs waiting (for memory, for scheduling). Target: > 80%. Static batching with variable-length sequences: typically 40–60%. Continuous batching with PagedAttention: typically 85–95%.

### Memory Utilization

$$\text{Memory Util} = \frac{\text{KV cache used} + \text{weights}}{\text{total VRAM}}$$

High memory utilization is desirable (more KV cache → more concurrent requests). vLLM targets > 90% utilization.

---

## 8. Tensor Parallelism at Inference

For single-request latency (TTFT and TBT), the bottleneck is sequential computation through the model. **Tensor parallelism** (TP) splits computation across multiple GPUs to reduce latency.

**Column parallelism**: Split weight matrices along the output dimension:

$$Y = XW^\top \approx [X W_1^\top \| X W_2^\top]$$

GPU 1 holds $W_1$; GPU 2 holds $W_2$. Both compute in parallel; results are concatenated (AllGather).

**Row parallelism**: Split weight matrices along the input dimension, sum results across GPUs (AllReduce).

**Standard MLP parallelism** (Megatron-style):
- First linear ($d_{\text{model}} \to 4d_{\text{model}}$): column-parallel
- Second linear ($4d_{\text{model}} \to d_{\text{model}}$): row-parallel, followed by AllReduce
- Net: 2 AllReduce per MLP layer

**Attention parallelism**:
- Split heads across GPUs: each GPU handles $H/N$ heads where $N$ is TP degree
- AllReduce after output projection

**TP scaling**:
- TP-2: 1.7–1.9× TTFT reduction (not 2× due to AllReduce overhead)
- TP-4: 2.5–3.5× TTFT reduction
- TP-8: 4–6× TTFT reduction

**NVLink vs PCIe**: AllReduce is a synchronization operation between GPUs. NVLink (NVIDIA GPU-to-GPU direct link) provides 600 GB/s bandwidth; PCIe provides ~64 GB/s. TP above 2 GPUs is only practical with NVLink.

**When to use TP**: When single-GPU memory is insufficient for the model, or when TTFT must be minimized (interactive applications). Does not improve throughput proportionally (AllReduce overhead).

**Pipeline parallelism**: Alternative to TP — split layers across GPUs (layer 0–20 on GPU 1, layers 21–40 on GPU 2). Reduces memory per GPU; introduces pipeline bubble latency. Less common for inference than TP.

---

## 9. Request Scheduling

### Priority Queues

Production systems serve requests with different priorities:
- Interactive user requests: high priority, tight TTFT SLA
- Background batch jobs: low priority, throughput-optimized
- API-tier pricing: paid tiers may get priority over free tiers

**Weighted fair queuing**: allocate GPU slots proportionally to request priority weights.

**Head-of-line blocking**: long requests at the front of the queue delay short requests. Solutions: prioritize short requests, or segment queues by expected length.

### SLA-Aware Scheduling

Monitor TTFT deadlines per request. If a request is approaching its TTFT SLA, prioritize it. This requires estimating processing time from prompt length and current batch load.

**Deadline-aware scheduling**: Sort the pending queue by deadline. Accept new requests only if they can be served before their deadline given current batch load.

### Preemption

If a high-priority request arrives and all GPU slots are occupied by low-priority requests:
1. Select a low-priority request to preempt
2. Swap its KV cache to CPU memory (requires fast CPU-GPU bandwidth; NVMe-backed tensors at ~3 GB/s is too slow; CPU DRAM at ~50 GB/s is feasible for short swaps)
3. Free its GPU slots
4. Insert the high-priority request
5. Resume the preempted request when slots become available

**vLLM preemption policies**:
- **Swap**: transfer KV cache to CPU DRAM (fast, resumes seamlessly)
- **Recompute**: discard KV cache and recompute from scratch when resumed (slower resume, no CPU memory requirement)

### Anti-Starvation

Low-priority requests can be indefinitely delayed if high-priority traffic is constant. Anti-starvation mechanisms: increase request priority as waiting time grows ("aging"), guarantee minimum throughput per priority class.

---

## 10. Prompt Caching in Serving Systems

System prompts repeated across thousands of requests represent the largest caching opportunity in LLM serving:

```
Every request: [System prompt: 1000 tokens] [User message: 50 tokens]

Without prompt caching: process 1050 tokens per request
With prompt caching:    process 50 tokens per request (1000 tokens cached)
→ 95% reduction in prefill compute for the prompt portion
```

**Implementation (vLLM)**: When processing the first request with a given prefix, store the computed KV blocks in a hash table keyed by token content hash. On subsequent requests with the same prefix, look up and reuse the cached blocks.

**Eviction**: KV cache blocks for prompts are evicted using LRU (Least Recently Used) when memory is full. Frequently-used system prompts stay cached; rare prompts are evicted.

**TTL-based caching** (Anthropic Claude API): Cache entries expire after 5 minutes by default. Users mark cache points with `cache_control` markers. Useful for caching long documents or system prompts for a session.

**Cold start**: On server restart or first request, the cache is empty — first request pays full prefill cost. Warm-up: send dummy requests with the system prompt before opening to traffic.

**Prefix caching effectiveness**: Depends on traffic patterns. If all requests share one system prompt, savings are proportional to system prompt fraction of total context. If requests have unique prefixes (no sharing), prefix caching provides no benefit.

---

## 11. Practical: vLLM Server

### Server Startup

```bash
# Install vLLM
pip install vllm

# Start server (OpenAI-compatible API)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-70b-chat-hf \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \          # split across 2 GPUs
    --max-model-len 4096 \              # max context length
    --gpu-memory-utilization 0.90 \     # use 90% of VRAM for KV cache
    --enable-prefix-caching \           # enable automatic prefix caching
    --max-num-seqs 256 \                # max concurrent sequences
    --port 8000

# With AWQ quantization (fits 70B on single A100):
python -m vllm.entrypoints.openai.api_server \
    --model TheBloke/Llama-2-70B-chat-AWQ \
    --quantization awq \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.95
```

### API Usage (OpenAI-compatible)

```python
from openai import OpenAI

# Point to local vLLM server
client = OpenAI(
    api_key="dummy",  # vLLM doesn't require authentication by default
    base_url="http://localhost:8000/v1",
)

# Standard completion
response = client.chat.completions.create(
    model="meta-llama/Llama-2-70b-chat-hf",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain transformer attention in 3 sentences."},
    ],
    temperature=0.7,
    max_tokens=200,
    stream=False,
)
print(response.choices[0].message.content)

# Streaming response
stream = client.chat.completions.create(
    model="meta-llama/Llama-2-70b-chat-hf",
    messages=[{"role": "user", "content": "Write a haiku about gradient descent."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Monitoring vLLM Performance

```bash
# vLLM exposes Prometheus metrics at /metrics endpoint
curl http://localhost:8000/metrics

# Key metrics:
# vllm:num_requests_running         - current batch size
# vllm:num_requests_waiting         - queue depth
# vllm:gpu_cache_usage_perc         - KV cache utilization
# vllm:prompt_tokens_total          - total input tokens processed
# vllm:generation_tokens_total      - total output tokens generated

# Watch GPU utilization
nvidia-smi dmon -s u -d 1
```

### Benchmarking

```bash
# vLLM benchmarking tool
python benchmarks/benchmark_serving.py \
    --backend vllm \
    --model meta-llama/Llama-2-70b-chat-hf \
    --dataset sharegpt \                  # use ShareGPT for realistic length distribution
    --num-prompts 1000 \
    --request-rate 10 \                   # requests per second
    --port 8000

# Reports: TTFT (mean/P50/P99), TBT (mean/P50/P99), throughput (tok/s)
```

---

## References

- Yu, G. et al. (2022). "Orca: A Distributed Serving System for Transformer-Based Generative Models." *OSDI 2022*. — Continuous batching / iteration-level scheduling.
- Kwon, W. et al. (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention." *SOSP 2023*. — PagedAttention, vLLM.
- Agrawal, A. et al. (2024). "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve." *OSDI 2024*. — Chunked prefill for TTFT/TBT optimization.
- Sheng, Y. et al. (2023). "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU." *ICML 2023*. — CPU/GPU offloading for single-GPU serving.
- NVIDIA TensorRT-LLM. [github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM).
- vLLM. [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm).
- HuggingFace TGI. [github.com/huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference).
- Shoeybi, M. et al. (2019). "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism." — Tensor parallelism patterns for transformer inference.

# Distributed Training

## Table of Contents
1. [Why Distribution is Necessary](#why-distribution-is-necessary)
2. [Data Parallelism and DDP](#data-parallelism-and-ddp)
3. [Tensor Parallelism](#tensor-parallelism)
4. [Pipeline Parallelism](#pipeline-parallelism)
5. [ZeRO: Zero Redundancy Optimizer](#zero-zero-redundancy-optimizer)
6. [FSDP: Fully Sharded Data Parallel](#fsdp-fully-sharded-data-parallel)
7. [3D Parallelism](#3d-parallelism)
8. [Communication Primitives](#communication-primitives)
9. [GPU Interconnect: NVLink and InfiniBand](#gpu-interconnect-nvlink-and-infiniband)
10. [Flash Attention](#flash-attention)
11. [Activation Checkpointing](#activation-checkpointing)
12. [Practical Configurations](#practical-configurations)
13. [References](#references)

---

## Why Distribution is Necessary

A single NVIDIA A100 GPU has 80GB of HBM2e memory. Training a language model requires storing:

| Component | Memory (bytes per parameter) |
|-----------|----------------------------|
| Model parameters (BF16) | 2 |
| Gradients (FP32 master copy) | 4 |
| Optimizer states (Adam: momentum + variance) | 8 |
| **Total** | **~14–16** |

For a 7B-parameter model: $7 \times 10^9 \times 16 \approx 112\text{GB}$ — already exceeds one A100. For a 70B model: ~1.1TB. For a 405B model: ~6.5TB.

Beyond memory, training throughput on a single GPU is insufficient. To train LLaMA 3 70B on 15T tokens in a reasonable time (months, not decades), you need thousands of GPUs running in parallel.

Distributed training thus solves two distinct problems:
1. **Memory**: model state does not fit on one device
2. **Throughput**: training time must be bounded to weeks, not years

Different parallelism strategies address these differently.

---

## Data Parallelism and DDP

### Basic Data Parallelism (DP)

The simplest approach: replicate the full model on $K$ GPUs, partition each global batch into $K$ micro-batches, run forward/backward independently on each GPU, then **all-reduce gradients** across all replicas before the optimizer step.

```
Global batch B
     │
     ├──B/K──► GPU 0 (model replica)
     ├──B/K──► GPU 1 (model replica)
     │                    ...
     └──B/K──► GPU K-1 (model replica)
          ↕ all-reduce gradients
     Single optimizer step, weights updated identically
```

Each GPU's gradient update is mathematically equivalent to computing the gradient over the full batch — the gradient of a sum is the sum of gradients.

**Memory**: no savings — each GPU stores the full model, gradients, and optimizer states.

**Throughput scaling**: near-linear in $K$ if communication is fast. The all-reduce must complete before the optimizer step begins, introducing a synchronization barrier.

### Distributed Data Parallel (PyTorch DDP)

PyTorch's `torch.nn.parallel.DistributedDataParallel` implements DP with two key optimizations:

1. **Gradient bucketing**: rather than all-reducing each gradient tensor individually (high per-message overhead), DDP groups parameter gradients into buckets (default 25MB) and launches all-reduce operations as soon as each bucket is full. This overlaps communication with the backward pass.

2. **Ring all-reduce**: the all-reduce operation is implemented as a ring: each GPU sends its gradient shard to its neighbor, which adds and forwards, until all GPUs have the sum. Ring all-reduce uses $2(K-1)/K$ of the total data bandwidth optimally, approaching 100% utilization at large $K$.

DDP communication overhead: $O(N_{\text{params}} / \text{bandwidth})$ per step, independent of $K$ (ring all-reduce is bandwidth-optimal). At 400Gbps InfiniBand, all-reducing 70B gradients (140GB) takes ~3 seconds — too slow to be negligible at small batch sizes.

**When DDP is sufficient**: model fits on one GPU, and you only need throughput scaling. Up to 7B models on A100-80GB (with mixed precision + optimization).

---

## Tensor Parallelism

Tensor parallelism (TP), developed in Megatron-LM (Shoeybi et al., 2019), partitions individual weight matrices across GPUs. Unlike DP (which replicates the model), TP shards the model itself.

### Column and Row Parallelism

Consider a linear layer $Y = XW$ where $W \in \mathbb{R}^{d \times 4d}$ (the FFN expansion in a Transformer) and $X \in \mathbb{R}^{B \times d}$.

**Column-parallel linear** (split output dimension):

$$W = [W_1 \mid W_2], \quad W_i \in \mathbb{R}^{d \times 2d}$$

$$Y_i = X W_i \quad \text{on GPU } i, \quad \text{no communication needed (X is replicated)}$$

**Row-parallel linear** (split input dimension):

$$W = \begin{bmatrix} W_1 \\ W_2 \end{bmatrix}, \quad X = [X_1 \mid X_2] \quad \text{(split along columns)}$$

$$Y = X_1 W_1 + X_2 W_2, \quad \text{requires all-reduce to sum partial results}$$

In Megatron-LM, FFN blocks use column parallelism for the first linear ($d \to 4d$) and row parallelism for the second ($4d \to d$), with a single all-reduce per FFN block:

```
Input X (replicated on all TP ranks)
  │
  ├──► GPU 0: col-parallel W1 → partial Y1
  └──► GPU 1: col-parallel W2 → partial Y2
                    │
              [col concat Y = [Y1|Y2]]
                    │
  ├──► GPU 0: row-parallel W3_1 → partial Z1
  └──► GPU 1: row-parallel W3_2 → partial Z2
                    │
              [all-reduce: Z = Z1 + Z2]
```

Multi-head attention is parallelized by distributing heads: each TP rank handles $H/T$ attention heads (where $H$ is total heads and $T$ is TP degree).

### Communication Cost

Each FFN block requires 2 all-reduce operations (one per layer in the attention block too). With TP degree $T$, this requires $T$ fast GPUs connected via NVLink (not InfiniBand). TP is effective only within a node (NVLink bandwidth: 600GB/s for A100 NVLink 3.0). Inter-node TP is bandwidth-constrained and typically avoided.

**Memory savings**: parameters are split across $T$ GPUs, reducing per-GPU memory by $\times T$.

---

## Pipeline Parallelism

Pipeline parallelism (PP) partitions the model's layers across GPUs, with each GPU responsible for a contiguous block of layers (a "stage").

```
GPU 0: Layers 0–19    GPU 1: Layers 20–39    GPU 2: Layers 40–59    GPU 3: Layers 60–79
         │                     │                      │                      │
      micro-batch 0 →→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→
         waiting...  micro-batch 0 →→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→
                        waiting...  micro-batch 0 →→→→→→→→→→→→→→→→→→→→
                                        waiting...  micro-batch 0 →→→→
```

### The Pipeline Bubble

Naive pipeline parallelism has a "bubble" — idle time at the beginning and end of each batch when some GPUs are waiting for activations:

$$\text{Bubble fraction} = \frac{P-1}{M + P - 1}$$

where $P$ is the number of pipeline stages and $M$ is the number of micro-batches. As $M \gg P$, the bubble fraction approaches 0.

The global batch is split into $M$ micro-batches; each micro-batch flows through the pipeline sequentially. GPUs execute forward passes for all micro-batches, then backward passes in reverse order (1F1B scheduling in PipeDream-Flush).

### GPipe vs PipeDream

| | GPipe | PipeDream-Flush (1F1B) |
|-|-------|------------------------|
| Forward pass | All $M$ micro-batches before any backward | Interleave 1 forward, 1 backward per micro-batch |
| Peak activation memory | $O(M \times$ activations per stage$)$ | $O(P \times$ activations per stage$)$ |
| Bubble fraction | $(P-1)/(M+P-1)$ | $(P-1)/(M+P-1)$ (same) |
| Communication | Activations + gradients at boundaries | Same |

PipeDream-Flush uses less memory because it does not hold all $M$ micro-batch activations simultaneously — it starts backward passes as soon as enough forward passes have accumulated. This is the scheme used in Megatron-LM.

### Communication

At each pipeline stage boundary, activations ($\text{batch size} \times \text{seq len} \times d_{\text{model}}$) must be transferred from GPU $i$ to GPU $i+1$. This point-to-point communication is typically done over InfiniBand.

PP is best for very large models where even TP within a node is insufficient.

---

## ZeRO: Zero Redundancy Optimizer

ZeRO (Rajbhandari et al., 2020) addresses a fundamental inefficiency of data parallelism: each GPU stores a full redundant copy of optimizer states, gradients, and parameters. For a 70B model with Adam:

- Parameters (BF16): 140GB
- Gradients (FP32): 280GB
- Optimizer states (FP32 momentum + variance): 560GB
- **Total per GPU in DP: ~980GB** (with 8 GPUs: 7.8TB total, but 980GB is stored redundantly on each!)

ZeRO eliminates this redundancy by partitioning across $K$ data-parallel ranks.

### ZeRO Stages

**ZeRO-1** — partition optimizer states:

Each rank $i$ stores optimizer states for parameters $[i \cdot N/K, (i+1) \cdot N/K)$. After the backward pass, gradients are **reduce-scattered** (each rank receives the gradient shard it owns), the optimizer step runs locally, then parameters are **all-gathered** before the next forward pass.

Memory per GPU: $4N + 2N + \frac{8N}{K}$ bytes ≈ $6N$ for large $K$.

**ZeRO-2** — partition optimizer states + gradients:

Gradients are also partitioned. Each rank computes all gradients during backward, then immediately discards gradients not owned by that rank after the reduce-scatter. 

Memory per GPU: $2N + \frac{4N + 8N}{K}$ bytes.

**ZeRO-3** — partition optimizer states + gradients + parameters:

Parameters are also partitioned. Each rank stores only $1/K$ of the parameters. Before each layer's forward pass, parameters are all-gathered from all ranks (and discarded after the layer). During backward, gradients are reduce-scattered.

Memory per GPU: $\frac{2N + 4N + 8N}{K} = \frac{14N}{K}$ bytes. For $K=64$: ~0.2GB per billion parameters — enabling a 70B model (14B bytes) on 64 A100-80GB GPUs.

**Communication overhead** of ZeRO-3: each forward and backward pass requires all-gather operations over all parameters. Total communication volume is $3\times$ that of vanilla DP (forward all-gather + backward all-gather + gradient reduce-scatter). This is the price of the memory savings.

```
ZeRO Memory per GPU (70B model, FP32 optimizer):

             Optimizer states    Gradients    Parameters    Total
ZeRO-0 (DP)    560 GB           280 GB        140 GB       980 GB
ZeRO-1         560/K GB         280 GB        140 GB       560/K+420 GB
ZeRO-2         (560+280)/K GB   280/K GB      140 GB       840/K+140 GB
ZeRO-3         (560+280+140)/K  280/K GB      140/K GB     980/K GB
```

At $K=64$: ZeRO-3 ≈ 15GB per GPU for the 70B model state (plus activations).

---

## FSDP: Fully Sharded Data Parallel

PyTorch's `FullyShardedDataParallel` (FSDP) is PyTorch's native implementation of ZeRO-3. Key differences from DeepSpeed ZeRO:

- **Unit of sharding**: FSDP shards at the module level (each `nn.Module` is a sharding unit), not the entire model monolithically. This allows mixing FSDP-wrapped and non-FSDP modules.
- **Prefetching**: FSDP automatically prefetches parameters for the next layer while computing the current layer, overlapping communication and compute.
- **Mixed precision**: FSDP natively integrates with PyTorch's `autocast` and maintains sharded BF16 parameters with FP32 parameter and optimizer state copies.
- **Checkpointing integration**: `StateDictType` allows saving/loading sharded or unsharded checkpoints.

FSDP is the standard approach for PyTorch-native training at 7B–70B scale without DeepSpeed. For larger scales or more fine-grained control (e.g., ZeRO-Infinity which spills to NVMe/CPU RAM), DeepSpeed provides more flexibility.

---

## 3D Parallelism

For frontier-scale models (100B+), no single parallelism strategy suffices. Megatron-DeepSpeed combines all three:

```
3D Parallelism layout (example: 512 GPUs, TP=8, PP=8, DP=8):

Node boundary        Layer boundary       Data boundary
     ↕                    ↕                    ↕
[TP group: 8 GPUs] × [PP group: 8 stages] × [DP group: 8 replicas]
     └── within node ──┘  └── across nodes ──┘ └── across nodes ──┘
```

**TP** handles weight matrix sharding *within* a node (uses fast NVLink).

**PP** handles layer distribution *across* nodes within a pipeline group (uses InfiniBand for activation transfer between stages).

**DP** handles data parallelism *across* independent pipeline replicas (uses InfiniBand for gradient all-reduce/ZeRO operations).

The hierarchy reflects hardware topology: NVLink within a node is 10–20× faster than InfiniBand between nodes, so TP (most communication-intensive) must stay within a node.

Training Megatron-LM GPT-3 (175B) required 1024 A100 GPUs with TP=8, PP=16, DP=8.

---

## Communication Primitives

Understanding communication primitives is essential for analyzing distributed training bottlenecks.

### Broadcast
One rank sends data to all other ranks.
- Cost: $\alpha + (K-1)\beta M / K$ (ring-optimal)
- Used: parameter broadcast at initialization

### All-Reduce
Every rank contributes a tensor; every rank receives the sum (or other reduction).
- Cost (ring): $2(K-1)/K \cdot M / \text{bandwidth}$
- Used: gradient synchronization in DP

### Reduce-Scatter
Every rank contributes; rank $i$ receives the $i$-th shard of the reduced result.
- Cost: $(K-1)/K \cdot M / \text{bandwidth}$
- Used: ZeRO gradient partitioning

### All-Gather
Every rank contributes a shard; every rank receives the concatenated full tensor.
- Cost: $(K-1)/K \cdot M / \text{bandwidth}$
- Used: ZeRO parameter reconstruction before forward pass

Note: All-Reduce = Reduce-Scatter + All-Gather. ZeRO-3 replaces one All-Reduce with Reduce-Scatter + All-Gather, decomposing the operation to interleave with computation.

### Point-to-Point (Send/Recv)
One rank sends to exactly one other. Cost: $\alpha + M/\text{bandwidth}$.
- Used: pipeline parallelism activation transfer between stages

---

## GPU Interconnect: NVLink and InfiniBand

### NVLink

NVLink is NVIDIA's proprietary high-speed interconnect between GPUs within a node or server. NVLink 4.0 (H100): 900 GB/s bidirectional per GPU pair. A100 NVLink 3.0: 600 GB/s bidirectional.

NVLink uses a mesh topology within a DGX server: each of 8 GPUs can communicate with any other at full bandwidth. This makes within-node TP with 8 GPUs nearly as fast as a single GPU's memory bandwidth.

### NVSwitch

NVSwitch chips on DGX nodes serve as a crossbar switch, enabling any-to-any GPU communication at full NVLink bandwidth simultaneously. Without NVSwitch, each GPU's NVLink bandwidth is shared across its connections.

### InfiniBand

InfiniBand (IB) connects nodes in a cluster. Key specs:
- HDR InfiniBand: 200 Gbps per port (25 GB/s)
- NDR InfiniBand: 400 Gbps per port (50 GB/s)
- Typical DGX A100 server: 8 × HDR IB ports = 1.6 Tbps total

InfiniBand provides **RDMA (Remote Direct Memory Access)**: the NIC transfers data between GPU memory on different nodes without involving the CPU, reducing latency to ~1µs and enabling efficient all-reduce operations at scale.

### Why Bandwidth Matters

An all-reduce of 70B gradients (280GB in FP32) over InfiniBand (NDR, 400 Gbps = 50 GB/s effective) takes:

$$t = 2 \times \frac{280\text{GB}}{50\text{ GB/s}} \approx 11\text{ seconds}$$

At a training step that takes ~30s on 1000 GPUs, 11s of communication is 37% overhead. This is why gradient compression (gradient sparsification, PowerSGD low-rank approximation) is sometimes used, though at the cost of gradient fidelity.

---

## Flash Attention

### The Memory Bottleneck of Standard Attention

Standard self-attention computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

For sequence length $n$ and heads dimension $d_k$, the attention score matrix $S = QK^\top / \sqrt{d_k}$ has shape $n \times n$ and must be materialized in GPU HBM. Memory cost: $O(n^2)$. At $n=8192$ and 80 attention heads: $8192^2 \times 80 \times 2\text{ bytes} \approx 10.7\text{TB}$ — impossible.

In practice, the $n \times n$ matrix is computed at full precision for a single head at a time, but this still requires $8192^2 \times 2 \approx 134\text{MB}$ per head per layer, and storing it for the backward pass multiplies this by the number of layers.

### Flash Attention: Tiled Computation

Flash Attention (Dao et al., 2022) avoids materializing the full $n \times n$ matrix by computing attention in **tiles** that fit in SRAM (the fast on-chip cache, ~20MB on A100).

The key insight: softmax over a vector can be computed incrementally using the **online softmax algorithm** (Milakov & Gimelshein, 2018):

Maintain running max $m$ and normalization constant $\ell$. For each new chunk of $K$ and $V$:

$$m_{\text{new}} = \max(m_{\text{old}}, \max(\mathbf{s}))$$

$$\ell_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}} \ell_{\text{old}} + \sum_j e^{s_j - m_{\text{new}}}$$

$$\mathbf{o}_{\text{new}} = \frac{e^{m_{\text{old}} - m_{\text{new}}} \ell_{\text{old}} \cdot \mathbf{o}_{\text{old}} + \sum_j e^{s_j - m_{\text{new}}} V_j}{\ell_{\text{new}}}$$

Flash Attention tiles both $Q$ (outer loop) and $K, V$ (inner loop) into blocks that fit in SRAM. Each tile's contribution to the output is computed and accumulated without ever writing the $n \times n$ matrix to HBM.

**Memory complexity**: $O(n)$ instead of $O(n^2)$
**Compute complexity**: same $O(n^2 d)$ operations, but with better hardware utilization (fewer HBM reads/writes, which are the bottleneck on modern GPUs)

Flash Attention 2 (2023) and Flash Attention 3 (2024) further optimize the kernel for H100 Tensor Cores and parallel decode, achieving >70% of theoretical A100 FLOP/s utilization (vs ~35% for standard attention).

### Impact on Training

- Enables training with sequences of 8k–32k+ tokens without OOM
- Reduces attention memory from ~30% of training memory to near-zero overhead
- Required for context length scaling (e.g., LLaMA 3 was trained at 8k context length with Flash Attention)
- Integrated natively in HuggingFace Transformers, PyTorch 2.x SDPA, and all major training frameworks

---

## Activation Checkpointing

### The Memory-Compute Trade-off

During the forward pass, intermediate activations (outputs of each layer) must be stored for use in the backward pass (chain rule). For a Transformer with $L$ layers, sequence length $n$, batch size $B$, and hidden size $d$:

$$\text{Activation memory} \approx L \times B \times n \times d \times \text{bytes per element}$$

For LLaMA 70B ($L=80$, $d=8192$), batch size 1, sequence 4096: $80 \times 1 \times 4096 \times 8192 \times 2 \approx 5\text{GB}$ per sample. At batch size 8 per GPU: 40GB just for activations.

### Gradient Checkpointing

Activation checkpointing (a.k.a. gradient checkpointing) discards activations after the forward pass and recomputes them during the backward pass when needed:

```
Standard training:
  Forward: [save act_1, save act_2, ..., save act_L] → loss
  Backward: [use act_L, use act_{L-1}, ..., use act_1]
  
Gradient checkpointing:
  Forward: [save checkpoint at act_k, discard rest] → loss  
  Backward: [reach act_k → re-run forward from checkpoint to get discarded acts] → gradients
```

**Memory**: $O(\sqrt{L})$ if checkpoints are spaced at $\sqrt{L}$ intervals (optimal spacing), plus the cost of the last segment's activations.

**Compute overhead**: requires one additional forward pass per checkpoint segment → ~33% extra compute in the worst case. In practice, 20–25% overhead is typical.

PyTorch: `torch.utils.checkpoint.checkpoint(function, *inputs)` wraps a module or function, discarding its intermediate activations.

In practice, checkpointing is applied at the Transformer block level: only the input to each block is saved, and the full block is recomputed during backward. This reduces activation memory by ~10–20× at a 20–30% compute cost.

---

## Practical Configurations

### DeepSpeed ZeRO-3 Config (70B training)

```json
{
  "zero_optimization": {
    "stage": 3,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true,
    "offload_optimizer": {
      "device": "none"
    },
    "offload_param": {
      "device": "none"
    }
  },
  "bf16": {
    "enabled": true
  },
  "gradient_clipping": 1.0,
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 4
}
```

Key settings:
- `overlap_comm`: pipeline communication with compute — reduces idle time
- `contiguous_gradients`: place gradients in contiguous memory for faster all-reduce
- `reduce_bucket_size`: bucket size for reduce-scatter — larger buckets → better bandwidth utilization but higher peak memory

### PyTorch FSDP Config (70B training)

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

bf16_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,    # gradients in FP32 for stability
    buffer_dtype=torch.bfloat16,
)

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3 equivalent
    mixed_precision=bf16_policy,
    auto_wrap_policy=transformer_auto_wrap_policy(
        transformer_layer_cls={LlamaDecoderLayer}
    ),
    device_id=torch.cuda.current_device(),
    sync_module_states=True,    # broadcast init params from rank 0
)
```

### Megatron-LM Launch Command (3D Parallelism)

```bash
torchrun --nproc_per_node=8 --nnodes=64 \
  pretrain_gpt.py \
  --tensor-model-parallel-size 8 \      # TP=8 (within node)
  --pipeline-model-parallel-size 4 \    # PP=4 (across 4 nodes per pipeline)
  --num-layers 80 \
  --hidden-size 8192 \
  --num-attention-heads 64 \
  --seq-length 4096 \
  --micro-batch-size 1 \
  --global-batch-size 1024 \
  --train-iters 500000 \
  --bf16 \
  --use-flash-attn
```

With 8 GPUs per node × 64 nodes = 512 GPUs total. TP=8 uses one node per tensor group; PP=4 uses 4 nodes per pipeline; DP = 512/(8×4) = 16 replicas.

---

## References

- Shoeybi, M., et al. (2019). *Megatron-LM: Training multi-billion parameter language models using model parallelism.* [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)
- Huang, Y., et al. (2019). *GPipe: Efficient training of giant neural networks using pipeline parallelism.* NeurIPS. [arXiv:1811.06965](https://arxiv.org/abs/1811.06965)
- Rajbhandari, S., et al. (2020). *ZeRO: Memory optimizations toward training trillion parameter models.* SC'20. [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)
- Narayanan, D., et al. (2021). *Efficient large-scale language model training on GPU clusters using Megatron-LM.* SC'21. [arXiv:2104.04473](https://arxiv.org/abs/2104.04473)
- Zhao, Y., et al. (2023). *PyTorch FSDP: Experiences on scaling fully sharded data parallel.* VLDB. [arXiv:2304.11277](https://arxiv.org/abs/2304.11277)
- Dao, T., et al. (2022). *FlashAttention: Fast and memory-efficient exact attention with IO-awareness.* NeurIPS. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Dao, T. (2023). *FlashAttention-2: Faster attention with better parallelism and work partitioning.* ICLR 2024. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
- Chen, T., et al. (2016). *Training deep nets with sublinear memory cost (gradient checkpointing).* [arXiv:1604.06174](https://arxiv.org/abs/1604.06174)
- Narayanan, D., et al. (2019). *PipeDream: Generalized pipeline parallelism for DNN training.* SOSP. [arXiv:1806.03377](https://arxiv.org/abs/1806.03377)

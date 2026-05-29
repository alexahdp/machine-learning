# Quantization

## Table of Contents

1. [Motivation: The Memory Problem](#1-motivation-the-memory-problem)
2. [PTQ vs QAT](#2-ptq-vs-qat)
3. [Weight vs Activation Quantization](#3-weight-vs-activation-quantization)
4. [INT8 Quantization Fundamentals](#4-int8-quantization-fundamentals)
5. [GPTQ](#5-gptq)
6. [AWQ](#6-awq)
7. [NF4 (Normal Float 4)](#7-nf4-normal-float-4)
8. [GGUF and llama.cpp](#8-gguf-and-llamacpp)
9. [bitsandbytes](#9-bitsandbytes)
10. [Perplexity Degradation by Precision](#10-perplexity-degradation-by-precision)
11. [Quantization Granularity](#11-quantization-granularity)
12. [Practical: Loading Quantized Models](#12-practical-loading-quantized-models)
13. [References](#references)

---

## 1. Motivation: The Memory Problem

Modern LLMs are too large for consumer and even prosumer hardware in their native FP32 or FP16 formats:

| Model | Parameters | FP16 Memory |
|-------|-----------|-------------|
| LLaMA-2 7B | 7B | 14 GB |
| LLaMA-2 13B | 13B | 26 GB |
| LLaMA-2 65B | 65B | 130 GB |
| LLaMA-2 70B | 70B | 140 GB |
| GPT-3 | 175B | 350 GB |
| PaLM-2 | ~340B | ~680 GB |

A single consumer GPU (RTX 4090) has 24 GB VRAM. A single server GPU (A100 SXM) has 80 GB. A 70B model in FP16 requires two 80 GB A100s just for weights — before counting KV cache and activations.

Quantization reduces precision of model tensors (weights, activations, or both) from FP16/FP32 to lower-bit formats:

| Format | Bits | Memory reduction vs FP16 |
|--------|------|--------------------------|
| BF16 | 16 | 1× (baseline) |
| INT8 | 8 | 2× |
| INT4 | 4 | 4× |
| INT3 | 3 | ~5.3× |
| INT2 | 2 | 8× |

A 70B model quantized to INT4 requires ~35 GB — feasible on two consumer GPUs or one A100.

---

## 2. PTQ vs QAT

### Post-Training Quantization (PTQ)

Quantize an already-trained FP16 model without any retraining. Requires only a small calibration dataset (typically 128–512 samples) to estimate activation statistics or compute optimal quantization parameters.

**Advantages**: No GPU-hours for retraining; applicable to any pretrained model; fast to apply.

**Disadvantages**: Quality loss is not corrected by the model itself; relies on calibration quality; harder at very low bitwidths (INT2–INT3).

**Methods**: GPTQ, AWQ, SmoothQuant, SPQR.

### Quantization-Aware Training (QAT)

Simulate quantization during training (or fine-tuning) using **straight-through estimators** (STE). During the forward pass, apply fake quantization (quantize then dequantize, keeping the same float format). During the backward pass, treat fake-quantized values as if they were full-precision (the "straight-through" step).

**Advantages**: Model learns to compensate for quantization error; better quality at low bitwidths (INT4 QAT ≈ PTQ INT8 quality possible); enables INT3/INT2 viably.

**Disadvantages**: Requires full or partial retraining (thousands of GPU-hours for large models); requires access to training data or fine-tuning data.

**Methods**: QLoRA (effectively QAT for fine-tuning), PEQA, LLM-QAT (Meta).

**In practice**: Most deployed quantized LLMs use PTQ because retraining 70B models is prohibitively expensive. QAT is used for smaller models or critical deployments where quality is paramount.

---

## 3. Weight vs Activation Quantization

### Weight-Only Quantization

Quantize weight matrices (e.g., $W_Q, W_K, W_V, W_O$, MLP weights). Activations remain in FP16 during the forward pass. Dequantize weights to FP16 on the fly before matrix multiplication.

**Why weights are easier to quantize**:
- Weights are fixed after training — their distribution is known and stable
- Weight distributions are approximately symmetric and normally distributed per channel
- Weights do not have inter-sequence variance (unlike activations, which vary by input)
- Dequantization overhead is small: one INT4 → FP16 conversion per weight before use

**Memory savings**: Full savings from lower precision, since weights are the dominant memory consumer.
**Compute**: Matrix multiplications still occur in FP16 (or FP16 × INT8 with hardware support). Some hardware (NVIDIA Ada/Hopper) supports INT8 GEMM natively.

### Activation Quantization

Quantize the activation tensors (outputs of attention, MLP, etc.) in addition to weights. Enables integer-only arithmetic during the forward pass.

**Why activations are harder**:
- Activations have **outliers**: a small fraction of activation values (~0.1%) are 100× larger than typical values (LLM.int8() paper). These outliers destroy quantization accuracy at INT8 unless handled specially.
- Activation ranges vary across sequences and positions — calibration statistics may not generalize
- Some layers (especially attention softmax outputs) are highly sensitive to quantization

**SmoothQuant** (Xiao et al., 2022) addresses activation outliers by mathematically shifting the quantization difficulty from activations to weights, which are easier to quantize:

$$Y = (X \text{diag}(s)^{-1}) \cdot (\text{diag}(s) W)$$

where $s_j \propto \max(|X_{:,j}|)^{\alpha} / \max(|W_{j,:}|)^{1-\alpha}$. This channel-wise scaling migrates outlier handling to the weight domain.

---

## 4. INT8 Quantization Fundamentals

### Absmax Quantization (Symmetric)

$$Q = \text{round}\!\left(\frac{X}{\Delta}\right), \quad \Delta = \frac{\max(|X|)}{127}$$

Dequantization: $\hat{X} = Q \cdot \Delta$

- Zero point: 0 (symmetric around zero)
- Range: $[-127, 127]$ (or $[-128, 127]$ for asymmetric)
- Works well for symmetric weight distributions

### Zero-Point Quantization (Asymmetric)

$$Q = \text{round}\!\left(\frac{X}{\Delta}\right) + z, \quad \Delta = \frac{\max(X) - \min(X)}{255}, \quad z = -\text{round}\!\left(\frac{\min(X)}{\Delta}\right)$$

Dequantization: $\hat{X} = (Q - z) \cdot \Delta$

- Allows arbitrary $[\min, \max]$ range to be mapped to $[0, 255]$
- Better for asymmetric distributions (e.g., ReLU outputs are always non-negative)
- Requires storing zero-point $z$ per quantized tensor

### Per-Tensor vs Per-Channel Quantization

**Per-tensor**: single $\Delta$ for the entire weight matrix. Fast but poor quality — one outlier channel dominates the scale, crushing precision everywhere else.

**Per-channel**: separate $\Delta_i$ for each output channel $i$ of a weight matrix. 2–5× better quality than per-tensor at INT8 with minimal overhead (store one scale per channel).

**Per-group**: separate $\Delta$ for each group of $g$ consecutive elements along a dimension (typical: $g = 128$). Finer-grained than per-channel. Essential for INT4 quality. More storage overhead (one scale per group).

---

## 5. GPTQ

**GPTQ** (Frantar et al., 2022) is the dominant INT3/INT4 PTQ method for LLMs. It performs **one-shot weight quantization** using second-order information (Hessian).

### Core Idea

Each layer's weight matrix $W$ is quantized column by column. After quantizing column $q$, the quantization error is compensated by adjusting the remaining unquantized columns using the layer's Hessian $H$:

$$\delta W_{:, q'} = -\frac{W_{:, q} - \hat{W}_{:, q}}{[H^{-1}]_{q, q}} \cdot [H^{-1}]_{q, q'} \quad \text{for } q' > q$$

The Hessian $H = 2 X X^\top$ where $X$ is the activation input to that layer over the calibration set. This error compensation ensures that the **output of the quantized layer** on calibration data matches the original as closely as possible, even though individual weights have quantization error.

### OBS Framework

GPTQ is built on the Optimal Brain Surgeon (OBS) framework. OBQ (Optimal Brain Quantization, the direct predecessor) quantizes weights one at a time in a greedy order. GPTQ simplifies by quantizing in fixed column order with batch compensation, reducing compute from $O(d^3)$ per weight to $O(d^2)$ per layer.

### Groupwise Quantization

Rather than a single scale per column, GPTQ uses **group quantization** (group size 128 by default): every 128 consecutive weights in a row share one scale and zero-point. This significantly improves quality at INT4 by reducing the effective quantization range each scale must cover.

**Memory overhead**: For INT4 with group size 128, each group of 128 weights requires 2 bytes for the scale (FP16) + 1 byte for the zero-point. Overhead = $2 + 1$ bytes per 128 weights × 4 bits/weight = 65 bytes overhead per 64 bytes of quantized weights ≈ 4.8% overhead. Effective bitwidth ≈ 4.18 bits.

**Quality at 4-bit GPTQ**:
- LLaMA-2 7B: perplexity increase ~0.3 points (WikiText-2) vs FP16
- LLaMA-2 70B: perplexity increase ~0.1 points — larger models quantize better
- At 3-bit: ~0.8–2.0 points degradation depending on model size

**Speed**: GPTQ takes 1–4 hours to quantize a 70B model on a single A100. Inference uses custom CUDA kernels (ExLlamaV2, AutoGPTQ) for fast INT4 matrix multiplication.

---

## 6. AWQ

**AWQ** (Activation-aware Weight Quantization, Lin et al., 2023) takes a different approach: rather than using Hessians, it identifies which weights are most critical by examining which activation channels have the largest magnitudes.

### Core Insight

Not all weights are equally important for quantization. A weight column $j$ that is multiplied by an activation channel with large magnitude contributes disproportionately to the output. If that column is poorly quantized, the error is amplified by the large activation.

**Salient channel identification**: For each input channel $j$, compute the average activation magnitude over the calibration set: $\bar{a}_j = \mathbb{E}[|x_j|]$. Channels with large $\bar{a}_j$ are salient.

### Per-Channel Scaling

Apply per-channel scaling to protect salient weights:

$$Y = X W^\top = (X \text{diag}(s)) (W \text{diag}(s)^{-1})^\top$$

Scale factor $s_j = \bar{a}_j^\alpha$ for some $\alpha \in [0, 1]$ (typically $\alpha = 0.5$). This increases the magnitude of salient weight columns relative to the quantization scale, effectively giving them more bits of precision. The scaling is absorbed into the input and weight, so no runtime overhead — just quantize $W' = W \text{diag}(s)^{-1}$.

**Optimal $s$ search**: AWQ searches for $s$ by minimizing the quantization error on calibration data using grid search over $\alpha$.

### AWQ vs GPTQ

| Aspect | GPTQ | AWQ |
|--------|------|-----|
| Method | Hessian-based error compensation | Activation-aware scaling |
| Calibration | ~128 samples | ~128 samples |
| Quality at 4-bit | Good | Slightly better on some tasks |
| Quantization time | 1–4 hours (70B) | ~30 min (70B) |
| Speed at inference | Fast with custom kernels | Fast with AWQ kernels |
| Groupwise support | Yes | Yes |

AWQ is preferred when: faster quantization time matters, or when tasks involve instruction following/chat (where it shows edge over GPTQ). GPTQ is preferred when: perplexity on language modeling is the primary metric.

---

## 7. NF4 (Normal Float 4)

**NF4** is a 4-bit data type designed to be **information-theoretically optimal for normally distributed data** (Dettmers et al., 2023, QLoRA paper).

### Motivation

Standard INT4 quantization uses uniformly spaced quantization levels. But weights in LLMs follow approximately normal distributions. Uniform spacing wastes levels on the tails (where few weights live) and under-represents the center (where most weights live).

### Construction

NF4 quantization levels are chosen so that each level represents an equal fraction of a standard normal distribution $\mathcal{N}(0, 1)$:

$$q_i = \Phi^{-1}\!\left(\frac{i}{2^k - 1}\right)$$

where $\Phi^{-1}$ is the inverse CDF of the normal distribution and $k = 4$ for NF4.

This produces 16 quantization levels that are densely packed near zero (where most weights are) and sparse at the extremes. The levels are normalized to $[-1, 1]$.

**Normalization**: Before quantizing, each group of weights is normalized by dividing by the absolute maximum: $W_{\text{norm}} = W / \max(|W|)$. The normalized weights are then mapped to the nearest NF4 level. At inference, dequantize by multiplying by the stored scale.

**Quality advantage**: NF4 achieves better perplexity than INT4 at the same 4-bit budget because its quantization levels match the actual weight distribution. QLoRA fine-tuning with NF4 base model + BF16 adapters achieves ~99% of FP16 fine-tuning quality.

**Used in**: QLoRA (Dettmers et al., 2023), bitsandbytes 4-bit loading.

---

## 8. GGUF and llama.cpp

**GGUF** (GPT-Generated Unified Format) is a file format for storing quantized LLM models for CPU (and GPU-offloaded) inference, used by **llama.cpp**.

### GGUF Format

A binary format containing:
- Model architecture metadata (JSON)
- Tokenizer vocabulary
- Quantized weight tensors
- Tensor metadata (shape, quantization type, scale factors)

Predecessor: GGML format (deprecated). GGUF is self-describing — no separate config file needed.

### Quantization Variants

GGUF supports multiple quantization types, designated by names like `Q4_K_M`:

```
Q<bits>_<variant>_<size>

Bits: 2, 3, 4, 5, 6, 8
Variant:
  (none): basic uniform quantization
  K:      k-quants — quantize in blocks with mixed precision
  S/M/L:  size variant (Small/Medium/Large) — different blocks use different bit depths
```

**Common variants and their effective bitwidths**:

| Variant | Bits per weight | PPL increase (7B) | Notes |
|---------|-----------------|-------------------|-------|
| Q8_0 | 8.5 | ~0.0 | Near-lossless |
| Q6_K | 6.6 | ~0.01 | Excellent quality |
| Q5_K_M | 5.7 | ~0.07 | Very good quality |
| Q4_K_M | 4.8 | ~0.19 | Best quality/size tradeoff |
| Q4_K_S | 4.6 | ~0.26 | Smaller than Q4_K_M |
| Q3_K_M | 3.9 | ~0.56 | Visible quality drop |
| Q2_K | 3.4 | ~1.8 | Significant degradation |

**k-quants**: A block-based quantization scheme where each block of 256 weights uses a mix of INT4 and INT6 (or INT4 and INT8) quantization. Some weights within the block get higher precision. Super-blocks of multiple blocks share a scale factor in FP16.

### CPU Inference

llama.cpp enables inference on CPUs (x86, ARM), enabling:
- Running 7B models on a MacBook Pro (unified memory: 16–96 GB)
- Running 70B models (Q4_K_M) on M2 Ultra Mac Studio (192 GB unified memory)
- Partial GPU offloading: send some layers to GPU, keep rest on CPU

**Throughput on CPU**: ~5–20 tokens/sec for 7B Q4 on modern laptop CPUs (M3 Max: ~25 tok/s for 7B Q4, ~7 tok/s for 70B Q4).

---

## 9. bitsandbytes

**bitsandbytes** is a HuggingFace library for 8-bit and 4-bit quantized LLM loading and inference on NVIDIA GPUs.

### LLM.int8()

Dettmers et al. (2022) found that LLMs have **activation outliers** — a small fraction (0.01%) of activation features that are up to 100× larger than typical values. These outliers emerge at ~6.7B parameters and are consistent across tokens and layers.

**Mixed-precision decomposition**: LLM.int8() handles outliers by decomposing the matrix multiplication:

1. Identify outlier activation columns (those with values > threshold, e.g., 6.0)
2. Extract those columns from both input $X$ and weight $W$
3. Compute the outlier portion in FP16: $Y_{\text{outlier}} = X_{\text{outlier}} W_{\text{outlier}}^\top$
4. Quantize the remaining (non-outlier) portions to INT8 and compute: $Y_{\text{main}} = \text{dequant}(Q(X_{\text{main}}) \cdot Q(W_{\text{main}})^\top)$
5. Sum: $Y = Y_{\text{outlier}} + Y_{\text{main}}$

**Result**: Near-zero perplexity degradation vs FP16 with 2× memory savings, at the cost of ~15–20% throughput reduction (mixed-precision overhead).

### 4-bit Loading

bitsandbytes also supports 4-bit loading with NF4 or FP4 data types, double quantization (quantize the scale factors themselves for additional memory savings), and is the backend for QLoRA.

**Double quantization**: Quantize the FP16 scale factors (one per group of 64 weights) from FP16 to FP8, saving ~0.37 bits per weight. Combined with NF4, achieves ~4.07 effective bits per weight.

---

## 10. Perplexity Degradation by Precision

Perplexity (lower is better) on WikiText-2 for LLaMA family models:

**LLaMA-2 7B**:

| Precision | PPL | Memory |
|-----------|-----|--------|
| FP16 (baseline) | 5.47 | 14 GB |
| INT8 (LLM.int8()) | 5.49 | 7 GB |
| Q4_K_M (GGUF) | 5.66 | 4.1 GB |
| INT4 (GPTQ, g128) | 5.62 | 3.9 GB |
| INT4 (AWQ, g128) | 5.60 | 3.9 GB |
| NF4 (bitsandbytes) | 5.63 | 4.0 GB |
| INT3 (GPTQ, g128) | 6.07 | 3.1 GB |
| Q2_K (GGUF) | 7.39 | 2.9 GB |

**LLaMA-2 70B** (quantizes better due to more parameters):

| Precision | PPL | Memory |
|-----------|-----|--------|
| FP16 | 3.32 | 140 GB |
| INT8 | 3.34 | 70 GB |
| INT4 (GPTQ, g128) | 3.42 | 38 GB |
| INT3 | 3.87 | 29 GB |

**Key observations**:
- INT8 is nearly lossless for any model size
- INT4 is acceptable for most applications (PPL increase < 0.5)
- INT3 causes visible quality drop; use only for memory-critical deployments
- Larger models tolerate quantization better (70B INT4 ≈ 7B INT8 quality)
- Group size 128 is much better than per-column at INT4

---

## 11. Quantization Granularity

| Granularity | Description | Bits overhead | Quality |
|-------------|-------------|---------------|---------|
| Per-tensor | One scale per matrix | Minimal | Poor at INT4 |
| Per-column (per-channel) | One scale per output channel | $d_{\text{out}}$ FP16 values | Good at INT8 |
| Per-group (g=128) | One scale per 128 weights | $d/128$ FP16 values | Excellent at INT4 |
| Per-row (g=1) | One scale per weight | 1:1 with weights | Full FP16 quality but defeats purpose |

**Rule of thumb**: For INT8, per-column is sufficient. For INT4, per-group with $g \leq 128$ is necessary for acceptable quality. For INT3, group size 32–64 is needed.

**Impact of group size on memory overhead**:

For a weight matrix of shape $(d_{\text{in}}, d_{\text{out}})$ with INT4 quantization and group size $g$:
- Quantized weights: $d_{\text{in}} \times d_{\text{out}} \times 4$ bits
- Scale factors: $(d_{\text{in}} / g) \times d_{\text{out}}$ × 16 bits
- At $d_{\text{in}} = d_{\text{out}} = 4096$, $g = 128$: overhead = $32 \times 4096 \times 2$ bytes = 256 KB per matrix vs 8 MB for quantized weights ≈ 3.1% overhead

---

## 12. Practical: Loading Quantized Models

### bitsandbytes INT8

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# INT8 loading
bnb_config_8bit = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    quantization_config=bnb_config_8bit,
    device_map="auto",  # automatically assigns layers to available GPUs
)
```

### bitsandbytes NF4 (4-bit)

```python
bnb_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NF4 vs FP4
    bnb_4bit_use_double_quant=True,     # double quantization for extra savings
    bnb_4bit_compute_dtype=torch.bfloat16,  # compute in BF16 after dequant
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-chat-hf",
    quantization_config=bnb_config_4bit,
    device_map="auto",
)
```

### GPTQ via AutoGPTQ

```python
from auto_gptq import AutoGPTQForCausalLM

# Load a pre-quantized GPTQ model (many available on HuggingFace)
model = AutoGPTQForCausalLM.from_quantized(
    "TheBloke/Llama-2-70B-chat-GPTQ",
    model_basename="model",
    use_safetensors=True,
    trust_remote_code=False,
    device="cuda:0",
    inject_fused_attention=True,     # use ExLlama kernel for faster inference
    quantize_config=None,
)

# Quantize your own model
from auto_gptq import BaseQuantizeConfig

quantize_config = BaseQuantizeConfig(
    bits=4,           # INT4
    group_size=128,   # per-group quantization
    desc_act=False,   # disable activation ordering (faster, slightly lower quality)
)
model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    quantize_config=quantize_config,
)

# Calibrate and quantize
examples = [tokenizer("Example calibration text " * 50, return_tensors="pt")]
model.quantize(examples)
model.save_quantized("./llama-2-7b-chat-gptq")
```

### AWQ via AutoAWQ

```python
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

quant_config = {
    "zero_point": True,    # asymmetric quantization
    "q_group_size": 128,   # group size
    "w_bit": 4,            # 4-bit weights
    "version": "GEMM",     # GEMM kernel (vs GEMV for batch size 1)
}
model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized("./llama-2-7b-chat-awq")
```

### llama.cpp / GGUF (CPU inference)

```bash
# Convert HuggingFace model to GGUF, then quantize
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
make -j

# Convert to GGUF FP16
python convert_hf_to_gguf.py /path/to/llama-2-7b --outtype f16 --outfile llama-2-7b-f16.gguf

# Quantize to Q4_K_M
./llama-quantize llama-2-7b-f16.gguf llama-2-7b-q4_k_m.gguf Q4_K_M

# Run inference
./llama-cli -m llama-2-7b-q4_k_m.gguf -p "The capital of France is" -n 50
```

---

## References

- Dettmers, T. et al. (2022). "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." *NeurIPS 2022*. — Mixed-precision INT8 decomposition.
- Frantar, E. et al. (2022). "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." *ICLR 2023*. — Hessian-based one-shot quantization.
- Lin, J. et al. (2023). "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." *MLSys 2024*. — Salient weight protection.
- Dettmers, T. et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs." *NeurIPS 2023*. — NF4 data type, double quantization.
- Xiao, G. et al. (2022). "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models." *ICML 2023*. — Activation quantization via channel scaling.
- Gerganov, G. (2023). llama.cpp. [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) — GGUF format and CPU inference.
- Hooper, C. et al. (2024). "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization."

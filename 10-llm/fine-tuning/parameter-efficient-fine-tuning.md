# Parameter-Efficient Fine-Tuning (PEFT)

## Table of Contents

1. [Motivation: The Memory Problem](#motivation-the-memory-problem)
2. [Adapter Layers (Houlsby et al.)](#adapter-layers-houlsby-et-al)
3. [Prefix Tuning](#prefix-tuning)
4. [Prompt Tuning](#prompt-tuning)
5. [LoRA: Low-Rank Adaptation](#lora-low-rank-adaptation)
6. [QLoRA: Quantized LoRA](#qlora-quantized-lora)
7. [IA³: Infused Adapter by Inhibiting and Amplifying Inner Activations](#ia-infused-adapter-by-inhibiting-and-amplifying-inner-activations)
8. [Comparison Table](#comparison-table)
9. [Choosing Rank and Target Modules for LoRA](#choosing-rank-and-target-modules-for-lora)
10. [Code Examples: PEFT Library](#code-examples-peft-library)
11. [References](#references)

---

## Motivation: The Memory Problem

Full fine-tuning of a large language model requires storing:
1. **Model parameters** in FP32: $N \times 4$ bytes
2. **Gradients** (same size as parameters): $N \times 4$ bytes
3. **Optimizer states** (AdamW maintains momentum + variance): $N \times 8$ bytes

Total: approximately $N \times 16$ bytes.

For a 70B-parameter model:

$$70 \times 10^9 \times 16 \text{ bytes} \approx 1{,}120 \text{ GB} \approx 1.1 \text{ TB}$$

No single machine has 1.1TB of GPU memory. Even with model parallelism across 16 × 80GB A100s (1.28TB total), there is almost no headroom for activations or data.

PEFT methods address this by training only a small fraction of parameters — typically 0.1–5% of the total — while keeping the pretrained weights frozen. This reduces optimizer state memory in proportion to the fraction of trained parameters, making large-model fine-tuning feasible on a single high-end GPU.

---

## Adapter Layers (Houlsby et al.)

Adapters (Houlsby et al., 2019) were the first widely adopted PEFT method. The idea: insert small trainable **bottleneck modules** into each Transformer layer, and freeze all original parameters.

### Architecture

An adapter module is inserted after the attention output and after the FFN output in each Transformer layer:

```
    Input
      │
  Layer Norm
      │
   Attention ◄── (frozen)
      │
  Add & Norm
      │
  [Adapter] ◄── (trainable)  ←─── inserted here
      │
   Layer Norm
      │
     FFN ◄── (frozen)
      │
  Add & Norm
      │
  [Adapter] ◄── (trainable)  ←─── and here
      │
    Output
```

Each adapter is a two-layer MLP with a bottleneck:

$$\text{Adapter}(h) = h + W_\text{up} \cdot \sigma(W_\text{down} \cdot h)$$

where $h \in \mathbb{R}^d$ is the hidden representation, $W_\text{down} \in \mathbb{R}^{r \times d}$ projects down to dimension $r \ll d$, $\sigma$ is a nonlinearity (GELU), and $W_\text{up} \in \mathbb{R}^{d \times r}$ projects back up. The residual connection ($h + ...$) ensures that at initialization (with $W_\text{up} \approx 0$), the adapter is an identity function — the pretrained model's behavior is exactly preserved at the start of training.

**Parameter count per adapter**: $2rd$ (ignoring bias terms). For $d=4096$, $r=64$: $\approx 524$K parameters per adapter, versus $16.8$M for a full attention block. With 32 layers and 2 adapters per layer: $\approx 34$M adapter parameters out of 7B total — about 0.5%.

### Inference Overhead

Adapters introduce a **latency penalty** at inference: the bottleneck computation is an additional forward pass that cannot be fused with the frozen weights without custom kernels. This is the primary practical weakness of the adapter approach.

---

## Prefix Tuning

Prefix Tuning (Li and Liang, 2021) takes a different approach: instead of inserting modules inside the transformer, it **prepends trainable "soft" token vectors** to the keys and values of every attention layer.

### Mechanism

In standard attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

Prefix tuning prepends $p$ trainable prefix vectors $P_K \in \mathbb{R}^{p \times d}$, $P_V \in \mathbb{R}^{p \times d}$ to the keys and values:

$$\text{PrefixAttention}(Q, K, V) = \text{softmax}\left(\frac{Q[P_K; K]^T}{\sqrt{d}}\right)[P_V; V]$$

Every query position can attend to the prefix tokens, which act as **soft, continuous task-specific context**. The prefix vectors are not actual token embeddings — they are unconstrained real-valued vectors in the key/value space, optimized directly by gradient descent.

**Key properties**:
- Only the prefix vectors ($2 \times p \times d \times L$ parameters, where $L$ is number of layers) are trained
- The frozen model sees the prefix as additional context at every layer, not just the input
- No inference overhead beyond the prefix length (attention cost is $O((n+p)^2)$ vs $O(n^2)$)
- Reparameterization trick: during training, Li and Liang found it helpful to parameterize the prefix through a small MLP to stabilize optimization, then discard the MLP at inference

**Limitation**: Prefix tuning is harder to optimize than adapter or LoRA methods. The prefix occupies context window space, reducing effective sequence length. Performance is competitive with adapters but can be sensitive to prefix length and initialization.

---

## Prompt Tuning

Prompt Tuning (Lester et al., 2021) is a simplified version of prefix tuning that prepends trainable soft tokens **only to the model input** (the embedding layer), rather than to every attention layer's keys and values.

```
[soft_token_1, soft_token_2, ..., soft_token_k, real_token_1, real_token_2, ...]
   trainable                                       frozen
```

This is the **minimal PEFT**: only $k \times d$ parameters are trained (e.g., for $k=100$, $d=4096$: 409K parameters).

**The key finding** (Lester et al.): at sufficient model scale (≥11B parameters), prompt tuning with only 100 soft tokens matches full model fine-tuning quality. At smaller scales, it falls behind.

**Why less expressive than prefix tuning**: The influence of input-layer soft tokens must propagate through all layers via attention. By the time the signal reaches deep layers, it has been mixed with many other representations. Prefix tuning's per-layer injection maintains the signal throughout.

**Advantage**: Prompt tuning enables an elegant **multi-task serving** scheme: a single frozen model can be used for multiple tasks by swapping only the soft prompt vectors (100K parameters) rather than maintaining full model copies.

---

## LoRA: Low-Rank Adaptation

LoRA (Hu et al., 2021) is currently the dominant PEFT method for LLMs. It sidesteps the inference overhead problem of adapters and the optimization difficulty of prefix tuning with a clean, mathematically grounded approach.

### Core Insight

Weight matrices in neural networks are over-parameterized — the intrinsic rank of the updates needed for fine-tuning is much lower than the full parameter matrix dimension. LoRA hypothesizes that the weight update $\Delta W$ learned during fine-tuning has **low intrinsic rank**.

### Mathematical Formulation

For a pretrained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, instead of learning the full update $\Delta W \in \mathbb{R}^{d \times k}$ (with $dk$ parameters), LoRA parameterizes:

$$\Delta W = BA$$

where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$, with rank $r \ll \min(d, k)$.

The modified forward pass becomes:

$$h = W_0 x + \Delta W x = W_0 x + BA x$$

During training, $W_0$ is frozen; only $A$ and $B$ are updated. The number of trainable parameters drops from $dk$ to $r(d + k)$.

**Initialization**: $A$ is initialized with a random Gaussian ($\mathcal{N}(0, \sigma^2)$); $B$ is initialized to zero. This ensures $\Delta W = BA = 0$ at the start of training — identical to the pretrained model's behavior.

**Scaling factor**: The update is scaled by $\frac{\alpha}{r}$, where $\alpha$ is a hyperparameter:

$$h = W_0 x + \frac{\alpha}{r} BA x$$

This scaling makes training more stable across different rank choices. Setting $\alpha = r$ means no scaling (equivalent to the unscaled formulation). In practice, $\alpha$ is often fixed at $\alpha = 2r$ or $\alpha = r$.

### Where to Apply LoRA

The original paper applied LoRA to the query ($W_Q$) and value ($W_V$) projection matrices only. Subsequent work showed applying to all attention and FFN matrices improves performance:

| Matrix | Apply LoRA? | Notes |
|--------|------------|-------|
| $W_Q$ | Yes (standard) | Query projection |
| $W_K$ | Yes (recommended) | Key projection |
| $W_V$ | Yes (standard) | Value projection |
| $W_O$ | Yes (recommended) | Output projection |
| $W_\text{up}$, $W_\text{gate}$ | Optional | FFN weights, larger impact |
| $W_\text{down}$ | Optional | FFN down-projection |
| Embeddings | Rarely | Large memory saving but risky |

**Heuristic**: Start with $Q, K, V, O$ matrices. If performance is insufficient, add FFN matrices. Adding more target modules increases expressiveness at the cost of more parameters.

### Rank Selection

Rank $r$ controls the expressiveness–efficiency trade-off:

| Rank | Parameters (7B model, Q+V only) | Use case |
|------|--------------------------------|---------|
| $r=4$ | ~2M | Lightweight style transfer, simple tasks |
| $r=8$ | ~4M | Most instruction-following tasks |
| $r=16$ | ~8M | Complex multi-turn dialogue |
| $r=32$ | ~16M | Domain adaptation, code |
| $r=64$ | ~32M | Approaching full fine-tuning quality |

Empirically, $r \in [8, 32]$ covers most use cases. Increasing beyond $r=64$ yields diminishing returns — at that point, full fine-tuning or QLoRA is often preferable.

### LoRA Merging at Inference

The key operational advantage of LoRA: **the adapter can be merged into the base model weights before deployment**, eliminating inference overhead entirely.

$$W_\text{merged} = W_0 + \frac{\alpha}{r} BA$$

After merging, the model is identical in architecture to the original — no extra computation per forward pass. Multiple LoRA adapters for different tasks can be maintained separately and merged on demand.

---

## QLoRA: Quantized LoRA

QLoRA (Dettmers et al., 2023) combines LoRA with aggressive quantization of the base model, enabling fine-tuning of very large models on a single GPU.

### The Memory Stack

QLoRA introduces three innovations:

**1. 4-bit NormalFloat (NF4) quantization**

Standard 4-bit quantization (e.g., INT4) assumes a uniform distribution of weights. Neural network weights are approximately normally distributed $\mathcal{N}(0, \sigma^2)$. **NF4** uses quantization levels that are optimal for a normal distribution: the quantile function of $\mathcal{N}(0,1)$ evaluated at equally-spaced intervals gives quantization points that minimize quantization error for normal data.

The NF4 quantization levels (for the normalized range $[-1, 1]$) are:

$$q_i = Q_N^{-1}\!\left(\frac{i}{2^k - 1}\right)$$

where $Q_N^{-1}$ is the quantile function of the standard normal distribution and $k=4$. This places more quantization levels near zero (where most weights cluster) and fewer at the tails.

**2. Double quantization**

The quantization constants themselves (scaling factors used per block of weights) are quantized from FP32 to 8-bit, saving an additional ~0.37 bits per parameter.

**3. Paged optimizers**

When GPU memory is exhausted during training (e.g., during long-sequence gradient accumulation), optimizer states are automatically offloaded to CPU RAM using NVIDIA's unified memory mechanism. This prevents out-of-memory crashes during long training runs.

### Memory Calculation

For a 65B-parameter model:
- **FP16 full fine-tuning**: $65 \times 10^9 \times 16 \text{ bytes} \approx 1{,}040 \text{ GB}$
- **QLoRA (NF4 base + BF16 adapters)**: $65 \times 10^9 \times 0.5 \text{ bytes} \approx 32 \text{ GB}$ (base) + small adapter overhead ≈ **<48 GB total**

QLoRA enables fine-tuning a 65B model on a single 48GB A100 — a reduction of >20x over full fine-tuning.

### Forward Pass in QLoRA

```
Input → [Dequantize W_0 (NF4→BF16)] → Frozen W_0 pass (BF16)
                                      → LoRA: B*A*x (BF16, trainable)
                                      → Add outputs → Continue
```

The base model weights are stored in NF4 but dequantized to BF16 for each forward pass computation. Gradients only flow through the LoRA adapters (BF16), not back through the quantized base weights. This is an approximation — the quantized base introduces a small but acceptable error in the gradient signal.

---

## IA³: Infused Adapter by Inhibiting and Amplifying Inner Activations

IA³ (Liu et al., 2022) is an extremely parameter-efficient PEFT method based on **element-wise rescaling** rather than additive low-rank updates.

For each adapted layer, IA³ introduces learned vectors $l \in \mathbb{R}^d$ that rescale intermediate activations:

- Keys: $K' = l_K \odot K$
- Values: $V' = l_V \odot V$
- FFN activations: $\text{FFN}' = l_\text{FF} \odot \text{FFN}(x)$

The vectors $l$ are initialized to all-ones (identity rescaling). Training adjusts them to suppress or amplify specific dimensions.

**Why element-wise**: This restricts the expressiveness of the adaptation — you can scale dimensions but not rotate or mix them. This is a much smaller parameter space than LoRA ($d$ vs $rd + rk$ per layer), making IA³ useful for **few-shot fine-tuning** where overfitting risk is high.

Parameter count: typically 0.01–0.1% of total model parameters — 10–100x fewer than LoRA.

**Trade-off**: IA³ is less expressive than LoRA and falls behind on tasks requiring significant distribution shift. It shines in low-data regimes (few-shot, <100 examples) where LoRA's additional parameters lead to overfitting.

---

## Comparison Table

| Method | Trainable Params | Inference Overhead | Memory Reduction | Task Performance | Best Use Case |
|--------|-----------------|-------------------|------------------|-----------------|---------------|
| Full Fine-tuning | 100% | None | None | Highest | Small models, ample resources |
| Adapter (Houlsby) | ~0.5-2% | Yes (bottleneck) | Large | Good | When merging is not required |
| Prefix Tuning | ~0.1-1% | Small (longer context) | Large | Good at scale | Multi-task serving |
| Prompt Tuning | ~0.01-0.1% | Small (longer context) | Very large | Good at ≥11B | Multi-task serving, ≥11B |
| LoRA | ~0.5-5% | None (after merge) | Large | Very good | General purpose, default choice |
| QLoRA | ~0.5-5% | None (after merge) | Very large (20x) | Very good | Large models on single GPU |
| IA³ | ~0.01% | None (after merge) | Very large | Moderate | Few-shot fine-tuning, low-data |

---

## Choosing Rank and Target Modules for LoRA

**A practical decision process**:

1. **Start with $r=16$, target $Q, K, V, O$**. This is the default that works well for most instruction-following tasks. Run a training job for 1 epoch.

2. **Evaluate on validation set**. If performance is sufficient, you're done. If not:
   - If the model is not expressive enough: increase $r$ to 32–64, add FFN matrices to targets
   - If overfitting: decrease $r$ to 8, remove FFN matrices, increase weight decay

3. **For domain-specific tasks** (medicine, law, code): use $r=32–64$ and include FFN matrices. The distribution shift from general text to domain text is larger and requires more expressive adapters.

4. **For style/format alignment** (chat formatting, output structure): $r=4–8$ targeting $Q, V$ only is usually sufficient. The pretrained model already knows how to write coherently; it just needs to learn when.

5. **For reasoning tasks** (math, code generation): $r=32–64$ with all attention and FFN matrices. Reasoning improvements appear to require updates distributed across the full computation graph.

**$\alpha$ heuristic**: Set $\alpha = r$ (scaling factor = 1) for stability. Some practitioners prefer $\alpha = 2r$, finding it speeds up convergence. Avoid very large $\alpha/r$ ratios as they amplify the LoRA updates and can destabilize training.

---

## Code Examples: PEFT Library

### LoRA Fine-tuning with PEFT

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import torch

model_name = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                            # rank
    lora_alpha=32,                   # scaling: alpha/r = 2 amplification
    lora_dropout=0.05,               # optional regularization
    target_modules=[                 # apply LoRA to these weight matrices
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",  # include FFN for stronger adaptation
    ],
    bias="none",                     # don't train bias terms
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 23,592,960 || all params: 3,235,512,320 || trainable%: 0.7293

# Training proceeds exactly like full fine-tuning
# The PEFT model wraps the base model and handles the LoRA forward pass
```

### QLoRA Fine-tuning

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import torch

model_name = "meta-llama/Llama-3.1-70B"

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # NF4 quantization
    bnb_4bit_use_double_quant=True,         # double quantization for constants
    bnb_4bit_quant_type="nf4",             # NormalFloat4 (optimal for normal distributions)
    bnb_4bit_compute_dtype=torch.bfloat16,  # compute in BF16 during forward pass
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

# Prepare model for k-bit training:
# - disable caching (incompatible with gradient checkpointing)
# - enable gradient checkpointing to reduce activation memory
# - cast layer norms and LM head to FP32 for stability
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# For 70B: ~40M trainable / 70B total — 0.057%
```

### Merging LoRA Adapters for Zero-overhead Inference

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load base model in FP16 (or BF16) for merging
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    torch_dtype=torch.bfloat16,
    device_map="cpu",  # merge on CPU to avoid GPU OOM
)

# Load the LoRA adapter
model = PeftModel.from_pretrained(base_model, "./lora-adapter-checkpoint")

# Merge LoRA weights into base model: W_merged = W_0 + (alpha/r) * B * A
model = model.merge_and_unload()

# Save the merged model — it's now a standard model with zero inference overhead
model.save_pretrained("./merged-model")
AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B").save_pretrained("./merged-model")
```

### Loading and Running Inference with a LoRA Adapter (without merging)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Load adapter directly — useful when maintaining multiple adapters for different tasks
model = PeftModel.from_pretrained(base_model, "./lora-adapter-checkpoint")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
result = pipe("Explain the Pythagorean theorem.", max_new_tokens=200)
print(result[0]["generated_text"])
```

---

## References

- Houlsby, N. et al. (2019). *Parameter-Efficient Transfer Learning for NLP*. ICML 2019.
- Li, X.L. and Liang, P. (2021). *Prefix-Tuning: Optimizing Continuous Prompts for Generation*. ACL 2021.
- Lester, B. et al. (2021). *The Power of Scale for Parameter-Efficient Prompt Tuning*. EMNLP 2021.
- Hu, E.J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022.
- Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS 2023.
- Liu, H. et al. (2022). *Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning* (IA³). NeurIPS 2022.
- HuggingFace. (2024). *PEFT: State-of-the-art Parameter-Efficient Fine-Tuning*. https://github.com/huggingface/peft

---

*← [Instruction Tuning](./instruction-tuning.md) | → [RLHF](./rlhf.md)*

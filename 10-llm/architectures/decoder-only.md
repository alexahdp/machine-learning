# Decoder-Only Models (Causal Language Models)

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [Causal Masking](#2-causal-masking)
3. [Training Objective](#3-training-objective)
4. [GPT-1, GPT-2, GPT-3: Scaling and Emergence](#4-gpt-1-gpt-2-gpt-3-scaling-and-emergence)
5. [GPT-4](#5-gpt-4)
6. [LLaMA Family](#6-llama-family)
7. [LLaMA 2 and LLaMA 3](#7-llama-2-and-llama-3)
8. [Mistral 7B](#8-mistral-7b)
9. [Falcon](#9-falcon)
10. [Architectural Innovations in Modern LLMs](#10-architectural-innovations-in-modern-llms)
11. [Why Decoder-Only Became Dominant](#11-why-decoder-only-became-dominant)
12. [Loading and Generating with HuggingFace](#12-loading-and-generating-with-huggingface)
13. [References](#references)

---

## 1. Design Philosophy

A decoder-only model is a stack of transformer blocks with a single, unified function: given tokens $x_1, x_2, \ldots, x_{t-1}$, predict the next token $x_t$. The same stack that processes the prompt also generates the continuation, one token at a time, conditioned on everything generated so far.

This is conceptually simpler than the full encoder-decoder architecture: there is no separate encoding stage, no cross-attention, and no separate set of parameters for encoding vs. decoding. The entire model is one self-attention stack with causal masking.

The key constraint — **causal masking** — ensures that when computing the representation of token $t$, the model has no access to tokens $t+1, t+2, \ldots$. This is not a limitation; it is what makes autoregressive generation coherent.

```
Prompt tokens          Generated tokens
[x₁] [x₂] [x₃]   →   [x₄] [x₅] [x₆] ...
  ↑     ↑     ↑
  Each token attends to all previous tokens only.
  Same model weights used for both prompt and generation.
```

---

## 2. Causal Masking

### The Attention Mask

Standard self-attention computes:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

For a sequence of length $n$, $\mathbf{Q}\mathbf{K}^T \in \mathbb{R}^{n \times n}$ is a full attention matrix: every token queries every other token.

Causal masking applies a mask $\mathbf{M}$ where $M_{ij} = -\infty$ if $j > i$ (token $i$ cannot attend to future token $j$), and $0$ otherwise:

$$\text{CausalAttention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^T + \mathbf{M}}{\sqrt{d_k}}\right)\mathbf{V}$$

After softmax, the $-\infty$ entries become exactly 0, so future tokens contribute zero weight. The resulting mask is lower-triangular:

```
Token: 1  2  3  4  5
1:    [1  0  0  0  0]
2:    [1  1  0  0  0]
3:    [1  1  1  0  0]
4:    [1  1  1  1  0]
5:    [1  1  1  1  1]
```

### Parallelism During Training

This masking enables **full parallelism during training**. All $n$ next-token predictions can be computed in a single forward pass: the model predicts $x_2$ from $x_1$, predicts $x_3$ from $x_1, x_2$, etc., simultaneously. Contrast with RNN-style training, where each time step required the previous hidden state.

During **inference**, however, the model generates sequentially: it must produce $x_{t}$ before it can produce $x_{t+1}$. The KV-cache mechanism (see [kv-cache.md](../inference/kv-cache.md)) stores past key/value computations to avoid recomputing them at each step.

---

## 3. Training Objective

### Next-Token Prediction (Causal Language Modeling)

Given a sequence of tokens $x = (x_1, x_2, \ldots, x_n)$, the loss is:

$$\mathcal{L}_{\text{CLM}} = -\frac{1}{n}\sum_{t=1}^{n} \log P(x_t \mid x_1, \ldots, x_{t-1}; \theta)$$

Every token position contributes to the loss (unlike MLM, which only trains on ~15% of tokens). This is simultaneously:

- **Efficient**: no tokens are "wasted" — every position contributes a gradient
- **Well-defined**: the factorization of the joint distribution $P(x_1, \ldots, x_n)$ into a product of conditionals is exact by the chain rule

$$P(x_1, \ldots, x_n) = \prod_{t=1}^{n} P(x_t \mid x_1, \ldots, x_{t-1})$$

### Perplexity

The standard evaluation metric derived from CLM loss:

$$\text{PPL} = \exp\!\left(-\frac{1}{n}\sum_{t=1}^{n} \log P(x_t \mid x_{<t})\right)$$

Lower perplexity = higher probability assigned to the evaluation text = better model. Perplexity is dataset-specific: a model trained on English has low perplexity on English, high on code (unless it was trained on code too).

---

## 4. GPT-1, GPT-2, GPT-3: Scaling and Emergence

### GPT-1 (Radford et al., 2018)

117M parameters, 12 layers, $d=768$, $h=12$. Trained on BooksCorpus (800M words). Introduced the **pretrain-then-fine-tune** paradigm for generative models. Demonstrated that a single pretrained model could be adapted to many downstream tasks via supervised fine-tuning with minimal architecture changes.

### GPT-2 (Radford et al., 2019)

| Variant | Layers | $d$ | Heads | Params |
|---------|--------|-----|-------|--------|
| Small | 12 | 768 | 12 | 117M |
| Medium | 24 | 1024 | 16 | 345M |
| Large | 36 | 1280 | 20 | 762M |
| XL | 48 | 1600 | 25 | 1.5B |

Trained on WebText (40GB, 8M Reddit-filtered documents). Key change: **zero-shot task framing**. GPT-2 showed that with prompting ("TL;DR:" for summarization, "Q: ... A:" for QA), the language model could perform tasks without any fine-tuning. This was the first evidence that prompting might be a viable alternative to fine-tuning.

Architectural differences from GPT-1: Layer norm moved before the attention/FFN sublayers (Pre-LN), weight initialization scaled by $1/\sqrt{N_{\text{layers}}}$ to stabilize deep networks.

### GPT-3 (Brown et al., 2020)

175B parameters, 96 layers, $d=12288$, 96 heads. Trained on ~570GB of filtered text (Common Crawl, WebText2, Books1/2, Wikipedia).

GPT-3's key contribution was demonstrating **in-context learning (ICL)**: by placing a few labeled examples in the prompt (few-shot learning), GPT-3 could perform many tasks without any weight updates, matching fine-tuned smaller models on many benchmarks.

This was not anticipated from GPT-2. It is an **emergent capability** — one that appeared at the GPT-3 scale but not at smaller scales:

```
Zero-shot:   Translate English to French: cheese =>
One-shot:    sea otter => loutre de mer
             cheese =>
Few-shot:    sea otter => loutre de mer
             peppermint => menthe poivrée
             plush girafe => girafe en peluche
             cheese =>
```

The model completes the pattern without gradient updates — pure prompt conditioning.

---

## 5. GPT-4

GPT-4 (OpenAI, 2023) is a closed model with limited architectural disclosure. Known or credibly reported details:

- **Mixture of Experts**: GPT-4 is widely reported to use a sparse MoE architecture, with ~8 experts and top-2 routing, giving ~1.8T total parameters but ~220B active per token. This is unconfirmed by OpenAI.
- **Multimodal input**: GPT-4V accepts image inputs via a vision encoder (likely a ViT-style model) that projects visual patches into the text token embedding space.
- **RLHF + Constitutional AI-style training**: extensive post-training alignment, resulting in substantially safer and more helpful behavior than raw pretraining.
- **128K context window** (GPT-4 Turbo): enabled by positional encoding modifications (likely RoPE with context scaling).

The performance jump from GPT-3.5 to GPT-4 is as much from better data, fine-tuning, and RLHF as from architectural changes.

---

## 6. LLaMA Family

### LLaMA 1 (Touvron et al., 2023)

LLaMA-1 was the first high-quality open-weight model family competitive with GPT-3. The key insight: **more tokens, smaller model** (anticipating Chinchilla laws). LLaMA-7B was trained on 1T tokens; LLaMA-65B on 1.4T tokens — substantially more data than most prior models of similar size.

| Variant | Layers | $d$ | Heads | Params |
|---------|--------|-----|-------|--------|
| LLaMA-7B | 32 | 4096 | 32 | 6.7B |
| LLaMA-13B | 40 | 5120 | 40 | 13B |
| LLaMA-33B | 60 | 6656 | 52 | 32.5B |
| LLaMA-65B | 80 | 8192 | 64 | 65.2B |

**Architectural innovations** over vanilla GPT-style transformer:

1. **RMSNorm** instead of LayerNorm (see §10)
2. **SwiGLU** activation in FFN (see §10)
3. **Rotary Positional Embedding (RoPE)** instead of learned absolute PE (see [positional-encoding.md](../foundations/positional-encoding.md))
4. **Pre-normalization**: norm applied before each sublayer (not after), improving training stability
5. No biases in linear layers (consistent with modern practice)

---

## 7. LLaMA 2 and LLaMA 3

### LLaMA 2 (Touvron et al., 2023b)

| Variant | Context | GQA | Params |
|---------|---------|-----|--------|
| LLaMA-2-7B | 4096 | No | 7B |
| LLaMA-2-13B | 4096 | No | 13B |
| LLaMA-2-70B | 4096 | Yes | 70B |

Key changes from LLaMA-1:

- **Grouped Query Attention (GQA)** for 70B (see §10): reduces KV cache memory without significant quality loss
- **40% more pretraining data**: 2T tokens (vs 1.4T for LLaMA-1-65B)
- **Longer context**: 4096 tokens (vs 2048)
- **Ghost Attention (GAtt)**: a fine-tuning technique that helps the model maintain system prompt instructions across many dialogue turns

LLaMA-2-Chat: instruction-tuned + RLHF variants with strong safety training and conversational ability.

### LLaMA 3 (Meta, 2024)

| Variant | Context | Vocab | Params |
|---------|---------|-------|--------|
| LLaMA-3-8B | 8192 | 128K | 8B |
| LLaMA-3-70B | 8192 | 128K | 70B |
| LLaMA-3.1-405B | 128K | 128K | 405B |

Key changes:

- **128K token vocabulary** (vs 32K in LLaMA 1/2): better tokenization efficiency for code and multilingual text
- **Grouped Query Attention for all sizes** (not just 70B)
- **8192 token context** for base; extended to 128K in LLaMA-3.1 via RoPE scaling
- **15T+ token pretraining corpus**: heavy emphasis on code and multilingual data
- **LLaMA-3.1-405B**: competitive with GPT-4-class models on many benchmarks, first fully open-weights model at this capability level

---

## 8. Mistral 7B

Mistral 7B (Mistral AI, 2023) demonstrated that careful architecture choices let a 7B model outperform LLaMA-2-13B on many benchmarks — a quality-per-parameter breakthrough.

### Sliding Window Attention (SWA)

Standard attention is $O(n^2)$ in sequence length. SWA restricts each token's attention to a window of $W$ previous tokens:

$$A_{ij} = 0 \text{ if } j < i - W$$

With $W = 4096$ in Mistral-7B. Information outside the window is not directly lost: stacking $L$ layers with window $W$ gives an effective receptive field of $L \times W$ tokens. For Mistral-7B ($L=32$, $W=4096$): 131K token receptive field.

Memory complexity drops from $O(n^2)$ to $O(n \times W)$, with $W \ll n$ for long sequences.

**Caveat**: SWA is useful for very long sequences but provides no benefit at sequence lengths $\leq W$. At 4K context (the typical case in 2023-2024), it is effectively equivalent to full attention.

### Grouped Query Attention

Mistral uses $h_q = 32$ query heads and $h_{kv} = 8$ KV heads (groups of 4 queries share one KV pair). This reduces KV cache memory by 4× during inference. See §10 for the full treatment.

### Performance

Mistral-7B outperforms LLaMA-2-13B on MMLU, HellaSwag, and reasoning benchmarks. The efficiency comes from: rolling KV cache (enabled by SWA), GQA, and a well-curated training mix.

---

## 9. Falcon

Falcon (Technology Innovation Institute, 2023) introduced **parallel attention + FFN computation** and **multi-query attention (MQA)**.

### Multi-Query Attention (MQA)

Extreme version of GQA: a single KV head shared across all query heads. If there are $h_q = 71$ query heads, there is $h_{kv} = 1$ KV head. The KV cache memory reduction is $h_q \times$ vs standard MHA.

$$\text{MQA:} \quad \mathbf{K}, \mathbf{V} \in \mathbb{R}^{n \times d_k} \quad \text{(shared across all query heads)}$$

Quality degradation is measurable but small for models trained from scratch with MQA. MQA is therefore most suitable when inference speed and memory are primary concerns.

### Parallel Attention + FFN

Standard transformers compute attention and FFN sequentially within each block:

$$\mathbf{h}' = \text{Attn}(\text{LayerNorm}(\mathbf{h})) + \mathbf{h}$$
$$\mathbf{h}'' = \text{FFN}(\text{LayerNorm}(\mathbf{h}')) + \mathbf{h}'$$

Falcon computes them in parallel:

$$\mathbf{h}' = \text{Attn}(\text{LayerNorm}(\mathbf{h})) + \text{FFN}(\text{LayerNorm}(\mathbf{h})) + \mathbf{h}$$

The two branches can be computed simultaneously on modern hardware, reducing the number of sequential operations per block from 2 to 1. This is a wall-clock speedup, not an FLOPs reduction.

---

## 10. Architectural Innovations in Modern LLMs

These are the key components that differentiate modern LLMs (LLaMA, Mistral, etc.) from vanilla GPT-2.

### RMSNorm vs LayerNorm

**LayerNorm** (Ba et al., 2016):

$$\text{LayerNorm}(\mathbf{x}) = \gamma \cdot \frac{\mathbf{x} - \mu}{\sigma + \epsilon} + \beta$$

where $\mu = \frac{1}{d}\sum_i x_i$ and $\sigma = \sqrt{\frac{1}{d}\sum_i (x_i - \mu)^2}$.

**RMSNorm** (Zhang & Sennrich, 2019):

$$\text{RMSNorm}(\mathbf{x}) = \gamma \cdot \frac{\mathbf{x}}{\text{RMS}(\mathbf{x}) + \epsilon}, \quad \text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_i x_i^2}$$

RMSNorm **removes the mean-centering step** (no $\mu$ computation, no $\beta$ parameter). This is ~15% faster than LayerNorm and works equally well empirically. The intuition: the centering step adds limited regularization benefit; the rescaling to unit norm is what matters for training stability.

### SwiGLU Activation

Standard FFN:

$$\text{FFN}(\mathbf{x}) = \text{ReLU}(\mathbf{x} W_1 + b_1) W_2 + b_2$$

Gated Linear Unit (GLU) family (Dauphin et al., 2017):

$$\text{GLU}(\mathbf{x}, \mathbf{W}, \mathbf{V}) = \mathbf{x}\mathbf{W} \otimes \sigma(\mathbf{x}\mathbf{V})$$

**SwiGLU** (Shazeer, 2020):

$$\text{SwiGLU}(\mathbf{x}) = (\mathbf{x}\mathbf{W} \otimes \text{SiLU}(\mathbf{x}\mathbf{V})) \cdot \mathbf{W}_2$$

where $\text{SiLU}(z) = z \cdot \sigma(z)$ (Sigmoid Linear Unit, also called Swish).

The $\otimes$ operator is element-wise multiplication. The "gate" $\text{SiLU}(\mathbf{x}\mathbf{V})$ modulates the FFN output: when the gate is near 0, the output is suppressed; when near 1, the linear projection passes through. This gating mechanism is more expressive than a simple nonlinearity.

**Implementation note**: SwiGLU requires 3 weight matrices ($W, V, W_2$) instead of 2. To keep total FLOPs constant, the inner dimension is typically scaled from $4d$ to $\frac{8}{3}d \approx 2.67d$.

Shazeer's paper showed SwiGLU consistently outperforms GeLU, ReLU, and other activations on language modeling, explaining its universal adoption in modern LLMs.

### Grouped Query Attention (GQA)

Multi-Head Attention (MHA):
- $h$ query heads, $h$ key heads, $h$ value heads
- KV cache size per token: $2 \times h \times d_k$ = $2d$ (full hidden dim)

Multi-Query Attention (MQA):
- $h$ query heads, 1 key head, 1 value head (shared)
- KV cache size per token: $2 \times 1 \times d_k$ = $\frac{2d}{h}$

Grouped Query Attention (Ainslie et al., 2023):
- $h_q$ query heads, $h_{kv}$ KV heads, $g = h_q / h_{kv}$ queries per KV group
- KV cache size per token: $2 \times h_{kv} \times d_k$

$$\text{GQA: } A_{i}^{(q)} = \text{softmax}\!\left(\frac{\mathbf{q}_i^{(q)} \mathbf{K}^{(\lfloor q \cdot h_{kv} / h_q \rfloor)T}}{\sqrt{d_k}}\right)\mathbf{V}^{(\lfloor q \cdot h_{kv} / h_q \rfloor)}$$

GQA generalizes: MHA is GQA with $h_{kv} = h_q$; MQA is GQA with $h_{kv} = 1$.

LLaMA-3-70B uses $h_q = 64$, $h_{kv} = 8$ → 8× KV cache reduction vs MHA, with minimal quality impact.

```
MHA:  Q₁K₁V₁  Q₂K₂V₂  Q₃K₃V₃  Q₄K₄V₄   (4 KV pairs)
GQA:  Q₁K₁V₁  Q₂K₁V₁  Q₃K₂V₂  Q₄K₂V₂   (2 KV pairs shared)
MQA:  Q₁K₁V₁  Q₂K₁V₁  Q₃K₁V₁  Q₄K₁V₁   (1 KV pair shared)
```

---

## 11. Why Decoder-Only Became Dominant

By 2022-2024, decoder-only models had displaced encoder-decoder models even for tasks (summarization, translation) where encoder-decoder seemed natural. The reasons:

1. **Simplicity scales better**: One unified pretraining objective (CLM) on one architecture with no task-specific structural modifications. More researcher bandwidth goes to data and scale.

2. **In-context learning**: CLM pretraining naturally produces models that can follow instructions and examples in the context window. This capability does not emerge as readily from MLM-based or encoder-decoder pretraining.

3. **Unified interface**: A decoder-only model can do classification, generation, translation, summarization, and QA via prompting, without architectural changes. Encoder-only models cannot generate; encoder-decoder models need separate encoder invocations.

4. **KV cache efficiency for generation**: The KV cache for decoder-only generation is straightforward — cached once, extended token by token. Encoder-decoder models need to run the encoder fresh for each new input and maintain both encoder and decoder KV caches.

5. **Transfer to fine-tuning**: RLHF and instruction tuning (which drove the ChatGPT-era models) are naturally formulated as sequence-to-sequence tasks via prompting, fitting the decoder-only paradigm perfectly.

6. **Emergent capabilities from scale**: GPT-3 demonstrated that massive scale in a decoder-only model produces qualitatively new behaviors. This motivated scaling investment in the same architecture.

---

## 12. Loading and Generating with HuggingFace

### Basic Text Generation

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # bfloat16 for modern GPUs
    device_map="auto",           # automatically place across GPUs/CPU
)
```

### Greedy Generation

```python
inputs = tokenizer("The Eiffel Tower is located in", return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,   # greedy decoding
    )
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Sampling with Temperature

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,     # lower = more deterministic
    top_p=0.9,           # nucleus sampling
    top_k=50,            # restrict to top-50 tokens at each step
    repetition_penalty=1.1,
)
```

### Chat Template (Instruction-Tuned Models)

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain gradient descent in two sentences."},
]

# Apply the model's chat template
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.6,
    )

# Decode only the generated portion
generated = outputs[0][inputs["input_ids"].shape[1]:]
print(tokenizer.decode(generated, skip_special_tokens=True))
```

### Streaming Generation

```python
from transformers import TextStreamer

streamer = TextStreamer(tokenizer, skip_special_tokens=True)
model.generate(**inputs, max_new_tokens=256, streamer=streamer)
```

### Memory-Efficient Loading (Quantization)

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
```

---

## References

- Radford, A., et al. (2018). *Improving Language Understanding by Generative Pre-Training*. OpenAI. [GPT-1](https://openai.com/research/language-understanding)
- Radford, A., et al. (2019). *Language Models are Unsupervised Multitask Learners*. OpenAI. [GPT-2](https://openai.com/research/better-language-models)
- Brown, T., et al. (2020). *Language Models are Few-Shot Learners*. NeurIPS 2020. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
- OpenAI. (2023). *GPT-4 Technical Report*. [arXiv:2303.08774](https://arxiv.org/abs/2303.08774)
- Touvron, H., et al. (2023). *LLaMA: Open and Efficient Foundation Language Models*. [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)
- Touvron, H., et al. (2023b). *Llama 2: Open Foundation and Fine-Tuned Chat Models*. [arXiv:2307.09288](https://arxiv.org/abs/2307.09288)
- Meta. (2024). *Meta Llama 3*. [Meta AI Blog](https://ai.meta.com/blog/meta-llama-3/)
- Jiang, A.Q., et al. (2023). *Mistral 7B*. [arXiv:2310.06825](https://arxiv.org/abs/2310.06825)
- Penedo, G., et al. (2023). *The Falcon Series of Open Language Models*. [arXiv:2311.16867](https://arxiv.org/abs/2311.16867)
- Zhang, B., Sennrich, R. (2019). *Root Mean Square Layer Normalization*. NeurIPS 2019. [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
- Shazeer, N. (2020). *GLU Variants Improve Transformer*. [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)
- Ainslie, J., et al. (2023). *GQA: Training Generalized Multi-Query Transformer Models*. EMNLP 2023. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)

---

*[← Encoder-Only Models](./encoder-only.md) | [← Back to Architectures](./README.md) | [→ Encoder-Decoder Models](./encoder-decoder.md)*

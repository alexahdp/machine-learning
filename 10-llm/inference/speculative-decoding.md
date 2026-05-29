# Speculative Decoding

## Table of Contents

1. [Why LLM Inference Is Memory-Bandwidth Bound](#1-why-llm-inference-is-memory-bandwidth-bound)
2. [The Core Insight](#2-the-core-insight)
3. [Speculative Decoding Algorithm](#3-speculative-decoding-algorithm)
4. [Acceptance-Rejection Sampling](#4-acceptance-rejection-sampling)
5. [Expected Speedup Analysis](#5-expected-speedup-analysis)
6. [Acceptance Rate Determinants](#6-acceptance-rate-determinants)
7. [Self-Speculative Decoding](#7-self-speculative-decoding)
8. [Medusa](#8-medusa)
9. [EAGLE](#9-eagle)
10. [Tree-Structured Drafting](#10-tree-structured-drafting)
11. [When Speculative Decoding Helps Most](#11-when-speculative-decoding-helps-most)
12. [Hardware Considerations](#12-hardware-considerations)
13. [Practical: HuggingFace Speculative Decoding](#13-practical-huggingface-speculative-decoding)
14. [References](#references)

---

## 1. Why LLM Inference Is Memory-Bandwidth Bound

At small batch sizes (typical for latency-sensitive serving), LLM inference is **memory-bandwidth bound**, not compute-bound.

**Arithmetic intensity**: For a matrix-vector product $y = Wx$ (the dominant operation in transformer inference):
- **FLOPs**: $2 \times d_{\text{out}} \times d_{\text{in}}$ (one multiply-add per weight)
- **Memory bytes read**: $d_{\text{out}} \times d_{\text{in}} \times 2$ (FP16 weights) + $d_{\text{in}} \times 2$ (input)

Arithmetic intensity = FLOPs / bytes read:
$$I = \frac{2 \times d_{\text{out}} \times d_{\text{in}}}{2 \times d_{\text{out}} \times d_{\text{in}} + 2 \times d_{\text{in}}} \approx 1 \text{ FLOP/byte}$$

An A100 SXM has:
- Peak FP16 compute: 312 TFLOPS
- Memory bandwidth: 2 TB/s
- **Roofline compute ceiling at 1 FLOP/byte**: 2 TFLOPS (not 312 TFLOPS)

**Implication**: At batch size 1, the GPU runs at ~0.6% of peak FLOP capacity. The bottleneck is reading 140 GB of weights from HBM memory, not executing multiplications.

**Batch size effect**: At batch size $B$, the matrix multiplication becomes $Y = WX$ where $X \in \mathbb{R}^{d_{\text{in}} \times B}$:
$$I \approx \frac{2B \times d_{\text{out}} \times d_{\text{in}}}{2 \times d_{\text{out}} \times d_{\text{in}}} = B$$

At $B = 100$: arithmetic intensity ≈ 100 FLOP/byte → the GPU approaches compute-bound regime.

**Key consequence**: Generating one token at batch size 1 takes approximately the same time as generating one batch of tokens at batch size $N$ (up to the hardware's compute ceiling). This "wasted parallelism" is the opportunity that speculative decoding exploits.

---

## 2. The Core Insight

**The bottleneck**: At small batch sizes, each autoregressive step reads all weights from memory. Generating $k$ tokens sequentially = $k$ full memory reads = $k \times$ latency of one step.

**The observation**: The target model (large) can verify $k$ tokens in a **single forward pass**, not $k$ passes, because:
- Attention is computed over all $k$ positions simultaneously (just like training)
- The verify pass produces the probability distribution at each of the $k$ positions in parallel

**The hypothesis**: If a smaller, faster "draft" model proposes $k$ candidate tokens, and most of them match what the large model would have generated, we can accept them all from one target model forward pass — achieving the equivalent of $k$ tokens from the cost of ~1 forward pass of the large model.

```
Without speculative decoding:
 Target │ ←← read 140GB ←←│ ←← read 140GB ←←│ ←← read 140GB ←←│
        │   generate x₁   │   generate x₂   │   generate x₃   │
         ─────────────────────────────────────────────────────────
         T                 T                 T                  = 3T

With speculative decoding (draft accepts all 3):
 Draft  │ generate x₁x₂x₃ │                                    │
 Target │                  │ ←← read 140GB once ←←│            │
        │                  │  verify x₁,x₂,x₃ in  │            │
        │                  │  parallel, accept all │            │
         ─────────────────────────────────────────────────────────
         t                                    T                 = t + T ≈ T
```

If $t \ll T$ (draft is much faster than target), the speedup approaches $k$ per target model call.

---

## 3. Speculative Decoding Algorithm

Proposed independently by Chen et al. (2023) ("Accelerating Large Language Model Decoding with Speculative Sampling") and Leviathan et al. (2023) ("Fast Inference from Transformers via Speculative Decoding").

**Inputs**:
- Draft model $M_q$ (small, fast): produces distributions $q(x \mid x_{<t})$
- Target model $M_p$ (large, high quality): produces distributions $p(x \mid x_{<t})$
- Draft length $k$ (tokens to propose per round)
- Current prefix $x_{1:t}$

**One round of speculative decoding**:

```
1. DRAFT PHASE
   Run M_q autoregressively for k steps:
   x̃_{t+1} ~ q(· | x_{1:t})
   x̃_{t+2} ~ q(· | x_{1:t}, x̃_{t+1})
   ...
   x̃_{t+k} ~ q(· | x_{1:t}, x̃_{t+1:t+k-1})
   
   (k fast forward passes of M_q)

2. VERIFY PHASE
   Run M_p once on the full sequence x_{1:t}, x̃_{t+1}, ..., x̃_{t+k}:
   → Gets distributions p(· | x_{1:t+i}) for i = 0, ..., k in parallel
   
   (1 forward pass of M_p, but processes k+1 positions)

3. ACCEPTANCE PHASE
   For i = 1, ..., k:
     r ~ Uniform(0, 1)
     if r < p(x̃_{t+i} | x_{1:t+i-1}) / q(x̃_{t+i} | x_{1:t+i-1}):
       Accept x̃_{t+i}  (continue to next draft token)
     else:
       Reject x̃_{t+i}
       Sample correction token from adjusted distribution
       Stop (return accepted prefix + correction token)
   
   If all k accepted:
     Sample one bonus token from p(· | x_{1:t+k})

4. RETURN
   Accepted tokens (at least 1, at most k+1)
   Update prefix and repeat
```

**The algorithm guarantees**: The output distribution is identical to sampling from $M_p$ directly. This is a critical property — speculative decoding is **lossless** (same quality, faster).

---

## 4. Acceptance-Rejection Sampling

The acceptance criterion comes from rejection sampling theory. We want samples from target distribution $p$ but only have samples from proposal distribution $q$.

**Standard rejection sampling**: Accept a sample $x \sim q$ with probability $\min(1, p(x)/q(x))$. This yields samples from $p$.

**Problem**: If $q(x) < p(x)$ (target assigns more mass to $x$ than draft), acceptance probability $> 1$ — invalid.

**Solution** (speculative sampling): When a draft token is rejected, don't just discard it — sample a correction token from the "residual" distribution:

$$p'(x) = \text{norm}\!\left(\max\!\left(0,\ p(x) - q(x)\right)\right)$$

This correction ensures the marginal distribution over accepted tokens is exactly $p$. The proof: let $\alpha = \min\!\left(1, p(\tilde{x}) / q(\tilde{x})\right)$ be the acceptance probability for draft token $\tilde{x}$. The probability of outputting token $x$ is:

$$\alpha \cdot \mathbf{1}[x = \tilde{x}] + (1 - \alpha) \cdot p'(x) = p(x)$$

which integrates to $p(x)$ over all $x$. (Full proof in Leviathan et al., 2023, Appendix A.)

---

## 5. Expected Speedup Analysis

Let $\alpha$ be the **acceptance rate** — the probability that any single draft token is accepted by the target model.

**Expected number of accepted tokens per round** (for $k$ draft tokens):

$$\mathbb{E}[\text{tokens accepted}] = \sum_{i=0}^{k-1} \alpha^i = \frac{1 - \alpha^k}{1 - \alpha}$$

Plus 1 bonus token (always generated from $M_p$ at the end), so:

$$\mathbb{E}[\text{total tokens per round}] = \frac{1 - \alpha^k}{1 - \alpha} + 1$$

**Speedup formula**: Let $c$ be the cost of one draft model forward pass relative to one target model forward pass ($c \ll 1$). Then:

$$\text{Speedup} = \frac{\mathbb{E}[\text{tokens per round}]}{k \cdot c + 1} = \frac{(1 - \alpha^k)/(1 - \alpha) + 1}{kc + 1}$$

**Example calculation**: $\alpha = 0.8$, $k = 5$, $c = 0.1$ (draft is 10× faster):

$$\mathbb{E}[\text{tokens}] = \frac{1 - 0.8^5}{0.2} + 1 = \frac{1 - 0.328}{0.2} + 1 = 3.36 + 1 = 4.36$$

$$\text{Speedup} = \frac{4.36}{5 \times 0.1 + 1} = \frac{4.36}{1.5} = 2.9\times$$

**Key insights from the formula**:
- Speedup is maximized when $\alpha$ is high AND $c$ is low
- At $\alpha = 1.0$ (draft always accepted): speedup = $(k+1) / (kc + 1) \approx 1/c$ for large $k$
- At $\alpha = 0.0$ (draft always rejected): speedup = $2/(0 + 1) = 2$ — wait, actually the bonus token is always accepted, so even at $\alpha = 0$ there is 1 accepted token per round, giving speedup $= 1/(kc + 1) < 1$ — **slower than baseline**. This means a poorly aligned draft model actually hurts performance.
- Optimal $k$: increasing $k$ past the acceptance-rate-determined ceiling gives diminishing returns and increases draft model overhead

**Reported speedups** (Chen et al., PaLM 540B + 8B draft): 2.9× speedup at $\alpha \approx 0.8$.

---

## 6. Acceptance Rate Determinants

Acceptance rate $\alpha$ depends on how well draft and target agree. It is NOT a fixed property of the model pair — it varies by task and context:

| Task | Typical $\alpha$ | Why |
|------|-----------------|-----|
| Code completion | 0.85–0.92 | Syntax is highly constrained; both models agree |
| Translation | 0.80–0.88 | Target language is constrained by source |
| Summarization | 0.75–0.85 | Content is constrained by source document |
| Open-ended chat | 0.60–0.75 | Many valid continuations; draft and target diverge more |
| Creative writing | 0.50–0.70 | High entropy; draft predictions frequently off |

**Draft-target model alignment**: The draft model should be a smaller version of the same architecture family. A 7B + 70B pair from the same training family has much higher acceptance rate than a mismatched pair (e.g., a small distilled model with a different architecture).

**Draft length $k$**: Longer drafts risk more rejections. Once $\alpha^k$ becomes small, most rounds produce only partial acceptance. Typical $k = 4$–8.

---

## 7. Self-Speculative Decoding

Instead of a separate draft model, use **early exit layers** of the target model as the draft.

**Mechanism**: Run the target model's first $L_d < L$ layers to get a draft distribution, then run all $L$ layers to verify. The partial forward pass is the draft; the full forward pass is the verification.

**Variants**:
- **LLMA** (early exit): stop at layer $L_d$, use that representation to predict next token
- **LayerSkip** (Meta, 2024): train the model with "early exit" training objectives so intermediate layers produce good token predictions. At inference, use early exit for draft, full model for verify.

**Advantages**: Only one model to load (memory savings); draft and target share compute (activations for first $L_d$ layers computed once and reused).

**Disadvantages**: Early exit drafts have lower acceptance rates than a well-tuned small model; limited speedup from shared-architecture early exit.

**Observed speedup**: ~1.3–1.8× for LayerSkip on LLaMA-2 7B — lower than cross-model speculative decoding but requires no second model.

---

## 8. Medusa

**Medusa** (Cai et al., 2024) replaces the draft model with multiple additional decoding heads attached to the last hidden state of the target model.

```
Target LLM backbone
        │
        ▼
  Hidden state h_t
        │
   ┌────┴──────────────────────┐
   │                           │
   ▼                           ▼
LM head              Medusa heads (1, 2, ..., k)
(predict x_t)        head_i predicts x_{t+i}
```

**Training**: The Medusa heads are trained while the backbone is frozen (or lightly fine-tuned). They learn to predict tokens $i$ steps ahead based on the current hidden state.

**Inference**: At each step:
1. Run target model forward pass → get hidden state $h_t$
2. Each Medusa head $i$ predicts a distribution over next+$i$ token
3. Select top candidates from each head (creating a tree of possible continuations)
4. Verify candidates using target model attention (one pass, processes the full tree)
5. Accept the longest consistent prefix

**No separate draft model**: Medusa adds only ~5% to model memory (a few linear layers) and no additional model-loading complexity.

**Speedup**: 2.0–3.0× on chat tasks with Medusa-1 (frozen backbone) or Medusa-2 (joint fine-tuning).

**Limitation**: Medusa heads are trained on a specific model — they are not transferable. The backbone must be fine-tuned or the heads must be trained from scratch for each model.

---

## 9. EAGLE

**EAGLE** (Li et al., 2024) — Extrapolation Algorithm for Greater Language-model Efficiency — uses a lightweight auto-regressive draft model that operates in the **feature space** rather than token space.

**Key insight**: Instead of training a small language model to predict tokens, EAGLE trains a small model to predict the **hidden features** (last-layer activations) of the target model. Feature prediction is an easier task than direct token prediction because features are in a continuous, smooth space.

```
EAGLE draft model:
  Input: [current features f_t, current token embedding e_t]
  Output: predicted features f̂_{t+1}
  
  f̂_{t+1} → LM head of target model → draft token distribution
```

**Architecture**: EAGLE draft is a single transformer layer (not a full model) + a copy of the target model's LM head. It autoregressively predicts feature vectors, then uses the target model's LM head to convert features to token distributions.

**EAGLE-2**: Introduces context-aware tree drafting — the draft tree shape is adapted dynamically based on the acceptance probability at each node, allocating more candidates where acceptance is likely.

**Speedup**: EAGLE achieves 3.0–4.0× speedup (higher than Medusa), with acceptance rates of 0.85–0.92 on code and 0.75–0.85 on chat. EAGLE-2 achieves up to 5× speedup on code tasks.

**Training cost**: EAGLE draft model trains in ~1–3 days on one A100 for a 7B model, using ~100K samples from the target model's outputs (no original training data required).

---

## 10. Tree-Structured Drafting

Single-sequence drafting proposes one linear sequence of $k$ tokens. **Tree drafting** proposes multiple sequences organized as a tree, increasing the total number of candidates verified in one target model pass.

```
Single draft (k=4):      Tree draft (branching factor 2, depth 3):
                          
x₁ → x₂ → x₃ → x₄       x₁ₐ → x₂ₐ → x₃ₐ
                          x₁ₐ → x₂ₐ → x₃ᵦ
                          x₁ₐ → x₂ᵦ → x₃ᵧ
                          x₁ᵦ → x₂꜀ → x₃ᵟ
                          ...
                          (7 candidate tokens in tree vs 4 in sequence)
```

**Verification**: The target model runs attention over the tree structure using a custom **tree attention mask** — each node only attends to its ancestors (its prefix path through the tree). This is implementable as a modified causal attention mask.

**Acceptance**: Walk the tree top-down, accepting the branch that matches the target model's greedy or sampling selection at each depth. The longest accepted path gives the final tokens.

**SpecInfer** (Miao et al., 2023): Proposes running multiple draft models simultaneously, each producing one sequence, assembled into a tree for batch verification.

**SpecTr**: Optimal tree construction via dynamic programming — given a budget of total tree nodes, allocate branching factors to maximize expected accepted tokens under a given acceptance rate model.

**Speedup vs single sequence**: Tree drafting can achieve 20–50% additional speedup over single-sequence drafting of the same depth, because the tree covers more of the probability space.

---

## 11. When Speculative Decoding Helps Most

**High acceptance rate scenarios** (best case):
- Code completion: syntax is highly predictable; draft model tracks target well
- Constrained generation: when output is constrained (JSON, function signatures), token space is small → high agreement
- Translation of technical content: limited vocabulary diversity
- Summarization with long source documents: target output is closely tied to source

**Low acceptance rate scenarios** (worst case):
- Open-ended creative generation: high entropy, many valid continuations
- Tasks requiring very different styles from the draft model's distribution
- When draft and target come from different model families

**Batch size effect**: Speculative decoding helps most at **small batch sizes** (1–16). At large batch sizes, the target model is already compute-bound — the parallel verification doesn't help because the GPU is already fully utilized. At batch size ~100, standard batching dominates and speculative decoding offers little benefit.

**Response length**: Long responses benefit more than short ones. The setup overhead (draft generation + verification) is amortized over more tokens.

**Summary of conditions for high speedup**:
- Batch size < 16
- Response length > 50 tokens
- Task is predictable (code, constrained output)
- Draft model is well-aligned with target (same model family, ~10× smaller)
- Hardware: fast interconnect between draft and target model (or co-located on same GPUs)

---

## 12. Hardware Considerations

**Memory for both models**: Speculative decoding requires loading both draft and target model. For LLaMA-2 70B + LLaMA-2 7B:
- 70B FP16: 140 GB
- 7B FP16: 14 GB
- Total: 154 GB (requires 2× A100 80GB minimum)

**Quantization to fit**: Quantizing the draft model (not the target) is common:
- 7B INT4: ~4 GB — fits alongside 70B on existing GPU allocation
- Target model quality is unaffected; draft quality degradation reduces acceptance rate slightly

**KV cache for both models**: Both draft and target maintain separate KV caches. For speculative decoding with $k = 5$, the draft KV cache holds the current sequence; the target KV cache holds the accepted prefix.

**GPU co-location vs separate GPUs**: If draft and target run on separate GPUs:
- Draft runs in parallel with target verification — latency is $\max(t_{\text{draft}}, 0) + t_{\text{verify}}$
- Requires fast NVLink for KV state transfer

If both run on the same GPUs (sequential):
- Simpler but draft and verify cannot overlap
- Still beneficial when draft is much smaller and faster

**Practical recommendation**: Use speculative decoding when:
- Serving with batch sizes ≤ 16 (latency-sensitive applications)
- GPU memory allows loading both models
- The draft model is from the same family (e.g., LLaMA-2 7B as draft for LLaMA-2 70B)

---

## 13. Practical: HuggingFace Speculative Decoding

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load target model (large)
target_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-chat-hf",
    torch_dtype=torch.float16,
    device_map="auto",
)

# Load draft model (small, same family)
draft_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    torch_dtype=torch.float16,
    device_map="cuda:0",
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf")

prompt = "Write a Python function to compute the nth Fibonacci number:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Speculative decoding via assistant model
output = target_model.generate(
    **inputs,
    assistant_model=draft_model,      # HuggingFace speculative decoding API
    do_sample=False,                  # greedy (higher acceptance rate)
    max_new_tokens=200,
    # Optional: num_assistant_tokens sets k (draft length per round)
    # num_assistant_tokens=5,         # default is adaptive
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

**Adaptive draft length** (HuggingFace default): HuggingFace's implementation dynamically adjusts the number of draft tokens based on the acceptance rate. If acceptance is high, it increases $k$; if acceptance drops, it decreases $k$. This avoids the fixed-$k$ inefficiency at low acceptance rates.

**Measuring acceptance rate**:

```python
from transformers import AutoModelForCausalLM, GenerationConfig
import time

# Time standard generation
t0 = time.time()
out_standard = target_model.generate(**inputs, max_new_tokens=200)
t_standard = time.time() - t0

# Time speculative generation
t0 = time.time()
out_speculative = target_model.generate(**inputs, assistant_model=draft_model, max_new_tokens=200)
t_speculative = time.time() - t0

speedup = t_standard / t_speculative
print(f"Speedup: {speedup:.2f}x")
```

---

## References

- Chen, C. et al. (2023). "Accelerating Large Language Model Decoding with Speculative Sampling." *arXiv 2302.01318*. — Original speculative sampling with token-level rejection.
- Leviathan, Y. et al. (2023). "Fast Inference from Transformers via Speculative Decoding." *ICML 2023*. — Parallel derivation; acceptance-rejection proof.
- Cai, T. et al. (2024). "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads." *ICML 2024*.
- Li, Y. et al. (2024). "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty." *ICML 2024*. — Feature-space drafting.
- Li, Y. et al. (2024). "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees." *EMNLP 2024*.
- Miao, X. et al. (2023). "SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification." *ASPLOS 2024*.
- Elhoushi, M. et al. (2024). "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding." *ACL 2024*.

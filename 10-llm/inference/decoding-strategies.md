# Decoding Strategies

## Table of Contents

1. [Auto-regressive Generation](#1-auto-regressive-generation)
2. [Greedy Decoding](#2-greedy-decoding)
3. [Beam Search](#3-beam-search)
4. [Temperature Sampling](#4-temperature-sampling)
5. [Top-k Sampling](#5-top-k-sampling)
6. [Top-p / Nucleus Sampling](#6-top-p--nucleus-sampling)
7. [Min-p Sampling](#7-min-p-sampling)
8. [Repetition Penalty](#8-repetition-penalty)
9. [Frequency Penalty vs Presence Penalty](#9-frequency-penalty-vs-presence-penalty)
10. [Combining Strategies](#10-combining-strategies)
11. [Contrastive Search](#11-contrastive-search)
12. [Constrained Decoding](#12-constrained-decoding)
13. [Practical: HuggingFace Generation Config](#13-practical-huggingface-generation-config)
14. [References](#references)

---

## 1. Auto-regressive Generation

A language model defines a distribution over sequences by factoring it left-to-right:

$$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \ldots, x_{t-1})$$

At each step $t$, the model takes the full prefix $x_{<t}$ and outputs a logit vector $z \in \mathbb{R}^{|V|}$ (one entry per vocabulary token). Applying softmax converts logits to a probability distribution:

$$P(x_t = v \mid x_{<t}) = \frac{\exp(z_v)}{\sum_{v' \in V} \exp(z_{v'})}$$

**Decoding** is the process of selecting one token from this distribution at each step. The selection rule is not specified by the model — it is a hyperparameter at inference time. Different rules trade off quality, diversity, and speed differently.

**Key consequence**: the model runs one forward pass per token generated. For a sequence of length $T$, that is $T$ sequential forward passes, each of which depends on the previous step's output. This sequentiality is the fundamental throughput bottleneck.

---

## 2. Greedy Decoding

At each step, select the most probable token:

$$x_t = \arg\max_{v \in V} P(x_t = v \mid x_{<t})$$

**Properties**:
- Deterministic: same input always produces same output
- Fast: no sampling or sorting beyond a single argmax
- **No guarantee of globally optimal sequence**: greedy is locally optimal but globally suboptimal. The highest-probability next token may lead to a low-probability continuation.

**Failure mode — repetition**: Greedy decoding frequently degenerates into repetitive loops. Once the model assigns high probability to a token given a certain prefix, that token being selected makes the same prefix more likely to recur. Observed in practice as outputs like "the the the the..." or restatements of the same sentence.

**When to use**: Exact-match tasks where there is one correct answer (arithmetic, code with a specific function signature, structured extraction). Not suitable for open-ended generation.

---

## 3. Beam Search

Beam search maintains $k$ candidate sequences (beams) simultaneously, expanding all and keeping the top $k$ at each step.

**Algorithm**:
```
Initialize: beams = [([], 0.0)]  # (token list, log-prob)

For each step t:
    candidates = []
    For each beam (seq, score):
        For each token v in vocabulary:
            new_score = score + log P(v | seq)
            candidates.append((seq + [v], new_score))
    beams = top-k candidates by score
```

**Scoring**: beams are scored by the sum of log-probabilities:

$$\text{score}(x_1, \ldots, x_T) = \sum_{t=1}^{T} \log P(x_t \mid x_{<t})$$

With beam width $k = 1$, beam search reduces to greedy decoding.

### Length Normalization

Longer sequences accumulate more negative log-probabilities, so beam search systematically prefers shorter outputs. Length normalization corrects this:

$$\text{score}_\text{norm}(x_{1:T}) = \frac{1}{T^\alpha} \sum_{t=1}^{T} \log P(x_t \mid x_{<t})$$

$\alpha \in [0, 1]$ controls normalization strength. $\alpha = 0$ gives raw log-prob (length-biased); $\alpha = 1$ gives mean log-prob (may over-correct). $\alpha = 0.6$ is a common default (used in Google's NMT).

### When Beam Search Hurts

Beam search improves performance on **constrained generation tasks**: machine translation, summarization, code generation with a target API, structured outputs. It finds higher log-probability sequences than greedy.

For **open-ended generation**, beam search produces generic, repetitive, "safe" outputs. This is because the highest-probability sequence is often the most generic one — beam search optimizes exactly the wrong objective for creative tasks. Empirically, humans prefer samples from language models over beam search outputs for dialogue and story generation (Holtzman et al., 2020).

**Compute cost**: $O(k \cdot |V|)$ candidates per step vs $O(|V|)$ for greedy, but only the top-k need their full forward pass, so in practice beam search is $k\times$ more compute with some implementation optimization.

---

## 4. Temperature Sampling

Before applying softmax, divide logits by temperature $T > 0$:

$$P_T(x_t = v \mid x_{<t}) = \frac{\exp(z_v / T)}{\sum_{v'} \exp(z_{v'} / T)}$$

**Effect on distribution**:
- $T \to 0$: distribution concentrates on argmax → equivalent to greedy decoding
- $T = 1$: model's native distribution (no modification)
- $T > 1$: distribution flattens → more uniform → higher entropy → more random
- $T \to \infty$: uniform distribution over vocabulary

**Intuition**: Temperature controls the "peakedness" of the distribution. At low temperature, high-probability tokens get even higher probability; at high temperature, probability mass spreads to lower-ranked tokens.

**Practical values**:
- $T = 0.0$ to $0.3$: deterministic or near-deterministic; good for factual Q&A, code
- $T = 0.7$: common default for chat models; balance of coherence and variety
- $T = 1.0$: unmodified model distribution
- $T > 1.2$: noticeable degradation in coherence; useful for creative exploration or data augmentation

**Warning**: Temperature alone is insufficient for high-quality open-ended generation because it still allows sampling from the full vocabulary, including extremely low-probability tokens that are almost certainly nonsensical. This is why temperature is usually combined with truncation methods (top-k or top-p).

---

## 5. Top-k Sampling

Restrict sampling to the $k$ most probable tokens, then renormalize:

$$V_k = \text{top-}k\text{ tokens by } P(x_t \mid x_{<t})$$

$$P'(x_t = v \mid x_{<t}) = \begin{cases} P(x_t = v) / Z & \text{if } v \in V_k \\ 0 & \text{otherwise} \end{cases}$$

where $Z = \sum_{v \in V_k} P(x_t = v)$ is the normalization constant.

**Common value**: $k = 50$ is a typical default.

**Problem with fixed $k$**: The optimal $k$ varies step-by-step. When the model is confident (distribution is peaked), the 5 top tokens might cover 95% of the probability — including token 50 would add nearly-impossible noise. When the model is uncertain (distribution is flat), the top 50 tokens might cover only 20% of the probability — excluding the remaining 80% cuts off legitimate candidates. Fixed $k$ handles neither case well.

---

## 6. Top-p / Nucleus Sampling

Instead of a fixed count, include the smallest set of tokens $V_p$ such that their cumulative probability reaches at least $p$:

$$V_p = \arg\min_{V' \subseteq V} |V'| \quad \text{s.t.} \quad \sum_{v \in V'} P(x_t = v \mid x_{<t}) \geq p$$

(where $V'$ is the set of tokens with highest probability, sorted in descending order)

Renormalize and sample from $V_p$.

**Why this is better than top-k**: The nucleus adapts dynamically to the model's confidence at each step. When the model is confident, $|V_p|$ is small (few tokens needed to reach $p$). When the model is uncertain, $|V_p|$ is large (many tokens included). This avoids both the "too many noise tokens" problem of large fixed $k$ and the "too few candidates" problem of small fixed $k$.

**Common values**: $p = 0.9$ or $p = 0.95$. At $p = 1.0$, all tokens are included (no truncation).

**Introduced by**: Holtzman et al., "The Curious Case of Neural Text Degeneration" (2020). This paper also introduced the concept of sequence probability distribution degeneration as the cause of repetitive generation.

---

## 7. Min-p Sampling

An alternative truncation scheme: remove tokens whose probability is below a fraction of the maximum probability:

$$V_{\min\text{-}p} = \{v : P(x_t = v \mid x_{<t}) \geq p_{\min} \cdot \max_{v'} P(x_t = v' \mid x_{<t})\}$$

**Example**: if $p_{\min} = 0.05$ and the top token has probability 0.6, all tokens with probability $< 0.03$ are excluded.

**Comparison to top-p**: Min-p scales the threshold relative to the top token's probability. When the model is very confident and the top token has probability 0.9, min-p cuts off everything below 0.045 — a tight nucleus. When the model is uncertain and the top token has probability 0.1, min-p cuts off only tokens below 0.005 — a wide nucleus. This makes min-p naturally adaptive in the same spirit as top-p, with a different parameterization that some practitioners find more intuitive.

**Typical values**: $p_{\min} = 0.05$ to $0.1$.

---

## 8. Repetition Penalty

Repetitive generation is a widespread failure mode. The repetition penalty (Keskar et al., 2019) discounts already-generated tokens:

$$z'_v = \begin{cases} z_v / \theta & \text{if } z_v > 0 \text{ and } v \in x_{<t} \\ z_v \cdot \theta & \text{if } z_v < 0 \text{ and } v \in x_{<t} \\ z_v & \text{if } v \notin x_{<t} \end{cases}$$

where $\theta > 1$ is the penalty parameter. The effect is to divide positive logits by $\theta$ (reducing probability) and multiply negative logits by $\theta$ (further reducing probability), for any token already in the generated sequence.

**Typical values**: $\theta = 1.1$ to $1.3$. Values above $1.5$ tend to cause the model to avoid repeating even necessary tokens (e.g., "the", "is") and produce incoherent outputs.

**Limitation**: The penalty applies to the entire generated history, so it discourages repeating tokens even when repetition is correct (e.g., repeating a name multiple times in a story).

---

## 9. Frequency Penalty vs Presence Penalty

OpenAI's API provides two distinct repetition controls:

**Frequency penalty** $\lambda_f$: penalizes tokens proportionally to how often they have appeared:

$$z'_v = z_v - \lambda_f \cdot \text{count}(v, x_{<t})$$

More appearances → larger penalty. Encourages reducing the *rate* of repetition.

**Presence penalty** $\lambda_p$: applies a flat penalty if the token has appeared at all:

$$z'_v = z_v - \lambda_p \cdot \mathbf{1}[v \in x_{<t}]$$

Binary: appeared once or many times — same penalty. Encourages introducing *new* tokens.

**When to use which**:
- Frequency penalty: reduces word overuse while allowing the token to still appear naturally. Better for longer documents where some repetition is expected.
- Presence penalty: strongly pushes toward introducing new vocabulary. More aggressive, can hurt coherence at high values.
- Both can be combined. OpenAI range is $[-2, 2]$ for both; typical values $0.1$–$0.5$.

---

## 10. Combining Strategies

Production systems typically combine multiple controls:

```
greedy        ← temperature=0, no sampling
standard chat ← temperature=0.7, top_p=0.9, repetition_penalty=1.1
creative      ← temperature=1.0, top_p=0.95, top_k=50, repetition_penalty=1.15
code gen      ← temperature=0.2, top_p=0.95 (or greedy)
factual QA    ← temperature=0.0 to 0.3
```

**Order of operations in HuggingFace**:
1. Apply logit processors (repetition penalty, grammar constraints, etc.)
2. Apply temperature scaling to logits
3. Apply top-k filtering (zero out below-threshold logits)
4. Apply top-p filtering (zero out below-nucleus logits)
5. Apply softmax to get probabilities
6. Sample (or argmax for greedy)

**Interaction effects**: Temperature and top-p interact. With $T = 0.5$ (sharpened distribution), top-p = 0.9 will include fewer tokens than with $T = 1.0$ because the sharpened distribution reaches $p = 0.9$ sooner. Setting both is redundant in some regimes — if $T$ already makes the distribution peaked, top-p adds little additional truncation.

---

## 11. Contrastive Search

Contrastive search (Su et al., 2022) simultaneously maximizes model confidence and minimizes similarity between the new token and the context:

$$x_t = \arg\max_{v \in V_k} \left[ (1 - \alpha) \cdot P(v \mid x_{<t}) - \alpha \cdot \max_{j \in [1, t-1]} \cos(h_v, h_{x_j}) \right]$$

where:
- $V_k$ is the top-$k$ candidates by model probability
- $h_v$ is the hidden representation of token $v$ at the current step
- $h_{x_j}$ is the hidden representation of the $j$-th generated token
- $\alpha \in [0, 1]$ controls the anti-repetition strength (typical: $\alpha = 0.6$)

**Intuition**: Select a token that is both probable under the model and unlike what has already been generated (measured in the model's representation space, not just surface form). This directly targets semantic repetition, not just lexical repetition.

**Advantages over sampling**:
- Deterministic (no randomness)
- Avoids both repetition and incoherence
- Empirically produces more human-like text than greedy or beam search on open-ended tasks

**Disadvantages**: Requires computing cosine similarities against all previous hidden states at each step — $O(T)$ extra compute per token. Also requires access to hidden states, not just output logits (not available through all serving APIs).

---

## 12. Constrained Decoding

Standard decoding produces free-form text. Many applications require structured output (JSON, SQL, code matching an API, outputs constrained to a known set).

### Grammar-Constrained Decoding

Maintain a parser state alongside the generation state. At each step, only allow tokens that produce a valid next state in the grammar (context-free grammar or regular expression). Any token that would produce an invalid parse state gets its logit set to $-\infty$ before sampling.

**Libraries**: Guidance, Outlines, LMQL, llama.cpp grammars.

**Overhead**: Parsing is fast relative to the forward pass; typically < 5% latency overhead.

### JSON-Constrained Decoding

A special case of grammar constraints. Given a JSON schema, construct a state machine that tracks valid JSON parse states. At each step, the allowed tokens are exactly those that continue a valid JSON document consistent with the schema.

**Example**: If the schema requires `{"name": <string>, "age": <integer>}` and the model has generated `{"name": "Alice", "age": `, then only digit tokens (and possibly `-`) are allowed next.

### Forced Tokens / Prefix Forcing

Forcing a specific prefix at the start of generation (e.g., forcing `{"` to start a JSON response). Some APIs support `logit_bias` to strongly boost or suppress specific tokens. Setting a token's logit bias to `+100` effectively forces it; `-100` effectively bans it.

---

## 13. Practical: HuggingFace Generation Config

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Greedy decoding
config_greedy = GenerationConfig(
    do_sample=False,
    max_new_tokens=256,
)

# Nucleus sampling (typical chat)
config_nucleus = GenerationConfig(
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    top_k=50,                    # also apply top-k as a safety net
    repetition_penalty=1.1,
    max_new_tokens=512,
)

# Beam search (translation/summarization)
config_beam = GenerationConfig(
    do_sample=False,
    num_beams=4,
    length_penalty=0.6,          # alpha for length normalization
    early_stopping=True,
    max_new_tokens=256,
)

# Constrained JSON output via logit processor
from transformers import LogitsProcessorList
from outlines.processors import JSONLogitsProcessor
import json

schema = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
processor = JSONLogitsProcessor(schema, tokenizer)

inputs = tokenizer("Extract: Alice is 30 years old.", return_tensors="pt")
output = model.generate(
    **inputs,
    logits_processor=LogitsProcessorList([processor]),
    max_new_tokens=50,
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
# → {"name": "Alice", "age": 30}
```

**Key `GenerationConfig` parameters**:

| Parameter | Type | Effect |
|-----------|------|--------|
| `do_sample` | bool | `False` = greedy/beam; `True` = sampling |
| `temperature` | float | Logit scaling before softmax |
| `top_k` | int | Keep top-k tokens |
| `top_p` | float | Nucleus probability threshold |
| `repetition_penalty` | float | Penalty multiplier for seen tokens |
| `num_beams` | int | Beam width (1 = greedy) |
| `length_penalty` | float | $\alpha$ for beam search length normalization |
| `max_new_tokens` | int | Hard cap on generation length |
| `min_new_tokens` | int | Force at least this many tokens (prevents early EOS) |

---

## References

- Holtzman, A. et al. (2020). "The Curious Case of Neural Text Degeneration." *ICLR 2020*. — Introduces nucleus sampling and the degeneration analysis.
- Keskar, N. S. et al. (2019). "CTRL: A Conditional Transformer Language Model for Controllable Generation." — Repetition penalty formulation.
- Vijayakumar, A. K. et al. (2018). "Diverse Beam Search." — Beam search diversity extension.
- Su, Y. et al. (2022). "A Contrastive Framework for Neural Text Generation." *NeurIPS 2022*. — Contrastive search.
- Willard, B. T. & Louf, R. (2023). "Efficient Guided Generation for Large Language Models." — Grammar-constrained decoding via finite-state machines.
- OpenAI API Reference. [platform.openai.com/docs/api-reference](https://platform.openai.com/docs/api-reference) — Frequency/presence penalty definitions.

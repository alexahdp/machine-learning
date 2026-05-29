# Language Modeling Objectives

## Table of Contents
1. [Why the Objective Defines the Model](#why-the-objective-defines-the-model)
2. [Causal Language Modeling (CLM)](#causal-language-modeling-clm)
3. [Masked Language Modeling (MLM)](#masked-language-modeling-mlm)
4. [Permutation Language Modeling (PLM)](#permutation-language-modeling-plm)
5. [Prefix Language Modeling](#prefix-language-modeling)
6. [Span Corruption / Denoising (T5)](#span-corruption--denoising-t5)
7. [UL2: Mixture of Denoisers](#ul2-mixture-of-denoisers)
8. [Contrastive Objectives](#contrastive-objectives)
9. [Objective Comparison and Downstream Task Fit](#objective-comparison-and-downstream-task-fit)
10. [Why CLM Has Won](#why-clm-has-won)
11. [References](#references)

---

## Why the Objective Defines the Model

The training objective is not a neutral implementation detail — it shapes the information structure encoded in the model's representations, its generalization behavior, and what downstream tasks it can perform well. An encoder trained with MLM learns rich bidirectional contextual representations but cannot generate text autoregressively. A decoder trained with CLM learns left-to-right conditional distributions and can generate fluently, but each token only attends to its past. These are architectural consequences of the objective, not hyperparameter choices.

The key question each objective answers: *given what the model has seen, what should it predict, and over what positions should the loss be computed?*

---

## Causal Language Modeling (CLM)

### Formulation

CLM factorizes the joint probability of a sequence $\mathbf{x} = (x_1, x_2, \ldots, x_T)$ as a product of conditional probabilities using the chain rule:

$$p(\mathbf{x}) = \prod_{t=1}^{T} p(x_t \mid x_1, x_2, \ldots, x_{t-1})$$

The training loss is the average negative log-likelihood over all token positions:

$$\mathcal{L}_{\text{CLM}} = -\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})$$

The causal (lower-triangular) attention mask enforces that position $t$ cannot attend to positions $> t$, making the factorization exact: each prediction uses only past context.

### Teacher Forcing

During training, the model always receives the *ground-truth* token at position $t-1$ when predicting $x_t$, regardless of what it would have generated. This is called **teacher forcing**. It makes training stable and parallelizable — the entire sequence can be processed in a single forward pass — but introduces a training/inference discrepancy: at inference time, the model sees its own (potentially wrong) previous predictions. This exposure bias is generally tolerable at scale.

### Perplexity

CLM loss maps directly to **perplexity**, the standard language model evaluation metric:

$$\text{PPL} = \exp\left(-\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})\right)$$

Perplexity has a clean interpretation: it is the geometric mean number of tokens the model is "confused between" at each step. Lower is better. Because CLM computes predictions at every position, perplexity is well-defined and directly comparable across models (modulo tokenizer differences).

### Models

GPT-2, GPT-3, GPT-4, LLaMA, Mistral, Falcon, Gemini, Claude (pretraining phase), PaLM.

---

## Masked Language Modeling (MLM)

### Formulation

Rather than predicting every token left-to-right, MLM selects a subset $\mathcal{M}$ of positions, replaces them with a special `[MASK]` token, and trains the model to recover the original tokens:

$$\mathcal{L}_{\text{MLM}} = -\frac{1}{|\mathcal{M}|} \sum_{t \in \mathcal{M}} \log p_\theta(x_t \mid \hat{\mathbf{x}})$$

where $\hat{\mathbf{x}}$ is the corrupted sequence. The model attends bidirectionally to all positions (both left and right of the mask), which cannot be done with CLM's causal mask.

### BERT's 80/10/10 Masking Strategy

Naively replacing all selected tokens with `[MASK]` creates a train/inference mismatch: the `[MASK]` token never appears at inference time, so the model learns representations conditioned on an artifact. BERT (Devlin et al., 2019) uses a heuristic to mitigate this:

For each selected position:
- With probability **80%**: replace with `[MASK]`
- With probability **10%**: replace with a random token from the vocabulary
- With probability **10%**: keep the original token unchanged

The 10% random replacement forces the model to maintain a useful representation for all tokens, not just masked ones, because any position could be "corrupted." The 10% unchanged forces the model to produce a good representation even when the token is correct (useful for downstream classification that reads the `[CLS]` token).

### Selection Rate

BERT masks 15% of tokens per sequence. This is a balance: too few and learning is slow; too many and the remaining context provides insufficient signal for recovery.

### Why MLM Cannot Generate

MLM is **not autoregressive**. It predicts masked positions given all other (non-masked) positions simultaneously. There is no principled way to generate a new sequence token-by-token with MLM — you would need to know which positions to mask before generating. This is the fundamental limitation of the MLM objective: excellent for understanding tasks, incompatible with generation.

### Models

BERT, RoBERTa, ALBERT, DeBERTa, ELECTRA (which replaces MLM with replaced token detection).

---

## Permutation Language Modeling (PLM)

### Motivation

PLM (XLNet, Yang et al., 2019) attempts to combine the best of CLM (autoregressive factorization, no mask token artifact) and MLM (bidirectional context). The insight: MLM fails because the mask token is an artifact and the model predicts all masked positions independently (ignoring their inter-dependencies). CLM is sound but strictly left-to-right.

### Formulation

For a sequence of length $T$, consider all $T!$ permutations of the position indices. For a permutation $\mathbf{z} = (z_1, z_2, \ldots, z_T)$, the autoregressive factorization is:

$$p_\theta(\mathbf{x}) = \prod_{t=1}^{T} p_\theta(x_{z_t} \mid x_{z_1}, x_{z_2}, \ldots, x_{z_{t-1}})$$

XLNet trains by sampling permutations and computing the CLM loss *in the permuted order*. The parameters are shared across all permutations. This means position $z_t$ conditions on $\{z_1, \ldots, z_{t-1}\}$, which can include positions to the *right* of $z_t$ in the original sequence.

$$\mathcal{L}_{\text{PLM}} = -\mathbb{E}_{\mathbf{z} \sim \mathcal{Z}_T} \left[ \sum_{t=1}^{T} \log p_\theta(x_{z_t} \mid \mathbf{x}_{z_{<t}}) \right]$$

In practice, XLNet only computes loss on the last $K$ tokens of each permutation (partial prediction target), since the early tokens have very little context and are noisy to train on.

### Two-Stream Self-Attention

A complication: in standard attention, the hidden state $h_t$ at position $t$ encodes *both* the token value $x_t$ and the position $t$. For PLM, we need to predict $x_{z_t}$ but the query representation cannot see $x_{z_t}$ (it's the target). XLNet resolves this with two attention streams:
- **Content stream** $h_{z_t}$: has access to $x_{z_t}$ and its context
- **Query stream** $g_{z_t}$: has access only to position $z_t$ and context $\mathbf{x}_{z_{<t}}$, *not* $x_{z_t}$

Predictions are made from the query stream. This doubles the attention computation during training.

### Practical Status

XLNet showed improvements over BERT on several NLU benchmarks, but the two-stream mechanism is complex and PLM never became the dominant pretraining objective. RoBERTa (which improved MLM training with more data and longer sequences) largely matched XLNet's gains at lower implementation cost.

---

## Prefix Language Modeling

### Formulation

Prefix LM (Raffel et al., 2020; also used in UniLM) is a hybrid: a prefix of length $P$ is encoded with **full bidirectional attention**, while the remainder is predicted autoregressively with causal attention:

$$\mathcal{L}_{\text{prefix}} = -\sum_{t=P+1}^{T} \log p_\theta(x_t \mid x_1, \ldots, x_{t-1})$$

The attention mask is:
```
Positions:  1  2  3 | 4  5  6  7
            [prefix ] [generated ]

Token 1 can attend to: 1, 2, 3
Token 2 can attend to: 1, 2, 3
Token 3 can attend to: 1, 2, 3
Token 4 can attend to: 1, 2, 3, 4
Token 5 can attend to: 1, 2, 3, 4, 5
Token 6 can attend to: 1, 2, 3, 4, 5, 6
```

### Why It's Useful

The prefix gets rich bidirectional context — the model can "read" the prefix before generating. This is natural for tasks like machine translation (read the source, generate the target) or summarization (read the document, generate the summary). T5 uses a variant where the entire input is treated as prefix and the decoder generates the output.

It also works as a zero-shot in-context learning setup: format examples as prefix, generation as completion.

---

## Span Corruption / Denoising (T5)

### Formulation

Span corruption (Raffel et al., 2020) is T5's pretraining objective. Rather than masking individual tokens, contiguous **spans** of tokens are replaced by single sentinel tokens (`<extra_id_0>`, `<extra_id_1>`, etc.), and the model (encoder-decoder) must recover the original spans as output:

**Input**: `"The quick <extra_id_0> fox jumps <extra_id_1> the lazy dog."`

**Target**: `"<extra_id_0> brown <extra_id_1> over"`

Formally, given a sequence $\mathbf{x}$, spans are sampled with mean length $\bar{\ell}$ (T5 uses $\bar{\ell} = 3$) and a corruption rate $\mu$ (T5 uses $\mu = 15\%$ of tokens). The loss is cross-entropy over the target sentinel-interleaved span sequence:

$$\mathcal{L}_{\text{span}} = -\sum_{k} \log p_\theta(s_k \mid \text{corrupted input}, s_{<k})$$

### Advantages Over Token-Level MLM

1. **Efficiency**: The target sequence is shorter than the input (spans are compressed to single tokens). The model spends compute predicting real tokens, not `[MASK]` artifacts.
2. **Bidirectional encoder**: The encoder processes the full (corrupted) input bidirectionally, then the decoder generates the missing spans autoregressively. This gives the best of both worlds for seq2seq tasks.
3. **Naturalness**: Contiguous span masking better models natural language structure (phrases, clauses) than random token masking.

### Sentinel Tokens

Sentinel tokens serve as positional delimiters in the output, allowing the model to produce multiple non-contiguous corrupted spans in a single target sequence without ambiguity. T5 uses 100 sentinel tokens (`<extra_id_0>` through `<extra_id_99>`).

---

## UL2: Mixture of Denoisers

### Motivation

No single objective is universally best. UL2 (Tay et al., 2022) proposes training a single model with a mixture of denoising objectives, enabling it to excel at both generation and understanding tasks.

### Three Denoiser Types

UL2 defines three regimes, denoted by mode tokens prepended to the input:

| Mode Token | Objective | Span Length | Corrupt Rate | Best For |
|------------|-----------|-------------|--------------|----------|
| `[S2S]` | Span corruption (T5-style) | Long (mean ~3–5) | ~15% | Seq2seq, summarization |
| `[NLU]` | Short span, high rate | Short (mean ~3) | ~15–30% | Classification, NLU |
| `[CLM]` | Causal LM (prefix = empty) | Full sequence | 0% | Generation, few-shot |

During training, the objective for each sample is sampled from this mixture. At inference, the mode token primes the model for the desired behavior.

The UL2 loss is:

$$\mathcal{L}_{\text{UL2}} = \sum_{m \in \{\text{S2S, NLU, CLM}\}} \lambda_m \mathcal{L}_m$$

### Result

UL2 20B (encoder-decoder) was competitive with GPT-3 175B (decoder-only) on few-shot tasks while being 8× smaller. This demonstrated that objective design is a significant lever independent of scale.

---

## Contrastive Objectives

### SimCSE

SimCSE (Gao et al., 2021) trains sentence encoders via contrastive learning. For a batch of $N$ sentences:

1. Each sentence $x_i$ is passed through the encoder **twice** with different dropout masks, producing embeddings $\mathbf{h}_i$ and $\mathbf{h}_i^+$.
2. $(\mathbf{h}_i, \mathbf{h}_i^+)$ are treated as positive pairs.
3. All other sentences in the batch serve as negatives.

The loss is **in-batch noise-contrastive estimation (NT-Xent)**:

$$\mathcal{L}_{\text{SimCSE}} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{e^{\text{sim}(\mathbf{h}_i, \mathbf{h}_i^+)/\tau}}{\sum_{j=1}^{N} e^{\text{sim}(\mathbf{h}_i, \mathbf{h}_j^+)/\tau}}$$

where $\text{sim}(\mathbf{u}, \mathbf{v}) = \mathbf{u}^\top \mathbf{v} / (|\mathbf{u}||\mathbf{v}|)$ is cosine similarity and $\tau$ is a temperature hyperparameter.

**Key insight**: Dropout acts as minimal data augmentation. Two passes of the same sentence through BERT-style dropout produce slightly different representations; contrastive training forces these to collapse to a point while pushing apart different sentences. This dramatically improves the uniformity of the embedding space (fixing the "anisotropy" problem of standard LM embeddings).

### Contrastive Objectives vs Language Modeling

Contrastive objectives produce better sentence-level embeddings for retrieval and similarity tasks. They are typically applied as a **fine-tuning objective** on top of a pretrained LM, not as a primary pretraining objective, because they require curated positive/negative pairs and do not teach the model language structure from scratch.

---

## Objective Comparison and Downstream Task Fit

| Objective | Attention | Loss computed on | Suitable for |
|-----------|-----------|-----------------|--------------|
| CLM | Causal (left-to-right) | All positions | Generation, few-shot, in-context learning |
| MLM | Bidirectional | Masked positions only | NLU, classification, NER, QA |
| PLM | Mixed (permuted AR) | All positions | NLU + generation (theoretical) |
| Prefix LM | Prefix bidir + causal | Non-prefix positions | Seq2seq, conditional generation |
| Span corruption | Encoder bidir + decoder causal | Output spans | Seq2seq, NLU, structured generation |
| UL2 | Mixed (multi-objective) | Varies by mode | General-purpose |
| Contrastive | Bidir (usually) | Pair-level | Retrieval, semantic similarity |

**Key patterns:**

- Tasks requiring generation (text, code, dialogue) → CLM or seq2seq with span corruption
- Tasks requiring deep understanding of the full input (sentiment, NLI, span extraction) → MLM or encoder-only
- Tasks requiring both input understanding and generation → encoder-decoder with prefix LM or span corruption
- Retrieval/embedding tasks → contrastive fine-tuning on top of any pretrained model

---

## Why CLM Has Won

Despite MLM producing empirically superior per-task NLU performance on BERT-era benchmarks (2018–2020), CLM (decoder-only) has become the dominant pretraining objective for frontier LLMs. Several reasons explain this convergence:

**1. Simplicity and scalability.** CLM has no masking hyperparameters, no sentinel tokens, no dual-stream attention. The loss is computed over every token in every sequence. This makes implementation simpler and eliminates entire categories of objective-specific bugs.

**2. Evaluation via perplexity is unambiguous.** Because CLM defines a proper probability distribution over sequences, model quality during training is directly measured by perplexity. MLM loss is not a proper language model probability and cannot be converted to perplexity — you cannot compare MLM losses across models with different masking rates.

**3. Emergent few-shot and in-context learning.** The CLM objective naturally teaches the model to complete sequences given a prefix, which is exactly the structure of few-shot prompting. MLM has no analogous mechanism — there is no clean way to provide examples in a masked-token prediction framework.

**4. Instruction following and RLHF compatibility.** Fine-tuning with human feedback (RLHF) produces rewards over generated token sequences. CLM generation is autoregressive and differentiable in the policy gradient sense. MLM models cannot be easily adapted to this paradigm.

**5. Instruction-tuned GPT-3 (InstructGPT) beat encoder-based models at understanding tasks.** Once CLM models were fine-tuned with instructions and RLHF, the empirical gap with MLM models on NLU tasks largely disappeared. This invalidated the main practical argument for MLM at scale.

**6. Unified architecture.** A single decoder-only model can handle classification (read-out from final token), generation, code completion, translation, and summarization — all with the same architecture and the same CLM loss. No task-specific heads required.

---

## References

- Devlin, J., et al. (2019). *BERT: Pre-training of deep bidirectional transformers for language understanding.* NAACL. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- Radford, A., et al. (2019). *Language models are unsupervised multitask learners.* OpenAI Blog. (GPT-2)
- Brown, T., et al. (2020). *Language models are few-shot learners.* NeurIPS. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
- Yang, Z., et al. (2019). *XLNet: Generalized autoregressive pretraining for language understanding.* NeurIPS. [arXiv:1906.08237](https://arxiv.org/abs/1906.08237)
- Raffel, C., et al. (2020). *Exploring the limits of transfer learning with a unified text-to-text transformer.* JMLR. [arXiv:1910.10683](https://arxiv.org/abs/1910.10683)
- Tay, Y., et al. (2022). *UL2: Unifying language learning paradigms.* ICLR 2023. [arXiv:2205.05131](https://arxiv.org/abs/2205.05131)
- Gao, T., et al. (2021). *SimCSE: Simple contrastive learning of sentence embeddings.* EMNLP. [arXiv:2104.08821](https://arxiv.org/abs/2104.08821)
- Liu, Y., et al. (2019). *RoBERTa: A robustly optimized BERT pretraining approach.* [arXiv:1907.11692](https://arxiv.org/abs/1907.11692)

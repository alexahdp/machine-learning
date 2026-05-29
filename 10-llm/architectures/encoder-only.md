# Encoder-Only Models (Masked Language Models)

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [Masked Language Modeling Objective](#2-masked-language-modeling-objective)
3. [Next Sentence Prediction](#3-next-sentence-prediction)
4. [BERT](#4-bert)
5. [RoBERTa](#5-roberta)
6. [ALBERT](#6-albert)
7. [DistilBERT](#7-distilbert)
8. [DeBERTa](#8-deberta)
9. [Use Cases and When NOT to Use Encoder-Only](#9-use-cases-and-when-not-to-use-encoder-only)
10. [Fine-Tuning Patterns](#10-fine-tuning-patterns)
11. [Model Comparison](#11-model-comparison)
12. [References](#references)

---

## 1. Design Philosophy

An encoder-only model processes an entire input sequence simultaneously, with every token attending to every other token (full, bidirectional self-attention). The result is a set of contextual representations — one per token — that encode the token's meaning in light of its full surrounding context.

This bidirectionality is the key distinction from decoder-only models. Consider the word "bank" in:

- "I deposited money at the **bank**."
- "We sat by the river **bank**."

A left-to-right model has only seen the words before "bank" when computing its representation. A bidirectional encoder has seen the entire sentence, so it can disambiguate the word correctly in a single forward pass.

**The trade-off is generation.** Full bidirectional attention is inconsistent with autoregressive generation: you cannot generate token $t$ while attending to tokens $t+1, t+2, \ldots$ that do not yet exist. Encoder-only models are therefore designed for *understanding* — mapping inputs to representations or labels — not for *producing* novel text sequences.

---

## 2. Masked Language Modeling Objective

### The Core Idea

Instead of predicting the next token (causal language modeling), MLM randomly masks some tokens in the input and trains the model to reconstruct them. This forces the model to build representations using both left and right context.

### BERT's Masking Strategy

Given a tokenized sequence, 15% of tokens are selected at random for "masking." Of those selected tokens:

- **80%** are replaced with the special `[MASK]` token
- **10%** are replaced with a random token from the vocabulary
- **10%** are left unchanged

The 80/10/10 split addresses a **training-inference mismatch**: `[MASK]` never appears during fine-tuning or inference. By sometimes keeping the original token or substituting a random one, the model cannot simply learn "predict a token when you see `[MASK]`" — it must maintain useful representations for every token position.

The training loss is computed only over the 15% selected positions:

$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log P(x_i \mid \tilde{x})$$

where $\mathcal{M}$ is the set of masked positions, $x_i$ is the original token, and $\tilde{x}$ is the corrupted input sequence.

### Why Not Autoregressive?

The MLM objective allows training on the entire context simultaneously: at every step, every selected position gets a gradient update informed by its full context. Autoregressive training on a sequence of length $n$ contributes $n$ prediction targets per sequence but each target only sees tokens $1, \ldots, t-1$. MLM is arguably more data-efficient per token for the representation learning goal, but it means the model never learns to generate coherently.

### Comparison of Pretraining Objectives

```
Causal LM (CLM):   [A] [B] [C] → predict [D]
                                 predict [C] given [A][B]
                                 predict [B] given [A]

Masked LM (MLM):   [A] [MASK] [C] [D] → predict [B]
                   Full bidirectional attention used.
```

---

## 3. Next Sentence Prediction

### Original Motivation

BERT's original paper paired MLM with **Next Sentence Prediction (NSP)**: given two segments A and B, predict whether B is the actual next sentence following A in the corpus (50% of the time yes, 50% a randomly sampled sentence).

The intended goal was to teach the model inter-sentence relationships needed for tasks like natural language inference and question answering.

### Why NSP Was Abandoned

RoBERTa (Liu et al., 2019) showed that **NSP hurts or is neutral** for downstream performance. The proposed explanations:

1. **Task is too easy.** A randomly sampled "negative" sentence from a different document is trivially distinguishable from a real continuation — the model can "cheat" using topic coherence cues rather than learning genuine entailment.
2. **Segment mixing degrades MLM quality.** NSP requires training on two short segments rather than one long sequence, which reduces the effective context length available for MLM.

ALBERT replaced NSP with **Sentence Order Prediction (SOP)**, which is harder: both segments are always adjacent (so topic is constant), but their order is sometimes swapped. The model must learn actual discourse coherence rather than topic matching.

---

## 4. BERT

### Architecture

BERT (Bidirectional Encoder Representations from Transformers, Devlin et al., 2018) introduced two model sizes:

| Variant | Layers ($L$) | Hidden ($d$) | Attention Heads ($h$) | Parameters |
|---------|-------------|--------------|----------------------|-----------|
| BERT-base | 12 | 768 | 12 | 110M |
| BERT-large | 24 | 1024 | 16 | 340M |

Each attention head dimension: $d_k = d / h = 64$.

FFN inner dimension: $4 \times d$ (3072 for base, 4096 for large).

Total parameters for BERT-base:

$$\underbrace{30522 \times 768}_{\text{token emb}} + \underbrace{512 \times 768}_{\text{position emb}} + \underbrace{3 \times 768}_{\text{segment emb}} + \underbrace{12 \times (\text{attn} + \text{FFN} + \text{LN})}_{\text{transformer layers}}$$

### Special Tokens

- **`[CLS]`**: Prepended to every input. After encoding, the `[CLS]` representation is a pooled summary of the entire sequence. Used as input to a classification head during fine-tuning.
- **`[SEP]`**: Appended after each segment, signaling segment boundaries.
- **`[MASK]`**: Used during MLM pretraining; never used during fine-tuning.
- **`[PAD]`**: Padding to a fixed sequence length; attended to with attention mask = 0.

### Input Representation

The input embedding is a sum of three learned embeddings:

$$\mathbf{e}_i = \mathbf{e}_{\text{token}}(x_i) + \mathbf{e}_{\text{position}}(i) + \mathbf{e}_{\text{segment}}(s_i)$$

where $s_i \in \{0, 1\}$ identifies which segment the token belongs to.

### Fine-Tuning Paradigm

BERT established the **pretrain-then-fine-tune** paradigm. A task-specific head (often a single linear layer) is placed on top of the `[CLS]` token or per-token representations. The entire model (all 110M/340M parameters) is fine-tuned end-to-end on downstream task data.

```
[CLS] token1 token2 ... [SEP]
  ↓
Classification head → label
```

For token-level tasks (NER, extractive QA):

```
token1 token2 ... tokenN
  ↓      ↓            ↓
 label  label  ...  label
```

---

## 5. RoBERTa

RoBERTa (Liu et al., 2019) is not a new architecture — it is BERT trained *correctly*. The changes are:

### Key Changes from BERT

**1. Remove NSP**
Training on single, full-length sequences instead of segment pairs allows longer context for MLM and removes a noisy auxiliary objective.

**2. Dynamic Masking**
BERT applied masking once to the corpus (static masking). RoBERTa regenerates the masking pattern each time a sequence is fed to the model during training. Over 40 epochs, the same text is seen with 40 different masks. This increases effective data diversity.

**3. Larger Batches**
BERT trained with batch size 256. RoBERTa trains with batch size 8,192 for the same total number of gradient steps, but with a proportionally larger learning rate (following linear scaling rule). Large batch training tends to improve optimization stability.

**4. More Data and Longer Training**
BERT: 16GB text (BooksCorpus + Wikipedia). RoBERTa: 160GB (+ CC-News, OpenWebText, Stories). Training for 500K steps vs BERT's 1M (with larger batches, this corresponds to more tokens seen).

**5. Byte-Pair Encoding (BPE) at Character Level**
RoBERTa uses a byte-level BPE tokenizer that never produces unknown tokens (`[UNK]`), at the cost of a slightly larger vocabulary (50,265 vs 30,522).

### Result

RoBERTa-base outperforms BERT-large on GLUE/SQuAD, demonstrating that BERT was significantly undertrained.

---

## 6. ALBERT

ALBERT (A Lite BERT, Lan et al., 2019) targets **parameter efficiency** — reducing the parameter count while maintaining or improving performance.

### Factorized Embedding Parameterization

BERT ties the vocabulary embedding size $V$ to the hidden dimension $d$: the embedding matrix is $V \times d$. This is wasteful because:
- Vocabulary embeddings benefit from large $V$ (broad coverage)
- But $d$ should be large to capture context (768–1024)
- These are independent requirements coupled unnecessarily

ALBERT factorizes the embedding into two smaller matrices:

$$E \in \mathbb{R}^{V \times e}, \quad W_{\text{proj}} \in \mathbb{R}^{e \times d}$$

with $e \ll d$ (e.g., $e = 128$). Token embeddings are looked up in $E$ and projected to $d$. Parameter saving:

$$V \times d \rightarrow V \times e + e \times d$$

For BERT-base: $30522 \times 768 = 23.4\text{M}$ vs $30522 \times 128 + 128 \times 768 = 3.9\text{M} + 0.1\text{M} = 4\text{M}$.

### Cross-Layer Parameter Sharing

All 12 (or 24) transformer layers share the **same weights**. This drastically reduces parameters: ALBERT-xxlarge (12 layers, $d=4096$) has 235M parameters but achieves BERT-large-equivalent performance.

Sharing works because the layers are learning to apply the same *type* of transformation — refining representations — not fundamentally different computations at each depth.

### Sentence Order Prediction (SOP)

NSP replacement. Both the positive and negative segment pairs are drawn from adjacent sentences; the only question is whether they appear in original or reversed order. This forces the model to model actual textual coherence.

### ALBERT Variants

| Variant | Layers | Hidden | Parameters |
|---------|--------|--------|-----------|
| ALBERT-base | 12 | 768 | 12M |
| ALBERT-large | 24 | 1024 | 18M |
| ALBERT-xxlarge | 12 | 4096 | 235M |

ALBERT-xxlarge matches or exceeds BERT-large (340M) at ~69% of the parameters.

---

## 7. DistilBERT

DistilBERT (Sanh et al., 2019) achieves a different kind of compression: **knowledge distillation** rather than architectural sharing.

### Knowledge Distillation

Knowledge distillation (Hinton et al., 2015) trains a smaller *student* network to mimic a larger *teacher* network. The student is not trained on hard labels alone; it also learns from the teacher's soft probability distribution over the vocabulary (which contains richer information about similarity between classes/tokens).

For MLM, the distillation loss for position $i$ is:

$$\mathcal{L}_{\text{distill}} = -\sum_v p_{\text{teacher}}(v \mid \tilde{x}, i) \log p_{\text{student}}(v \mid \tilde{x}, i)$$

The full DistilBERT loss combines:

$$\mathcal{L} = \alpha \mathcal{L}_{\text{MLM}} + \beta \mathcal{L}_{\text{distill}} + \gamma \mathcal{L}_{\text{cos}}$$

where $\mathcal{L}_{\text{cos}}$ is a cosine embedding loss that aligns hidden state directions between teacher and student.

### Architecture Reduction

DistilBERT reduces BERT-base by:
- Halving the number of layers: 6 instead of 12
- Keeping the same hidden size ($d = 768$) and heads ($h = 12$)
- Removing the token type embeddings (they found these unimportant)

This yields **40% fewer parameters** (66M vs 110M) and runs **60% faster** at inference while retaining ~97% of BERT's GLUE performance.

Layer initialization: student layers are initialized from every other teacher layer (layers 0, 2, 4, 6, 8, 10), exploiting the fact that consecutive layers in BERT are highly correlated.

---

## 8. DeBERTa

DeBERTa (He et al., 2020) introduces two architectural innovations that significantly improve over BERT/RoBERTa.

### Disentangled Attention

Standard BERT computes attention from a single embedding $\mathbf{h}_i$ that conflates both content and position information. DeBERTa **disentangles** content and position into separate vectors.

For tokens $i$ and $j$:

$$A_{ij} = \mathbf{Q}_c^{(i)} \mathbf{K}_c^{(j)T} + \mathbf{Q}_c^{(i)} \mathbf{K}_r^{(i \to j)T} + \mathbf{Q}_r^{(i \to j)} \mathbf{K}_c^{(j)T}$$

- $\mathbf{Q}_c, \mathbf{K}_c$: content-based query/key (what the token means)
- $\mathbf{Q}_r, \mathbf{K}_r$: position-based query/key (where the token is relative to others)

The third term (content-to-position) captures "this specific token cares about tokens at this relative distance." The fourth term (position-to-content) is dropped for efficiency (DeBERTa-v2 adds it back selectively).

This disentanglement is more expressive than adding position and content embeddings before computing attention (as BERT/RoBERTa do), because it models their interactions explicitly rather than forcing a single combined representation.

### Enhanced Mask Decoder (EMD)

During MLM, BERT predicts masked tokens using the encoder output directly. DeBERTa adds an absolute position embedding only at this final decoding stage. The motivation: absolute position is important for predicting masked tokens (a noun after "the" has different plausible words than a noun after "with"), but you don't want absolute position to dominate content-based attention throughout all layers.

### Performance

DeBERTa-xxlarge (1.5B parameters) surpassed human performance on SuperGLUE in 2021. DeBERTa-v3 uses ELECTRA-style pretraining (replaced-token detection instead of MLM) on top of the disentangled attention.

---

## 9. Use Cases and When NOT to Use Encoder-Only

### When to Use

Encoder-only models excel when:

- **You need rich contextual representations**: classification, regression, semantic similarity
- **Input context matters in both directions**: NER, slot filling, extractive QA
- **You want maximum representation quality per parameter** for discriminative tasks
- **Inference speed matters**: at 7B-70B decoder-only scale, even BERT-large is tiny

| Task | Approach |
|------|----------|
| Text classification | `[CLS]` → linear head |
| Named entity recognition | Per-token → linear head |
| Extractive QA (SQuAD) | Per-token start/end logits |
| Semantic similarity | `[CLS]` embeddings + cosine distance |
| Cross-encoder re-ranking | `[CLS]` of concatenated query+doc |

### When NOT to Use

- **Text generation**: encoder-only models cannot generate — they have no decoder and were not trained autoregressively. Use decoder-only or encoder-decoder.
- **Open-ended tasks**: tasks that require producing novel text sequences (summarization, translation, dialogue) require a generative model.
- **Zero/few-shot prompting**: encoder-only models do not do in-context learning — they require task-specific fine-tuning. GPT-style models are better for prompt-based zero-shot use.
- **Very long documents**: BERT's maximum sequence length is 512 tokens. For longer documents, you need chunking, longformer, or a long-context decoder-only model.

---

## 10. Fine-Tuning Patterns

### Sequence Classification

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base", num_labels=2
)

inputs = tokenizer(
    "This movie was fantastic!",
    return_tensors="pt",
    truncation=True,
    max_length=512,
)
outputs = model(**inputs)
logits = outputs.logits  # shape: (1, 2)
predicted_class = logits.argmax(dim=-1).item()
```

### Token Classification (NER)

```python
from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    "dbmdz/bert-large-cased-finetuned-conll03-english"
)
outputs = model(**inputs)
# outputs.logits: (batch, seq_len, num_labels)
```

### Extractive Question Answering

```python
from transformers import AutoModelForQuestionAnswering, pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2"
)
result = qa_pipeline(
    question="What is the capital of France?",
    context="Paris is the capital and most populous city of France."
)
# result: {'score': ..., 'start': ..., 'end': ..., 'answer': 'Paris'}
```

### Sentence Embeddings (Semantic Similarity)

```python
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()
    ).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / \
           torch.clamp(input_mask_expanded.sum(1), min=1e-9)

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-roberta-large-v1")
model = AutoModel.from_pretrained("sentence-transformers/all-roberta-large-v1")

sentences = ["A cat sat on the mat.", "A dog lay on the rug."]
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

embeddings = mean_pooling(outputs, inputs["attention_mask"])
embeddings = F.normalize(embeddings, p=2, dim=1)
similarity = (embeddings[0] @ embeddings[1]).item()
```

### Training Loop (Condensed)

```python
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_ratio=0.06,
    weight_decay=0.01,
    learning_rate=2e-5,          # typical range: 1e-5 to 5e-5 for BERT
    lr_scheduler_type="linear",
    evaluation_strategy="epoch",
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()
```

**Key fine-tuning tips**:
- Learning rates in $[1 \times 10^{-5}, 5 \times 10^{-5}]$ — BERT is sensitive to LR
- Warmup for ~6% of training steps prevents early instability
- Fine-tuning typically converges in 2–4 epochs; more causes catastrophic forgetting
- Use `[CLS]` pooling for classification; mean-pooling over all tokens often works better for similarity

---

## 11. Model Comparison

| Model | Params | GLUE | Speed (rel.) | Key Innovation |
|-------|--------|------|--------------|----------------|
| BERT-base | 110M | 79.6 | 1× | Bidirectional MLM + NSP |
| BERT-large | 340M | 82.1 | 0.3× | Scale |
| RoBERTa-base | 125M | 83.2 | 0.9× | Better training recipe, no NSP |
| RoBERTa-large | 355M | 86.4 | 0.25× | Scale + better training |
| ALBERT-base | 12M | 79.6 | 1.7× | Param sharing, factorized emb |
| ALBERT-xxlarge | 235M | 90.9 | 0.35× | Scale + sharing |
| DistilBERT | 66M | 77.0 | 1.6× | Knowledge distillation |
| DeBERTa-base | 140M | 84.5 | 0.7× | Disentangled attention |
| DeBERTa-xxlarge | 1.5B | 91.0+ | 0.1× | Best pre-GPT-4 NLU |

*GLUE scores are approximate; vary by task and fine-tuning recipe.*

**When to choose which**:
- **Latency-critical production**: DistilBERT or ALBERT-base
- **Maximum NLU quality**: DeBERTa-xxlarge or RoBERTa-large
- **Baseline/prototype**: RoBERTa-base (best quality/speed trade-off at reasonable scale)
- **Multilingual**: mBERT, XLM-R (RoBERTa-style, trained on 100 languages)

---

## References

- Devlin, J., et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL 2019. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- Liu, Y., et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach*. [arXiv:1907.11692](https://arxiv.org/abs/1907.11692)
- Lan, Z., et al. (2019). *ALBERT: A Lite BERT for Self-supervised Learning of Language Representations*. ICLR 2020. [arXiv:1909.11942](https://arxiv.org/abs/1909.11942)
- Sanh, V., et al. (2019). *DistilBERT, a distilled version of BERT*. [arXiv:1910.01108](https://arxiv.org/abs/1910.01108)
- He, P., et al. (2020). *DeBERTa: Decoding-enhanced BERT with Disentangled Attention*. ICLR 2021. [arXiv:2006.03654](https://arxiv.org/abs/2006.03654)
- Hinton, G., Vinyals, O., Dean, J. (2015). *Distilling the Knowledge in a Neural Network*. [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)

---

*[← Back to Architectures](./README.md) | [→ Decoder-Only Models](./decoder-only.md)*

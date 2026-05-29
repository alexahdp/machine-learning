# Encoder-Decoder Models (Sequence-to-Sequence)

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [Cross-Attention](#2-cross-attention)
3. [T5: Text-to-Text Transfer Transformer](#3-t5-text-to-text-transfer-transformer)
4. [BART: Denoising Pretraining](#4-bart-denoising-pretraining)
5. [Multilingual Variants: mT5 and mBART](#5-multilingual-variants-mt5-and-mbart)
6. [UL2: Unifying Pretraining Objectives](#6-ul2-unifying-pretraining-objectives)
7. [When to Use Encoder-Decoder](#7-when-to-use-encoder-decoder)
8. [Encoder-Decoder vs Decoder-Only for Generation](#8-encoder-decoder-vs-decoder-only-for-generation)
9. [Code Examples](#9-code-examples)
10. [References](#references)

---

## 1. Design Philosophy

The encoder-decoder architecture maintains the original Transformer split (Vaswani et al., 2017): a separate encoder stack processes the input, and a separate decoder stack generates the output, connected by cross-attention.

The **key structural insight** is that encoder and decoder play fundamentally different roles:

- **Encoder**: reads the input *in full*, bidirectionally, without any generation pressure. Every encoder token can attend to every other encoder token. The encoder produces a rich, contextualized representation of the input.
- **Decoder**: generates the output autoregressively, token by token, attending to past output tokens (via causal self-attention) *and* attending to the full encoder output (via cross-attention).

This separation is natural for tasks where input and output are distinct objects:

```
Input:  [x₁ x₂ x₃ x₄]   ← Encoder (bidirectional attention)
              ↓ (cross-attention)
Output: [y₁ y₂ y₃]       ← Decoder (causal + cross-attention)
```

The encoder-decoder architecture predates large-scale pretraining — it was the standard for neural machine translation (seq2seq, Bahdanau attention) before BERT and GPT. What T5 and BART contributed was demonstrating how to *pretrain* this architecture effectively on large corpora.

---

## 2. Cross-Attention

### Mechanism

In each decoder block, after the causal self-attention sublayer, there is a cross-attention sublayer. Queries come from the decoder hidden state; keys and values come from the encoder output:

$$\mathbf{Q}_{\text{dec}} = \mathbf{h}_{\text{dec}} W_Q, \quad \mathbf{K}_{\text{enc}} = \mathbf{h}_{\text{enc}} W_K, \quad \mathbf{V}_{\text{enc}} = \mathbf{h}_{\text{enc}} W_V$$

$$\text{CrossAttn} = \text{softmax}\!\left(\frac{\mathbf{Q}_{\text{dec}} \mathbf{K}_{\text{enc}}^T}{\sqrt{d_k}}\right) \mathbf{V}_{\text{enc}}$$

The cross-attention score matrix is $(n_{\text{dec}} \times n_{\text{enc}})$: each decoder position produces one attention distribution over all encoder positions. This is the mechanism by which the decoder "reads" the input at each generation step.

### Structural Comparison

```
Decoder-only block:
  Input → CausalSelfAttn → FFN → Output

Encoder-decoder decoder block:
  Input → CausalSelfAttn → CrossAttn(encoder output) → FFN → Output
```

Cross-attention adds parameters ($W_Q, W_K, W_V$ for the cross-attention sublayer in each decoder layer) and computation. For an encoder with $n_{\text{enc}}$ tokens and decoder with $n_{\text{dec}}$ tokens, the cross-attention cost is $O(n_{\text{dec}} \times n_{\text{enc}})$ per layer.

### Information Flow

The encoder runs once per input sequence. Its output $\mathbf{h}_{\text{enc}}$ is fixed during decoding. Each decoder step queries the entire encoder output via cross-attention, extracting whichever encoder positions are relevant for the current generation step.

This is why encoder-decoder models are efficient for tasks like translation: the encoder processes the source sentence once ($O(n_{\text{enc}}^2)$), then all decoder steps share this cached encoding.

---

## 3. T5: Text-to-Text Transfer Transformer

### Text-to-Text Framing

T5 (Raffel et al., 2019) introduces a radical simplification: **every task is reformulated as text-in, text-out**. There are no task-specific output heads. Classification, regression, translation, summarization, QA — all are presented to the same model with the same loss (cross-entropy on the target text):

| Task | Input | Target |
|------|-------|--------|
| Sentiment | `sentiment: This movie is great.` | `positive` |
| Translation | `translate English to German: That is good.` | `Das ist gut.` |
| Summarization | `summarize: [long article...]` | `[summary]` |
| QA | `question: Who wrote Hamlet? context: Hamlet is a play by Shakespeare.` | `Shakespeare` |
| NLI | `mnli hypothesis: I am sick. premise: I am at the doctor.` | `entailment` |

Task prefixes (like `summarize:`, `translate English to German:`) tell the model which task it is performing. The model is a single encoder-decoder with no task-specific components.

This framing simplifies multi-task learning: a single model, single loss, different text prefixes.

### Architecture

T5 closely follows the original Transformer with modifications:

- **Relative positional bias** (not absolute): rather than adding a positional embedding to token embeddings, T5 adds a learned scalar bias to attention logits based on the relative distance between positions. This supports generalization to different sequence lengths.
- **Simplified Layer Norm**: no bias, no learned additive shift (similar to RMSNorm).
- **Pre-norm**: layer norm applied before each sublayer.

| Variant | Encoder Layers | Decoder Layers | $d_{\text{model}}$ | Params |
|---------|---------------|---------------|---------------------|--------|
| T5-small | 6 | 6 | 512 | 60M |
| T5-base | 12 | 12 | 768 | 220M |
| T5-large | 24 | 24 | 1024 | 770M |
| T5-XL | 24 | 24 | 2048 | 3B |
| T5-XXL | 24 | 24 | 4096 | 11B |

T5's encoder and decoder have the same number of layers, which is unusual — typically the encoder and decoder layers can differ.

### Span Corruption Objective

T5 does not use MLM (predict individual masked tokens). Instead, it uses **span corruption** (also called span masking):

1. Sample contiguous spans of tokens to mask (mean span length = 3 tokens)
2. Replace each span with a single sentinel token (`<extra_id_0>`, `<extra_id_1>`, etc.)
3. Target: the sentinel tokens followed by their original spans

```
Input:  "The quick brown <extra_id_0> over the <extra_id_1> dog."
Target: "<extra_id_0> fox jumps <extra_id_1> lazy"
```

Span corruption is more efficient than token-level MLM because:
- Fewer sentinel tokens = shorter input sequence = more context fits in the window
- Generating spans (not single tokens) as targets provides a stronger generative training signal

The default masking rate is 15% (same as BERT), but masking is on spans rather than individual tokens.

### T5-v1.1 and Flan-T5

**T5-v1.1** (2020): bug fixes and improvements over original T5:
- GeGLU activation instead of ReLU
- No pre-training on downstream tasks (pure unsupervised pretraining)
- Larger FFN hidden size (multiplier changed from 4 to 2/3 × 8 = 5.33 in some variants)

**Flan-T5** (Chung et al., 2022): T5 fine-tuned on a massive collection of instruction-formatted tasks (FLAN: Finetuned Language Models are Zero-Shot Learners). Flan-T5-XXL substantially outperforms T5-XXL on held-out tasks by learning to generalize from the instruction format rather than the task prefix.

---

## 4. BART: Denoising Pretraining

### Core Idea

BART (Lewis et al., 2019) pretrains the full encoder-decoder Transformer using a **denoising objective**: corrupt the input text in various ways and train the model to reconstruct the original.

This is more general than T5's span corruption. The encoder sees the corrupted input; the decoder autoregressively reconstructs the original sequence.

### Corruption Strategies

BART experiments with several noise functions:

**1. Token Masking** (like BERT-MLM)
Individual tokens are replaced with `[MASK]`. Decoder must reconstruct them.

**2. Token Deletion**
Random tokens are removed entirely (not masked — they are simply absent). The decoder must determine *which positions* were deleted as well as *what values* to fill in.

**3. Text Infilling**
Spans of tokens (Poisson-distributed length, $\lambda=3$) are replaced with a single `[MASK]` token. The decoder must reconstruct the original span, which may be longer or shorter than the single mask token. This is similar to T5 span corruption but in a denoising framing.

**4. Sentence Permutation**
Sentences are shuffled randomly. Decoder reconstructs the original sentence order.

**5. Document Rotation**
The document is rotated so it starts at a random token. Decoder reconstructs the original start.

BART finds that **text infilling alone** (or combined with sentence permutation for summarization) works best for most downstream tasks.

### Architecture

BART uses a standard encoder-decoder Transformer:
- Encoder: bidirectional attention (like RoBERTa/BERT)
- Decoder: causal attention + cross-attention (like GPT)

BART-base: 6 encoder + 6 decoder layers (139M params)
BART-large: 12 + 12 layers (400M params)

The encoder is initialized from a BERT-style model; the decoder from a GPT-style model. This initialization is important: it lets BART leverage both bidirectional and unidirectional pretraining simultaneously.

### Why BART Excels at Summarization and Translation

Summarization requires both understanding (encoder: read the full document) and generation (decoder: write a fluent, concise summary). The denoising objective closely mirrors the summarization task: reconstruct the original from a corrupted version is structurally similar to generating a clean summary from a long, redundant document.

BART-large achieves state-of-the-art results on CNN/DailyMail and XSum summarization, and competitive results on WMT translation tasks.

---

## 5. Multilingual Variants: mT5 and mBART

### mT5

mT5 (Xue et al., 2020) scales T5 to 101 languages by pretraining on mC4 (multilingual Common Crawl). Architecture and objectives are identical to T5.

Key design question: **shared vocabulary**. mT5 uses a 250K sentencepiece vocabulary (vs T5's 32K), sampling text from each language proportional to $p_L^{\alpha}$ where $\alpha=0.7$ (upsampling low-resource languages).

Challenge: **language forgetting**. High-resource languages (English) dominate training data even with upsampling. Flan-mT5 addresses this with multilingual instruction tuning.

### mBART

mBART (Liu et al., 2020) extends BART's denoising pretraining to 25 languages simultaneously. Key design choices:

- Single shared vocabulary (250K BPE) across all languages
- Language ID token appended to source and target sequences
- Full denoising (text infilling + sentence permutation) on each language independently

mBART enables **zero-shot and few-shot multilingual translation**: after pretraining multilingually, fine-tuning on a few thousand parallel sentences in any language pair achieves competitive translation quality.

mBART-50 (Tang et al., 2020) extends to 50 languages and achieves strong many-to-many translation.

---

## 6. UL2: Unifying Pretraining Objectives

UL2 (Tay et al., 2022) argues that no single pretraining objective is optimal for all downstream tasks. Different tasks benefit from different pretraining biases:

| Pretraining Objective | Best For |
|----------------------|----------|
| Causal LM (CLM) | Generation, open-ended tasks |
| Masked LM (MLM) | Discriminative tasks, NLU |
| Prefix LM | Tasks with structured inputs |
| Span corruption | Summarization, translation |

UL2 unifies these via **Mixture-of-Denoisers (MoD)**. Three "denoiser" modes:

- **R-denoiser** (Regular, like T5 span corruption): short spans, low masking rate. Models local span reconstruction.
- **S-denoiser** (Sequential, like CLM): the model must predict a large suffix of the document from the prefix. Trains on sequential generation.
- **X-denoiser** (Extreme): very long spans or high masking rates. Models document-level reconstruction.

Each training example is prepended with a mode token (`[R]`, `[S]`, `[X]`), and the model learns all three behaviors simultaneously.

UL2 20B outperforms T5-XXL 11B on most tasks and GPT-3 175B on many tasks while being 8× smaller. Flan-UL2 (UL2 + instruction tuning) is competitive with much larger models.

---

## 7. When to Use Encoder-Decoder

### Strong Cases

**Translation**: The clearest win for encoder-decoder. Source and target are in different languages with different lengths. The encoder processes the source completely; the decoder generates the translation with full source access via cross-attention.

**Abstractive Summarization**: Long source document → short summary. The encoder processes the full document (potentially with hierarchical chunking for very long documents). The decoder generates the summary using cross-attention to the salient source positions.

**Structured Prediction**: Tasks where the output is structured relative to the input — table-to-text, data-to-text, code generation from specifications. The explicit encoding step captures input structure; the decoder generates the output format.

**Conditional Generation with Complex Inputs**: When the input conditioning is complicated (multiple documents, structured data, long context) and the output is a relatively short generated text.

### Marginal Cases (Decoder-Only Often Equivalent)

Since ~2022, large decoder-only models handle summarization and translation effectively via prompting. The cross-attention benefit only materializes when:
- The input is substantially longer than the output
- The model explicitly benefits from reading the full input before generating
- You have dedicated fine-tuning data and need maximum quality at small model size

For production systems at small scale (< 1B params), T5 or BART fine-tuned on a specific task often outperforms a prompt-based decoder-only model of similar size.

---

## 8. Encoder-Decoder vs Decoder-Only for Generation

### Architectural Trade-offs

| Aspect | Encoder-Decoder | Decoder-Only |
|--------|----------------|--------------|
| Input processing | Bidirectional full attention | Causal (left-to-right) |
| Input-output coupling | Explicit (cross-attention) | Implicit (input in context window) |
| Parameters for same task | Encoder + Decoder (~2× base) | Single stack |
| Prefill efficiency | O(n²) encoder + O(n²) decoder | O(n²) single pass |
| Decode efficiency | KV cache for decoder only | KV cache grows with total length |
| In-context learning | Weak | Strong (GPT-3 era discovery) |
| Few-shot adaptation | Requires fine-tuning | Prompt engineering viable |

### KV Cache Implications

For a decoder-only model processing a prompt of length $L_p$ and generating $L_g$ tokens:
- Total KV cache = $O(L_p + L_g)$ growing with generation

For an encoder-decoder model with input $L_p$ and generating $L_g$ tokens:
- Encoder KV: $O(L_p)$, computed once and fixed
- Decoder KV: $O(L_g)$, growing with generation
- Cross-attention keys/values: $O(L_p)$, computed once from encoder, accessed at every decode step

The encoder-decoder model has a fixed encoder cache that does not grow. For long-input, short-output tasks (e.g., summarizing a 10K-word document to 100 words), the encoder-decoder has better memory characteristics.

### Why Decoder-Only Won Anyway

1. Instruction tuning collapses the task structure: with a good instruction-tuned decoder-only model, you don't need separate encoder-decoder fine-tuning — you just write a prompt.
2. Scaling investments concentrated on decoder-only (GPT, LLaMA, etc.). Research effort produces better decoder-only models faster.
3. Unified inference: one model, one serving stack, handles all tasks.

---

## 9. Code Examples

### T5 for Summarization

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-large",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

article = """
The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical
rainforest in the Amazon biome that covers most of the Amazon basin of South America.
This basin encompasses 7,000,000 km² (2,700,000 sq mi), of which
5,500,000 km² (2,100,000 sq mi) are covered by the rainforest.
"""

# T5-style task prefix
inputs = tokenizer(
    "summarize: " + article,
    return_tensors="pt",
    max_length=512,
    truncation=True,
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        num_beams=4,             # beam search for higher quality
        early_stopping=True,
        no_repeat_ngram_size=3,  # avoid repetition
    )

summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(summary)
```

### BART for Summarization

```python
from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    torch_dtype=torch.bfloat16,
    device=0,
)

result = summarizer(
    article,
    max_length=130,
    min_length=30,
    do_sample=False,   # deterministic beam search
)
print(result[0]["summary_text"])
```

### T5 for Translation

```python
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")

inputs = tokenizer(
    "translate English to French: Machine learning is a subset of artificial intelligence.",
    return_tensors="pt"
)

outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# → L'apprentissage automatique est un sous-ensemble de l'intelligence artificielle.
```

### Fine-Tuning T5 on a Custom Task

```python
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

training_args = Seq2SeqTrainingArguments(
    output_dir="./t5-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=5e-5,
    predict_with_generate=True,   # use generate() for evaluation metrics
    fp16=True,
    generation_max_length=128,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100,      # ignore padding in loss
    pad_to_multiple_of=8,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,  # e.g., ROUGE for summarization
)
trainer.train()
```

### Accessing Encoder Representations

```python
# Get encoder hidden states (useful for retrieval or classification)
inputs = tokenizer("The cat sat on the mat.", return_tensors="pt")
with torch.no_grad():
    encoder_outputs = model.encoder(**inputs)
# encoder_outputs.last_hidden_state: (batch, seq_len, d_model)
# Use mean pooling for sentence-level representation
sentence_embedding = encoder_outputs.last_hidden_state.mean(dim=1)
```

---

## References

- Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Raffel, C., et al. (2019). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*. JMLR 2020. [arXiv:1910.10683](https://arxiv.org/abs/1910.10683)
- Lewis, M., et al. (2019). *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension*. ACL 2020. [arXiv:1910.13461](https://arxiv.org/abs/1910.13461)
- Xue, L., et al. (2020). *mT5: A massively multilingual pre-trained text-to-text transformer*. NAACL 2021. [arXiv:2010.11934](https://arxiv.org/abs/2010.11934)
- Liu, Y., et al. (2020). *Multilingual Denoising Pre-training for Neural Machine Translation*. TACL 2020. [arXiv:2001.08210](https://arxiv.org/abs/2001.08210)
- Tay, Y., et al. (2022). *UL2: Unifying Language Learning Paradigms*. ICLR 2023. [arXiv:2205.05131](https://arxiv.org/abs/2205.05131)
- Chung, H.W., et al. (2022). *Scaling Instruction-Finetuned Language Models*. [arXiv:2210.11416](https://arxiv.org/abs/2210.11416)

---

*[← Decoder-Only Models](./decoder-only.md) | [← Back to Architectures](./README.md) | [→ Mixture of Experts](./mixture-of-experts.md)*

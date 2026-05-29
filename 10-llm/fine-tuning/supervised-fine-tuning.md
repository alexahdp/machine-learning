# Supervised Fine-Tuning (SFT)

## Table of Contents

1. [What SFT Is](#what-sft-is)
2. [Data Format and Prompt Structure](#data-format-and-prompt-structure)
3. [Data Requirements: Quality vs Quantity](#data-requirements-quality-vs-quantity)
4. [Learning Rate Selection](#learning-rate-selection)
5. [Full Fine-tuning vs Head-only Fine-tuning](#full-fine-tuning-vs-head-only-fine-tuning)
6. [Training Duration and Overfitting](#training-duration-and-overfitting)
7. [Catastrophic Forgetting](#catastrophic-forgetting)
8. [The Alignment Tax](#the-alignment-tax)
9. [Evaluation During Fine-tuning](#evaluation-during-fine-tuning)
10. [Common Pitfalls](#common-pitfalls)
11. [When to Use Full Fine-tuning vs PEFT](#when-to-use-full-fine-tuning-vs-peft)
12. [Code Example: HuggingFace Trainer](#code-example-huggingface-trainer)
13. [References](#references)

---

## What SFT Is

Supervised Fine-Tuning is conceptually simple: take a pretrained model and continue training it on a dataset of (input, output) pairs, using standard next-token prediction loss. The model has already learned the mechanics of language during pre-training; SFT teaches it *what kind of outputs to produce* in response to *what kind of inputs*.

The training objective is identical to pre-training — minimize cross-entropy loss over the output tokens:

$$\mathcal{L}_\text{SFT} = -\sum_{t=1}^{T} \log P_\theta(y_t \mid x, y_{<t})$$

where $x$ is the input (instruction or context), $y_{<t}$ are the preceding output tokens, and $y_t$ is the target token at position $t$. Critically, the loss is **only computed over the output tokens**, not the input. The input is fed as context but does not contribute to the gradient, preventing the model from "learning the question" rather than "learning to answer."

This is different from raw language modeling, where loss is computed over every token in the sequence. Masking the input tokens is essential — without it, the model would be pushed to memorize the input format rather than generalize to new inputs.

---

## Data Format and Prompt Structure

Every SFT example must be formatted as a complete string the model can process. The canonical structure is:

```
[SYSTEM PROMPT]
[USER INPUT / INSTRUCTION]
[ASSISTANT RESPONSE]
```

In practice, special tokens delimit these sections. For example, using ChatML format:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Explain the difference between precision and recall.<|im_end|>
<|im_start|>assistant
Precision measures what fraction of predicted positives are actually positive...
<|im_end|>
```

The loss mask then covers only the tokens from `Precision` onward. The system and user portions contribute to the attention context but not to the gradient.

**Why the format matters**: Inconsistent formatting across examples is one of the most common sources of poor SFT performance. The model will learn whatever structural patterns appear in the training data. If 80% of examples use one prompt template and 20% use another, the model will exhibit inconsistent behavior at inference time when it sees the minority template.

---

## Data Requirements: Quality vs Quantity

The relationship between data quantity and fine-tuning quality is roughly logarithmic — doubling the dataset yields diminishing returns far faster than in pre-training. The dominant factor is **quality**, defined as:

- **Correctness**: the output is factually accurate and logically sound
- **Completeness**: the response fully addresses the input
- **Consistency**: similar inputs produce consistently formatted and styled outputs
- **Coverage**: the dataset covers the distribution of inputs you expect at deployment

The seminal finding from InstructGPT (Ouyang et al., 2022) is that RLHF fine-tuning with ~13,000 high-quality demonstrations beat the 175B prompted GPT-3 on human preference ratings, while a 1.3B model fine-tuned with SFT+RLHF was preferred over the 175B prompt-only baseline. The quality of the fine-tuning data, not its volume, was the key lever.

More concretely, Lima (Zhou et al., 2023) fine-tuned LLaMA-65B on exactly 1,000 carefully curated examples and matched or exceeded models trained on datasets 100x larger. The explanation: the pretrained model already knows *how* to reason, write, and explain. SFT is teaching it *when* and *in what style* to apply those capabilities. A few thousand high-quality demonstrations are sufficient to unlock this.

**Practical heuristics**:
- Aim for 1K–100K examples depending on task complexity
- Prefer a diverse 1K over a redundant 10K
- Filter aggressively: remove examples where the output is wrong, truncated, or stylistically inconsistent
- Sample examples from the deployment distribution — SFT generalizes poorly to out-of-distribution inputs

---

## Learning Rate Selection

Fine-tuning learning rates must be dramatically smaller than pre-training rates. Pre-training uses large rates (e.g., $3 \times 10^{-4}$ for AdamW on LLaMA) because the model starts random and must make large parameter updates. Fine-tuning starts from a well-converged checkpoint — large updates destroy the knowledge already encoded.

Typical SFT learning rates: $1 \times 10^{-5}$ to $5 \times 10^{-5}$ (1–2 orders of magnitude smaller than pre-training).

The optimal rate depends on:
- **Model size**: larger models are more sensitive; 70B models often need rates at the low end ($\sim 1 \times 10^{-5}$)
- **Dataset size**: more data tolerates slightly higher rates
- **Number of epochs**: with more epochs, lower rates prevent over-fitting
- **Divergence from pre-training distribution**: larger domain shift may require slightly higher rates to update representations

A cosine schedule with linear warmup (typically 3–10% of total steps) is standard. The warmup prevents large early updates from destabilizing the checkpoint.

**AdamW weight decay**: $0.01$–$0.1$ is typical. Unlike pre-training where $0.1$ is standard, fine-tuning sometimes uses lighter regularization ($0.01$) to allow the model to adapt more freely to the task distribution.

---

## Full Fine-tuning vs Head-only Fine-tuning

**Head-only fine-tuning** (also called *linear probing*) freezes all transformer parameters and trains only the final classification head. This is common in discriminative settings (sentiment classification, NLI) where the task is mapping the model's representations to a small output space.

**Full fine-tuning** updates all parameters — every attention matrix, FFN weight, and embedding. This is standard for generative fine-tuning where the model must produce arbitrary text.

The trade-off:

| | Head-only | Full Fine-tuning |
|---|---|---|
| Memory (params) | Very low | Full model |
| Memory (optimizer) | Very low | 2–3x model size |
| Catastrophic forgetting | None | Significant risk |
| Task performance (sufficient data) | Lower | Higher |
| Training time | Very fast | Slow |
| Typical use | Classifiers, embeddings | Generative tasks, chat |

For generative LLMs, head-only fine-tuning is rarely used because the output space is the full vocabulary — there is no separate "head" in the sense of a discriminative model. The relevant alternative is PEFT (see [parameter-efficient-fine-tuning.md](./parameter-efficient-fine-tuning.md)).

---

## Training Duration and Overfitting

Unlike pre-training where models are typically trained for a single epoch over a vast corpus, SFT typically runs for **1–5 epochs** over a smaller dataset. With a 10K-example dataset, 3 epochs might be perfectly adequate; with 1,000 examples and 5 epochs, overfitting is a serious risk.

**Signs of overfitting in SFT**:
- Training loss continues to decrease while validation loss plateaus or rises
- Model outputs on held-out prompts become shorter and more terse (mode collapse toward high-confidence, low-entropy responses)
- Repetitive phrasing: the model echoes phrases from training examples verbatim
- Benchmark regression: general capabilities (MMLU, GSM8K) decline as training progresses

**Monitoring strategy**:
- Hold out 5–10% of the dataset as a validation split
- Evaluate validation loss every 100–500 steps
- Track 2–3 downstream benchmarks (e.g., MMLU, a task-specific benchmark) every epoch
- Use early stopping: halt when validation loss stops improving for 1–2 evaluation intervals

The key insight is that **overfitting in language models manifests behaviorally before it manifests numerically**. Validation loss may decrease (or remain flat) while behavioral quality degrades. Regular qualitative evaluation — reading actual model outputs on test prompts — is as important as numeric monitoring.

---

## Catastrophic Forgetting

When a pretrained model is fine-tuned on a narrow dataset, gradient updates that optimize for the fine-tuning objective can degrade parameters that encode pre-training knowledge. This is **catastrophic forgetting**: the model's general capabilities erode as it specializes.

Formally, the problem arises because the gradient of the fine-tuning loss $\nabla_\theta \mathcal{L}_\text{SFT}$ may point in a direction that conflicts with the curvature of $\mathcal{L}_\text{pretrain}$ — parameters important for general language understanding are modified in ways that harm those capabilities.

**Elastic Weight Consolidation (EWC)** (Kirkpatrick et al., 2017) addresses this by adding a regularization term that penalizes changes to parameters weighted by their importance to the original task:

$$\mathcal{L}_\text{EWC} = \mathcal{L}_\text{SFT} + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_i^*)^2$$

where $\theta_i^*$ are the pretrained parameter values, $F_i$ is the Fisher information (an approximation of the parameter's importance to the pre-training objective), and $\lambda$ controls the regularization strength.

EWC is computationally expensive (requires computing Fisher information over the pre-training data) and rarely used in practice for LLMs. More practical mitigations:

1. **Lower learning rate**: smaller updates = less forgetting
2. **Fewer epochs**: don't over-train on narrow data
3. **Data mixing**: include a small percentage of pre-training-like data in the SFT dataset
4. **PEFT methods**: LoRA only modifies a small fraction of the parameter space, limiting forgetting
5. **Replay**: periodically include general-domain examples in training batches

---

## The Alignment Tax

Fine-tuning for safety or instruction-following can paradoxically reduce a model's raw capabilities — this is called the **alignment tax**. It manifests as performance regression on reasoning benchmarks (GSM8K, MATH, MMLU) after safety fine-tuning.

The mechanism is subtle. Safety fine-tuning often includes refusal examples, where the model declines to answer. But the model cannot perfectly discriminate "genuinely harmful request" from "looks superficially like a harmful request." It learns a broader refusal pattern that also triggers on benign edge cases. Additionally, safety data often emphasizes hedged, qualified language ("it depends," "there are multiple perspectives") which can reduce the confidence and directness that improves performance on benchmark tasks.

InstructGPT observed that their 1.3B SFT+RLHF model scored worse than the raw 1.3B on MMLU — the alignment tax in action. Mitigations:

- **Data blending**: include capability-maintaining data alongside alignment data
- **Staged fine-tuning**: optimize for capability first, then apply alignment fine-tuning with light regularization
- **PEFT for alignment**: apply alignment-specific adapters, preserving base model weights

---

## Evaluation During Fine-tuning

Track two complementary signals throughout training:

**1. Loss-based metrics**
- Training loss (expected to decrease)
- Validation loss on a held-out split (should decrease then plateau; rising indicates overfitting)

**2. Downstream task metrics**
- Task-specific accuracy on a test split
- General benchmarks (MMLU, HellaSwag) to detect capability regression
- Human preference ratings or LLM-as-judge scores for generative quality

A common mistake is tracking only training loss. A model can achieve very low training loss (by memorizing training examples) while failing completely on new inputs. The validation loss and downstream metrics tell the real story.

For generative models, include at minimum:
- A qualitative review of 20–50 outputs on held-out prompts every epoch
- Automated metrics on a fixed test set (ROUGE, BERTScore, or task-specific accuracy)
- One or two capability regression benchmarks

---

## Common Pitfalls

**Mode collapse**: The model converges to producing a narrow range of outputs regardless of input. Common when training data is too homogeneous (all examples have the same structure or domain). Fix: increase data diversity.

**Repetitive outputs**: The model loops or repeats phrases. Often caused by too many training epochs, too high a learning rate, or training data that itself contains repetitive patterns. Fix: early stopping, lower LR, filter training data.

**Formatting artifacts**: The model produces excessive markdown headers, bullet points, or special tokens that bleed from training examples into unrelated outputs. Fix: normalize formatting in training data; use a formatting-aware loss mask.

**Instruction following regression**: After SFT, the model follows training instructions correctly but fails on paraphrases or structural variations. Indicates overfitting to surface-level patterns. Fix: augment with rephrased versions of instructions.

**Length bias**: The model learns to produce outputs of a specific length (often the mean length of training outputs). Long training responses cause the model to pad; short ones cause it to truncate. Fix: verify the length distribution in training data reflects the desired output distribution.

**Prompt injection from training**: If training examples embed instructions within the input (e.g., "Respond formally"), the model may fail to follow system-level instructions at inference time when the format differs. Fix: use a single, consistent prompt structure throughout training.

---

## When to Use Full Fine-tuning vs PEFT

| Scenario | Recommendation |
|---|---|
| Model < 7B, ample GPU memory | Full fine-tuning |
| Model ≥ 13B, single or few GPUs | LoRA or QLoRA |
| Need to deploy multiple task-specific variants | LoRA (merge adapters per variant) |
| Catastrophic forgetting is a concern | PEFT (limits parameter updates) |
| Maximum task performance is the priority | Full fine-tuning (with careful regularization) |
| Fast iteration / experimentation | LoRA (faster, cheaper) |
| Safety fine-tuning at scale | Full fine-tuning or LoRA depending on model size |

The general rule: **full fine-tuning is preferred when resources allow and forgetting is managed**; PEFT is preferred when resources are constrained or forgetting is a primary concern. For models above 30B parameters, full fine-tuning without model parallelism is infeasible on most hardware setups.

---

## Code Example: HuggingFace Trainer

```python
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
import torch

# Load model and tokenizer
model_name = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # LLaMA has no pad token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # BF16 for training stability
    device_map="auto",
)

# Prepare dataset — assumes {"instruction": str, "output": str} format
dataset = load_dataset("json", data_files="train.jsonl", split="train")
val_dataset = load_dataset("json", data_files="val.jsonl", split="train")

SYSTEM = "You are a helpful assistant."

def format_and_tokenize(example):
    # Build the full text in ChatML format
    full_text = (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction']}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['output']}<|im_end|>"
    )
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=2048,
        padding=False,
        return_tensors=None,
    )

    # Build loss mask: -100 on input tokens, actual ids on output tokens
    # Find where the assistant response starts
    input_text = (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    input_ids_len = len(tokenizer(input_text, add_special_tokens=False)["input_ids"])
    labels = [-100] * input_ids_len + tokenized["input_ids"][input_ids_len:]

    tokenized["labels"] = labels
    return tokenized

train_dataset = dataset.map(format_and_tokenize, remove_columns=dataset.column_names)
eval_dataset = val_dataset.map(format_and_tokenize, remove_columns=val_dataset.column_names)

# Training arguments
training_args = TrainingArguments(
    output_dir="./sft-output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,       # effective batch = 16
    learning_rate=2e-5,                  # 10x-100x smaller than pre-training
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    bf16=True,                           # BF16 training
    logging_steps=25,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True, pad_to_multiple_of=8
    ),
)

trainer.train()
trainer.save_model("./sft-final")
tokenizer.save_pretrained("./sft-final")
```

Key decisions in the above:
- `bf16=True`: BF16 is more numerically stable than FP16 for fine-tuning LLMs (same dynamic range as FP32, fewer mantissa bits)
- Loss mask via `labels=-100`: HuggingFace automatically ignores tokens with label `-100` in the loss computation
- `gradient_accumulation_steps=4`: simulates a larger effective batch without requiring more GPU memory
- `load_best_model_at_end=True`: saves the checkpoint with lowest validation loss, guarding against over-training

---

## References

- Ouyang, L. et al. (2022). *Training language models to follow instructions with human feedback*. NeurIPS. [InstructGPT]
- Zhou, C. et al. (2023). *LIMA: Less Is More for Alignment*. NeurIPS.
- Kirkpatrick, J. et al. (2017). *Overcoming catastrophic forgetting in neural networks*. PNAS. [EWC]
- Touvron, H. et al. (2023). *LLaMA 2: Open Foundation and Fine-Tuned Chat Models*. arXiv:2307.09288.
- Wei, J. et al. (2022). *Emergent Abilities of Large Language Models*. TMLR.
- HuggingFace. (2024). *TRL: Transformer Reinforcement Learning*. https://github.com/huggingface/trl

---

*← [Fine-tuning Overview](./README.md) | → [Instruction Tuning](./instruction-tuning.md)*

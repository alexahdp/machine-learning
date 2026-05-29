# LLM Fine-tuning

Pre-training gives a model broad competence — it learns grammar, facts, reasoning patterns, and world knowledge from raw text. But a pre-trained model is not yet useful: it completes text continuations, not instructions. It may hallucinate, refuse nothing, or produce outputs in whatever format was most common in its training data. **Fine-tuning** is the process of reshaping that raw competence into reliable, targeted behavior.

The core challenge is that pre-training and fine-tuning pursue different objectives. Pre-training minimizes next-token prediction loss over a vast, diverse corpus. Fine-tuning optimizes for specific human-valued properties — helpfulness, harmlessness, honesty, task accuracy — that are hard to capture in a language modeling objective. Bridging that gap requires a progression of techniques, each addressing limitations of the previous.

## The Fine-tuning Pipeline

Modern LLM fine-tuning follows a staged pipeline, each stage building on the last:

```
Pretrained Base Model
        │
        ▼
┌─────────────────┐
│  Supervised     │  Train on (instruction, response) pairs
│  Fine-Tuning    │  — shapes output format, style, compliance
│  (SFT)         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Reward Model   │  Learn human preferences from comparison data
│  Training       │  — captures what "good" means beyond format
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RL Fine-tuning │  Optimize policy against reward signal
│  (RLHF / DPO)  │  — align behavior with human preferences
└─────────────────┘
```

Not every deployment needs all three stages. A domain-specific assistant might need only SFT. A safety-critical application demands the full RLHF pipeline. Resource-constrained settings use PEFT methods like LoRA to make any stage feasible at scale.

## Why Pretrained Models Need Adaptation

A pretrained causal LM is a **next-token predictor**, not an assistant. Given the prompt `"What is 2+2?"`, it will likely complete with `"What is 2+2? What is 3+3? What is 4+4?"` — continuing the pattern of a math worksheet rather than answering the question. Three types of adaptation are needed:

1. **Format alignment**: Teaching the model to respond rather than continue, to follow conversation structure, to use appropriate output formats.
2. **Capability alignment**: Specializing knowledge or reasoning for a domain (code, medicine, law).
3. **Value alignment**: Teaching the model to be helpful, refuse harmful requests, maintain honesty.

## Contents

| File | Description |
|------|-------------|
| [supervised-fine-tuning.md](./supervised-fine-tuning.md) | Full fine-tuning on labeled examples: learning rates, catastrophic forgetting, data quality, training dynamics |
| [instruction-tuning.md](./instruction-tuning.md) | Fine-tuning on (instruction, response) pairs for generalization: FLAN, Alpaca, chat templates, template formats |
| [parameter-efficient-fine-tuning.md](./parameter-efficient-fine-tuning.md) | LoRA, QLoRA, adapters, prefix tuning — fine-tuning large models without updating all weights |
| [rlhf.md](./rlhf.md) | Reinforcement Learning from Human Feedback: reward modeling, PPO for LLMs, RLAIF, Constitutional AI |
| [direct-preference-optimization.md](./direct-preference-optimization.md) | DPO and successors (IPO, SimPO, KTO, ORPO) — preference optimization without a reward model |

## Key Themes

**Quality beats quantity in fine-tuning data.** Pre-training benefits from scale above almost all else. Fine-tuning inverts this: 1,000 carefully curated examples frequently outperform 100,000 noisy ones. The model already knows how to reason; fine-tuning is teaching it *which* behaviors to express.

**Catastrophic forgetting is real.** Aggressively fine-tuning on a narrow dataset erodes the broad capabilities acquired during pre-training. Regularization techniques, careful learning rate selection, and PEFT methods all mitigate this.

**Parameter efficiency unlocks scale.** Full fine-tuning of a 70B model is impractical for most organizations. LoRA, QLoRA, and adapters make fine-tuning feasible on a single GPU while recovering most of the performance of full fine-tuning.

**RLHF captures preferences that SFT cannot.** Human preferences over outputs — preferring a response that is accurate, concise, and safe — cannot be fully specified in labeled examples. Reward models learn these preferences from comparisons, and RL fine-tuning optimizes for them directly.

**DPO simplifies the pipeline without sacrificing quality.** Direct preference optimization eliminates the separate reward model and PPO training loop, expressing the same alignment objective as a binary classification loss. For many use cases, DPO matches or exceeds RLHF quality with dramatically less engineering complexity.

## Navigation

- [← Back to LLM Overview](../README.md)
- [← Previous: Pre-training](../pretraining/README.md)
- [→ Next: Evaluation](../evaluation/README.md)

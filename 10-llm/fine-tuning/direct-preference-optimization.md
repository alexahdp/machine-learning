# Direct Preference Optimization (DPO) and Successors

## Table of Contents

1. [Motivation: The RLHF Engineering Problem](#motivation-the-rlhf-engineering-problem)
2. [DPO: The Core Insight](#dpo-the-core-insight)
3. [DPO Derivation](#dpo-derivation)
4. [The DPO Loss and Its Intuition](#the-dpo-loss-and-its-intuition)
5. [The β Hyperparameter](#the-β-hyperparameter)
6. [Data Requirements](#data-requirements)
7. [Reference Model: Why It Matters](#reference-model-why-it-matters)
8. [DPO Limitations](#dpo-limitations)
9. [IPO: Identity Preference Optimization](#ipo-identity-preference-optimization)
10. [SimPO: Simple Preference Optimization](#simpo-simple-preference-optimization)
11. [KTO: Kahneman-Tversky Optimization](#kto-kahneman-tversky-optimization)
12. [ORPO: Odds Ratio Preference Optimization](#orpo-odds-ratio-preference-optimization)
13. [DPO vs PPO: When to Use Each](#dpo-vs-ppo-when-to-use-each)
14. [Code Example: TRL DPO Trainer](#code-example-trl-dpo-trainer)
15. [References](#references)

---

## Motivation: The RLHF Engineering Problem

Standard RLHF requires maintaining and training four separate models simultaneously:
1. $\pi_\theta$ — the policy model being trained (full model, trainable)
2. $\pi_\text{ref}$ — the reference policy (frozen SFT checkpoint)
3. $r_\phi$ — the reward model (separately trained, frozen during RL)
4. $V_\psi$ — the value function (trainable)

Plus the PPO training loop itself, which requires rollout generation (slow autoregressive sampling), advantage estimation, and multiple gradient update steps per batch. This infrastructure is:

- **Complex**: Bugs in the value function, KL computation, or reward normalization silently degrade training
- **Unstable**: PPO hyperparameters (clip ratio $\epsilon$, entropy coefficient, learning rates for policy and value) require careful tuning
- **Expensive**: Keeping 4 models in GPU memory simultaneously, plus generating samples
- **Slow to iterate**: Each experiment requires retraining the reward model and then running PPO

**DPO** (Rafailov et al., 2023) eliminates this complexity by showing that the RLHF objective has a **closed-form optimal solution** that can be expressed as a simple binary classification loss — no reward model, no PPO, no value function required.

---

## DPO: The Core Insight

The RLHF objective optimizes:

$$\max_{\pi_\theta} \mathbb{E}_{x \sim D,\, y \sim \pi_\theta(y|x)}\bigl[r(x, y)\bigr] - \beta \cdot \text{KL}\bigl(\pi_\theta(\cdot|x) \,\|\, \pi_\text{ref}(\cdot|x)\bigr)$$

This has a **known closed-form optimal solution**. For any reward function $r$, the optimal policy $\pi^*$ satisfying the KL-constrained objective is:

$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_\text{ref}(y \mid x) \exp\!\left(\frac{1}{\beta} r(x, y)\right)$$

where $Z(x) = \sum_y \pi_\text{ref}(y \mid x) \exp\!\left(\frac{1}{\beta} r(x, y)\right)$ is the partition function (a normalizing constant).

**The key move**: Rearrange this expression to express the reward in terms of the policy:

$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_\text{ref}(y \mid x)} + \beta \log Z(x)$$

The reward can be **implicitly represented** by the ratio of the optimal policy to the reference policy. We no longer need to train a separate reward model — the policy *is* the reward model.

---

## DPO Derivation

Starting from the Bradley-Terry preference model:

$$P(y_w \succ y_l \mid x) = \sigma\bigl(r(x, y_w) - r(x, y_l)\bigr)$$

Substitute the reparameterized reward $r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)} + \beta \log Z(x)$:

$$P(y_w \succ y_l \mid x) = \sigma\!\left(\beta \log \frac{\pi^*(y_w|x)}{\pi_\text{ref}(y_w|x)} + \cancel{\beta \log Z(x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_\text{ref}(y_l|x)} - \cancel{\beta \log Z(x)}\right)$$

The partition function $Z(x)$ cancels because it depends only on $x$, not on $y_w$ or $y_l$:

$$P(y_w \succ y_l \mid x) = \sigma\!\left(\beta \log \frac{\pi^*(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)$$

Now substitute the parameterized policy $\pi_\theta$ in place of $\pi^*$ and write the maximum likelihood objective over the preference dataset $\mathcal{D} = \{(x, y_w, y_l)\}$:

$$\boxed{\mathcal{L}_\text{DPO}(\pi_\theta; \pi_\text{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)}\right)\right]}$$

This is the **DPO loss**. It is a binary cross-entropy loss where the "logit" is the difference in implicit rewards (log probability ratios) between the preferred and rejected responses.

---

## The DPO Loss and Its Intuition

Expand the implicit reward notation: let $\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}$.

The DPO loss becomes:

$$\mathcal{L}_\text{DPO} = -\mathbb{E}\left[\log \sigma\!\left(\hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l)\right)\right]$$

Minimizing this loss does two things simultaneously:

1. **Increases $\hat{r}_\theta(x, y_w)$**: Increases $\pi_\theta(y_w \mid x)$ relative to $\pi_\text{ref}(y_w \mid x)$ — the model assigns higher probability to the preferred response *compared to where it started*
2. **Decreases $\hat{r}_\theta(x, y_l)$**: Decreases $\pi_\theta(y_l \mid x)$ relative to $\pi_\text{ref}(y_l \mid x)$ — the model assigns lower probability to the rejected response

Crucially, these are **relative** adjustments. The model doesn't blindly increase $\pi_\theta(y_w)$ regardless of how probable $y_w$ was to start — it increases the ratio to the reference. If $y_w$ was already highly probable under $\pi_\text{ref}$, the gradient is small. This is the implicit KL regularization built into the formulation.

**Gradient dynamics**: The gradient of the DPO loss with respect to $\theta$ is:

$$\nabla_\theta \mathcal{L}_\text{DPO} = -\beta \mathbb{E}\left[\sigma\!\left(\hat{r}_\theta(y_l) - \hat{r}_\theta(y_w)\right) \cdot \left(\nabla_\theta \log \pi_\theta(y_w) - \nabla_\theta \log \pi_\theta(y_l)\right)\right]$$

The factor $\sigma(\hat{r}_\theta(y_l) - \hat{r}_\theta(y_w))$ is the probability that the model currently "gets the preference wrong" (assigns higher reward to $y_l$ than $y_w$). This is a **self-weighting**: pairs the model already ranks correctly receive smaller gradient updates, while pairs where the model is confused receive larger updates. This is exactly the behavior you want — focus learning effort on examples where the model is still wrong.

---

## The β Hyperparameter

$\beta$ controls the divergence from the reference policy:

- **Small $\beta$ (0.01–0.1)**: The model can move far from $\pi_\text{ref}$. More aggressive optimization, higher risk of forgetting or reward hacking.
- **Large $\beta$ (0.5–1.0)**: The model stays close to $\pi_\text{ref}$. Conservative updates, better knowledge preservation.

In practice, $\beta = 0.1$–$0.3$ is the standard range. Unlike PPO where the KL coefficient requires careful scheduling, DPO's $\beta$ is fixed throughout training and less sensitive to the exact value chosen.

**Interpretation**: $\beta$ is inversely related to the "temperature" of the alignment — how strongly the model favors the preference signal over the original distribution. High $\beta$ = prefer staying near pre-training distribution; low $\beta$ = prefer satisfying preferences even at cost of distribution shift.

---

## Data Requirements

DPO requires a dataset of **(prompt, chosen, rejected) triplets**:

```python
{
    "prompt": "Explain the difference between nuclear fission and fusion.",
    "chosen": "Nuclear fission splits heavy atoms (like uranium-235) to release energy...",
    "rejected": "Fission and fusion are both nuclear reactions. Fission is when atoms split..."
}
```

**Sources of preference data**:
- **Human annotations**: Most reliable, most expensive. Labelers compare model outputs pairwise.
- **AI annotations**: GPT-4 or Claude used as a judge to rate pairs. Cheaper, scalable.
- **Existing RLHF datasets**: Anthropic HH-RLHF, OpenAssistant, UltraFeedback, Capybara.
- **Synthetic construction**: Generate multiple responses with different models or sampling temperatures; use a strong LLM to label which is better.

**Quality vs. quantity**: Similar to SFT, quality is paramount. 10K carefully curated preference pairs often outperform 100K noisily labeled pairs. Key quality criteria:
- The chosen and rejected responses should be meaningfully different — pairs where both are equivalently good or both are equivalently bad provide little training signal
- Avoid pairs where the length difference is the dominant distinguishing feature (the model may learn to be verbose rather than good)
- Include adversarial prompts to teach the model to refuse appropriately

**The chosen/rejected gap**: Pairs with larger quality differences train more effectively. When choosing which pairs to include, prefer pairs where a human (or strong AI) judge has high confidence in the preference, over pairs where the judgment is marginal.

---

## Reference Model: Why It Matters

The reference model $\pi_\text{ref}$ is typically the **SFT checkpoint** — the model fine-tuned on instruction-following demonstrations before preference optimization. This choice is critical:

**If $\pi_\text{ref}$ is the base model (not SFT)**: The reference model has no instruction-following capability. The DPO gradient will need to simultaneously teach format compliance and preference alignment, pulling in two directions. Training becomes unstable and requires more data.

**If $\pi_\text{ref}$ is a well-trained SFT model**: The reference model already knows how to follow instructions and produce coherent responses. DPO only needs to nudge the model toward preferred behaviors, not teach it to respond at all.

**If $\pi_\text{ref}$ is the current policy** (like in iterative DPO / online DPO): The reference is periodically updated, requiring more careful management of the training loop.

The reference model also serves as an implicit constraint: behaviors that $\pi_\text{ref}$ assigns near-zero probability to will not be affected by DPO training (the log ratio goes to $-\infty$, producing extreme values that destabilize training). This means DPO cannot easily teach behaviors that the SFT model never produces. For novel capabilities, SFT must be done first.

---

## DPO Limitations

**1. Distribution mismatch**: DPO is trained on responses generated by $\pi_\text{SFT}$ (the reference model) or some other policy. If the current policy $\pi_\theta$ has drifted from $\pi_\text{ref}$, the preference data may no longer be on-distribution for $\pi_\theta$. This is less problematic than it sounds in practice but motivates **online DPO** (iteratively generating new preference pairs from the current policy).

**2. Overfitting to deterministic preferences**: When one response in a pair is clearly preferred (high-confidence label), DPO loss drives the probability of the rejected response toward zero. This can cause pathological behavior — the model becomes overconfident on the training distribution and refuses to generate alternative responses. IPO (below) addresses this.

**3. No explicit reward signal**: Unlike PPO, there is no reward model that can be queried for arbitrary inputs. This makes it harder to diagnose *why* the model is or isn't improving, and impossible to score new examples without a separately trained reward model.

**4. Winning/losing responses must come from comparable distributions**: If chosen responses are much longer, more detailed, or stylistically different from rejected responses in a systematic way, DPO will learn superficial correlates (length, formatting) rather than genuine preference dimensions.

---

## IPO: Identity Preference Optimization

IPO (Azar et al., 2023) fixes DPO's overfitting problem. The core issue: DPO's loss can be driven to zero when the model assigns probability 1 to $y_w$ and 0 to $y_l$. Real-world preferences are noisy — even "preferred" responses are not always correct, and the optimal policy should not be deterministic.

IPO replaces the Bradley-Terry log-sigmoid with an **identity (squared error) loss**:

$$\mathcal{L}_\text{IPO}(\pi_\theta; \pi_\text{ref}) = \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\!\left[\left(\log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)} - \frac{1}{2\beta}\right)^2\right]$$

The target is $\frac{1}{2\beta}$ rather than infinity. This means the optimal solution is for the model to assign a log-ratio difference of exactly $\frac{1}{2\beta}$ — a finite gap, not an infinite one. The model learns to prefer $y_w$ over $y_l$ by a margin proportional to $\frac{1}{\beta}$, rather than making $y_l$ impossible.

**Effect**: IPO is more robust to noisy preference labels and produces better-calibrated models that maintain the ability to generate diverse outputs.

---

## SimPO: Simple Preference Optimization

SimPO (Meng et al., 2024) takes a different approach to DPO's limitations: it removes the reference model entirely.

The key observation: using a reference model introduces a dependency on $\pi_\text{ref}$, which requires loading a separate (frozen) model during training and doubles memory requirements.

SimPO replaces the log-ratio implicit reward with **average log-likelihood** as the reward:

$$\hat{r}_\text{SimPO}(x, y) = \frac{1}{|y|} \log \pi_\theta(y \mid x) = \frac{1}{|y|} \sum_t \log \pi_\theta(y_t \mid x, y_{<t})$$

The loss becomes:

$$\mathcal{L}_\text{SimPO}(\pi_\theta) = -\mathbb{E}\!\left[\log \sigma\!\left(\frac{\beta}{|y_w|} \log \pi_\theta(y_w \mid x) - \frac{\beta}{|y_l|} \log \pi_\theta(y_l \mid x) - \gamma\right)\right]$$

where $\gamma > 0$ is a **target reward margin** — the model must prefer $y_w$ over $y_l$ by at least $\gamma$, not just by any positive amount.

**Why length normalization**: Without normalization, the model has an incentive to prefer longer responses (longer sequences have lower raw log-likelihood as probabilities multiply). Length normalization neutralizes this bias.

**The $\gamma$ margin**: Unlike DPO where the reference model provides implicit regularization, SimPO needs the explicit margin to prevent trivial solutions. Setting $\gamma > 0$ ensures the model must make a meaningful distinction, not just a marginal one.

**Practical advantage**: Removing $\pi_\text{ref}$ cuts memory requirements approximately in half compared to DPO — critical for large models.

---

## KTO: Kahneman-Tversky Optimization

KTO (Ethayarajh et al., 2023) breaks from the pairwise comparison paradigm entirely. Named after behavioral economists Kahneman and Tversky (prospect theory), it uses **unpaired feedback** — just binary labels (good/bad) rather than explicit comparisons.

**Motivation**: Generating preference pairs requires two responses to the same prompt. Collecting unpaired feedback (labelers simply rate whether a single response is good or bad) is 2x more efficient and covers more of the response space.

**The KTO objective**:

$$\mathcal{L}_\text{KTO}(\pi_\theta; \pi_\text{ref}) = \mathbb{E}\!\left[\lambda_w \cdot \sigma\!\left(\hat{r}_\theta(x, y_w) - z_0\right) + \lambda_l \cdot \sigma\!\left(z_0 - \hat{r}_\theta(x, y_l)\right)\right]$$

where $z_0 = \text{KL}(\pi_\theta \| \pi_\text{ref})$ is the KL divergence (used as a baseline, similar to the value function in RL), $\lambda_w$ and $\lambda_l$ are weights on desired and undesired examples, and $\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}$ is the same implicit reward as in DPO.

**The prospect theory connection**: KTO maximizes "utility" in the sense of Kahneman-Tversky: losses (undesired outputs) are weighted more heavily than gains (desired outputs) of equivalent magnitude. This asymmetry ($\lambda_l > \lambda_w$ typically) reflects how humans psychologically weight negative outcomes more than positive ones — making KTO's preference model more aligned with human psychology than the symmetric Bradley-Terry model.

**Practical benefit**: Any dataset with scalar quality ratings or binary thumbs-up/thumbs-down labels can be used directly, without requiring matched pairs. This dramatically expands the usable data pool.

---

## ORPO: Odds Ratio Preference Optimization

ORPO (Hong et al., 2024) combines SFT and preference optimization into a single training objective, eliminating the need for a separate SFT stage.

**The key idea**: DPO and its variants require a pretrained SFT checkpoint as the reference model. This means the SFT stage must complete before preference optimization begins — two sequential training runs. ORPO unifies them.

**The ORPO loss**:

$$\mathcal{L}_\text{ORPO} = \mathbb{E}\!\left[-\log \pi_\theta(y_w \mid x) - \lambda \cdot \log \sigma\!\left(\log \frac{\text{OR}_\theta(y_w \mid x)}{\text{OR}_\theta(y_l \mid x)}\right)\right]$$

where the **odds ratio** $\text{OR}_\theta(y \mid x) = \frac{\pi_\theta(y \mid x)}{1 - \pi_\theta(y \mid x)}$ is the ratio of the probability of generating $y$ to the probability of *not* generating $y$.

**The first term** $-\log \pi_\theta(y_w \mid x)$ is the standard SFT loss on the chosen response — the model learns to produce good outputs.

**The second term** $-\lambda \log \sigma\!\left(\log \frac{\text{OR}_\theta(y_w)}{\text{OR}_\theta(y_l)}\right)$ penalizes the model for assigning higher odds to the rejected response than the chosen response — the preference signal.

**Why odds ratio instead of log ratio**: The log ratio $\log \frac{\pi_\theta(y_w)}{\pi_\text{ref}(y_w)}$ requires a reference model. The odds ratio $\frac{\pi_\theta(y)}{1-\pi_\theta(y)}$ is self-referential — it measures how much the model *itself* prefers generating $y$ versus not generating $y$, without needing an external reference.

**Practical impact**: ORPO trains in a single stage, halves the compute overhead compared to SFT-then-DPO, and requires no reference model at inference. For resource-constrained settings, this is a significant practical advantage.

---

## DPO vs PPO: When to Use Each

| Factor | DPO | PPO |
|--------|-----|-----|
| Engineering complexity | Low (single training loop) | High (4 models, rollout generation) |
| Compute cost | Low (no sampling during training) | High (sampling is the bottleneck) |
| Memory requirement | 2× model (policy + reference) | 4× model + optimizer states |
| Reward model needed? | No | Yes |
| Online exploration | No (offline, static dataset) | Yes (generates new samples during training) |
| Handles distribution shift | Less robustly | More robustly (online generation) |
| Hyperparameter sensitivity | Low ($\beta$ is fairly stable) | High (5+ critical hyperparameters) |
| Best for | Most preference fine-tuning tasks | Tasks with clear reward signals, long-horizon tasks |
| Typical use case | Safety alignment, instruction following, style | RLHF at frontier scale, code execution feedback |

**When to use DPO**: Start here. For most alignment and instruction-following fine-tuning tasks, DPO achieves comparable or better results than PPO with dramatically less complexity. The offline nature is rarely a limitation if you have a good preference dataset.

**When to use PPO**: When you have a verifiable reward signal (code that compiles and passes tests, math answers that can be checked), online generation is valuable because the model can explore and learn from its own mistakes. PPO is also preferred when the preference dataset would be too small to use offline, but the reward model can generalize — the RL loop generates more training signal.

---

## Code Example: TRL DPO Trainer

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
import torch

model_name = "meta-llama/Llama-3.2-3B-Instruct"  # SFT checkpoint

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Policy model (trainable) — start from SFT checkpoint
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Reference model (frozen) — same SFT checkpoint
# TRL creates this automatically from model if ref_model=None,
# or you can pass a separate model
ref_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Dataset format: must have "prompt", "chosen", "rejected" columns
# Each of "chosen" and "rejected" should be the full response (not including prompt)
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train_prefs")

# For chat models, apply the chat template to format prompt + responses
def format_example(example):
    # "prompt" is the user message; "chosen" / "rejected" are assistant responses
    # TRL's DPOTrainer handles the chat template formatting internally
    # if you provide messages in the correct format
    return example

training_args = DPOConfig(
    output_dir="./dpo-output",
    beta=0.1,                        # KL penalty coefficient
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,   # effective batch = 16
    learning_rate=5e-7,              # DPO LR should be very small (1e-7 to 5e-6)
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=200,
    max_length=2048,                 # max sequence length (prompt + chosen/rejected)
    max_prompt_length=1024,          # max prompt length
    remove_unused_columns=False,
    report_to="none",
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,             # frozen reference model
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model("./dpo-final")
```

**Using LoRA with DPO** (recommended for large models):

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

model = get_peft_model(model, lora_config)

# When using PEFT, set ref_model=None:
# TRL will automatically use the base (non-LoRA) model as the reference,
# which is memory-efficient — only one model copy needed
trainer = DPOTrainer(
    model=model,
    ref_model=None,  # TRL infers reference from the frozen base model when using PEFT
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)
```

**Key hyperparameters and their effects**:

| Hyperparameter | Typical Range | Effect |
|---|---|---|
| `beta` | 0.01–0.5 | Lower = more divergence from reference, higher = more conservative |
| `learning_rate` | 1e-7 – 5e-6 | Much lower than SFT; too high causes collapse |
| `max_prompt_length` | 256–1024 | Truncates prompt if too long; important for memory management |
| `label_smoothing` | 0.0–0.1 | Reduces overconfidence; similar to IPO's effect |

---

## References

- Rafailov, R. et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. NeurIPS 2023.
- Azar, M.G. et al. (2023). *A General Theoretical Paradigm to Understand Learning from Human Feedback* (IPO). AISTATS 2024.
- Meng, Y. et al. (2024). *SimPO: Simple Preference Optimization with a Reference-Free Reward*. NeurIPS 2024.
- Ethayarajh, K. et al. (2023). *KTO: Model Alignment as Prospect Theoretic Optimization*. ICML 2024.
- Hong, J. et al. (2024). *ORPO: Monolithic Preference Optimization without Reference Model*. arXiv:2403.07691.
- Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- HuggingFace TRL. (2024). *TRL: Transformer Reinforcement Learning*. https://github.com/huggingface/trl
- Kahneman, D. and Tversky, A. (1979). *Prospect Theory: An Analysis of Decision under Risk*. Econometrica.

---

*← [RLHF](./rlhf.md) | → [Fine-tuning Overview](./README.md)*

# Reinforcement Learning from Human Feedback (RLHF)

## Table of Contents

1. [Motivation: The Limits of SFT](#motivation-the-limits-of-sft)
2. [The RLHF Pipeline Overview](#the-rlhf-pipeline-overview)
3. [Stage 1: Supervised Fine-Tuning](#stage-1-supervised-fine-tuning)
4. [Stage 2: Reward Model Training](#stage-2-reward-model-training)
5. [Stage 3: RL Fine-tuning with PPO](#stage-3-rl-fine-tuning-with-ppo)
6. [Reward Hacking and Overoptimization](#reward-hacking-and-overoptimization)
7. [Practical Challenges](#practical-challenges)
8. [RLAIF: RL from AI Feedback](#rlaif-rl-from-ai-feedback)
9. [Constitutional AI (Anthropic)](#constitutional-ai-anthropic)
10. [InstructGPT Results: What RLHF Actually Achieves](#instructgpt-results-what-rlhf-actually-achieves)
11. [References](#references)

---

## Motivation: The Limits of SFT

Supervised fine-tuning has a fundamental limitation: it optimizes a model to produce outputs that *look like* human-written demonstrations, but it cannot capture *human preferences over outputs*.

Consider two responses to "Explain quantum entanglement":

> **Response A**: Quantum entanglement is a phenomenon where two particles become correlated such that measuring one instantly affects the other, regardless of distance.

> **Response B**: Quantum entanglement is when particles are like best friends — they always know what the other is doing, even if they're really far apart!

Response A is accurate and precise. Response B is a misleading analogy. Both might appear in a pre-training corpus or even in an SFT dataset. The SFT loss treats them identically — both are valid continuations given some context. The model has no mechanism to distinguish "accurate and useful" from "superficially friendly but misleading."

Human preferences are implicit, context-dependent, and multidimensional. They involve:
- Accuracy vs. simplicity trade-offs
- Appropriate length for the context
- Safety: refusing harmful requests appropriately
- Helpfulness: actually addressing the user's need, not a related but simpler question
- Honesty: acknowledging uncertainty rather than confabulating

RLHF captures these preferences by learning a **reward model** from human comparison data, then using reinforcement learning to optimize the LLM's behavior against that reward signal.

---

## The RLHF Pipeline Overview

```
                       ┌─────────────────────────────────┐
                       │  1. SUPERVISED FINE-TUNING (SFT) │
                       │                                   │
                       │  Pretrained LM + curated demos   │
                       │  → SFT Policy π_SFT              │
                       └────────────────┬────────────────-─┘
                                        │
                                        ▼
                       ┌─────────────────────────────────┐
                       │  2. REWARD MODEL TRAINING        │
                       │                                   │
                       │  π_SFT generates responses y_1,  │
                       │  y_2 for same prompt x           │
                       │  → Human labels: y_w ≻ y_l       │
                       │  → Train RM to predict preference│
                       └────────────────┬─────────────────┘
                                        │
                                        ▼
                       ┌─────────────────────────────────┐
                       │  3. RL FINE-TUNING (PPO)         │
                       │                                   │
                       │  π_θ generates responses         │
                       │  RM scores them                  │
                       │  PPO updates π_θ to maximize     │
                       │  reward - β·KL(π_θ || π_SFT)    │
                       └─────────────────────────────────┘
```

---

## Stage 1: Supervised Fine-Tuning

The RLHF pipeline begins with SFT on a curated **demonstration dataset** — human-written responses to a diverse set of prompts. The purpose of this stage is twofold:

1. **Format alignment**: The base model learns to respond to prompts rather than continue text completions
2. **Reference policy creation**: The SFT checkpoint $\pi_\text{SFT}$ serves as the reference policy for the KL penalty in stage 3

In InstructGPT, labelers were contractors with specific hiring criteria (high agreement on sensitive content guidelines, strong English comprehension). They wrote responses from scratch and also ranked model outputs. The demonstration dataset was ~13K examples.

---

## Stage 2: Reward Model Training

### Preference Data Collection

The SFT model generates multiple responses ($y_1$, $y_2$, ...) for the same prompt $x$. Human labelers compare these and produce a ranking or pairwise preference: $y_w \succ y_l$ (preferred over rejected).

Labeler instructions specify what "good" means along multiple dimensions:
- **Helpfulness**: Does the response actually address the request?
- **Harmlessness**: Does it avoid producing dangerous, harmful, or dishonest content?
- **Honesty**: Does the model express appropriate uncertainty?

Collecting these labels at scale is the most expensive part of RLHF. InstructGPT collected ~33K comparison pairs; Anthropic's Constitutional AI collected ~300K pairs.

### The Bradley-Terry Reward Model

The reward model $r_\phi(x, y)$ maps a prompt-response pair to a scalar reward. It is trained to predict human preferences using the **Bradley-Terry model** of paired comparisons.

Given a preference pair $(y_w, y_l)$ for prompt $x$, the probability that $y_w$ is preferred over $y_l$ is:

$$P(y_w \succ y_l \mid x) = \sigma\bigl(r_\phi(x, y_w) - r_\phi(x, y_l)\bigr)$$

where $\sigma$ is the sigmoid function. This is simply saying: the higher the reward difference, the more confidently we expect humans to prefer $y_w$.

The training objective is binary cross-entropy:

$$\mathcal{L}_\text{RM}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim D}\Bigl[\log \sigma\bigl(r_\phi(x, y_w) - r_\phi(x, y_l)\bigr)\Bigr]$$

**Architecture**: The reward model is typically initialized from the SFT checkpoint, with the LM head replaced by a linear layer projecting to a scalar value. The entire sequence is fed through the transformer, and the reward is read from the last token's representation:

```python
# Pseudocode for reward model
class RewardModel(nn.Module):
    def __init__(self, pretrained_lm):
        self.transformer = pretrained_lm.transformer  # frozen or partially frozen
        self.reward_head = nn.Linear(hidden_dim, 1)  # replace LM head

    def forward(self, input_ids, attention_mask):
        hidden = self.transformer(input_ids, attention_mask)
        # Use last non-padded token's representation
        last_token_hidden = hidden[:, -1, :]
        return self.reward_head(last_token_hidden).squeeze(-1)  # scalar per sequence
```

**Practical considerations**:
- The RM should be trained to be **well-calibrated**, not just accurate: overconfident rewards mislead the RL optimizer
- Initialize from SFT (not the base model) to ensure the RM understands instruction-following contexts
- Regularize with dropout and early stopping; RM overfitting is a primary source of reward hacking downstream

---

## Stage 3: RL Fine-tuning with PPO

With a reward model in hand, we want to fine-tune the LLM policy $\pi_\theta$ to maximize reward. The naive approach — maximize $r_\phi(x, y)$ directly — quickly fails due to **reward hacking**: the policy finds degenerate outputs (e.g., repetitive text, certain phrasing patterns) that the reward model scores highly but humans would not prefer.

The solution is a **KL-penalized reward objective**:

$$R(x, y) = r_\phi(x, y) - \beta \cdot \text{KL}\bigl(\pi_\theta(\cdot \mid x) \,\|\, \pi_\text{ref}(\cdot \mid x)\bigr)$$

where $\pi_\text{ref}$ is the frozen SFT model and $\beta$ controls how strongly we penalize divergence from it.

In practice, the KL term is computed token-by-token:

$$\text{KL}\bigl(\pi_\theta \,\|\, \pi_\text{ref}\bigr) = \sum_{t=1}^{T} \log \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_\text{ref}(y_t \mid x, y_{<t})}$$

This is added as a per-token penalty to the reward signal before backpropagation through PPO.

### PPO for LLMs

PPO (Proximal Policy Optimization, Schulman et al., 2017) is the standard RL algorithm for RLHF due to its stability. Adapted to the LLM setting:

**Policy**: The LLM $\pi_\theta$ maps prompts to response probability distributions (token by token).

**Trajectory**: A prompt $x$ sampled from a distribution, followed by the LLM generating a response $y$ token by token. The entire sequence is one "episode" in the RL sense.

**Reward**: Sparse — the KL-penalized reward is assigned only at the end of generation ($t = T$). Per-token rewards are zero except for the KL term.

**Value function**: PPO requires a value function $V_\psi(x, y_{<t})$ to estimate the expected future reward from each state. This is typically a separate model (or a separate head on $\pi_\theta$) trained alongside the policy.

**PPO clip objective**: At each optimization step, the policy update is clipped to prevent too-large policy changes:

$$\mathcal{L}_\text{PPO}(\theta) = \mathbb{E}\left[\min\!\left(\rho_t A_t,\ \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

where $\rho_t = \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_\text{old}(y_t \mid x, y_{<t})}$ is the probability ratio and $A_t$ is the advantage estimate (estimated using the value function via GAE).

The clipping prevents any single gradient step from moving the policy too far, which stabilizes training — crucial because LLM-as-policy has an enormous action space (vocabulary size) and small gradient noise can cause large behavioral changes.

**In practice**, a PPO training step for LLMs involves:
1. Sample a batch of prompts
2. Generate responses with $\pi_\theta$ (stored with log-probabilities)
3. Score responses with $r_\phi$ and compute KL against $\pi_\text{ref}$
4. Compute advantages using the value function
5. Run $k$ gradient steps on the PPO clip objective
6. Update $\pi_\theta$ and $V_\psi$

---

## Reward Hacking and Overoptimization

As RL training progresses, the policy increasingly optimizes against the reward model rather than true human preferences. This is Goodhart's Law:

> *"When a measure becomes a target, it ceases to be a good measure."*

The reward model $r_\phi$ is an imperfect proxy for human preferences, trained on finite data. As $\pi_\theta$ moves further from $\pi_\text{ref}$, it explores regions of response space that were not represented in the reward model's training distribution. In these out-of-distribution regions, $r_\phi$ produces inaccurate predictions that the policy exploits.

**Observed hacking behaviors**:
- Repetitive high-confidence assertions ("This is absolutely correct. I am completely certain.")
- Excessive agreement with the user (sycophancy)
- Padding responses with filler phrases that pattern-match to "thorough" responses the reward model liked
- Switching to a different language mid-response (confusing the RM)
- Excessively long or excessively short responses if the RM has a length bias

**The empirical finding** (Gao et al., 2022): When measuring gold-standard human preference (the "true" reward) as a function of KL distance from $\pi_\text{ref}$:
- Initially: both RM score and gold reward increase together
- Past an optimal KL: RM score continues increasing while gold reward plateaus then **decreases**
- This "overoptimization" region is where reward hacking occurs

**Mitigations**:
- **KL penalty** $\beta$: The primary defense. Higher $\beta$ = more conservative, less hacking but also less improvement
- **Early stopping**: Stop RL training before the policy drifts too far
- **Reward model ensembles**: Average rewards from multiple independently trained RMs; harder to hack simultaneously
- **Conservative RL**: Penalize high-variance policies, not just high-KL ones
- **Iterative RLHF**: Retrain the reward model on comparisons between current policy outputs periodically

---

## Practical Challenges

**Memory**: PPO for LLMs requires simultaneously loading: the policy $\pi_\theta$ (trainable), the reference policy $\pi_\text{ref}$ (frozen copy), the reward model $r_\phi$, and the value model $V_\psi$. For a 7B base model, this is 4 × 14GB ≈ 56GB minimum, before optimizer states. DeepSpeed ZeRO or model sharding is essential.

**Sample efficiency**: PPO is notoriously sample-inefficient. Each training step requires generating responses (slow autoregressive sampling), scoring them, and then performing a relatively small number of gradient updates. Generating text is the bottleneck; separating generation onto inference GPUs while training runs on separate accelerators is a common production architecture.

**KL coefficient tuning**: $\beta$ is the most important hyperparameter. Too small: reward hacking. Too large: no improvement over SFT. Common practice: start with $\beta = 0.1$–$0.3$ and adjust based on monitoring KL divergence and downstream benchmarks during training.

**Reward model generalization**: The RM is trained on comparison data from $\pi_\text{SFT}$ but used to score outputs from $\pi_\theta$, which drifts as RL training progresses. As the distribution gap grows, RM accuracy degrades. Iterative reward model retraining is expensive but improves robustness.

**Instability**: PPO training can collapse — the policy converges to degenerate behaviors (outputting only one response, or very short outputs that are easy to generate). Monitoring generation statistics (average length, vocabulary diversity, reward distribution) throughout training is essential.

---

## RLAIF: RL from AI Feedback

**RLAIF** (RL from AI Feedback, Bai et al., 2022; Lee et al., 2023) replaces human labelers with a separate AI model (the "preference model") to generate preference labels.

**Motivation**: Human annotation is expensive ($\sim$\$1–10 per comparison), slow (hours to days for a batch), and introduces human judgment variability. An AI labeler can generate millions of comparisons in hours at near-zero marginal cost.

**The pipeline**:
1. Sample pairs of responses $(y_1, y_2)$ from the SFT model
2. Present the pair to a strong AI judge (GPT-4, Claude, or a powerful open-source model) with a detailed rubric
3. The judge outputs a preference: $y_1 \succ y_2$ or vice versa, sometimes with a confidence score
4. Train a reward model on these AI-generated labels
5. Apply PPO exactly as in standard RLHF

**Key finding** (Lee et al., 2023): RLAIF with Claude 2 as the preference annotator matched RLHF with human annotators on a summarization task (helpfulness ratings). AI feedback generalizes better at scale — humans show fatigue and inconsistency on large annotation batches, while AI judges are consistent.

**Limitations**:
- AI judges have their own biases (verbosity bias, position bias, favoring their own outputs)
- Circular reasoning: if the AI judge is similar to the policy, it may rate its own outputs highly regardless of quality
- Does not capture human value nuances that the AI judge itself lacks

---

## Constitutional AI (Anthropic)

Constitutional AI (Bai et al., 2022) extends RLAIF with a structured approach to generating preference data based on explicit **principles** (the "constitution").

**The two-phase CAI pipeline**:

### Phase 1: Supervised Learning from AI Feedback (SL-CAF)

1. **Red-team prompting**: Present the model with prompts designed to elicit harmful outputs
2. **Critique**: Prompt the model to critique its own response against specific constitutional principles:
   > "Identify specific ways in which the assistant's last response is harmful, unethical, racist, sexist, toxic, dangerous, or illegal."
3. **Revision**: Prompt the model to revise its response to be less harmful while maintaining helpfulness:
   > "Please rewrite the assistant response to remove all harmful, unethical, racist, sexist, toxic, dangerous, or illegal content."
4. Fine-tune the model on the (original prompt, revised response) pairs

The model learns to be less harmful *without human labelers ever seeing the harmful content*.

### Phase 2: RL from AI Feedback (RLAIF)

1. Generate pairs of responses (one from the original model, one from the revised model)
2. Use the model itself (or a strong judge) to rate which is more aligned with each constitutional principle
3. Train a **preference model** (reward model) on these AI-generated labels
4. Apply PPO with the preference model as the reward signal

**The key insight**: By making the principles explicit (the "constitution"), Anthropic created a transparent and auditable alignment process. The constitutional principles serve as a formalizable proxy for human values, enabling rapid iteration and evaluation.

**Example constitutional principles**:
- "Choose the response that is least likely to contain harmful, unethical, or illegal content."
- "Choose the response that is most helpful, most concise, and most directly answers the question."
- "Choose the response that is most truthful and honest."

This approach powers Claude's alignment training and has been shown to produce models that are more consistently aligned across diverse prompts than standard RLHF with human preferences alone.

---

## InstructGPT Results: What RLHF Actually Achieves

InstructGPT (Ouyang et al., 2022) was the first public demonstration of RLHF for aligning a large LLM. Key results:

**Human preference study**: Human labelers preferred InstructGPT 1.3B outputs over GPT-3 175B outputs **85% of the time**, despite the 100x parameter difference. Alignment matters more than raw scale.

**The alignment tax**: InstructGPT 1.3B (SFT+RLHF) was preferred over InstructGPT 175B SFT alone on helpfulness — but the 175B SFT+RLHF model scored slightly worse than 175B SFT on certain knowledge benchmarks (MMLU). Fine-tuning for alignment trades a small capability penalty for a large preference improvement.

**GPT-3 with prompting vs InstructGPT with fine-tuning**: InstructGPT clearly outperformed prompted GPT-3 across all human preference dimensions. The performance gap was most pronounced on sensitive topics (safety, factual accuracy, appropriate refusal).

**Toxicity and bias**: InstructGPT generated fewer toxic outputs and showed reduced demographic biases compared to GPT-3, without explicitly optimizing for these properties — the RLHF training signal for "helpfulness and harmlessness" generalized to reduce toxicity.

**The fundamental lesson**: A pretrained model's raw capabilities are largely preserved through RLHF; what changes is *how it applies those capabilities*. RLHF does not teach the model new knowledge — it teaches it to use existing knowledge in ways humans prefer.

---

## References

- Ouyang, L. et al. (2022). *Training language models to follow instructions with human feedback* (InstructGPT). NeurIPS.
- Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- Bradley, R.A. and Terry, M.E. (1952). *Rank Analysis of Incomplete Block Designs*. Biometrika.
- Bai, Y. et al. (2022). *Constitutional AI: Harmlessness from AI Feedback*. Anthropic, arXiv:2212.08073.
- Lee, H. et al. (2023). *RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback*. arXiv:2309.00267.
- Gao, L. et al. (2022). *Scaling Laws for Reward Model Overoptimization*. ICML 2023.
- Stiennon, N. et al. (2020). *Learning to summarize from human feedback*. NeurIPS.
- Ziegler, D.M. et al. (2019). *Fine-Tuning Language Models from Human Preferences*. arXiv:1909.08593.

---

*← [Parameter-Efficient Fine-Tuning](./parameter-efficient-fine-tuning.md) | → [Direct Preference Optimization](./direct-preference-optimization.md)*

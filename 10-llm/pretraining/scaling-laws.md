# Scaling Laws

## Table of Contents
1. [Why Scaling Laws Matter](#why-scaling-laws-matter)
2. [Kaplan et al. 2020: The Original OpenAI Scaling Laws](#kaplan-et-al-2020-the-original-openai-scaling-laws)
3. [Chinchilla: Compute-Optimal Scaling](#chinchilla-compute-optimal-scaling)
4. [Emergent Abilities](#emergent-abilities)
5. [Data Scaling and Diminishing Returns](#data-scaling-and-diminishing-returns)
6. [Architecture Scaling: Depth vs Width](#architecture-scaling-depth-vs-width)
7. [Test-Time Compute Scaling](#test-time-compute-scaling)
8. [Practical Implications: Allocating a Compute Budget](#practical-implications-allocating-a-compute-budget)
9. [Limitations of Scaling Laws](#limitations-of-scaling-laws)
10. [References](#references)

---

## Why Scaling Laws Matter

Scaling laws are quantitative relationships between model size, data volume, compute budget, and loss. They matter because they transform LLM training from an empirical art into a partially predictable engineering discipline. Given a fixed budget (measured in FLOPs), scaling laws tell you how to divide that budget between model parameters and training tokens to achieve the lowest possible loss — without running the expensive experiment at full scale.

The discovery that loss follows smooth **power laws** across many orders of magnitude (from 1M to 100B parameters, from 1B to 1T tokens) is itself a remarkable empirical finding: it implies that the phenomena underlying language modeling are not qualitatively different at different scales, and that extrapolation from small runs is valid.

---

## Kaplan et al. 2020: The Original OpenAI Scaling Laws

### Setup

Kaplan et al. (2020) trained language models (decoder-only Transformers with CLM) spanning 7 orders of magnitude in compute, 5 in parameters, and 4 in dataset size, measuring validation loss on a held-out text corpus.

Three fundamental quantities:
- $N$ — number of **non-embedding** parameters
- $D$ — number of training **tokens** (dataset size)
- $C$ — total training **compute** (FLOPs), approximately $C \approx 6ND$ for a single pass

### Power Law Findings

When other variables are unconstrained (i.e., infinite data and compute), loss decreases as a power law in each variable independently:

$$L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076$$

$$L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095$$

$$L(C) \approx \left(\frac{C_c}{C}\right)^{\alpha_C}, \quad \alpha_C \approx 0.057$$

where $N_c$, $D_c$, $C_c$ are constants fit from data. The exponents are small: doubling parameters only reduces loss by $2^{0.076} - 1 \approx 5\%$ in relative terms. This means large absolute improvements require enormous multiplicative increases in scale.

### The Joint Loss Equation

When both $N$ and $D$ are finite, the approximate combined loss is:

$$L(N, D) \approx \left[\left(\frac{N_c}{N}\right)^{\frac{\alpha_N}{\alpha_D}} + \frac{D_c}{D}\right]^{\alpha_D}$$

This has two important regimes:
- **Parameter-limited** ($N$ small, $D$ large): loss is dominated by model capacity
- **Data-limited** ($D$ small, $N$ large): loss is dominated by token count; model is undertrained

### The Kaplan Scaling Recommendation

Given a fixed compute budget $C$, the Kaplan paper argued that optimal compute allocation strongly favors **larger models trained for fewer tokens**. Specifically:

$$N^* \propto C^{0.73}, \quad D^* \propto C^{0.27}$$

This means: as compute grows, scale the model much faster than the data. For example, going from 1× to 10× compute, increase parameters by $10^{0.73} \approx 5.4\times$ but only increase tokens by $10^{0.27} \approx 1.9\times$.

**This recommendation turned out to be wrong.** GPT-3 (175B parameters, 300B tokens) was trained following roughly this logic and was significantly undertrained relative to what Chinchilla later showed.

---

## Chinchilla: Compute-Optimal Scaling

### The Hoffmann et al. 2022 Correction

Hoffmann et al. (2022) from DeepMind re-ran the scaling law analysis with three methodologies (fixing $N$ and sweeping $D$, fixing $C$ and sweeping $N$, fitting a parametric loss model), finding that Kaplan's recommendation was off because their experiments had not adequately explored the data-scaling axis.

### The Chinchilla Scaling Law

The Hoffmann et al. loss model is:

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

where $E \approx 1.69$ (irreducible entropy), $\alpha \approx 0.34$, $\beta \approx 0.28$, and $A$, $B$ are fitted constants.

For a fixed compute budget $C \approx 6ND$, minimizing $L$ subject to $6ND = C$ gives:

$$N^* \propto C^{0.5}, \quad D^* \propto C^{0.5}$$

The compute-optimal model scales **parameters and tokens equally**. Each doubling of compute should double both parameters and tokens.

More concretely, the Chinchilla rule of thumb:

$$D^* \approx 20 \times N^*$$

Train each parameter on approximately **20 tokens** for compute-optimal performance.

### The Chinchilla Empirical Result

| Model | Parameters | Training Tokens | Compute | Test Loss |
|-------|-----------|----------------|---------|-----------|
| Gopher | 280B | 300B | ~84 × 10²³ FLOPs | Higher |
| Chinchilla | 70B | 1.4T | ~84 × 10²³ FLOPs | Lower |

Same compute budget. Chinchilla (70B × 1.4T) beats Gopher (280B × 300B) on virtually every benchmark. The 280B model was massively **undertrained**; it had four times the capacity but had only seen each token ~1 time.

### Implications for the Field

The Chinchilla result restructured LLM development:

1. **LLaMA** (Touvron et al., 2023): trained 7B, 13B, 33B, 65B models on up to 1.4T tokens, explicitly following the Chinchilla philosophy. LLaMA 7B outperforms GPT-3 175B on many benchmarks because it is properly trained despite being 25× smaller.

2. **Inference cost matters**: A smaller, well-trained model has lower inference cost per query. For a system running millions of queries per day, the inference cost dominates training cost by orders of magnitude. Chinchilla-optimal training produces models that are *both* better and cheaper to serve.

3. **"Training beyond Chinchilla"**: For models intended for wide deployment (many inference calls), it can be economically rational to train *beyond* the compute-optimal point — using more tokens on a fixed-size model — because the additional training improves model quality at no inference cost increase. LLaMA 3 (8B on 15T tokens) exemplifies this.

---

## Emergent Abilities

### Definition

Emergent abilities (Wei et al., 2022) are capabilities that are absent (near-random performance) in smaller models and appear "suddenly" as model scale crosses some threshold.

Examples from BIG-Bench:
- **3-digit arithmetic**: Below ~50B parameters, models perform at chance. Above ~100B, accuracy jumps significantly.
- **Chain-of-thought reasoning**: Only models above ~50–100B parameters benefit from CoT prompting.
- **Multi-step word problems**: Sudden accuracy jumps at scale.
- **Multi-language translation (zero-shot)**: Near-zero below threshold, then functional above it.

### The Phase Transition Metaphor

The apparent "sharpness" of emergence looks like a phase transition: smooth input (compute/parameters), discontinuous output (capability). This has prompted speculation about qualitative changes in model behavior.

### The Metric Artifact Argument

Schaeffer et al. (2023) argued that many apparent emergence phenomena are **artifacts of nonlinear metrics**. If a task requires getting all $k$ sub-steps correct and each step improves smoothly with scale, then:

$$P(\text{all correct}) = \prod_{i=1}^{k} P(\text{step } i \text{ correct})$$

This product of gradually improving probabilities produces a sharp-looking curve even though each component is smooth. When measuring with a continuous metric (e.g., token-level accuracy), the sharp jump disappears and smooth improvement is visible.

**Both views have evidence.** Some capabilities do appear gradually when measured with more sensitive metrics. Others — particularly multi-hop reasoning that requires combining multiple sub-capabilities that each threshold at similar scales — may show genuine non-smoothness. The debate is not fully resolved.

### Why It Matters for Engineering

Whether or not emergence is "real," the practical implication is clear: capabilities you care about may not be visible in smaller training runs. You cannot always extrapolate from 7B to 70B performance on multi-step tasks. This makes capability evaluation fundamentally harder than loss evaluation.

---

## Data Scaling and Diminishing Returns

### Quality vs. Quantity

Scaling laws assume a fixed data distribution. In practice, data quality is a major variable. A small, high-quality dataset (e.g., curated academic papers, verified code) produces better per-token learning signal than a large noisy one (raw Common Crawl). Concretely:

- MiniLM trained on curated data can outperform larger models trained on raw web text on specific domains
- The LLaMA 3 paper (Meta, 2024) reports aggressive quality filtering that reduced eligible tokens from raw Common Crawl by ~10× but improved downstream performance significantly

### Epoch Scaling and Repeated Data

If the available token budget $D_{\text{available}} < D^*$ (the compute-optimal token count), should you repeat data? Muennighoff et al. (2023) found that repeating data up to 4× incurs minimal performance degradation relative to using unique data. Beyond 4× repeats, performance degrades noticeably. The Chinchilla formula $D^* = 20N^*$ is implicitly a bound on training with unique tokens; in data-constrained settings, modest repetition is acceptable.

### Code Data as a Special Case

Code training data is disproportionately valuable. Models trained with substantial code data (GitHub repositories) consistently show improved reasoning on non-code tasks: symbolic manipulation, structured problem-solving, step-by-step logic. The hypothesis is that code's explicit structure (control flow, variable binding, formal specifications) teaches the model a precision of reasoning not easily learned from prose. This is one reason why all frontier LLMs include code data even for general-purpose applications. See [data-curation.md](./data-curation.md) for details.

---

## Architecture Scaling: Depth vs Width

### What Scales Better?

For a fixed parameter budget $N$, how should you allocate parameters between depth ($L$ layers) and width ($d$ model dimension)?

Empirically (Kaplan et al., 2020; later confirmed by others):
- Loss is relatively insensitive to aspect ratio (depth/width) over a broad range
- Very deep, narrow models and very wide, shallow models both perform slightly worse than balanced architectures
- The "sweet spot" for most decoder-only models is roughly $d \approx 128\sqrt{L}$ (width scales with square root of depth)

For reference, GPT-3 (175B) uses $L=96$ layers, $d=12288$; LLaMA 3 70B uses $L=80$ layers, $d=8192$.

### Compute Per Layer

Each Transformer layer at hidden dimension $d$ and sequence length $n$ costs approximately $O(nd^2 + n^2 d)$ FLOPs per forward pass. For typical training sequence lengths ($n \leq 4096$), the $nd^2$ (FFN and attention projection) term dominates. Width thus scales compute quadratically while depth scales it linearly — which pushes optimal architectures toward more depth relative to naive intuition.

---

## Test-Time Compute Scaling

### Beyond Training: Scaling Inference

Recent work (Snell et al., 2024; OpenAI o1 technical report, 2024) has established a **second scaling axis**: inference (test-time) compute. Rather than using more training compute to get a better model, you can use more inference compute to get better answers from a fixed model.

### Best-of-N Sampling

The simplest test-time scaling: generate $N$ independent solutions, score them with a verifier (a reward model or ground-truth checker), and return the best:

$$\hat{y} = \arg\max_{y_i} r(x, y_i), \quad y_i \sim p_\theta(\cdot \mid x)$$

Accuracy on math benchmarks (e.g., MATH) scales as a power law in $N$:

$$P(\text{correct}) \approx 1 - (1 - p)^N \approx 1 - \exp(-pN)$$

where $p$ is per-sample accuracy. This is sublinear in the compute budget $N$ but can dramatically improve performance on hard reasoning tasks.

### Chain-of-Thought and Compute Scaling

Longer chains of thought allocate more tokens (more inference compute) to a problem. Wei et al. (2022) showed CoT prompting enables reasoning that zero-shot prompting fails at entirely. Critically, the quality of CoT scales with:
1. **Model size** (smaller models produce incoherent chains)
2. **Chain length** (more tokens = more steps = better for complex problems)

### Tree Search and Process Reward Models

More structured test-time scaling uses tree search (MCTS, beam search over reasoning steps) guided by a **process reward model (PRM)** that scores intermediate reasoning steps. OpenAI o1 and its successors use this paradigm: the model generates a "chain of thought" internally (reasoning tokens), with test-time compute allocated dynamically based on problem difficulty.

The scaling law for this regime (Snell et al., 2024):

$$L(\text{C}_{\text{test}}) \propto C_{\text{test}}^{-\gamma}$$

for some $\gamma > 0$, analogous to training-time scaling laws. This suggests **a smooth trade between training compute and test-time compute** for a fixed capability level.

---

## Practical Implications: Allocating a Compute Budget

Given a fixed compute budget $C$ (in FLOPs), the following decision procedure follows from scaling law analysis:

**Step 1: Decide training vs inference allocation.**
If you will run many inference queries (production system), consider training a smaller, better-trained model (even beyond Chinchilla-optimal) to minimize per-query inference cost.

**Step 2: Apply Chinchilla optimal allocation.**
$$N^* \approx \sqrt{\frac{C \cdot A}{6B}} \cdot \left(\frac{\alpha}{\beta}\right)^{1/2}, \quad D^* \approx \frac{C}{6N^*}$$

Practically: for most budgets, target $D \approx 20N$ to $D \approx 100N$ depending on how much you want to trade inference cost for marginal quality gains.

**Step 3: Run small-scale ablations.**
Run models at 3–5 scales that are 100–1000× smaller than the target, fitting power law curves to extrapolate loss. Check whether the loss curve follows the predicted trajectory and whether the architecture/objective choices hold at scale.

**Step 4: Monitor for instabilities.**
Large training runs encounter instabilities (loss spikes, numerical issues) that small runs do not. Maintain checkpoints and budget for ~10–15% of training compute as "retry budget" for failed runs.

**Step 5: Reserve capability evaluation for scale.**
Emergent capabilities cannot be reliably extrapolated from small ablations. Plan a capability evaluation checkpoint at ~10% and ~50% of the full run.

---

## Limitations of Scaling Laws

Scaling laws are powerful but not universal. The following assumptions are implicit and can be violated:

1. **Fixed data distribution.** Scaling laws are fit on a specific dataset. If you change the data mixture (add code, multilingual data, synthetic data), the constants change and extrapolations break.

2. **Fixed architecture.** Power law exponents change with attention type, positional encoding, activation function, and normalization. Scaling law fits from dense Transformers do not directly apply to MoE models or SSMs.

3. **Fixed optimizer.** The Kaplan and Chinchilla results use Adam with standard hyperparameters. Different learning rate schedules, batch sizes, or optimizers shift the curves.

4. **Loss vs task performance.** Scaling laws predict *loss*, not benchmark accuracy. The relationship between loss and downstream task performance is nonlinear and task-dependent. A 10% reduction in perplexity does not map to a predictable improvement on GSM8K.

5. **Quality of training data.** Scaling laws assume consistent data quality throughout training. In practice, data pipelines change; contamination is discovered; quality degrades as the internet is exhausted.

6. **Post-training interventions.** RLHF, instruction tuning, and constitutional AI can dramatically change model behavior independent of pretraining loss. Scaling laws for aligned models remain less well-characterized.

---

## References

- Kaplan, J., et al. (2020). *Scaling laws for neural language models.* [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
- Hoffmann, J., et al. (2022). *Training compute-optimal large language models (Chinchilla).* NeurIPS. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- Wei, J., et al. (2022). *Emergent abilities of large language models.* TMLR. [arXiv:2206.07682](https://arxiv.org/abs/2206.07682)
- Schaeffer, R., et al. (2023). *Are emergent abilities of large language models a mirage?* NeurIPS. [arXiv:2304.15004](https://arxiv.org/abs/2304.15004)
- Muennighoff, N., et al. (2023). *Scaling data-constrained language models.* NeurIPS. [arXiv:2305.16264](https://arxiv.org/abs/2305.16264)
- Snell, C., et al. (2024). *Scaling LLM test-time compute optimally can be more effective than scaling model parameters.* [arXiv:2408.03314](https://arxiv.org/abs/2408.03314)
- Touvron, H., et al. (2023). *LLaMA: Open and efficient foundation language models.* [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)
- Meta AI (2024). *The LLaMA 3 herd of models.* [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)
- OpenAI (2024). *OpenAI o1 system card.* [openai.com](https://openai.com/research/openai-o1-system-card)

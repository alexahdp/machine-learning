# Training Stability

## Table of Contents
1. [Why Stability is a First-Class Engineering Problem](#why-stability-is-a-first-class-engineering-problem)
2. [Loss Spikes](#loss-spikes)
3. [Gradient Norm Monitoring](#gradient-norm-monitoring)
4. [Mixed Precision Training](#mixed-precision-training)
5. [Learning Rate Schedule](#learning-rate-schedule)
6. [Gradient Clipping](#gradient-clipping)
7. [Weight Initialization and μP](#weight-initialization-and-μp)
8. [Checkpointing Strategy](#checkpointing-strategy)
9. [Common Failure Modes](#common-failure-modes)
10. [Training Monitoring](#training-monitoring)
11. [LARS and LAMB: Large-Batch Optimizers](#lars-and-lamb-large-batch-optimizers)
12. [References](#references)

---

## Why Stability is a First-Class Engineering Problem

Training a 70B model for six weeks on 2,000 GPUs costs millions of dollars. A single unrecovered failure — a loss spike that does not recover, a NaN cascade, a hardware failure at the wrong checkpoint — wastes days of compute. At scale, instabilities that are hypothetically possible at small scale become *guaranteed* in expectation over a long training run:

- Hardware ECC memory errors occur at rate ~$10^{-6}$/hour/GPU; over 2,000 GPUs × 1,000 hours = 2,000,000 GPU-hours, expect ~2,000 uncorrectable errors.
- Loss spike probability per step might be $10^{-5}$; over 500,000 training steps, expect ~5 loss spikes per run.
- Bad data batches (malformed documents, encoding errors) occur at non-zero frequency in any real dataset.

Stability engineering is the set of techniques that make training robust to these events: detecting problems early, recovering gracefully, and structuring the training run so that no single failure is catastrophic.

---

## Loss Spikes

### What They Are

A loss spike is a sudden, large increase in training loss — typically 2–10× the running average — that may or may not self-recover. In a smooth training run, loss decreases roughly monotonically (with small fluctuations). A spike looks like:

```
Loss
│
│      ╭─╮
│     ╭╯  ╰─╮
│    ╭╯      ╰─────────────────────── (recovered)
│   ╭╯
│──╮╯
│  ╰────────────────────────────────
│
└──────────────────────────────────── Steps
```

### Root Causes

**1. Bad data batches.** A document with encoding errors, extreme token lengths, or highly anomalous content can produce an unusually large gradient that transiently destabilizes training. Solution: robust data filtering, gradient clipping, and monitoring loss per-batch to identify the culprit batch.

**2. Learning rate too high.** If the learning rate is at the boundary of stability, small perturbations push it over. The effective learning rate increases when gradient norms are small (relative to the weight magnitudes), which can happen at specific training phases.

**3. Numerical accumulation in residual streams.** In deep Transformers, small numerical errors in residual additions can accumulate over many layers. With FP16 (narrow dynamic range), values can overflow or underflow. This is why BF16 or FP32 is preferred (see below).

**4. Optimizer state divergence.** In Adam, the second moment estimate $v_t$ approaches zero for rarely-updated parameters (embeddings for rare tokens), causing the effective learning rate for those parameters ($\eta / \sqrt{v_t + \epsilon}$) to explode when those parameters are encountered in a batch.

**5. Hardware transients.** Bit flips in GPU memory, NaN propagation from a faulty CUDA operation, or a hardware-induced computation error can inject NaN or Inf into the computation graph.

### Recovery

Most spikes self-recover if gradient clipping is in place and the learning rate is reasonable. For spikes that do not recover:
- Roll back to the last checkpoint before the spike
- Optionally skip the offending data batch (or reshuffle)
- Optionally reduce the learning rate temporarily (LR cooldown after spike)

LLaMA 3 training (Meta, 2024) encountered and recovered from several loss spikes by rolling back 50–100 steps.

---

## Gradient Norm Monitoring

### The Global Gradient Norm

The global gradient norm is the $\ell_2$ norm of all parameter gradients concatenated:

$$\|\nabla\|_2 = \sqrt{\sum_{p \in \text{params}} \sum_{i} \left(\frac{\partial \mathcal{L}}{\partial p_i}\right)^2}$$

This single scalar is the most important per-step monitoring signal in large-scale LLM training.

**What it tells you:**

| Gradient norm behavior | Interpretation |
|------------------------|---------------|
| Smoothly decreasing over training | Healthy — model is converging |
| Stable plateau around a value | Training in steady state |
| Sudden spike (10× or more) | Loss spike event; check data batch |
| Slowly increasing trend | Potential instability building |
| Near-zero for many steps | Possible learning rate issue or NaN |
| NaN or Inf | Catastrophic failure |

### Per-Layer Gradient Norms

Beyond the global norm, tracking gradient norms per layer reveals structural problems:
- **Embedding gradient norm** much larger than FFN: embedding learning is dominating; may need lower LR for embeddings
- **First layer gradient near zero, last layer large**: vanishing gradients (check residual connections and initialization)
- **Specific layer suddenly spiking**: that layer's data path has a numerical issue

In PyTorch, per-layer gradient norms:

```python
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.data.norm(2).item()
        wandb.log({f"grad_norm/{name}": grad_norm})
```

Logging all per-layer norms is expensive (thousands of tensors); in practice, log them for a representative sample of layers (first, middle, last) plus the global norm for every step.

---

## Mixed Precision Training

### The Problem with FP32

Training in 32-bit floating point is safe (wide dynamic range, high precision) but expensive: parameter storage and gradient computation at FP32 doubles memory requirements vs FP16/BF16. For 70B parameters:

- FP32 parameters: 280GB
- FP32 gradients + optimizer states: 840GB
- Total FP32: ~1.1TB

This is prohibitive for modern training runs. Mixed precision reduces the footprint while managing the precision risks.

### FP16 vs BF16

Both use 16 bits total. The difference is in bit allocation:

```
FP32:  1 sign | 8 exponent | 23 mantissa  bits   range: ±3.4×10³⁸
FP16:  1 sign | 5 exponent | 10 mantissa  bits   range: ±65,504
BF16:  1 sign | 8 exponent | 7  mantissa  bits   range: ±3.4×10³⁸
```

**FP16 problems**:
- Overflow at values > 65,504. Gradient norms, logits, and loss values routinely exceed this during LLM training → Inf.
- Underflow: small gradient values (< ~$6 \times 10^{-8}$) flush to zero → gradients lost.
- Requires **dynamic loss scaling** to keep gradients in representable range (see below).

**BF16 advantages**:
- Same exponent range as FP32 → no overflow for any value that FP32 can represent
- Lower mantissa precision (7 vs 23 bits) → ~0.78% relative error per operation
- No loss scaling required
- A100 and H100 have native BF16 tensor core support

**BF16 is now the standard** for LLM pretraining. NVIDIA Ampere (A100) and Hopper (H100) architectures support BF16 matrix multiplication at the same throughput as FP16.

### Loss Scaling for FP16

If using FP16 (required for older GPUs without BF16 support), dynamic loss scaling prevents underflow:

1. Multiply the loss by a scaling factor $S$ before backprop: $\tilde{\mathcal{L}} = S \cdot \mathcal{L}$
2. Gradients are computed at scale $S$ (no underflow for moderate $S$)
3. Before the optimizer step, divide gradients by $S$: $g \leftarrow g / S$
4. **If any gradient is Inf/NaN** (overflow): skip the optimizer step, halve $S$
5. **If no Inf/NaN for $K$ consecutive steps**: double $S$

PyTorch's `torch.cuda.amp.GradScaler` implements this automatically. Typical initial $S = 2^{16} = 65536$.

### Master Weights in FP32

In mixed precision training, the standard approach is:

- **Compute in BF16/FP16**: forward pass, backward pass, gradient communication
- **Store parameters in FP32**: a "master copy" maintained by the optimizer
- **Optimizer step in FP32**: updates applied to FP32 master weights
- **Cast to BF16** for the next forward pass

This is crucial because optimizer updates are small ($\eta \cdot g \approx 10^{-4}$ to $10^{-6}$), smaller than the precision of BF16 relative to typical weight magnitudes. If you apply updates directly to BF16 weights, small updates round to zero — the weights stop learning.

In ZeRO-3 and FSDP with BF16, the optimizer states (FP32 master weights + Adam moments) are stored as sharded FP32 tensors, while the compute parameters are BF16.

---

## Learning Rate Schedule

### Warmup

Starting training with a large learning rate causes instability: the random initial weights produce large, noisy gradients, and a large learning rate amplifies these into large parameter updates that push the model into bad regions of loss landscape from which recovery is slow.

**Linear warmup**: increase the learning rate linearly from 0 (or a small value) to the peak learning rate over $W$ steps:

$$\eta_t = \eta_{\max} \cdot \frac{t}{W}, \quad t \leq W$$

Typical warmup duration: $W \approx 2000$–$4000$ steps for frontier-scale models. The intuition: small early updates let the optimizer's second-moment estimates ($v_t$ in Adam) "warm up" — without warmup, $v_t$ starts near zero, causing extremely large effective learning rates ($\eta/\sqrt{v_t + \epsilon}$) in the first few steps.

### Cosine Decay

After warmup, the learning rate follows a cosine annealing schedule:

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{\pi(t - W)}{T - W}\right)\right), \quad W < t \leq T$$

where $T$ is the total training steps and $\eta_{\min}$ is a final LR (typically $0.1\eta_{\max}$).

Cosine decay:
- Starts with a relatively large LR for fast progress in the early-middle training phase
- Gradually slows as training approaches the end, allowing fine-grained optimization near the minimum
- The "U-shape" of the cosine provides a natural curriculum from exploration to exploitation

Alternative: **linear decay**, **constant with final decay** (train at max LR until 90% of budget, then linearly decay), or **WSD schedule** (warmup–stable–decay): the WSD schedule (Hu et al., 2024) maintains constant LR during the bulk of training and applies a steep decay only in the final 10% of steps. This facilitates training with continuous data, since you can extend training by continuing the stable phase without redoing the decay.

### Practical Learning Rate Values

| Model scale | Typical $\eta_{\max}$ |
|-------------|----------------------|
| 1B | 3×10⁻⁴ |
| 7B | 2×10⁻⁴ – 3×10⁻⁴ |
| 13B | 1.5×10⁻⁴ – 2×10⁻⁴ |
| 70B | 1×10⁻⁴ – 1.5×10⁻⁴ |
| 405B | 5×10⁻⁵ – 1×10⁻⁴ |

Learning rate generally scales down with model size because larger models have larger effective gradient norms per step.

---

## Gradient Clipping

### Max Norm Clipping

Before applying the optimizer update, the global gradient norm is checked. If it exceeds a threshold $\tau$, all gradients are scaled down proportionally:

$$g_t \leftarrow \begin{cases} g_t & \text{if } \|g_t\|_2 \leq \tau \\ \tau \cdot \frac{g_t}{\|g_t\|_2} & \text{if } \|g_t\|_2 > \tau \end{cases}$$

This operation **preserves the gradient direction** (unlike per-element clipping) while bounding the update magnitude.

Typical $\tau = 1.0$ for Transformer LLM training. This is nearly universally applied across all large-scale training runs.

### Why Clipping Works

Gradient clipping addresses the specific pathology of **gradient explosion**: a few extremely large gradients from anomalous batches or numerical issues that would otherwise cause large weight updates and push the model to a bad region. By capping the maximum update size, clipping ensures that no single step can do serious damage.

### Interaction with Adaptive Optimizers

Adam adapts per-parameter learning rates using second moment estimates. Gradient clipping applies *before* Adam's adaptation — it clips the raw gradients, not the effective updates. The effective update magnitude is:

$$\|\Delta \theta\|_2 \approx \eta \cdot \left\|\frac{g_t}{\sqrt{v_t + \epsilon}}\right\|_2$$

This can still be large if $v_t$ is small (newly activated parameters, rare tokens). Clipping does not directly bound the effective update — it only bounds the input gradient. This is why warmup (to grow $v_t$) is also necessary.

### Clip Norm Monitoring

The fraction of steps where clipping fires is diagnostic:
- **<5% of steps**: LR is appropriate, training is smooth
- **20–50% of steps**: LR may be too high, or data has frequent noisy batches
- **>80% of steps**: LR is definitely too high, or there is a systematic numerical issue

Track both the gradient norm and whether clipping fired at each step.

---

## Weight Initialization and μP

### Why Initialization Matters at Scale

Poor weight initialization causes two problems:
1. **Gradient explosion/vanishing at init**: if weights are too large, activations grow exponentially with depth; if too small, they shrink. In a 80-layer Transformer, wrong initialization makes training infeasible.
2. **Hyperparameter transfer failure**: the optimal learning rate for a 1B model may not transfer to a 70B model trained with the same architecture and data, because the gradient magnitudes differ.

### Standard Initialization

For a linear layer with weight $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$:

$$W_{ij} \sim \mathcal{N}\left(0, \frac{2}{d_{\text{in}} + d_{\text{out}}}\right) \quad \text{(Glorot/Xavier)}$$

$$W_{ij} \sim \mathcal{N}\left(0, \frac{2}{d_{\text{in}}}\right) \quad \text{(He, for ReLU)}$$

LLaMA uses a variant: $W_{ij} \sim \mathcal{N}(0, \sigma^2)$ with $\sigma = 0.02$ for most weights, and **scaled residual initialization**: the output projection of attention and the second linear of FFN are initialized with $\sigma = 0.02 / \sqrt{2L}$ (where $L$ is the number of layers). This ensures that the residual stream does not grow with depth.

### μP: Maximal Update Parametrization

Greg Yang et al. (2022) introduced **μP (mu-Parametrization)**, a principled reparametrization of Transformer weights that ensures:
1. Activations and gradient norms are stable across model widths at initialization
2. Optimal hyperparameters (especially learning rate) transfer across model scales

The core idea: the learning rate for a weight matrix should scale inversely with the fan-in (the number of input features). For a weight $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$:

$$\eta_W \propto \frac{1}{d_{\text{in}}}$$

Under standard parametrization, increasing width increases the gradient norm at the output (because more inputs sum into the output), requiring LR to be retuned. Under μP, the LR naturally adjusts so that the *effective* update magnitude stays constant as width scales.

**Practical implication**: under μP, you can tune hyperparameters (LR, optimizer momentum, weight decay) at a small proxy model (e.g., 100M parameters) and transfer them directly to the full-scale model (70B+). This dramatically reduces the cost of hyperparameter search for large models.

Microsoft used μP for their Phi models; it is implemented in the `mup` Python package.

---

## Checkpointing Strategy

### What to Checkpoint

A complete checkpoint contains:
- Model parameters (BF16 or FP32)
- Optimizer states (Adam: FP32 first and second moments for all parameters)
- Learning rate schedule state (current step, warmup state)
- Random number generator states (for all GPUs, to ensure reproducible data ordering)
- Data loader state (which batches have been seen, shuffling seed)

A full ZeRO-3 checkpoint for a 70B model is ~1–2TB in sharded form. Saving and loading takes 5–30 minutes depending on storage bandwidth.

### Checkpoint Frequency

A balance between storage cost and recovery overhead:
- **Every 1,000 steps** is common for multi-week training runs (~50 checkpoints total for 500k steps)
- **Keep the last 3–5 checkpoints**: allows rolling back across multiple loss spikes or hardware failures
- **Permanent checkpoints every 10% of training**: for analysis, early model releases, and ablation studies
- **Emergency checkpoints**: some systems checkpoint every 100 steps during known unstable phases (post-spike recovery)

### Asynchronous Checkpointing

Writing a 1TB checkpoint synchronously blocks training for 5–30 minutes. Asynchronous checkpointing (saving to a CPU buffer in a background thread while training continues) eliminates this pause. PyTorch FSDP supports this via `StateDictType.SHARDED_STATE_DICT` with async I/O to distributed storage (e.g., AWS S3, GPFS).

### Resuming from Checkpoint

To resume correctly:
1. Load model weights and optimizer states from the checkpoint
2. Restore the LR schedule state (so warmup/decay continues correctly)
3. Restore the RNG state on each GPU (so data is not repeated or skipped)
4. Skip ahead in the dataset to the correct position (or use a deterministic shuffled dataset index)

Incorrect resumption — e.g., restarting the LR from a warmup instead of the correct cosine decay position — can waste hundreds of steps or cause instability.

---

## Common Failure Modes

### Gradient Explosion

**Symptoms**: gradient norm grows over many steps; then loss spikes and does not recover; eventually NaN.

**Causes**: learning rate too high; initialization too large; missing gradient clipping; numerical overflow in FP16.

**Diagnosis**: plot gradient norm over training. If it grows monotonically over thousands of steps, intervention is needed before the explosion.

**Fix**: reduce learning rate; add or reduce gradient clip threshold; check for FP16 overflow; inspect per-layer norms for the exploding layer.

### NaN Losses

**Symptoms**: loss becomes NaN at step $t$; all subsequent losses are NaN (NaN propagates through the graph).

**Causes**: division by zero (e.g., log(0) in cross-entropy); sqrt of negative number; FP16 overflow; bad input (zero-length sequence, all-masked batch); hardware bit flip.

**Diagnosis**: binary search by inserting `torch.isnan().any()` checks after each layer; if NaN appears at layer $k$ but not $k-1$, the source is in layer $k$'s computation.

**Fix**: roll back to last checkpoint; identify the bad batch (log batch IDs and replay); add `assert not torch.isnan(loss)` after each major computation; use BF16 instead of FP16.

### Collapsed or Dead Layers

**Symptoms**: training loss plateaus prematurely; gradient norms for specific layers drop to near zero; layer outputs become constant.

**Causes**: initialization issues; learning rate too low for specific layers; softmax saturation (attention heads always attend to same position); ReLU death in FFN.

**Diagnosis**: monitor per-layer activation statistics (mean, variance) and gradient norms. A layer producing constant outputs is collapsed.

**Fix**: reduce model depth or add layer normalization; check initialization; for attention saturation, check if temperature scaling is needed; consider SwiGLU or GELU activations (less prone to death than ReLU).

### Rank Collapse in Attention

At scale, attention weight matrices can develop **low effective rank**: most attention heads learn nearly identical patterns, wasting capacity. This is not a catastrophic failure but reduces model efficiency.

**Detection**: compute the effective rank of $QK^\top$ matrices periodically (fraction of singular value energy in top-$k$ singular values).

**Prevention**: QK-normalization (normalize Q and K before the dot product); used in LLaMA 3 and Gemma 2.

---

## Training Monitoring

### Essential Metrics (every step)

| Metric | What it tells you |
|--------|-----------------|
| Training loss | Primary optimization signal |
| Gradient norm | Stability; spike detection |
| Learning rate | Confirm schedule is correct |
| Tokens/second | Throughput; detect hardware issues |
| GPU utilization (%) | Detect communication bottlenecks |
| GPU memory usage (GB) | Detect memory leaks |
| Loss scale (FP16 only) | Numerical stability |

### Diagnostic Metrics (every N steps)

| Metric | What it tells you |
|--------|-----------------|
| Validation perplexity (held-out set) | Generalization; overfitting detection |
| Per-domain validation loss | Which domains the model is learning |
| Gradient norm per layer | Structural health of the model |
| Activation statistics per layer | Layer collapse detection |
| Weight norm per layer | Implicit regularization |
| Attention entropy per head | Attention diversity |

### Loss Curve Shape as Diagnosis

```
Healthy:
Loss │╲
     │  ╲
     │    ╲───────────────────────────
     └────────────────────────────────

Too high LR (oscillating):
Loss │╲╱╲╱╲
     │      ╲╱╲╱╲
     │              ╲───────────────
     └────────────────────────────────

Data quality issue (plateau):
Loss │╲
     │  ╲
     │    ────────────────────────────  ← stuck at high loss
     └────────────────────────────────

Catastrophic (NaN cascade):
Loss │╲
     │  ╲      ╭─∞
     │    ╲___/
     └────────────────────────────────
```

### Tooling

- **Weights & Biases (wandb)**: standard for LLM training runs; handles distributed logging, metric aggregation, checkpoint tracking, and alerting
- **TensorBoard**: simpler alternative; integrated in PyTorch Lightning and HuggingFace Trainer
- **NVIDIA Nsight Systems**: GPU-level profiling for identifying compute vs communication bottlenecks
- **PyTorch Profiler**: step-level CPU/GPU timeline; identifies which operations are slow

---

## LARS and LAMB: Large-Batch Optimizers

### The Large-Batch Training Problem

Training efficiency requires large batch sizes (more tokens per step → more GPU utilization). However, naively scaling batch size degrades generalization: the model converges faster (fewer steps to same loss) but to a worse minimum. This is the "generalization gap" (Keskar et al., 2017).

The fix: scale the learning rate with the batch size (linear scaling rule: $\eta \propto B$). But beyond a certain batch size, linear scaling causes instability because the gradient steps become too large relative to the curvature of the loss landscape.

### LARS

LARS (Layer-wise Adaptive Rate Scaling, You et al., 2017) scales the learning rate of each layer $l$ by the ratio of weight norm to gradient norm:

$$\eta_l = \lambda \cdot \frac{\|w_l\|_2}{\|g_l\|_2 + \beta \|w_l\|_2}$$

where $\lambda$ is a global learning rate, $\beta$ is a trust coefficient, and $\|w_l\|_2 / \|g_l\|_2$ is the local "trust ratio." This prevents any single layer from taking steps larger than the natural scale of its weights, enabling very large batch sizes (up to 32k images in ImageNet training).

LARS works well for **SGD with momentum** (used in image classification). It is less effective for Transformers trained with Adam.

### LAMB

LAMB (Layer-wise Adaptive Moments optimizer for Batch training, You et al., 2019) adapts the LARS concept to Adam:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{u}_t = \frac{m_t / (1-\beta_1^t)}{\sqrt{v_t / (1-\beta_2^t)} + \epsilon}$$
$$\eta_l = \lambda \cdot \frac{\|w_l\|_2}{\|\hat{u}_l\|_2}$$
$$w_t = w_{t-1} - \eta_l \cdot \hat{u}_t$$

LAMB computes the Adam update direction $\hat{u}_t$ and then applies a LARS-style trust ratio scaling per layer. This allows stable training at very large batch sizes (up to 32k sequences).

LAMB was used to train BERT in 76 minutes (vs 3 days with Adam) by scaling to large batches on TPU pods. For LLM pretraining, LAMB is less commonly used than AdamW (which, combined with appropriate LR schedules and warmup, achieves sufficient stability at practical batch sizes of 1M–4M tokens).

---

## References

- Zhao, W., et al. (2024). *GaLore: Memory-efficient LLM training by gradient low-rank projection.* ICML. [arXiv:2403.03507](https://arxiv.org/abs/2403.03507)
- Micikevicius, P., et al. (2018). *Mixed precision training.* ICLR. [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)
- Yang, G., et al. (2022). *Tensor programs V: Tuning large neural networks via zero-shot hyperparameter transfer (μP).* [arXiv:2203.03466](https://arxiv.org/abs/2203.03466)
- You, Y., et al. (2019). *Large batch optimization for deep learning: Training BERT in 76 minutes (LAMB).* ICLR. [arXiv:1904.00962](https://arxiv.org/abs/1904.00962)
- You, Y., et al. (2017). *Large batch training of convolutional networks (LARS).* [arXiv:1708.03888](https://arxiv.org/abs/1708.03888)
- Keskar, N., et al. (2017). *On large-batch training for deep learning: Generalization gap and sharp minima.* ICLR. [arXiv:1609.04836](https://arxiv.org/abs/1609.04836)
- Glorot, X., Bengio, Y. (2010). *Understanding the difficulty of training deep feedforward neural networks.* AISTATS.
- Hu, S., et al. (2024). *MiniCPM: Scaling small language models with WSD LR schedule.* [arXiv:2404.06395](https://arxiv.org/abs/2404.06395)
- Meta AI (2024). *The LLaMA 3 herd of models.* [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)
- Dubey, A., et al. (2024). *The LLaMA 3 herd of models.* [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)
- Loshchilov, I., Hutter, F. (2019). *Decoupled weight decay regularization (AdamW).* ICLR. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)

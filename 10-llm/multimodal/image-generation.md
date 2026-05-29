# Image Generation

## Table of Contents

1. [Overview](#overview)
2. [Historical Context: GANs and VQ-VAE](#historical-context-gans-and-vq-vae)
3. [Diffusion Models](#diffusion-models)
4. [Latent Diffusion Models (Stable Diffusion)](#latent-diffusion-models-stable-diffusion)
5. [DALL-E 2 and DALL-E 3](#dall-e-2-and-dall-e-3)
6. [FLUX: Flow Matching and MMDiT](#flux-flow-matching-and-mmdit)
7. [Fine-tuning Techniques](#fine-tuning-techniques)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Inference Speed: Distillation and Consistency Models](#inference-speed-distillation-and-consistency-models)
10. [Safety Considerations](#safety-considerations)
11. [Code Example](#code-example)
12. [References](#references)

---

## Overview

Text-to-image generation is the task of synthesizing a photorealistic (or artistic) image from a natural language description. It sits at the intersection of computer vision, language understanding, and generative modeling.

Why is this hard?

1. **Combinatorial complexity**: "A red cube on top of a blue sphere to the left of a green cylinder" requires correctly binding attributes to objects and encoding spatial relations — both of which are notoriously difficult.
2. **High dimensionality**: a 512×512 RGB image is a 786,432-dimensional object. Learning a distribution over this space is intractable without strong inductive biases.
3. **Semantic alignment**: the output must faithfully reflect the text semantics, not just look photorealistic.
4. **Mode diversity**: generating "a dog" 10 times should yield 10 different dogs, not the same dog 10 times.

The field has gone through three distinct phases: GANs (dominant 2014–2021), discrete token models (DALL-E 1, 2021), and diffusion models (dominant 2022–present). Diffusion models won because they produce higher fidelity, better diversity, and more text-faithful outputs than GANs, while being easier to train stably.

---

## Historical Context: GANs and VQ-VAE

### GAN-based Generation

Generative Adversarial Networks (Goodfellow et al., 2014) train a generator $G$ and discriminator $D$ in a minimax game:

$$\min_G \max_D \; \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

**StyleGAN** (Karras et al., 2019) achieved photorealistic face synthesis at 1024px via style-based generator with AdaIN layers, progressive growing, and path length regularization. State-of-the-art on faces, but:
- **Mode collapse**: GANs tend to ignore parts of the training distribution
- **Training instability**: sensitive to hyperparameters, requires careful tuning
- **Text conditioning was brittle**: conditioning GANs on language (StackGAN, AttnGAN) rarely produced coherent compositional scenes

### DALL-E 1: VQ-VAE + Transformer

**Paper**: Ramesh et al., OpenAI, 2021.

DALL-E 1 uses a two-stage pipeline:
1. **VQ-VAE** (Vector Quantized VAE): train an encoder-decoder that compresses 256×256 images into a 32×32 grid of discrete tokens (from a codebook of 8192 entries). Each image → 1024 discrete tokens.
2. **Autoregressive Transformer**: train a 12B parameter GPT-like model to predict image tokens autoregressively, conditioned on text tokens. Text (up to 256 BPE tokens) + image tokens (1024) = 1280-token sequence.

At inference: sample image tokens one-by-one from the transformer, then decode with the VQ-VAE decoder.

**Limitations**: slow (1024 autoregressive steps), limited resolution (256×256), output quality below contemporaneous diffusion models.

---

## Diffusion Models

Diffusion models (Sohl-Dickstein et al., 2015; Ho et al., 2020) define a **forward process** that gradually corrupts data with Gaussian noise, then learn a **reverse process** that denoises.

### Forward Process

Define a Markov chain that adds noise over $T$ steps (typically $T=1000$):

$$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t;\; \sqrt{1-\beta_t}\, x_{t-1},\; \beta_t \mathbf{I})$$

where $\{\beta_t\}_{t=1}^T$ is a **noise schedule** (linear, cosine, or learned). As $t \to T$, $x_T \approx \mathcal{N}(0, \mathbf{I})$.

Using the reparameterization with $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$:

$$q(x_t \mid x_0) = \mathcal{N}(x_t;\; \sqrt{\bar{\alpha}_t}\, x_0,\; (1-\bar{\alpha}_t)\mathbf{I})$$

This closed-form lets you **sample any noisy step directly** from $x_0$:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

### Reverse Process

The reverse process $p_\theta(x_{t-1} \mid x_t)$ is intractable directly. We parameterize it as a Gaussian:

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1};\; \mu_\theta(x_t, t),\; \sigma_t^2 \mathbf{I})$$

and learn $\mu_\theta$ by training a neural network $\epsilon_\theta$ (a U-Net) to predict the noise $\epsilon$ given $x_t$ and $t$:

### DDPM Training Objective

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{x_0, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

At each training step:
1. Sample a real image $x_0$ from the dataset
2. Sample a timestep $t \sim \text{Uniform}(1, T)$
3. Sample noise $\epsilon \sim \mathcal{N}(0, \mathbf{I})$
4. Compute noisy image $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$
5. Predict noise $\hat{\epsilon} = \epsilon_\theta(x_t, t)$
6. Minimize $\|\epsilon - \hat{\epsilon}\|^2$

**Why predict noise rather than $x_0$ directly?** Empirically, noise prediction leads to better sample quality. Theoretically, at high noise levels, predicting $x_0$ from $x_T$ is nearly impossible (too much information destroyed), while predicting the noise that was added is a well-conditioned regression problem.

### DDPM Sampling

Reverse diffusion samples from $p_\theta$ iteratively:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right) + \sigma_t z, \quad z \sim \mathcal{N}(0, \mathbf{I})$$

This requires $T=1000$ forward passes through the U-Net — slow, but parallelizable per-image.

### DDIM: Deterministic Sampling

DDIM (Song et al., 2020) redefines the reverse process as **non-Markovian and deterministic**:

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\underbrace{\frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta}{\sqrt{\bar{\alpha}_t}}}_{\text{predicted }x_0} + \sqrt{1-\bar{\alpha}_{t-1}}\,\epsilon_\theta$$

DDIM uses the same trained $\epsilon_\theta$ but allows **subsampling**: instead of all 1000 steps, take 20–50 steps on a subset of timesteps. At 50 DDIM steps, quality is comparable to 1000 DDPM steps. At $\eta=0$ the sampling is fully deterministic (same seed → same image), enabling reproducibility and interpolation in latent space.

### U-Net Architecture

The noise prediction network $\epsilon_\theta$ is a **U-Net** with:
- **Downsampling path**: successive ResNet blocks + downsampling, each adding spatial detail via skip connections
- **Bottleneck**: transformer blocks with self-attention (for global coherence)
- **Upsampling path**: transposed convolutions with skip connections from encoder
- **Time conditioning**: timestep $t$ is embedded via sinusoidal encoding + MLP, added to every ResNet block via AdaGN (adaptive group normalization)

```
Input x_t [512×512×3]
    ↓ Conv
[512×512×128]──────────────────────────────────────┐
    ↓ ResBlock + Attention                          │ skip
[256×256×256]────────────────────────────────┐     │
    ↓ ResBlock + Attention                   │ skip │
[128×128×512]────────────────────────┐       │     │
    ↓ ResBlock + Attention (+ cross) │ skip  │     │
[64×64×1024] ←────── Bottleneck ────→│       │     │
    ↑ ResBlock + Attention            │       │     │
[128×128×512]────────────────────────┘       │     │
    ↑ ResBlock + Attention                   │     │
[256×256×256]────────────────────────────────┘     │
    ↑ ResBlock + Attention                          │
[512×512×128]──────────────────────────────────────┘
    ↓ Conv
Predicted noise ε̂ [512×512×3]
```

Text conditioning enters via **cross-attention** in the transformer blocks: keys and values come from text embeddings, queries from spatial features.

---

## Latent Diffusion Models (Stable Diffusion)

**Paper**: Rombach et al., CompVis/Stability AI, 2022. *High-Resolution Image Synthesis with Latent Diffusion Models.*

### The Key Insight

Running diffusion in pixel space at 512×512 is expensive: 1000 U-Net passes over $512 \times 512 \times 3 = 786{,}432$-dimensional inputs. The key insight of LDM: **most perceptual information lives in a much lower-dimensional latent space**.

Train a VAE to compress images to a small latent:
- Encoder: $\mathcal{E}(x) = z$, where $z \in \mathbb{R}^{H/f \times W/f \times C}$, typically $f=8$, $C=4$
- For 512×512 images: $z \in \mathbb{R}^{64 \times 64 \times 4}$ — **64× fewer dimensions**
- Decoder: $\mathcal{D}(z) \approx x$

Then run diffusion entirely in $z$-space:
$$\mathcal{L}_{LDM} = \mathbb{E}_{z_0 = \mathcal{E}(x_0), \epsilon, t}\left[\|\epsilon - \epsilon_\theta(z_t, t, \tau_\theta(y))\|^2\right]$$

where $y$ is the text prompt and $\tau_\theta$ is the text encoder.

**Why this works**: the VAE latent space is nearly Gaussian (regularized by KL divergence during VAE training), making it an appropriate space for Gaussian diffusion. The VAE handles low-level perceptual details (sharp edges, textures); the diffusion model focuses on semantic structure.

**Speed advantage**: operating in 64×64×4 vs 512×512×3 gives roughly 48× reduction in U-Net input size, enabling faster training and inference.

### Components

```
Text prompt y
      ↓
CLIP Text Encoder (or OpenCLIP ViT-H)
      ↓
Text embeddings τ(y)  [77 × 768]
      ↓ (cross-attention into U-Net)

Image x → VAE Encoder → z₀ → [noising] → zₜ
                                     ↓
                          U-Net ε_θ(zₜ, t, τ(y))
                                     ↓
                          Predicted noise ε̂
                                     ↓
                          [denoising loop: T steps]
                                     ↓
                              z₀ (estimated)
                                     ↓
                          VAE Decoder → generated image x̂
```

### Classifier-Free Guidance (CFG)

**Paper**: Ho and Salimans, 2021.

The tension in text-to-image generation: a model that perfectly maximizes diversity generates images unrelated to the prompt; a model that perfectly maximizes text alignment generates one image for each prompt (low diversity). CFG allows explicit control over this trade-off at inference time without any additional models.

During training, randomly drop the text conditioning with probability $p_{uncond} \approx 0.1$, training the model for both conditional ($\epsilon_\theta(z_t, t, y)$) and unconditional ($\epsilon_\theta(z_t, t, \varnothing)$) generation.

At inference, **extrapolate in the direction of the conditional**:

$$\tilde{\epsilon}_\theta(z_t, t, y) = \epsilon_\theta(z_t, t, \varnothing) + w \cdot \left(\epsilon_\theta(z_t, t, y) - \epsilon_\theta(z_t, t, \varnothing)\right)$$

where $w$ is the **CFG scale** (guidance weight).

- $w = 1$: no guidance, maximum diversity, potentially low text alignment
- $w = 7.5$: typical Stable Diffusion default, good balance
- $w = 15$+: high text alignment, but oversaturation and reduced diversity

**Why CFG works geometrically**: the unconditional model estimates the score of the marginal distribution $p(z)$; the conditional model estimates the score of $p(z | y)$. Their difference $\epsilon_\theta(z_t, y) - \epsilon_\theta(z_t, \varnothing)$ approximates $\nabla_z \log p(y|z)$, the gradient of how much the image "matches" the text. Scaling by $w > 1$ amplifies this gradient, pushing the sample toward regions of high text-image alignment.

---

## DALL-E 2 and DALL-E 3

### DALL-E 2

**Paper**: Ramesh et al., OpenAI, 2022. *Hierarchical Text-Conditional Image Generation with CLIP Latents.*

DALL-E 2 operates in CLIP's embedding space rather than pixel space. The pipeline has three components:

1. **Prior**: text → CLIP image embedding. A diffusion model (or autoregressive transformer) that generates the CLIP image embedding $z_i$ given the CLIP text embedding $z_t$: $p(z_i | z_t)$
2. **Decoder**: CLIP image embedding → image. A diffusion model conditioned on the CLIP image embedding (+ optional text) that generates the actual pixels
3. **CLIP**: provides the shared embedding space

Why generate in CLIP space first? CLIP image embeddings are highly semantic (they capture what is in the image, not pixel-level details) and already aligned with text. The decoder then fills in perceptual details consistent with the CLIP embedding.

**CLIP space manipulation** enables semantic image editing: moving linearly in CLIP space between two image embeddings interpolates meaningfully between their visual concepts. Adding the CLIP embedding of "in watercolor style" to a photo embedding produces a watercolor version.

### DALL-E 3

**Paper**: Betker et al., OpenAI, 2023.

DALL-E 3's key innovation is not architectural but **data-driven**: **caption recaptioning**.

Web-scraped image-text pairs have low-quality captions (alt text like "img_4923.jpg" or generic descriptions). DALL-E 3 used a fine-tuned image captioner to generate **detailed, accurate, descriptive** synthetic captions for all training images. Training on these synthetic captions dramatically improved prompt adherence.

Other improvements:
- **Integrated with ChatGPT**: ChatGPT rewrites user prompts to be more descriptive before sending to DALL-E 3, improving results for casual users
- **Text rendering**: DALL-E 3 is the first widely-deployed model that can reliably render legible text in images ("a sign that says STOP")
- **Aspect ratios**: native support for 1:1, 16:9, 9:16 without content distortion

---

## FLUX: Flow Matching and MMDiT

**Released**: Black Forest Labs, 2024. (Former Stability AI team members.)

FLUX.1 is the current state-of-the-art open-weight image generation model. It introduces two architectural innovations:

### Flow Matching

Instead of the DDPM noise-prediction objective, FLUX uses **Rectified Flow Matching** (Liu et al., 2022; Lipman et al., 2022):

- Define a simple **linear interpolation path** between noise $x_1 \sim \mathcal{N}(0, I)$ and data $x_0$:
  $$x_t = (1 - t) x_0 + t x_1, \quad t \in [0, 1]$$
- The velocity field $v = x_1 - x_0$ is **constant along each path**
- Train a neural network to predict this velocity: $v_\theta(x_t, t)$
- Objective: $\mathcal{L} = \mathbb{E}_{t, x_0, x_1}\left[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2\right]$

Flow matching has **straighter trajectories** than DDPM's curved diffusion paths, allowing accurate generation in fewer NFE (neural function evaluations): 4 steps vs 20-50 for DDIM. This is because optimal transport flow matching minimizes trajectory curvature.

### Multimodal Diffusion Transformer (MMDiT)

FLUX replaces the U-Net with a **Transformer** architecture (following DiT — Diffusion Transformer, Peebles and Xie, 2022). The key innovation is the **Multimodal Diffusion Transformer**:

```
Text tokens (T5-XXL + CLIP embeddings)     Image tokens (patchified latent)
         |                                          |
         v                                          v
    ┌──────────────────────────────────────────────────────┐
    │            MMDiT Block                               │
    │   Text self-attention ←── cross-attention ──→ Image self-attention │
    │                 ↕                                    │
    │          Bidirectional attention: both modalities    │
    │          attend to each other symmetrically          │
    └──────────────────────────────────────────────────────┘
                          ↓
                 ... repeat N times ...
                          ↓
                 Image tokens only → unpatchify → generated image
```

Unlike UNet-based models where text enters only as cross-attention queries in specific layers, MMDiT treats text and image tokens **jointly**: they share the same attention mechanism. This allows richer, more symmetric information exchange between modalities. Text is encoded with both CLIP (semantic, 768-dim) and T5-XXL (syntactic/detailed, 4096-dim) for richer conditioning.

**FLUX variants**:
- FLUX.1-dev: 12B parameters, guidance-distilled, non-commercial
- FLUX.1-schnell: distilled for 1-4 step inference, Apache 2.0
- FLUX.1-pro: highest quality, API-only

---

## Fine-tuning Techniques

### DreamBooth

**Paper**: Ruiz et al., Google, 2022.

**Goal**: teach the model a specific subject (person, pet, object) from 3–5 reference images, represented by a rare identifier token like `sks`.

**Method**:
- Fine-tune the **entire diffusion model** on the subject images with prompts like "A photo of [sks] dog"
- To prevent **language drift** (forgetting what "dog" means), use **prior preservation loss**: simultaneously sample from the model's prior distribution of the class ("a photo of a dog") and include those samples in the training batch

$$\mathcal{L} = \mathbb{E}\left[\|\epsilon - \epsilon_\theta(z_t, t, \text{"[sks] dog"})\|^2\right] + \lambda \mathbb{E}\left[\|\epsilon' - \epsilon_\theta(z'_t, t, \text{"a dog"})\|^2\right]$$

**Limitation**: fine-tuning all parameters is expensive (~10GB VRAM) and risks overfitting on 3–5 images.

### LoRA for Diffusion Models

LoRA (Low-Rank Adaptation) applies to diffusion U-Nets the same way it applies to LLMs: add low-rank matrices to the attention weight matrices of the U-Net, train only those parameters.

- Typical rank: $r = 4$ to $r = 128$
- Train only: $W_Q, W_K, W_V, W_O$ in U-Net attention layers
- ~4MB of weights vs ~4GB full model
- Can be combined: multiple style LoRAs can be merged at inference with linear weighting

LoRA for diffusion is the dominant fine-tuning approach in the open-source community (Civitai has hundreds of thousands of community LoRAs).

### ControlNet

**Paper**: Zhang et al., 2023.

**Goal**: add spatial conditioning inputs (depth maps, pose skeletons, edge maps, segmentation masks) to a pre-trained diffusion model while keeping the original model weights frozen.

**Architecture**: duplicate the U-Net encoder (trainable copy), connect it to the frozen decoder via "zero convolutions" (initialized to output zero):

```
                      ControlNet (trainable copy of encoder)
Conditioning input → Encoder copy → Zero Convolutions → ┐
                                                          ├→ Frozen U-Net Decoder
     Noisy latent → Frozen Encoder → ────────────────── ┘
```

Zero convolutions ensure that at initialization, ControlNet outputs zero — the model starts as the original pre-trained model and gradually learns to use the conditioning.

**Supported conditioning types** (separate ControlNet for each):
- Canny edges → detailed structure preservation
- Depth maps → 3D-consistent generation
- Human pose (OpenPose) → character posing
- Segmentation masks → region-controlled generation
- Normal maps, line art, scribbles, etc.

### Textual Inversion

**Paper**: Gal et al., 2022.

**Goal**: represent a new concept as a new text token embedding, without changing any model weights.

- Add a new token `<my-concept>` to the tokenizer
- Initialize its embedding vector randomly (or from an existing word like "dog")
- Keep all model weights frozen; train only the single embedding vector
- After training: use `"a photo of <my-concept> in Paris"` etc.

**Trade-off vs DreamBooth**: Textual Inversion is parameter-efficient (one 768-dim vector) and preserves the model exactly, but has less capacity to capture complex subjects than DreamBooth's full fine-tune.

---

## Evaluation Metrics

### FID (Fréchet Inception Distance)

The dominant metric for image quality and diversity.

$$\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$

where $\mu_r, \Sigma_r$ are the mean and covariance of Inception-v3 features extracted from **real** images, and $\mu_g, \Sigma_g$ from **generated** images.

FID measures how similar the **distribution** of generated images is to real images, capturing both quality (low FID requires sharp, realistic images) and diversity (low FID requires covering the real distribution's spread).

**Limitations**: FID requires ~50K samples for reliable estimates; it doesn't directly measure text alignment; it's sensitive to the preprocessing pipeline (resolution, center-crop vs resize).

### CLIP Score

Measures **text-image alignment**:

$$\text{CLIP Score}(c, I) = w \cdot \max(100 \cdot \cos(\mathbf{e}_c, \mathbf{e}_I), 0)$$

where $\mathbf{e}_c$ is the CLIP text embedding of caption $c$ and $\mathbf{e}_I$ is the CLIP image embedding. Higher is better; typical values for good generations are 25–35.

**Limitation**: measures CLIP-alignment, not human judgment. A model trained to maximize CLIP score can "cheat" (produce images that fool CLIP but look bad to humans).

### IS (Inception Score)

$$\text{IS} = \exp\left(\mathbb{E}_x \left[\text{KL}(p(y|x) \| p(y))\right]\right)$$

where $p(y|x)$ is the Inception-v3 class distribution for generated image $x$ and $p(y) = \mathbb{E}_x[p(y|x)]$ is the marginal.

IS rewards two properties: **quality** (generated images should be classified confidently, so $p(y|x)$ is peaked) and **diversity** (the marginal $p(y)$ should be uniform — different images should belong to different classes). However, IS doesn't compare to real images and can be gamed by generating diverse but unrealistic images.

### Human Evaluation

The ground truth. Typical protocol: A/B comparison between models on text-image alignment and overall image quality, rated by human annotators. DrawBench, PartiPrompts, and GenEval provide standardized challenging prompts for human evaluation.

**GenEval** (Ghosh et al., 2023) is an automatic compositional evaluation: tests attribute binding ("red cube"), counting ("three cats"), spatial relations ("to the left of") — exactly the failure modes that FID/CLIP miss.

---

## Inference Speed: Distillation and Consistency Models

Standard LDM/SD inference requires 20–50 DDIM steps (U-Net calls), taking 2–5 seconds on an A100. Several approaches dramatically accelerate this:

### Progressive Distillation

Train a student model to match the output of a teacher model in **half the steps**. Repeat: 1000→500→250→...→4 steps. Each round of distillation doubles inference speed with small quality loss.

### SDXL-Turbo (Adversarial Diffusion Distillation)

ADD (Sauer et al., Stability AI, 2023): combine score distillation (matching teacher distributions) with adversarial training (discriminator that distinguishes real from generated). Achieves 1-step generation at acceptable quality.

### LCM (Latency Consistency Models)

**Paper**: Luo et al., 2023.

Consistency models (Song et al., 2023) enforce **self-consistency**: starting from any point $x_t$ on the trajectory, the model should predict the same $x_0$. This allows single-step or multi-step generation with no iterative refinement.

$$f_\theta(x_t, t) = f_\theta(x_{t'}, t'), \quad \forall t, t' \text{ on the same trajectory}$$

LCM distills a pre-trained LDM into a consistency model via distillation, achieving 2–4 step generation without noticeable quality loss.

**LCM-LoRA**: apply consistency distillation only to LoRA weights, making it plug-and-play with existing community models.

### FLUX.1-schnell

As discussed: flow matching's straighter trajectories allow accurate 1–4 step generation natively, without distillation.

---

## Safety Considerations

### NSFW Filtering

Most deployed systems implement two-stage filtering:
1. **Prompt filtering**: block or modify prompts requesting explicit content (keyword lists, classifier on prompt text)
2. **Output filtering**: run a safety classifier on the generated image before returning (CLIP-based or dedicated CNN classifier for NSFW, violence, etc.)

Stable Diffusion includes a Safety Checker based on CLIP features. It has both false positives (incorrectly blocking benign medical/artistic content) and false negatives (novel NSFW content not in training distribution).

### Watermarking

**C2PA (Coalition for Content Provenance and Authenticity)**: metadata standard that cryptographically signs generated images with creator, tool, and timestamp information. Adopted by OpenAI (DALL-E 3), Adobe (Firefly), and others.

**Invisible watermarks**: steganographic signals embedded in pixel values, surviving moderate compression. **Stable Signature** (Fernandez et al., Meta 2023) fine-tunes the VAE decoder to embed an imperceptible watermark in every generated image.

**Limitation**: watermarks are removed by adversarial cropping, re-diffusion, or GAN-based washing.

### Style Mimicry

Training on artist data without consent raises copyright concerns. The community has developed:
- **Glaze** (SAND Lab): adversarial perturbations on artwork that cause VLMs to misclassify style
- **Anti-DreamBooth**: perturbations that prevent DreamBooth from fine-tuning on specific faces
- **Copyright law**: unsettled; Stable Diffusion and related lawsuits (Andersen v. Stability AI) are still being litigated

---

## Code Example

```python
# Stable Diffusion XL inference with Classifier-Free Guidance
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
# Use faster DPM-Solver++ scheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "A photorealistic orange tabby cat wearing a tiny wizard hat, studio lighting, 4K"
negative_prompt = "blurry, low quality, cartoon, painting, watermark"

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=25,    # DPM-Solver++ is fast; 25 steps is sufficient
    guidance_scale=7.5,        # CFG scale
    height=1024,
    width=1024,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]

image.save("output.png")

# --- ControlNet: pose-conditioned generation ---
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.utils import load_image
import numpy as np
import cv2

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",  # edge-conditioned ControlNet
    torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
).to("cuda")

# Prepare Canny edge map as conditioning
reference_image = load_image("reference.jpg")
image_array = np.array(reference_image)
edges = cv2.Canny(image_array, 100, 200)
edges_rgb = np.stack([edges] * 3, axis=-1)
from PIL import Image
control_image = Image.fromarray(edges_rgb)

generated = pipe(
    prompt="A futuristic cityscape with neon lights, cyberpunk style",
    image=control_image,
    num_inference_steps=30,
    guidance_scale=7.5,
    controlnet_conditioning_scale=0.8,  # strength of conditioning
).images[0]

# --- LCM: ultra-fast 4-step generation ---
from diffusers import LCMScheduler, AutoPipelineForText2Image

pipe = AutoPipelineForText2Image.from_pretrained(
    "Lykon/dreamshaper-8",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

# Load LCM-LoRA
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# Generate in 4 steps (vs 25+ for standard DDIM)
image = pipe(
    prompt=prompt,
    num_inference_steps=4,
    guidance_scale=1.5,     # LCM uses low CFG scale (1.0–2.0)
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]
```

---

## References

- Ho et al. (2020). *Denoising Diffusion Probabilistic Models* (DDPM). https://arxiv.org/abs/2006.11239
- Song et al. (2020). *Denoising Diffusion Implicit Models* (DDIM). https://arxiv.org/abs/2010.02502
- Rombach et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models* (LDM/Stable Diffusion). https://arxiv.org/abs/2112.10752
- Ho and Salimans (2021). *Classifier-Free Diffusion Guidance*. https://arxiv.org/abs/2207.12598
- Ramesh et al. (2022). *Hierarchical Text-Conditional Image Generation with CLIP Latents* (DALL-E 2). https://arxiv.org/abs/2204.06125
- Betker et al. (2023). *Improving Image Generation with Better Captions* (DALL-E 3). https://cdn.openai.com/papers/dall-e-3.pdf
- Black Forest Labs (2024). *FLUX.1*. https://blackforestlabs.ai
- Liu et al. (2022). *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*. https://arxiv.org/abs/2209.03003
- Peebles and Xie (2022). *Scalable Diffusion Models with Transformers* (DiT). https://arxiv.org/abs/2212.09748
- Ruiz et al. (2022). *DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation*. https://arxiv.org/abs/2208.12242
- Zhang and Agrawala (2023). *Adding Conditional Control to Text-to-Image Diffusion Models* (ControlNet). https://arxiv.org/abs/2302.05543
- Gal et al. (2022). *An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion*. https://arxiv.org/abs/2208.01618
- Luo et al. (2023). *Latent Consistency Models*. https://arxiv.org/abs/2310.04378
- Karras et al. (2019). *A Style-Based Generator Architecture for Generative Adversarial Networks* (StyleGAN). https://arxiv.org/abs/1812.04948

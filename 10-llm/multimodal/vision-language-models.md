# Vision-Language Models

## Table of Contents

1. [Motivation](#motivation)
2. [Early Cross-Modal Approaches](#early-cross-modal-approaches)
3. [CLIP](#clip-contrastive-language-image-pre-training)
4. [BLIP and BLIP-2](#blip-and-blip-2)
5. [Flamingo](#flamingo)
6. [LLaVA](#llava-large-language-and-vision-assistant)
7. [Proprietary Natively Multimodal Models](#proprietary-natively-multimodal-models)
8. [Open-Source VLMs: InternVL](#open-source-vlms-internvl)
9. [Connecting Images to LLMs: Projection Methods](#connecting-images-to-llms-projection-methods)
10. [Visual Token Count and Context Window Pressure](#visual-token-count-and-context-window-pressure)
11. [High-Resolution Images: Dynamic Tiling](#high-resolution-images-dynamic-tiling)
12. [Benchmarks](#benchmarks)
13. [Code Example](#code-example)
14. [References](#references)

---

## Motivation

Vision is the dominant human sensory channel: roughly half the human cerebral cortex is devoted to visual processing. Language, however, is how humans communicate knowledge. The gap between perceiving the visual world and reasoning about it in language is precisely what Vision-Language Models (VLMs) bridge.

Practical consequences are immediate. A model that understands both images and text can:
- Answer questions about a photograph ("Is there a stop sign in this image?")
- Generate captions for medical images, satellite imagery, or UI screenshots
- Follow visual instructions ("Click the blue button")
- Perform chart and document understanding (reading tables, graphs, PDFs)
- Ground spatial relationships ("What is to the left of the dog?")

Pure language models hallucinate about images they cannot see; pure vision models cannot express nuanced reasoning. VLMs combine the expressive power of large language models with the perceptual representations of vision encoders.

---

## Early Cross-Modal Approaches

### Visual Question Answering (VQA)

VQA (Antol et al., 2015) posed the problem: given an image and a natural language question, produce a natural language answer. Early models fused image features (from CNNs like ResNet or VGG) with question features (from LSTMs) via simple concatenation or bilinear pooling, then classified over a fixed answer vocabulary. These models:
- Required task-specific fine-tuning
- Could not generalize to open-ended answers
- Struggled with spatial reasoning and counting

### Image Captioning

Early captioning models used an encoder-decoder pipeline: CNN feature extractor → LSTM decoder trained with cross-entropy on ground-truth captions. Show, Attend and Tell (Xu et al., 2015) added soft attention over spatial CNN features, producing the first glimpse of cross-modal attention. Still, these models were brittle, domain-specific, and lacked transferable representations.

The fundamental limitation of this era: **representations were learned separately per task**. There was no shared embedding space between vision and language. The breakthrough that changed this was CLIP.

---

## CLIP (Contrastive Language-Image Pre-training)

**Paper**: Radford et al., OpenAI, 2021.

### Core Idea

CLIP learns a **shared embedding space** for images and text via contrastive learning on 400 million image-text pairs scraped from the internet. Rather than training a model to predict exact captions (a noisy, ambiguous objective for web data), CLIP trains two encoders to produce representations where matching image-text pairs have high cosine similarity and non-matching pairs have low similarity.

### Architecture

```
Image → [Vision Transformer (ViT) or ResNet] → image embedding (512-dim)
Text  → [Transformer text encoder]            → text embedding (512-dim)

Both projected into a joint L2-normalized embedding space.
```

The vision encoder is a Vision Transformer (ViT-B/32, ViT-L/14, etc.). The text encoder is a standard Transformer with masked self-attention, capped at 77 tokens. Both encoders output a single vector (via [CLS] token or pooling), projected to the same dimensionality via a linear layer.

### Training: InfoNCE Contrastive Loss

Given a batch of $N$ image-text pairs $\{(v_i, t_i)\}_{i=1}^N$, define the cosine similarity matrix $S \in \mathbb{R}^{N \times N}$ where:

$$S_{ij} = \frac{v_i \cdot t_j}{\|v_i\| \|t_j\| \cdot \tau}$$

Here $\tau$ is a learnable temperature parameter (initialized to 0.07). The loss treats each row as an $N$-way classification (which text matches this image?) and each column as an $N$-way classification (which image matches this text?):

$$\mathcal{L} = -\frac{1}{2N} \sum_{i=1}^{N} \left[ \log \frac{\exp(S_{ii})}{\sum_{j=1}^{N} \exp(S_{ij})} + \log \frac{\exp(S_{ii})}{\sum_{j=1}^{N} \exp(S_{ji})} \right]$$

This is the **symmetric InfoNCE** objective. The diagonal entries $S_{ii}$ are positive pairs; all off-diagonal entries are negatives. With large batch sizes (up to 32,768 in CLIP), the model sees many hard negatives without explicit mining.

Why this works better than captioning loss: the contrastive objective is **noise-robust**. Web alt-text is often wrong or irrelevant; a model need not perfectly match the caption, just rank the correct image-text pair above all others in the batch.

### Zero-Shot Image Classification

CLIP's most striking property: zero-shot transfer to arbitrary image classification tasks.

```
1. For each class c ∈ {c₁, ..., cK}:
   - Construct text prompt: "A photo of a {c}"
   - Compute text embedding tᶜ = TextEncoder("A photo of a {c}")
2. Compute image embedding v = ImageEncoder(image)
3. Predict: argmax_c cosine_similarity(v, tᶜ)
```

On ImageNet, CLIP ViT-L/14 achieves ~76% zero-shot accuracy — matching a ResNet-50 trained on 1.28M labeled ImageNet examples. This is remarkable: CLIP never saw ImageNet labels during training.

**Prompt engineering matters**: "A photo of a {c}, a type of food" outperforms "A photo of a {c}" for fine-grained food classification. Ensembling 80 prompt templates further improves accuracy.

### Limitations

- **Spatial reasoning**: CLIP embeds global semantics; it does not explicitly encode "X is to the left of Y"
- **Object counting**: "three dogs" and "four dogs" produce similar embeddings
- **Fine-grained distinctions**: distinguishing between 200 species of birds is much harder than coarse-grained classification
- **OCR and text in images**: CLIP was not trained specifically for text recognition
- **CLIP does not generate text**: it is a discriminative model, not a generative one

---

## BLIP and BLIP-2

### BLIP (Bootstrapping Language-Image Pre-training)

**Paper**: Li et al., Salesforce, 2022.

BLIP addresses two problems simultaneously:
1. Web data is noisy (alt-text is often irrelevant or misleading)
2. Prior VLMs were either good at understanding (contrastive) or generation (captioning), not both

**Architecture: MED (Multimodal mixture of Encoder-Decoder)**

BLIP uses a single vision encoder (ViT) and a text model with three modes of operation:
- **Unimodal text encoder**: for image-text contrastive learning (like CLIP)
- **Image-grounded text encoder**: cross-attention to image features for image-text matching (ITM)
- **Image-grounded text decoder**: causal self-attention for language generation (captioning)

**CapFilt (Captioner + Filter)**: BLIP's key innovation for data quality.

```
Noisy web image-text pairs
         ↓
  Captioner (fine-tuned on COCO) generates synthetic captions
         ↓
  Filter (fine-tuned on COCO) removes low-quality pairs
  (both noisy alt-text and synthetic captions filtered)
         ↓
  Clean training data for second round of pre-training
```

This bootstrapping loop dramatically improves data quality without human annotation.

### BLIP-2

**Paper**: Li et al., Salesforce, 2023.

BLIP-2 solves a harder problem: how to connect **frozen** pre-trained vision encoders to **frozen** pre-trained LLMs, without expensive end-to-end fine-tuning. The key component is the **Q-Former**.

**Q-Former (Querying Transformer)**

```
Image (224×224) → [Frozen ViT-g/14] → 257 patch tokens
                                              ↓
                               ┌─────────────────────────┐
     N learnable query vectors → Cross-Attention         │
     (N=32, d=768)             │  Q: learned queries     │
                               │  K,V: image patch tokens│
                               └─────────────────────────┘
                                              ↓
                               32 output query vectors (fixed-size)
                                              ↓
                               Linear Projection
                                              ↓
                               Frozen LLM (OPT or FlanT5)
```

The Q-Former is a 12-layer BERT-style Transformer. The $N=32$ learnable query vectors interact with image patch tokens via cross-attention and with each other via self-attention. This compresses the variable-length visual representation into a **fixed 32-token sequence** that the LLM can condition on.

**Two-stage training:**

Stage 1 — Vision-Language Representation Learning (Q-Former + ViT, LLM frozen):
- Image-text contrastive (ITC)
- Image-text matching (ITM)
- Image-grounded text generation (ITG)

Stage 2 — Vision-to-Language Generative Learning (full pipeline, LLM still frozen):
- Q-Former outputs projected to LLM embedding dimension
- LLM generates text conditioned on Q-Former output
- Only Q-Former + projection trained; LLM weights unchanged

**Why freeze the LLM?** Training a 7B+ LLM from scratch with vision data is expensive and risks catastrophic forgetting of language capabilities. The Q-Former acts as an information bottleneck — it must extract only what the LLM needs, discarding visually irrelevant detail.

BLIP-2 with FlanT5-XXL achieves zero-shot VQAv2 performance competitive with Flamingo-80B at 54× fewer trainable parameters.

---

## Flamingo

**Paper**: Alayrac et al., DeepMind, 2022.

Flamingo was the first model to demonstrate strong **few-shot** visual language understanding — learning new tasks from just a few image-text examples in context, analogously to how GPT-3 does few-shot text tasks.

### Architecture

Flamingo connects a **Perceiver Resampler** to a frozen **Chinchilla LLM** (70B) using gated cross-attention layers inserted between every Transformer block.

**Perceiver Resampler**

The vision encoder (NFNet) produces variable-length image patch features (size depends on resolution). The Perceiver Resampler compresses these to a fixed 64-token output:

```
Variable image patches (T_feat tokens)
         ↓
Perceiver Resampler:
  - 64 fixed learnable latent vectors
  - Cross-attention: latents attend to image patches
  - Self-attention: latents attend to each other
  - Repeated L times
         ↓
64 fixed visual tokens
```

This is efficient: regardless of input resolution, the LLM always sees 64 visual tokens per image.

**Gated Cross-Attention Layers**

Flamingo inserts new cross-attention layers before every feed-forward layer in the frozen LLM:

```
LLM block:
  Self-Attention (frozen) → [Gated Cross-Attention (new)] → FFN (frozen)
```

The gated cross-attention has a learnable $\tanh$-gating mechanism: at initialization, $\tanh(0) = 0$, so the new layers have zero contribution and the model starts as the original LLM. Training gradually opens the gate. Only the Perceiver Resampler, cross-attention layers, and input/output projections are trained.

**Interleaved image-text sequences**: Flamingo can handle arbitrarily interleaved sequences:

```
<image> What is in this image? A dog. <image> And this one? A cat. <image> And this?
```

This enables in-context few-shot learning across image-text pairs — a capability most contemporaneous VLMs lacked.

**Results**: Flamingo-80B achieves state-of-the-art few-shot performance on VQAv2, COCO captioning, and TextVQA at the time of publication.

---

## LLaVA (Large Language and Vision Assistant)

**Paper**: Liu et al., 2023 (LLaVA), 2023 (LLaVA-1.5), 2024 (LLaVA-Next).

LLaVA is the most influential open-source VLM architecture. Its simplicity is a feature: a linear projection connects CLIP's vision encoder to an LLM, making the design easy to understand, replicate, and improve.

### LLaVA Architecture

```
Image → [CLIP ViT-L/14 @ 336px] → 576 patch tokens (1024-dim each)
                                          ↓
                              Linear Projection (W_proj)
                                          ↓
                               576 visual tokens (4096-dim)
                                          ↓
                    Prepend to tokenized text → [LLaMA-13B]
                                          ↓
                                   Text output
```

The projection matrix $W_{proj} \in \mathbb{R}^{1024 \times 4096}$ is the only new parameter. Everything else (CLIP ViT and LLaMA) comes pretrained.

**Training procedure (two stages)**:

Stage 1 — Feature alignment (LLM frozen):
- Train only $W_{proj}$ on 595K image-caption pairs (CC3M filtered)
- Goal: map CLIP visual features into LLaMA's word embedding space
- Think of it as teaching LLaMA a new "visual vocabulary"

Stage 2 — Visual instruction tuning (full fine-tune):
- Fine-tune $W_{proj}$ + LLaMA on 158K multimodal instruction-following samples
- Data generated by GPT-4: given an image's COCO annotations, prompt GPT-4 to write diverse conversations, detailed descriptions, and complex reasoning Q&A

**Key insight**: GPT-4 can generate high-quality visual instruction data from text descriptions of images (captions + bounding boxes), without ever seeing the images itself.

### LLaVA-1.5

LLaVA-1.5 upgrades three things:
1. **MLP connector**: replace linear $W_{proj}$ with a two-layer MLP ($\text{GELU}$): `Linear → GELU → Linear`. This substantially improves alignment quality
2. **Higher resolution**: CLIP at 336px instead of 224px → 576 tokens instead of 256
3. **Better data**: ShareGPT4V (high-quality GPT-4V generated data), academic task data (VQA, OCR, referring)

LLaVA-1.5 with Vicuna-13B achieves competitive performance with Flamingo-80B on most benchmarks while being orders of magnitude smaller.

### LLaVA-Next (LLaVA-1.6)

Key additions:
- **Dynamic resolution (AnyRes)**: divide high-resolution images into tiles (up to 4 tiles × 336px + 1 thumbnail), process each separately with CLIP, concatenate patch tokens. Enables effective resolution up to 1344×672
- **Multi-image support**: context window can accommodate multiple images
- **Better reasoning data**: more diverse instruction-following data

---

## Proprietary Natively Multimodal Models

### GPT-4V and GPT-4o

OpenAI's GPT-4V (2023) and GPT-4o (2024) are the most capable proprietary VLMs.

GPT-4V added vision to the GPT-4 text model via post-hoc integration (details undisclosed). Capabilities: chart understanding, OCR, spatial reasoning, multi-image comparison, GUI interaction.

GPT-4o ("omni") is **natively multimodal**: trained from the beginning to handle text, images, and audio in a unified architecture. Key advance: end-to-end audio handling without a separate ASR step, enabling low-latency voice conversation (~320ms response time). The model processes raw audio tokens directly, preserving prosody and emotion that is lost in a text transcription intermediate.

GPT-4o also features real-time image understanding during screen sharing and video calls.

### Gemini

Google's Gemini family (Ultra, Pro, Flash, Nano) was designed as **natively multimodal from pretraining**: images, audio, video, and text are interleaved as tokens during pre-training, not added post-hoc. This is architecturally distinct from CLIP-then-LLM designs.

Gemini uses a modified ViT architecture trained jointly with the language model. Images are encoded as sequences of tokens using a learned tokenizer, which are then interleaved with text tokens in the full sequence fed to the Transformer.

Gemini 1.5 Pro introduced a 1M token context window capable of processing hours of video, thousands of images, or entire codebases simultaneously.

---

## Open-Source VLMs: InternVL

**InternVL** (Shanghai AI Lab) is the strongest open-source VLM family as of 2024-2025.

Key design choices:
- **InternViT**: a custom Vision Transformer scaled to 6B parameters (unlike CLIP ViT-L/14 at 307M), pre-trained with contrastive loss on a massive internal dataset
- **Dynamic resolution**: similar to LLaVA-Next's AnyRes tiling
- **InternLM or Qwen backbone**: paired with strong open-source LLMs
- **Strong OCR**: document understanding, chart reading, math problems with equations
- **InternVL2**: further scaling with data quality improvements

InternVL2-40B outperforms GPT-4V on several benchmarks including MMMU and MathVista.

---

## Connecting Images to LLMs: Projection Methods

All VLMs must solve the same fundamental problem: the vision encoder outputs a set of feature vectors, but the LLM expects token embeddings from its vocabulary space. These are different dimensionalities and semantic spaces. How they bridge this gap is the primary architectural choice:

| Method | Used By | Mechanism | Tokens | Strengths | Weaknesses |
|--------|---------|-----------|--------|-----------|------------|
| Linear projection | LLaVA | $W \cdot f_{vis}$, single matrix | All patches (576) | Simple, preserves spatial info | Less expressive alignment |
| MLP connector | LLaVA-1.5 | 2-layer MLP with GELU | All patches (576) | Better feature mapping | Slightly higher compute |
| Q-Former | BLIP-2 | Cross-attention, learned queries | Fixed $N=32$ | Compact, frozen LLM compatible | May lose fine-grained spatial detail |
| Perceiver Resampler | Flamingo | Cross-attention, fixed latents | Fixed 64 | Handles variable resolution | Loses spatial structure |

**Trade-off**: Q-Former and Perceiver Resampler compress heavily (576→32 or 64 tokens), reducing LLM context pressure but potentially losing spatial information. Linear/MLP connectors preserve all patch tokens (576+), maintaining spatial structure at the cost of more LLM context.

**Why does the connector matter?**

The vision encoder (CLIP) was trained to align image embeddings with text embeddings of captions — a global semantic alignment. It was not trained to align with the specific token embedding space of LLaMA or Mistral. The connector must learn this non-trivial mapping. A more expressive connector (MLP vs. linear) learns a better-calibrated mapping, which is why LLaVA-1.5's MLP outperforms LLaVA's linear projection substantially on VQA benchmarks.

---

## Visual Token Count and Context Window Pressure

The choice of vision encoder resolution has direct implications for context window usage:

| Setup | Tokens per Image |
|-------|-----------------|
| CLIP ViT-B/32 @ 224px | 49 tokens |
| CLIP ViT-L/14 @ 224px | 256 tokens |
| CLIP ViT-L/14 @ 336px | 576 tokens |
| LLaVA-Next: 4 tiles × 576 + 1 thumbnail | 2,304 + 256 = 2,560 tokens |

With a 4096-token context window and 576 tokens per image, a single image consumes 14% of the context budget. With dynamic tiling at 2,560 tokens per image, multi-image inputs quickly saturate typical context limits.

This motivates two directions:
1. **Longer context windows** (LLaVA-HD, Gemini 1.5 with 1M tokens)
2. **Token compression** (Q-Former style reduction, token merging via [FastV](https://arxiv.org/abs/2403.06764), etc.)

**Token merging intuition**: adjacent image patches in ViT are highly redundant (especially in flat background regions). Methods like FastV or TokenPacker prune or merge visual tokens based on attention weights, reducing 576 tokens to ~144 with minimal accuracy loss.

---

## High-Resolution Images: Dynamic Tiling

Low-resolution images miss fine-grained detail critical for OCR, chart reading, and detailed scene understanding. The standard solution is **dynamic tiling (AnyRes)**:

```
High-resolution image (e.g. 1344 × 672)
         ↓
Divide into M tiles that fit the encoder's native resolution (336×336)
         ↓
Process each tile independently with CLIP ViT-L/14 → 576 tokens each
         ↓
Also process a downsampled thumbnail → 576 tokens (global context)
         ↓
Concatenate: M × 576 + 576 tokens → feed to LLM
```

For a 1344×672 image divided into 4 tiles: $4 \times 576 + 576 = 2,880$ tokens.

**Why the thumbnail?** Individual tiles lose global context (e.g., a tile showing "43" in a chart is uninterpretable without knowing it's the y-axis label). The thumbnail provides the whole-image semantic context while tiles provide local detail.

**Aspect ratio adaptation**: tile layout adapts to image aspect ratio. Portrait images get a vertical stack; landscape images get a horizontal strip. This avoids aggressive cropping that would discard content.

---

## Benchmarks

| Benchmark | Task | What It Tests |
|-----------|------|---------------|
| **VQAv2** | Open-ended VQA | Broad visual understanding, commonsense |
| **GQA** | Compositional VQA | Spatial relations, logical reasoning |
| **TextVQA** | VQA on images with text | OCR + reasoning |
| **MMMU** | College-level multi-discipline | Expert-level vision+knowledge reasoning |
| **MMBench** | 20 sub-task VQA | Holistic VLM evaluation, bilingual |
| **ScienceQA** | Science multiple-choice | Scientific visual reasoning, mostly diagrams |
| **POPE** | Object hallucination | Whether VLMs hallucinate absent objects |
| **MMStar** | Contamination-aware | Challenges memorized answers |
| **OCRBench** | Document, chart, text understanding | OCR, table parsing, math formulas |

MMMU is currently the most discriminative benchmark: it requires integrating domain knowledge (medicine, law, engineering) with visual reasoning, testing whether models truly understand versus pattern-match.

POPE (Polling-based Object Probing Evaluation) tests a critical failure mode: VLMs often hallucinate objects that are plausible but absent. POPE asks simple yes/no questions ("Is there a chair?") using carefully chosen absent objects.

---

## Code Example

```python
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import requests
import torch

# Load LLaVA-Next (LLaVA-1.6 with Mistral-7B backbone)
model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
processor = LlavaNextProcessor.from_pretrained(model_id)
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Load an image
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)

# Prepare inputs using chat template
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image in detail. What is the animal doing?"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

# Generate
with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
    )

# Decode (skip input tokens)
response = processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)

# --- CLIP zero-shot classification example ---
from transformers import CLIPProcessor, CLIPModel

clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Zero-shot classification
candidate_labels = ["a cat sleeping", "a dog playing", "a bird flying"]
inputs = clip_processor(
    text=candidate_labels,
    images=image,
    return_tensors="pt",
    padding=True
)

with torch.no_grad():
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image  # shape: [1, num_labels]
    probs = logits_per_image.softmax(dim=-1)

for label, prob in zip(candidate_labels, probs[0]):
    print(f"{label}: {prob:.3f}")
```

---

## References

- Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). OpenAI. https://arxiv.org/abs/2103.00020
- Li et al. (2022). *BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation*. https://arxiv.org/abs/2201.12086
- Li et al. (2023). *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models*. https://arxiv.org/abs/2301.12597
- Alayrac et al. (2022). *Flamingo: a Visual Language Model for Few-Shot Learning*. DeepMind. https://arxiv.org/abs/2204.14198
- Liu et al. (2023). *Visual Instruction Tuning* (LLaVA). https://arxiv.org/abs/2304.08485
- Liu et al. (2023). *Improved Baselines with Visual Instruction Tuning* (LLaVA-1.5). https://arxiv.org/abs/2310.03744
- Liu et al. (2024). *LLaVA-NeXT: Improved reasoning, OCR, and world knowledge*. https://llava-vl.github.io/blog/2024-01-30-llava-next/
- Chen et al. (2024). *InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks*. https://arxiv.org/abs/2312.14238
- Team, G. (2023). *Gemini: A Family of Highly Capable Multimodal Models*. Google DeepMind. https://arxiv.org/abs/2312.11805
- OpenAI (2023). *GPT-4V(ision) System Card*. https://openai.com/research/gpt-4v-system-card

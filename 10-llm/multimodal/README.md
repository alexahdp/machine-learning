# Multimodal LLMs

Language models began as purely textual systems: tokens in, tokens out. Multimodal LLMs dissolve that boundary. They extend the same Transformer machinery that processes language to handle images, audio, video, and other modalities — enabling visual question answering, image generation from text, speech understanding, and cross-modal reasoning in a unified framework.

The core insight is architectural: different modalities can be projected into the same embedding space that language models already operate in. Once an image or audio clip is a sequence of vectors in that space, the rest of the model proceeds exactly as for text. The challenge lies in the projection: how do you faithfully compress the 576 patch tokens from a ViT-L/14 encoder, or a 30-second audio spectrogram, into a representation that the LLM's attention layers can work with?

This section covers the three dominant modality pairs: vision+language (by far the most mature), text-to-image generation (a separate generative paradigm), and audio+language (the fastest-growing area, driven by voice interfaces).

## Contents

| File | Description |
|------|-------------|
| [vision-language-models.md](./vision-language-models.md) | CLIP, BLIP-2, Flamingo, LLaVA, GPT-4V, Gemini — connecting image encoders to LLMs, visual instruction tuning, benchmark landscape |
| [image-generation.md](./image-generation.md) | Diffusion models, latent diffusion, DALL-E, Stable Diffusion, FLUX, fine-tuning techniques (DreamBooth, ControlNet, LoRA), evaluation metrics |
| [audio-language-models.md](./audio-language-models.md) | Audio representations, Whisper ASR, audio tokenization (EnCodec, AudioLM), speech LLMs, TTS (VALL-E, Voicebox), music generation |

## Navigation

- [← Back to LLM Overview](../README.md)
- [← Previous: Agents](../agents/README.md)

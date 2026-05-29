# Audio-Language Models

## Table of Contents

1. [Audio Representations](#audio-representations)
2. [Speech Recognition Background](#speech-recognition-background)
3. [Whisper](#whisper)
4. [Audio Tokenization](#audio-tokenization)
5. [Speech LLMs](#speech-llms)
6. [Text-to-Speech](#text-to-speech)
7. [Voice Interfaces: GPT-4o Audio](#voice-interfaces-gpt-4o-audio)
8. [Music Generation](#music-generation)
9. [Benchmarks](#benchmarks)
10. [Key Challenges](#key-challenges)
11. [Code Example](#code-example)
12. [References](#references)

---

## Audio Representations

Audio is a continuous pressure wave — a 1D signal sampled at discrete time steps. A 1-second clip at 16kHz sampling rate is 16,000 floating-point values. At 44.1kHz (CD quality): 44,100 values. Working directly with raw waveforms is possible but computationally expensive and difficult because the model must discover structure at multiple timescales simultaneously (phonemes: ~50ms; words: ~300ms; sentences: seconds).

Different representations trade dimensionality for interpretability and task-suitability:

### Raw Waveform

The raw time-domain signal $x[n]$ where each sample is an amplitude in $[-1, 1]$.

- **Dimensionality**: $F_s \cdot T$ (sampling rate × duration) — very high
- **Used by**: WaveNet, EnCodec, wav2vec 2.0 (input only)
- **Advantage**: lossless, no information discarded
- **Disadvantage**: long sequences; model must learn multi-scale structure from scratch; no localized frequency information

### Mel Spectrogram

The standard input for most audio models. Computation pipeline:

```
Raw waveform x[n]
      ↓ Short-Time Fourier Transform (STFT)
      |  Window: Hamming, 25ms, stride 10ms
      |  FFT size: 512 or 1024
      ↓ Power Spectrogram |STFT(x)|²  [F_bins × T_frames]
      ↓ Mel Filterbank (M triangular filters on mel scale)
      |  Mel scale: m = 2595 · log₁₀(1 + f/700)
      |  Compresses high frequencies (perceptual weighting)
      ↓ Log compression: log(mel + 1e-8)
      ↓ Log-Mel Spectrogram [M × T_frames], M=80 or 128
```

The **mel scale** approximates human auditory frequency perception: humans are more sensitive to differences at low frequencies than high frequencies. Equal steps on the mel scale correspond to equal perceived pitch differences.

- **Dimensionality**: $M \times T$, typically $80 \times 3000$ for 30 seconds at 16kHz
- **Standard for**: Whisper (80-dim), most ASR systems
- **Advantage**: compact, perceptually meaningful, models learn features quickly
- **Disadvantage**: lossy (phase discarded), non-invertible without Griffin-Lim or neural vocoders

### MFCC (Mel-Frequency Cepstral Coefficients)

Apply DCT to log-mel filterbank outputs, keep the first $C$ coefficients (typically $C=13$):

$$\text{MFCC}[c] = \sum_{m=1}^{M} \log S[m] \cdot \cos\left(\frac{\pi c (m - 0.5)}{M}\right)$$

MFCCs decorrelate the filterbank outputs (filterbank channels are highly correlated; DCT diagonalizes this). They were the dominant feature for classical ASR (GMM-HMM systems) and speaker recognition.

- **Dimensionality**: very compact ($13 \times T$)
- **Used by**: GMM-HMM ASR, speaker verification
- **Largely superseded by** learned features (wav2vec, HuBERT) for deep learning tasks

### Learned Representations: wav2vec 2.0 and HuBERT

Self-supervised approaches learn features directly from raw audio:

**wav2vec 2.0** (Baevski et al., 2020): 
- CNN encoder extracts local features from raw waveform
- Transformer contextualizes over time
- Contrastive loss: predict discretized representations at masked timesteps
- Produces rich contextual audio features used downstream for ASR, speaker verification, emotion

**HuBERT** (Hsu et al., 2021):
- CNN encoder + Transformer (similar structure)
- Training: cluster offline (k-means on MFCC or prior HuBERT features) → use cluster IDs as pseudo-labels → train with masked prediction loss (like BERT for audio)
- Produces semantic audio tokens (phoneme-level content, ignoring speaker/noise) — critical for AudioLM

---

## Speech Recognition Background

Automatic Speech Recognition (ASR) maps audio to text. The core challenge is the **alignment problem**: audio and text are different lengths and the correspondence is unknown.

### CTC (Connectionist Temporal Classification)

CTC (Graves et al., 2006) introduces a special blank token `<b>` and defines a distribution over all possible frame-level label sequences that collapse to the target transcript:

$$P(y | x) = \sum_{\pi \in \mathcal{B}^{-1}(y)} \prod_t P(\pi_t | x)$$

where $\mathcal{B}$ collapses repeated characters and blank tokens. CTC allows the model to output characters at every frame without knowing when each character occurs, using dynamic programming (forward-backward algorithm) for training.

**Advantage**: monotonic alignment, efficient, no external language model required.
**Disadvantage**: conditional independence assumption (each frame output is independent of others given input) limits language model integration.

### Attention-Based Seq2Seq

Encoder-decoder architectures (LAS, Chorowski et al., 2015): the encoder produces a sequence of audio representations; the decoder generates text tokens autoregressively via cross-attention over encoder outputs. No conditional independence assumption — the decoder can attend to any part of the audio at each step. Requires paired audio-text data and careful handling of long audio sequences.

### External Language Model Integration

Both CTC and seq2seq benefit from LM integration (shallow fusion):

$$\log P(y | x) = \log P_{ASR}(y | x) + \lambda \log P_{LM}(y)$$

Adding a character or word LM improves WER on out-of-domain text, but Whisper's multitask training largely internalizes LM knowledge, reducing the need for explicit external LMs.

---

## Whisper

**Paper**: Radford et al., OpenAI, 2022. *Robust Speech Recognition via Large-Scale Weak Supervision.*

Whisper is arguably the most important ASR development since the deep learning revolution. Its insight: **scale weak supervision**. Rather than carefully curating a small labeled dataset, collect 680,000 hours of audio from the internet paired with whatever text transcripts exist (subtitles, captions, human transcriptions) — even if the alignment is imperfect.

### Architecture

Whisper is a standard **Transformer encoder-decoder**, with no architectural novelty. The model is not SOTA because of a new architecture; it's SOTA because of the data.

```
30-second audio chunk
      ↓ Resample to 16kHz
      ↓ Log-Mel Spectrogram (80-dim, 25ms window, 10ms stride)
      → [3000 × 80] input tensor
      ↓
  Conv1d(80, d_model, kernel=3, stride=1) → ReLU
  Conv1d(d_model, d_model, kernel=3, stride=2) → ReLU
  (subsamples time by 2x → 1500 frames)
      ↓
  Sinusoidal positional encoding
      ↓
  Transformer Encoder (N layers, d_model, h heads)
      ↓ audio representations
  Transformer Decoder (N layers)
  ← text token embeddings (autoregressive)
      ↓
  Text tokens output
```

The convolutional frontend serves as a local feature extractor before the global attention in the Transformer encoder. Stride=2 in the second conv halves the sequence length to 1500 frames.

### Multitask via Special Tokens

Whisper is a single model handling multiple tasks, controlled entirely by prompt tokens prepended to the decoder:

```
[<|startoftranscript|>] [<|en|>] [<|transcribe|>] [<|notimestamps|>]
    ↓                       ↓           ↓                   ↓
  begin            language ID    task: transcribe      no timestamps
                   (one of 99)    vs. <|translate|>
```

This elegant formulation means:
- **Language ID**: the model simultaneously identifies the spoken language
- **Translation**: conditioning on `<|translate|>` makes Whisper translate speech directly to English text, without a text intermediate in the source language
- **Timestamps**: conditioning on `<|timestamps|>` produces `<|t_start|>` ... `<|t_end|>` tokens interleaved with transcript words

### Model Sizes

| Model | Parameters | Layers | d_model | Heads | Speed (relative) |
|-------|-----------|--------|---------|-------|-----------------|
| tiny | 39M | 4+4 | 384 | 6 | ~32× |
| base | 74M | 6+6 | 512 | 8 | ~16× |
| small | 244M | 12+12 | 768 | 12 | ~6× |
| medium | 769M | 24+24 | 1024 | 16 | ~2× |
| large-v3 | 1,550M | 32+32 | 1280 | 20 | 1× |

Speed is relative to large-v3. tiny.en (English-only) is fast enough to run in real-time on CPU.

### Key Properties and Limitations

**Strengths**:
- Robust to accents, background noise, music, and non-native speakers
- Handles 99 languages without separate per-language models
- Zero-shot — no fine-tuning required for new domains (usually)
- Timestamps enable downstream speaker diarization, alignment, subtitle generation

**Limitations**:
- **Hallucination on silence**: Whisper hallucinates plausible-sounding text when given silent audio or unintelligible noise. This is a consequence of the encoder-decoder architecture and the autoregressive decoder: given silence, the decoder must still produce tokens, and it generates its "prior" about what transcripts look like.
- **Repetition loops**: on long audio, the decoder occasionally falls into repetitive loops
- **Fixed 30-second window**: Whisper processes exactly 30 seconds; shorter audio is zero-padded, longer audio requires chunking

### Long Audio Handling

For audio longer than 30 seconds:
1. **Chunk with overlap**: split into 30s windows with 5s overlap; merge at silence boundaries
2. **VAD (Voice Activity Detection) preprocessing**: detect speech segments, skip silence, feed only voiced segments to Whisper. Reduces hallucination on silence dramatically.
3. **Word-timestamp-based alignment**: use Whisper's timestamps to stitch chunks without duplicating words

Libraries like `faster-whisper` and `whisperX` implement efficient chunking with VAD and forced alignment.

---

## Audio Tokenization

To use audio with generative models (language model-style generation), audio must be converted to **discrete tokens** — the audio equivalent of BPE tokens for text. This is the job of neural audio codecs.

### Why Discrete Tokens?

A language model operates over a discrete vocabulary. To generate audio autoregressively with an LLM (like generating text), you need to represent audio as sequences of integers from a finite codebook. The codec compresses audio to tokens; the decoder synthesizes audio from tokens.

### EnCodec

**Paper**: Défossez et al., Meta, 2022.

EnCodec is a neural audio codec trained with a combination of spectral reconstruction loss and adversarial losses (discriminators operating in time and frequency domains).

**Architecture:**
```
Audio waveform (24kHz)
      ↓
  Encoder (convolutional, 1D) → compressed latent
  (4x, 5x, 8x, 16x downsampling strides → 75Hz frame rate)
      ↓
  RVQ: Residual Vector Quantization
      ↓
  Q₁ quantizes latent → residual r₁ = z - z_q1
  Q₂ quantizes r₁   → residual r₂ = r₁ - z_q2
  ...
  Q₈ quantizes r₇   → residual r₈
      ↓
  8 codebooks × 1024 entries each
  → 8 discrete codes per time frame, at 75 Hz
  → bit rate: 75 × 8 × log₂(1024) = 6,000 bits/second ≈ 6 kbps
      ↓
  Decoder (convolutional) → reconstructed waveform
```

**Residual Vector Quantization (RVQ)**: instead of quantizing the entire latent in one step (which would require a huge codebook), quantize the residual error at each step. The first codebook captures the coarse structure; later codebooks refine progressively finer details. This hierarchy is crucial: the first codebook captures phoneme identity (semantic content), while later codebooks encode speaker voice, prosody, and acoustic detail.

**At 6kbps**: EnCodec compresses 24kHz audio 75× compared to uncompressed PCM. At 24kbps (8 codebooks × 3kbps each), reconstruction quality approaches CD quality.

### DAC (Descript Audio Codec)

**Paper**: Kumar et al., 2023.

DAC improves on EnCodec by:
- Codebook utilization: EnCodec codes collapse (many entries unused); DAC uses **random projection codebook initialization** ensuring all entries are used
- **Improved discriminators**: multi-band, multi-scale STFT discriminators
- **Higher quality at same bitrate**: competitive with Opus (a classical codec) at much lower bitrates
- Supports 44.1kHz (music quality) in addition to 16kHz and 24kHz

### AudioLM: Hierarchical Audio Generation

**Paper**: Borsos et al., Google, 2022.

AudioLM generates speech by combining two token types in a two-stage hierarchy:

**Stage 1 — Semantic tokens** (HuBERT discrete units):
- Encode phoneme-level content (what is said), speaker-independent
- 25Hz token rate, 500-token vocabulary from k-means on HuBERT features
- Stage 1 model: language model trained to predict semantic tokens autoregressively

**Stage 2 — Acoustic tokens** (SoundStream/EnCodec RVQ tokens):
- Encode speaker voice, prosody, acoustic environment (how it's said)
- 75Hz × 8 codebooks per time step
- Stage 2 model: conditioned on semantic tokens, generates acoustic tokens

```
Prompt audio → Semantic Tokens  → [Stage 1 LM] → Predicted Semantic Tokens
                                                           ↓
Prompt audio → Acoustic Tokens  → [Stage 2 LM: conditioned on semantic] → Predicted Acoustic Tokens
                                                           ↓
                                              Neural Codec Decoder
                                                           ↓
                                                   Synthesized audio
```

The hierarchy solves a fundamental problem: modeling 8 RVQ codebooks at 75Hz autoregressively would require an LM over $(75 \times 8) = 600$ tokens per second — for 10 seconds of audio that's 6,000 tokens. AudioLM delegates semantic structure to the first stage (manageable sequence length) and acoustic fine-grained detail to the second stage (conditioned, so easier).

### SoundStream vs. EnCodec vs. DAC

| Codec | Sample Rate | Codebooks | Frame Rate | Bit Rate | Strength |
|-------|------------|-----------|-----------|----------|----------|
| SoundStream | 24kHz | 8 | 75Hz | 3–18kbps | First RVQ neural codec |
| EnCodec | 24kHz | 8 | 75Hz | 1.5–24kbps | Open weights, widely adopted |
| DAC | 44.1kHz | 12 | 86Hz | ~8kbps | Best quality, codebook utilization |

All three use RVQ. The main differences are the discriminator design, codebook utilization, and supported sample rates.

---

## Speech LLMs

### SpeechGPT

**Paper**: Zhang et al., 2023.

SpeechGPT is the first model to perform **direct speech-to-speech dialogue** — the model takes speech tokens as input and generates speech tokens as output, without any text intermediate.

Pipeline:
1. Discretize speech: apply HuBERT k-means to get discrete speech units
2. Pretrain LLaMA on a mixture of: text tokens + speech tokens (interleaved)
3. Instruction-tune on speech-in/speech-out tasks: ASR, TTS, spoken QA, etc.

The model learns a shared vocabulary spanning both modalities. It can follow spoken instructions and respond in speech — crucially, preserving prosody and emotional tone that is lost when routing through text.

**Limitation**: poor audio quality (HuBERT tokens are semantic, not acoustic; synthesis from HuBERT units requires a separate vocoder), and the model is not truly end-to-end (the HuBERT tokenizer is a frozen pre-processing step).

### AudioPaLM

**Paper**: Rubenstein et al., Google, 2023.

AudioPaLM merges the audio tokenizer of AudioLM with the text backbone of PaLM-2. The model operates over a combined vocabulary of text tokens (256K BPE) and audio tokens (SoundStream). It handles:
- Speech recognition (speech → text)
- Text-to-speech (text → speech)
- Speech-to-speech translation
- Spoken question answering

Fine-tuned jointly on all tasks, AudioPaLM achieves SOTA on FLEURS (multilingual ASR) and outperforms Whisper on many languages while also being capable of TTS.

### Llama 3 Audio

Meta's Llama 3 includes native audio capabilities via a separate audio encoder (similar to Whisper's encoder) whose outputs are projected into Llama 3's embedding space. This follows the VLM connector pattern: audio encoder → projection layer → LLM. Unlike SpeechGPT's end-to-end token approach, the audio is not generated natively — outputs are text, with TTS applied post-hoc for spoken responses.

---

## Text-to-Speech

TTS has evolved from signal-processing pipelines (concatenative synthesis, formant synthesis) to deep learning approaches that achieve human-parity voice quality.

### Neural TTS Evolution

**Tacotron 2** (Wang et al., 2017): Seq2Seq with attention. Text → mel spectrogram (via encoder + attention decoder), then WaveNet vocoder → waveform. Two-stage pipeline; slow inference due to WaveNet.

**FastSpeech 2** (Ren et al., 2020): Parallel TTS — predicts mel spectrogram in a single forward pass (no autoregression). Duration predictor aligned to phonemes via forced alignment; pitch and energy predictors for expressiveness. ~38× faster than Tacotron 2 at inference.

**VITS** (Kim et al., 2021): End-to-end TTS using Variational Inference with adversarial Training. Combines VAE (latent variable for speaking style) + normalizing flows (invertible transforms) + adversarial training. The first single-stage TTS to match two-stage quality.

### VALL-E (Zero-Shot TTS from 3-Second Speaker Sample)

**Paper**: Wang et al., Microsoft, 2023.

VALL-E reframes TTS as a **language modeling problem over codec tokens**:

Given:
- Text transcript
- 3-second audio prompt from a target speaker

Generate: codec tokens (EnCodec, 8 codebooks at 75Hz) that sound like the target speaker reading the transcript.

**Architecture**: Two-stage codec language model:
1. **AR model**: autoregressively generate first EnCodec codebook tokens conditioned on text + acoustic prompt tokens
2. **NAR model**: non-autoregressively generate codebooks 2–8 conditioned on codebook 1 + text + prompt

```
Text:    "The quick brown fox"
Prompt:  [3s audio from target speaker] → EnCodec → [C₁, C₂, ..., C₈] (speaker prompt tokens)

AR LM:   P(C₁_generated | C₁_prompt, text) — autoregressive over time
NAR LM:  P(Cₖ_generated | C₁_generated, ..., Cₖ₋₁_generated, Cₖ_prompt, text), k=2..8
          — parallel over time, conditioned on previous codebooks
```

This zero-shot speaker cloning works because the model has learned to disentangle content (from text) from speaker voice (from the acoustic prompt). VALL-E can preserve emotion and speaking style from the prompt.

**VALL-E 2** (Chen et al., 2024): adds repetition-aware sampling (reduces repetition loops) and grouped code modeling (speeds up inference). First TTS to achieve human parity on VCTK benchmark.

### Voicebox (Meta AI, 2023)

**Paper**: Le et al., Meta, 2023.

Voicebox uses **flow matching** (the same approach as FLUX for images) for speech synthesis. It is a non-autoregressive diffusion model trained on 50,000 hours of speech.

Key capability: **in-context generation** — given surrounding audio context, generate a speech segment that seamlessly fills a gap (useful for audio editing: replacing a word without re-recording the whole sentence).

Voicebox achieves SOTA WER and speaker similarity on zero-shot TTS while being 20× faster than VALL-E at inference.

### Bark

An open-source, permissively-licensed TTS model (Suno AI). Uniquely handles:
- Laughing, sighing, crying, sound effects embedded in speech
- Music generation (though limited quality)
- Non-speech sounds: "[clears throat]", "[music]"
- 100+ languages and accents

Bark is a hierarchical autoregressive model over semantic tokens (from a text-conditioned GPT-like model) then acoustic tokens (from an EnCodec-like codec).

---

## Voice Interfaces: GPT-4o Audio

GPT-4o's audio capability (released 2024) represents a qualitative shift in voice interfaces. Prior approaches:

```
[Traditional pipeline]
User speech → ASR (Whisper) → text → LLM → text → TTS → speech response
```

Problems: latency (~1–2s for ASR + inference + TTS), emotional information discarded at ASR step, prosody in TTS is generic and doesn't match response emotion.

GPT-4o audio processes audio as native tokens (similar to SpeechGPT but at production scale):

```
[GPT-4o end-to-end]
User speech tokens → GPT-4o → speech response tokens → waveform
```

Measured latency: ~230ms mean, ~320ms median — comparable to human conversational response time. The model generates speech output directly, so it can:
- Match response emotion to content ("Good news! Your package arrived." said with warmth)
- Interrupt and adjust based on vocal cues (sighing, hesitation)
- Preserve regional accents and speaking style in multi-turn conversation

The technical details of GPT-4o's audio tokenization are undisclosed, but from the paper, it uses custom tokenizers separate from text BPE, trained with a combination of ASR, TTS, and conversational objectives.

---

## Music Generation

Music generation is harder than speech: music involves rhythm, harmony, melody, instrument timbre, and structure at multiple timescales.

### MusicGen (Meta AI, 2023)

**Paper**: Copet et al., 2023.

MusicGen is an autoregressive language model over EnCodec tokens (4 codebooks at 50Hz), conditioned on text descriptions. Key innovation: **efficient multi-stream token modeling**.

Instead of generating 4 codebooks sequentially (which quadruples sequence length), MusicGen uses a **delay pattern**: codebook $k$ is delayed by $k$ timesteps relative to codebook 1, then all codebooks at a given time step can be predicted in parallel within a single Transformer forward pass.

```
Time:     t=1   t=2   t=3   t=4
CB1:      C1₁   C1₂   C1₃   C1₄
CB2:      [pad] C2₁   C2₂   C2₃
CB3:      [pad] [pad] C3₁   C3₂
CB4:      [pad] [pad] [pad] C4₁

At t=4: predict C1₄, C2₃, C3₂, C4₁ simultaneously
```

Conditioning: text is encoded with a frozen T5-XXL encoder and enters via cross-attention.

MusicGen supports:
- Unconditional generation
- Text-conditioned generation: "upbeat jazz with piano and bass, 120 BPM"
- Melody conditioning: provide a hummed melody, generate music following it

### MusicLM (Google, 2023)

**Paper**: Agostinelli et al., 2023.

MusicLM is a hierarchical sequence-to-sequence model that uses:
- **MuLan** text-music embedding (analogous to CLIP, trained on 370K music+description pairs) for text conditioning
- **SoundStream** RVQ tokens as generation targets

Three hierarchical models: semantic (low-rate, 3Hz), coarse acoustic (25Hz, top 4 codebooks), fine acoustic (25Hz, all codebooks).

**Story generation**: given a sequence of text descriptions at different timestamps ("a calm forest scene... then a thunderstorm approaches..."), MusicLM generates music that evolves accordingly.

### AudioCraft (Meta, 2023)

Meta's umbrella of generative audio models, combining:
- **MusicGen**: music generation
- **AudioGen**: sound effect generation ("rain on a tin roof", "dog barking in distance")
- **EnCodec**: shared neural codec

All open-sourced under MIT license.

---

## Benchmarks

### ASR Benchmarks

**LibriSpeech**: 1000 hours of English audiobooks read by volunteers. Standard benchmark with `test-clean` (professional readers, clean audio) and `test-other` (accented, harder). WER measured; Whisper large-v3: 2.7% on test-clean.

**FLEURS** (Few-shot Learning Evaluation of Universal Representations of Speech): 102 languages, 10-12 hours per language, parallel across all languages. Primary benchmark for multilingual ASR. Whisper competitive; SeamlessM4T (Meta) strong.

**Common Voice** (Mozilla): crowd-sourced, 100+ languages, highly variable quality. Tests robustness to diverse speakers.

**AESRC2020**: accented English from 8 accent groups (Indian, Chinese, British, etc.).

### Comprehensive Audio Benchmarks

**SUPERB** (Speech processing Universal PERformance Benchmark): 13 tasks including ASR, speaker identification, intent classification, emotion recognition, keyword spotting — all from a shared self-supervised representation. Measures how well a frozen pretrained encoder generalizes.

**AIR-Bench** (Audio Instruct-Response Benchmark): evaluates audio LLMs on open-ended audio understanding (environmental sounds, music, speech) with LLM-as-judge evaluation.

### TTS Benchmarks

**VCTK**: 109 English speakers with different accents. WER on ASR of generated speech + speaker similarity (cosine similarity of speaker embeddings). Human parity: ~5% WER, >0.93 speaker similarity.

**AudioCaps**: 46K audio clips with 5 human-written captions each. Used for audio captioning evaluation (METEOR, SPIDEr metrics).

---

## Key Challenges

### Speaker Diarization

"Who spoke when?" — segmenting an audio stream by speaker identity. Required for:
- Meeting transcription (multi-speaker, overlapping speech)
- Podcast transcription
- Legal recordings

Classical approaches: spectral clustering over speaker embeddings (x-vectors, d-vectors). Recent approaches: end-to-end neural diarization (EEND) with self-attention, handling overlapping speech. Whisper does not natively diarize; pyannote.audio provides state-of-the-art neural diarization that combines with Whisper (whisperX pipeline).

### Real-Time Processing

Streaming ASR requires processing audio as it arrives, not in complete 30-second chunks:
- **Chunk-and-process**: 2–5s chunks with overlap; introduces latency
- **CTC streaming**: CTC models can emit characters as they become confident, enabling low-latency streaming
- **End-of-utterance detection**: must detect when the speaker has finished before responding

### Emotional Prosody

Current models separate content and style (speaker identity) but struggle with fine-grained emotion. "I love Mondays" said sarcastically vs. sincerely requires tonal understanding beyond phoneme sequences. Self-supervised representations (wav2vec, HuBERT) do capture some emotional information (shown on emotion recognition benchmarks), but injecting emotional intent into TTS remains an open problem.

### Low-Resource Languages

Of the world's ~7,000 languages, ASR exists for perhaps 100. Low-resource languages lack training data. Approaches:
- **Multilingual transfer**: train on high-resource languages, fine-tune on low-resource
- **Cross-lingual transfer**: shared phoneme inventory across languages
- **Data synthesis**: TTS for low-resource languages using speech from diaspora communities
- **Whisper's weak supervision**: effective because internet audio in many languages has subtitles

### Noise Robustness

Real-world audio is never clean: background noise, room reverberation, microphone quality, network compression artifacts (codec noise in phone calls). Whisper's training on diverse web audio (rather than clean studio recordings) is why it is more robust than contemporaneous models trained on curated datasets.

---

## Code Example

```python
# Whisper ASR — transcribe audio with timestamps
import torch
import whisper

model = whisper.load_model("large-v3")  # or "base", "medium", etc.

# Basic transcription
result = model.transcribe(
    "audio.mp3",
    language="en",          # optional: auto-detect if omitted
    task="transcribe",      # or "translate" for English output from any language
    verbose=True,
)
print(result["text"])

# With word timestamps (requires alignment)
result = model.transcribe(
    "audio.mp3",
    word_timestamps=True,
    condition_on_previous_text=False,  # reduces hallucination loops
)
for segment in result["segments"]:
    print(f"[{segment['start']:.2f}s → {segment['end']:.2f}s]: {segment['text']}")

# --- faster-whisper: CTranslate2-optimized, 4× faster with same accuracy ---
from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cuda", compute_type="float16")
segments, info = model.transcribe(
    "audio.mp3",
    vad_filter=True,                # Voice Activity Detection — kills hallucination on silence
    vad_parameters={"min_silence_duration_ms": 500},
    beam_size=5,
)
print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
for segment in segments:
    print(f"[{segment.start:.2f}s → {segment.end:.2f}s] {segment.text}")

# --- VALL-E-style zero-shot TTS with Bark ---
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

preload_models()

# Speaker presets: see bark/assets/prompts/
# v2/en_speaker_6: American male, v2/en_speaker_9: American female
text_prompt = """
    Hello! My name is [clears throat] Suno. I'm going to tell you about
    diffusion models. [laughs] They're actually quite elegant when you think about it.
"""
audio_array = generate_audio(text_prompt, history_prompt="v2/en_speaker_6")
write_wav("bark_output.wav", SAMPLE_RATE, audio_array)

# --- EnCodec: audio tokenization/detokenization ---
from encodec import EncodecModel
from encodec.utils import convert_audio
import torchaudio

# Load model
encodec_model = EncodecModel.encodec_model_24khz()
encodec_model.set_target_bandwidth(6.0)  # 6kbps = 8 codebooks × 75Hz
encodec_model.eval()

# Load and convert audio
wav, sr = torchaudio.load("audio.wav")
wav = convert_audio(wav, sr, encodec_model.sample_rate, encodec_model.channels)
wav = wav.unsqueeze(0)  # [1, C, T]

# Encode to discrete tokens
with torch.no_grad():
    encoded_frames = encodec_model.encode(wav)

# encoded_frames: list of (codes, scale) tuples
# codes.shape: [1, n_codebooks, T_encoded]
codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
print(f"Audio shape: {wav.shape}")         # [1, 1, 72000] for 3s at 24kHz
print(f"Codes shape: {codes.shape}")       # [1, 8, 225] for 3s (75Hz × 3s = 225 frames)

# Decode back to audio
with torch.no_grad():
    reconstructed = encodec_model.decode(encoded_frames)
torchaudio.save("reconstructed.wav", reconstructed[0].cpu(), encodec_model.sample_rate)

# --- MusicGen: text-to-music generation ---
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained("facebook/musicgen-large")
model.set_generation_params(
    duration=30,    # seconds
    temperature=1.0,
    top_k=250,
)

descriptions = [
    "upbeat jazz piano trio, walking bass line, swing rhythm, 120 BPM",
    "ambient electronic, slow evolving pads, minor key, introspective",
]
wav = model.generate(descriptions)  # [2, 1, T] at 32kHz

for i, (description, one_wav) in enumerate(zip(descriptions, wav)):
    audio_write(
        f"music_{i}",
        one_wav.cpu(),
        model.sample_rate,
        strategy="loudness",    # normalize loudness
    )
```

---

## References

- Radford et al. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper). OpenAI. https://arxiv.org/abs/2212.04356
- Défossez et al. (2022). *High Fidelity Neural Audio Compression* (EnCodec). Meta. https://arxiv.org/abs/2210.13438
- Kumar et al. (2023). *High-Fidelity Audio Compression with Improved RVQGAN* (DAC). https://arxiv.org/abs/2306.06546
- Borsos et al. (2022). *AudioLM: a Language Modeling Approach to Audio Generation*. Google. https://arxiv.org/abs/2209.03143
- Zhang et al. (2023). *SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities*. https://arxiv.org/abs/2305.11000
- Rubenstein et al. (2023). *AudioPaLM: A Large Language Model That Can Speak and Listen*. Google. https://arxiv.org/abs/2306.12925
- Wang et al. (2023). *VALL-E: Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers*. Microsoft. https://arxiv.org/abs/2301.02111
- Chen et al. (2024). *VALL-E 2: Neural Codec Language Models are Human Parity Zero-Shot Text to Speech Synthesizers*. https://arxiv.org/abs/2406.05370
- Le et al. (2023). *Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale*. Meta. https://arxiv.org/abs/2306.15687
- Copet et al. (2023). *Simple and Controllable Music Generation* (MusicGen). Meta. https://arxiv.org/abs/2306.05284
- Agostinelli et al. (2023). *MusicLM: Generating Music From Text*. Google. https://arxiv.org/abs/2301.11325
- Baevski et al. (2020). *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*. https://arxiv.org/abs/2006.11477
- Hsu et al. (2021). *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units*. https://arxiv.org/abs/2106.07447
- Yang et al. (2021). *SUPERB: Speech Processing Universal PERformance Benchmark*. https://arxiv.org/abs/2105.01051
- Kim et al. (2021). *Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech* (VITS). https://arxiv.org/abs/2106.06103
- OpenAI (2024). *GPT-4o System Card*. https://openai.com/index/gpt-4o-system-card/

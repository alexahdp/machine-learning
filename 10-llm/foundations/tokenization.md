# Tokenization

## Table of Contents

1. [Why Tokenization Exists](#why-tokenization-exists)
2. [Granularity Levels and Trade-offs](#granularity-levels-and-trade-offs)
3. [Byte Pair Encoding (BPE)](#byte-pair-encoding-bpe)
4. [WordPiece](#wordpiece)
5. [SentencePiece](#sentencepiece)
6. [Vocabulary Size and Its Effects](#vocabulary-size-and-its-effects)
7. [Special Tokens](#special-tokens)
8. [Tokenization Artifacts](#tokenization-artifacts)
9. [Token Count vs. Word Count](#token-count-vs-word-count)
10. [Out-of-Vocabulary Handling](#out-of-vocabulary-handling)
11. [Using HuggingFace Tokenizers](#using-huggingface-tokenizers)
12. [References](#references)

---

## Why Tokenization Exists

Neural networks operate on fixed-size numerical tensors. Text is a variable-length sequence of Unicode characters from an open-ended character set. **Tokenization** is the bridge: a bijective mapping from text strings to integer sequences that a model can process.

The central tension is:

- **Vocabulary too small** (e.g., character-level): sequences become very long, increasing computational cost quadratically (attention is $O(n^2)$), and the model must learn composition from scratch.
- **Vocabulary too large** (e.g., word-level): rare words and morphological variants appear infrequently in training, receive poor embeddings, and unseen words are unhandled.

Subword tokenization is the practical resolution: decompose text into units that are smaller than words but larger than characters, letting the vocabulary cover both common words (as single tokens) and rare words (as sequences of subword pieces).

---

## Granularity Levels and Trade-offs

### Character-Level

Every Unicode character (or byte) is a token.

**Vocabulary size**: ~130 (ASCII bytes) to ~143,859 (all Unicode characters).

**Pros**: No OOV problem; handles any text in any language; compact vocabulary.

**Cons**: Sequences are very long (a 500-word document → ~2500 characters); the model must learn all morphology, syntax, and semantics from individual letters; memory/compute scales quadratically with sequence length.

**Used in**: Byte-level models (ByT5 from Xue et al., 2022); character-level language models for research.

### Word-Level

Whitespace (or punctuation) tokenization. Vocabulary is the set of observed words.

**Vocabulary size**: Easily millions in natural text corpora.

**Pros**: Tokens align with human linguistic intuition; short sequences.

**Cons**: Severe OOV problem (any typo, proper noun, or word form not seen in training is unhandled); inflected languages (Finnish, Turkish) explode vocabulary; "running," "runs," "ran" are three unrelated tokens.

**Used in**: Early neural NLP (Word2Vec embeddings), no longer standard for LLM input.

### Subword-Level

Split words into morphologically meaningful pieces, reserving frequently used words as single tokens and decomposing rare words into known subpieces.

**Vocabulary size**: 10k–100k (tunable).

**Pros**: OOV-free for any text (can always fall back to individual characters/bytes); balances sequence length vs. vocabulary coverage; multilingual friendly.

**Used in**: All modern LLMs.

---

## Byte Pair Encoding (BPE)

BPE (Sennrich et al., 2016, originally a compression algorithm) is the most widely used subword algorithm.

### Algorithm

**Input**: Raw text corpus.

**Step 1 — Initialize**: Start with a vocabulary of all individual characters (or bytes) present in the corpus, plus a special end-of-word symbol (e.g., `</w>`). Represent every word as a sequence of its characters.

```
"low low low lower lowest"

Initial:
  l o w </w>        (frequency 3)
  l o w e r </w>    (frequency 1)
  l o w e s t </w>  (frequency 1)
```

**Step 2 — Count pairs**: For each adjacent pair of symbols, count how many times it occurs across all word representations.

```
Pair counts:
  (l, o): 5   (o, w): 5   (w, </w>): 3
  (w, e): 2   (e, r): 1   (e, s): 1   ...
```

**Step 3 — Merge most frequent pair**: The pair with the highest count is merged into a new symbol.

```
Merge (l, o) → lo:
  lo w </w>        (freq 3)
  lo w e r </w>    (freq 1)
  lo w e s t </w>  (freq 1)
```

**Step 4 — Repeat**: Count pairs again on the updated representations, merge the next most frequent pair. Continue for a fixed number of merges (which determines vocabulary size).

```
Merge 2: (lo, w) → low
  low </w>        (freq 3)
  low e r </w>    (freq 1)
  low e s t </w>  (freq 1)

Merge 3: (low, </w>) → low</w>
  ...
```

Each merge adds one symbol to the vocabulary. Starting from $V_0$ base symbols and performing $K$ merges gives a final vocabulary of size $V_0 + K$.

### Encoding at Inference

Given a new word, repeatedly apply learned merge rules in the order they were learned:

```
Word: "lowest"
Chars: l o w e s t </w>

Apply merge 1 (l+o → lo): lo w e s t </w>
Apply merge 2 (lo+w → low): low e s t </w>
Apply merge 3 (low+</w> → low</w>): not applicable (no </w> adjacent)
...
Apply merge (e+s → es): low es t </w>
Apply merge (es+t → est): low est </w>
Final: ["low", "est", "</w>"]  →  ["low", "est##"]
```

### Byte-Level BPE

GPT-2 and its descendants use **byte-level BPE**: the base vocabulary is the 256 UTF-8 bytes rather than Unicode characters. This guarantees that *any* text can be tokenized — even arbitrary binary content — because every byte is in the base vocabulary. There is no OOV at the byte level.

---

## WordPiece

Developed at Google (Schuster & Nakamura, 2012; used in BERT). Similar to BPE but with a different merge criterion.

### Key Difference from BPE

BPE merges the pair that appears most frequently. WordPiece merges the pair that **maximizes the likelihood of the training corpus** under a unigram language model.

Concretely, the score for merging symbols $a$ and $b$ into $ab$ is:

$$\text{score}(a, b) = \frac{\text{freq}(ab)}{\text{freq}(a) \times \text{freq}(b)}$$

This is the pointwise mutual information (PMI) between $a$ and $b$. A pair scores highly if it co-occurs much more than expected by chance — i.e., if the pair forms a cohesive unit, not just two common symbols that happen to be adjacent.

**Effect**: WordPiece is more conservative about merging very frequent character pairs that don't form meaningful units. In practice, the vocabularies look similar to BPE for standard English text.

### WordPiece Notation

WordPiece uses the `##` prefix to mark continuation subwords:

```
"tokenization"  →  ["token", "##ization"]
"unaffected"    →  ["un", "##affect", "##ed"]
```

The `##` signals that the piece should be concatenated to the preceding piece without a space.

---

## SentencePiece

Developed by Kudo & Richardson (2018) at Google. A language-agnostic tokenization library that implements both BPE and a unigram language model approach.

### Key Innovation: Treating Text as a Raw Character Stream

Word-level tokenization and WordPiece assume pre-tokenized input (whitespace separated). This is inappropriate for:
- Japanese, Chinese (no word boundaries)
- Arabic, Hebrew (complex morphology)
- Any language where tokenization rules differ from English whitespace splitting

SentencePiece treats the **input as a raw Unicode string** and learns to segment it from scratch. Whitespace is treated as a regular character (represented as `▁` or `_` in the vocabulary):

```
"Hello world"  →  ["▁Hello", "▁world"]
"世界"         →  ["▁世界"]  or  ["▁世", "界"]  (depending on frequency)
```

This makes SentencePiece fully language-agnostic.

### Unigram Language Model

The unigram variant (Kudo, 2018) takes a different approach from BPE entirely:

1. Start with a large candidate vocabulary (all substrings up to a certain length).
2. Assign each piece a log-probability and define the probability of a segmentation as the product of its piece probabilities.
3. Use the Viterbi algorithm to find the most likely segmentation of each training sentence.
4. Iteratively prune pieces that contribute least to the overall likelihood (EM-style optimization).
5. Repeat until target vocabulary size is reached.

The unigram model provides a probability distribution over *all possible segmentations* of a sentence (not just the greedy-merge segmentation BPE produces), enabling stochastic sampling for data augmentation.

**Used in**: T5, XLNet, ALBERT, mBART, LLaMA (via SentencePiece BPE on bytes).

---

## Vocabulary Size and Its Effects

| Model | Tokenizer | Vocab Size |
|-------|-----------|-----------|
| BERT | WordPiece | 30,522 |
| GPT-2 | Byte-level BPE | 50,257 |
| T5 | SentencePiece | 32,000 |
| LLaMA 1/2 | SentencePiece BPE | 32,000 |
| LLaMA 3 | Tiktoken (BPE) | 128,256 |
| GPT-4 | Tiktoken (BPE) | ~100,000 |

### Effects of Vocabulary Size on Model Behavior

**On embedding matrix size**: The embedding table has shape $(V \times d)$. With $V = 100{,}000$ and $d = 4096$, this is 1.6B parameters — a substantial fraction of smaller models' total parameter count.

**On sequence length**: A larger vocabulary tokenizes text into fewer tokens. GPT-4's 100K vocabulary produces roughly 30% fewer tokens than BERT's 30K vocabulary for the same English text. Fewer tokens → less attention compute → effectively longer context for the same sequence length budget.

**On multilingual coverage**: A vocabulary trained primarily on English will allocate most tokens to English morphemes. For other scripts, characters are rarely merged, producing 4–10× more tokens per word than English. LLaMA 3's 128K vocabulary was specifically enlarged to improve multilingual token efficiency.

**On rare words**: A larger vocabulary means more words are represented as single tokens, improving their embedding quality. But very rare items in the vocabulary receive minimal gradient signal and may be poorly trained.

---

## Special Tokens

Special tokens serve structural roles and are always in the vocabulary. Their exact form is model/framework-specific:

| Token | Model(s) | Purpose |
|-------|----------|---------|
| `[CLS]` | BERT, RoBERTa | "Classification" token prepended to every input. Its final hidden state is used as the aggregate sequence representation for classification tasks. |
| `[SEP]` | BERT | Separates two segments (e.g., question and passage). Also marks end of each segment. |
| `[PAD]` | BERT, most models | Padding token to equalize sequence lengths in a batch. Attention is masked over these positions. |
| `[MASK]` | BERT | Replaces 15% of tokens during masked language model pretraining. The model predicts the original token. |
| `<s>` | RoBERTa, many others | Beginning of sequence token (same role as `[CLS]` in practice). |
| `</s>` | RoBERTa, T5, BART | End of sequence token; signals generation completion to the decoder. |
| `<\|endoftext\|>` | GPT-2, GPT-3 | Marks document boundaries in training data; signals end of generation. |
| `<\|im_start\|>`, `<\|im_end\|>` | ChatML format | Conversation role delimiters (instruct-tuned models). |
| `[BOS]`, `[EOS]` | Generic names | Beginning/end of sequence; exact tokens vary by model. |
| `<unk>` | Older models | Unknown token fallback (mostly eliminated by byte-level tokenizers). |

Special tokens are added to the vocabulary outside the normal merge/split process and are never split into subpieces.

---

## Tokenization Artifacts

### Capitalization

Most tokenizers are case-sensitive. "Cat" and "cat" are different tokens (unless using a lowercased vocabulary like BERT-uncased). The model must learn that capitalized and uncapitalized versions share meaning, which it does through co-occurrence patterns but at the cost of some embedding separation.

### Leading Whitespace

SentencePiece encodes "world" and " world" as different tokens (`world` vs. `▁world`). BPE-based tokenizers similarly distinguish leading-space variants. This means the same word at the start of a sentence vs. in the middle is tokenized differently:

```python
tokenizer.encode("Hello world")  →  ["Hello", "▁world"]
tokenizer.encode("world")        →  ["world"]  # different token!
```

### Numbers

Numbers are usually tokenized digit-by-digit or in small groups:

```
"12345" → ["12", "345"]   or  ["1", "2", "3", "4", "5"]
```

This makes arithmetic reasoning hard: "12345 + 1 = 12346" requires the model to learn that `12346` is `12345 + 1` — a relationship not explicit in the token sequence.

### Code

Programming languages have identifiers, operators, and syntax that tokenize inconsistently:

```python
"variable_name" → ["variable", "_", "name"]   # 3 tokens
"variableName"  → ["variable", "Name"]         # 2 tokens (CamelCase is one unit)
```

Code-specialized tokenizers (e.g., those in Codex/StarCoder) include common code patterns and merge frequently occurring code tokens.

### Non-Latin Scripts

For languages like Arabic, Chinese, or Hindi, a vocabulary trained primarily on English typically assigns one token per character (no merges). A 10-word Arabic sentence might tokenize to 50+ tokens, consuming context window and degrading performance.

### Emojis and Special Characters

Multi-byte Unicode characters tokenize to multiple byte tokens in byte-level BPE:

```
"😊" (U+1F60A) → bytes [F0, 9F, 98, 8A] → 4 tokens
```

---

## Token Count vs. Word Count

The oft-cited rule of thumb: **1 token ≈ 0.75 words** for English text (or equivalently, **1 word ≈ 1.33 tokens**). This is derived from empirical observation on English prose with standard vocabularies (~30–50K):

- Common words ("the", "and", "is") → 1 token each.
- Medium words ("tokenization") → 2–3 tokens ("token", "##ization").
- Rare words, names, compounds → 3–5+ tokens.

The average across typical English text lands near 1.3 tokens/word.

**But this varies enormously**:

| Content type | Tokens/word |
|-------------|------------|
| Simple English prose | 1.0–1.2 |
| Technical writing | 1.3–1.5 |
| Code | 2–4 |
| Arabic/Hebrew text | 3–5 |
| Chinese/Japanese | 1.5–3 |
| Emojis | 4–10 (bytes) |

Practical implication: when estimating whether a document fits in a context window, multiply word count by 1.3 for English, higher for other content types.

---

## Out-of-Vocabulary Handling

With character-level or byte-level base vocabularies, OOV is effectively eliminated. Any string can be decomposed into its constituent characters or bytes, which are always in the vocabulary.

For word-level tokenizers (legacy), OOV is handled by:
1. Replacing the word with a special `<unk>` token (information is lost).
2. Character decomposition (rarely implemented in word-level systems).

For subword tokenizers: a word not in the vocabulary is split into subword pieces. If a subpiece also isn't in the vocabulary, it's split further, eventually reaching individual characters or bytes. This guarantees full coverage.

The `<unk>` token still exists in most modern tokenizers but should appear with near-zero frequency in practice for byte-level systems.

---

## Using HuggingFace Tokenizers

The `transformers` and `tokenizers` libraries (HuggingFace) provide a unified API:

```python
from transformers import AutoTokenizer

# Load a tokenizer (e.g., LLaMA-2)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Basic encoding
text = "The quick brown fox jumps."
tokens = tokenizer.tokenize(text)
# ['▁The', '▁quick', '▁brown', '▁fox', '▁jumps', '.']

ids = tokenizer.encode(text)
# [1, 450, 4996, 17354, 1701, 29916, 12230, 29879, 29889]
# (1 is <s> BOS token automatically prepended)

# Decode back
decoded = tokenizer.decode(ids)
# "<s> The quick brown fox jumps."

# Batch encoding with padding and truncation
batch = tokenizer(
    ["First sentence.", "Second, longer sentence here."],
    padding=True,          # Pad to longest in batch
    truncation=True,       # Truncate to model's max length
    max_length=512,
    return_tensors="pt"    # Return PyTorch tensors
)
# batch["input_ids"]: tensor of shape [2, max_len]
# batch["attention_mask"]: 1 for real tokens, 0 for padding

# Inspect special tokens
print(tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token)
# <s>  </s>  None (LLaMA-2 has no pad token by default)

# Add a pad token if needed
tokenizer.pad_token = tokenizer.eos_token
```

**Counting tokens without full encoding**:

```python
# Fast estimation using fast tokenizer
n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
```

**Tokenizer fast vs. slow**: The `tokenizers` library (Rust backend) provides `Fast` tokenizer variants that are 10–100× faster and support offset mapping (linking token positions back to character positions in the original string), useful for span extraction tasks.

---

## References

- Sennrich, R., Haddow, B., & Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units.* ACL 2016.
- Schuster, M., & Nakamura, K. (2012). *Japanese and Korean Voice Search.* ICASSP 2012.
- Kudo, T., & Richardson, J. (2018). *SentencePiece: A simple and language independent subword tokenizer.* EMNLP 2018.
- Kudo, T. (2018). *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates.* ACL 2018.
- Devlin, J., et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL 2019.
- Radford, A., et al. (2019). *Language Models are Unsupervised Multitask Learners.* OpenAI Blog (GPT-2).
- Xue, L., et al. (2022). *ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models.* TACL 2022.

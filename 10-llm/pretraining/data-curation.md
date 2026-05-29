# Data Curation for LLM Pre-training

## Table of Contents
1. [Data is the Underrated Variable](#data-is-the-underrated-variable)
2. [Primary Data Sources](#primary-data-sources)
3. [Quality vs. Quantity Tension](#quality-vs-quantity-tension)
4. [Filtering Pipeline](#filtering-pipeline)
5. [Deduplication](#deduplication)
6. [Data Mixing Ratios](#data-mixing-ratios)
7. [Evaluation Set Contamination](#evaluation-set-contamination)
8. [The Role of Code Data](#the-role-of-code-data)
9. [Synthetic and Model-Generated Data](#synthetic-and-model-generated-data)
10. [Tokenization Efficiency Across Data Types](#tokenization-efficiency-across-data-types)
11. [Dataset Documentation: Cards, Provenance, Licensing](#dataset-documentation-cards-provenance-licensing)
12. [References](#references)

---

## Data is the Underrated Variable

Post-Chinchilla, the field learned that parameters alone do not determine quality — tokens matter equally. But tokens are not interchangeable: a token from a carefully curated scientific paper contributes more to model quality than a token from a spam blog. Data curation is the engineering discipline of transforming raw internet-scale text into a training corpus where every token carries high learning signal.

Most frontier lab papers reveal surprisingly little about their data pipelines. The ones that do (LLaMA 3, RefinedWeb, Dolma, RedPajama-V2) reveal that data processing is iterative, expensive, and often the key differentiator between models of the same nominal scale.

---

## Primary Data Sources

### Common Crawl

Common Crawl is a nonprofit that crawls the web continuously and releases compressed archives of raw HTML (~300–400TB per crawl). It is the largest freely available text source, representing the primary data supply for most open-weight LLMs.

**Raw quality is low**: spam, SEO content, templated pages, near-duplicate documents, and foreign-language content mixed with target language account for a large fraction of raw tokens. After filtering, typically 1–10% of raw Common Crawl tokens survive into high-quality training data.

Key processed versions:
- **C4** (Raffel et al., 2020): filtered with simple heuristics (terminal punctuation, no bad words, dedup). 750GB after filtering.
- **CC-Net** (Wenzek et al., 2020): language identification + perplexity filtering with a 5-gram LM trained on Wikipedia. Used by Meta's early models.
- **RefinedWeb** (Penedo et al., 2023): Falcon's web corpus. Uses trafilatura for text extraction, fuzzy deduplication, and quality filtering. Available as a dataset.
- **FineWeb** (HuggingFace, 2024): 15T tokens from 96 Common Crawl snapshots, with per-snapshot deduplication and quality classifiers.

### Books

Books provide long-form text with coherent narratives, diverse vocabulary, and domain depth not found in web data.

- **Books3** (part of The Pile): ~100GB of books, sourced from Bibliotik. Removed from some datasets due to copyright concerns.
- **Project Gutenberg**: Public domain books (pre-1929). Legally unambiguous. ~3GB processed text.
- **BookCorpus**: Used in original BERT; ~800M words from self-published books. Not publicly available in original form.

### Wikipedia and Knowledge Bases

Wikipedia is small (~20GB) but extremely high quality: verified facts, good prose, structured knowledge. Its impact per token on knowledge-intensive benchmarks (TriviaQA, Natural Questions) is disproportionately high. All major LLMs include it at elevated weight.

Wikidata and knowledge graph triples are sometimes included to improve structured reasoning.

### Code

GitHub repositories, StackOverflow Q&A, programming documentation, and coding tutorials. The Stack (BigCode) is the largest permissively licensed code dataset: ~3.1TB of source code in 358 programming languages, filtered by license.

Code is typically upweighted substantially relative to its token count, for reasons discussed in [The Role of Code Data](#the-role-of-code-data).

### Scientific Papers

arXiv (~270GB), PubMed, Semantic Scholar Open Research Corpus (S2ORC). Key for:
- LaTeX math notation (teaches LLMs to reason about symbolic expressions)
- Technical vocabulary and precision
- Long-document reasoning (abstracts → related work → methods → results structure)

arXiv includes LaTeX source, which preserves equation structure better than rendered PDF.

### Multilingual and Specialized Sources

- **mC4**: multilingual version of C4, covering 101 languages
- **CulturaX**: cleaned multilingual web data (24 languages)
- Domain-specific corpora: legal (case law), medical (clinical notes, textbooks), financial (SEC filings)

LLaMA 3 (Meta, 2024) trained on ~15T tokens across these sources, with roughly:
- ~50% web (filtered Common Crawl)
- ~25% code
- ~10% books/academic papers
- ~5% Wikipedia/knowledge bases
- ~10% other (domain-specific, multilingual)

---

## Quality vs. Quantity Tension

The fundamental tension: **volume is available in the trillions of tokens; quality is scarce and expensive to label.**

Two empirical findings that resolve the tension in practice:

**Finding 1: Aggressive filtering usually wins.** When FineWeb-Edu (an educationally-filtered subset of FineWeb using a classifier trained on 500k GPT-4-judged pages) was used to train 1.8B models, it outperformed models trained on 5× more unfiltered tokens on knowledge benchmarks. Signal density matters more than corpus size once you are above the minimum data quantity for the model size.

**Finding 2: Domain coverage matters for capability coverage.** A model trained only on curated encyclopedic text will fail on casual dialogue, code, and informal reasoning. Quality filtering should be within-domain, not cross-domain: filter low-quality web text, but keep diverse web text styles (question-answer, forum discussion, technical documentation).

The practical resolution: filter heavily within each domain but maintain diversity across domains.

---

## Filtering Pipeline

A production data pipeline has multiple ordered stages. Each stage is designed to remove a specific type of noise while minimizing false positives (discarding good data).

```
Raw HTML/text
    │
    ▼
[1] Text Extraction
    (trafilatura / jusText / resiliparse)
    │
    ▼
[2] Language Identification
    (fastText LID classifier)
    │
    ▼
[3] Heuristic Quality Filters
    (line length, ratio rules, word count)
    │
    ▼
[4] Perplexity Filtering
    (5-gram LM or small neural LM)
    │
    ▼
[5] Classifier-Based Quality Scoring
    (trained on human/LLM quality ratings)
    │
    ▼
[6] Safety Filtering
    (toxicity, CSAM, PII removal)
    │
    ▼
[7] Deduplication
    (exact + near-dedup, document + paragraph level)
    │
    ▼
Filtered corpus
```

### Stage 1: Text Extraction

Raw HTML must be converted to clean text. This is non-trivial: boilerplate (navigation menus, footers, ads) must be stripped; encoding issues must be resolved; poorly rendered unicode must be normalized.

Tools:
- **trafilatura**: heuristic-based, fast, good recall on news/blog content
- **jusText**: tuned for article extraction
- **resiliparse**: fast Rust-based extraction used by RefinedWeb/FineWeb

### Stage 2: Language Identification

**fastText language identification model** (Joulin et al., 2016) classifies text into one of 176 languages. Applied at the document level; threshold typically 0.65–0.80 confidence for the target language.

Documents below the threshold are discarded or routed to a multilingual corpus bucket.

### Stage 3: Heuristic Quality Filters

Rule-based filters that are fast and cheap to apply:

| Filter | Typical Threshold | Rationale |
|--------|-----------------|-----------|
| Minimum word count | > 50 words | Removes stubs, navigation text |
| Maximum word count | < 100,000 words | Removes concatenated junk |
| Mean word length | 3–10 characters | Detects garbled text |
| Fraction of alphabetic characters | > 0.7 | Detects symbol-heavy junk |
| Fraction of lines ending in punctuation | > 0.6 | Prose structure signal |
| Stop word ratio | Top-N stop words present | Language quality signal |
| Duplicate line ratio | < 0.2 | Detects boilerplate/spam |
| Fraction of lines starting with bullet chars | < 0.9 | Detects pure listing pages |

These filters are derived from C4 (Raffel et al.) and refined across subsequent work. Thresholds are tuned empirically on development sets.

### Stage 4: Perplexity Filtering

A language model trained on high-quality reference data (e.g., Wikipedia) is used to score documents. Documents with **high perplexity** (the LM is surprised by them) are likely to be noisy, machine-generated spam, or foreign-language text that slipped through language ID.

CC-Net used a 5-gram KenLM model per language, trained on Wikipedia. Documents above a perplexity threshold (typically the 90th percentile of Wikipedia perplexity) are discarded.

**Caution**: perplexity filtering can inadvertently remove valid but unusual text (code, mathematics, poetry, legal boilerplate). The threshold must be set carefully per-domain.

### Stage 5: Classifier-Based Quality Scoring

More expensive but more accurate than heuristics. A classifier is trained to predict document quality, typically using:

- **Positive examples**: Wikipedia, curated books, high-scoring academic text
- **Negative examples**: raw Common Crawl samples

The classifier (often a fastText or small neural classifier for speed) assigns a quality score; documents below a threshold are dropped.

FineWeb-Edu trains a quality classifier specifically for *educational content* (GPT-4 is used to label 500k documents on a 0–5 educational score scale, then a regression model is trained). This domain-specific classifier dramatically outperforms generic quality filtering for knowledge tasks.

### Stage 6: Safety Filtering

Three categories:

**Toxicity removal**: Documents are scored with a toxicity classifier (e.g., Perspective API, fine-tuned classifiers). Documents above a threshold are removed. This is coarse — the goal is to remove the most egregious content, not to achieve perfect safety (which is addressed in alignment, not pretraining).

**CSAM detection**: Child sexual abuse material is detected and removed using perceptual hashing (PhotoDNA) for images and keyword/classifier methods for text. This is non-negotiable and typically done by infrastructure partners.

**PII removal**: Personally identifiable information (email addresses, phone numbers, social security numbers, IP addresses) is identified with regex and named entity recognition and either removed or replaced with placeholder tokens. Regex patterns cover common PII formats; NER catches names in context.

---

## Deduplication

Deduplication is one of the highest-impact steps in data curation. Lee et al. (2022) showed that training on deduplicated data produces models with lower perplexity and better factual accuracy, even with fewer total tokens. Duplicate data teaches nothing new; it can also cause models to memorize specific phrases (privacy risk) and can distort the frequency distribution of rare knowledge.

### Exact Deduplication

Two methods:

1. **SHA-256 hashing**: compute the hash of each document; discard exact duplicates. Catches only byte-for-byte identical documents. Fast, O(N) in corpus size.

2. **N-gram suffix arrays**: build a suffix array over the concatenated corpus (Manber & Myers), identify all repeated substrings above length $k$ (e.g., $k=50$ tokens). This catches exact substring matches across documents (not just identical documents). Used in MassiveText (Gopher data).

### Near-Deduplication with MinHash LSH

Near-deduplication catches documents that are nearly identical (e.g., two web pages that differ only in navigation text or a timestamp). The standard approach uses **MinHash + Locality Sensitive Hashing (LSH)**:

**Step 1: Shingling.** Represent each document as the set of all overlapping $n$-grams (typically 5-grams or 13-grams of characters). A document with $L$ characters has $L - n + 1$ shingles.

**Step 2: MinHashing.** Apply $k$ hash functions $h_1, \ldots, h_k$ to the shingle set and take the minimum value under each hash. The $k$-dimensional MinHash signature approximates the Jaccard similarity between documents:

$$\Pr[h_i(\text{min}(A)) = h_i(\text{min}(B))] = J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

**Step 3: LSH banding.** Divide the $k$ hash values into $b$ bands of $r = k/b$ hashes each. Two documents are candidate duplicates if their signatures are identical in at least one band. This creates a threshold effect: pairs with Jaccard similarity above approximately $(1/b)^{1/r}$ are identified with high probability.

**Step 4: Verification and removal.** Among candidates, verify Jaccard similarity directly and remove near-duplicates (keeping one representative per cluster). The criterion for "near-duplicate" is typically $J > 0.8$.

MinHash LSH scales to trillion-token corpora via MapReduce/Spark; it requires $O(Nk)$ memory for $N$ documents and $k$ hash functions.

### Document-Level vs Paragraph-Level Deduplication

- **Document-level dedup**: removes whole documents that are near-duplicates. Misses cases where paragraphs are shared across otherwise different documents (boilerplate footers, common citations, news wire text reprinted across outlets).
- **Paragraph-level dedup**: shingles at the paragraph level, removes shared paragraphs. More aggressive; can fragment documents. Used in The Pile (v2) and some Meta pipelines.

RefinedWeb and FineWeb use document-level MinHash dedup, which is the more common choice.

---

## Data Mixing Ratios

A training corpus is not a single dataset — it is a weighted mixture of multiple domains. The mixing ratio is a hyperparameter that determines how frequently each domain appears in training batches.

### Why Mixing Ratios Matter

A model trained on 90% web text and 10% code will show different behavior than one trained on 60% web + 30% code + 10% academic. The proportional representation of a domain roughly corresponds to the model's exposure to that domain's style, vocabulary, and reasoning patterns.

### LLaMA 3 as a Case Study

Meta's LLaMA 3 (8B and 70B, 2024) trained on 15T tokens with the following approximate mixture:

| Domain | Approximate Proportion | Rationale |
|--------|----------------------|-----------|
| General web (filtered CC) | ~45–50% | Broad world knowledge, diverse style |
| Code (The Stack, GitHub) | ~17–20% | Reasoning, structured thinking |
| Academic/scientific | ~8–10% | Technical precision, STEM knowledge |
| Books | ~6–8% | Long-form coherence, narrative |
| Wikipedia/knowledge bases | ~3–5% | Factual accuracy |
| Multilingual web | ~5–8% | Non-English coverage |
| Other specialized | ~5% | Legal, financial, domain-specific |

These are **upweighted** relative to their natural occurrence (web text would be >90% of raw token counts). Books, Wikipedia, and academic text are upweighted because their signal density per token is higher.

### How to Set Mixing Ratios

Empirically: train small models (1B–7B) with different mixing ratios and evaluate on a battery of downstream tasks. The optimal ratio is task-distribution-dependent. For a model intended for coding: upweight code. For a model intended for biomedical: upweight medical literature.

Domain weighting is typically implemented via **weighted sampling** during data loading: at each step, a domain is sampled with probability proportional to its weight, then a document is sampled uniformly from that domain. This allows weights to be changed without rebuilding the dataset.

---

## Evaluation Set Contamination

### The Problem

If documents from evaluation benchmarks (e.g., MMLU questions, GSM8K problems, HumanEval) appear verbatim in the training corpus, the model may have "memorized" the answers rather than reasoning from scratch. This **inflates reported benchmark scores** and makes honest comparison between models impossible.

### Detection Methods

1. **N-gram overlap**: compute the fraction of $n$-grams (typically $n=8$ or $n=13$) in each evaluation example that appear in the training corpus. Examples above a threshold (e.g., 80% 8-gram overlap) are flagged as potentially contaminated.

2. **Membership inference attacks**: train a classifier to distinguish whether a sample was in the training data (using loss, perplexity, or other signals). More principled but expensive.

3. **Canary injection**: inject synthetic "canary" strings into the training data and test whether the model reproduces them verbatim at inference — a proxy for memorization capacity.

### Mitigation

- Remove known benchmark examples from training data before training (requires knowing the benchmarks in advance)
- Report contamination analysis alongside results (standard practice in reputable papers)
- Use hold-out benchmarks that were created after the training data cutoff

---

## The Role of Code Data

Frontier LLMs consistently include 10–20% code in their training mixture, far above the natural occurrence of code on the internet. This is not primarily for code generation capability — it is for **reasoning quality**.

The empirical evidence:
- Wei et al. (2022) found that chain-of-thought reasoning ability appears sharply in models above a scale threshold; code-trained models exhibit it at lower scale.
- Madaan et al. (2022) showed that training on code improves multi-step reasoning on non-code tasks (grade school math, symbolic manipulation).
- Ablation studies (Guo et al., 2024) confirm code removal degrades math and reasoning benchmarks even for models not tested on code.

**Why does code help reasoning?**

Code has properties that prose lacks:
1. **Explicit control flow**: `if`, `for`, `while` structures require tracking state across steps — exactly what multi-step reasoning demands
2. **Variable binding**: identifiers refer to specific values; the model learns referential consistency
3. **Formal specification**: function signatures and docstrings specify inputs, outputs, and contracts precisely
4. **Long-range dependencies**: a variable defined 50 lines up must be correctly used 50 lines down; code training incentivizes long-range coherence
5. **Verifiability**: code is right or wrong in a well-defined sense, providing clean gradient signal

For these reasons, code is arguably the highest signal-density data source for general reasoning, not just programming tasks.

---

## Synthetic and Model-Generated Data

### The Data Flywheel

A "data flywheel" is a self-reinforcing cycle where better models generate better training data:
1. Pretrain model $M_0$ on human-generated data
2. Use $M_0$ to generate or curate higher-quality data $D_1$
3. Train model $M_1$ on $D_0 \cup D_1$
4. Repeat

This is the basis for constitutional AI (Anthropic), RLHF (OpenAI), and synthetic data generation (Phi series, Microsoft).

### Phi: Quality over Quantity

The Phi series (Gunasekar et al., 2023; Li et al., 2023) trained small models (1.3B–3.8B) primarily on **synthetically generated "textbook-quality" data**: GPT-3.5/4 was prompted to write educational passages, reasoning traces, and worked examples. Phi-1.5 (1.3B) outperformed models trained on 50× more raw web tokens on reasoning benchmarks.

Key insight: LLMs trained on GPT-generated explanations learn problem-solving *procedures*, not just pattern matches. The signal is cleaner than scraped web text.

### Risks of Synthetic Data

1. **Model collapse**: if a model is trained on its own outputs (or outputs of a similar model), it can converge to a degenerate distribution that loses diversity and coverage of the original data. Repeated rounds amplify this (Shumailov et al., 2024).

2. **Error propagation**: synthetic data from an imperfect model contains systematic errors. If these are included in fine-tuning data without verification, they can be learned.

3. **Distribution shift**: GPT-4-generated "textbook math problems" follow a specific style that may not generalize to the diversity of real-world mathematical notation.

Mitigation: always mix synthetic data with human-generated data; use verification (compiler, unit tests, human checking) for synthetic reasoning traces; monitor diversity metrics.

---

## Tokenization Efficiency Across Data Types

Different data types have very different **tokens-per-character ratios** under a fixed tokenizer. This matters because training costs scale with token count, not character count.

Under a 100k BPE vocabulary (e.g., LLaMA 3 tokenizer):

| Data Type | Approx. Tokens per Character | Notes |
|-----------|----------------------------|-------|
| English prose | ~0.25 (4 chars/token avg) | Longest words become 1–2 tokens |
| Python code | ~0.30–0.35 | Indentation whitespace is expensive |
| JSON/XML | ~0.4–0.5 | Repetitive structural characters |
| LaTeX math | ~0.5–0.8 | Many single-char tokens, escape sequences |
| Chinese/Japanese | ~0.8–2.5 | CJK characters often map to 1–3 tokens |
| Minified JS | ~0.5–0.7 | Dense symbol sequences |

**Implications:**
- Chinese text costs ~3–10× more tokens per byte than English with the same vocabulary. Multilingual LLMs need larger vocabularies to remain efficient across languages.
- Math-heavy training (LaTeX) is more expensive than prose training per character of content.
- Code training cost depends heavily on indentation style (tabs vs 4-space indentation) and whether trailing whitespace is preserved.

This affects mixing ratios: if code costs 1.4× more tokens per byte than prose, you may want to adjust nominal token counts when computing effective compute per domain.

---

## Dataset Documentation: Cards, Provenance, Licensing

### Data Cards

Following Gebru et al. (2018) "Datasheets for Datasets," data cards document:
- **Composition**: what sources, what languages, what time ranges
- **Collection methodology**: crawl dates, filtering criteria
- **Preprocessing**: what transformations were applied
- **Uses**: what the dataset is suitable for; known limitations
- **Distribution**: licensing, access restrictions
- **Maintenance**: versioning, updates

Without data cards, it is impossible to diagnose downstream model failures, reproduce results, or identify contamination.

### Licensing Considerations

Not all text on the internet is freely usable for commercial model training. Key categories:

| License Type | Examples | Training Use |
|-------------|---------|-------------|
| Public domain | Gutenberg, government documents | Unrestricted |
| Creative Commons BY/BY-SA | Wikipedia, OpenWebText | Generally OK |
| CC BY-NC | Some academic papers | Non-commercial only |
| Copyright (all rights reserved) | Books3, news articles, NYT | Legally contested |
| Code: MIT/Apache/BSD | Most open source code | Permissive |
| Code: GPL | Linux kernel, many libraries | Copyleft — complex |
| Code: proprietary | Closed-source repos | Not permitted |

The New York Times vs OpenAI lawsuit (2023) and Books3 takedowns have made copyright in training data an active legal issue. The "fair use" defense for training data in the US is not yet settled law.

Projects aiming for clear legal standing (The Stack, The Pile v2, ROOTS, Dolma) explicitly filter by license, accepting only permissively licensed content.

---

## References

- Raffel, C., et al. (2020). *Exploring the limits of transfer learning with a unified text-to-text transformer.* JMLR. [arXiv:1910.10683](https://arxiv.org/abs/1910.10683)
- Wenzek, G., et al. (2020). *CCNet: Extracting high quality monolingual datasets from web crawl data.* LREC. [arXiv:1911.00359](https://arxiv.org/abs/1911.00359)
- Penedo, G., et al. (2023). *The RefinedWeb dataset for Falcon LLM.* NeurIPS. [arXiv:2306.01116](https://arxiv.org/abs/2306.01116)
- Lee, K., et al. (2022). *Deduplicating training data makes language models better.* ACL. [arXiv:2107.06499](https://arxiv.org/abs/2107.06499)
- Gunasekar, S., et al. (2023). *Textbooks are all you need.* [arXiv:2306.11644](https://arxiv.org/abs/2306.11644)
- Shumailov, I., et al. (2024). *AI models collapse when trained on recursively generated data.* Nature. [arXiv:2402.07350](https://arxiv.org/abs/2402.07350)
- Longpre, S., et al. (2023). *The Data Provenance Initiative.* [arXiv:2310.16787](https://arxiv.org/abs/2310.16787)
- Soldaini, L., et al. (2024). *Dolma: An open corpus of three trillion tokens for language model pretraining research.* ACL. [arXiv:2402.00159](https://arxiv.org/abs/2402.00159)
- Meta AI (2024). *The LLaMA 3 herd of models.* [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)
- Gebru, T., et al. (2018). *Datasheets for datasets.* Communications of the ACM. [arXiv:1803.09010](https://arxiv.org/abs/1803.09010)
- Joulin, A., et al. (2016). *Bag of tricks for efficient text classification (fastText).* [arXiv:1607.01759](https://arxiv.org/abs/1607.01759)

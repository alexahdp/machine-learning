# Generation Metrics

## Table of Contents

1. [The Evaluation Problem for Open-Ended Generation](#the-evaluation-problem-for-open-ended-generation)
2. [Perplexity](#perplexity)
3. [BLEU](#bleu)
4. [ROUGE](#rouge)
5. [METEOR](#meteor)
6. [BERTScore](#bertscore)
7. [BLEURT](#bleurt)
8. [Faithfulness Metrics](#faithfulness-metrics)
9. [Diversity Metrics](#diversity-metrics)
10. [Human Evaluation](#human-evaluation)
11. [Metric Choice by Task](#metric-choice-by-task)
12. [Why No Single Metric Is Sufficient](#why-no-single-metric-is-sufficient)
13. [References](#references)

---

## The Evaluation Problem for Open-Ended Generation

Given a prompt, a good LLM response might look like any of thousands of valid strings. Classic supervised learning metrics — accuracy, F1 on label sets — do not apply. The alternatives are:

1. **Reference-based metrics**: Compare the generated output to one or more gold references using string overlap, n-gram statistics, or learned similarity.
2. **Reference-free metrics**: Score the output directly (fluency, factuality, coherence) without a reference.
3. **Human evaluation**: Have humans rate quality on defined dimensions.

No automatic metric strongly correlates with human judgment across all tasks and domains. The practical choice is to pick metrics that are calibrated for your specific task and to validate them against human judgments on a sample.

---

## Perplexity

Perplexity (PP) measures how well a language model predicts a held-out text. It is the inverse geometric mean token probability under the model — equivalently, the exponential of the average cross-entropy loss.

For a tokenized sequence $W = (w_1, w_2, \ldots, w_N)$:

$$\text{PP}(W) = P(w_1, w_2, \ldots, w_N)^{-1/N} = \exp\!\left(-\frac{1}{N} \sum_{i=1}^{N} \log P(w_i \mid w_1, \ldots, w_{i-1})\right)$$

This is exactly $\exp(\mathcal{L})$ where $\mathcal{L}$ is the average negative log-likelihood per token.

**Intuition**: A model with perplexity 50 is, on average, as uncertain as if it had to pick uniformly from 50 equally likely options at each step. Lower is better.

**What perplexity captures**:
- Fluency and grammaticality: implausible token sequences have high perplexity.
- Calibration: a well-calibrated model assigns probability proportional to actual frequency.
- Compression efficiency: perplexity per bit is directly related to bits-per-character, a lossless compression measure.

**What perplexity does not capture**:
- Factual accuracy: a confidently wrong statement can have low perplexity if it matches training data patterns.
- Coherence across long spans: local token prediction is good but long-range consistency may fail.
- Helpfulness or relevance to a prompt.

**Dataset dependence**: Perplexity is not comparable across different tokenizers or corpora. A model trained on web text will have lower perplexity on web text than on biomedical literature — not because it is more capable but because of domain match. Comparisons are only meaningful when using the same tokenizer and evaluation corpus.

**Typical values**: GPT-2 achieves ~18 perplexity on WikiText-103; GPT-3 175B achieves ~10.8; current frontier models are in the 5–8 range on standard benchmarks.

---

## BLEU

**Bilingual Evaluation Understudy** (Papineni et al., 2002) was introduced for machine translation. Given a generated hypothesis $\hat{y}$ and one or more reference translations, BLEU computes a clipped n-gram precision score modified by a brevity penalty.

### N-gram Precision

For each n-gram order $n \in \{1, 2, 3, 4\}$, count how many n-grams in the hypothesis appear in any reference, clipping each n-gram's count to the maximum it appears in any single reference:

$$p_n = \frac{\sum_{\text{ngram} \in \hat{y}} \min\!\left(\text{Count}(\text{ngram}, \hat{y}),\; \max_r \text{Count}(\text{ngram}, r)\right)}{\sum_{\text{ngram} \in \hat{y}} \text{Count}(\text{ngram}, \hat{y})}$$

The clipping prevents gaming by repeating common n-grams (e.g., "the the the the").

### Brevity Penalty

Short hypotheses trivially achieve high precision by only generating n-grams they are sure about. The brevity penalty discourages this:

$$\text{BP} = \begin{cases} 1 & \text{if } |\hat{y}| > |r^*| \\ e^{1 - |r^*|/|\hat{y}|} & \text{if } |\hat{y}| \leq |r^*| \end{cases}$$

where $|r^*|$ is the length of the reference closest to the hypothesis length.

### BLEU Score

$$\text{BLEU} = \text{BP} \cdot \exp\!\left(\sum_{n=1}^{4} w_n \log p_n\right) = \text{BP} \cdot \exp\!\left(\frac{1}{4} \sum_{n=1}^{4} \log p_n\right)$$

using uniform weights $w_n = 1/4$ in the standard BLEU-4 variant. The exponential of the weighted sum of log-precisions is the geometric mean of the precision scores, scaled by BP.

**Range**: 0–100 (usually expressed as a percentage). Translation system scores in the 2002 era were 25–35 on WMT tasks; modern neural MT systems achieve 35–45.

**Limitations**:

1. **No semantic understanding**: "The cat ate the mouse" and "The feline consumed the rodent" share zero n-grams but are semantically equivalent.
2. **Single reference bias**: If only one reference exists, valid paraphrases are penalized.
3. **Poor correlation with human judgment for free-form text**: In summarization, dialogue, and creative writing, BLEU correlates poorly with human ratings (Callison-Burch et al., 2006).
4. **4-gram ceiling**: Long-range coherence is not captured by 4-gram statistics.

BLEU remains the standard metric in machine translation for historical reasons and because translation has a restricted output space with multiple references available. For other tasks, its use is largely discouraged.

---

## ROUGE

**Recall-Oriented Understudy for Gisting Evaluation** (Lin, 2004) was designed for automatic summarization. Unlike BLEU (precision-oriented), ROUGE measures how much of the reference content is captured in the hypothesis.

### ROUGE-N

N-gram recall between hypothesis $\hat{y}$ and reference $r$:

$$\text{ROUGE-N} = \frac{\sum_{\text{ngram} \in r} \min\!\left(\text{Count}(\text{ngram}, r),\, \text{Count}(\text{ngram}, \hat{y})\right)}{\sum_{\text{ngram} \in r} \text{Count}(\text{ngram}, r)}$$

**ROUGE-1** uses unigrams; **ROUGE-2** uses bigrams. ROUGE-2 is more sensitive to phrase overlap and better correlated with human judgment for summarization.

### ROUGE-L

Based on the longest common subsequence (LCS) between hypothesis and reference:

$$\text{ROUGE-L} = F_1(\text{LCS}) = \frac{(1+\beta^2) \cdot R_{lcs} \cdot P_{lcs}}{R_{lcs} + \beta^2 \cdot P_{lcs}}$$

where $R_{lcs} = \text{LCS}(|\hat{y}|, |r|) / |r|$ and $P_{lcs} = \text{LCS}(|\hat{y}|, |r|) / |\hat{y}|$. The LCS-based approach captures sentence-level structural similarity without requiring contiguous matches.

### Typical Values

On CNN/DailyMail (the standard summarization benchmark), extractive systems score ROUGE-1 ~41, ROUGE-2 ~18; abstractive systems circa 2022 score ROUGE-1 ~44, ROUGE-2 ~21. These absolute numbers are only meaningful relative to other systems on the same dataset.

**Limitations**:
- Recall-oriented; doesn't penalize hallucinated content in the hypothesis that doesn't appear in the reference.
- Sensitive to stopword handling and preprocessing.
- Poor for abstractive summaries that paraphrase rather than extract.
- Multiple references improve reliability but are expensive to collect.

ROUGE is the universal standard in summarization, largely for historical reasons and reproducibility, despite known weaknesses.

---

## METEOR

**Metric for Evaluation of Translation with Explicit ORdering** (Banerjee & Lavie, 2005) addresses two key BLEU limitations: it uses recall as well as precision, and it handles morphological variants and synonyms.

### Alignment

METEOR first finds an alignment between hypothesis and reference unigrams:
1. Exact match: identical tokens.
2. Stem match: tokens with the same stemmed form (e.g., "running" = "run").
3. Synonym match: tokens that are WordNet synonyms.

Only one alignment is computed (using a greedy algorithm), and each token can align to at most one token in the other string.

### Score

Given the aligned unigrams, compute unigram precision $P$ and recall $R$:

$$\text{METEOR} = (1 - \text{Pen}) \cdot F_{\alpha}$$

where $F_{\alpha} = \frac{P \cdot R}{\alpha P + (1-\alpha)R}$ (harmonic mean weighted toward recall, with default $\alpha = 0.9$), and the penalty term discourages many short, fragmented matches:

$$\text{Pen} = \gamma \left(\frac{\text{chunks}}{m}\right)^\theta$$

Here $m$ is the number of aligned unigrams, "chunks" is the number of contiguous, consecutively matched unigram groups in the hypothesis, and $\gamma, \theta$ are tunable parameters. Fewer longer chunks (more contiguous matches) → lower penalty → higher score.

**Correlation with human judgment**: METEOR correlates better with human segment-level scores than BLEU on most MT datasets, because stem and synonym matching captures more genuine translation equivalents.

**Limitations**: WordNet-based synonym lookup is English-specific and misses domain-specific vocabulary. The alignment algorithm is $O(n^2)$ and slow for long texts. Less widely adopted outside MT than BLEU/ROUGE.

---

## BERTScore

**BERTScore** (Zhang et al., 2020) replaces n-gram exact matching with contextual embedding similarity. Each token in the hypothesis is matched to the most similar token in the reference using cosine similarity of BERT representations.

### Formulation

Given hypothesis tokens $\hat{y} = (\hat{y}_1, \ldots, \hat{y}_m)$ and reference tokens $y = (y_1, \ldots, y_n)$, with their contextual embeddings $\{\hat{\mathbf{e}}_i\}$ and $\{\mathbf{e}_j\}$ from a pretrained BERT model:

**Precision**: For each hypothesis token, find the most similar reference token:
$$P_{\text{BERT}} = \frac{1}{m} \sum_{i=1}^{m} \max_{j} \frac{\hat{\mathbf{e}}_i \cdot \mathbf{e}_j}{\|\hat{\mathbf{e}}_i\| \cdot \|\mathbf{e}_j\|}$$

**Recall**: For each reference token, find the most similar hypothesis token:
$$R_{\text{BERT}} = \frac{1}{n} \sum_{j=1}^{n} \max_{i} \frac{\hat{\mathbf{e}}_i \cdot \mathbf{e}_j}{\|\hat{\mathbf{e}}_i\| \cdot \|\mathbf{e}_j\|}$$

**F1**:
$$F_{\text{BERT}} = \frac{2 \cdot P_{\text{BERT}} \cdot R_{\text{BERT}}}{P_{\text{BERT}} + R_{\text{BERT}}}$$

Importance weighting (using IDF weights for each token from the reference corpus) can further improve correlation.

**Why contextual embeddings help**: "automobile" and "car" map to similar embedding vectors; "president" and "chairman" may not, despite surface similarity. Contextual representations also capture polysemy — "bank" in a financial context is far from "bank" (river) in BERTScore similarity.

**Correlation with human judgment**: BERTScore consistently outperforms BLEU and ROUGE in correlation with human ratings in MT and summarization, particularly when paraphrasing is common.

**Model choice matters**: BERTScore computed with DeBERTa-xlarge achieves higher correlations than with BERT-base. The choice of layer (default: last few layers) also affects quality.

**Limitations**:
- Computationally expensive (must encode every token via a transformer forward pass).
- Scores are not interpretable in absolute terms; only relative comparisons are meaningful.
- Depends on the quality and domain of the underlying BERT model.
- Still does not detect factual errors: if the hypothesis says "Paris is the capital of Germany," BERTScore may be high if the sentence structure matches the reference.

---

## BLEURT

**Bilingual Evaluation Understudy with Representations from Transformers** (Sellam et al., 2020) goes further than BERTScore by **learning** an evaluation function directly from human judgments.

BLEURT fine-tunes a BERT model on WMT human ratings data: the input is a (reference, hypothesis) pair, and the target is a real-valued human quality score. The model learns a mapping from text pairs to scalar quality estimates.

**Training procedure**:
1. Pre-train on synthetic tasks (predicting BLEU/ROUGE scores, backtranslation acceptability) to expose the model to evaluation-relevant features.
2. Fine-tune on human ratings from WMT shared tasks (2015–2018), predicting normalized segment-level ratings.

**Performance**: BLEURT achieves Pearson $r \approx 0.80$ with human segment-level ratings on WMT19, compared to BLEU's $r \approx 0.57$ and BERTScore's $r \approx 0.74$.

**Limitation**: BLEURT is calibrated to WMT translation human judgments. Generalization to other tasks (summarization, dialogue) is uncertain, and human judgments themselves vary by annotator pool and guidelines.

---

## Faithfulness Metrics

For tasks like summarization and QA, faithfulness — whether the output is factually consistent with the source document or real-world knowledge — is often more important than n-gram overlap with a reference.

### FactScore

**FactScore** (Min et al., 2023) evaluates factuality by decomposing generated text into atomic claims and verifying each claim independently against a knowledge source (Wikipedia). The score is the fraction of atomic claims that are supported:

$$\text{FactScore} = \frac{\text{supported atomic claims}}{\text{total atomic claims}}$$

Verification uses a retrieval system to find relevant Wikipedia passages and then a classifier (or LLM) to judge support. The atomic decomposition step itself uses an LLM.

**Results**: Models like ChatGPT achieve FactScore ~58% on biography generation; Claude is typically higher at ~65%+. These numbers highlight how common hallucination is in open-ended generation.

### Hallucination Detection

For summarization, faithfulness can be measured by checking whether all claims in the summary are entailed by the source document. Common approaches:

- **NLI-based**: A natural language inference model classifies (source, summary sentence) pairs as entailment/neutral/contradiction. SummaC (Laban et al., 2022) uses NLI at the sentence level with a consistency aggregation step.
- **QA-based**: Generate questions from the summary, answer them from the source, and compare answers. QAGS (Wang et al., 2020) and QAFactEval follow this approach.
- **LLM-based**: Ask an LLM to rate whether the summary is factually consistent with the source.

No approach is fully reliable; all have false positive and false negative rates of 10–20% in practice.

---

## Diversity Metrics

When evaluating generative models beyond single-output quality, diversity of the model's output distribution matters — especially for creative tasks, dialogue, and open-ended generation where repetition is a failure mode.

### Distinct-N

**Distinct-1** and **Distinct-2** (Li et al., 2016) measure the fraction of unique n-grams in a generated corpus:

$$\text{Distinct-N} = \frac{|\text{unique N-grams}|}{|\text{total N-grams}|}$$

A model that generates "I am fine, thank you. I am fine, thank you." repeatedly has Distinct-2 close to 0. Higher values indicate more lexically diverse outputs.

**Limitation**: Distinct-N captures lexical diversity but not semantic diversity — different phrasings of the same idea score high on Distinct-N but do not represent genuine diversity of content.

### Self-BLEU

**Self-BLEU** (Zhu et al., 2018) measures inter-sample diversity: given a set of generated samples, compute the BLEU score of each sample against all other samples, then average. High Self-BLEU → low diversity (samples are similar to each other).

$$\text{Self-BLEU} = \frac{1}{|S|} \sum_{s \in S} \text{BLEU}(s,\; S \setminus \{s\})$$

Self-BLEU is used to evaluate the diversity of sampling strategies: top-k sampling produces lower Self-BLEU (more diverse) than greedy decoding or low-temperature sampling.

---

## Human Evaluation

Automatic metrics all approximate human judgment. For production evaluation and research papers making strong claims, human evaluation remains necessary.

### Rating Dimensions

Human evaluators typically rate responses on separate dimensions:

| Dimension | Definition | Scale |
|-----------|-----------|-------|
| **Adequacy** | Does the output convey the meaning of the source/reference? | 1–5 |
| **Fluency** | Is the output grammatical and natural? | 1–5 |
| **Coherence** | Is the output internally consistent and logically organized? | 1–5 |
| **Relevance** | Does the output address the prompt? | 1–5 |
| **Factual accuracy** | Are the stated facts correct? | 1–5 |
| **Faithfulness** | Are all claims supported by the source? | Binary or 1–5 |

Different tasks prioritize different dimensions. Summarization emphasizes faithfulness and adequacy. Dialogue emphasizes coherence and relevance. Creative writing emphasizes fluency and coherence.

### Inter-Annotator Agreement

Human evaluation is noisy. Measuring agreement between annotators is essential for knowing whether ratings are reliable.

**Cohen's kappa** for categorical ratings:

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

where $p_o$ is observed agreement and $p_e$ is expected agreement by chance. $\kappa > 0.6$ is considered substantial agreement; $\kappa < 0.4$ is poor agreement, meaning ratings may be unreliable.

**Krippendorff's alpha** generalizes to any scale (ordinal, interval) and any number of annotators. Preferable for continuous rating scales.

For LLM output quality, inter-annotator agreement is often low ($\kappa \approx 0.3$–$0.5$) on holistic quality ratings but higher ($\kappa \approx 0.6$–$0.8$) on pairwise preference comparisons ("which is better, A or B?"). This is why Chatbot Arena uses pairwise comparisons rather than absolute ratings.

### Crowdsourcing vs. Expert Evaluation

**Crowdsourcing** (via platforms like MTurk or Prolific) is cheap and scalable but workers may not detect subtle errors in technical content. **Expert evaluation** (domain experts, linguists) is expensive but reliable for specialized content. The choice depends on the domain and the types of errors expected.

---

## Metric Choice by Task

| Task | Primary Metrics | Notes |
|------|----------------|-------|
| Machine Translation | BLEU, COMET, BLEURT | COMET (Rei et al., 2020) is now preferred over BLEU in MT research |
| Summarization | ROUGE-1/2/L, BERTScore, SummaC (faithfulness) | ROUGE for coverage; faithfulness metrics essential |
| Dialogue / Chat | Perplexity, Distinct-N, human ratings | No automatic metric reliably captures dialogue quality |
| Factual QA | Exact match, F1 (token overlap), FactScore | For open-domain QA; FactScore for long-form |
| Code Generation | pass@k (unit test execution) | Execution-based; no text metric substitute |
| Creative Writing | Human ratings (fluency, coherence, originality) | No reliable automatic metric |
| Summarization faithfulness | QAFactEval, SummaC, FactScore | Faithfulness orthogonal to ROUGE |

---

## Why No Single Metric Is Sufficient

Text quality is multi-dimensional: a response can be fluent but hallucinated, relevant but verbose, creative but incoherent, accurate but unengaging. No scalar captures all dimensions.

Empirical evidence supports this:

1. **BLEU vs. human in MT**: Callison-Burch et al. (2006) showed BLEU scores disagreed with human rankings in many WMT system comparisons. The MT community now recommends COMET + human evaluation.
2. **ROUGE vs. human in summarization**: Graham (2015) and Peyrard (2019) showed ROUGE rankings correlate poorly with human rankings when comparing diverse systems.
3. **BERTScore limitations**: Despite better human correlation than BLEU, BERTScore fails to detect factual errors, omissions, and hallucinations.
4. **The metric gaming problem**: Any single metric, once adopted as a target, becomes subject to Goodhart's Law — models optimized against it achieve high scores while losing other quality dimensions. This is most visible in RLHF: reward models trained to predict human preferences can be gamed by verbosity or sycophantic tone.

**Best practice**: Use multiple complementary metrics (one for relevance/coverage, one for faithfulness, one for fluency), validate against human judgments on a sample, and report all metrics rather than selecting the one that looks best.

---

## References

- Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002). *BLEU: A Method for Automatic Evaluation of Machine Translation.* ACL 2002.
- Lin, C.-Y. (2004). *ROUGE: A Package for Automatic Evaluation of Summaries.* ACL Workshop 2004.
- Banerjee, S., & Lavie, A. (2005). *METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments.* ACL Workshop 2005.
- Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). *BERTScore: Evaluating Text Generation with BERT.* ICLR 2020.
- Sellam, T., Das, D., & Parikh, A. P. (2020). *BLEURT: Learning Robust Metrics for Text Generation.* ACL 2020.
- Min, S., et al. (2023). *FActScoring: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* EMNLP 2023.
- Li, J., Galley, M., Brockett, C., Gao, J., & Dolan, W. B. (2016). *A Diversity-Promoting Objective Function for Neural Conversation Models.* NAACL 2016.
- Zhu, Y., Lu, S., Zheng, L., Guo, J., Zhang, W., Wang, J., & Yu, Y. (2018). *Texygen: A Benchmarking Platform for Text Generation Models.* SIGIR 2018.
- Callison-Burch, C., Osborne, M., & Koehn, P. (2006). *Re-evaluating the Role of BLEU in Machine Translation Research.* EACL 2006.
- Rei, R., Stewart, C., Farinha, A. C., & Lavie, A. (2020). *COMET: A Neural Framework for MT Evaluation.* EMNLP 2020.
- Laban, P., Schnabel, T., Bennett, P. N., & Hearst, M. A. (2022). *SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization.* TACL 2022.
- Wang, A., et al. (2020). *Asking and Answering Questions to Evaluate the Factual Consistency of Summaries.* ACL 2020.

# LLM Benchmarks

## Table of Contents

1. [Why Benchmarks?](#why-benchmarks)
2. [Knowledge & Reasoning Benchmarks](#knowledge--reasoning-benchmarks)
   - [MMLU](#mmlu)
   - [HellaSwag](#hellaswag)
   - [ARC-Easy and ARC-Challenge](#arc-easy-and-arc-challenge)
   - [WinoGrande](#winogrande)
3. [Math & Coding Benchmarks](#math--coding-benchmarks)
   - [GSM8K](#gsm8k)
   - [MATH](#math)
   - [HumanEval](#humaneval)
   - [MBPP](#mbpp)
   - [SWE-bench](#swe-bench)
4. [Long-Context Benchmarks](#long-context-benchmarks)
   - [Needle-in-a-Haystack](#needle-in-a-haystack)
   - [RULER](#ruler)
   - [LongBench](#longbench)
   - [SCROLLS](#scrolls)
5. [Instruction Following](#instruction-following)
   - [MT-Bench](#mt-bench)
   - [IFEval](#ifeval)
6. [BIG-Bench and BIG-Bench Hard](#big-bench-and-big-bench-hard)
7. [Open-Ended Chat Evaluation](#open-ended-chat-evaluation)
   - [Chatbot Arena / LMSYS](#chatbot-arena--lmsys)
   - [AlpacaEval](#alpacaeval)
8. [Leaderboards](#leaderboards)
9. [Benchmark Contamination](#benchmark-contamination)
10. [Benchmark Saturation](#benchmark-saturation)
11. [References](#references)

---

## Why Benchmarks?

Benchmarks operationalize capabilities into measurable numbers. A well-designed benchmark should satisfy:

1. **Validity**: the score actually reflects the capability it claims to measure.
2. **Reliability**: scores are consistent across runs and evaluation configurations.
3. **Discriminability**: the benchmark separates models of different ability levels.
4. **Resistance to gaming**: it is hard to improve scores without genuinely improving capability.

In practice, no existing LLM benchmark fully satisfies all four criteria. The practical value of a benchmark lies in understanding its specific failure modes and using multiple benchmarks that cover different capability axes.

---

## Knowledge & Reasoning Benchmarks

### MMLU

**Massive Multitask Language Understanding** (Hendrycks et al., 2021) is the de facto standard knowledge benchmark. It covers 57 subjects spanning STEM, humanities, social sciences, and professional domains — from elementary mathematics to medical licensing exam questions to professional law.

**Format**: 4-choice multiple choice. Each question has exactly one correct answer. Evaluated in 0-shot or 5-shot format; the 5-shot version includes 5 examples from the same subject prepended to the prompt.

**Scale**: ~15,908 questions in the test set; ~285 per subject on average.

**Scoring**: accuracy (fraction of correct answers). Reported per-subject and as a macro-average across all 57 subjects.

**Why it became standard**: It tests declarative knowledge across a breadth of domains, making it hard to game by over-indexing on a single area. The 4-choice forced-choice format makes evaluation cheap and deterministic.

**Model performance (approximate, as of 2024–2025)**:

| Model | MMLU (5-shot) |
|-------|--------------|
| GPT-3 (175B) | 43.9% |
| GPT-4 | 86.4% |
| Claude 3 Opus | 86.8% |
| Llama 3 70B | 82.0% |
| Llama 3.1 405B | 87.3% |
| Gemini 1.5 Pro | 85.9% |

Random baseline is 25%. Human expert performance is roughly 89–90% depending on subject.

**Limitations**:
- Questions are static multiple choice; models can sometimes eliminate wrong answers by style rather than knowledge.
- Many high-performing models were likely trained on data containing MMLU questions or very similar material.
- The benchmark is now saturated for frontier models (>85%), making it ineffective at discriminating between them.

A harder variant, **MMLU-Pro** (Wang et al., 2024), expands to 10-choice questions and integrates more reasoning-required problems. Top models score around 60–65% on MMLU-Pro versus 85%+ on MMLU.

### HellaSwag

**HellaSwag** (Zellers et al., 2019) evaluates commonsense reasoning about everyday physical and social situations. Given a short context paragraph (often from WikiHow or ActivityNet Captions), the model selects which of four continuations is most plausible.

**Key design**: HellaSwag uses **adversarial filtering** (AF) to make wrong answers hard. A discriminator model (BERT) is used to reject candidate wrong answers that it finds obviously wrong; only those that fool the discriminator are kept. This makes simple surface statistics insufficient.

**Performance**:

| Model | HellaSwag (0-shot) |
|-------|-------------------|
| GPT-3 (175B) | 78.9% |
| GPT-4 | ~95.3% |
| Llama 3 70B | 93.0% |
| Human | ~95.6% |

The benchmark is now effectively saturated for large models. Its usefulness has shifted to evaluating smaller models (7B–13B range) where meaningful gaps still exist.

### ARC-Easy and ARC-Challenge

**AI2 Reasoning Challenge** (Clark et al., 2018) tests science question answering drawn from standardized US school tests (grades 3–9). The dataset is partitioned into:

- **ARC-Easy**: Questions that retrieval-based baselines can answer correctly.
- **ARC-Challenge**: Questions where retrieval methods fail — requiring inference, not just lookup.

**Format**: 4-choice multiple choice. ~7,787 questions total; ~1,172 Challenge test questions.

**Why ARC-Challenge matters**: It was specifically designed so that simple n-gram overlap with retrieved Wikipedia passages is insufficient. Models must perform genuine multi-step inference.

**Performance (ARC-Challenge, 25-shot)**:

| Model | ARC-Challenge |
|-------|--------------|
| GPT-3 (175B) | 51.4% |
| Llama 2 70B | 67.3% |
| Llama 3 70B | 92.9% |
| Random | 25% |

### WinoGrande

**WinoGrande** (Sakagami et al., 2021) is a large-scale commonsense reasoning dataset using Winograd schemas — sentences with pronoun ambiguity that requires world knowledge to resolve.

**Example**: "The trophy didn't fit in the suitcase because **it** was too big. What was too big?" Answer: the trophy (not the suitcase).

**Scale**: 44,000 problems (versus 273 in the original Winograd Schema Challenge). Also uses adversarial filtering to remove problems solvable by surface statistics.

**Performance** ranges from ~70% (smaller models) to ~90%+ (GPT-4 class), with human performance at ~94%.

---

## Math & Coding Benchmarks

### GSM8K

**Grade School Math 8K** (Cobbe et al., 2021) contains 8,500 linguistically diverse elementary math word problems requiring 2–8 reasoning steps. Problems are entirely in natural language; no equations.

**Why it matters**: GSM8K was designed specifically to test multi-step reasoning, not just retrieval or pattern matching. When chain-of-thought (CoT) prompting was demonstrated to dramatically improve performance on GSM8K, it became the canonical benchmark for studying reasoning in LLMs.

**Scoring**: Exact match on the final numerical answer. The model's scratchpad (reasoning steps) is not scored — only the final answer.

**Performance**:

| Model / Method | GSM8K |
|---------------|-------|
| GPT-3 (few-shot) | 17.9% |
| GPT-3 + CoT (8-shot) | 56.4% |
| GPT-4 | 92.0% |
| Claude 3 Opus | 95.0% |
| Llama 3.1 405B | 96.8% |
| GPT-4o | 97.0% |

The large jump from 17.9% to 56.4% with CoT was one of the key results motivating the CoT research direction (Wei et al., 2022).

**Current status**: Saturated for frontier models. The successor **GSM8K-Platinum** re-annotated problems with verified solutions; some problems had incorrect gold answers, partially inflating prior scores.

### MATH

**MATH** (Hendrycks et al., 2021) contains 12,500 competition-level mathematics problems drawn from AMC, AIME, and similar competitions. Problems are categorized into 7 subjects (Algebra, Counting & Probability, Geometry, Intermediate Algebra, Number Theory, Precalculus, Prealgebra) and 5 difficulty levels (1–5).

**Why it's hard**: Level 4–5 problems require genuinely creative multi-step mathematical reasoning. Even with CoT, base models performed poorly.

**Scoring**: Exact symbolic equivalence of the final answer (accounting for equivalent forms like $\frac{1}{2}$ vs $0.5$).

**Performance**:

| Model | MATH |
|-------|------|
| GPT-3 + CoT | 4.2% |
| Minerva (540B) | 33.6% |
| GPT-4 | 52.9% |
| GPT-4 + process supervision (PRM) | 78.2% |
| Claude 3.5 Sonnet | 71.1% |
| o1 (OpenAI, with extended thinking) | 94.8% |

The gap between standard GPT-4 (~53%) and o1 (~95%) on MATH reflects the impact of test-time compute scaling (chain-of-thought search / process reward models).

### HumanEval

**HumanEval** (Chen et al., 2021) evaluates functional code generation. Given a Python function signature and docstring, the model must generate the function body. Correctness is evaluated by running the generated code against a suite of unit tests.

**Scale**: 164 hand-crafted problems. Tests cover algorithms, data structures, string manipulation, and math.

**Metric**: **pass@k** — the probability that at least one of $k$ generated samples passes all unit tests:

$$\text{pass@}k = 1 - \frac{\binom{n - c}{k}}{\binom{n}{k}}$$

where $n$ is the total number of samples generated per problem and $c$ is the number that pass. Using $n > k$ samples with the formula above gives an unbiased estimate. In practice, pass@1 (generate one sample) and pass@10 are most commonly reported.

**Performance (pass@1)**:

| Model | HumanEval pass@1 |
|-------|-----------------|
| Codex (12B) | 28.8% |
| GPT-4 | 67.0% |
| GPT-4 Turbo | 87.1% |
| Claude 3.5 Sonnet | 89.0% |
| DeepSeek-Coder-V2 | 90.2% |

**Limitations**: 164 problems is a small sample — variance is high. Many problems are template-like and may be memorized from GitHub training data. **HumanEval+** adds more comprehensive unit tests for the same 164 problems, reducing pass rates by roughly 10–15 percentage points.

### MBPP

**Mostly Basic Python Problems** (Austin et al., 2021) provides 374 crowd-sourced Python programming problems, each with 3 test cases. Less focused on algorithmic reasoning than HumanEval; more representative of entry-level programming tasks.

Typically evaluated with pass@1 and pass@80. GPT-4 achieves ~80–85% pass@1.

### SWE-bench

**SWE-bench** (Jimenez et al., 2024) is qualitatively more challenging than HumanEval. It consists of 2,294 real GitHub issues from 12 popular Python repositories (e.g., Django, scikit-learn, pytest). The model must produce a patch that makes the repository's existing test suite pass.

**Why it's hard**: The model must (a) understand a codebase it hasn't been fine-tuned on, (b) localize the bug from a natural language issue description, (c) produce a syntactically and semantically correct multi-file patch, and (d) not break other existing tests.

**SWE-bench Verified** (a 500-problem subset with verified gold patches) is now the standard split.

**Performance (resolve rate)**:

| System | SWE-bench Verified |
|--------|-------------------|
| GPT-4 (zero-shot, no scaffolding) | ~1–2% |
| SWE-agent + GPT-4 | 12.5% |
| Claude 3.5 Sonnet (with scaffolding) | 49.0% |
| OpenAI o1 + scaffolding | ~48% |
| Best agent systems (2025) | ~55–60% |

SWE-bench remains far from solved; it is the current frontier benchmark for software engineering capability.

---

## Long-Context Benchmarks

### Needle-in-a-Haystack

A synthetic stress test: a "needle" (a specific short fact) is injected at a specified position within a long "haystack" document (often fiction or news articles). The model is asked the one question answerable only from the needle.

The test is parameterized by (a) context length (1K–1M tokens) and (b) needle position (start, middle, end), producing a 2D heatmap of retrieval accuracy. Well-known failure modes: models that claim 128K context windows often show significant performance degradation for needles placed in the middle of long contexts — the "lost in the middle" phenomenon.

Results are typically visualized as a heatmap where x-axis = context length, y-axis = needle depth (position as fraction of document). High-quality frontier models (GPT-4 Turbo, Claude 3) show near-perfect recall across the heatmap up to their advertised context lengths.

### RULER

**RULER** (Hsieh et al., 2024) extends the Needle-in-a-Haystack idea with more diverse task types: multi-hop retrieval (find A, use A to find B), aggregation (count occurrences), and question answering with in-context facts. RULER is harder to game because it tests both retrieval and reasoning over retrieved content.

### LongBench

**LongBench** (Bai et al., 2023) covers 21 bilingual long-context tasks across 6 categories: single-document QA, multi-document QA, summarization, few-shot learning, code completion, and synthetic tasks. Context lengths range from 5K to 30K tokens.

Frontier models score 50–70% on LongBench; smaller models degrade sharply at longer contexts.

### SCROLLS

**SCROLLS** (Shaham et al., 2022) — Standardized CompaRison Over Long Language Sequences — aggregates 7 existing long-document NLP datasets into a unified benchmark: GovReport, SumScroll, QMSum, QASPER, NarrativeQA, QuALITY, and ContractNLI. Covers summarization, QA, and NLI over documents ranging from legal contracts to academic papers.

---

## Instruction Following

### MT-Bench

**MT-Bench** (Zheng et al., 2023) evaluates instruction-following quality in multi-turn conversation. It contains 80 high-quality questions across 8 categories: writing, roleplay, extraction, reasoning, math, coding, knowledge (STEM), and knowledge (humanities). Each question is a two-turn conversation: the first turn poses the main task; the second turn adds a follow-up that requires maintaining context.

**Evaluation**: GPT-4 rates each response on a 1–10 integer scale. Scores are averaged across turns and categories.

**Why multi-turn matters**: Many benchmarks test single-turn factual recall. Real assistants must maintain coherent context across turns, handle clarifications, and satisfy follow-up constraints without contradicting previous answers.

**Performance**:

| Model | MT-Bench (avg score) |
|-------|---------------------|
| GPT-4 | 8.99 |
| GPT-3.5-Turbo | 7.94 |
| Claude-v1 | 7.90 |
| Llama 2 70B Chat | 6.27 |
| Vicuna 13B | 6.57 |

MT-Bench scores are bounded by GPT-4's self-evaluation bias — GPT-4 rating GPT-4 may not be fully objective.

### IFEval

**Instruction-Following Evaluation** (Zhou et al., 2023) takes a different approach: it uses verifiable instructions that can be checked programmatically. Examples: "respond with exactly 500 words," "begin every sentence with a capital letter," "include the keyword 'quantum' at least 3 times," "do not use the word 'however'."

**Why verifiable**: Unlike MT-Bench, IFEval eliminates judge subjectivity. The score is either correct or incorrect based on parseable output properties.

**Metrics**: Prompt-level accuracy (did the model follow all instructions in a prompt) and instruction-level accuracy (did the model follow each individual instruction). GPT-4 achieves ~80–83% prompt accuracy; smaller models often struggle below 60%.

---

## BIG-Bench and BIG-Bench Hard

**BIG-Bench** (Srivastava et al., 2023) is a collaborative benchmark with 204 tasks contributed by 450+ researchers, probing capabilities from logical reasoning and creative writing to understanding social norms and generating code. Tasks vary wildly in format: multiple choice, generation, cloze, and more.

The aggregated **BIG-Bench score** is less useful than individual task analysis because averaging over incomparable formats loses information.

**BIG-Bench Hard (BBH)** (Suzgun et al., 2023) identified 23 tasks from BIG-Bench where models (GPT-3, PaLM) performed worse than the average human rater. These tasks are specifically those where chain-of-thought prompting provides the largest gains. BBH has become a standard reasoning benchmark.

BBH tasks include: temporal sequences, tracking shuffled objects, reasoning about colored objects, formal fallacies, causal judgment, and word sorting.

**Performance**:

| Model / Setting | BBH (3-shot CoT) |
|----------------|-----------------|
| GPT-3 (text-davinci-003) | 49.0% |
| GPT-4 | 83.1% |
| Llama 3 70B | 81.0% |
| Claude 3 Opus | 86.8% |

---

## Open-Ended Chat Evaluation

### Chatbot Arena / LMSYS

**Chatbot Arena** (Chiang et al., 2024) is a crowdsourced platform where users submit a prompt, two anonymized model responses are shown side-by-side, and users vote for the better response. Votes aggregate into an **ELO rating** system analogous to chess rankings.

**Why ELO from pairwise comparisons is principled**: Under the Bradley-Terry model, the probability that model $A$ beats model $B$ in a pairwise comparison is:

$$P(A > B) = \frac{e^{\beta_A}}{e^{\beta_A} + e^{\beta_B}}$$

where $\beta_A, \beta_B$ are latent quality parameters. Maximum likelihood estimation of $\{\beta_i\}$ from pairwise votes gives a consistent ranking under mild assumptions. ELO is an online approximation of this.

**Advantages over benchmarks**:
- Real user prompts; not benchmark-style questions.
- Blind evaluation; no position or identity bias.
- Covers the full distribution of user interest, not a curated task set.
- Resistant to contamination (users bring novel prompts).

**Disadvantages**:
- Dominated by fluency and superficial appeal; coherent-sounding wrong answers can win.
- Users are not expert evaluators for technical questions.
- Coverage biased toward English, writing tasks, and casual use.

**ELO scores (approximate, 2024–2025 era)**:

| Model | Chatbot Arena ELO |
|-------|-----------------|
| GPT-4o | ~1310 |
| Gemini 1.5 Pro | ~1300 |
| Claude 3.5 Sonnet | ~1295 |
| Llama 3.1 405B | ~1270 |
| GPT-3.5-Turbo | ~1112 |

Scores are relative; only differences matter.

### AlpacaEval

**AlpacaEval** (Li et al., 2023) evaluates models against a reference model (originally text-davinci-003; later GPT-4 Turbo as reference in AlpacaEval 2.0) using an LLM judge (GPT-4). The score is the **win rate**: percentage of responses where the evaluated model is preferred over the reference.

**AlpacaEval 2.0** uses 805 instructions from diverse sources and reports **length-controlled win rate** — correcting for the systematic preference for longer responses by regressing win rate on response length difference. This was a major improvement because verbosity bias artificially inflated win rates.

**Correlation with Chatbot Arena**: AlpacaEval 2.0 length-controlled win rate correlates with Arena ELO at $r \approx 0.98$, making it a fast automated proxy for the slower crowdsourced evaluation.

---

## Leaderboards

Three primary leaderboards track model performance:

**Open LLM Leaderboard (Hugging Face)**: Aggregates performance on a standardized set of benchmarks (MMLU, ARC, HellaSwag, TruthfulQA, Winogrande, GSM8K) for open-weight models. Focused on reproducible evaluation with fixed prompting protocols. Scores are easy to game if models are specifically fine-tuned for the benchmark format.

**LMSYS Chatbot Arena**: Human preference ELO as described above. The most externally valid measure of assistant quality but slow to accumulate data.

**LiveBench** (White et al., 2024): A contamination-resistant benchmark that generates new questions monthly from recent sources (news, math competitions, programming contest problems dated after the training cutoffs of evaluated models). Avoids saturation by construction. Harder questions focus on math, coding, and reasoning. Top models score 60–75%.

---

## Benchmark Contamination

Contamination occurs when examples from a benchmark's test set (or semantically similar content) appear in the training data. For internet-scale corpora, this is nearly inevitable — MMLU questions appear on study-help websites, HumanEval problems appear on coding forums, and so on.

**Detection methods**:

1. **N-gram overlap**: Count the fraction of benchmark questions with high n-gram overlap with training data shards. A common threshold is 13-gram overlap — if any 13-gram from the question appears in training data, the example is flagged. This misses paraphrased versions.

2. **Membership inference**: Statistical tests for whether the model assigns higher likelihood to verbatim test examples than to perturbed versions. Requires access to model log-probabilities.

3. **Temporal cutoff analysis**: Compare model performance on questions created before vs. after its training cutoff date. If performance drops sharply for post-cutoff questions, it is evidence of contamination rather than capability.

4. **Canary strings**: Embed unique random strings in benchmark examples and check if the model can complete them, which would require memorization.

**Practical impact**: Several papers have found that removing likely-contaminated examples changes benchmark scores by 2–10 percentage points. This is significant when models are separated by less than 1 percentage point on saturated benchmarks.

**Decontamination strategies** used in practice: filtering training data by n-gram overlap with benchmark test sets (used by GPT-4, Llama papers), randomized question-answer ordering, dynamic benchmark generation.

---

## Benchmark Saturation

Once the top models achieve >85% on a benchmark, it loses discriminative power. Saturation occurs faster than expected because:

1. **Capability overhang**: Models are trained on vast data; they often acquire capabilities on benchmarks they were not specifically optimized for.
2. **Architecture and scale improvements**: Each generation of frontier models is substantially more capable than the last.
3. **Contamination**: Training data inclusion inflates scores.

**Current saturation status (2025)**:

| Benchmark | Top Model Score | Human Baseline | Saturated? |
|-----------|----------------|----------------|-----------|
| HellaSwag | ~95% | ~95% | Yes |
| ARC-Easy | ~98% | ~99% | Yes |
| WinoGrande | ~90% | ~94% | Approaching |
| MMLU | ~87% | ~90% | Approaching |
| GSM8K | ~97% | ~100% | Yes |
| HumanEval | ~90% | ~100% | Approaching |
| ARC-Challenge | ~92% | ~100% | Approaching |
| MATH | ~95% (o1) | ~100% | Approaching |
| BBH | ~87% | ~100% | Approaching |
| SWE-bench Verified | ~55% | ~100% | Not yet |
| MMLU-Pro | ~65% | ~80% | Not yet |

The response to saturation has been to (a) design harder benchmarks (MMLU-Pro, LiveBench, FrontierMath), (b) shift from knowledge recall to reasoning and agent tasks (SWE-bench, GAIA), and (c) adopt preference-based evaluations where saturation is less meaningful.

---

## References

- Hendrycks, D., et al. (2021). *Measuring Massive Multitask Language Understanding.* ICLR 2021.
- Zellers, R., et al. (2019). *HellaSwag: Can a Machine Really Finish Your Sentence?* ACL 2019.
- Clark, P., et al. (2018). *Think You Have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge.* arXiv:1803.05457.
- Sakaguchi, K., et al. (2021). *WinoGrande: An Adversarial Winograd Schema Challenge at Scale.* AAAI 2021.
- Cobbe, K., et al. (2021). *Training Verifiers to Solve Math Word Problems.* arXiv:2110.14168.
- Hendrycks, D., et al. (2021). *Measuring Mathematical Problem Solving With the MATH Dataset.* NeurIPS 2021.
- Chen, M., et al. (2021). *Evaluating Large Language Models Trained on Code.* arXiv:2107.03374.
- Jimenez, C. E., et al. (2024). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR 2024.
- Srivastava, A., et al. (2023). *Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models.* TMLR 2023.
- Suzgun, M., et al. (2023). *Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them.* ACL Findings 2023.
- Zheng, L., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS 2023.
- Chiang, W.-L., et al. (2024). *Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference.* ICML 2024.
- Hsieh, C.-Y., et al. (2024). *RULER: What's the Real Context Size of Your Long-Context Language Models?* arXiv:2404.06654.
- Zhou, J., et al. (2023). *Instruction-Following Evaluation for Large Language Models.* arXiv:2311.07911.
- White, C., et al. (2024). *LiveBench: A Challenging, Contamination-Free LLM Benchmark.* arXiv:2406.19314.
- Wei, J., et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS 2022.
- Wang, Y., et al. (2024). *MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark.* arXiv:2406.01574.

# LLM-as-Judge

## Table of Contents

1. [Motivation](#motivation)
2. [The LLM-as-Judge Paradigm](#the-llm-as-judge-paradigm)
3. [MT-Bench](#mt-bench)
4. [Chatbot Arena / LMSYS](#chatbot-arena--lmsys)
5. [AlpacaEval](#alpacaeval)
6. [Prometheus: Open-Source Judge Models](#prometheus-open-source-judge-models)
7. [Judge Prompt Design](#judge-prompt-design)
8. [Biases in LLM Judges](#biases-in-llm-judges)
9. [Mitigation Strategies](#mitigation-strategies)
10. [Agreement with Human Judgment](#agreement-with-human-judgment)
11. [Cascade Judging](#cascade-judging)
12. [G-Eval](#g-eval)
13. [When to Trust LLM Judges](#when-to-trust-llm-judges)
14. [References](#references)

---

## Motivation

Human evaluation is the gold standard for LLM quality, but it is slow, expensive, and does not scale to the continuous evaluation required during model development. Automatic metrics (BLEU, ROUGE, BERTScore) are fast but fail to capture the dimensions that matter most for instruction-following assistants: reasoning quality, helpfulness, stylistic appropriateness, factual accuracy, and safety.

The **LLM-as-judge** paradigm exploits a key observation: capable LLMs have internalized human quality preferences well enough that their evaluations correlate strongly with human judgments, often as well as or better than individual human annotators on many tasks. This allows cheap, fast, scalable evaluation at the cost of trusting another model's judgment.

The trade-off is real: LLM judges are not neutral. They have systematic biases that must be understood and mitigated.

---

## The LLM-as-Judge Paradigm

An LLM judge receives a prompt describing the evaluation task and a formatted input containing the question and the response(s) to evaluate. It produces either:

- **A score**: typically an integer 1–10 rating, possibly per dimension.
- **A preference**: pairwise comparison ("which response is better, A or B?").
- **A critique plus score**: chain-of-thought reasoning followed by a score.

**Two evaluation modes**:

| Mode | Format | Strengths | Weaknesses |
|------|--------|-----------|-----------|
| **Pointwise** | Judge rates a single response | Absolute scoring; easy to aggregate | No reference point; hard to calibrate |
| **Pairwise** | Judge picks the better of two responses | More reliable; easier for judges; matches Chatbot Arena | Only gives ordering, not absolute quality; $O(n^2)$ comparisons for $n$ models |

Both modes can optionally include:
- A **reference answer** (ground truth or expert response) to anchor the evaluation.
- A **rubric** specifying what dimensions to consider and how to weight them.

---

## MT-Bench

**MT-Bench** (Zheng et al., 2023) is the canonical LLM-as-judge benchmark. It contains 80 carefully constructed multi-turn questions across 8 categories:

| Category | Example Topic |
|----------|--------------|
| Writing | Compose a persuasive essay on X |
| Roleplay | Act as a customer service agent for a software product |
| Extraction | Extract named entities from this legal document |
| Reasoning | Logic puzzles, spatial reasoning, sequence problems |
| Math | Word problems, algebra, calculus concepts |
| Coding | Write and explain a function; debug a program |
| Knowledge (STEM) | Explain the mechanism of action of aspirin |
| Knowledge (Humanities) | Analyze the historical significance of Y |

**Multi-turn structure**: Turn 1 poses the main question. Turn 2 is a follow-up that requires (a) maintaining context from Turn 1 and (b) adding a constraint or refinement. This tests coherence maintenance, not just single-turn capability.

**Evaluation protocol**: GPT-4 rates each response on a 1–10 scale. Two variants are used:
1. **Single-answer grading**: GPT-4 sees one response and rates it.
2. **Pairwise comparison**: GPT-4 sees two responses and picks the better one.

To reduce position bias (see Biases section), each pairwise comparison is run twice with A/B swapped; a win is only counted if the same model is preferred both times.

**Results on MT-Bench**:

| Model | MT-Bench Score (1–10) |
|-------|----------------------|
| GPT-4 | 8.99 |
| Claude-v1 | 7.90 |
| GPT-3.5-Turbo | 7.94 |
| Llama-2-70B-Chat | 6.27 |
| Vicuna-33B | 7.12 |
| Vicuna-13B | 6.57 |

**Validation**: The paper demonstrated that GPT-4 judge agreement with human expert preference is 85%+, comparable to human-human agreement (~81%). This established that GPT-4 can serve as a proxy for expensive human evaluation.

---

## Chatbot Arena / LMSYS

**Chatbot Arena** (Chiang et al., 2024) is not purely LLM-as-judge — it uses crowdsourced human judgments — but it established the ELO rating system for chat models that LLM-as-judge methods attempt to replicate cheaply.

**Protocol**:
1. A user submits a prompt.
2. Two anonymized models generate responses simultaneously (the user does not know which models).
3. The user votes for the better response or declares a tie.
4. Votes update ELO scores using the Bradley-Terry model.

**The ELO update rule**: After a match where model $A$ beats model $B$:

$$R_A \leftarrow R_A + K(1 - E_A), \quad R_B \leftarrow R_B + K(0 - E_B)$$

where $E_A = 1/(1 + 10^{(R_B - R_A)/400})$ is the expected win probability for $A$, and $K$ is the update factor. After many comparisons, ELO converges to the maximum likelihood estimate of the latent quality parameters.

**Scale**: The arena has accumulated millions of votes across hundreds of models, making it the most externally valid large-scale LLM quality ranking. New models typically need 500–1,000 battles for stable ELO estimates.

**Coverage bias**: Arena users skew toward English speakers, tech-savvy demographics, and conversational rather than technical tasks. Models optimized for casual chat may outperform stronger reasoning models on the Arena.

---

## AlpacaEval

**AlpacaEval** (Li et al., 2023) uses 805 diverse instructions from sources including OASST, Koala, ShareGPT, and Self-Instruct. The model under evaluation generates responses; an LLM judge (originally GPT-4) compares each response to a reference response from a baseline model and reports the **win rate**.

**Version history**:

| Version | Reference Model | Judge | Key Change |
|---------|----------------|-------|-----------|
| AlpacaEval 1.0 | text-davinci-003 | GPT-4 | Original |
| AlpacaEval 2.0 | GPT-4 Turbo | GPT-4 | LC win rate; harder reference |

**Length-controlled (LC) win rate**: A major problem with AlpacaEval 1.0 was that models learned to write longer responses and win because the judge exhibited verbosity bias. AlpacaEval 2.0 regresses win rate on response length difference and reports the length-controlled win rate — the estimated win rate if both models produced responses of equal length.

$$\text{LC Win Rate} = \text{logit}^{-1}\!\left(\hat{\alpha} + \hat{\beta} \cdot \Delta\ell\right)\bigg|_{\Delta\ell = 0}$$

where $\hat{\alpha}$ and $\hat{\beta}$ are estimated by logistic regression with $\Delta\ell$ (length difference) as a covariate. Setting $\Delta\ell = 0$ gives the debiased win rate.

**Correlation with Chatbot Arena**: AlpacaEval 2.0 LC win rate achieves Spearman $\rho \approx 0.98$ with Arena ELO rankings, making it one of the most reliable fast automated proxies for human preference.

---

## Prometheus: Open-Source Judge Models

**Prometheus** (Kim et al., 2023) is a family of LLMs fine-tuned specifically to evaluate other LLMs. It addresses two problems with using proprietary models (GPT-4) as judges:

1. **Cost**: GPT-4 API calls accumulate at scale.
2. **Reproducibility**: API responses change as the model is updated; evaluations cannot be reproduced exactly.
3. **Transparency**: No access to GPT-4 internals; can't debug judge failures.

**Training**: Prometheus is trained on a synthetic dataset of (instruction, response, rubric, reference answer, evaluation, score) tuples generated by GPT-4, covering diverse tasks and rubrics. The model learns to follow rubrics faithfully and produce consistent scores.

**Prometheus-2** (Kim et al., 2024) improves by training on both pointwise and pairwise evaluation data, achieving agreement with human judgments comparable to GPT-4 on several benchmarks.

**Use case**: Prometheus is recommended when proprietary API access is unavailable, when reproducibility is required, or when a domain-specific rubric needs to be consistently applied.

---

## Judge Prompt Design

The quality of LLM-as-judge evaluation is highly sensitive to prompt design. Key design decisions:

### Pairwise vs. Pointwise Prompt Structure

**Pairwise prompt** (abbreviated):

```
[System]
You are a fair judge. Your task is to evaluate which of the following
two responses is better for the given question.

[User]
Question: {question}

Response A: {response_a}
Response B: {response_b}

Which response is better? State "A" or "B" and explain your reasoning.
```

**Pointwise prompt** (abbreviated):

```
[System]
You are a strict judge. Rate the following response on a scale from 1 to 10.
Use the full range of the scale.

[User]
Question: {question}
Response: {response}
[Rubric]
10: Excellent — accurate, complete, well-written
7-9: Good — mostly correct with minor issues
4-6: Mediocre — partially correct or poorly structured
1-3: Poor — mostly wrong or unhelpful

Score (integer only):
```

### Rubric Specification

An explicit rubric reduces score variance and makes criteria transparent. Without rubrics, judges default to holistic impressions that vary by prompt wording. Effective rubrics:

- Define what each score level means concretely.
- List the dimensions to consider (accuracy, completeness, clarity).
- Provide examples of responses at different score levels (few-shot examples).

### Reference-Guided Evaluation

Including a gold reference answer in the prompt anchors the judge and reduces hallucination of quality criteria. This is especially important for factual tasks where the judge might not know the correct answer.

---

## Biases in LLM Judges

LLM judges have systematic biases that can invalidate evaluations if not controlled.

### Position Bias

When presented with two responses in pairwise comparison, LLMs systematically prefer the response in a certain position — often the first ("primacy bias") or sometimes the second ("recency bias"). The effect magnitude varies by model but can be 5–15 percentage points.

**Detection**: Run each comparison both ways (A vs B, then B vs A) and measure how often the same response wins both times. If the model always prefers the first-position response, win rates under one ordering will be 100% and the other 0%.

### Verbosity Bias

LLMs tend to prefer longer responses, even when the additional content is redundant or off-topic. This is because length correlates with perceived thoroughness and effort. GPT-4 judges are particularly susceptible.

**Quantification**: AlpacaEval 2.0 found that win rate increases by ~1 percentage point for each 100-token length advantage, approximately. The length-controlled win rate described above corrects for this.

### Self-Preference Bias

Models prefer responses that match their own stylistic patterns (e.g., use of lists, hedging language, response length). This is problematic when using a model to evaluate its own outputs: GPT-4 judging GPT-4 may be biased toward GPT-4-like responses from other models.

**Evidence**: Panickssery et al. (2024) showed self-preference rates of 10–20% above what human judges select for several frontier models.

### Sycophancy

If the prompt contains any hint of the evaluator's preference (e.g., "I think Response A is better, but what do you think?"), LLM judges will often agree with the implied preference even if Response B is objectively superior. This is a form of the broader sycophancy problem in LLMs.

**Control**: Never include preference hints in judge prompts. Double-blind protocols where the judge has no context about which model produced which response are preferable.

### Familiarity Bias

Judges may systematically favor responses that look like their own training distribution — well-formatted markdown, hedged language with "I think/I believe," numbered steps for procedural answers. Responses from models with different stylistic norms may be penalized purely on style grounds.

### Summary of Biases

| Bias | Description | Magnitude | Mitigation |
|------|-------------|-----------|-----------|
| Position bias | Prefers first or last response | 5–15 pp | Swap ordering; take consistent wins only |
| Verbosity bias | Prefers longer responses | ~1 pp / 100 tokens | Length-controlled scoring |
| Self-preference | Prefers same-model style | 10–20 pp | Use diverse judges |
| Sycophancy | Agrees with stated preferences | High if hints present | Never include preference hints |
| Familiarity | Prefers in-distribution style | 3–10 pp | Use multiple judges; calibrate |

---

## Mitigation Strategies

**Order swapping**: Run pairwise comparisons in both orders. Count a preference only when the same response wins in both positions. This eliminates position bias at the cost of doubling API calls.

**Multiple judges**: Use several different judge models (GPT-4, Claude, Gemini) and aggregate by majority vote or average score. Biases that are uncorrelated across judges cancel out; biases shared by all frontier models (systematic human preference patterns) remain.

**Reference answers**: Include gold reference answers in the prompt to anchor factual evaluations. This is especially important for STEM questions where the judge might not have reliable knowledge.

**Calibration**: On a sample where human judgments are available, fit a calibration function mapping judge scores to human-aligned scores. A simple linear recalibration can correct for score inflation/deflation.

**Explicit rubrics**: Detailed rubrics reduce implicit biases by forcing the judge to reason against a specified framework rather than holistic impression.

**Chain-of-thought before scoring**: Requiring the judge to write reasoning before giving a score improves consistency and makes biased reasoning visible.

---

## Agreement with Human Judgment

Quantifying LLM judge reliability requires comparison to human judgments on the same examples.

**Metrics**:
- **Agreement rate**: fraction of examples where the judge and human agree on pairwise preference.
- **Cohen's kappa**: agreement beyond chance.
- **Spearman correlation**: rank correlation between judge and human scores across models.

**Empirical results**:

| Judge | Human Agreement (pairwise) | Notes |
|-------|---------------------------|-------|
| GPT-4 | 85.2% | MT-Bench study |
| Human-Human | 81.0% | MT-Bench baseline |
| GPT-3.5-Turbo | 72.3% | Significantly worse than GPT-4 |
| Claude-3-Opus | ~84% | Comparable to GPT-4 |
| Prometheus-7B | 75–80% | Open judge; varies by task |

Note that human-human agreement is ~81%, which creates a ceiling: a perfect judge can agree with a given human at most ~81% of the time (the rest of disagreement is irreducible human variance). GPT-4's 85.2% exceeding this ceiling suggests it may be closer to the "true" preference than either individual human annotator — or it may share systematic biases with human raters.

**When judges are unreliable**:
- Highly technical content where the judge lacks domain knowledge (specialized math, medicine, law).
- Long outputs where the judge may not read the entire response carefully.
- Tasks requiring code execution or mathematical verification.
- Culturally specific content.

---

## Cascade Judging

Running the most capable (and expensive) judge on all examples is wasteful when many comparisons are clear-cut. **Cascade judging** uses a cheap judge first and escalates to an expensive judge only when uncertain:

```
Response A vs. Response B
        |
   Cheap Judge (GPT-3.5 or local model)
        |
   ┌────┴────┐
 Clear    Uncertain
   |          |
 Done    Expensive Judge (GPT-4)
                |
              Done
```

**Threshold setting**: The cheap judge expresses uncertainty as a score margin or confidence. If the margin exceeds a threshold $\tau$, the cheap verdict is accepted; otherwise escalate. Choosing $\tau$ trades off cost against accuracy.

**Empirical efficiency**: In practice, 60–80% of pairwise comparisons are "easy" (clear preference), allowing most examples to be resolved cheaply with minimal accuracy loss.

---

## G-Eval

**G-Eval** (Liu et al., 2023) is a framework for structured LLM evaluation using chain-of-thought scoring. It makes the evaluation criteria explicit and uses the model's generation probabilities to compute a soft-weighted score.

**Protocol**:
1. Define evaluation dimensions (coherence, consistency, fluency, relevance) and a rubric for each.
2. Generate a chain-of-thought evaluation: ask the model to reason step-by-step about each dimension.
3. Ask the model to assign a score (1–5) for each dimension.
4. Compute a **probability-weighted score**: instead of taking the maximum-likelihood score token, compute the expected value over the distribution of score tokens:

$$\text{Score} = \sum_{s \in \{1,2,3,4,5\}} s \cdot P(\text{score} = s \mid \text{context})$$

This soft score is more granular than sampling a single integer and reduces variance from stochastic generation.

**Advantage over direct scoring**: Chain-of-thought before scoring forces the judge to articulate reasoning, which both improves consistency (the reasoning commits the model to a score) and provides a human-interpretable audit trail for judge decisions.

**Results**: G-Eval with GPT-4 achieves Pearson $r \approx 0.51$ with human ratings on SummEval (summarization), compared to $r \approx 0.45$ for BERTScore and $r \approx 0.23$ for ROUGE-L.

---

## When to Trust LLM Judges

LLM judges are reliable when:
- The evaluation task is within the judge's knowledge domain.
- Outputs are short enough for careful reading (< 1,000 tokens typically).
- Position bias is controlled (order-swapped pairwise or pointwise).
- A rubric is provided to anchor scoring.
- The judge's biases are known and calibrated on a human-judged sample.

LLM judges are unreliable when:
- Evaluating highly technical content the judge cannot verify (complex proofs, specialized code).
- The outputs are very long (the judge loses track of early content).
- The task requires actual execution or verification (code correctness, numerical computation).
- Stylistic similarity between judge and one evaluated model creates self-preference.

**Best practice**: Use LLM judges for scalable filtering and relative ranking. Reserve expensive human evaluation for final assessments, calibration samples, and high-stakes comparisons.

---

## References

- Zheng, L., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS 2023.
- Chiang, W.-L., et al. (2024). *Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference.* ICML 2024.
- Li, X., et al. (2023). *AlpacaEval: An Automatic Evaluator of Instruction-following Models.* GitHub (NeurIPS 2023 Workshop).
- Kim, S., et al. (2023). *Prometheus: Inducing Fine-grained Evaluation Capability in Language Models.* ICLR 2024.
- Kim, S., et al. (2024). *Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models.* arXiv:2405.01535.
- Liu, Y., et al. (2023). *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment.* EMNLP 2023.
- Panickssery, A., et al. (2024). *LLM Evaluators Recognize and Favor Their Own Generations.* arXiv:2404.13076.
- Wang, P., et al. (2023). *Large Language Models are not Fair Evaluators.* arXiv:2305.17926.
- Shen, T., et al. (2023). *Large Language Models are Better Reasoners with Self-Verification.* EMNLP 2023.
- Dubois, Y., et al. (2024). *Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators.* arXiv:2404.04475.

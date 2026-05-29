# Hallucination

## Table of Contents

1. [Definition](#definition)
2. [Taxonomy of Hallucination](#taxonomy-of-hallucination)
3. [Why Models Hallucinate](#why-models-hallucinate)
4. [Factuality vs Faithfulness](#factuality-vs-faithfulness)
5. [Detection Approaches](#detection-approaches)
6. [Mitigation Strategies](#mitigation-strategies)
7. [Benchmarks](#benchmarks)
8. [Measurement](#measurement)
9. [The Capability Paradox](#the-capability-paradox)
10. [References](#references)

---

## Definition

Hallucination in LLMs is the generation of text that is fluent, confident, and plausible-sounding but factually incorrect, ungrounded in provided context, or fabricated. The term is borrowed from psychology (perceiving things that are not there) and captures the core failure: the model does not know it is wrong.

Two properties make hallucination particularly dangerous:

1. **Fluency is decoupled from accuracy**: The same generation process that produces well-formed sentences does not distinguish true from false statements. A hallucinated claim looks identical to a correct one in surface form.

2. **Confidence is miscalibrated**: Models often express hallucinated content with high apparent confidence, providing no signal to the reader that verification is needed.

The practical stakes are significant. Hallucination in legal documents, medical advice, scientific summaries, and code generation causes real harm. A model that hallucinates 5% of cited papers, drugs, or statutes is unusable in high-stakes domains even if it is 95% accurate overall.

---

## Taxonomy of Hallucination

Maynez et al. (2020) and Ji et al. (2023) provide the standard taxonomies. The key distinctions:

**By relationship to source (for conditional generation tasks)**:

- **Intrinsic hallucination**: the generated output directly contradicts information present in the source/context. In summarization, this would be stating the opposite of what the document says. This is verifiable and unambiguous.
- **Extrinsic hallucination**: the generated output adds information not present in the source that cannot be verified from it. The information might be true (drawn from the model's parametric memory) or false, but it was not grounded in the provided source.

**By relationship to world knowledge**:

- **Factual hallucination**: the model asserts a real-world claim that is false — wrong dates, non-existent publications, incorrect scientific claims, fabricated people or events. This is the most commonly studied type.
- **Entity hallucination**: fabricating names, places, organizations, or identifiers that do not exist. Particularly common when models are asked to list resources, papers, or examples.

**By relationship to the task**:

- **Faithful vs unfaithful generation**: in conditional generation tasks (summarization, translation, RAG-based QA), faithfulness refers to consistency with the provided source. A faithful model only makes claims supported by the source. Unfaithful generation is a form of hallucination even if the added information happens to be true — the model is not grounded.

These distinctions matter practically: intrinsic hallucinations indicate a model that contradicts its own context (perhaps a training failure); extrinsic hallucinations are harder to detect without external verification; factual hallucinations require world knowledge to identify.

---

## Why Models Hallucinate

Understanding the causes helps design targeted mitigations. The mechanisms are multiple and overlapping:

**1. Training objective does not encode factuality**

The standard language modeling objective is maximum likelihood over next-token prediction:

$$\mathcal{L} = -\sum_t \log P(x_t \mid x_{<t})$$

This objective rewards producing likely continuations given context, not factually correct ones. A model trained on a corpus where "the Eiffel Tower is in Paris" appears frequently and "the Eiffel Tower is in Berlin" never appears will correctly learn to prefer Paris. But for long-tail facts that appear rarely or with noise in the training data, the model has no strong signal for factual accuracy — only for distributional plausibility.

**2. Internet training data contains errors, contradictions, and misinformation**

Web-scraped corpora inevitably include factually incorrect content. Models learn the distribution of what is written, not what is true. If a claim (even a false one) appears in many sources, the model will represent it as likely. Common misconceptions, outdated information, and confident misinformation in the training data become embedded in the model's parameters.

**3. Models cannot represent "I don't know"**

The default decoding process requires the model to always produce a next token. There is no privileged token or mechanism for expressing uncertainty by default. If asked about a topic for which training data is sparse or absent, the model has no way to "not answer" — it can only produce the most likely continuation, which may be a plausible-sounding fabrication. Teaching models to express calibrated uncertainty requires explicit training (see mitigation strategies).

**4. Parametric memory is static and bounded**

All world knowledge a model has is encoded in its parameters at training time. After the training cutoff, the model has no mechanism to update its beliefs. Even before the cutoff, knowledge about entities and events with sparse internet coverage is unreliable. The model's parametric memory is high-capacity but imperfect — it stores statistical regularities, not a database of verified facts.

**5. Fluency optimization can diverge from factuality**

Models are trained to produce fluent, coherent text. This sometimes creates a pressure toward confident, well-formed claims even when the underlying belief is uncertain. A vague or hedged statement might score lower under some evaluation criteria than a confident, specific (but wrong) one.

**6. Context window attention limitations**

In long-context tasks (summarization, document QA), models sometimes fail to attend to relevant parts of the context and instead rely on parametric knowledge or generate based on patterns. This produces extrinsic hallucinations even when the correct answer is present in the context. The "lost in the middle" phenomenon (Liu et al., 2023) shows that information in the middle of long contexts is retrieved less reliably than information at the beginning or end.

---

## Factuality vs Faithfulness

These are related but distinct properties that are often conflated:

**Factuality** is an absolute property: does the claim correspond to ground truth in the real world? It requires external verification against knowledge outside the conversation.

**Faithfulness** is a relative property: does the output stay consistent with the provided source material? In RAG-based systems and summarization, faithfulness means the model only asserts what the retrieved documents or input text support.

The distinction matters for system design:

| Setting | Relevant Property | Failure Mode |
|---------|-------------------|--------------|
| Open-domain QA (no retrieval) | Factuality | Parametric hallucination |
| Summarization | Faithfulness | Intrinsic/extrinsic hallucination |
| RAG-based QA | Both | Faithful to retrieved docs that are themselves wrong |
| Closed-book generation | Factuality | Training-time knowledge errors |

A model can be faithful but not factual (faithfully summarizing a document that contains errors) or factual but not faithful (correctly stating world knowledge that contradicts the provided source). In RAG systems, faithfulness is the proximate goal because we choose the retrieved documents; factuality depends on source quality.

---

## Detection Approaches

**Self-consistency sampling** (Wang et al., 2022): Generate $k$ independent completions for the same prompt using different random seeds (temperature > 0). Measure agreement across completions — claims that appear in most completions are more likely to be accurate; claims that vary across completions indicate uncertainty. This exploits the insight that hallucinations tend to be more variable than correctly memorized facts.

Formally, for a claim $c$ extracted from model outputs:

$$P(\text{claim is correct}) \approx \frac{|\{i : c \in \text{output}_i\}|}{k}$$

Works well for factual claims with clear right/wrong answers; less useful for open-ended generation where legitimate variation is expected.

**FactScore** (Min et al., 2023): A principled framework for factuality evaluation at the claim level.
1. Decompose the generated text into atomic facts — small, standalone, verifiable claims
2. Verify each atomic fact against a knowledge source (Wikipedia, retrieved documents)
3. Report the precision of atomic facts: $\text{FactScore} = \frac{\text{supported atomic facts}}{\text{total atomic facts}}$

This provides interpretable, fine-grained hallucination detection rather than a binary hallucinated/not-hallucinated judgment. FactScore showed substantial hallucination rates even in capable models: GPT-4 on biography generation had a FactScore significantly below 1.0.

**NLI-based detection**: Frame hallucination detection as a natural language inference (NLI) task. Given a premise (the source document or retrieved context) and a hypothesis (a claim in the model's output), an NLI model classifies the claim as entailed, neutral, or contradicted. Claims classified as contradicted are intrinsic hallucinations; claims classified as neutral are extrinsic.

This approach requires a reliable NLI model and works best when there is a clear source document to compare against (summarization, RAG). Performance is limited by the NLI model's own accuracy and coverage.

**Citation checking**: For models that produce citations or source references, verify that:
- The cited source actually exists (entity hallucination check)
- The cited source contains the claim being made (faithfulness check)
- The citation actually supports the assertion (logical support check)

This is straightforward to implement but limited to outputs with explicit citations. Guo et al. (2022) showed that LLMs frequently hallucinate citations even when they exist — citing real papers for claims those papers do not make.

**Uncertainty estimation via token probabilities**: Examine the log-probabilities assigned to key tokens in the generated output. Low-confidence token sequences (high perplexity at the individual token level) may indicate hallucinations. More sophisticated approaches (Kuhn et al., 2023 — "Semantic Entropy") cluster semantically equivalent responses and measure entropy at the semantic level rather than the token level, providing a more reliable uncertainty signal.

---

## Mitigation Strategies

**Retrieval-Augmented Generation (RAG)**: Ground generation in retrieved documents, reducing reliance on parametric memory. When the answer is present in the retrieved context and the model is conditioned on it, factual claims can be anchored to verifiable sources. RAG reduces but does not eliminate hallucination — models still sometimes ignore retrieved content, hallucinate additional claims beyond what is retrieved, or faithfully reproduce incorrect retrieved information.

**Fine-tuning on factual QA data**: Training on datasets with clear factual answers (Natural Questions, TriviaQA) improves the model's tendency to refuse or hedge when it lacks information. The key is including negative examples — questions for which the model should say "I don't know" rather than fabricate an answer.

**Chain-of-thought with explicit verification steps**: Prompting the model to reason step by step and explicitly check intermediate claims ("Before answering, verify: is this consistent with what I know?") can reduce hallucination rates on reasoning tasks. This works because CoT externalizes the reasoning process, making errors visible and correctable before they propagate to the final answer.

**Uncertainty calibration training**: Explicitly train models to predict when they are uncertain. This can be done by:
- Including examples where the model should say "I'm not sure" or hedge its claims
- Training with a calibration objective that penalizes confident-wrong answers more than uncertain-wrong answers
- Self-evaluation prompting: "How confident are you in this answer? Rate 1-5."

Kadavath et al. (2022) ("Language Models (Mostly) Know What They Know") showed that large LLMs have reasonably calibrated self-assessment — they can often identify when they are likely wrong — but this ability requires elicitation.

**Post-hoc fact-checking with external tools**: Deploy a verification stage after generation using search engines, knowledge bases, or specialized fact-checking models. This is effective in principle but expensive at inference time and requires well-defined factual claims that can be verified externally.

**Contrastive decoding**: Subtract the logits of a weaker "amateur" model from the stronger model to reduce repetition and hallucination (Li et al., 2022). The intuition is that the amateur model captures shallow statistical patterns (including hallucination tendencies), and subtracting its predictions amplifies the expert model's distinctive knowledge.

---

## Benchmarks

**TruthfulQA** (Lin et al., 2022): 817 questions designed to elicit hallucinations based on common human misconceptions. Questions span health, law, finance, and conspiracies. The benchmark measures whether models give truthful answers versus plausible-but-false ones. Notably, larger models are not necessarily more truthful on TruthfulQA — GPT-3 variants showed flat or slightly worse truthfulness at scale, suggesting scaling alone does not solve the problem.

**HaluEval** (Li et al., 2023): A large-scale benchmark of hallucinated samples across three tasks: QA, dialogue, and summarization. 35,000 hallucinated/non-hallucinated pairs generated by GPT-3.5 with human verification. Tests models' ability to detect hallucinations in each other's outputs.

**FaithDial** (Dziri et al., 2022): A faithful dialogue benchmark where each response is grounded in a specific Wikipedia passage. Evaluates faithfulness in multi-turn dialogue — a harder setting than single-turn because hallucinations can compound across turns.

**FactScore Benchmark** (Min et al., 2023): Evaluates factuality in long-form generation using the FactScore metric. Tests biography generation and measures atomic fact accuracy against Wikipedia. Provides model-level factuality rankings comparable across systems.

**SelfAware** (Yin et al., 2023): Tests whether models know what they don't know — a calibration benchmark for out-of-scope and unanswerable questions.

---

## Measurement

**Hallucination rate**: The fraction of outputs (or atomic facts within outputs) that are factually incorrect or unfaithful. Requires human evaluation or an automated fact-checking pipeline. Often reported as (1 - FactScore) for long-form generation.

**Factual precision/recall**: In tasks with a defined set of correct facts, precision is the fraction of generated claims that are correct; recall is the fraction of correct facts that are mentioned. Precision targets hallucination; recall targets completeness.

**Faithfulness score**: For conditional generation tasks, the fraction of generated claims that are entailed by the source. Measured via NLI models or human judgment. ROUGE and BLEU do not directly measure faithfulness — high n-gram overlap can coexist with hallucination.

**Self-consistency score**: The fraction of $k$ sampled responses that agree on a specific claim. Higher is more reliable. Used as a proxy for model confidence and a signal for hallucination detection.

---

## The Capability Paradox

A counterintuitive empirical finding: more capable models hallucinate less frequently overall but more convincingly when they do hallucinate.

**Less frequently**: Larger models have better parametric knowledge — they have encoded more correct information during training and can retrieve it more reliably. This reduces the rate of outright factual errors on well-covered topics.

**More convincingly**: When larger models hallucinate, they produce more fluent, contextually coherent, and internally consistent fabrications. A hallucination from a less capable model may be obviously wrong in form; a hallucination from a frontier model may be indistinguishable from correct information without external verification. This is a safety concern: the harm from a believable hallucination is greater than from an obviously wrong one.

Additionally, more capable models are used for higher-stakes tasks. A user who trusts a frontier model enough to use it for legal research or medical information faces greater risk from a convincing hallucination than a user who uses a weaker model for casual queries.

This paradox implies that capability improvements alone are insufficient for safety in high-stakes domains. Hallucination mitigation must be treated as a first-class objective alongside capability.

---

## References

- Ji, Z. et al. (2023). *Survey of Hallucination in Natural Language Generation*. ACM Computing Surveys.
- Maynez, J. et al. (2020). *On Faithfulness and Factuality in Abstractive Summarization*. ACL.
- Min, S. et al. (2023). *FActScoring: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation*. arXiv:2305.14251.
- Lin, S. et al. (2022). *TruthfulQA: Measuring How Models Mimic Human Falsehoods*. ACL.
- Li, J. et al. (2023). *HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models*. arXiv:2305.11747.
- Dziri, N. et al. (2022). *FaithDial: A Faithful Benchmark for Information-Seeking Dialogue*. TACL.
- Wang, X. et al. (2022). *Self-Consistency Improves Chain of Thought Reasoning in Language Models*. arXiv:2203.11171.
- Kuhn, L. et al. (2023). *Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation*. ICLR.
- Kadavath, S. et al. (2022). *Language Models (Mostly) Know What They Know*. arXiv:2207.05221.
- Liu, N.F. et al. (2023). *Lost in the Middle: How Language Models Use Long Contexts*. TACL.
- Li, X. et al. (2022). *Contrastive Decoding: Open-ended Text Generation as Optimization*. arXiv:2210.15097.
- Guo, Z. et al. (2022). *Bringing Order to Chaos: A Comparative Analysis of Generation Biases in GPT-2, GPT-3, and Davinci*. arXiv:2202.07785.
- Yin, Z. et al. (2023). *Do Large Language Models Know What They Don't Know?* arXiv:2305.18153.

---

*← [Alignment Fundamentals](./alignment-fundamentals.md) | → [Safety Techniques](./safety-techniques.md)*

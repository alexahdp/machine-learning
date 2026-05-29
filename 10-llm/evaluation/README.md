# LLM Evaluation

Evaluating large language models is one of the most contested and actively evolving areas in ML. Unlike supervised classification — where accuracy against a held-out test set is both cheap and reliable — LLM evaluation faces a fundamental mismatch: the output space is enormous, tasks are open-ended, and the behaviors we care about most (truthfulness, reasoning depth, helpfulness, safety) are not directly measurable by any single automatic metric.

## Why LLM Evaluation Is Unsolved

**The output space problem.** A classifier maps inputs to a fixed label set; a language model maps inputs to an effectively unbounded string space. Comparing two strings requires either (a) restricting to structured outputs where exact match is meaningful, (b) computing approximate overlap via n-gram or embedding similarity, or (c) asking a human or another model to judge. All three approaches have serious failure modes.

**Benchmark contamination.** LLMs are trained on internet-scale corpora that likely include benchmark questions and answers verbatim. A model that memorizes MMLU answers achieves high accuracy without reasoning. Detecting contamination is hard — n-gram overlap checks miss paraphrased versions, and models rarely reveal when they have memorized a specific example.

**Benchmark saturation.** Top models now exceed 85% on MMLU and approach ceiling on HellaSwag and ARC. Once a benchmark saturates, it ceases to discriminate between models and loses diagnostic value. The field responds by creating harder benchmarks, which saturate in turn.

**Evaluation gaming.** Because benchmark scores drive model selection decisions (papers, leaderboards, product announcements), there is systematic pressure to optimize for benchmark performance specifically. This degrades benchmarks even without outright memorization — targeted fine-tuning on benchmark-adjacent data, prompt engineering for specific formats, cherry-picking evaluation configurations.

**Multi-dimensional quality.** Text quality is not a scalar. A response can be factually accurate but stylistically poor, or coherent and engaging but subtly wrong. No automatic metric captures all dimensions simultaneously, and their correlations with human judgment vary by task.

The files in this section cover three complementary approaches to the evaluation problem: structured benchmarks that test specific capabilities, generation metrics that quantify automatic similarity to reference outputs, LLM-as-judge methods that proxy human judgment, and safety-specific evaluations.

## Contents

| File | Description |
|------|-------------|
| [benchmarks.md](./benchmarks.md) | Major evaluation benchmarks: MMLU, GSM8K, HumanEval, BIG-Bench Hard, LMSYS Chatbot Arena, and the contamination/saturation problem |
| [generation-metrics.md](./generation-metrics.md) | Automatic metrics: perplexity, BLEU, ROUGE, METEOR, BERTScore, BLEURT, faithfulness and diversity metrics |
| [llm-as-judge.md](./llm-as-judge.md) | Using LLMs to evaluate LLMs: MT-Bench, AlpacaEval, judge biases, mitigation strategies, G-Eval |
| [safety-evaluation.md](./safety-evaluation.md) | TruthfulQA, bias benchmarks, toxicity evaluation, red-teaming, jailbreak attacks, refusal quality |

## Navigation

- [← Back to LLM Overview](../README.md)
- [← Previous: Fine-tuning](../fine-tuning/README.md)
- [→ Next: Inference](../inference/README.md)

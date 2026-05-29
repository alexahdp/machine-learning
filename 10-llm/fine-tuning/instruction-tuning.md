# Instruction Tuning

## Table of Contents

1. [What Instruction Tuning Is](#what-instruction-tuning-is)
2. [Why Instruction Tuning Enables Generalization](#why-instruction-tuning-enables-generalization)
3. [FLAN: The Foundational Work](#flan-the-foundational-work)
4. [Scaling Instruction Tuning: FLAN-T5 and FLAN-PaLM](#scaling-instruction-tuning-flan-t5-and-flan-palm)
5. [Open-source Instruction Datasets](#open-source-instruction-datasets)
6. [Synthetic Data Generation: Distillation from Stronger Models](#synthetic-data-generation-distillation-from-stronger-models)
7. [Chat Templates and Special Tokens](#chat-templates-and-special-tokens)
8. [Template Format Reference](#template-format-reference)
9. [Instruction Diversity: Task Coverage and Quality](#instruction-diversity-task-coverage-and-quality)
10. [Evaluation: Measuring Instruction Following](#evaluation-measuring-instruction-following)
11. [How Instruction Tuning Enables In-context Learning Without Few-shot Examples](#how-instruction-tuning-enables-in-context-learning-without-few-shot-examples)
12. [References](#references)

---

## What Instruction Tuning Is

Instruction tuning is a specific form of supervised fine-tuning where the training data consists of **(instruction, response) pairs** covering a diverse range of tasks and phrasings. The goal is not to specialize the model for one task but to teach it the *meta-skill* of following natural language instructions in general.

A standard SFT example for sentiment classification looks like:

```
Input: "The movie was terrible, a complete waste of time."
Output: "Negative"
```

An instruction-tuned version wraps this in a natural language instruction:

```
Instruction: "Classify the sentiment of the following review as Positive, Negative, or Neutral."
Input: "The movie was terrible, a complete waste of time."
Output: "Negative"
```

This framing seems superficially similar, but the effect is profound. When the model is trained on thousands of tasks *all expressed in this instruction-following format*, it learns a general mapping from task descriptions to appropriate outputs. At inference time, it can follow instructions for tasks it has **never seen during fine-tuning** — it has learned to read and interpret the instruction itself as a specification of the desired transformation.

---

## Why Instruction Tuning Enables Generalization

A pretrained LLM already encodes the knowledge needed to perform most tasks — it knows grammar, facts, reasoning patterns, code syntax, and so on. The bottleneck is not knowledge but *elicitation*: how to reliably extract the right capability from the model given a natural language prompt.

Instruction tuning solves a fundamental problem with naive prompting: **prompt sensitivity**. Raw pretrained models are brittle to prompt phrasing. Rephrasing "Classify the sentiment" as "What is the tone of this review?" can produce completely different outputs. Instruction tuning, by exposing the model to many phrasings of the same task, builds robustness to surface-level variation.

The formal picture: instruction tuning fine-tunes the conditional distribution $P_\theta(y \mid \text{instruction}, x)$ such that for any semantically equivalent instruction $I$, the output $y$ is consistent and correct. The model learns to *parse and execute* the instruction rather than pattern-match to a specific prompt template.

This is categorically different from task-specific fine-tuning, which overrides the model's behavior for one task but leaves it unchanged on others. Instruction tuning modifies the model's *interface* — how it interprets inputs — rather than just its outputs for a fixed task.

---

## FLAN: The Foundational Work

**FLAN** (Finetuned Language Net, Wei et al., 2021) was the first systematic study of instruction tuning at scale. The key experimental design:

- Took a 137B-parameter GPT-style model
- Curated 62 NLP datasets, grouped into 12 task clusters (sentiment, NLI, commonsense, reading comprehension, etc.)
- For each dataset, wrote 10 natural language instruction templates expressing the same task differently
- Fine-tuned on all datasets except the held-out cluster being evaluated
- Evaluated zero-shot on the held-out cluster

**The central finding**: FLAN dramatically improved zero-shot performance on unseen task clusters — outperforming the same base model prompted with few-shot examples on 20 of 25 datasets, and even outperforming 175B GPT-3 (with few-shot examples) on several tasks.

What made FLAN work:

1. **Template diversity**: Each task appeared in 10 phrasings, preventing the model from learning a template-specific pattern
2. **Task diversity**: Holding out a task cluster and still generalizing to it proves the model learned general instruction following, not task-specific mapping
3. **Scale**: The effect was much stronger at 137B than at 8B, suggesting instruction tuning requires a capable base model to unlock

The FLAN result demonstrated that **zero-shot instruction following is an emergent capability** that can be elicited through fine-tuning without needing in-context examples at inference time.

---

## Scaling Instruction Tuning: FLAN-T5 and FLAN-PaLM

Subsequent work scaled instruction tuning dramatically:

**FLAN-T5** (Chung et al., 2022): Applied FLAN-style instruction tuning to the T5 encoder-decoder family. Trained on 1,836 tasks (vs. FLAN's 62), covering the full Flan 2022 collection. Showed that:
- More tasks consistently improve performance
- Chain-of-thought (CoT) data included in the mix improves reasoning even on tasks without CoT
- T5 encoder-decoder models particularly benefit from instruction tuning due to their seq2seq training objective

**FLAN-PaLM** (Chung et al., 2022): Instruction-tuned PaLM (540B) on the same collection. Key result: FLAN-PaLM substantially outperforms PaLM on most benchmarks and achieves near-human performance on reasoning tasks when combined with CoT prompting.

The **scaling conclusion**: instruction tuning benefits increase with both model scale and task diversity. More tasks is more important than more examples per task, up to a point.

---

## Open-source Instruction Datasets

### Alpaca (Stanford, 2023)

Alpaca used the **Self-Instruct** method (Wang et al., 2022) to generate 52,000 instruction-response pairs:

1. Start with 175 seed human-written instruction examples
2. Prompt GPT-3.5-Turbo to generate new instruction-response pairs, using the seeds as context
3. Filter for quality and deduplicate
4. Fine-tune LLaMA-7B on the resulting dataset

Cost: approximately $500 in API calls. The resulting model was surprisingly capable — comparable to GPT-3.5 on many instruction-following tasks, at 1/24th the parameter count. The lesson: *the distillation of a stronger teacher into a weaker student via synthetic data is a highly effective instruction-tuning strategy*.

Limitations: Alpaca exhibited hallucination (GPT-3.5 hallucinations propagated to training data) and limited instruction diversity (the seed set was not very broad).

### Vicuna and ShareGPT

Rather than synthetic generation, Vicuna (Chiang et al., 2023) used **real user conversations** with ChatGPT scraped from ShareGPT.com — approximately 70K multi-turn conversations. This had two advantages:
- Conversations reflect real user needs rather than researcher-imagined instructions
- Multi-turn format teaches the model to maintain coherence across dialogue turns

Vicuna-13B was rated by GPT-4 as achieving 92% of ChatGPT's response quality — a strong result from 70K examples of real conversations.

### OpenHermes, Orca, and WizardLM

**Orca** (Mukherjee et al., 2023): Augmented instruction data with **step-by-step reasoning traces** generated by GPT-4. The insight: models that learn from explanations (not just answers) generalize better. Orca-13B matched or exceeded GPT-3.5 on reasoning benchmarks.

**WizardLM** (Xu et al., 2023): Introduced **Evol-Instruct** — a method to automatically increase instruction complexity and diversity by prompting an LLM to rewrite simple instructions into harder variants (adding constraints, asking for step-by-step reasoning, combining multiple tasks). Training on evolved instructions taught models to handle complex, multi-part instructions.

**OpenHermes**: Community-curated collection combining Alpaca, Vicuna, Orca-style data, and code data. Demonstrated that dataset curation and mixing strategy significantly outweighs raw dataset size.

---

## Synthetic Data Generation: Distillation from Stronger Models

The pattern across Alpaca, Orca, and WizardLM is **model distillation**: using a stronger teacher model to generate training data for a weaker student. This is powerful because:

1. The teacher's outputs encode implicit reasoning and style that would be expensive to collect from humans at scale
2. The student learns to mimic the teacher's *interface* (how it presents information) not just its *knowledge*
3. The cost of API calls to GPT-4 is orders of magnitude less than human annotation at equivalent scale

**Risks of distillation**:
- **Hallucination propagation**: If the teacher hallucinates, the student learns to hallucinate confidently
- **Homogenization**: All distilled models converge toward the teacher's style, reducing diversity in the open-source ecosystem
- **License restrictions**: OpenAI's terms of service prohibit using GPT-4 outputs to train competing models — legally ambiguous for distillation datasets
- **Capability ceiling**: A student cannot exceed the teacher's capabilities by distillation alone; it can only approach them

---

## Chat Templates and Special Tokens

A **chat template** is the formatting scheme that converts a list of messages (with roles) into a single string of tokens the model processes. Chat templates are not optional — a model trained with a specific template will perform poorly when a different template is used at inference time.

Templates must:
1. Distinguish between system, user, and assistant turns unambiguously
2. Indicate where the model should begin generating (immediately after the final assistant prompt)
3. Support multi-turn conversations by chaining turns
4. Handle special tokens for beginning/end of sequences

---

## Template Format Reference

### ChatML (OpenAI, used by Mistral, Qwen, etc.)

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>
<|im_start|>user
What language do they speak there?<|im_end|>
<|im_start|>assistant
```

The model generates from the final `<|im_start|>assistant` token onward. `<|im_end|>` is the end-of-turn token; generation stops when the model produces it.

### LLaMA-2 Chat Format (Meta)

```
<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

What is the capital of France? [/INST] The capital of France is Paris. </s><s>[INST] What language do they speak there? [/INST]
```

Key differences from ChatML:
- System prompt embedded in the first user turn via `<<SYS>>...<</SYS>>`
- `[INST]...[/INST]` wraps user turns
- `<s>` and `</s>` mark sequence boundaries per turn
- No explicit `<|im_end|>` equivalent; `</s>` plays the role of turn terminator

### Mistral Instruct v0.1/v0.2 Format

```
<s>[INST] What is the capital of France? [/INST] The capital of France is Paris.</s>[INST] What language do they speak there? [/INST]
```

Mistral's format has **no system prompt** in the original v0.1 design — a deliberate choice to avoid over-constraining the model. v0.3 added system prompt support.

### Applying Templates via HuggingFace

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "What language do they speak there?"},
]

# apply_chat_template handles all formatting and special tokens automatically
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,  # adds the final assistant turn prompt
)
print(text)
```

Using `apply_chat_template` is strongly preferred over manual string formatting. It:
- Uses the tokenizer's built-in template (defined in `tokenizer_config.json`)
- Handles edge cases (e.g., missing system prompt)
- Ensures consistency between fine-tuning and inference

---

## Instruction Diversity: Task Coverage and Quality

The single most important factor in instruction dataset quality is **task diversity** — not the number of examples per task, but the breadth of task types covered.

Evidence from FLAN and subsequent work shows that:

$$\text{Performance}_\text{zero-shot} \propto \log(\text{num\_tasks})$$

Adding the 100th task type improves generalization more than doubling the examples per existing task. The intuition: each new task type teaches the model a new dimension of instruction interpretation. At some point (roughly 1,000+ task types), returns diminish.

**The axes of diversity**:
- **Format diversity**: questions, commands, fill-in-the-blank, role-play, step-by-step
- **Task type diversity**: classification, generation, summarization, translation, code, math, reasoning
- **Domain diversity**: science, law, medicine, creative writing, programming
- **Complexity diversity**: single-hop vs multi-hop, simple vs multi-constraint instructions
- **Length diversity**: short instructions, long instructions, multi-paragraph contexts

A dataset scoring high on all axes but with only 1,000 examples will outperform a 100,000-example dataset that is homogeneous in task type and format.

---

## Evaluation: Measuring Instruction Following

Instruction following is hard to measure because outputs are free-form. Common evaluation strategies:

**Format adherence**: Does the output match the requested format (JSON, bullet list, specific length)? Easy to measure automatically.

**Task accuracy**: For tasks with ground truth (classification, QA), measure accuracy directly.

**Instruction-following rate**: For tasks without ground truth, use an LLM judge (GPT-4) to score whether the response actually followed the instruction, on a Likert scale. MT-Bench and AlpacaEval use this approach.

**Human preference**: A/B comparison by human raters between the instruction-tuned model and a baseline. Gold standard but expensive.

**Benchmark regression tracking**: Monitor MMLU, HellaSwag, GSM8K throughout fine-tuning to detect capability regression. The goal is improved instruction following without degraded general capabilities.

For the specific evaluation of format diversity (does the model handle instructions in many phrasings?), test the same underlying task expressed in 5+ different instruction formulations and measure variance in output quality. High variance indicates the model is pattern-matching to specific phrasings rather than genuinely parsing instructions.

---

## How Instruction Tuning Enables In-context Learning Without Few-shot Examples

A key practical benefit of instruction tuning is that it largely eliminates the need for few-shot examples in the prompt. Consider the difference:

**Without instruction tuning** (raw pretrained model):
```
Text: "I loved this film, it was fantastic!"
Sentiment: Positive

Text: "The acting was dreadful."
Sentiment: Negative

Text: "An average movie, nothing special."
Sentiment:
```
The model requires examples to understand what is being asked.

**With instruction tuning**:
```
Classify the sentiment of the following text as Positive, Negative, or Neutral.

Text: "An average movie, nothing special."
Sentiment:
```
The instruction itself conveys the task. No examples needed.

**Why this works**: Instruction tuning teaches the model to use the instruction as a *program specification*. Rather than inferring the task from examples (which is what in-context learning does), the model reads the specification directly. This is more efficient (shorter prompts), more robust (no sensitivity to example ordering or phrasing), and more generalizable (the model can follow specifications for entirely new tasks).

The relationship between instruction tuning and in-context learning is complementary, not competitive. Instruction-tuned models still benefit from few-shot examples for very complex tasks. But the *baseline* capability — handling simple, clearly specified tasks — no longer requires examples, freeing up context window space for more substantive content.

---

## References

- Wei, J. et al. (2021). *Finetuned Language Models Are Zero-Shot Learners* (FLAN). ICLR 2022.
- Chung, H.W. et al. (2022). *Scaling Instruction-Finetuned Language Models* (FLAN-T5/FLAN-PaLM). JMLR.
- Taori, R. et al. (2023). *Alpaca: A Strong, Replicable Instruction-Following Model*. Stanford CRFM.
- Chiang, W.-L. et al. (2023). *Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90% ChatGPT Quality*. LMSYS Blog.
- Mukherjee, S. et al. (2023). *Orca: Progressive Learning from Complex Explanation Traces of GPT-4*. arXiv:2306.02707.
- Xu, C. et al. (2023). *WizardLM: Empowering Large Language Models to Follow Complex Instructions*. ICLR 2024.
- Wang, Y. et al. (2022). *Self-Instruct: Aligning Language Models with Self-Generated Instructions*. ACL 2023.
- OpenAI. (2022). *ChatML format specification*. https://github.com/openai/openai-python/blob/main/chatml.md

---

*← [Supervised Fine-Tuning](./supervised-fine-tuning.md) | → [Parameter-Efficient Fine-Tuning](./parameter-efficient-fine-tuning.md)*

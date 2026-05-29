# Chain-of-Thought Prompting

## Table of Contents

1. [Motivation: Why LLMs Struggle with Multi-Step Reasoning](#motivation)
2. [Chain-of-Thought (CoT)](#chain-of-thought-cot)
3. [Zero-Shot CoT](#zero-shot-cot)
4. [Automatic CoT](#automatic-cot)
5. [Self-Consistency](#self-consistency)
6. [Tree of Thoughts](#tree-of-thoughts)
7. [Graph of Thoughts](#graph-of-thoughts)
8. [Program of Thoughts](#program-of-thoughts)
9. [Least-to-Most Prompting](#least-to-most-prompting)
10. [Step-Back Prompting](#step-back-prompting)
11. [ReAct (Preview)](#react-preview)
12. [Limitations](#limitations)
13. [Choosing a Technique](#choosing-a-technique)
14. [References](#references)

---

## Motivation

Standard language model decoding produces one token at a time, each conditioned on all prior tokens. For a multi-step reasoning problem like "If a train travels at 60 mph and the station is 45 miles away, how long is the journey?", the model must somehow collapse all intermediate arithmetic into a single direct output token. This is a severe constraint: the model's intermediate computation is bounded by its residual stream capacity, not by the depth of reasoning the problem requires.

**Chain-of-thought prompting** sidesteps this bottleneck by externalizing intermediate reasoning into the context itself. Each reasoning step becomes tokens — tokens the model then conditions on when generating the next step. The context window becomes a scratch pad, and the effective computation depth grows with generated length rather than being fixed by model width.

The empirical effect is striking: CoT prompting with PaLM-540B achieved 56.9% on the GSM8K grade-school math benchmark, up from 17.9% with standard prompting — a 3× improvement, achieved purely by changing the prompt format.

---

## Chain-of-Thought (CoT)

### The Original Technique

Wei et al. (2022) introduced few-shot CoT: include examples in which the reasoning trace is written out before the final answer.

```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?

A: Roger started with 5 balls. He bought 2 × 3 = 6 more balls.
5 + 6 = 11. The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and
bought 6 more, how many apples do they have?

A:
```

The model, seeing the pattern (problem → reasoning trace → "The answer is N"), generates its own reasoning trace before the answer. This is not rule-following — it is in-context learning of a reasoning style.

### Why It Works

Three complementary explanations:

1. **Computation depth**: Each reasoning step is a forward pass; chaining them lets the model perform deeper effective computation than a single step allows.
2. **Error localization**: Writing out steps reveals where errors occur, and the model implicitly self-corrects by seeing that an intermediate step leads to an inconsistent state.
3. **Training distribution shift**: Reasoning traces with intermediate steps appear in math textbooks, worked examples, and Stack Overflow explanations — domains well-represented in training data. CoT moves the generation into a high-probability region for correct reasoning patterns.

### Emergence Threshold

CoT is primarily effective in models with ≥ ~100B parameters (or equivalent capability). Smaller models produce incoherent reasoning traces that can actively mislead. The threshold is not sharp — it varies by task — but as a rule of thumb, if you are using a model smaller than GPT-3.5 class, CoT may not help.

This is an **emergent ability**: not a gradual improvement but a phase transition. The model either has the capability to follow multi-step reasoning or it does not.

### CoT Prompt Design

```
# System message
You are a careful mathematical reasoner. Solve problems step by step,
showing all work. State the final answer at the end as:
"The answer is: [answer]"

# Few-shot examples
Q: A store has 48 apples. They sell 1/3 of them in the morning and
   half of the remainder in the afternoon. How many are left?

A: The store starts with 48 apples.
   Morning: sold 48 × (1/3) = 16 apples. Remaining: 48 - 16 = 32.
   Afternoon: sold 32 × (1/2) = 16 apples. Remaining: 32 - 16 = 16.
   The answer is: 16

Q: {{problem}}
A:
```

---

## Zero-Shot CoT

Kojima et al. (2022) discovered that appending "Let's think step by step" to a prompt dramatically improves reasoning accuracy on math and logic benchmarks — without any examples.

```
Q: A bat and a ball cost $1.10 together. The bat costs $1 more than
   the ball. How much does the ball cost?

Let's think step by step.
```

The phrase activates the model's internalized representation of worked-problem style. The intuitive answer ($0.10) is wrong; CoT leads the model through: if ball = $x, bat = $x + $1, total = $2x + $1 = $1.10, so $x = $0.05.

**Other effective zero-shot CoT triggers**:
- "Let's work through this carefully."
- "Let's solve this step by step."
- "Take a deep breath and think step by step." (Large et al., 2023 — this phrasing improved GPT-4 performance on math benchmarks by ~4 percentage points)
- "First, let me understand what the problem is asking..."

**Two-stage zero-shot CoT**: The original Kojima et al. approach uses two sequential prompts:
1. `[Problem] Let's think step by step.` → generates reasoning
2. `[Problem] [Reasoning] Therefore, the answer is:` → extracts final answer

This prevents the model from abandoning the reasoning trace before reaching the answer.

---

## Automatic CoT

Manual CoT example curation is labor-intensive. Auto-CoT (Zhang et al., 2022) automates it:

1. Cluster training questions by embedding similarity.
2. Sample one representative question per cluster.
3. Generate reasoning traces for each using zero-shot CoT ("Let's think step by step").
4. Use these auto-generated (question, reasoning, answer) triples as few-shot examples.

The key insight: diverse questions (from different clusters) produce diverse reasoning patterns, reducing the chance that all examples reinforce the same failure mode. Auto-CoT achieves performance close to hand-crafted CoT while requiring no human annotation.

---

## Self-Consistency

Wang et al. (2022) identified that greedy CoT is brittle: a single reasoning path can go wrong and confidently produce an incorrect answer. **Self-consistency** samples multiple reasoning paths and takes a majority vote.

### Algorithm

```
for i in range(k):  # k = 20–40 typically
    reasoning_i, answer_i = model.generate(problem, temperature=0.7)
final_answer = majority_vote([answer_0, answer_1, ..., answer_{k-1}])
```

### Why It Works

Correct reasoning paths tend to converge on the same answer because the mathematical or logical constraints of the problem strongly determine the result. Incorrect reasoning paths fail in diverse ways — there are many ways to be wrong, but typically one way to be right. Majority voting amplifies the signal from correct paths.

**Empirical improvement**: Self-consistency improves over greedy CoT by 10–17% on arithmetic and commonsense reasoning benchmarks. The improvement scales with $k$ but with diminishing returns beyond ~20 samples.

**Cost**: $k$ samples means $k$ times the inference cost. At production scale, self-consistency is expensive. It is most justified when:
- Correctness is critical (medical, legal, financial).
- The task is one where CoT errors are diverse but correct reasoning converges.
- You can afford the latency/cost.

**Weighted voting**: Rather than uniform majority vote, you can weight votes by the model's confidence (logit scores of the answer tokens), though this adds complexity without consistent gains.

---

## Tree of Thoughts

Yao et al. (2023) generalized CoT from a linear chain of steps to a **tree search** over reasoning steps. At each step, the model generates multiple candidate next steps ("thoughts"), evaluates them, and searches the tree using BFS or DFS.

### Algorithm

```
State: (problem, reasoning_so_far)
Expansion: sample k candidate next steps from current state
Evaluation: for each candidate, ask model: "Is this step correct? Rate 1–10."
Search: BFS (explore all nodes at current depth before going deeper)
      or DFS (explore one path deeply before backtracking)
Termination: reach final answer or exceed step budget
```

### Prompt Structure

**Generation prompt** (produce candidate next steps):
```
Problem: {{problem}}
Current reasoning: {{reasoning_so_far}}

Generate 3 possible next steps in the reasoning. List them as:
Step A: ...
Step B: ...
Step C: ...
```

**Evaluation prompt** (score each candidate):
```
Problem: {{problem}}
Proposed reasoning step: {{step}}

Is this step logically correct and helpful toward solving the problem?
Rate from 1 (incorrect/unhelpful) to 10 (correct and helpful).
Rating:
```

### When ToT Outperforms CoT

ToT is most valuable when:
- The problem has genuine branching structure (mathematical proof steps, planning tasks, puzzles).
- A single wrong step in a linear chain is catastrophic.
- The search space is tractable (a few steps, a few branches per step).

**Benchmark**: On the 24-point game (arrange 4 numbers with arithmetic to make 24), ToT achieved 74% success vs. 4% for standard CoT.

**Cost**: ToT multiplies inference calls by branching factor × depth. For production use, this typically means restricting it to low-depth problems or using smaller models for evaluation.

---

## Graph of Thoughts

Besta et al. (2023) generalized ToT to directed graphs, allowing:

- **Aggregation**: merge outputs from multiple reasoning paths (analogous to ensemble combination).
- **Refinement cycles**: revisit and improve a previous thought node.
- **Parallel exploration**: multiple independent reasoning chains whose results are later combined.

The graph structure is appropriate for tasks that are inherently graph-structured: planning problems where subgoals can be achieved in parallel, multi-document synthesis where partial answers from different documents are merged, or iterative refinement tasks.

In practice, GoT is more complex to implement and the gains over ToT are task-specific. It is primarily a research framework demonstrating the generality of thought-graph architectures.

---

## Program of Thoughts

Chen et al. (2022) — **Program of Thoughts (PoT)** — replaces natural language reasoning chains with code. The model generates Python (or another language) that, when executed, computes the answer.

```
Q: A rectangle has length 3x + 2 and width 2x - 1. If the perimeter
   is 38, find the area.

A:
```python
# Solve for x from perimeter equation
# Perimeter = 2*(length + width) = 38
# 2*((3x+2) + (2x-1)) = 38 → 2*(5x+1) = 38 → 10x+2=38 → x=3.6
x = (38/2 - 1) / 5
length = 3*x + 2
width = 2*x - 1
area = length * width
print(f"x = {x}, area = {area}")
```

**Execution**: The generated code is executed by a Python interpreter; the output is the final answer.

### Why PoT Outperforms CoT on Math

1. **Exact arithmetic**: Code execution is numerically exact; CoT arithmetic is done by the language model and makes errors on multi-digit operations.
2. **Separation of concerns**: The model focuses on problem setup and formula derivation; the interpreter handles computation.
3. **Verifiability**: Code can be inspected and debugged.

**PAL** (Gao et al., 2022 — Program-Aided Language models) is the related approach that generates code with natural language comments as the reasoning trace.

**Limitations**: PoT requires a code execution environment, is vulnerable to code generation errors (syntax errors, wrong library usage), and does not generalize to non-computational tasks.

---

## Least-to-Most Prompting

Zhou et al. (2022) — **Least-to-Most** — decomposes a complex problem into sub-problems and solves them sequentially, each solution informing the next.

### Two-Stage Process

**Stage 1 — Decomposition**:
```
To solve this problem, I need to first solve these sub-problems:
{{list of sub-problems in order from simplest to hardest}}
```

**Stage 2 — Sequential solving**:
```
Sub-problem 1: {{simplest sub-problem}}
Solution: {{solution}}

Sub-problem 2: {{next sub-problem}} (using the fact that {{solution 1}})
Solution: {{solution}}

...

Main problem: {{original problem}} (using {{all sub-solutions}})
Solution:
```

### Why It Outperforms Standard CoT

Standard CoT works well when intermediate steps are independent. It fails when sub-problems have complex dependencies. Least-to-Most makes dependencies explicit by feeding solved sub-problems into subsequent prompts as established facts.

**Benchmark**: On the SCAN compositional generalization benchmark, Least-to-Most achieved 99.7% accuracy vs. 16.2% for standard CoT.

### Example

```
Main problem: A farmer has cows and chickens. He counts 50 heads and
130 legs. How many cows does he have?

Decomposition:
1. How do we represent the number of legs in terms of cows (c) and
   chickens (k)?
2. How do we set up the system of equations?
3. How do we solve for c?

Step 1: Cows have 4 legs, chickens have 2. Total legs: 4c + 2k = 130.
Step 2: Total heads: c + k = 50. So k = 50 - c.
Step 3: Substituting: 4c + 2(50 - c) = 130 → 4c + 100 - 2c = 130
        → 2c = 30 → c = 15.

The farmer has 15 cows.
```

---

## Step-Back Prompting

Zheng et al. (2023) — **Step-Back Prompting** — asks the model to first identify the abstract principle or concept needed to solve a problem, then applies that principle to the specific instance.

```
# Step 1: Abstract principle retrieval
Q: What is the general physics principle needed to solve problems
   about projectile motion under gravity?
A: The motion can be decomposed into independent horizontal (constant
   velocity) and vertical (constant acceleration due to gravity)
   components. The equations of motion under constant acceleration apply.

# Step 2: Apply principle to specific problem
Q: A ball is thrown horizontally at 10 m/s from a cliff 45m high.
   How far from the base does it land?
A: Applying the principle: vertical fall time t: 45 = ½ × 9.8 × t²
   → t ≈ 3.03 s. Horizontal distance: 10 × 3.03 = 30.3 m.
```

Step-Back is particularly useful when the model has the relevant knowledge but fails to retrieve it given the specific problem framing. Stepping back to the principle level ensures the model accesses the right knowledge before tackling the specific case.

---

## ReAct (Preview)

ReAct (Yao et al., 2022) interleaves **Re**asoning and **Act**ion: the model alternates between generating reasoning traces (Thought:) and taking actions (Action:) like calling tools, which return observations (Observation:).

```
Thought: I need to find the population of Tokyo to answer this question.
Action: search("Tokyo population 2024")
Observation: Tokyo's population is approximately 13.96 million in 2024.
Thought: Now I can compare with the other cities mentioned.
Action: search("New York City population 2024")
Observation: New York City's population is approximately 8.26 million.
Thought: Tokyo has the larger population. I can now answer.
Final Answer: Tokyo, with ~14M vs NYC's ~8.3M.
```

ReAct is covered in depth in [../agents/agentic-frameworks.md](../agents/agentic-frameworks.md).

---

## Limitations

Despite their power, CoT techniques have important failure modes:

### 1. Confident but Wrong

CoT can produce a fully coherent, well-structured reasoning trace that leads to an incorrect answer. The model does not "know" when its steps are wrong — it generates tokens that look like correct reasoning steps. This is particularly dangerous when the answer is used without verification.

### 2. Factual Recall Is Not Improved

CoT improves reasoning over known facts but does not improve the accuracy of those facts. If the model has a wrong belief ("The capital of Australia is Sydney"), CoT may reason correctly from that wrong premise to a confidently stated wrong answer. Retrieval-augmented generation ([retrieval-augmented-generation.md](./retrieval-augmented-generation.md)) is the solution for factual errors, not CoT.

### 3. Length Bias and Verbosity

Models can generate verbose reasoning to appear thorough without actually performing useful computation. Self-consistency partially mitigates this by checking whether diverse paths converge, but individual CoT traces can still be misleading.

### 4. Emergent Threshold

Below ~100B parameters, CoT typically hurts performance — the model generates plausible-looking but incorrect reasoning, which anchors the final answer to an incorrect trajectory.

### 5. Overthinking on Simple Problems

For facts and simple lookups, CoT adds latency and tokens without benefit. Use zero-shot direct answers for simple queries; reserve CoT for tasks with genuine multi-step structure.

---

## Choosing a Technique

| Task Type | Recommended Approach |
|-----------|---------------------|
| Single-step fact recall | Zero-shot direct |
| Multi-step arithmetic | PoT (code execution) |
| Math word problems | Self-consistent CoT or PoT |
| Logical reasoning / puzzles | Tree of Thoughts |
| Complex planning | Tree of Thoughts / Least-to-Most |
| Multi-document reasoning | Least-to-Most |
| Domain knowledge application | Step-Back + CoT |
| Tool-augmented reasoning | ReAct |
| Any reasoning, higher reliability | Self-consistency over CoT |

---

## References

- Wei, J. et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." NeurIPS. [Original CoT paper]
- Kojima, T. et al. (2022). "Large Language Models are Zero-Shot Reasoners." NeurIPS. [Zero-shot CoT "Let's think step by step"]
- Wang, X. et al. (2022). "Self-Consistency Improves Chain of Thought Reasoning in Language Models." ICLR 2023. [Self-consistency]
- Zhang, Z. et al. (2022). "Automatic Chain of Thought Prompting in Large Language Models." ICLR 2023. [Auto-CoT]
- Yao, S. et al. (2023). "Tree of Thoughts: Deliberate Problem Solving with Large Language Models." NeurIPS. [ToT]
- Besta, M. et al. (2023). "Graph of Thoughts: Solving Elaborate Problems with Large Language Models." AAAI 2024. [GoT]
- Chen, W. et al. (2022). "Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks." TMLR. [PoT]
- Gao, L. et al. (2022). "PAL: Program-aided Language Models." ICML 2023. [PAL]
- Zhou, D. et al. (2022). "Least-to-Most Prompting Enables Complex Reasoning in Large Language Models." ICLR 2023. [Least-to-Most]
- Zheng, H. et al. (2023). "Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models." ICLR 2024. [Step-Back]
- Yao, S. et al. (2022). "ReAct: Synergizing Reasoning and Acting in Language Models." ICLR 2023. [ReAct]
- Large, J. et al. (2023). "Are Large Language Models Zero-Shot Human Simulators?" arXiv. ["Take a deep breath" prompting]

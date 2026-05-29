# Agentic Frameworks

## Table of Contents

1. [What Makes a System Agentic](#1-what-makes-a-system-agentic)
2. [ReAct: Reasoning + Acting](#2-react-reasoning--acting)
3. [Reflexion: Verbal Reinforcement Learning](#3-reflexion-verbal-reinforcement-learning)
4. [Plan-and-Execute](#4-plan-and-execute)
5. [Scratchpad and Working Memory](#5-scratchpad-and-working-memory)
6. [MRKL Systems](#6-mrkl-systems)
7. [OpenAI Assistants API](#7-openai-assistants-api)
8. [Toolformer](#8-toolformer)
9. [Agent Evaluation Benchmarks](#9-agent-evaluation-benchmarks)
10. [Failure Modes](#10-failure-modes)
11. [References](#references)

---

## 1. What Makes a System Agentic

The term "agentic" is used loosely. A useful framework: agentic systems exist on a spectrum defined by **degree of autonomy** and **scope of action**.

```
Single-pass LLM
    │
    ├─ + Tool use (one call, fixed)       → Augmented LLM
    │
    ├─ + Loop + conditional branching     → Basic Agent
    │
    ├─ + Multi-step planning              → Planning Agent
    │
    ├─ + Self-reflection / error recovery → Reflexive Agent
    │
    └─ + Long-horizon goals, multi-agent  → Autonomous Agent System
```

The minimal ingredients of an agentic system:

1. **Multi-step reasoning**: the LLM is invoked multiple times within a task, each invocation conditioned on outputs of previous steps
2. **External actions**: the system modifies world state beyond the context window (API calls, file writes, web searches)
3. **Observation-conditioned continuation**: action results re-enter context, influencing subsequent decisions
4. **Goal-directedness**: the system is trying to complete a task, not just respond to a prompt

Chain-of-Thought (CoT) is not agentic — it is single-pass. RAG is not fully agentic — tool use is fixed, not dynamic. A system where the model decides *which* tool to call and *when to stop* crosses the threshold.

---

## 2. ReAct: Reasoning + Acting

**Paper**: Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)

ReAct is the canonical framework for agentic LLMs. The key insight: **interleaving explicit reasoning traces with action selection** dramatically outperforms taking actions without reasoning (Act-only) and reasoning without actions (CoT-only) on knowledge-intensive and decision-making tasks.

### The Thought → Action → Observation Loop

```
User Task
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ Thought: I need to find the capital of the country where        │
│          Einstein was born. Let me search for Einstein first.   │
├─────────────────────────────────────────────────────────────────┤
│ Action: search("Albert Einstein birthplace")                    │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
Observation: "Albert Einstein was born on March 14, 1879, in Ulm,
             in the Kingdom of Württemberg in the German Empire."
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ Thought: Einstein was born in Ulm, which was in Germany.       │
│          Now I need to find the capital of Germany.            │
├─────────────────────────────────────────────────────────────────┤
│ Action: search("capital of Germany")                            │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
Observation: "Berlin is the capital and largest city of Germany."
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ Thought: Einstein was born in Germany; the capital is Berlin.  │
├─────────────────────────────────────────────────────────────────┤
│ Action: finish("Berlin")                                        │
└─────────────────────────────────────────────────────────────────┘
```

### Concrete ReAct Prompt Example

ReAct is typically implemented via few-shot prompting. Each example in the prompt shows the full Thought/Action/Observation sequence:

```
Solve a question answering task with interleaved Thought, Action, Observation steps.
Thought: reason about the current situation and what to do next.
Action: one of [search[query], lookup[term], finish[answer]]
Observation: result of the action (provided by the environment)

Example:
Question: Were Pavel Durov and Mark Zuckerberg born in the same country?

Thought: I need to find where Pavel Durov was born.
Action: search[Pavel Durov]
Observation: Pavel Durov is a Russian-born entrepreneur, born in Leningrad, USSR (now Saint Petersburg, Russia).

Thought: Durov was born in Russia (USSR). Now I need Zuckerberg's birthplace.
Action: search[Mark Zuckerberg birthplace]
Observation: Mark Zuckerberg was born in White Plains, New York, United States.

Thought: Durov was born in Russia; Zuckerberg in the United States. Different countries.
Action: finish[No]

---
Question: {user_question}

Thought:
```

The model is expected to complete the trace from `Thought:` onward, with the application intercepting each `Action:` line to execute the tool and injecting the `Observation:` result.

### Why Thought Traces Help

Without explicit reasoning, the model selects actions based on pattern matching the query to tool patterns. With Thought traces, the model:

1. **Decomposes the task**: breaking a complex query into sub-questions
2. **Tracks state**: noting what has been learned and what remains unknown
3. **Recovers from errors**: reasoning about why an action failed and choosing a different strategy
4. **Decides to stop**: recognizing when enough information has been gathered

Empirically, on HotpotQA (multi-hop reasoning), ReAct outperformed CoT by ~6 points and Act-only by ~14 points in exact match accuracy (Yao et al. 2023). The improvement is larger on tasks requiring multiple retrieval steps.

### ReAct vs Chain-of-Thought

| Dimension | Chain-of-Thought | ReAct |
|-----------|-----------------|-------|
| Knowledge source | Parametric only | Parametric + external tools |
| Number of LLM calls | 1 | N (per action taken) |
| Error recovery | Cannot — output already generated | Yes — observe and adapt |
| Hallucination risk | High on knowledge queries | Lower — grounded by observations |
| Cost | Low | Higher (multiple calls + tool costs) |
| Use case | Math, logic, coding | Knowledge-intensive QA, web tasks |

---

## 3. Reflexion: Verbal Reinforcement Learning

**Paper**: Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning" (NeurIPS 2023)

Reflexion extends ReAct by adding a **self-reflection** step when the agent fails. Instead of updating weights (which requires gradient descent), Reflexion stores verbal self-critiques in a persistent memory buffer and uses them in subsequent attempts.

### Architecture

```
                 ┌─────────────────────────┐
                 │   Episodic Memory       │
                 │   Buffer (text)         │
                 │  "Last attempt failed   │
                 │   because I searched    │
                 │   too broadly. Next     │
                 │   time, search for      │
                 │   the specific year."   │
                 └────────────┬────────────┘
                              │ retrieved at start
                              ▼
Task ──► Actor (ReAct loop) ──► Trajectory ──► Evaluator ──► Success?
                                                                │
                                              Yes ──────────── No
                                               │                │
                                            Return          Reflector
                                                           generates
                                                           critique ──► Memory Buffer
```

**Three components**:

1. **Actor**: the ReAct agent that takes actions and produces a trajectory
2. **Evaluator**: assesses whether the trajectory achieved the goal (binary or scalar reward)
3. **Reflector**: given the failed trajectory, generates a verbal critique of what went wrong and what to try differently

**Reflection generation prompt** (simplified):

```
You attempted the following task and failed.

Task: {task}
Your attempt:
{trajectory}
Outcome: FAILURE — {failure_reason}

Write a concise paragraph reflecting on why you failed and what you should do differently next time.
Reflection:
```

The reflection is stored in the episodic memory buffer. On the next attempt, the buffer is prepended to the prompt, giving the actor context about past failures.

### Why It Works

Reflexion performs verbal reinforcement: the reflection is a policy update in natural language. It is significantly cheaper than fine-tuning because no gradients are computed, and significantly more transparent because the "policy update" is human-readable text.

Results from the paper:
- AlfWorld (household task simulation): 97% task completion with Reflexion vs 75% for ReAct
- HotpotQA: +8% exact match over ReAct
- HumanEval (coding): 91% pass@1 with Reflexion vs 80% for standard prompting

**Limitations**:
- Memory buffer grows with attempts — context window limits how many reflections can be stored
- Reflection quality is bounded by the model's ability to diagnose its own failures
- Does not generalize reflections across tasks — each reflection is task-specific

---

## 4. Plan-and-Execute

**Reference**: LangChain's Plan-and-Execute agent; also related to "HuggingGPT" and "TaskMatrix"

ReAct interleaves planning and execution within the same context. Plan-and-Execute **separates them into distinct phases** and often uses distinct models for each:

```
┌────────────────────────────────────────────────────────────────┐
│                        PLANNER LLM                            │
│  Input: user goal                                             │
│  Output: ordered list of steps                                │
│                                                               │
│  Plan:                                                        │
│  1. Search for recent quarterly revenue of Apple Inc.         │
│  2. Search for recent quarterly revenue of Microsoft.         │
│  3. Calculate the revenue ratio Apple/Microsoft.              │
│  4. Summarize the comparison in 2 sentences.                  │
└───────────────────────────┬────────────────────────────────────┘
                            │ plan
                            ▼
┌────────────────────────────────────────────────────────────────┐
│                       EXECUTOR LLM                            │
│  For each step:                                               │
│    - Receive step description                                 │
│    - Receive results of previous steps                        │
│    - Execute step (may call tools)                            │
│    - Return result                                            │
└────────────────────────────────────────────────────────────────┘
```

**Re-planning**: if a step fails or produces an unexpected result, a re-planning call is triggered. The re-planner receives the original goal, the completed steps so far, and the failure, and produces a revised plan for remaining steps.

**Advantages over ReAct**:
- The plan is explicit and auditable — a human or verification system can inspect it before execution
- The planner can be a large, expensive model; the executor can be faster/cheaper
- Parallelism: independent steps in the plan can be executed simultaneously

**Disadvantages**:
- Plans can become stale — step 3 may depend on step 2 in a way that isn't known at planning time
- Extra LLM calls (planning + per-step execution) increase cost
- Planner may over-specify or under-specify steps

**When to use**: tasks with predictable structure (report generation, data pipeline, multi-source research). ReAct is better for unpredictable tasks where the next action depends heavily on the previous result.

---

## 5. Scratchpad and Working Memory

An agent often needs to maintain **intermediate state** that doesn't neatly fit as a tool result. Examples:

- A running tally of numbers found across multiple searches
- A growing outline for a document being written
- A list of already-visited URLs to avoid repetition
- Hypotheses under consideration

The **scratchpad** approach: give the model a special section in its response (or a dedicated tool) where it can write and read its own working notes:

```
[SCRATCHPAD]
- Apple Q3 2024 revenue: $85.8B (found in step 1)
- Microsoft Q3 2024 revenue: $64.7B (found in step 2)
- Ratio: 85.8 / 64.7 = 1.326
- Need to check if these are same quarter year-over-year or sequential
[END SCRATCHPAD]
```

The scratchpad is included in every subsequent prompt so the model can reference it. This is pure in-context memory — it grows with usage and is bounded by context window size.

**Structured working memory**: for tasks with known structure, a typed schema helps:

```python
from pydantic import BaseModel
from typing import Optional, List

class AgentState(BaseModel):
    goal: str
    completed_steps: List[str] = []
    key_findings: dict = {}
    current_hypothesis: Optional[str] = None
    next_action: Optional[str] = None
```

Serializing this to JSON and including it in the prompt gives the model a structured representation of its state. More reliable than free-form scratchpad for complex tasks.

---

## 6. MRKL Systems

**Paper**: Karpas et al., "MRKL Systems: A modular, neuro-symbolic architecture" (2022)

MRKL (Modular Reasoning, Knowledge and Language, pronounced "miracle") is an architecture where an LLM acts as a **router** that dispatches sub-tasks to specialized expert modules:

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│          Neural Router (LLM)                │
│  Analyzes query, selects expert module      │
└──────┬──────────┬──────────┬────────────────┘
       │          │          │
       ▼          ▼          ▼
  Calculator  Search    Code Runner
  (symbolic)  (neural)  (symbolic)
       │          │          │
       └──────────┴──────────┘
                  │
                  ▼
             Result aggregated
             by Router LLM
                  │
                  ▼
              Final answer
```

MRKL predates the modern tool-use API. The router LLM decides which expert to call via text generation (e.g., generating `CALCULATOR: 3 * 47 + 12`), and parsing rules extract the module name and argument.

Modern function-calling APIs are essentially a more robust implementation of MRKL routing — the key ideas are the same: specialized modules for different knowledge/capability types, with an LLM deciding how to route.

MRKL introduced the important insight that **not all sub-tasks benefit from neural solutions**. A symbolic calculator is more reliable for arithmetic than asking the LLM to compute. This "neuro-symbolic" principle remains relevant: use deterministic, exact systems for precise computation; use neural systems for understanding, generation, and routing.

---

## 7. OpenAI Assistants API

The Assistants API (2023) provides a managed abstraction for stateful agents, removing the need to manage conversation history manually.

**Key abstractions**:

```
Assistant ──── definition: model, instructions, tools, files
    │
    └─── Thread ──── persistent conversation (messages stored server-side)
             │
             └─── Run ──── one execution of the assistant on the thread
                      │
                      └─── Run Steps ──── individual tool calls within a run
```

**Thread**: a persistent message history stored by OpenAI. You add user messages to the thread; the assistant's responses are automatically appended. No need to manage a local `messages` list.

**Run lifecycle**:

```
queued → in_progress → [requires_action] → in_progress → completed
                            │
                    (submit tool outputs)
```

When the run enters `requires_action`, it means the model has made one or more tool calls. You retrieve the pending tool calls, execute them, and submit results:

```python
import openai, json, time

client = openai.OpenAI()

# Create assistant with tools
assistant = client.beta.assistants.create(
    model="gpt-4o",
    instructions="You are a research assistant. Use tools to answer questions accurately.",
    tools=[{"type": "code_interpreter"}, {"type": "file_search"}]
)

# Create a thread and add user message
thread = client.beta.threads.create()
client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="What is the standard deviation of [4, 8, 15, 16, 23, 42]?"
)

# Start a run
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id
)

# Poll until completion
while run.status in ("queued", "in_progress"):
    time.sleep(0.5)
    run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

    if run.status == "requires_action":
        tool_calls = run.required_action.submit_tool_outputs.tool_calls
        outputs = []
        for tc in tool_calls:
            result = run_tool(tc.function.name, json.loads(tc.function.arguments))
            outputs.append({"tool_call_id": tc.id, "output": result})
        run = client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread.id, run_id=run.id, tool_outputs=outputs
        )

# Retrieve final message
messages = client.beta.threads.messages.list(thread_id=thread.id)
print(messages.data[0].content[0].text.value)
```

**Built-in tools**: Code Interpreter (sandboxed Python execution), File Search (vector-indexed document retrieval). These require no implementation from you.

**Trade-offs**: The Assistants API trades flexibility for convenience. You cannot easily inspect or modify the full message list, cannot control exactly how the history is formatted, and are locked into OpenAI's infrastructure. For research or production systems requiring full control, the raw chat completions API with manual state management is often preferable.

---

## 8. Toolformer

**Paper**: Schick et al., "Toolformer: Language Models Can Teach Themselves to Use Tools" (NeurIPS 2023)

Toolformer takes a different approach: instead of prompting a general-purpose model to use tools, **fine-tune the model to self-learn when to call tools**. The model learns to insert API calls into its own text generation:

```
Original text:
"The population of Germany is about 84 million."

Toolformer-augmented:
"The population of Germany is about [Calculator(84000000)] 84 million."
```

**Self-supervised data generation**:

1. Use a large LLM (GPT-3) to annotate a text corpus with candidate API call positions via in-context learning
2. Filter: only keep annotations where calling the API *actually helps* predict the continuation (measured by perplexity reduction)
3. Fine-tune a smaller model on this filtered, annotated corpus

**Result**: a model that spontaneously inserts tool calls during generation without being explicitly prompted to do so — the tool use behavior is baked into the weights.

**Supported tools in the paper**: Calculator, Q&A (question answering API), Wikipedia search, Machine Translation, Calendar.

**Significance**: Toolformer demonstrated that tool use need not be a prompting artifact — it can emerge from fine-tuning in a mostly self-supervised manner. Smaller models (GPT-J 6.7B) with Toolformer-style fine-tuning outperformed much larger models (OPT 66B) on tasks requiring tool use.

**Limitation**: fine-tuning is required per tool set. If you add a new tool, you need to re-collect data and fine-tune. Modern API-based tool use (function calling) is more flexible since tools can be added at inference time.

---

## 9. Agent Evaluation Benchmarks

Evaluating agents is harder than evaluating single-pass models. Standard benchmarks measure accuracy on a fixed dataset. Agents require evaluating **behavior in interactive environments** — tasks where the agent must take sequences of actions and the environment responds dynamically.

### AgentBench

**Paper**: Liu et al., "AgentBench: Evaluating LLMs as Agents" (2023)

A multi-environment benchmark with 8 environments:
- Web browsing (WebShop, WebArena)
- Coding (HumanEval in interactive mode)
- OS interaction (shell commands)
- Database queries
- Game playing
- Knowledge graph operations

Evaluation metric: **task success rate** (binary per task). Models including GPT-4, Claude, open-source LLMs are tested. Key finding: GPT-4 significantly outperforms open-source models (~4x better), suggesting agent capability is strongly correlated with base model quality.

### WebArena

**Paper**: Zhou et al., "WebArena: A Realistic Web Environment for Building Autonomous Agents" (2023)

812 tasks across 5 realistic websites (e-commerce, Reddit-like forum, code repo, CMS, map). Tasks require realistic multi-step web interactions: search, filter, fill forms, navigate, compare information.

Success rate of best models at release: ~14-18% — deliberately hard. Web interaction requires understanding rendered HTML, tracking state across page loads, handling unexpected UI states.

### OSWorld

**Paper**: Xie et al., "OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments" (2024)

369 tasks requiring actual desktop computer interaction (screenshots, mouse clicks, keyboard) in Linux/macOS/Windows VMs. Evaluates vision-language model agents. Success rates of best models: ~20%.

### GAIA

**Paper**: Mialon et al., "GAIA: a benchmark for General AI Assistants" (2023)

Questions requiring multi-step reasoning with real-world tools — browsing, code execution, file parsing. 3 difficulty levels. Designed so a human can solve them in minutes but LLMs struggle. GPT-4 with plugins achieved ~30% on level 1, near 0% on level 3 at release.

---

## 10. Failure Modes

Understanding how agents fail is as important as understanding how they succeed. The failure modes are distinct from single-pass LLM failures.

### Infinite Loops

The agent calls the same tool repeatedly without progress. Causes: the model doesn't recognize that it's in a loop, or the stopping condition is not well-specified.

**Mitigation**: enforce a maximum step count. Track action history in the scratchpad; include a reminder like "If you've already tried this search, try different search terms." Check for repeated identical tool calls.

```python
MAX_STEPS = 20
step = 0
while not done and step < MAX_STEPS:
    step += 1
    ...
```

### Compounding Errors

An error in step 2 corrupts the agent's understanding, causing errors in steps 3, 4, 5. Because the agent reasons over its own prior outputs, errors propagate and amplify. A hallucinated intermediate fact becomes a premise for subsequent reasoning.

**Mitigation**: ground each step in retrieved/computed facts, not in the model's own generated text. Reflexion-style verification: periodically check intermediate results against the original goal.

### Over-confidence

The model asserts a tool result is correct when it is ambiguous or wrong. LLMs have a tendency to produce confident-sounding text regardless of certainty.

**Mitigation**: prompt the model to explicitly express uncertainty. Use multiple tools to cross-validate important facts. Include "If you are uncertain about a tool result, say so explicitly" in the system prompt.

### Action Hallucination

The model generates a plausible-looking tool call with arguments that don't correspond to anything real — a URL that doesn't exist, a function name it invented, arguments in wrong format.

**Mitigation**: validate tool calls before executing them. Use JSON Schema validation on arguments. Use `enum` types in schemas to constrain string values. Structured output / constrained decoding for generating tool call JSON.

### Sycophancy in Multi-Turn

The model changes its assessment of a tool result based on apparent user preference rather than the actual content. If the user seems displeased with a result, the model may re-interpret the same data more favorably.

**Mitigation**: keep tool result interpretation separate from user interaction. Use a critic sub-agent to independently assess results.

### Scope Creep

The agent takes on sub-tasks not requested by the user, expanding scope autonomously. A user asks to "find a restaurant" and the agent books it, emails a confirmation, and adds it to the calendar without being asked.

**Mitigation**: conservative default behavior for side-effect tools. Explicit confirmation before irreversible actions. Define the agent's scope in the system prompt with explicit boundaries.

---

## References

- Yao, S. et al. (2023). *ReAct: Synergizing Reasoning and Acting in Language Models*. ICLR 2023.
- Shinn, N. et al. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*. NeurIPS 2023.
- Schick, T. et al. (2023). *Toolformer: Language Models Can Teach Themselves to Use Tools*. NeurIPS 2023.
- Karpas, E. et al. (2022). *MRKL Systems: A modular, neuro-symbolic architecture that combines large language models, external knowledge sources and discrete reasoning*. arXiv:2205.00445.
- Liu, X. et al. (2023). *AgentBench: Evaluating LLMs as Agents*. arXiv:2308.03688.
- Zhou, S. et al. (2023). *WebArena: A Realistic Web Environment for Building Autonomous Agents*. arXiv:2307.13854.
- Mialon, G. et al. (2023). *GAIA: a benchmark for General AI Assistants*. arXiv:2311.12983.
- Xie, T. et al. (2024). *OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments*. arXiv:2404.07972.
- Chase, H. (2022). *LangChain*. https://github.com/langchain-ai/langchain

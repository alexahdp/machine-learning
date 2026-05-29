# Multi-Agent Systems

## Table of Contents

1. [Why Multi-Agent](#1-why-multi-agent)
2. [Multi-Agent Architectures](#2-multi-agent-architectures)
3. [Role Specialization](#3-role-specialization)
4. [Debate and Self-Play](#4-debate-and-self-play)
5. [Orchestration Frameworks](#5-orchestration-frameworks)
6. [Communication Protocols](#6-communication-protocols)
7. [Challenges](#7-challenges)
8. [Evaluation](#8-evaluation)
9. [References](#references)

---

## 1. Why Multi-Agent

A single agent is bounded by the context window, the model's capability, and the sequential nature of its reasoning. Multi-agent systems address these limits through three mechanisms:

**Parallelism**: independent sub-tasks can be assigned to multiple agents working simultaneously. A research task that requires searching 10 sources takes 10x longer with a single sequential agent than with 10 parallel agents.

**Specialization**: different agents can be optimized for different tasks. A code-generation agent uses a system prompt tuned for programming, has access to a code execution tool, and is evaluated on test passage rates. A research agent has web search tools and is tuned for information synthesis. Specialization improves per-task quality beyond what a single generalist prompt achieves.

**Verification through critique**: one agent producing an answer and a second agent independently evaluating it catches errors that self-consistency checking within a single model cannot. The evaluator agent is not anchored by the generator's reasoning path.

**Context isolation**: each agent has its own context window. An orchestrator managing a large workflow can delegate deep sub-tasks to agents that go deep on their portion without polluting the orchestrator's context with low-level details.

The cost: coordination overhead, error propagation, consistency management, and significantly higher API costs. Multi-agent is not always better — for straightforward tasks, the added complexity introduces more failure modes than it solves.

---

## 2. Multi-Agent Architectures

### Sequential Pipeline

Agents form a linear chain. Each agent's output is the next agent's input. Analogous to a Unix pipe.

```
User Query
    │
    ▼
┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐
│  Agent A  │──► │  Agent B  │──► │  Agent C  │──► │  Agent D  │──► Output
│ (Research)│    │ (Analyst) │    │  (Writer) │    │ (Editor)  │
└───────────┘    └───────────┘    └───────────┘    └───────────┘
```

**When to use**: tasks with well-defined sequential stages where each stage's input is cleanly the previous stage's output. Report generation: (research → outline → draft → edit → format).

**Failure mode**: errors from early stages propagate. If Agent A misunderstands the query, every subsequent agent operates on a flawed premise.

### Hierarchical (Orchestrator + Sub-agents)

An orchestrator LLM receives the top-level goal and breaks it into sub-tasks, dispatching each to a specialized sub-agent. Sub-agents may themselves orchestrate further sub-agents.

```
                    ┌─────────────────────┐
                    │    Orchestrator     │
                    │  (high-level plan)  │
                    └──────────┬──────────┘
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
     ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
     │  Code Agent   │ │ Search Agent  │ │  Data Agent   │
     │ (Python exec) │ │ (web search)  │ │ (SQL queries) │
     └───────────────┘ └───────────────┘ └───────────────┘
             │                 │                 │
             └─────────────────┴─────────────────┘
                               │ results
                               ▼
                        ┌─────────────────────┐
                        │    Orchestrator     │
                        │   (synthesize,      │
                        │    respond)         │
                        └─────────────────────┘
```

The orchestrator sees high-level results; sub-agents see only their specific sub-task. This is a **divide-and-conquer** pattern.

**When to use**: complex tasks with heterogeneous sub-tasks requiring different tools and expertise. Software development (plan → code → test → deploy), research workflows.

**Key design question**: how much information does the orchestrator need from each sub-agent? Over-returning detail re-introduces the context bloat problem. Sub-agents should return summaries, not raw output.

### Parallel

Multiple agents work simultaneously on independent tasks. An aggregator collects results when all are done.

```
               ┌──────────────────────────────────────┐
               │           Dispatcher                 │
               └────┬─────────┬─────────┬─────────────┘
                    │         │         │
                    ▼         ▼         ▼
             ┌──────────┐ ┌──────────┐ ┌──────────┐
             │ Agent 1  │ │ Agent 2  │ │ Agent 3  │
             │(source A)│ │(source B)│ │(source C)│
             └──────────┘ └──────────┘ └──────────┘
                    │         │         │
                    └─────────┼─────────┘
                              ▼
                       ┌─────────────┐
                       │  Aggregator │
                       │  (merge,    │
                       │   deduplicate│
                       │   synthesize)│
                       └─────────────┘
```

**When to use**: tasks decomposable into independent sub-problems (search multiple databases simultaneously, run multiple experiments in parallel).

**Implementation**: use `asyncio` or `concurrent.futures.ThreadPoolExecutor` to issue API calls concurrently:

```python
import asyncio
import openai

async def run_agent(task: str, system_prompt: str) -> str:
    client = openai.AsyncOpenAI()
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task}
        ]
    )
    return response.choices[0].message.content

async def parallel_research(query: str) -> dict:
    sources = [
        ("academic", "You are a research assistant specializing in academic papers. Find relevant academic findings."),
        ("news", "You are a news analyst. Find recent news articles and developments."),
        ("technical", "You are a technical expert. Focus on technical details and implementation."),
    ]
    tasks = [run_agent(query, prompt) for _, prompt in sources]
    results = await asyncio.gather(*tasks)
    return dict(zip([name for name, _ in sources], results))
```

### Debate

Agents take opposing positions on a question and argue. A third agent (arbitrator) or a final consensus round produces the answer.

```
Question
    │
    ├──────────────────────────────────┐
    │                                  │
    ▼                                  ▼
┌──────────┐                    ┌──────────┐
│ Agent A  │                    │ Agent B  │
│ (argues  │ ◄── challenge ──►  │ (argues  │
│  "Yes")  │                    │  "No")   │
└────┬─────┘                    └─────┬────┘
     │         Round 2               │
     │ A reads B's arg, refutes     │ B reads A's arg, refutes
     │                               │
     └───────────────┬───────────────┘
                     │ debate transcript
                     ▼
               ┌─────────────┐
               │ Arbitrator  │
               │ (or final   │
               │  consensus  │
               │  round)     │
               └─────────────┘
                     │
                     ▼
                  Answer
```

**Du et al. (2023)** showed that debate significantly improves accuracy on reasoning tasks (GSM8K math, MMLU knowledge) over single-model responses. The key mechanism: agents are forced to explicitly justify their reasoning and identify weaknesses in opposing arguments, which exposes logical errors.

### Network (Peer-to-Peer)

Agents communicate freely with each other in a shared message space. No fixed hierarchy. Any agent can message any other agent.

```
┌──────────┐ ◄────────────────► ┌──────────┐
│ Agent A  │                    │ Agent B  │
└────┬─────┘                    └─────┬────┘
     │ ▲                             │ ▲
     │ │                             │ │
     ▼ │                             ▼ │
┌──────────┐ ◄────────────────► ┌──────────┐
│ Agent C  │                    │ Agent D  │
└──────────┘                    └──────────┘
```

Flexible but hard to reason about and debug. Message routing can be unpredictable. Primarily a research architecture; most production systems use hierarchical or pipeline designs for controllability.

---

## 3. Role Specialization

Each agent in a multi-agent system has a **role** defined primarily through its system prompt and available tools. Role design is one of the most consequential decisions in multi-agent architecture.

### Core Role Taxonomy

**Planner**: decomposes high-level goals into actionable sub-tasks. System prompt: "You are a strategic planner. Given a complex task, produce a numbered, ordered list of concrete steps to achieve it. Focus on structure and completeness. Do not execute any steps yourself."

**Executor**: takes a specific task and executes it using available tools. System prompt: "You are an execution agent. You will be given a specific task. Use your tools to accomplish it. Report results concisely. Do not deviate from the assigned task."

**Critic / Verifier**: evaluates an output for correctness, completeness, or quality. System prompt: "You are a critical reviewer. Given an answer or output, identify factual errors, logical inconsistencies, and missing information. Be specific about what is wrong and why. Do not simply say 'looks good'."

**Summarizer**: condenses long outputs into concise summaries. System prompt: "You are a summarization specialist. Compress the provided content to its essential points. Preserve key facts, numbers, and conclusions. Aim for 20% of original length."

### Domain Expert Roles

For technical tasks, role specialization mirrors professional teams:

| Role | System Prompt Focus | Tools |
|------|--------------------|----|
| Code Agent | Software engineering, debugging, code review | Python exec, code search |
| Research Agent | Information synthesis, source evaluation | Web search, academic APIs |
| Math Agent | Formal mathematical reasoning | Calculator, symbolic math (SymPy) |
| Data Agent | Statistical analysis, visualization | SQL, Python (pandas/numpy) |
| QA Agent | Verification, test case generation | Code execution |

**System prompt engineering for roles**: be explicit about what the agent should and should not do. Vague role definitions lead to scope creep and inconsistent behavior:

```python
CODE_AGENT_SYSTEM_PROMPT = """
You are a senior software engineer. Your responsibilities:
- Write clean, well-commented Python code
- Debug code when given error messages
- Review code for correctness and efficiency

You do NOT:
- Make architectural decisions (escalate to the orchestrator)
- Communicate directly with users (report results to the orchestrator)
- Execute code without explicitly stating what it will do first

Output format: Always begin with a brief plan, then provide the code, then explain what it does.
"""
```

---

## 4. Debate and Self-Play

### Society of Mind Prompting

A single model can simulate multiple perspectives without using multiple API calls by explicitly asking it to reason from different viewpoints:

```
"Consider this question from three perspectives:
1. As a skeptic who doubts the claim
2. As an advocate who supports the claim
3. As a neutral arbiter who weighs both sides

Question: Is nuclear energy a net positive for climate change mitigation?

Skeptic: ...
Advocate: ...
Arbiter: ..."
```

This is cheap (one API call) but limited — the model is not actually running independent reasoning chains. The perspectives are correlated because they share weights and the same forward pass. Actual multi-agent debate avoids this correlation.

### Actual Multi-Agent Debate

**Paper**: Du et al., "Improving Factuality and Reasoning in Language Models through Multiagent Debate" (2023)

Protocol:

1. Each agent independently produces an initial response
2. Each agent reads all other agents' responses
3. Each agent produces a revised response, acknowledging and addressing others' arguments
4. Repeat for N rounds
5. Final answers are aggregated (majority vote or dedicated arbitrator)

```python
async def multi_agent_debate(question: str, num_agents: int = 3, rounds: int = 2) -> str:
    # Round 0: independent initial responses
    agent_responses = await asyncio.gather(*[
        run_agent(
            f"Answer this question: {question}",
            f"You are agent {i+1}. Reason carefully and give your best answer."
        )
        for i in range(num_agents)
    ])

    # Debate rounds
    for round_num in range(rounds):
        other_responses_prompt = "\n\n".join([
            f"Agent {i+1}'s answer: {resp}"
            for i, resp in enumerate(agent_responses)
        ])

        agent_responses = await asyncio.gather(*[
            run_agent(
                f"""Question: {question}

Other agents' answers:
{other_responses_prompt}

Your previous answer: {agent_responses[i]}

Review the other agents' arguments. Update your answer if you find their reasoning
compelling, or maintain your position with stronger justification if you disagree.""",
                f"You are agent {i+1} in round {round_num+1} of a debate."
            )
            for i in range(num_agents)
        ])

    # Aggregate: simple majority or arbitrator
    return arbitrate(question, agent_responses)
```

**Results** (Du et al. 2023): on GSM8K math benchmarks, 3-agent debate with 2 rounds improves accuracy by ~10% over single-model. The gain is larger on tasks requiring multi-step reasoning than on factual recall.

### Consensus Mechanisms

| Mechanism | Method | Use Case |
|-----------|--------|----------|
| Majority vote | Most common answer wins | Classification, multiple-choice |
| Weighted vote | Weight by agent confidence or past accuracy | When agent quality differs |
| Arbitrator LLM | Separate judge reads debate transcript | Open-ended questions |
| Delphi method | Anonymous initial answers, then discuss, revote | Sensitive topics |

---

## 5. Orchestration Frameworks

### AutoGen (Microsoft)

**Concept**: Conversational multi-agent framework. Agents communicate by sending messages to each other, just like a human conversation. The core abstraction is the `ConversableAgent`, and the key pattern is `UserProxyAgent` + `AssistantAgent`.

```
UserProxyAgent ◄──────────────────────────────────── AssistantAgent
   (executes      sends code for execution             (generates code,
    code, gets                                          plans, responds)
    feedback)     ──────────────────────────────────►
                  returns execution result + error
```

`UserProxyAgent` acts as the human proxy — it executes code returned by the assistant and sends back results. This enables automatic code debugging loops: assistant writes code → proxy executes → proxy sends traceback → assistant debugs.

```python
import autogen

config = {"config_list": [{"model": "gpt-4o", "api_key": "..."}]}

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=config,
    system_message="You are an expert Python programmer. Write code to solve the user's task."
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",    # fully automated
    max_consecutive_auto_reply=10,
    code_execution_config={"work_dir": "/tmp/autogen", "use_docker": True},
    is_termination_msg=lambda msg: "TASK COMPLETE" in msg.get("content", "")
)

# Start the conversation
user_proxy.initiate_chat(
    assistant,
    message="Compute the first 20 Fibonacci numbers and plot them."
)
```

**GroupChat**: AutoGen supports N-agent group chats where a `GroupChatManager` decides which agent speaks next (either round-robin, random, or LLM-based routing).

### CrewAI

**Concept**: agents are defined as members of a **crew** with explicit roles, goals, and backstories. Tasks are assigned to agents explicitly. A `Crew` coordinates execution.

```python
from crewai import Agent, Task, Crew, Process

researcher = Agent(
    role="Senior Research Analyst",
    goal="Find comprehensive, accurate information on {topic}",
    backstory="You are a veteran research analyst with 20 years of experience.",
    tools=[web_search_tool],
    verbose=True
)

writer = Agent(
    role="Technical Writer",
    goal="Write clear, well-structured reports based on research findings",
    backstory="You specialize in making complex topics accessible to expert audiences.",
    tools=[]  # no tools needed
)

research_task = Task(
    description="Research the current state of quantum computing, focusing on recent breakthroughs.",
    expected_output="A structured summary with key findings, 500-800 words.",
    agent=researcher
)

writing_task = Task(
    description="Write a technical report based on the research findings.",
    expected_output="A polished 1000-word technical report with sections and citations.",
    agent=writer,
    context=[research_task]  # depends on research task output
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential
)

result = crew.kickoff(inputs={"topic": "quantum error correction"})
```

CrewAI supports `Process.sequential` (linear pipeline) and `Process.hierarchical` (manager LLM orchestrates). Role-playing is central to the design — the model plays its character.

### LangGraph

**Concept**: agent workflows as **state machines** represented as graphs. Nodes are Python functions (or LLM calls); edges define transitions. State is a typed dictionary that flows through the graph.

```
                    ┌──────────────────┐
                    │    START node    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   LLM node       │◄──────────────────┐
                    │ (decide action)  │                   │
                    └────────┬─────────┘                   │
                             │                             │
                    ┌────────┴──────────┐                  │
                    │   Conditional     │                  │
                    │   edge router     │                  │
                    └──┬────────────────┘                  │
                       │               │                   │
                tool calls           done              tool results
                       │               │                   │
                       ▼               ▼                   │
               ┌──────────────┐  ┌──────────────┐         │
               │  Tool node   │  │    END node  │         │
               │(executes     │  └──────────────┘         │
               │ tool calls)  │                           │
               └──────────────┘                           │
                       │                                   │
                       └───────────────────────────────────┘
```

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]   # messages accumulate
    next_action: str

graph = StateGraph(AgentState)

graph.add_node("llm", call_llm)
graph.add_node("tools", execute_tools)

graph.set_entry_point("llm")

def should_call_tools(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END

graph.add_conditional_edges("llm", should_call_tools)
graph.add_edge("tools", "llm")   # after tools, go back to LLM

app = graph.compile()
result = app.invoke({"messages": [HumanMessage("Research quantum computing")], "next_action": ""})
```

**LangGraph's strength**: explicit state management, checkpointing (resume from any node), time-travel debugging (replay from any past state), human-in-the-loop (pause graph, get human input, resume).

### Swarm (OpenAI)

**Concept**: an experimental, lightweight framework for **agent handoffs**. An agent can transfer control to another agent by returning a new agent object. Minimal boilerplate; the key abstraction is the handoff.

```python
from swarm import Swarm, Agent

client = Swarm()

def transfer_to_billing():
    return billing_agent

triage_agent = Agent(
    name="Triage Agent",
    instructions="Route customer questions to the appropriate specialist.",
    functions=[transfer_to_billing]
)

billing_agent = Agent(
    name="Billing Agent",
    instructions="You handle billing and payment questions. Be precise about amounts and dates."
)

response = client.run(
    agent=triage_agent,
    messages=[{"role": "user", "content": "My last bill seems wrong."}]
)
```

When `triage_agent` calls `transfer_to_billing()`, control passes to `billing_agent`, which continues the conversation with the full history. **Handoff** is the central design primitive: agents are specialists that pass work to each other, not an orchestrator coordinating them from above.

**Comparison**:

| Framework | Mental model | Strengths | Limitations |
|-----------|-------------|-----------|-------------|
| AutoGen | Conversational agents | Code execution, debugging loops | Complex setup |
| CrewAI | Team with roles | Easy to read, role-playing | Less flexible routing |
| LangGraph | State machine / graph | Explicit state, checkpointing | More verbose |
| Swarm | Agent handoff | Simple, lightweight | Experimental, limited features |

---

## 6. Communication Protocols

### Structured Message Passing

Agents communicate via typed message objects rather than raw strings:

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class AgentMessage:
    sender: str
    recipient: str
    message_type: str          # "task", "result", "error", "handoff"
    content: str
    task_id: str
    requires_response: bool = True
    metadata: dict = None

# Example: orchestrator delegates to code agent
msg = AgentMessage(
    sender="orchestrator",
    recipient="code_agent",
    message_type="task",
    content="Write a Python function to compute edit distance between two strings",
    task_id="task_42",
    requires_response=True
)
```

Structured messages make routing, logging, and debugging tractable. Avoid passing raw strings between agents — you lose the ability to filter, prioritize, or route messages.

### Shared Memory (Blackboard Architecture)

Agents read and write to a shared state object rather than exchanging messages directly:

```
┌──────────────────────────────────────────────────────────────────┐
│                       Shared Blackboard                         │
│                                                                  │
│  task_queue: [task1, task2, task3]                              │
│  completed: {task1: result1}                                     │
│  agent_states: {A: "working on task2", B: "idle"}               │
│  global_context: {user_goal: ..., key_facts: {...}}             │
└──────┬───────────────┬────────────────────┬──────────────────────┘
       │               │                    │
       ▼               ▼                    ▼
  Agent A          Agent B             Agent C
  (reads task,     (reads task,        (reads results,
   writes result)   writes result)      synthesizes)
```

**Advantage**: any agent can see the full state without explicit message routing. **Disadvantage**: concurrent writes require locking; hard to scale.

### Turn-Based vs Event-Driven

**Turn-based**: agents take turns in a fixed order. Simple, predictable, easy to debug. Used in AutoGen group chats.

**Event-driven**: agents act when triggered by events (new message, tool result, timer). More efficient but harder to reason about. Suitable for production systems with variable latency per agent.

---

## 7. Challenges

### Error Propagation

In a pipeline A → B → C, an error in A corrupts B's input, which corrupts C's input. The final output can be wildly wrong with no clear signal of where the failure occurred.

**Mitigation**: each agent should validate its input before processing. Return a structured error object (not just wrong output) when input is malformed. The orchestrator should detect "wrong type" errors and re-run the upstream agent.

```python
class AgentResult:
    success: bool
    output: Optional[str]
    error: Optional[str]
    confidence: float   # 0-1; low confidence triggers verification
```

### Coordination Overhead

Agents spend time waiting for each other. The total wall-clock time of a sequential pipeline is the sum of each agent's latency, plus communication overhead. Parallel pipelines reduce this but require careful dependency analysis.

**Amdahl's Law applies**: if 20% of the work must be sequential, parallelizing the rest gives at most 5x speedup regardless of how many parallel agents you add.

### Consistency

Agents may contradict each other. Agent A concludes the meeting should be on Tuesday; Agent B schedules it for Wednesday. Without a consistency layer, the final output is incoherent.

**Mitigation**: designate one agent as the **single source of truth** for each domain. Write consensus decisions to a shared state that all agents read before making related decisions. Use an arbitration step to resolve contradictions before final output.

### Cost Multiplication

If a single agent task costs $0.01, a 5-agent system with 3 rounds of communication costs ~$0.15. With parallel searches, debate rounds, and verification, costs scale quickly.

**Cost management**:
- Use large/expensive models only for complex reasoning; use small/fast models for routing, formatting, summarization
- Cache identical or near-identical requests (semantic caching with embeddings)
- Set budget limits per task; abort and return partial results if exceeded
- Profile which agent roles consume most tokens; optimize those system prompts

### Debugging

Multi-agent systems fail in non-obvious ways. An incorrect final answer may be traceable to a single bad tool call 8 steps back.

**Tracing**: log every inter-agent message with: timestamp, sender, recipient, message type, token count, latency. Use distributed tracing tools (LangSmith, LangFuse, Phoenix by Arize) to visualize agent call graphs.

---

## 8. Evaluation

### GAIA Benchmark

**Paper**: Mialon et al., "GAIA: a benchmark for General AI Assistants" (2023)

Real-world multi-step tasks requiring web browsing, code execution, file reading, arithmetic. 3 levels:
- Level 1: 1-5 steps, single tool
- Level 2: 5-10 steps, multiple tools
- Level 3: 10+ steps, complex reasoning chains

Human experts achieve near 100%. GPT-4 with plugins achieved ~30% on Level 1, ~15% on Level 2, ~8% on Level 3 at initial release. Gap to human performance remains large.

### WebArena

812 tasks on realistic websites. Evaluates whether an agent correctly completes a web task (search, buy, post, compare), not just whether it generates a plausible response. Binary success metric.

State-of-the-art models (2024): ~35-45% success rate. Human baseline: ~78%.

### Multi-Agent Task Completion Rate

For custom multi-agent systems, define task-specific evaluation:

```python
def evaluate_agent_system(task_suite: list[Task], agent_system) -> dict:
    results = []
    for task in task_suite:
        try:
            output = agent_system.run(task.input)
            success = task.evaluate(output)   # task-specific scorer
            results.append({
                "task_id": task.id,
                "success": success,
                "latency": ...,
                "total_tokens": ...,
                "tool_calls": ...,
                "cost": ...
            })
        except Exception as e:
            results.append({"task_id": task.id, "success": False, "error": str(e)})

    return {
        "success_rate": sum(r["success"] for r in results) / len(results),
        "avg_latency": ...,
        "avg_cost": ...,
        "failure_analysis": ...   # categorize failure modes
    }
```

**Metrics beyond success rate**:
- **Efficiency**: number of agent steps per successful task (fewer is better)
- **Cost per task**: total token spend normalized by task complexity
- **Robustness**: success rate under adversarial inputs, noisy tool results
- **Consistency**: variance in output across multiple runs of the same task

---

## References

- Du, Y. et al. (2023). *Improving Factuality and Reasoning in Language Models through Multiagent Debate*. ICML 2024. arXiv:2305.14325.
- Mialon, G. et al. (2023). *GAIA: a benchmark for General AI Assistants*. arXiv:2311.12983.
- Zhou, S. et al. (2023). *WebArena: A Realistic Web Environment for Building Autonomous Agents*. arXiv:2307.13854.
- Wu, Q. et al. (2023). *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation*. arXiv:2308.08155.
- Hong, S. et al. (2023). *MetaGPT: Meta Programming for a Multi-Agent Collaborative Framework*. arXiv:2308.00352.
- Li, G. et al. (2023). *CAMEL: Communicative Agents for "Mind" Exploration of Large Scale Language Model Society*. NeurIPS 2023.
- OpenAI. *Swarm*. https://github.com/openai/swarm
- LangChain. *LangGraph Documentation*. https://langchain-ai.github.io/langgraph/
- CrewAI. *CrewAI Documentation*. https://docs.crewai.com/

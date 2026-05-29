# Agents

LLMs started as single-pass text completion engines: given a prompt, produce a response. Agents represent a qualitative shift. An **agent** is a system that uses an LLM as its reasoning core but wraps it in a loop: the model can take actions, observe the results of those actions, and use observations to inform subsequent reasoning steps. The output is not just text — it is behavior extended through time.

The shift matters because many real tasks cannot be solved in one forward pass. Finding a bug requires reading code, forming a hypothesis, running tests, interpreting output, forming a new hypothesis. Booking a flight requires searching, comparing, filling forms, confirming. These tasks require **planning, external state, and iterative refinement** — capabilities that emerge from agent architectures, not from scaling the model alone.

This section covers the foundational building blocks of LLM-based agent systems: how models call external tools, the reasoning loops that drive multi-step behavior, how agents manage and persist memory, and how multiple agents can collaborate.

## Contents

| File | Description |
|------|-------------|
| [tool-use.md](./tool-use.md) | Function calling, schema design, structured tool call/result format (OpenAI and Anthropic APIs), parallel tool calls, security and error handling |
| [agentic-frameworks.md](./agentic-frameworks.md) | ReAct, Reflexion, Plan-and-Execute, MRKL, OpenAI Assistants, Toolformer — reasoning loops and agent architectures |
| [memory-systems.md](./memory-systems.md) | In-context, episodic (vector DB), semantic, and procedural memory; Generative Agents; MemGPT; memory consolidation |
| [multi-agent-systems.md](./multi-agent-systems.md) | Sequential, hierarchical, parallel, and debate architectures; AutoGen, CrewAI, LangGraph, Swarm; orchestration and failure modes |

## What Makes a System "Agentic"

The term is often overloaded. A minimal working definition:

1. **Multi-step reasoning**: the LLM reasons across more than one inference call
2. **External actions**: the system can modify state outside the context window (call APIs, write files, search the web)
3. **Feedback loops**: observations from actions re-enter the model's context, influencing subsequent decisions
4. **Goal-directedness**: behavior is organized around completing a task, not just responding to a single prompt

Single-call RAG pipelines are not agents. A system that calls a tool once in a fixed pipeline and returns is borderline. True agents exhibit **dynamic branching** — the model decides at runtime which tool to call, when to stop, and how to recover from failures.

## Navigation

- [← Back to LLM Overview](../README.md)
- [← Previous: Alignment](../alignment/README.md)
- [→ Next: Multimodal](../multimodal/README.md)

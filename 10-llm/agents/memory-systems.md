# Memory Systems

## Table of Contents

1. [Memory Types in Agents](#1-memory-types-in-agents)
2. [In-Context Memory Management](#2-in-context-memory-management)
3. [Vector-Database Memory (Episodic)](#3-vector-database-memory-episodic)
4. [Memory Consolidation](#4-memory-consolidation)
5. [Generative Agents](#5-generative-agents)
6. [MemGPT: OS-like Memory Management](#6-memgpt-os-like-memory-management)
7. [External Knowledge Bases](#7-external-knowledge-bases)
8. [Memory Privacy](#8-memory-privacy)
9. [References](#references)

---

## 1. Memory Types in Agents

Memory in agents mirrors a four-tier taxonomy borrowed from cognitive science:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AGENT MEMORY HIERARCHY                          │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Procedural Memory (in weights)                             │  │
│  │  • Skills, habits, language understanding                   │  │
│  │  • Fixed at inference; updated only by fine-tuning          │  │
│  │  • Example: "how to write Python", "what is an API"        │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Semantic Memory (structured knowledge)                     │  │
│  │  • Facts, concepts, relationships                           │  │
│  │  • External: knowledge graph, SQL DB, knowledge base        │  │
│  │  • Example: product catalog, user profile schema            │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Episodic Memory (past events)                              │  │
│  │  • Records of specific interactions, observations           │  │
│  │  • External: vector database, log store                     │  │
│  │  • Example: "User said they prefer Python over JS (May 3)"  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  In-Context Memory (working memory)                         │  │
│  │  • Current conversation + retrieved memories                │  │
│  │  • Bounded by context window size                           │  │
│  │  • Fast, but ephemeral — lost at session end                │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### In-Context Memory

Everything in the model's current context window. Includes: system prompt, conversation history, retrieved documents, tool results, agent scratchpad. This is the only memory the model can directly "see" — all other memory must be retrieved and injected here.

**Capacity**: bounded by context length (typically 8K–1M tokens depending on model). Modern long-context models (Gemini 1.5, Claude with 200K context) dramatically expand what can be held in-context.

**Speed**: zero retrieval latency — the model attends to all in-context content simultaneously.

**Durability**: session-scoped. When the conversation ends, in-context memory is gone unless explicitly saved elsewhere.

### External / Episodic Memory

A persistent store of past events, encoded as text (or embeddings) and stored outside the model. Retrieved semantically (via embedding similarity) and injected into context at the start of a new turn or session.

**Capacity**: unbounded (scales with storage)

**Retrieval latency**: ~10-100ms for vector DB queries

**Durability**: persistent across sessions

### Semantic Memory

Structured knowledge: relational databases, knowledge graphs, ontologies. More structured than episodic memory. Queried via SQL, SPARQL, or via an LLM-generated query layer.

### Procedural Memory

Skills and behaviors encoded in model weights. Not modifiable at inference time — changing procedural memory requires fine-tuning or RLHF. This is why you cannot tell a base model to "remember to always use metric units" in one conversation and have it persist across sessions without external storage.

The fundamental architectural constraint: **the model's weights are frozen during inference**. All learning during a session must be stored externally or in the context window.

---

## 2. In-Context Memory Management

### The Context Window Budget

A context window has a finite token budget. A production agent needs to allocate this budget across competing needs:

```
┌──────────────────────────────────────────────────────────────────┐
│                    Context Window Budget                        │
├──────────────────┬───────────────────────────────────────────────┤
│ System prompt    │ 500–2000 tokens (instructions, persona, tools)│
├──────────────────┼───────────────────────────────────────────────┤
│ Retrieved memory │ 1000–5000 tokens (relevant past interactions) │
├──────────────────┼───────────────────────────────────────────────┤
│ Current task     │ 500–2000 tokens (user query, task context)    │
├──────────────────┼───────────────────────────────────────────────┤
│ Tool results     │ 1000–10000 tokens (tool outputs, documents)   │
├──────────────────┼───────────────────────────────────────────────┤
│ History          │ remaining tokens                              │
└──────────────────┴───────────────────────────────────────────────┘
```

When the context fills, old content must be removed or compressed. Three main strategies:

### Conversation Summarization

Replace the oldest portion of the conversation with an LLM-generated summary:

```python
def summarize_history(messages: list[dict], keep_last_n: int = 10) -> list[dict]:
    """
    Keep the most recent N messages verbatim; summarize older messages.
    """
    if len(messages) <= keep_last_n:
        return messages

    to_summarize = messages[:-keep_last_n]
    recent = messages[-keep_last_n:]

    # Format history for summarization
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in to_summarize
        if isinstance(m.get('content'), str)
    )

    summary_response = client.chat.completions.create(
        model="gpt-4o-mini",  # cheaper model for summarization
        messages=[{
            "role": "user",
            "content": (
                "Summarize the following conversation history concisely, "
                "preserving key facts, decisions, and user preferences:\n\n"
                f"{history_text}"
            )
        }]
    )
    summary = summary_response.choices[0].message.content

    return [
        {"role": "system", "content": f"[Previous conversation summary]: {summary}"},
        *recent
    ]
```

**Trade-off**: summarization loses detail. If the user said something important 20 turns ago, it may be summarized away. Mitigate by tagging important messages and protecting them from summarization.

### Selective Retention

Not all messages are equally important. A classification step decides which messages to keep verbatim:

```
For each message, classify:
  - CRITICAL: preserve verbatim (user preferences, key decisions, error messages)
  - CONTEXT: keep in summary form
  - EPHEMERAL: can discard (filler messages, acknowledgments)
```

This can be done with a lightweight classifier or by scoring messages based on heuristics (contains numbers/dates → likely important; contains "thanks" → likely ephemeral).

### Working Memory as Separate State

Instead of managing the conversation history directly, maintain a **structured working memory** object that is always included in context:

```python
class WorkingMemory:
    user_preferences: dict        # "prefers Python", "metric units"
    task_progress: list[str]      # completed steps
    key_facts: dict               # important findings from tools
    open_questions: list[str]     # things still to resolve

    def to_context_string(self) -> str:
        return f"""
[WORKING MEMORY]
User preferences: {json.dumps(self.user_preferences)}
Task progress: {self.task_progress}
Key facts: {json.dumps(self.key_facts)}
Open questions: {self.open_questions}
[END WORKING MEMORY]
"""
```

The working memory is updated by the agent at each step. The conversation history can then be more aggressively pruned because key state is captured in structured form.

---

## 3. Vector-Database Memory (Episodic)

### Architecture

```
WRITE PATH:
Interaction/Observation
    │
    ▼
Text Chunk ──► Embedding Model ──► vector (1536-dim) ──► Vector DB
                                                           + metadata
                                                           (timestamp, source,
                                                            importance score,
                                                            raw text)

READ PATH:
Query
    │
    ▼
Embedding Model ──► query vector
                        │
                        ▼
                   Vector DB
                (ANN similarity search)
                        │
                        ▼
              Top-k (text, score, metadata)
                        │
                        ▼
             Injected into context
```

**Common vector databases**: Pinecone, Weaviate, Qdrant, Chroma (local), pgvector (PostgreSQL extension), FAISS (library, not a service).

### Storing Memories

Not every observation should be stored. A memory formation policy decides:

```python
class MemoryStore:
    def __init__(self, embedding_model, vector_db, importance_threshold=0.6):
        self.embed = embedding_model
        self.db = vector_db
        self.importance_threshold = importance_threshold

    def maybe_store(self, text: str, metadata: dict):
        importance = self._score_importance(text)
        if importance < self.importance_threshold:
            return  # not worth storing

        vector = self.embed(text)
        self.db.upsert(
            id=generate_id(text, metadata['timestamp']),
            vector=vector,
            metadata={
                "text": text,
                "timestamp": metadata['timestamp'],
                "source": metadata.get('source', 'conversation'),
                "importance": importance
            }
        )

    def _score_importance(self, text: str) -> float:
        # Heuristic: score based on presence of named entities, numbers,
        # explicit user preferences, or stated facts
        # Can also use a small LLM for importance scoring
        ...
```

**When to create a new memory vs update existing**: use similarity search to check if a very similar memory already exists (cosine similarity > 0.95 → likely a duplicate or update, not a new memory). Update the existing entry's text and timestamp rather than creating a near-duplicate.

### Retrieval

Retrieval is not just nearest-neighbor by semantic similarity. Three signals should be combined:

**1. Relevance** — cosine similarity to the current query/context:

$$\text{sim}(q, m) = \frac{q \cdot m}{\|q\| \cdot \|m\|}$$

**2. Recency** — exponential decay over time:

$$\text{recency}(m, t) = \exp\!\left(-\lambda \cdot (t - t_m)\right)$$

where $t$ is current time, $t_m$ is the memory's timestamp, $\lambda$ controls the decay rate.

**3. Importance** — a stored score reflecting how significant the memory was judged to be at write time.

Combined retrieval score:

$$\text{score}(m) = \alpha \cdot \text{relevance}(m) + \beta \cdot \text{recency}(m) + \gamma \cdot \text{importance}(m)$$

The weights $\alpha, \beta, \gamma$ are hyperparameters tuned to the application. For a personal assistant, recency matters a lot. For a knowledge base, importance and relevance dominate.

---

## 4. Memory Consolidation

Human memory consolidation happens during sleep: experiences move from short-term hippocampal storage to long-term cortical storage, with redundancy removed and patterns abstracted.

For agents, analogous consolidation means periodically processing episodic memories to extract higher-level knowledge and store it in a more structured form:

```
Episodic memories (raw observations):
  "User asked for Python code on 2024-05-01"
  "User asked for Python code on 2024-05-03"
  "User said they don't use JavaScript on 2024-05-05"
  "User corrected my use of JS and asked for Python on 2024-05-07"
                    │
                    ▼ consolidation (LLM synthesis)
                    │
Semantic fact:
  "User strongly prefers Python. Avoid JavaScript unless explicitly requested."
```

**Consolidation trigger**: run nightly/hourly as a background job, or when episodic memory exceeds a size threshold.

**Consolidation process**:

1. Retrieve recent episodic memories (e.g., last 24 hours or last N interactions)
2. Prompt an LLM to synthesize patterns: "What stable facts, preferences, or behaviors can you infer from these observations?"
3. Store high-confidence inferences in the semantic memory layer (structured key-value store or knowledge graph)
4. Optionally: remove or archive episodic memories that have been consolidated

This is essentially **unsupervised preference learning** — the agent builds a model of the user's preferences from observed behavior.

---

## 5. Generative Agents

**Paper**: Park et al., "Generative Agents: Interactive Simulacra of Human Behavior" (2023)

Generative Agents is a landmark paper demonstrating that LLMs with rich memory systems can simulate believable human social behavior in a virtual town (inspired by The Sims). 25 agents interact, form relationships, plan days, and exhibit emergent social behaviors (gossip spreading, spontaneous party organization).

The memory architecture has three interacting components:

### Memory Stream

A timestamped log of observations and actions — the agent's diary:

```
[2024-01-15 08:00] Klaus Mueller woke up and started his morning routine
[2024-01-15 08:15] Klaus observed: Maria is in the kitchen making coffee
[2024-01-15 08:20] Klaus said to Maria: "Good morning, how did you sleep?"
[2024-01-15 09:00] Klaus started writing his research paper on climate change
[2024-01-15 11:30] Klaus felt frustrated because he couldn't find a key reference
```

Each entry has: timestamp, description text, and an **importance score** generated by the LLM at write time via the prompt: "On a scale of 1-10, how important is this observation for Klaus's long-term goals?"

### Retrieval

At each decision point, the agent retrieves the most relevant memories from the stream. The retrieval score combines three components:

$$\text{score}(m_i) = \alpha \cdot s_{\text{recency}}(m_i) + \beta \cdot s_{\text{importance}}(m_i) + \gamma \cdot s_{\text{relevance}}(m_i, q)$$

- **Recency**: $s_{\text{recency}}(m_i) = \exp(-\delta \cdot (t_{\text{now}} - t_i))$ — exponential decay over hours
- **Importance**: the 1-10 score generated at write time, normalized
- **Relevance**: cosine similarity between memory embedding and query embedding

The paper uses $\alpha = \beta = \gamma = 1$ and normalizes each component to [0,1] before combining.

### Reflection

Periodically (triggered when the sum of recent importance scores exceeds a threshold), the agent generates higher-level insights about its own experience:

```
Reflection trigger: sum of recent importance scores > 150

Prompt:
"Given the following recent observations about Klaus Mueller, what 3 high-level
insights can you infer about his situation, goals, or relationships?

[recent observations...]

Insights:"

Output:
1. "Klaus is making progress on his research paper but feels frustrated about gaps in his knowledge."
2. "Klaus has a warm relationship with Maria and enjoys their morning conversations."
3. "Klaus values productivity and is stressed when he cannot focus on his work."
```

These reflections are stored back into the memory stream with high importance scores, where they influence future behavior. Reflections about reflections are possible — a multi-level abstraction hierarchy emerges.

### Planning

Agents create plans for the day using their memories and current context:

```
Morning plan (rough): Wake up → Morning routine → Work on research paper →
                       Lunch → Read for 2 hours → Dinner → Social activity

Decomposed (detailed): 8:00 AM: Shower and breakfast (30 min)
                        8:30 AM: Review notes from yesterday (45 min)
                        ...
```

Plans are re-evaluated dynamically when unexpected events occur (another agent asks to meet, a task takes longer than expected).

---

## 6. MemGPT: OS-like Memory Management

**Paper**: Packer et al., "MemGPT: Towards LLMs as Operating Systems" (2023)

MemGPT draws an analogy between OS memory management and LLM context management. The OS has a limited amount of fast RAM and a much larger slow disk. It uses **virtual memory** to give processes the illusion of unlimited memory by paging between RAM and disk.

MemGPT maps this to LLMs:

```
OS concept          MemGPT concept
─────────────────   ──────────────────────────────────────
Physical RAM    →   Context window (fast, limited)
Virtual memory  →   External storage (slow, unlimited)
Page table      →   Memory index (what's stored where)
Page fault      →   Context miss (needed info not in context)
Page swap       →   Memory function call (load from / save to storage)
OS kernel       →   LLM (makes memory management decisions)
```

### Architecture

The LLM has access to three memory regions:

1. **Main context** (in-window): the active conversation + immediately relevant content. Always available.

2. **Archival storage** (external): a vector database of past interactions. Accessed via `archival_memory_search` and `archival_memory_insert` function calls.

3. **Recall storage** (external): a chronological log of all past messages. Accessed via `conversation_search`.

```
┌───────────────────────────────────────────────────────────────────┐
│                    LLM (Memory Controller)                       │
│                                                                   │
│  Main context:                                                    │
│  [System prompt | Core memory (user facts) | Conversation slice] │
│                                                                   │
│  Available tools:                                                 │
│  - core_memory_append(section, content)                          │
│  - core_memory_replace(section, old, new)                        │
│  - archival_memory_insert(content)                               │
│  - archival_memory_search(query) → top-k results                 │
│  - conversation_search(query, page) → paginated history          │
└───────────────────────────────────────────────────────────────────┘
```

**Core memory** is a special section of the system prompt that the model can modify: a persistent, brief summary of the user and the agent's persona. When the model learns something important about the user, it calls `core_memory_append` or `core_memory_replace` to update this persistent section. Core memory persists across sessions by being included in every new conversation's system prompt.

**Key insight**: the LLM itself decides when to load or save information — it is the memory controller. Unlike static RAG, the model is not passively given retrieved context; it actively manages its own memory through function calls.

**Example: updating core memory**

```
User: "By the way, I just moved from New York to Berlin."

Model (internal reasoning): This is important long-term information about the user's location.
                            I should update my core memory.

Model (function call):
core_memory_replace(
    section="human",
    old="The human lives in New York.",
    new="The human recently moved from New York to Berlin (as of 2024-05)."
)

Model (response): "That's a big move! How are you finding Berlin so far?"
```

---

## 7. External Knowledge Bases

### Knowledge Graphs

Semantic memory encoded as (subject, predicate, object) triples:

```
(Albert Einstein, born_in, Ulm)
(Ulm, located_in, Germany)
(Germany, capital, Berlin)
```

**Querying**: via SPARQL (for RDF stores like Wikidata, DBpedia) or via Cypher (for property graph stores like Neo4j).

LLMs can generate SPARQL/Cypher queries from natural language. The workflow:

```
User: "Where was Einstein born and what is the capital of that country?"
    │
    ▼
LLM generates SPARQL:
    SELECT ?birthplace ?capital WHERE {
      <dbr:Albert_Einstein> dbo:birthPlace ?city .
      ?city dbo:country ?country .
      ?country dbp:capital ?capital .
    }
    │
    ▼
Graph store executes query
    │
    ▼
Result: birthplace=Ulm, capital=Berlin
    │
    ▼
LLM formats natural language response
```

The model needs the graph's schema (ontology) to generate valid queries. For Wikidata, this means understanding Wikidata property IDs (P19 = place of birth, P36 = capital).

### Relational Databases (SQL)

For structured tabular data:

```
User: "Which of our customers bought more than 5 products last month?"
    │
    ▼
LLM given schema:
    CREATE TABLE orders (id, customer_id, product_id, date, amount);
    CREATE TABLE customers (id, name, email);
    │
    ▼
LLM generates SQL:
    SELECT c.name, COUNT(o.id) as orders
    FROM customers c
    JOIN orders o ON c.id = o.customer_id
    WHERE o.date >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH)
    GROUP BY c.id
    HAVING orders > 5;
    │
    ▼
Database executes query, returns rows
    │
    ▼
LLM formats response
```

**Text-to-SQL** is a well-studied problem. Models like GPT-4 achieve >80% accuracy on Spider benchmark. Provide table schema with column descriptions, sample rows for context, and few-shot examples of query patterns specific to your database.

**Safety**: never run LLM-generated SQL with write permissions on production data. Use read-only credentials. Consider a query allowlist or whitelist of safe SQL operations.

---

## 8. Memory Privacy

As agents accumulate memory about users, significant privacy risks emerge.

### Data Categories in Agent Memory

| Category | Sensitivity | Example |
|----------|------------|---------|
| Preferences | Low | "Prefers Python over Java" |
| Behavioral patterns | Medium | "Usually asks questions in the evening" |
| Personal facts | Medium-High | "Has diabetes", "Lives in Berlin" |
| Financial information | High | "Mentioned a salary of $X" |
| Health information | Very High | "Disclosed depression diagnosis" |
| Relationship information | High | "Having issues with partner" |

Agents often capture high-sensitivity information casually — users say things in conversational context not expecting them to be stored indefinitely.

### Retention Policies

**Time-bounded retention**: memories older than N days are automatically deleted or anonymized. The agent forgets by design.

**Category-scoped retention**: sensitive categories (health, financial) are stored with shorter TTL or not stored at all.

**User-controlled deletion**: provide tools for users to inspect, correct, and delete agent memories:

```python
# Tools exposed to users for memory management
{"name": "list_my_memories", "description": "Show what I remember about you"},
{"name": "delete_memory", "parameters": {"memory_id": {"type": "string"}}},
{"name": "update_memory", "parameters": {"memory_id": ..., "new_content": ...}},
{"name": "clear_all_memories", "description": "Delete all stored memories"}
```

### Regulatory Considerations

- **GDPR (EU)**: right to access, right to erasure ("right to be forgotten"), data minimization principle — collect only what is necessary
- **CCPA (California)**: right to know what data is collected, right to deletion
- **HIPAA (US)**: health information requires special handling, including encryption at rest and in transit

**Practical implication**: any agent storing episodic memory about EU users must implement deletion and export functionality, cannot retain data indefinitely without legal basis, and must declare the retention policy at collection time.

### Minimization by Design

The strongest privacy protection: don't store sensitive information in the first place. Memory formation policies should exclude sensitive categories:

```python
NEVER_STORE_PATTERNS = [
    r'\b\d{3}-\d{2}-\d{4}\b',          # SSN
    r'\b\d{16}\b',                       # credit card numbers
    r'(cancer|HIV|depression|anxiety)',  # health conditions
    r'\$\d+[k]?\s*(salary|income)',      # financial info
]

def should_store(text: str) -> bool:
    return not any(re.search(p, text, re.IGNORECASE) for p in NEVER_STORE_PATTERNS)
```

Pattern matching alone is insufficient (users may express sensitive information indirectly), but combined with human oversight and periodic audits it reduces the risk significantly.

---

## References

- Park, J.S. et al. (2023). *Generative Agents: Interactive Simulacra of Human Behavior*. UIST 2023. arXiv:2304.03442.
- Packer, C. et al. (2023). *MemGPT: Towards LLMs as Operating Systems*. arXiv:2310.08560.
- Zhu, X. et al. (2023). *Ghost in the Minecraft: Generally Capable Agents for Open-World Environments via Large Language Models with Text-based Knowledge and Memory*. arXiv:2305.17144.
- Weaviate. *Vector Database Concepts*. https://weaviate.io/developers/weaviate/concepts
- Johnson, J. et al. (2019). *Billion-scale similarity search with GPUs*. IEEE TPAMI. (FAISS)
- Shen, Y. et al. (2023). *HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace*. NeurIPS 2023.

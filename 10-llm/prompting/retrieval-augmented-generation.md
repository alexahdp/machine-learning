# Retrieval-Augmented Generation (RAG)

## Table of Contents

1. [Motivation](#motivation)
2. [Basic RAG Pipeline](#basic-rag-pipeline)
3. [Chunking Strategies](#chunking-strategies)
4. [Embedding Models](#embedding-models)
5. [Vector Databases](#vector-databases)
6. [Sparse Retrieval: BM25 and TF-IDF](#sparse-retrieval)
7. [Hybrid Search](#hybrid-search)
8. [Re-ranking](#re-ranking)
9. [Advanced RAG Patterns](#advanced-rag-patterns)
10. [Evaluation](#evaluation)
11. [GraphRAG](#graphrag)
12. [References](#references)

---

## Motivation

LLMs have three fundamental limitations for factual question-answering:

1. **Stale knowledge**: Pre-training has a cutoff date. A model trained through mid-2024 has no knowledge of events after that. For rapidly evolving domains (stock prices, news, software documentation), this is immediately problematic.

2. **Hallucination**: LLMs generate fluent, confident text by predicting plausible continuations. When the correct answer is absent or uncertain in parametric memory, the model generates plausible-sounding but incorrect text. Hallucination rates on specific factual queries can exceed 20% even for frontier models.

3. **Private data**: An organization's internal documents, databases, and proprietary knowledge never appeared in pre-training data. The model genuinely has no knowledge of them.

**RAG** addresses all three by providing the model with relevant documents at inference time. Instead of relying on memory baked into weights, the model grounds its answer in retrieved context:

```
Without RAG: "Who is the current CEO of Anthropic?"
Model: "Dario Amodei" ← correct only because it's in training data

With RAG: retrieved doc = Anthropic's about page (2025)
Model: "According to Anthropic's website, Dario Amodei is the CEO." ← grounded
```

**The key trade-off**: RAG improves factual grounding but adds retrieval latency and introduces retrieval errors (the wrong documents can mislead the model more than no context).

---

## Basic RAG Pipeline

RAG consists of three phases:

```
                    INDEXING (offline)
                    ─────────────────
Documents → Chunker → Embedding Model → Vector DB
                         ↑ embeds each chunk

                    RETRIEVAL (online)
                    ─────────────────
User Query → Embedding Model → Query Vector
                                    ↓
                              Vector DB search → Top-k Chunks

                    GENERATION (online)
                    ─────────────────
[System prompt] + [Retrieved chunks] + [User query] → LLM → Answer
```

### Indexing

```python
import openai
import faiss
import numpy as np

def embed(texts: list[str], model="text-embedding-3-small") -> np.ndarray:
    response = openai.embeddings.create(input=texts, model=model)
    return np.array([r.embedding for r in response.data], dtype="float32")

# Chunk documents
chunks = chunk_documents(documents)  # See chunking section

# Embed all chunks
chunk_embeddings = embed(chunks)  # shape: (N, D)

# Build FAISS index
D = chunk_embeddings.shape[1]  # embedding dimension, e.g., 1536
index = faiss.IndexFlatIP(D)   # Inner product (equivalent to cosine after normalization)
faiss.normalize_L2(chunk_embeddings)
index.add(chunk_embeddings)
```

### Retrieval and Generation

```python
def rag_query(
    user_query: str,
    index: faiss.Index,
    chunks: list[str],
    k: int = 5,
    llm_model: str = "gpt-4o"
) -> str:
    # Embed query
    query_embedding = embed([user_query])
    faiss.normalize_L2(query_embedding)

    # Retrieve top-k chunks
    distances, indices = index.search(query_embedding, k)
    retrieved_chunks = [chunks[i] for i in indices[0]]

    # Build context
    context = "\n\n---\n\n".join(retrieved_chunks)

    # Generate grounded answer
    response = openai.chat.completions.create(
        model=llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the user's question "
                    "based ONLY on the provided context. If the answer is not "
                    "in the context, say 'I don't have information about that.' "
                    "Never make up information."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {user_query}"
            }
        ]
    )
    return response.choices[0].message.content
```

---

## Chunking Strategies

Chunking is the most underappreciated component of RAG. The quality of retrieval depends entirely on chunks being appropriately sized and semantically coherent.

### Fixed-Size Chunking

Split text into chunks of N tokens (or characters) with an overlap of M tokens.

```
Text: "Transformers use self-attention. Self-attention computes..."
      ← chunk 1 (512 tokens) → ← overlap → ← chunk 2 (512 tokens) →
```

**Overlap** (typically 10–20% of chunk size) ensures that sentences near chunk boundaries appear in at least one chunk with full surrounding context.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,          # tokens approximately
    chunk_overlap=64,        # overlap
    separators=["\n\n", "\n", ".", " ", ""]  # try to split at paragraph → sentence → word
)
chunks = splitter.split_text(document_text)
```

### Sentence and Paragraph Chunking

Split at natural linguistic boundaries. Sentence splitting uses a library like spaCy or NLTK:

```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
sentences = [sent.text for sent in doc.sents]
# Group into chunks of N sentences with overlap
```

Paragraph chunking uses double newlines or markdown headers as split points. This is the default approach for structured documents (technical docs, wikis).

### Semantic Chunking

Split where the semantic content changes, detected by a drop in cosine similarity between consecutive sentence embeddings.

```python
def semantic_chunk(sentences: list[str], threshold: float = 0.3) -> list[list[str]]:
    embeddings = embed(sentences)
    chunks = []
    current_chunk = [sentences[0]]
    for i in range(1, len(sentences)):
        sim = cosine_similarity(embeddings[i-1], embeddings[i])
        if sim < threshold:
            chunks.append(current_chunk)
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])
    chunks.append(current_chunk)
    return chunks
```

### Chunk Size Trade-offs

| Chunk Size | Precision | Context | Use Case |
|------------|-----------|---------|----------|
| Small (128–256 tokens) | High | Low | QA over dense technical docs, lookup tasks |
| Medium (512–1024 tokens) | Balanced | Balanced | General RAG, most production use |
| Large (2K–4K tokens) | Low | High | Summarization, synthesis tasks |

**The retrieval precision vs. context trade-off**: Small chunks match queries more precisely (higher retrieval recall for specific facts) but may lack the surrounding context needed for a coherent answer. Large chunks provide full context but may match queries on irrelevant content.

---

## Embedding Models

The embedding model determines the semantic representation of chunks and queries. The embedding space defines what "similar" means for retrieval.

| Model | Dimension | Context | Strengths |
|-------|-----------|---------|-----------|
| OpenAI text-embedding-3-small | 1536 | 8K tokens | Low cost, good baseline |
| OpenAI text-embedding-3-large | 3072 | 8K tokens | Best OpenAI quality |
| BGE-M3 (BAAI) | 1024 | 8K tokens | Open source, multilingual, dense+sparse |
| E5-large-v2 (Microsoft) | 1024 | 512 tokens | Strong on MTEB, open source |
| Instructor-XL | 768 | 512 tokens | Task-conditioned embeddings |
| GTE-large | 1024 | 512 tokens | Fast, good quality/cost ratio |

**MTEB** (Massive Text Embedding Benchmark) is the standard evaluation suite. Check the leaderboard at huggingface.co/spaces/mteb/leaderboard for up-to-date rankings.

**Key consideration**: Asymmetric vs symmetric embedding. For QA, queries are short and documents are long. Some models (Instructor-XL, E5) use different prompts for query vs document embedding:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# BGE requires "Represent this sentence:" prefix for queries
query_embedding = model.encode("Represent this sentence: " + query)
doc_embedding = model.encode(document_chunk)  # no prefix for documents
```

---

## Vector Databases

A vector database stores embeddings and supports approximate nearest neighbor (ANN) search — retrieving the $k$ most similar vectors by cosine or dot-product similarity without comparing against every stored vector.

### FAISS (Facebook AI Similarity Search)

FAISS is a library (not a service) for in-process vector search. Key index types:

| Index | Description | Trade-off |
|-------|-------------|-----------|
| `IndexFlatIP` | Exact inner product search | Exact but O(N) per query |
| `IndexIVFFlat` | Inverted file index: cluster then search | ~100× speedup, small recall loss |
| `IndexHNSWFlat` | Hierarchical Navigable Small World graph | Best recall/speed for < 100M vectors |
| `IndexIVFPQ` | IVF + Product Quantization | Lowest memory, higher recall loss |

```python
# HNSW index — good default for <10M vectors
import faiss
d = 1536  # embedding dimension
M = 32    # number of neighbors in HNSW graph (higher = better recall, more memory)
index = faiss.IndexHNSWFlat(d, M)
index.hnsw.efConstruction = 200  # build quality (higher = better, slower build)
index.hnsw.efSearch = 50         # search quality (higher = better, slower search)
index.add(embeddings)
```

### Managed Vector Databases

| Service | Strengths | Notes |
|---------|-----------|-------|
| Pinecone | Fully managed, metadata filtering, hybrid search | Closed source, usage-based pricing |
| Weaviate | Open source, GraphQL API, built-in vectorizers | Self-hosted or cloud |
| Chroma | Lightweight, embedded, great for prototyping | Python-native, no external services |
| Qdrant | High performance, Rust-based, rich filtering | Open source, self-hosted or cloud |
| pgvector | PostgreSQL extension — store vectors in existing DB | Convenient if already on Postgres |

**pgvector** is notable for organizations already running PostgreSQL: it avoids a separate service and allows combining vector search with SQL joins.

```sql
-- pgvector: find k most similar chunks to a query
SELECT id, text, 1 - (embedding <=> $1::vector) AS similarity
FROM document_chunks
ORDER BY embedding <=> $1::vector
LIMIT 5;
```

---

## Sparse Retrieval

Dense embedding-based retrieval excels at semantic similarity but can miss exact keyword matches. **Sparse retrieval** uses term frequency statistics.

### BM25 (Best Match 25)

BM25 is a probabilistic ranking function and the dominant sparse retrieval method. The score of document $d$ for query $q$ is:

$$\text{BM25}(d, q) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})}$$

Where:
- $f(t, d)$ = term frequency of term $t$ in document $d$
- $|d|$ = document length; $\text{avgdl}$ = average document length
- $k_1 \in [1.2, 2.0]$ controls term frequency saturation
- $b \in [0, 1]$ controls length normalization (0.75 typical)
- $\text{IDF}(t) = \log \frac{N - n(t) + 0.5}{n(t) + 0.5}$, where $N$ = total docs, $n(t)$ = docs containing $t$

```python
from rank_bm25 import BM25Okapi

tokenized_corpus = [doc.lower().split() for doc in chunks]
bm25 = BM25Okapi(tokenized_corpus)

query_tokens = query.lower().split()
scores = bm25.get_scores(query_tokens)
top_k_indices = np.argsort(scores)[::-1][:k]
```

**When BM25 outperforms dense retrieval**:
- Exact terminology matters: product names, medical codes, legal citations.
- Short, specific queries ("GDPR Article 17 right to erasure").
- Out-of-domain text where embedding models were not trained on the vocabulary.

---

## Hybrid Search

Hybrid search combines dense (embedding) and sparse (BM25/TF-IDF) retrieval to get the benefits of both.

### Reciprocal Rank Fusion (RRF)

RRF combines ranked lists without needing score normalization:

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + r(d)}$$

Where $r(d)$ is the rank of document $d$ in ranked list $r$, and $k = 60$ is a smoothing constant.

```python
def reciprocal_rank_fusion(
    dense_results: list[tuple[int, float]],  # (doc_id, score)
    sparse_results: list[tuple[int, float]],
    k: int = 60
) -> list[int]:
    rrf_scores: dict[int, float] = {}
    
    for rank, (doc_id, _) in enumerate(dense_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
    
    for rank, (doc_id, _) in enumerate(sparse_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
    
    return sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)
```

### Linear Combination

Alternatively, normalize both score distributions to [0, 1] and combine:

$$\text{hybrid\_score}(d) = \alpha \cdot \text{dense\_score}(d) + (1 - \alpha) \cdot \text{sparse\_score}(d)$$

$\alpha = 0.5$ is a reasonable default; tune on a validation set.

---

## Re-ranking

After initial retrieval of top-$k$ candidates (typically $k = 20–50$), re-ranking applies a more expensive but more accurate model to reorder the candidates. This two-stage design is a classic precision-recall trade-off: use fast retrieval for recall (get relevant documents in top-k), then precise re-ranking for precision (push the most relevant to the top).

### Cross-Encoder Re-ranking

A cross-encoder takes (query, document) as a single input and produces a relevance score. Unlike bi-encoders (embedding models that encode query and document independently), cross-encoders can model fine-grained query-document interactions.

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Score each (query, chunk) pair
pairs = [(query, chunk) for chunk in retrieved_chunks]
scores = reranker.predict(pairs)

# Sort by relevance score
reranked = sorted(
    zip(retrieved_chunks, scores),
    key=lambda x: x[1],
    reverse=True
)
top_chunks = [chunk for chunk, score in reranked[:5]]
```

**BGE-Reranker** (BAAI) and **ColBERT** are strong open-source options. Cohere Rerank is a high-quality API-based option.

**ColBERT** uses late interaction — query and document are encoded separately, but relevance is computed via a MaxSim operation over token-level representations. It is faster than cross-encoders while more accurate than bi-encoders.

---

## Advanced RAG Patterns

### HyDE (Hypothetical Document Embeddings)

Gao et al. (2022): instead of embedding the user query directly, generate a hypothetical document that would answer the query and embed that instead.

**Rationale**: The query "What causes aurora borealis?" is in a different embedding space than "Aurora borealis is caused by charged particles from the solar wind colliding with atmospheric gases..." A hypothetical answer is in the same embedding space as actual answer documents.

```python
def hyde_retrieve(query: str, index, chunks, k=5):
    # Generate hypothetical answer
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "user",
            "content": f"Write a passage that answers: {query}"
        }]
    )
    hypothetical_doc = response.choices[0].message.content
    
    # Embed hypothetical document, not the query
    hyp_embedding = embed([hypothetical_doc])
    faiss.normalize_L2(hyp_embedding)
    _, indices = index.search(hyp_embedding, k)
    return [chunks[i] for i in indices[0]]
```

### Multi-Query Retrieval

A single user query may be phrased in a way that misses relevant documents phrased differently. Generate multiple query reformulations and take the union of retrieved chunks.

```python
def multi_query_retrieve(query: str, index, chunks, k_per_query=3):
    # Generate query variations
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": (
                f"Generate 4 different ways to ask the following question. "
                f"Output one per line, no numbering.\n\nQuestion: {query}"
            )
        }]
    )
    queries = [query] + response.choices[0].message.content.strip().split("\n")
    
    # Retrieve for each, union results
    all_chunks = set()
    for q in queries:
        q_emb = embed([q])
        faiss.normalize_L2(q_emb)
        _, indices = index.search(q_emb, k_per_query)
        all_chunks.update(indices[0])
    
    return [chunks[i] for i in all_chunks]
```

### Parent-Child Chunking

Index small chunks for precise retrieval, but return the larger parent chunk to the LLM for more context.

```
Document
├── Parent Chunk 1 (1024 tokens) ← returned to LLM
│   ├── Child Chunk 1a (128 tokens) ← indexed for retrieval
│   ├── Child Chunk 1b (128 tokens)
│   └── Child Chunk 1c (128 tokens)
└── Parent Chunk 2 (1024 tokens)
    ├── Child Chunk 2a (128 tokens)
    └── ...
```

When a child chunk is retrieved, look up its parent ID and return the full parent. This combines retrieval precision with generation-time context richness.

### Self-RAG

Asai et al. (2023) — Self-RAG trains a model to:
1. **Decide when to retrieve**: generate a `[Retrieve]` token if retrieval is needed.
2. **Evaluate retrieved content**: generate `[Relevant]`/`[Irrelevant]` tokens per retrieved chunk.
3. **Reflect on output**: generate `[Supported]`/`[Unsupported]` tokens assessing grounding.

Self-RAG outperforms standard RAG because it avoids retrieving when the model already knows the answer and skips irrelevant retrieved chunks. It requires fine-tuning a base model with reflection tokens.

---

## Evaluation

RAG systems have two evaluation targets: retrieval quality and generation quality.

### Retrieval Metrics

**Recall@k**: Fraction of queries for which at least one relevant document appears in the top-k results. The primary metric for retrieval.

$$\text{Recall@k} = \frac{|\text{relevant} \cap \text{retrieved}_k|}{|\text{relevant}|}$$

**MRR (Mean Reciprocal Rank)**: Measures how highly the first relevant document is ranked.

$$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_q}$$

**NDCG@k** (Normalized Discounted Cumulative Gain): Accounts for graded relevance and position.

### Generation Metrics

| Metric | Description | How to Measure |
|--------|-------------|----------------|
| **Faithfulness** | Does the answer only make claims supported by retrieved context? | LLM judge: "Is claim X supported by context?" |
| **Answer Relevance** | Does the answer actually address the question? | LLM judge or embedding cosine similarity |
| **Context Precision** | Are retrieved chunks relevant? | Fraction of retrieved chunks that support the answer |
| **Context Recall** | Are all facts needed for the answer present in retrieved context? | Compare answer to ground truth |

### RAGAS Framework

RAGAS (Es et al., 2023) is a reference-free evaluation framework that uses LLMs as judges for all four metrics above, enabling evaluation without human-annotated ground truth.

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

results = evaluate(
    dataset=dataset,          # HuggingFace dataset with question/context/answer
    metrics=[faithfulness, answer_relevancy, context_recall]
)
print(results)
# {'faithfulness': 0.82, 'answer_relevancy': 0.91, 'context_recall': 0.74}
```

---

## GraphRAG

Standard RAG retrieves isolated text chunks. For queries requiring multi-hop reasoning across entities (e.g., "What companies were founded by people who worked at Bell Labs?"), chunk-level retrieval is insufficient.

**GraphRAG** (Edge et al., 2024 — Microsoft Research) builds a **knowledge graph** from the document corpus:
1. Use an LLM to extract entities and relations from all documents.
2. Build a graph: nodes = entities, edges = relations with supporting text.
3. Run community detection (Leiden algorithm) to find clusters of related entities.
4. Summarize each community with an LLM.

At query time:
- **Local search**: find relevant entities via graph traversal from matched nodes.
- **Global search**: match query against community summaries (for corpus-level questions like "What are the main themes in these documents?").

GraphRAG excels at queries requiring aggregation ("How many companies in the corpus are in healthcare?") and multi-hop reasoning. It has significantly higher indexing cost than vector RAG.

```
Vector RAG:    "What does document 47 say about CRISPR?"  ✓
GraphRAG:      "Who are the key researchers connected to CRISPR across all documents?" ✓✓
GraphRAG:      "What is the overall sentiment about AI regulation in this corpus?" ✓✓
```

---

## References

- Lewis, P. et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS. [Original RAG paper]
- Gao, L. et al. (2022). "Precise Zero-Shot Dense Retrieval without Relevance Labels." ACL 2023. [HyDE]
- Asai, A. et al. (2023). "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." ICLR 2024. [Self-RAG]
- Es, S. et al. (2023). "RAGAS: Automated Evaluation of Retrieval Augmented Generation." EACL 2024. [RAGAS]
- Edge, D. et al. (2024). "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." arXiv:2404.16130. [GraphRAG]
- Robertson, S., Zaragoza, H. (2009). "The Probabilistic Relevance Framework: BM25 and Beyond." Foundations and Trends in IR. [BM25]
- Cormack, G.V. et al. (2009). "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods." SIGIR. [RRF]
- Khattab, O., Zaharia, M. (2020). "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT." SIGIR. [ColBERT]
- Johnson, J. et al. (2019). "Billion-scale similarity search with GPUs." IEEE Transactions on Big Data. [FAISS]

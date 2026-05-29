# Embeddings

## Table of Contents

1. [Token Embedding Matrix](#token-embedding-matrix)
2. [Static Word Embeddings](#static-word-embeddings)
3. [Contextual Embeddings](#contextual-embeddings)
4. [Embedding Geometry](#embedding-geometry)
5. [Embedding Arithmetic](#embedding-arithmetic)
6. [Sentence and Document Embeddings](#sentence-and-document-embeddings)
7. [Embedding Dimensionality](#embedding-dimensionality)
8. [Initialization and Learning During Training](#initialization-and-learning-during-training)
9. [Cross-Lingual Embeddings](#cross-lingual-embeddings)
10. [Retrieval and Similarity](#retrieval-and-similarity)
11. [References](#references)

---

## Token Embedding Matrix

Every LLM begins with an **embedding lookup**: each input token index $t \in \{0, 1, \ldots, V-1\}$ is mapped to a dense vector by indexing into a matrix $\mathbf{E} \in \mathbb{R}^{V \times d_{\text{model}}}$:

$$\mathbf{x}_t = \mathbf{E}[t] \in \mathbb{R}^{d_{\text{model}}}$$

This is mathematically equivalent to multiplying a one-hot vector by $\mathbf{E}$, but embedding lookup avoids materializing the one-hot vector.

### Weight Tying

In most modern LLMs, the output projection that converts the final hidden states back to vocabulary logits is the **transpose of the same embedding matrix**:

$$\text{logits} = \mathbf{H}_L \mathbf{E}^\top \in \mathbb{R}^{n \times V}$$

This **weight tying** (Press & Wolf, 2017) shares the $V \times d$ parameters between input and output, which:
- Reduces parameter count by $V \times d$ (e.g., 50,257 × 768 ≈ 38.6M for GPT-2).
- Enforces consistency: the token vector that the model uses to recognize token $t$ as input is the same vector it learns to produce when predicting token $t$ as output.
- Empirically improves perplexity, especially for smaller models.

### Parameter Count

The embedding matrix is often the third or fourth largest component of a model:

| Model | $V$ | $d$ | Embedding params |
|-------|-----|-----|-----------------|
| BERT-Base | 30,522 | 768 | 23.4M (≈21% of total) |
| GPT-2 | 50,257 | 768 | 38.6M (≈33% of total) |
| LLaMA-7B | 32,000 | 4096 | 131M (≈1.9% of total) |
| LLaMA-3-70B | 128,256 | 8192 | 1.05B (≈1.5% of total) |

The fraction decreases as model size grows because other components (attention, FFN) grow as $d^2$ while the embedding grows as $V \times d$.

---

## Static Word Embeddings

Before contextual LLMs, the dominant approach was to train a single, context-free vector for each word — the same vector regardless of the word's usage.

### Word2Vec (Mikolov et al., 2013)

Word2Vec trains embeddings using a shallow neural network on a large corpus, using one of two objectives:

**Skip-gram**: Given a center word, predict surrounding context words. Maximize:

$$\mathcal{L} = \sum_{(w, c) \in \mathcal{D}} \log P(c \mid w) = \sum_{(w, c)} \log \frac{\exp(\mathbf{v}_c \cdot \mathbf{u}_w)}{\sum_{c'} \exp(\mathbf{v}_{c'} \cdot \mathbf{u}_w)}$$

where $\mathbf{u}_w$ is the center word vector and $\mathbf{v}_c$ is the context word vector. Words with similar contexts end up with similar $\mathbf{u}_w$ vectors.

**CBOW** (Continuous Bag of Words): Given surrounding context words, predict the center word. Averages context vectors and predicts the center. Faster to train; slightly worse on rare words.

**Negative sampling**: The full softmax over $V$ words is expensive. In practice, the objective is replaced with binary classification: distinguish the true context word from $k$ randomly sampled "noise" words:

$$\mathcal{L}_{\text{NS}} = \log \sigma(\mathbf{v}_c \cdot \mathbf{u}_w) + \sum_{j=1}^{k} \mathbb{E}_{n_j \sim P_n} [\log \sigma(-\mathbf{v}_{n_j} \cdot \mathbf{u}_w)]$$

where $P_n$ is a noise distribution (empirically, $P_n(w) \propto f(w)^{3/4}$ where $f(w)$ is word frequency).

### GloVe (Pennington et al., 2014)

GloVe (Global Vectors) trains embeddings to predict **word co-occurrence statistics** from a global co-occurrence matrix $\mathbf{X}$ where $X_{ij}$ counts how often word $j$ appears in the context of word $i$ in the corpus.

The objective:

$$\mathcal{L} = \sum_{i,j=1}^{V} f(X_{ij})\left(\mathbf{u}_i \cdot \mathbf{v}_j + b_i + b_j - \log X_{ij}\right)^2$$

where $f(X_{ij})$ is a weighting function that down-weights very frequent co-occurrences ($f(x) = \min(1, (x/x_{\max})^{3/4})$).

The key insight: the ratio $\log(P(k \mid i) / P(k \mid j))$ encodes the relationship between words $i$ and $j$ with respect to probe word $k$. GloVe trains embeddings so that the dot product $\mathbf{u}_i \cdot \mathbf{v}_j$ approximates $\log X_{ij}$, capturing these ratio relationships.

GloVe is faster to train than Word2Vec (operates on precomputed counts) and generally produces higher-quality embeddings on word analogy tasks.

### Limitations of Static Embeddings

- **No context sensitivity**: "bank" (river bank) and "bank" (financial institution) share one vector.
- **Morphological blindness**: "run" and "running" have unrelated vectors in word-level models.
- **Fixed vocabulary**: OOV words have no embedding.
- **Surface-form bias**: Rare words are poorly trained regardless of their semantic richness.

---

## Contextual Embeddings

Modern LLMs produce **contextual embeddings**: the representation of a token is a function of *all* tokens in its context, not just the token itself.

In BERT, the hidden state $\mathbf{h}_i^{(L)}$ at the final layer for position $i$ is a rich representation that encodes:
- The identity of token $i$ (from the initial embedding).
- The syntactic role of token $i$ in the sentence.
- The semantic contribution of token $i$ given all surrounding context.

Concretely: the word "bank" in:
- "I deposited money in the **bank**."
- "We sat on the river **bank**."

produces measurably different BERT representations (Peters et al., 2018 showed this using probing classifiers). A static embedding produces the same vector in both cases.

### Probing Classifiers

The classic tool for analyzing what contextual embeddings encode: train a simple linear classifier on frozen embeddings to predict a linguistic property (POS tag, dependency relation, entity type). If a simple linear model achieves high accuracy, the information is linearly encoded in the embedding. This is how researchers established that:

- Lower layers of BERT → surface-level and syntactic features (POS tags, chunking).
- Middle layers → syntax (parsing, NER).
- Upper layers → semantic features (coreference, semantic roles).

---

## Embedding Geometry

### Isotropy

An embedding space is **isotropic** if the embeddings are uniformly distributed across all directions — no direction is preferred. For Word2Vec embeddings, Mu & Viswanath (2018) found that the top few principal components of the embedding matrix dominate variance: most embedding vectors point in roughly similar directions, and the "effective" dimensionality of the space is much less than $d$.

**Consequences**: If all embeddings cluster around a narrow cone, cosine similarity between arbitrary word pairs is artificially high. Retrieval and classification degrade because there's limited discrimination between vectors.

**Anisotropy in contextual embeddings**: BERT and GPT produce severely anisotropic representations (Ethayarajh, 2019). Contextual embeddings cluster in a narrow cone of the representation space, more so in upper layers. This means that average cosine similarity between random sentence embeddings is > 0.95 — almost all sentences look similar in cosine space.

**Fixes**: Post-hoc whitening (BERT-whitening, Su et al., 2021) applies PCA and rescales the principal directions to have unit variance. This dramatically improves semantic similarity tasks.

### Hubness Problem

In high-dimensional spaces, certain vectors become "hubs" — they are the nearest neighbor of many other vectors. This is a consequence of the geometry of high-dimensional spaces (the curse of dimensionality). In embedding-based retrieval, hub vectors appear as false positives, retrieved as neighbors of many unrelated queries.

**Mitigation**: Cross-match scoring, CSLS (Cross-domain Similarity Local Scaling) from Conneau et al. (2018), or dimensionality reduction.

---

## Embedding Arithmetic

One of the most celebrated properties of Word2Vec-style embeddings is that **linear vector arithmetic captures semantic relationships**:

$$\mathbf{v}(\text{king}) - \mathbf{v}(\text{man}) + \mathbf{v}(\text{woman}) \approx \mathbf{v}(\text{queen})$$

### Why This Works

Word2Vec trains embeddings so that words appearing in similar contexts have similar vectors. Consider contexts where "king" and "queen" appear: they are syntactically interchangeable in many sentences, so they learn similar embeddings. The *difference* $\mathbf{v}(\text{king}) - \mathbf{v}(\text{queen})$ encodes the "gender" direction in embedding space, shared with $\mathbf{v}(\text{man}) - \mathbf{v}(\text{woman})$.

More formally, from the Skip-gram objective with negative sampling, the embedding dot product approximates pointwise mutual information (shifted by a constant $k$):

$$\mathbf{u}_w \cdot \mathbf{v}_c \approx \text{PMI}(w, c) - \log k$$

PMI captures co-occurrence patterns, and the ratio of PMIs between word pairs encodes relational structure. When that relational structure is consistent across many context words, the corresponding geometry appears in the embedding space.

### When It Doesn't Work

- **Rare words**: Words with few training occurrences have high-variance embeddings; arithmetic over them is unreliable.
- **Polysemy**: A single vector cannot capture multiple senses; arithmetic conflates them.
- **Contextual embeddings**: The analogy property largely disappears in BERT representations, because each embedding is context-specific rather than a "prototypical" representation of the word.
- **Large analogies**: The linear structure holds for systematic morphological/semantic relationships but breaks for complex relational reasoning (e.g., capital cities of more obscure countries).

---

## Sentence and Document Embeddings

LLMs produce token-level representations. Obtaining a single fixed-size vector for a sentence or document requires **pooling**:

### CLS Token Pooling

In BERT-style models, the `[CLS]` token is prepended to every input. Its final hidden state $\mathbf{h}_{[\text{CLS}]}^{(L)}$ is designed to aggregate sequence-level information for classification tasks.

**Limitation**: BERT's `[CLS]` embedding is not well-calibrated for semantic similarity tasks out of the box — it was trained with a Next Sentence Prediction objective, not with a similarity loss. Fine-tuning on sentence similarity data (SBERT) is typically required.

### Mean Pooling

Average the hidden states at the final layer over all non-padding token positions:

$$\mathbf{s} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{h}_i^{(L)}$$

Often outperforms CLS token pooling for semantic similarity without fine-tuning. The averaging dilutes position-specific information but captures the overall semantic content.

### Max Pooling

Take the element-wise maximum across all positions:

$$s_j = \max_i h_{ij}^{(L)}$$

Retains the strongest activation for each feature dimension, regardless of position. Useful when the most salient feature in the sequence should dominate.

### SBERT (Sentence-BERT)

Reimers & Gurevych (2019) fine-tune BERT with a Siamese network architecture using a contrastive loss on sentence pairs. The resulting model produces sentence embeddings that directly optimize for semantic similarity:

$$\mathcal{L} = \max(0, \epsilon - \text{cos}(\mathbf{s}_a, \mathbf{s}_b^+) + \text{cos}(\mathbf{s}_a, \mathbf{s}_b^-))$$

where $\mathbf{s}_b^+$ is a semantically similar sentence and $\mathbf{s}_b^-$ is a negative example.

### For Decoder-Only Models

GPT-style models are not trained with CLS tokens. Common approaches:
- Use the last token's hidden state (the one position that has "seen" all other tokens).
- Mean-pool all token positions.
- Fine-tune with contrastive objectives (E5, LLM2Vec).

---

## Embedding Dimensionality

Typical embedding dimensions and their tradeoffs:

| Dimension | Models | Notes |
|-----------|--------|-------|
| 128–256 | DistilBERT variants, character models | Research/distillation; limited capacity |
| 768 | BERT-Base, GPT-2, RoBERTa-Base | The canonical "base" size |
| 1024 | BERT-Large, GPT-2-Large | Standard "large" size |
| 2048 | Intermediate-scale models | |
| 4096 | LLaMA-7B, LLaMA-13B | Modern large models |
| 8192 | LLaMA-65B, LLaMA-3-70B | Very large models |

**Why higher dimensionality helps**:
- More capacity to represent fine-grained semantic distinctions.
- Reduces anisotropy (more orthogonal directions available).
- Better utilization of multi-head attention (heads operate on $d/h$ dimensions each; larger $d$ gives each head more expressive capacity).

**Why higher dimensionality hurts**:
- Quadratic cost of attention is unchanged (attention acts on $n \times d$, but FFN costs $n \times d^2$).
- Embedding matrix grows as $V \times d$.
- Training data requirements increase; larger models need more data to fill their capacity.

The empirical consensus from scaling laws (Kaplan et al., 2020): at fixed compute budget, increasing model width and depth together yields better scaling efficiency than increasing only one dimension.

---

## Initialization and Learning During Training

### Random Initialization

Token embeddings are typically initialized from a small-variance normal or uniform distribution:

$$\mathbf{E}_{ij} \sim \mathcal{N}(0, \sigma^2)$$

where $\sigma$ is often $d^{-1/2}$ or a small fixed value (e.g., 0.02). This ensures that at initialization, all token embeddings are approximately equally similar (all near zero), preventing the model from biasing toward specific tokens before seeing any data.

### Learning Rate Considerations

Embedding parameters are often trained with a slightly higher effective learning rate than other parameters, because they receive gradients only for the tokens that appear in each batch (sparse updates). Adaptive optimizers (Adam) handle this well by maintaining per-parameter learning rate estimates, but SGD would under-train rare tokens.

### Embedding Drift

During training, the embedding matrix shifts significantly from initialization. Studies (Geva et al., 2022) show that the model increasingly "de-embeds" early token information, with later layers representing higher-level abstractions that differ substantially from the raw embedding space. The final hidden state at layer $L$ retains little direct correspondence to the input embedding.

---

## Cross-Lingual Embeddings

### Multilingual Models

Models like mBERT (Devlin et al., 2019) and XLM-R (Conneau et al., 2020) train on text from 100+ languages with a shared vocabulary and shared Transformer weights. Despite never seeing cross-lingual parallel data, these models develop a **language-neutral representation space**: the English word "dog" and the French word "chien" have similar hidden representations.

This emerges because:
1. The shared vocabulary contains byte-level overlap between languages (numbers, URLs, code, proper nouns).
2. The Transformer must use the same weights to process all languages — it is induced to find representations that work across languages.
3. Related languages (French/Spanish, German/Dutch) share subword tokens, directly aligning their embedding spaces.

### Cross-Lingual Alignment

The degree of alignment can be measured by how well a classifier trained in English transfers to another language (zero-shot cross-lingual transfer). XLM-R achieves strong results on NER and QA tasks in languages it was never explicitly trained on for those tasks.

**Alignment is imperfect**: High-resource languages (English, Chinese) are better aligned than low-resource ones. Languages with non-Latin scripts and small corpora often occupy separate regions of the embedding space.

---

## Retrieval and Similarity

### Cosine Similarity

For normalized vectors ($\|\mathbf{u}\| = \|\mathbf{v}\| = 1$):

$$\text{cos}(\mathbf{u}, \mathbf{v}) = \mathbf{u} \cdot \mathbf{v} = \frac{\mathbf{u}^\top \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$

Cosine similarity is invariant to vector magnitude, measuring only directional alignment. This is appropriate when the "meaning" of an embedding is in its direction (as in word vectors) rather than its magnitude.

**When to use instead**: If embeddings have meaningful magnitudes (e.g., if a longer document embedding has larger norm), dot product may be more appropriate than cosine similarity.

### Dot Product

$$\text{sim}(\mathbf{u}, \mathbf{v}) = \mathbf{u} \cdot \mathbf{v}$$

Used in dense retrieval (DPR, Karpukhin et al., 2020) and in attention itself. Dot product is faster than cosine (no normalization), but sensitive to embedding magnitude.

### FAISS (Facebook AI Similarity Search)

For nearest-neighbor search over millions of embeddings, exact search is $O(Vd)$ per query. FAISS provides approximate nearest neighbor search with:

- **IndexFlatL2 / IndexFlatIP**: Exact search; brute force. Correct but slow.
- **IndexIVFFlat**: Inverted file index. Clusters vectors; at query time, only searches clusters near the query. ~100× speedup at slight recall loss.
- **IndexIVFPQ**: Product quantization compresses vectors to lower bit-width, fitting more in RAM. Fastest; best for billion-scale retrieval.
- **HNSW**: Graph-based index. Excellent recall-speed tradeoff; most commonly used in production vector databases (Weaviate, Qdrant, Chroma).

```python
import faiss
import numpy as np

d = 768          # embedding dimension
n = 100_000      # number of vectors

embeddings = np.random.randn(n, d).astype(np.float32)
faiss.normalize_L2(embeddings)  # normalize for cosine similarity

# Build flat (exact) index
index = faiss.IndexFlatIP(d)   # inner product = cosine for normalized vectors
index.add(embeddings)

# Query
query = np.random.randn(1, d).astype(np.float32)
faiss.normalize_L2(query)

k = 5
scores, indices = index.search(query, k)
# scores: [batch, k] cosine similarities
# indices: [batch, k] indices of nearest neighbors
```

### Re-ranking

In retrieval-augmented generation (RAG), a two-stage pipeline is common:
1. **Retrieval** (fast): Approximate nearest neighbor search returns top-$k$ candidates.
2. **Re-ranking** (slow): A cross-encoder model (which jointly encodes the query and each candidate) scores pairs more accurately. Cross-encoders are much more accurate than bi-encoders for similarity but cannot be pre-computed.

---

## References

- Mikolov, T., et al. (2013). *Distributed Representations of Words and Phrases and their Compositionality.* NeurIPS 2013.
- Pennington, J., Socher, R., & Manning, C. D. (2014). *GloVe: Global Vectors for Word Representation.* EMNLP 2014.
- Peters, M. E., et al. (2018). *Deep contextualized word representations (ELMo).* NAACL 2018.
- Devlin, J., et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL 2019.
- Press, O., & Wolf, L. (2017). *Using the Output Embedding to Improve Language Models.* EACL 2017.
- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP 2019.
- Ethayarajh, K. (2019). *How Contextual are Contextualized Word Representations?* EMNLP 2019.
- Mu, J., & Viswanath, P. (2018). *All-but-the-Top: Simple and Effective Postprocessing for Word Representations.* ICLR 2018.
- Conneau, A., et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale (XLM-R).* ACL 2020.
- Su, J., et al. (2021). *Whitening Sentence Representations for Better Semantics and Faster Retrieval.* arXiv:2103.15316.
- Karpukhin, V., et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP 2020.

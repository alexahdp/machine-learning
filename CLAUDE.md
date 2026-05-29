# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is a Markdown-only knowledge base covering machine learning from mathematical foundations to advanced architectures. There are no code files, build systems, package managers, or test suites — all content is `.md` files.

## Content Architecture

Sections are numbered to reflect a progressive learning dependency:

```
01-fundamentals → 02-core-concepts → 03-classical-ml → 04-evaluation-metrics
     → 05-neural-networks → 06-famous-architectures → 07-advanced-topics
     → 08-practical-guides → 09-resources
```

Each numbered directory has a `README.md` that indexes its contents and cross-links to adjacent sections. Topic files within a section go from simpler to more complex.

**Section roles:**
- `01-fundamentals/` — math prerequisites (linear algebra, calculus, statistics, data preprocessing)
- `02-core-concepts/` — cross-cutting ML principles (bias-variance, regularization, cross-validation, hyperparameter tuning)
- `03-classical-ml/` — algorithms organized by problem type: `regression/`, `classification/`, `clustering/`, `tree-methods/`
- `04-evaluation-metrics/` — how to measure models (metrics, loss functions, confusion matrices, ROC/AUC)
- `05-neural-networks/` — deep learning split into `fundamentals/`, `architectures/`, `training-techniques/`
- `06-famous-architectures/` — landmark CNN and segmentation models, organized by task (`computer-vision/`, `segmentation/`)
- `07-advanced-topics/` — high-level overviews of NLP, RL, GANs, advanced DL (not exhaustive — detailed coverage lives in separate repos)
- `08-practical-guides/` — end-to-end workflow: data cleaning, feature engineering, experiment planning, model selection, deployment
- `09-resources/` — glossary, cheat sheets, dataset references, tools/libraries, further reading

## Conventions

- Every section directory has a `README.md` that lists its files and provides navigation links (`← Back`, `→ Next`).
- Each topic file includes: concept explanation, key formulas/intuition, and practical context for when/why it applies.
- `09-resources/glossary.md` is the canonical place for terminology definitions — add new terms there when introducing them.
- `07-advanced-topics/` is intentionally high-level; deep dives belong in separate repositories.

## Diagrams

Use diagrams to illustrate architecture flows, data transformations, and structural concepts. Follow this priority order:

1. **Mermaid first** — write diagrams as fenced ` ```mermaid ` blocks. GitHub renders them natively as SVG. Use Mermaid for: data flow, layer-by-layer architectures, computation graphs, parallel branches (e.g. Inception module), skip connections, anything boxes-and-arrows. Keep node labels concise.

2. **Embed local images** when a real visual is significantly clearer than a text diagram — particularly for: spatial feature map progressions, multi-column layouts, or cases where the original paper figure is the canonical reference. Download images with `curl` into a local `images/` subdirectory within the same section folder (e.g. `06-famous-architectures/computer-vision/images/`). Reference them with a relative path: `![alt text](images/filename.svg)`. Prefer SVG over PNG; prefer CC-licensed sources (e.g. Wikimedia Commons).

3. **ASCII art** is acceptable for small inline illustrations (matrices, simple two-node flows) where a Mermaid block would be overkill.

**Never hotlink external images.** Always copy images locally so the repo is self-contained.

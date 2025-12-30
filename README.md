# Machine Learning Knowledge Base

A comprehensive guide to machine learning fundamentals, from mathematical foundations to advanced neural network architectures. This repository provides structured learning materials with a focus on building strong foundational knowledge.

## Table of Contents

- [Overview](#overview)
- [Learning Path](#learning-path)
- [Repository Structure](#repository-structure)
- [How to Use This Repository](#how-to-use-this-repository)
- [Prerequisites](#prerequisites)
- [Contributing](#contributing)

## Overview

This knowledge base is designed for:
- **Beginners** starting their machine learning journey
- **Practitioners** looking to strengthen their fundamentals
- **Students** seeking structured learning materials
- **Engineers** needing quick reference guides

The content emphasizes deep understanding of core concepts over superficial coverage of advanced topics. Each section builds upon previous knowledge, creating a solid foundation for machine learning expertise.

## Learning Path

The repository follows a progressive learning structure:

```
Fundamentals → Core Concepts → Classical ML → Evaluation → Neural Networks → Advanced Topics
```

### Recommended Study Sequence

1. **Start with Fundamentals** (01-fundamentals/)
   - Brush up on mathematics: linear algebra, calculus, statistics
   - Learn proper data preprocessing techniques

2. **Master Core Concepts** (02-core-concepts/)
   - Understand learning paradigms and model evaluation
   - Learn about bias-variance tradeoff and regularization

3. **Practice Classical ML** (03-classical-ml/)
   - Implement regression and classification algorithms
   - Experiment with ensemble methods and clustering

4. **Learn Evaluation Metrics** (04-evaluation-metrics/)
   - Understand how to measure model performance
   - Learn when to use specific metrics

5. **Explore Neural Networks** (05-neural-networks/)
   - Build from perceptrons to deep architectures
   - Master training techniques and optimization

6. **Study Famous Architectures** (06-famous-architectures/)
   - Learn battle-tested CNN architectures
   - Understand design principles behind successful models

7. **Survey Advanced Topics** (07-advanced-topics/)
   - Get high-level overview of NLP, RL, and GANs
   - Identify areas for deeper future study

8. **Apply Practical Skills** (08-practical-guides/)
   - Learn experiment planning and model selection
   - Understand deployment and MLOps basics

## Repository Structure

```
machine-learning/
├── 01-fundamentals/              Mathematical and statistical foundations
├── 02-core-concepts/             Essential ML concepts and principles
├── 03-classical-ml/              Traditional ML algorithms in depth
├── 04-evaluation-metrics/        Model evaluation and performance metrics
├── 05-neural-networks/           Neural network fundamentals and training
├── 06-famous-architectures/      Industry-standard model architectures
├── 07-advanced-topics/           High-level overview of advanced areas
├── 08-practical-guides/          Real-world application and deployment
└── 09-resources/                 Glossary, tools, datasets, and references
```

### Detailed Content Map

#### 01. Fundamentals
Essential mathematical and statistical knowledge for machine learning.
- Statistics and probability theory
- Linear algebra for ML
- Calculus basics (derivatives and gradients)
- Data preprocessing techniques

[→ Go to Fundamentals](./01-fundamentals/README.md)

#### 02. Core Concepts
Fundamental principles that apply across all machine learning methods.
- Learning paradigms (supervised, unsupervised, reinforcement)
- Bias-variance tradeoff
- Regularization techniques
- Cross-validation strategies
- Hyperparameter tuning

[→ Go to Core Concepts](./02-core-concepts/README.md)

#### 03. Classical Machine Learning
In-depth coverage of traditional ML algorithms.
- **Regression**: Linear, polynomial, ridge, lasso, KNN
- **Classification**: Logistic regression, SVM, Naive Bayes, KNN
- **Tree Methods**: Decision trees, Random Forest, XGBoost, LightGBM
- **Clustering**: K-means, hierarchical, DBSCAN, PCA, t-SNE

[→ Go to Classical ML](./03-classical-ml/README.md)

#### 04. Evaluation Metrics
Understanding how to measure and compare model performance.
- Regression metrics (MSE, RMSE, MAE, R²)
- Classification metrics (accuracy, precision, recall, F1)
- Confusion matrices and interpretation
- ROC curves and AUC
- Loss functions

[→ Go to Evaluation Metrics](./04-evaluation-metrics/README.md)

#### 05. Neural Networks
From basic perceptrons to modern deep learning architectures.
- **Fundamentals**: Perceptron, MLP, activation functions, backpropagation
- **Architectures**: CNNs, RNNs/LSTMs, autoencoders
- **Training**: Optimizers, batch normalization, dropout, transfer learning

[→ Go to Neural Networks](./05-neural-networks/README.md)

#### 06. Famous Architectures
Proven architectures that shaped modern deep learning.
- **Computer Vision**: LeNet, AlexNet, VGG, ResNet, Inception, EfficientNet
- **Segmentation**: U-Net, FCN, Mask R-CNN

[→ Go to Famous Architectures](./06-famous-architectures/README.md)

#### 07. Advanced Topics
High-level overviews of specialized areas (detailed coverage in separate repositories).
- Natural Language Processing overview
- Advanced deep learning concepts
- Reinforcement learning basics
- Generative Adversarial Networks

[→ Go to Advanced Topics](./07-advanced-topics/README.md)

#### 08. Practical Guides
Real-world application of machine learning knowledge.
- Experiment planning and scientific method
- Data cleaning for production systems
- Feature engineering strategies
- Model selection frameworks
- Production deployment and MLOps

[→ Go to Practical Guides](./08-practical-guides/README.md)

#### 09. Resources
Reference materials and external resources.
- Comprehensive glossary (A-Z terminology)
- Cheat sheets and quick references
- Public datasets for practice
- Tools and libraries overview
- Recommended books, courses, and papers

[→ Go to Resources](./09-resources/README.md)

## How to Use This Repository

### For Complete Beginners

1. Start with **01-fundamentals/** to ensure you have the mathematical foundation
2. Work through sections sequentially (01 → 02 → 03 → 04)
3. Practice with code examples and exercises in each section
4. Use **09-resources/glossary.md** whenever you encounter unfamiliar terms
5. Apply knowledge with **08-practical-guides/** as you progress

### For Experienced Practitioners

1. Use the repository as a reference guide
2. Jump directly to specific topics of interest
3. Review fundamentals if you need to refresh concepts
4. Consult **04-evaluation-metrics/** when choosing metrics for projects
5. Reference **06-famous-architectures/** for architecture design decisions

### For Quick Reference

1. Check **09-resources/glossary.md** for term definitions
2. Use **09-resources/cheat-sheets.md** for formula references
3. Browse section README files for topic overviews
4. Search specific algorithm files for implementation details

## Prerequisites

### Recommended Background

- **Programming**: Python proficiency (NumPy, Pandas, Matplotlib)
- **Mathematics**:
  - Linear algebra (vectors, matrices, operations)
  - Calculus (derivatives, partial derivatives, chain rule)
  - Statistics (probability, distributions, hypothesis testing)
- **Tools**: Jupyter notebooks, Git basics

### If You're Missing Prerequisites

- The **01-fundamentals/** section provides refreshers on key mathematical concepts
- Each section includes references to external resources for deeper learning
- The glossary explains mathematical notation and terminology

## Navigation Tips

- Each directory contains a `README.md` with section overview and contents
- Files are organized by topic complexity (simple → complex)
- Related topics are grouped in subdirectories
- Cross-references link related concepts across sections
- Code examples use clear comments and standard libraries

## Content Principles

This repository follows these principles:

1. **Depth over Breadth**: Thorough coverage of fundamentals rather than superficial coverage of everything
2. **Progressive Complexity**: Each section builds on previous knowledge
3. **Practical Focus**: Theory accompanied by practical examples and use cases
4. **Clear Explanations**: Mathematical concepts explained with intuition and examples
5. **Modern Relevance**: Focus on techniques used in current practice

## Contributing

This is a living knowledge base. Contributions are welcome:

- Corrections and clarifications
- Additional examples and visualizations
- Updated best practices
- New practical guides
- Resource recommendations

## Quick Access by Topic

### By Algorithm Type
- **Regression**: [03-classical-ml/regression/](./03-classical-ml/regression/)
- **Classification**: [03-classical-ml/classification/](./03-classical-ml/classification/)
- **Clustering**: [03-classical-ml/clustering/](./03-classical-ml/clustering/)
- **Neural Networks**: [05-neural-networks/](./05-neural-networks/)

### By Use Case
- **Tabular Data**: Classical ML algorithms (section 03)
- **Image Data**: CNNs (section 05), Computer Vision architectures (section 06)
- **Text Data**: NLP overview (section 07)
- **Time Series**: RNN/LSTM (section 05)

### By Learning Goal
- **Understanding Math**: [01-fundamentals/](./01-fundamentals/)
- **Building Models**: [03-classical-ml/](./03-classical-ml/) and [05-neural-networks/](./05-neural-networks/)
- **Evaluating Models**: [04-evaluation-metrics/](./04-evaluation-metrics/)
- **Deploying Models**: [08-practical-guides/production-deployment.md](./08-practical-guides/production-deployment.md)

---

**Last Updated**: December 2025

**Status**: Active development - content being migrated and enhanced from previous version

For questions or suggestions, please open an issue in the repository.

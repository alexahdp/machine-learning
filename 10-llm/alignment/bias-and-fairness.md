# Bias and Fairness in LLMs

## Table of Contents

1. [Sources of Bias](#sources-of-bias)
2. [Types of Bias in LLMs](#types-of-bias-in-llms)
3. [Measurement Benchmarks](#measurement-benchmarks)
4. [Debiasing Approaches](#debiasing-approaches)
5. [Fairness Definitions and Their Incompatibility](#fairness-definitions-and-their-incompatibility)
6. [Limitations of Current Approaches](#limitations-of-current-approaches)
7. [Practical Guidance](#practical-guidance)
8. [References](#references)

---

## Sources of Bias

Bias in LLMs does not arise from a single cause — it is accumulated across the full data and training pipeline. Understanding the sources is prerequisite to addressing them.

### Training Data Bias

Web-scraped corpora (C4, The Pile, Common Crawl) reflect the demographics and views of internet authors, who are not representative of the global population. English is vastly over-represented; educated, Western, younger, and male perspectives dominate. Content that is commonly written and shared online (news, Wikipedia, Reddit, GitHub) reflects the biases of those communities.

Beyond representation, the internet reproduces and amplifies societal stereotypes. Correlations between demographic groups and traits, occupations, or behaviors in training data encode historical inequalities that may be unjust to reproduce in model outputs. Bolukbasi et al. (2016) demonstrated that word2vec embeddings trained on Google News articles encoded gender stereotypes (man:doctor :: woman:nurse) — the same phenomenon manifests in LLMs at larger scale.

### Annotation Bias

Human-labeled data introduces annotator demographic biases. Sap et al. (2019) showed that toxicity annotations are skewed by annotator race — annotators with less exposure to African American Vernacular English (AAVE) systematically rated AAVE-using text as more toxic than Standard American English text with equivalent semantic content. Sap et al. (2022) extended this finding to show that state-of-the-art models trained on such data inherit the bias.

Annotators also show political, cultural, and subcultural biases. Labeling guidelines attempt to standardize judgments but cannot eliminate disagreement, especially on subjective or culturally variable content.

### Historical Bias

Training data encodes historical patterns. An LLM trained on historical data will reproduce the fact that most historical doctors were male, most CEOs were white, most criminals in court records were from specific demographics. Even if these are historically accurate reflections, applying them to present-day predictions or generation tasks perpetuates injustice — historical patterns often reflect unjust structural constraints rather than intrinsic properties of individuals.

This is particularly problematic for language tasks that involve inference about individuals from group membership (coreference resolution, sentiment inference, text completion about named individuals).

### Representation Bias

Languages and dialects with sparse internet presence are dramatically underrepresented in training data. Joshi et al. (2020) categorized world languages by their digital resource availability — the vast majority fall in the lowest-resource categories, meaning models trained on web data effectively ignore them. A model with poor performance on low-resource languages is unfair to speakers of those languages and is less useful as a general-purpose tool.

Within English, non-standard dialects (AAVE, Scottish English, Indian English) are underrepresented in high-quality training sources (Wikipedia, books, academic text), while being present in some lower-quality sources (social media). This creates models that perform worse on non-standard dialect inputs and that may associate non-standard dialects with lower credibility.

---

## Types of Bias in LLMs

### Stereotyping

LLMs associate demographic groups with traits, roles, or behaviors in ways that reflect and reinforce stereotypes. This manifests in both explicit and implicit forms:

- **Occupation stereotyping**: text completions about "the nurse" default to feminine pronouns; completions about "the engineer" default to masculine pronouns
- **Nationality stereotyping**: completions about people from specific countries reflect common stereotypes rather than individual variation
- **Religious stereotyping**: different sentiment, word associations, and attribute predictions for different religious groups

Sheng et al. (2019) documented sentiment disparities in GPT-2 completions — prompts introducing a "Black man" received completions with more negative sentiment than prompts introducing a "White man," across multiple completion generations.

### Toxicity

Models are more likely to generate or continue toxic text involving certain demographic groups than others. Gehman et al. (2020) — the RealToxicityPrompts study — showed that GPT-2 generated toxic content at rates that varied systematically by the demographic groups mentioned in the prompt, even for non-toxic prompts.

This is not only a problem with harmful input — models can spontaneously introduce toxicity that was not implied by the prompt when certain groups are mentioned.

### Sentiment Disparity

Different demographic groups receive systematically different sentiment in model-generated text. This includes:
- Adjective associations: different descriptive adjectives generated for the same role applied to different groups
- Sentiment in narratives: descriptions of members of different groups show systematic positive/negative valence differences
- Evaluation asymmetries: the same action receives different moral evaluation when attributed to different groups

Huang et al. (2019) systematically measured sentiment disparities across gender and race in GPT-2 across multiple occupational contexts.

### Coreference Resolution Bias

**WinoBias** (Zhao et al., 2018) exposed a specific failure mode in coreference resolution — the task of determining which pronouns refer to which entities. The benchmark uses sentences like:

> "The doctor asked the nurse to help her with the operation."

Models with gender bias default to resolving "her" based on gender stereotypes (doctors are male → "her" refers to nurse) rather than syntactic structure. WinoBias measured pronoun resolution accuracy across pro-stereotypical and anti-stereotypical configurations. Models of the time (2018) showed dramatic accuracy disparities between stereotypical and counter-stereotypical resolutions — over 20 percentage point gaps.

Subsequent work showed that many fine-tuning approaches reduce but do not eliminate this gap.

### Multilingual Bias

Models perform substantially worse on non-English inputs and exhibit different bias profiles across languages. A model may be tested extensively for English bias and show acceptable behavior, while exhibiting pronounced biases in other languages where testing is sparse. This is a fairness concern: users who interact in non-English languages receive less well-calibrated and potentially more biased outputs.

Nationality biases in sentiment also shift across languages — a model prompted in Spanish may show different cultural associations than the same model prompted in English, because the training data and cultural contexts differ.

---

## Measurement Benchmarks

### WinoBias and WinoGender

**WinoBias** (Zhao et al., 2018): 3,160 sentence pairs testing gendered coreference resolution in occupational contexts. Each pair has a pro-stereotypical version (pronoun aligns with occupational stereotype) and an anti-stereotypical version. The gap between accuracy on pro- and anti-stereotypical versions quantifies gender bias.

**WinoGender** (Rudinger et al., 2018): Similar design, focuses on 720 sentences with a broader set of occupations and includes a "someone" condition to control for linguistic factors.

**Key limitation**: These benchmarks test gender bias in a specific linguistic task (coreference). They do not measure stereotyping in generation, sentiment disparity, or bias in downstream tasks.

### StereoSet

**StereoSet** (Nadeem et al., 2021): Measures stereotypical bias across race, gender, religion, and profession with 17,000 samples. Two subtasks:

- **Intrasentence**: given a sentence with a blank, choose between a stereotypical, anti-stereotypical, or unrelated word
- **Intersentence**: given a context sentence, choose the most appropriate continuation from stereotypical, anti-stereotypical, or unrelated options

Two scores are reported:
- **Language Model Score (LMS)**: how often the model prefers meaningful completions over unrelated ones (measures language modeling quality)
- **Stereotype Score (SS)**: how often the model prefers stereotypical completions over anti-stereotypical ones (50% = no bias, >50% = stereotypical bias)

The ideal model has high LMS and 50% SS. A model that achieves low SS by being incoherent (choosing anti-stereotypical completions that are also ungrammatical) is not debiased — it is broken. The ICAT (Idealized CAT) score combines both: $\text{ICAT} = \text{LMS} \times \min(\text{SS}, 100 - \text{SS}) / 50$.

### BBQ (Bias Benchmark for QA)

**BBQ** (Parrish et al., 2022): 58,492 examples covering nine social dimensions (age, disability status, gender, nationality, physical appearance, race/ethnicity, religion, socioeconomic status, sexual orientation). Each question is presented in two contexts:

- **Ambiguous context**: the answer cannot be determined from the context (the model should say "Unknown")
- **Disambiguated context**: the context provides enough information to answer correctly

Bias is measured as how often the model relies on stereotypes in the ambiguous context (when it should say "Unknown") and how often it answers correctly in the disambiguated context. BBQ found that even models with strong benchmark performance rely heavily on stereotypes under ambiguity.

### CrowS-Pairs

**CrowS-Pairs** (Nangia et al., 2020): 1,508 sentence pairs covering nine bias categories. Each pair consists of a more stereotyped and a less stereotyped sentence, differing minimally in wording. Bias is measured as the rate at which the model assigns higher likelihood to the more stereotyped sentence. A perfectly unbiased model would assign equal likelihood to each pair.

**Limitation**: Neveol et al. (2022) identified significant annotation quality issues in CrowS-Pairs, with many sentence pairs not satisfying the intended criteria. Results on this benchmark should be interpreted cautiously.

### BOLD (Bias in Open-Ended Language Generation)

**BOLD** (Dhamala et al., 2021): Evaluates bias in open-ended generation. 23,679 prompts derived from Wikipedia introductions to real people, grouped by demographic dimensions (race, gender, religious affiliation, political ideology, profession). Generated completions are analyzed for sentiment, toxicity, and regard (overall portrayal) using automated classifiers.

BOLD provides a more ecologically valid measure than coreference or sentence completion tasks — it tests generation that resembles real deployment conditions.

---

## Debiasing Approaches

### Data-Level Approaches

**Counterfactual data augmentation (CDA)**: Swap demographic identifiers in training examples and include both versions (original and swapped). For gender bias, replace "he/she/his/her/him/her" systematically and include both gendered versions in training. This teaches the model to produce symmetric outputs across groups.

Limitations: CDA is effective for binary binary attributes (gender) but complex for intersectional or multi-valued attributes. Swapping tokens can produce semantically awkward text ("the queen became king") that degrades overall language quality.

**Balanced dataset sampling**: Ensure training data represents demographic groups proportionally for attributes that should be independent of the model's judgments. In practice, this requires identifying and balancing which attributes, which is non-trivial for large web-scraped corpora.

**Deduplication**: Removing duplicate text reduces the amplification of any single source's biases, since duplicated content implicitly receives higher weight in training.

### Training-Level Approaches

**Adversarial training**: Train an adversary to predict demographic attributes from the model's internal representations. Add a loss term that penalizes predictability of demographics from representations that should be demographically invariant:

$$\mathcal{L} = \mathcal{L}_\text{task} - \lambda \cdot \mathcal{L}_\text{adversary}$$

This incentivizes the model to learn representations that are useful for the task but not predictive of demographic attributes, a form of fairness through representation.

**Debiasing objectives**: Add explicit fairness regularization terms to the training loss. For example, penalize differences in output distributions across demographic groups:

$$\mathcal{L}_\text{fairness} = \sum_{g_1, g_2} \text{KL}(P(y | g_1) \| P(y | g_2))$$

This requires defining which output distributions should be equal across groups, which requires normative judgments about what equality means.

**INLP (Iterative Nullspace Projection)** (Ravfogel et al., 2020): Iteratively identify and project out directions in the embedding space that are predictive of a protected attribute. After $k$ iterations, the attribute is approximately linearly unpredictable from the representations. Effective for probing-measurable biases in contextual embeddings.

### Post-Processing Approaches

**Output filtering and calibration**: Apply demographic parity constraints post-hoc: if the model produces systematically different sentiment across groups, re-calibrate outputs to equalize distributions. This is straightforward to implement but may require knowing the demographic group being discussed, and may degrade quality by imposing artificial equality.

**Re-ranking**: Generate multiple completions and select the one that best satisfies fairness criteria alongside quality criteria.

### Prompt-Level Approaches

**Instruction-based debiasing**: Instruct the model to "ignore the gender/race/religion of the people in the scenario" or "respond in a way that does not depend on demographic characteristics." Studies show partial effectiveness: models can be instructed to attend less to demographic signals, but compliance is imperfect, especially for strong stereotypes encoded in the training distribution. Ganguli et al. (2023) showed that explicit debiasing instructions reduce but do not eliminate bias in generated text.

---

## Fairness Definitions and Their Incompatibility

Fairness is not a single concept — there are competing formal definitions, and Chouldechova (2017) and Kleinberg et al. (2016) independently proved that most cannot be simultaneously satisfied (except in degenerate cases).

**Demographic parity (statistical parity)**: Outputs should be equally distributed across demographic groups. Formally: $P(\hat{Y} = 1 | A = 0) = P(\hat{Y} = 1 | A = 1)$ where $A$ is the protected attribute and $\hat{Y}$ is the model output.

**Equalized odds**: True positive rates and false positive rates should be equal across groups. Formally: $P(\hat{Y} = 1 | Y = y, A = 0) = P(\hat{Y} = 1 | Y = y, A = 1)$ for $y \in \{0, 1\}$.

**Calibration (individual fairness)**: If two individuals are equally likely to belong to the positive class, they should receive equal predictions, regardless of group membership. Formally: $P(Y = 1 | \hat{Y} = p, A = 0) = P(Y = 1 | \hat{Y} = p, A = 1) = p$.

**The incompatibility theorems**: When the base rates of the outcome differ across groups ($P(Y = 1 | A = 0) \neq P(Y = 1 | A = 1)$) — which is common in practice due to historical structural inequalities — demographic parity, equalized odds, and calibration cannot all be simultaneously satisfied by a non-trivial classifier. Improving on one metric necessarily degrades another.

This is not merely a technical constraint — it reflects a genuine normative disagreement about what fairness requires. Should a model produce equal rates of positive sentiment across groups (demographic parity), even if the true underlying rates differ? Or should it accurately reflect underlying rates (calibration), at the cost of reproducing unequal outcomes? These are political and ethical questions that cannot be resolved through optimization alone.

For LLMs, these tensions manifest concretely: a model that generates equally positive descriptions of all demographic groups (demographic parity) may be inaccurate in contexts where group-specific facts are relevant; a model that accurately represents historical group-level statistics (calibration) may perpetuate historical injustices.

---

## Limitations of Current Approaches

**Narrow benchmark coverage**: Current bias benchmarks test specific, measurable manifestations of bias (coreference resolution, sentence completion likelihood ratios). They do not capture the full range of social harms LLMs can cause, including subtle forms of othering, microaggressions, or harm through cumulative small biases in long-form text.

**Static benchmarks become saturated**: As benchmarks become public, models can be inadvertently or deliberately over-fitted to them. High benchmark scores may reflect narrow benchmark-specific improvements rather than general debiasing.

**Debiasing is not robustly generalizing**: Reducing bias on WinoBias does not reliably reduce bias in BOLD or BBQ. Techniques that are effective for one bias type or domain often do not generalize. There is no universal debiasing approach.

**Intersectionality is understudied**: Most benchmarks test single-axis bias (gender OR race OR religion). Intersectional identities (Black women, LGBTQ+ people of color) may experience bias differently and more severely than single-axis analysis predicts. Intersection-aware evaluation and debiasing remains a research gap.

**Defining harm is contested**: What counts as harmful stereotyping versus accurate statistical description is culturally contested. Benchmark design implicitly encodes normative choices about which associations are harmful.

---

## Practical Guidance

**Document model limitations explicitly**: Model cards (Mitchell et al., 2019) should specify known bias risks, the demographic groups on which the model was evaluated, and known failure modes. This is a minimum standard for responsible deployment.

**Test on diverse populations before deployment**: Do not rely only on standard benchmarks. Test model outputs on prompts and scenarios relevant to the deployment population, including populations that are underrepresented in benchmark design.

**Audit regularly and at multiple levels**: Bias can emerge at the intersection of the model, the system prompt, and the application context. A model that is unbiased in isolation may exhibit bias in a specific deployment configuration. Audit at the deployed system level, not just the base model level.

**Prefer task-specific bias evaluation**: Generic bias benchmarks may not capture the bias risks most relevant to a specific application. Design custom evaluations for the demographic groups and decision types relevant to the deployment context.

**Involve affected communities**: Bias evaluation and mitigation design is most effective when it incorporates input from the communities most likely to be harmed. External fairness audits by third parties are more credible than self-audits.

**Report error rates, not just aggregate metrics**: Aggregate performance metrics hide disparities across subgroups. Always report performance broken down by demographic group and identify which groups experience the largest error rates.

---

## References

- Bolukbasi, T. et al. (2016). *Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings*. NeurIPS.
- Zhao, J. et al. (2018). *Gender Bias in Coreference Resolution: Evaluation and Debiasing Methods*. NAACL. [WinoBias]
- Rudinger, R. et al. (2018). *Gender Bias in Coreference Resolution*. NAACL. [WinoGender]
- Nadeem, M. et al. (2021). *StereoSet: Measuring stereotypical bias in pretrained language models*. ACL.
- Parrish, A. et al. (2022). *BBQ: A Hand-Built Bias Benchmark for Question Answering*. ACL Findings.
- Nangia, N. et al. (2020). *CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models*. EMNLP.
- Dhamala, J. et al. (2021). *BOLD: Dataset and Metrics for Measuring Biases in Open-Ended Language Generation*. FAccT.
- Sheng, E. et al. (2019). *The Woman Worked as a Babysitter: On Biases in Language Generation*. EMNLP.
- Gehman, S. et al. (2020). *RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models*. EMNLP Findings.
- Sap, M. et al. (2019). *Annotators with Attitudes: How Annotator Beliefs And Identities Bias Toxic Language Detection*. NAACL.
- Sap, M. et al. (2022). *Annotators with Attitudes: How Annotator Beliefs And Identities Bias Toxic Language Detection*. ACL.
- Ravfogel, S. et al. (2020). *Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection*. ACL. [INLP]
- Chouldechova, A. (2017). *Fair Prediction with Disparate Impact: A Study of Bias in Recidivism Prediction Instruments*. Big Data.
- Kleinberg, J. et al. (2016). *Inherent Trade-Offs in the Fair Determination of Risk Scores*. ITCS.
- Ganguli, D. et al. (2023). *The Capacity for Moral Self-Correction in Large Language Models*. arXiv:2302.07459.
- Joshi, P. et al. (2020). *The State and Fate of Linguistic Diversity and Inclusion in the NLP World*. ACL.
- Huang, P. et al. (2019). *Reducing Sentiment Bias in Language Models via Counterfactual Evaluation*. EMNLP Findings.
- Mitchell, M. et al. (2019). *Model Cards for Model Reporting*. FAccT.
- Neveol, A. et al. (2022). *French CrowS-Pairs: Extending a challenge dataset for measuring social bias in masked language models to a language other than English*. ACL.

---

*← [Safety Techniques](./safety-techniques.md) | → [Agents Overview](../agents/README.md)*

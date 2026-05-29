# Alignment

Alignment is the challenge of ensuring that AI systems — and LLMs in particular — pursue the goals we actually intend, behave safely, and remain honest. The canonical framing is **HHH**: Helpful, Harmless, Honest. These properties sound straightforward but are technically subtle: they can conflict with each other, are hard to specify precisely, and are difficult to measure reliably.

Alignment is an **active, unsolved research area**. Current techniques (RLHF, Constitutional AI, DPO) produce substantially better behavior than unaligned base models, but no method provides guarantees. Reward hacking, hallucination, jailbreaks, and systematic bias remain open problems with significant practical and safety consequences.

## Contents

| File | Description |
|------|-------------|
| [alignment-fundamentals.md](./alignment-fundamentals.md) | Value alignment, the specification problem, Goodhart's Law, reward hacking, mesa-optimization, RLHF limitations, Constitutional AI, scalable oversight |
| [hallucination.md](./hallucination.md) | Types and causes of hallucination, factuality vs faithfulness, detection methods (FactScore, NLI, self-consistency), mitigation strategies, benchmarks |
| [safety-techniques.md](./safety-techniques.md) | Training-time and inference-time safety, red-teaming, jailbreaks, Constitutional AI, output filtering, watermarking, safety-capability trade-offs |
| [bias-and-fairness.md](./bias-and-fairness.md) | Sources of bias in LLMs, stereotyping and toxicity, fairness benchmarks (WinoBias, StereoSet, BBQ), debiasing approaches, fairness definition incompatibilities |

## Navigation

- [← Back to LLM Overview](../README.md)
- [← Previous: Prompting](../prompting/README.md)
- [→ Next: Agents](../agents/README.md)

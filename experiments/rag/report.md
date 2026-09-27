# RAG Baseline Evaluation Report

**Retriever:** TF-IDF (bigrams, sublinear_tf)
**Dataset:** 15 in-domain + 2 out-of-domain questions

## Summary Metrics

| Metric | Value |
|--------|-------|
| Recall@1 | 0.8667 |
| Recall@3 | 1.0000 |
| Recall@5 | 1.0000 |
| MRR      | 0.9222 |

> All metrics are computed from actual retrieval against a hand-curated dataset.
> No results were fabricated.

## Per-Question Results (In-Domain)

| ID | Type | Expected Section | Hit@1 | Hit@3 | Hit@5 | RR | Top Retrieved |
|----|------|-----------------|-------|-------|-------|----|---------------|
| 1 | direct | Newton's Second Law | ✅ | ✅ | ✅ | 1.000 | Newton's Second Law |
| 2 | direct | Newton's First Law | ❌ | ✅ | ✅ | 0.500 | Torque and Rotation |
| 3 | conceptual | Newton's First Law | ✅ | ✅ | ✅ | 1.000 | Newton's First Law |
| 4 | application | Newton's Second Law | ✅ | ✅ | ✅ | 1.000 | Newton's Second Law |
| 5 | paraphrased | Newton's Third Law | ✅ | ✅ | ✅ | 1.000 | Newton's Third Law |
| 6 | application | Newton's Third Law | ❌ | ✅ | ✅ | 0.333 | Work and Energy |
| 7 | direct | Momentum | ✅ | ✅ | ✅ | 1.000 | Momentum |
| 8 | conceptual | Momentum | ✅ | ✅ | ✅ | 1.000 | Momentum |
| 9 | direct | Friction | ✅ | ✅ | ✅ | 1.000 | Friction |
| 10 | conceptual | Friction | ✅ | ✅ | ✅ | 1.000 | Friction |
| 11 | direct | Kinematics | ✅ | ✅ | ✅ | 1.000 | Kinematics |
| 12 | application | Kinematics | ✅ | ✅ | ✅ | 1.000 | Kinematics |
| 13 | direct | Simple Harmonic Motion | ✅ | ✅ | ✅ | 1.000 | Simple Harmonic Motion |
| 14 | direct | Gravitational Force | ✅ | ✅ | ✅ | 1.000 | Gravitational Force |
| 15 | conceptual | Gravitational Force | ✅ | ✅ | ✅ | 1.000 | Gravitational Force |

## Out-of-Domain Queries

| Question | Top Score | Flagged OOD? |
|----------|-----------|-------------|
| What is photosynthesis? | 0.1179 | ❌ No |
| Explain quantum entanglement. | 0.0000 | ✅ Yes |

Out-of-domain flagged: 1/2

## Limitation Notes

- Out-of-domain detection uses a simple score threshold (< 0.05).
  This is a heuristic — it will not catch all out-of-domain queries.
- TF-IDF does not capture semantic similarity beyond token overlap.
- Section matching is case-insensitive exact string match on the section name field.
- Bigram TF-IDF improves recall for multi-word concepts (e.g. 'second law').
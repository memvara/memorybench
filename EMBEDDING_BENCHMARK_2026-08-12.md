# Embedding benchmark: Gemini vs OpenAI

Date: 2026-08-12

## Executive summary

Supermemory's cloud embedding configuration is `gemini-embedding-001` at 1,536 dimensions, with `RETRIEVAL_DOCUMENT` for indexed content and `RETRIEVAL_QUERY` for queries. This benchmark used the same model ID and task configuration through the Gemini Developer API; production prefers Vertex, so this is a quality comparison rather than a production-serving latency benchmark.

There is no universal winner:

- **General retrieval:** Gemini is substantially stronger. Across NFCorpus, SciFact, and ArguAna, Gemini achieved 0.680 pooled nDCG@10 versus 0.448 for OpenAI small and 0.472–0.474 for OpenAI large.
- **Conversational evidence retrieval:** OpenAI large is stronger overall. At 1,536 dimensions it achieved 80.0% Recall@10 versus 75.6% for production-configured Gemini.
- **Gemini's main product weaknesses:** multi-hop evidence, world-knowledge implications, preference evidence, and implicit connections. These are cases where the query and decisive evidence do not share direct wording.
- **Temporal retrieval is not a weakness:** production Gemini reached 75.3% Recall@10 on official LoCoMo temporal questions and beat both OpenAI models. `SEMANTIC_SIMILARITY` raised it to 83.5% but regressed product-specific preference, implicit, and update cases.
- **More dimensions do not solve the problem:** Gemini 768, 1,536, and 3,072 were effectively tied. OpenAI large 3,072 was also effectively tied with its 1,536-dimensional form.
- **Update traversal works when the latest node is retrieved:** production Gemini seeded the latest node in 64.3% of valid update chains, then backward traversal recovered the old parent. The remaining bottleneck is latest-node seed recall.

## Production configuration verified

The current cloud path uses:

- model: `gemini-embedding-001`
- dimensions: 1,536 (`BIG_DIMENSIONS_NEW`)
- indexed content: `RETRIEVAL_DOCUMENT`
- search queries: `RETRIEVAL_QUERY`
- normalized vectors and cosine similarity

Relevant implementation:

- `../mono/packages/lib/vertex-embeddings.ts`
- `../mono/packages/lib/constants.ts`
- `../mono/packages/lib/embeddings.ts`
- `../mono/apps/api/src/services/workflows/ai-steps/generate-embeddings.ts`
- `../mono/apps/api/src/routes/search/handlers.ts`

Self-hosted Supermemory uses a separate local BGE-base model and is outside this comparison.

## Product-specific benchmark

### Dataset and method

- LoCoMo: 1,527 non-adversarial questions with completely resolvable evidence IDs
- Official LoCoMo categories: 1 multi-hop, 2 temporal, 3 world-knowledge/open-domain, 4 single-hop, 5 adversarial
- LoCoMo adversarial questions are excluded from positive-qrel retrieval because their annotated evidence is a tempting distractor; abstention requires a separate negative-retrieval metric
- ConvoMem: deterministic sample of 50 candidates per category; 202 cases had complete, duplicate-safe evidence-message matches
- Total: 1,729 cases, 15,937 unique document representations, 1,718 unique queries
- Search scope: each query searched only its own conversation haystack
- Representation: speaker + source message + available session date
- No reranker, BM25, query rewriting, memory extraction, LLM judge, or answer model

The representation is production-shaped but not production-identical: Supermemory normally embeds contextualized chunks or extracted memories. This benchmark embeds evidence-bearing conversation messages with speaker/date context and isolates embedding retrieval rather than the complete memory pipeline.

### Overall results

| Configuration | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|
| Gemini retrieval, 1,536d (production) | 67.0% | 75.6% | 0.582 | 0.598 |
| Gemini QA query, 1,536d | 65.9% | 75.0% | 0.574 | 0.590 |
| Gemini fact verification query, 1,536d | 66.1% | 75.4% | 0.576 | 0.593 |
| Gemini semantic similarity, 1,536d | 68.8% | 77.6% | 0.592 | 0.611 |
| Gemini retrieval, 768d | 66.7% | 75.2% | 0.575 | 0.591 |
| Gemini retrieval, 3,072d | 66.7% | 75.4% | 0.580 | 0.596 |
| OpenAI small, 1,536d | 65.7% | 75.5% | 0.558 | 0.583 |
| OpenAI large, 1,536d | 70.4% | **80.0%** | 0.601 | 0.627 |
| OpenAI large, 3,072d | **70.9%** | 79.7% | **0.607** | **0.631** |

### Where production Gemini is weak

| Use case | Gemini R@10 | OpenAI large 1,536d R@10 | Finding |
|---|---:|---:|---|
| LoCoMo multi-hop | 49.1% | 55.6% | Weak for all models; OpenAI large is moderately better when evidence must be combined |
| LoCoMo world knowledge | 46.8% | 44.5% | Weak for all models; Gemini is slightly better, so a model swap is not the answer |
| ConvoMem preferences | 77.1% | 91.4% | Gemini often retrieves the preference discussion but misses the decisive statement |
| ConvoMem implicit connections | 53.2% | 68.6% | Gemini struggles when a request must connect to indirectly expressed personal context |
| ConvoMem changing evidence | 87.5% | 96.4% | Gemini is good but more likely to miss one side of a fully matched old→new pair |

Gemini is strong on direct retrieval:

- LoCoMo single-hop: 84.7% Recall@10
- LoCoMo temporal: 75.3%
- ConvoMem user evidence: 100%
- ConvoMem assistant facts: 100%

Representative failures where Gemini missed top 10 and OpenAI large succeeded include:

- Inferring political leaning from a negative encounter with religious conservatives and support for LGBTQ rights.
- Inferring road-trip aversion after a frightening family accident.
- Recommending project tools from a stated preference for visual Kanban flow.
- Recommending a rental car from a prior preference for an engaging manual-transmission hatchback.

The recurring pattern is semantic implication: Gemini's highest-scoring results often repeat query vocabulary but omit the decisive fact.

## Gemini task-type ablation

| Query task | Document task | Overall R@10 | Assessment |
|---|---|---:|---|
| `RETRIEVAL_QUERY` | `RETRIEVAL_DOCUMENT` | 75.6% | Best balanced production default |
| `QUESTION_ANSWERING` | `RETRIEVAL_DOCUMENT` | 75.0% | Slight overall regression |
| `FACT_VERIFICATION` | `RETRIEVAL_DOCUMENT` | 75.4% | No overall win |
| `SEMANTIC_SIMILARITY` | `SEMANTIC_SIMILARITY` | 77.6% | Strong LoCoMo aggregate, but product-specific regressions |
| `RETRIEVAL_DOCUMENT` | `RETRIEVAL_DOCUMENT` | 70.5% | Confirms asymmetric query/document pairing matters |
| `CLASSIFICATION` | `RETRIEVAL_DOCUMENT` | 69.7% | Negative control; worse |
| `CLUSTERING` | `RETRIEVAL_DOCUMENT` | 61.9% | Worst negative control |
| `CODE_RETRIEVAL_QUERY` | `RETRIEVAL_DOCUMENT` | 70.6% | Domain-mismatched negative control; worse |

`SEMANTIC_SIMILARITY` deserves a narrow follow-up, not a global switch. Compared with production retrieval, it improved:

- temporal: 75.3% → 83.5%
- single-hop: 84.7% → 87.0%
- multi-hop: 49.1% → 50.6%

But it regressed:

- world knowledge: 46.8% → 36.4%
- preferences: 77.1% → 60.0%
- changing evidence: 87.5% → 76.8%
- implicit connections: same Recall@10 but materially worse MRR

Google documents semantic similarity as a symmetric similarity task rather than retrieval. Keep the production pairing unless routing can be evaluated by query type.

## Traversal and context expansion

Supermemory stores parent→child `updates`, `extends`, and `derives` edges. The benchmark keeps context-window expansion separate from graph traversal.

### Context expansion

Adding one previous/next message around top-five results helped production Gemini on:

- preferences: 62.9% direct Recall@5 → 80.0%
- implicit connections: 45.5% → 60.9%
- multi-hop: 38.0% → 41.6%

This supports preserving or fetching conversational neighborhood context after retrieval.

### Update history

Twenty-eight sampled changing-evidence cases had complete, chronologically ordered old→new evidence pairs. To mirror production eligibility:

- only the latest node can seed traversal;
- the old parent is not eligible for vector-search seeding;
- history is recovered by following the stored old→new edge backward from the latest child.

| Model | Eligible direct evidence R@5 | Latest node seeded@5 | After history expansion R@5 |
|---|---:|---:|---:|
| Gemini production | 32.1% | 64.3% | 64.3% |
| OpenAI small | 41.1% | 82.1% | 82.1% |
| OpenAI large 1,536d | 41.1% | 82.1% | 82.1% |
| OpenAI large 3,072d | 39.3% | 78.6% | 78.6% |

The eligible direct metric counts only the latest child because stale parents are excluded from production vector search. History expansion then recovers the old parent, doubling evidence recall whenever the child is seeded. The edges are dataset-derived oracle relations, not independently extracted relations, so this does not measure relation-construction accuracy. Improving latest-node seed recall remains the priority.

## General retrieval benchmark

A separate exact-cosine benchmark used official IDs and qrels from NFCorpus, SciFact, and ArguAna:

- 17,490 corpus IDs represented by 17,400 unique embedded texts
- 2,029 test query IDs represented by 1,921 unique embedded texts
- duplicate texts share embeddings but retain every original ID and qrel
- exact blocked NumPy ranking

### Query-weighted pooled result

| Configuration | nDCG@10 | Recall@10 | Recall@100 | MRR@10 |
|---|---:|---:|---:|---:|
| Gemini retrieval, 1,536d | **0.680** | **0.841** | **0.903** | **0.649** |
| OpenAI small, 1,536d | 0.448 | 0.731 | 0.889 | 0.383 |
| OpenAI large, 1,536d | 0.472 | 0.757 | 0.897 | 0.404 |
| OpenAI large, 3,072d | 0.474 | 0.758 | 0.898 | 0.407 |

### Per dataset nDCG@10

| Configuration | NFCorpus | SciFact | ArguAna | Dataset macro average |
|---|---:|---:|---:|---:|
| Gemini retrieval, 1,536d | **0.433** | **0.897** | **0.691** | **0.674** |
| OpenAI small, 1,536d | 0.385 | 0.724 | 0.403 | 0.504 |
| OpenAI large, 1,536d | 0.419 | 0.774 | 0.420 | 0.537 |
| OpenAI large, 3,072d | 0.421 | 0.782 | 0.420 | 0.541 |

Gemini wins every dataset, so the conclusion is not an artifact of ArguAna's larger query count. This is strong evidence against replacing Gemini globally based only on conversational evidence retrieval.

## Dimensions, storage, and price

Quality was effectively flat across Gemini dimensions:

- 768d: 75.2% product Recall@10
- 1,536d: 75.6%
- 3,072d: 75.4%

OpenAI large also showed no meaningful gain from 1,536d to 3,072d. Therefore:

- Do not increase Gemini to 3,072 dimensions for quality.
- A 768-dimensional Gemini shadow-index experiment is credible: it halves raw vector storage and distance-computation work with negligible observed quality change.
- OpenAI large at 1,536 dimensions is preferable to 3,072 for this workload.

Official list pricing at benchmark time:

- Gemini Embedding 001: $0.15 per 1M input tokens
- OpenAI text-embedding-3-small: $0.02 per 1M input tokens
- OpenAI text-embedding-3-large: $0.13 per 1M input tokens

The benchmark did not record authoritative token counts, so it does not claim exact run cost. Price also excludes batching, retries, storage, serving architecture, and migration costs.

## Recommendations

1. **Keep Gemini retrieval task types in production.** No alternate task type is a safe global improvement.
2. **Do not move to 3,072 dimensions.** It adds storage and compute without measurable quality.
3. **Evaluate Gemini 768d in an online shadow index.** It is the clearest storage/compute opportunity.
4. **Prioritize multi-hop, world-knowledge implications, preferences, and implicit connections.** Use query rewriting and richer contextual memory representations.
5. **Preserve conversational neighborhood context.** Context expansion produced larger gains than dimension changes.
6. **Improve latest-node retrieval before relying on graph history.** Production traversal cannot help when the latest child is not seeded.
7. **Consider OpenAI large only for a targeted conversational-memory index or fallback.** It improves preferences, updates, implicit connections, and multi-hop evidence but loses badly on all three general retrieval datasets.
8. **Run a production-shadow A/B before changing models.** Re-embed actual extracted memories/chunks and use the real Turbopuffer filters, reranking, answer generation, latency, and cost path.

## Reproduction and artifacts

```bash
node scripts/embedding-model-benchmark.mjs \
  --suite all \
  --convomem-sample 50 \
  --output product-final-2026-08-12

node scripts/embedding-model-benchmark.mjs \
  --suite general \
  --configs gemini-retrieval-1536,openai-small-1536,openai-large-1536,openai-large-3072 \
  --output general-embeddings-2026-08-12

python3 scripts/evaluate-general-retrieval.py \
  --output general-standard-ndcg-2026-08-12
```

Datasets, binary embedding caches, and run JSON are stored under ignored `data/embedding-model-benchmark/`. The final product summary was derived from the completed per-case artifact by applying official LoCoMo labels, excluding adversarial and partially resolved qrels, and recalculating production-eligible update traversal. A clean rerun of the corrected script was attempted, but the sandbox process was killed during re-embedding after the changed case set invalidated the cache hash; this does not affect the completed per-case vectors or filtered metrics.

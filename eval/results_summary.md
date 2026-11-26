# Evaluation Results Summary

## Overall by Mode

| Mode | N | Correctness | Faithfulness | Citation Acc | Override Obed | Mean Latency (ms) | Median Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| rag | 60 | 3.417 | 4.85 | 4.983 | 0.7 | 1880.17 | 1218.6 |
| zero_shot | 60 | 2.2 | 4.933 | 4.983 | 0.0 | 1725.44 | 1001.44 |

## Breakdown by Mode & Slice

| Mode | Slice | N | Correctness | Faithfulness | Citation Acc | Override Obed | Mean Latency (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| rag | static | 20 | 1.8 | 5.0 | 5.0 | - | 936.85 |
| rag | live | 20 | 3.85 | 4.55 | 4.95 | - | 3008.95 |
| rag | temporal_override | 10 | 4.2 | 5.0 | 5.0 | 0.7 | 2586.69 |
| rag | unanswerable | 10 | 5.0 | 5.0 | 5.0 | - | 802.72 |
| zero_shot | static | 20 | 1.45 | 4.9 | 5.0 | - | 2039.38 |
| zero_shot | live | 20 | 1.0 | 5.0 | 5.0 | - | 1010.11 |
| zero_shot | temporal_override | 10 | 3.3 | 4.8 | 4.9 | 0.0 | 3291.28 |
| zero_shot | unanswerable | 10 | 5.0 | 5.0 | 5.0 | - | 962.4 |

## Judge vs. Human Agreement

- Matched items: 20 (pooled ordinal scores: 60)

| Dimension | N | Exact Match % | Within-1 % | Bucket Match % |
|---|---:|---:|---:|---:|
| correctness | 20 | 60.0 | 70.0 | 70.0 |
| faithfulness | 20 | 85.0 | 95.0 | 95.0 |
| citation_accuracy | 20 | 95.0 | 95.0 | 95.0 |
| **pooled** | 60 | 80.0 | 86.67 | 86.67 |

- `override_obedience` exact match (N=4): 100.0%

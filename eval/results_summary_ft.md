# Evaluation Results Summary

## Overall by Mode

| Mode | N | Correctness | Faithfulness | Citation Acc | Override Obed | Mean Latency (ms) | Median Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| rag | 60 | 3.5 | 4.833 | 4.65 | 0.9 | 834.91 | 816.53 |
| zero_shot | 60 | 2.567 | 4.217 | 4.2 | 0.0 | 612.85 | 614.39 |

## Breakdown by Mode & Slice

| Mode | Slice | N | Correctness | Faithfulness | Citation Acc | Override Obed | Mean Latency (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| rag | static | 20 | 1.8 | 5.0 | 5.0 | - | 498.76 |
| rag | live | 20 | 3.9 | 4.55 | 3.95 | - | 1174.34 |
| rag | temporal_override | 10 | 4.6 | 4.9 | 5.0 | 0.9 | 1249.45 |
| rag | unanswerable | 10 | 5.0 | 5.0 | 5.0 | - | 413.8 |
| zero_shot | static | 20 | 1.7 | 4.65 | 4.8 | - | 735.49 |
| zero_shot | live | 20 | 2.1 | 3.6 | 3.4 | - | 683.6 |
| zero_shot | temporal_override | 10 | 2.8 | 3.8 | 3.8 | 0.0 | 478.89 |
| zero_shot | unanswerable | 10 | 5.0 | 5.0 | 5.0 | - | 360.05 |

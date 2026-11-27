# Evaluation Results Summary

## Overall by Mode

| Mode | N | Correctness | Faithfulness | Citation Acc | Override Obed | Mean Latency (ms) | Median Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| rag | 60 | 3.783 | 4.933 | 4.967 | 0.8 | 834.91 | 816.53 |
| zero_shot | 60 | 1.867 | 2.15 | 4.133 | 0.0 | 612.85 | 614.39 |

## Breakdown by Mode & Slice

| Mode | Slice | N | Correctness | Faithfulness | Citation Acc | Override Obed | Mean Latency (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| rag | static | 20 | 2.0 | 4.8 | 5.0 | - | 498.76 |
| rag | live | 20 | 4.45 | 5.0 | 4.9 | - | 1174.34 |
| rag | temporal_override | 10 | 4.8 | 5.0 | 5.0 | 0.8 | 1249.45 |
| rag | unanswerable | 10 | 5.0 | 5.0 | 5.0 | - | 413.8 |
| zero_shot | static | 20 | 1.6 | 1.25 | 3.6 | - | 735.49 |
| zero_shot | live | 20 | 1.2 | 1.4 | 3.8 | - | 683.6 |
| zero_shot | temporal_override | 10 | 1.0 | 3.0 | 5.0 | 0.0 | 478.89 |
| zero_shot | unanswerable | 10 | 4.6 | 4.6 | 5.0 | - | 360.05 |

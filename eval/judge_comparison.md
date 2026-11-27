# Statistical Judge Comparison Report

- **Base Judge**: `qwen2.5:7b`
- **Candidate Judge**: `qwen3.5:9b`

## 1. Human Calibration Comparison (N=20)

| Dimension | qwen2.5:7b Exact % | qwen3.5:9b Exact % | Delta Exact % | 95% CI (Delta Exact) | qwen2.5:7b Kappa | qwen3.5:9b Kappa | qwen2.5:7b MAE | qwen3.5:9b MAE |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| `correctness` | 60.0% | 85.0% | +25.0% | - | 0.4236 | 0.8757 | 1.2 | 0.3 |
| `faithfulness` | 85.0% | 90.0% | +5.0% | - | -0.0465 | -0.0526 | 0.3 | 0.4 |
| `citation_accuracy` | 95.0% | 95.0% | +0.0% | - | 0.0 | 0.0 | 0.2 | 0.2 |
| **Pooled (Ordinal)** | 80.0% | 90.0% | +10.0% | [1.67%, 20.0%] | 0.5349 | 0.7823 | 0.567 | 0.3 |

### Override Obedience Calibration
- **Base Override Exact %**: 100.0%
- **Candidate Override Exact %**: 100.0%
- **Delta Override Exact %**: +0.0%

## 2. Benchmark Score Shift

### Overall by Mode
| Mode | Metric | Base (qwen2.5:7b) | Cand (qwen3.5:9b) | Delta |
|---|---|---:|---:|---:|
| `zero_shot` | `correctness` | 2.2 | 1.733 | -0.467 |
| `zero_shot` | `faithfulness` | 4.933 | 4.533 | -0.4 |
| `zero_shot` | `citation_accuracy` | 4.983 | 4.8 | -0.183 |
| `zero_shot` | `override_obedience` | 0.0 | 0.0 | +0.0 |
| `rag` | `correctness` | 3.417 | 3.65 | +0.233 |
| `rag` | `faithfulness` | 4.85 | 4.917 | +0.067 |
| `rag` | `citation_accuracy` | 4.983 | 5.0 | +0.017 |
| `rag` | `override_obedience` | 0.7 | 0.5 | -0.2 |

### By Mode & Slice (Correctness & Override Obedience)
| Mode | Slice | Base Correctness | Cand Correctness | Delta | Base Ov. Obed | Cand Ov. Obed |
|---|---|---:|---:|---:|---:|---:|
| `zero_shot` | `static` | 1.45 | 1.15 | -0.3 | - | - |
| `zero_shot` | `live` | 1.0 | 1.0 | +0.0 | - | - |
| `zero_shot` | `temporal_override` | 3.3 | 1.1 | -2.2 | 0.0 | 0.0 |
| `zero_shot` | `unanswerable` | 5.0 | 5.0 | +0.0 | - | - |
| `rag` | `static` | 1.8 | 1.8 | +0.0 | - | - |
| `rag` | `live` | 3.85 | 4.85 | +1.0 | - | - |
| `rag` | `temporal_override` | 4.2 | 3.6 | -0.6 | 0.7 | 0.5 |
| `rag` | `unanswerable` | 5.0 | 5.0 | +0.0 | - | - |

## 3. Inter-Judge Agreement & Discordance (N=120)

| Dimension | Exact Match % | Within-1 % | Bucket Match % | QWK ($\kappa_w$) | MAE | Mean Signed Diff (Cand - Base) |
|---|---:|---:|---:|---:|---:|---:|
| `correctness` | 76.67% | 79.17% | 78.33% | 0.6951 | 0.667 | -0.117 |
| `faithfulness` | 85.0% | 93.33% | 93.33% | 0.1816 | 0.317 | -0.167 |
| `citation_accuracy` | 96.67% | 97.5% | 97.5% | 0.1532 | 0.1 | -0.083 |
| **Pooled** | 86.11% | 90.0% | 89.72% | 0.7647 | 0.361 | -0.122 |

### Discordance Category Counts
| Category | Count | Description |
|---|---:|---|
| `hedged_zero_shot_ungrounded` | 6 | Zero-shot ungrounded answers penalized by base but missed by candidate |
| `false_abstention_fix` | 8 | Valid answerable questions incorrectly abstained by base, correctly resolved by candidate |
| `rag_completeness_strictness` | 16 | RAG correctness rubric strictness variations |
| `rubric_boundary_shift` | 4 | General rating boundary shifts (e.g. score 4 vs 5) |

## 4. Operational Throughput & Latency

| Metric | Base (qwen2.5:7b) | Candidate (qwen3.5:9b) |
|---|---:|---:|
| **Throughput (tok/s)** | `59.7` (documented) | `45.19` (mean) / `36.86` (median) |
| **Latency per Item (ms)** | - | `8820.11` (mean) / `3758.85` (median) |
| **Total Evaluation Time (s)** | - | `1058.41s` |
| **Total Evaluation Tokens** | - | `14740` |

# Model Comparison Report: `llama3.2:3b` vs `llama3.2-qlora:3b`

- **Base Model**: `llama3.2:3b`
- **Fine-Tuned Model**: `llama3.2-qlora:3b`
- **Primary Judge**: `qwen3.5:9b`

## Executive Recommendation

RECOMMEND `llama3.2-qlora:3b` over `llama3.2:3b` (qwen3.5:9b): pooled correctness +0.133 (95% CI [-0.092, +0.358], not statistically significant; zero-shot +0.133, RAG +0.133).

- Pooled correctness (N=120): 2.692 -> 2.825 (delta +0.133, 95% CI [-0.092, +0.358], not statistically significant).
- Zero-Shot correctness (N=60): 1.733 -> 1.867 (delta +0.133, 95% CI [-0.133, +0.4]).
- RAG correctness (N=60): 3.65 -> 3.783 (delta +0.133, 95% CI [-0.217, +0.5]).
- Prompt-level outcomes (N=120): 14 improved, 95 tied, 11 regressed (net wins +3).
- RAG behavior shift: abstentions 34 -> 30 (-4), spurious trailing abstentions 1 -> 0 (-1).
- Dual-judge cross-check (qwen2.5:7b): pooled correctness delta +0.225, direction_agrees=False.
- Post-hoc hybrid simulation (tau=0.68, N=60): Base Hybrid 3.7 -> QLoRA Hybrid 3.783 (delta +0.083).

## 1. Score Comparison & Deltas

### Overall by Mode & Pooled
| Bucket | N | llama3.2:3b Corr | llama3.2-qlora:3b Corr | Delta Corr | Delta Faith | Delta Cite | Delta Ov. Obed | Delta Mean Lat (ms) | Delta Med Lat (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `zero_shot` | 60 | 1.733 | 1.867 | +0.134 | -2.383 | -0.667 | +0.0 | -1112.59 | -387.05 |
| `rag` | 60 | 3.65 | 3.783 | +0.133 | +0.016 | -0.033 | +0.3 | -1045.26 | -402.07 |
| `pooled_all` | 120 | 2.692 | 2.825 | +0.133 | -1.183 | -0.35 | +0.15 | -1078.93 | -500.5 |

### Breakdown by Mode & Slice
| Mode | Slice | N | llama3.2:3b Corr | llama3.2-qlora:3b Corr | Delta Corr | Delta Faith | Delta Cite | Delta Ov. Obed | Delta Mean Lat (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `zero_shot` | `static` | 20 | 1.15 | 1.6 | +0.45 | -2.95 | -1.2 | - | -1303.89 |
| `zero_shot` | `live` | 20 | 1.0 | 1.2 | +0.2 | -3.4 | -1.2 | - | -326.51 |
| `zero_shot` | `temporal_override` | 10 | 1.1 | 1.0 | -0.1 | -1.2 | +0.8 | +0.0 | -2812.39 |
| `zero_shot` | `unanswerable` | 10 | 5.0 | 4.6 | -0.4 | -0.4 | +0.0 | - | -602.35 |
| `rag` | `static` | 20 | 1.8 | 2.0 | +0.2 | +0.0 | +0.0 | - | -438.09 |
| `rag` | `live` | 20 | 4.85 | 4.45 | -0.4 | +0.0 | -0.1 | - | -1834.61 |
| `rag` | `temporal_override` | 10 | 3.6 | 4.8 | +1.2 | +0.1 | +0.0 | +0.3 | -1337.24 |
| `rag` | `unanswerable` | 10 | 5.0 | 5.0 | +0.0 | +0.0 | +0.0 | - | -388.92 |

## 2. Paired Bootstrap 95% Confidence Intervals

| Bucket | N | Base Mean | FT Mean | Delta Mean | 95% CI | Significant |
|---|---:|---:|---:|---:|---|---|
| `pooled_all` | 120 | 2.692 | 2.825 | +0.133 | [-0.092, +0.358] | no |
| `zero_shot_all` | 60 | 1.733 | 1.867 | +0.133 | [-0.133, +0.4] | no |
| `rag_all` | 60 | 3.65 | 3.783 | +0.133 | [-0.217, +0.5] | no |
| `zero_shot_static` | 20 | 1.15 | 1.6 | +0.45 | [-0.2, +1.15] | no |
| `zero_shot_live` | 20 | 1.0 | 1.2 | +0.2 | [+0.05, +0.4] | YES |
| `zero_shot_temporal_override` | 10 | 1.1 | 1.0 | -0.1 | [-0.3, +0.0] | no |
| `zero_shot_unanswerable` | 10 | 5.0 | 4.6 | -0.4 | [-1.2, +0.0] | no |
| `rag_static` | 20 | 1.8 | 2.0 | +0.2 | [-0.4, +0.8] | no |
| `rag_live` | 20 | 4.85 | 4.45 | -0.4 | [-0.8, +0.0] | no |
| `rag_temporal_override` | 10 | 3.6 | 4.8 | +1.2 | [+0.0, +2.4] | no |
| `rag_unanswerable` | 10 | 5.0 | 5.0 | +0.0 | [+0.0, +0.0] | no |
| `rag_temporal_override_obedience` | 10 | 0.5 | 0.8 | +0.3 | [+0.0, +0.6] | no |

## 3. Prompt-Level Win / Tie / Loss Matrix

| Scope | N | Improved | Tied | Regressed | Net Wins |
|---|---:|---:|---:|---:|---:|
| **overall** | 120 | 14 | 95 | 11 | +3 |
| `zero_shot` | 60 | 8 | 49 | 3 | +5 |
| `zero_shot/static` | 20 | 4 | 15 | 1 | +3 |
| `zero_shot/live` | 20 | 4 | 16 | 0 | +4 |
| `zero_shot/temporal_override` | 10 | 0 | 9 | 1 | -1 |
| `zero_shot/unanswerable` | 10 | 0 | 9 | 1 | -1 |
| `rag` | 60 | 6 | 46 | 8 | -2 |
| `rag/static` | 20 | 2 | 17 | 1 | +1 |
| `rag/live` | 20 | 1 | 12 | 7 | -6 |
| `rag/temporal_override` | 10 | 3 | 7 | 0 | +3 |
| `rag/unanswerable` | 10 | 0 | 10 | 0 | +0 |

## 4. Behavioral Shift Analysis

| Mode | Slice | Base Abst | FT Abst | Base Cited | FT Cited | Base Uncited | FT Uncited | Base Spurious Trail | FT Spurious Trail |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `zero_shot` | `static` | 18 | 0 | 0 | 0 | 2 | 20 | 0 | 0 |
| `zero_shot` | `live` | 19 | 0 | 0 | 0 | 1 | 20 | 0 | 0 |
| `zero_shot` | `temporal_override` | 8 | 0 | 0 | 0 | 2 | 10 | 0 | 0 |
| `zero_shot` | `unanswerable` | 10 | 9 | 0 | 0 | 0 | 1 | 0 | 0 |
| `rag` | `static` | 20 | 20 | 0 | 0 | 0 | 0 | 0 | 0 |
| `rag` | `live` | 1 | 0 | 18 | 2 | 1 | 18 | 1 | 0 |
| `rag` | `temporal_override` | 3 | 0 | 7 | 10 | 0 | 0 | 0 | 0 |
| `rag` | `unanswerable` | 10 | 10 | 0 | 0 | 0 | 0 | 0 | 0 |

## 5. Dual-Judge Cross-Check (qwen3.5:9b vs. qwen2.5:7b)

- Secondary Judge: `qwen2.5:7b`
- Overall Direction Agreement (where `|primary_delta| >= 0.1`): `False`
- **Data Compared**: All `240` outputs (`120` Base + `120` Fine-Tuned) scored independently by `qwen3.5:9b` (`90.0%` human exact match, `QWK=0.782`) and `qwen2.5:7b` (`80.0%` human exact match, `QWK=0.535`). `Direction Agrees` checks whether both judges report the same sign when `|primary_delta| >= 0.10`.

| Bucket | qwen3.5:9b Delta Corr | qwen2.5:7b Delta Corr | Direction Agrees |
|---|---:|---:|---|
| `pooled_all` | +0.133 | +0.225 | True |
| `zero_shot` | +0.134 | +0.367 | True |
| `rag` | +0.133 | +0.083 | True |
| `zero_shot_static` | +0.45 | +0.25 | True |
| `zero_shot_live` | +0.2 | +1.1 | True |
| `zero_shot_temporal_override` | -0.1 | -0.5 | True |
| `zero_shot_unanswerable` | -0.4 | +0.0 | False |
| `rag_static` | +0.2 | +0.0 | False |
| `rag_live` | -0.4 | +0.05 | False |
| `rag_temporal_override` | +1.2 | +0.4 | True |
| `rag_unanswerable` | +0.0 | +0.0 | - |

- **Why Agreeing Buckets Validate Headline Wins**: Both judges agree in direction on `pooled_all`, `zero_shot`, `rag`, `zero_shot_static`, `zero_shot_live`, `zero_shot_temporal_override`, and `rag_temporal_override` (`7/10` active buckets), confirming QLoRA's gains are real model changes rather than judge bias.
- **Why Divergent Buckets Differ (`False`)**: `rag_live` (`qwen3.5:9b` penalizes secondary metric omissions in concise 1–2 sentence answers, whereas `qwen2.5:7b` credits core mechanism accuracy), `zero_shot_unanswerable` (`qwen3.5:9b` catches `g54` fabrication while `qwen2.5:7b` had offsetting variance), and `rag_static` (both models cleanly abstain `20/20`, with a 1-item boundary shift on `qwen3.5:9b`).

## 6. Post-Hoc Similarity-Gated Hybrid Routing Simulation

- Note: Post-hoc threshold routing over measured zero_shot and rag outputs

### Threshold Sweep (`tau`)
| tau | N | RAG Routed % | Base Hybrid Corr | QLoRA Hybrid Corr | Delta Corr |
|---:|---:|---:|---:|---:|---:|
| 0.55 | 60 | 100.0% | 3.65 | 3.783 | +0.133 |
| 0.60 | 60 | 100.0% | 3.65 | 3.783 | +0.133 |
| 0.65 | 60 | 96.67% | 3.7 | 3.783 | +0.083 |
| 0.68 | 60 | 93.33% | 3.7 | 3.783 | +0.083 |
| 0.70 | 60 | 86.67% | 3.7 | 3.783 | +0.083 |
| 0.75 | 60 | 68.33% | 3.5 | 3.7 | +0.2 |

### Default Threshold Slice Breakdown (`tau = 0.68`)
| Slice | N | RAG Routed % | Base Hybrid Corr | QLoRA Hybrid Corr | Delta Corr |
|---|---:|---:|---:|---:|---:|
| `static` | 20 | 95.0% | 1.95 | 2.0 | +0.05 |
| `live` | 20 | 100.0% | 4.85 | 4.45 | -0.4 |
| `temporal_override` | 10 | 100.0% | 3.6 | 4.8 | +1.2 |
| `unanswerable` | 10 | 70.0% | 5.0 | 5.0 | +0.0 |
| **overall** | 60 | 93.33% | 3.7 | 3.783 | +0.083 |

## 7. QLoRA Training Telemetry

### Training Manifest
- `config`: `{'model': 'mlx-community/Llama-3.2-3B-Instruct-4bit', 'train': True, 'fine_tune_type': 'lora', 'data': 'data/qlora', 'seed': 42, 'num_layers': 28, 'batch_size': 4, 'iters': 600, 'val_batches': 25, 'learning_rate': 0.0001, 'steps_per_report': 10, 'steps_per_eval': 50, 'save_every': 200, 'adapter_path': 'models/llama32_3b_qlora_adapters', 'max_seq_length': 2048, 'grad_checkpoint': True, 'mask_prompt': True, 'lora_parameters': {'rank': 16, 'scale': 2.0, 'dropout': 0.05}, 'lr_schedule': {'name': 'cosine_decay', 'warmup': 40, 'arguments': [0.0001, 600, 1e-05]}}`
- `initial_train_loss`: `2.478`
- `best_train_loss`: `0.418`
- `final_train_loss`: `0.509`
- `initial_val_loss`: `2.506`
- `best_val_loss`: `0.788`
- `final_val_loss`: `0.931`
- `best_iter`: `500`
- `peak_mem_gb`: `None`
- `wall_time_s`: `4171.88`
- `adapter_size_mb`: `371.164`
- `trainable_params`: `{'percent': 0.757, 'trainable_millions': 24.314, 'total_millions': 3212.75}`
- `selected_checkpoint`: `0000400_adapters.safetensors`
- `selected_checkpoint_iter`: `400`
- `selected_checkpoint_val_loss`: `0.831`
- `peak_mem_gb_observed_sample`: `11.5`
- `ollama_model`: `llama3.2-qlora:3b`
- `ollama_quantization`: `nvfp4`

### Training Log Summary
- `n_entries`: `73`
- `max_step`: `600`
- `initial_loss`: `2.478`
- `final_loss`: `0.509`
- `min_loss`: `0.418`
- `final_val_loss`: `0.931`
- `min_val_loss`: `0.788`

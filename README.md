# arXiv Hybrid LLM: Date-Split Corpus, Golden Set & Baselines

Leakage-safe temporal RAG benchmark on arXiv `cs.CL` running locally on an **Apple Mac mini (M5 Pro, 48 GB)** via **Ollama** (`llama3.2:3b` base model + `qwen2.5:7b` baseline judge / `qwen3.5:9b` stronger MLX judge) and **FAISS + `BAAI/bge-small-en-v1.5`**.

---

## Start Here: Interactive 12-Page Experiment Book (`experiment_book.html`)

> [!IMPORTANT]
> **Best way to review this experiment and all results:** open **[`experiment_book.html`](experiment_book.html)**.
> It compiles the full methodology, multi-agent orchestration, slice definitions with real positive/negative failure transcripts, interactive Plotly charts, judge-vs-human audit tables, QLoRA fine-tuning results, and 3 QLoRA & RAG research continuations into a single self-contained, Tailwind + shadcn styled book.

### Ways to Open

1. **Instant online preview (no clone required):** [Open rendered book via htmlpreview.github.io](https://htmlpreview.github.io/?https://github.com/etai83/arxiv-qlora-rag/blob/main/experiment_book.html)
2. **Local (zero-server, opens directly in browser):**
   ```bash
   open experiment_book.html   # macOS (or double-click experiment_book.html)
   ```
3. **Regenerate from raw artifacts:**
   ```bash
   uv run python main.py book
   ```

### Book Structure (12 Pages + Clickable Agenda)

| Page | Section | What You Will Find |
| :--- | :--- | :--- |
| **Page 1** | **Prologue: Abstract & Clickable Agenda** | Executive KPIs, experimental design thesis, and direct chapter navigation |
| **Page 2** | **Ch. 1: Slice Definitions, Metrics & Agents** | Static/Live/Override/Unanswerable specs, 1–5 rubric + Override Obedience, 9-agent pipeline, and real `+`/`-` output examples per slice |
| **Page 3** | **Ch. 2: Date-Split Corpus & Cutoff T** | `2024-09-25` Llama-3.2 cutoff provenance, 300 static + 200 live `cs.CL` papers, zero-overlap check |
| **Page 4** | **Ch. 3: Vector Ingestion & Retriever** | `BAAI/bge-small-en-v1.5` 384-d embeddings, FAISS `IndexFlatIP`, `retrieve(query, k, after=...)` |
| **Page 5** | **Ch. 4: 60-Prompt Golden Benchmark** | Slice composition, temporal override pairs, unanswerable trap construction |
| **Page 6** | **Ch. 5: Evaluation Rubric & Calibration** | Cross-family judge schema and calibration rules for Qwen2.5-7B and Qwen3.5-9B |
| **Page 7** | **Ch. 6: Baselines & Judge Agreement** | Zero-shot vs RAG (`k=4`) scores, latency breakdown, 20-sample blind human agreement |
| **Page 8** | **Ch. 7: Failure Modes & QLoRA Outlook** | Citation leakage, override resistance, and QLoRA mitigation targets |
| **Page 9** | **Ch. 8: QLoRA Fine-Tuning (`llama3.2-qlora:3b` vs. `llama3.2:3b`)** | MLX 4-bit QLoRA training telemetry, head-to-head scores & bootstrap CIs, win/tie/loss matrix, behavioral shift, and post-hoc hybrid routing |
| **Page 10** | **Appendix A: Hardware & Sub-Agent Trace** | Mac mini M5 Pro runtime telemetry, Ollama throughput, and 9-agent execution trace |
| **Page 11** | **Appendix B: 3 Future Research Directions** | Temporal DPO/ORPO, Abstention-Calibrated QLoRA, and Cross-Family Judge Distillation |
| **Page 12** | **Appendix C: Stronger Judge (Qwen3.5-9B vs. Qwen2.5-7B)** | MLX `nvfp4` evaluation calibration, human agreement lift (`80.0% -> 90.0%`), and discordance analysis |

---

## Temporal Split & Cutoff Provenance

| Parameter | Value |
|---|---|
| Category | `cs.CL` (Computation and Language) |
| Base Model | `Llama-3.2-3B-Instruct` (`ollama: llama3.2:3b`) |
| Base Model Training Cutoff | **December 2023** (released `2024-09-25`) |
| Cutoff Source | [Meta Llama 3.2 Model Card](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| Chosen Split Cutoff **`T`** | **`2025-01-01`** |
| Static Corpus (`< T`) | **300** abstracts (`2020-05-22` to `2024-01-06`) -> `corpus/static.jsonl` |
| Live Corpus (`>= T`) | **200** abstracts (`2025-01-01` to `2026-07-01`) -> `corpus/live.jsonl` |
| ID Overlap Count | **0** (asserted in `src/fetch_corpus.py` & `tests/test_pipeline.py`) |
| Judge Model | `Qwen2.5-7B-Instruct` (`ollama: qwen2.5:7b`, separate model family) |
| Embeddings / Store | `BAAI/bge-small-en-v1.5` (384-d L2-normalized) + `faiss.IndexFlatIP` |

---

## Evaluation Results (`eval/results_summary.md`)

### Overall by Mode (`60` Golden Prompts)

| Mode | N | Correctness | Faithfulness | Citation Acc | Override Obed | Mean Latency (ms) | Median Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Zero-Shot (`llama3.2:3b`) | 60 | 2.200 | 4.933 | 4.983 | 0.0 | 1725.44 | 1001.44 |
| RAG `k=4` (`llama3.2:3b` + `bge-small-en-v1.5`) | 60 | **3.417** | 4.850 | 4.983 | **0.7** | 1880.17 | 1218.60 |

### Breakdown by Mode & Slice

| Mode | Slice | N | Correctness | Faithfulness | Citation Acc | Override Obed | Mean Latency (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| Zero-Shot | `static` | 20 | 1.45 | 4.90 | 5.00 | - | 2039.38 |
| Zero-Shot | `live` | 20 | 1.00 | 5.00 | 5.00 | - | 1010.11 |
| Zero-Shot | `temporal_override` | 10 | 3.30 | 4.80 | 4.90 | 0.0 | 3291.28 |
| Zero-Shot | `unanswerable` | 10 | 5.00 | 5.00 | 5.00 | - | 962.40 |
| RAG (`k=4`) | `static` | 20 | 1.80 | 5.00 | 5.00 | - | 936.85 |
| RAG (`k=4`) | `live` | 20 | **3.85** | 4.55 | 4.95 | - | 3008.95 |
| RAG (`k=4`) | `temporal_override` | 10 | **4.20** | 5.00 | 5.00 | **0.7** | 2586.69 |
| RAG (`k=4`) | `unanswerable` | 10 | **5.00** | 5.00 | 5.00 | - | 802.72 |

### Judge vs. Human Agreement (`20` Stratified Blind Items / `60` Ordinal Scores)

| Dimension | N | Exact Match % | Within-1 % | Bucket Match % |
|---|---:|---:|---:|---:|
| `correctness` | 20 | 60.0 | 70.0 | 70.0 |
| `faithfulness` | 20 | 85.0 | 95.0 | 95.0 |
| `citation_accuracy` | 20 | 95.0 | 95.0 | 95.0 |
| **Pooled** | **60** | **80.0** | **86.67** | **86.67** |

- **`override_obedience` exact match (`N=4`)**: **100.0%**

---

## QLoRA Fine-Tuning Experiment: `llama3.2-qlora:3b` vs. `llama3.2:3b` (Issue #2)

To test whether domain-specific pre-cutoff fine-tuning improves static `cs.CL` recall, RAG synthesis, and temporal override handling without leaking post-T knowledge, we trained a 4-bit MLX QLoRA adapter on `mlx-community/Llama-3.2-3B-Instruct-4bit` distilled strictly from pre-T `corpus/static.jsonl` (`< 2025-01-01`), fused it into Ollama as **`llama3.2-qlora:3b`** (`nvfp4`, MLX runner), and evaluated both generators head-to-head across all `120` runs (`60` zero-shot + `60` RAG) using **`qwen3.5:9b`** as primary judge and **`qwen2.5:7b`** as secondary cross-check.

> [!IMPORTANT]
> **Bottom-Line Verdict (Issue #2)**: **`llama3.2-qlora:3b` delivers modest pooled correctness gains (`2.692 -> 2.825`, `+0.133`, 95% CI `[-0.092, +0.358]` under `qwen3.5:9b`; `+0.225` under `qwen2.5:7b`) at `2.5×` faster pooled inference (`-1,078.93 ms` mean latency: `1,802.81 ms -> 723.88 ms`), driven by major breakthroughs in RAG `temporal_override` (`3.60 -> 4.80`, `+1.20`; override obedience `0.50 -> 0.80`; `10/10` cited, `0/10` abstentions) and Zero-Shot `static` (`1.15 -> 1.60`, `+0.45`; abstentions `18/20 -> 0/20`).** However, those gains come with clear calibration trade-offs: zero-shot overconfidence reduces `zs/unanswerable` (`5.00 -> 4.60`) and `zs` faithfulness (`-2.383`), and concise 1–2 sentence distillation drops minor secondary details in `rag/live` (`4.85 -> 4.45` under `qwen3.5:9b`).

### Training & Leakage-Audit Summary

| Parameter / Audit Check | Value |
|---|---|
| Base Weights & Framework | `mlx-community/Llama-3.2-3B-Instruct-4bit`, `mlx_lm 0.32.2` |
| LoRA Architecture | `rank=16`, `scale=2.0`, `dropout=0.05`, `num_layers=28`, `max_seq_length=2048` |
| Trainable Parameters | **`0.757%`** (`24.314M / 3212.750M`), `batch_size=4`, `iters=600`, `lr=1e-4` (`cosine_decay`, warmup `40`, min `1e-5`) |
| SFT Dataset (`data/qlora/`) | **`1,060` train / `120` valid** (`1,180` total): `570/30` ZS QA, `100/20` ZS abstention, `260/40` RAG synthesis, `130/30` RAG abstention |
| Temporal Leakage Audit | Distilled strictly from `corpus/static.jsonl` (`300` papers, `2020-05-22` to `2024-01-06`): **`0`** live paper IDs, **`0`** golden prompts/non-abstention refs |
| Training Telemetry | `train_loss 2.478 -> 0.509` (best `0.418`), `val_loss 2.506 -> 0.831` at selected **`iter 400`** checkpoint (`best_val_loss 0.788` at `iter 500`), wall time `4,171.88 s`, adapter `371.164 MB`, peak mem `11.5 GB` |
| Export & Ollama Deployment | Fused via `mlx_lm fuse --dequantize`, registered as `llama3.2-qlora:3b` (`nvfp4`, native `MLX` runner, `2.80 GB`) |

### Head-to-Head Mode & Slice Comparison (`llama3.2:3b` vs. `llama3.2-qlora:3b`)

Primary judge: **`qwen3.5:9b`** (with **`qwen2.5:7b`** cross-check deltas and 2,000-resample paired bootstrap 95% CIs on `correctness`).

| Scope | N | `llama3.2:3b` Corr | `llama3.2-qlora:3b` Corr | Delta (`qwen3.5:9b`) | 95% CI | `qwen2.5:7b` Delta | Delta Faith | Delta Cite | Delta Ov. Obed | Delta Mean Lat (ms) |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| **Pooled All** | **120** | 2.692 | **2.825** | **+0.133** | `[-0.092, +0.358]` | **+0.225** | -1.183 | -0.350 | +0.15 | **-1,078.93** |
| **Zero-Shot (All)** | **60** | 1.733 | **1.867** | **+0.133** | `[-0.133, +0.400]` | **+0.367** | -2.383 | -0.667 | +0.00 | **-1,112.59** |
| **RAG `k=4` (All)** | **60** | 3.650 | **3.783** | **+0.133** | `[-0.217, +0.500]` | **+0.083** | **+0.016** | -0.033 | **+0.30** | **-1,045.26** |
| `zero_shot / static` | 20 | 1.15 | **1.60** | **+0.45** | `[-0.20, +1.15]` | +0.25 | -2.95 | -1.20 | - | -1,303.89 |
| `zero_shot / live` | 20 | 1.00 | **1.20** | **+0.20** (sig) | `[+0.05, +0.40]` | +1.10 | -3.40 | -1.20 | - | -326.51 |
| `zero_shot / temporal_override` | 10 | **1.10** | 1.00 | -0.10 | `[-0.30, +0.00]` | -0.50 | -1.20 | +0.80 | 0.00 | -2,812.39 |
| `zero_shot / unanswerable` | 10 | **5.00** | 4.60 | -0.40 | `[-1.20, +0.00]` | 0.00 | -0.40 | 0.00 | - | -602.35 |
| `rag / static` | 20 | 1.80 | **2.00** | **+0.20** | `[-0.40, +0.80]` | 0.00 | 0.00 | 0.00 | - | -438.09 |
| `rag / live` | 20 | **4.85** | 4.45 | -0.40 | `[-0.80, +0.00]` | +0.05 | 0.00 | -0.10 | - | -1,834.61 |
| `rag / temporal_override` | 10 | 3.60 | **4.80** | **+1.20** | `[+0.00, +2.40]` | **+0.40** | **+0.10** | 0.00 | **+0.30** (`0.50 -> 0.80`) | -1,337.24 |
| `rag / unanswerable` | 10 | **5.00** | **5.00** | 0.00 | `[+0.00, +0.00]` | 0.00 | 0.00 | 0.00 | - | -388.92 |

### Win / Tie / Loss, Behavioral Shift & Post-Hoc Hybrid Routing

- **Prompt-Level Outcomes (`N=120`)**: **`14` improved / `95` tied / `11` regressed** (**`+3` net wins**):
  - **Zero-Shot (`60`)**: `8` improved / `49` tied / `3` regressed (`+5` net; `static` `+3`, `live` `+4`, `temporal_override` `-1`, `unanswerable` `-1`).
  - **RAG (`60`)**: `6` improved / `46` tied / `8` regressed (`-2` net; `temporal_override` `+3`, `static` `+1`, `unanswerable` `0`, `live` `-6`).
- **Behavioral Shift**:
  - **Zero-Shot**: Abstentions dropped `55/60 -> 9/60` (`static` `18 -> 0`, `live` `19 -> 0`, `temporal_override` `8 -> 0`, `unanswerable` `10 -> 9`).
  - **RAG (`k=4`)**: Abstentions dropped `34/60 -> 30/60` (`temporal_override` `3 -> 0` with `10/10` cited, `live` `1 -> 0`), `spurious_trailing_abstention` dropped `1 -> 0`, while `rag/static` held `20/20` clean abstentions and `rag/unanswerable` held `10/10` clean abstentions. Explicit `[arxiv_id]` brackets in `rag/live` dropped `18/20 -> 2/20` (`18/20` uncited, though `qwen3.5:9b` still scored `citation_accuracy 4.90`).
- **Post-Hoc Similarity-Gated Hybrid Routing (`N=60`)**:
  - At **`tau=0.68`** (`93.33%` routed to RAG, including `95.0%` of `static` and `70.0%` of `unanswerable`): Base Hybrid **`3.700`** -> QLoRA Hybrid **`3.783`** (**`+0.083`**).
  - At **`tau=0.75`** (`68.33%` routed to RAG): Base Hybrid **`3.500`** -> QLoRA Hybrid **`3.700`** (**`+0.200`**).

---

## Stronger Judge Experiment: Qwen3.5-9B (MLX) vs. Qwen2.5-7B (Issue #1)

To test evaluation robustness, we benchmarked a stronger candidate LLM-as-a-Judge: **`qwen3.5:9b`** (`9.7B` parameters, `nvfp4`, native Apple Silicon `MLX` runner on Ollama with `think=False`, `temperature=0.0`, `num_ctx=4096`) against the baseline **`qwen2.5:7b`** (`7.6B`, `Q4_K_M` GGUF).

> [!IMPORTANT]
> **Bottom-Line Verdict**: **YES — Use `qwen3.5:9b` (MLX `nvfp4`, `think=False`) over `qwen2.5:7b` (GGUF `Q4_K_M`)** as the default benchmark judge on Apple Silicon (48 GB M5 Pro). It improves pooled exact agreement vs. humans by **+10.0 pp** (`80.0% -> 90.0%`), correctness agreement by **+25.0 pp** (`60.0% -> 85.0%`), and severity-weighted agreement (Quadratic Weighted Kappa $\kappa_w$) from **0.5349 (moderate) to 0.7823 (strong)**, **AND** runs **2.06× faster per item** (`3,758.85 ms` vs `7,754.56 ms` median, `-3,995.71 ms` saved per item; `36.86` vs `25.01` median `tok/s`) even while generating nearly `2×` longer rationales (`119.5` vs `63.0` median tokens/item, `+56.5` tokens). Keep `qwen2.5:7b` only on RAM-constrained machines (`<= 16 GB`) where `4.68 GB` fits more comfortably than `7.64 GB`.

### How to Read Pooled Exact % & Weighted Kappa ($\kappa_w$)
- **Exact Score Match vs. Human (Pooled Exact %)**: Each of the 20 blind audit items is graded on 3 ordinal 1–5 dimensions (`correctness`, `faithfulness`, `citation_accuracy`), giving `20 × 3 = 60` total human ratings. `90.0%` means `54 / 60` exact matches (`80.0%` = `48 / 60`).
- **Severity-Weighted Agreement / Quadratic Weighted Kappa ($\kappa_w$)**: On `-1` to `+1` (`0` = random chance, `1` = perfect human match), QWK adjusts for chance agreement (since `5` is common) and penalizes large errors quadratically — mistaking a `1` for a `5` hurts **$16\times$ more** than `4 vs 5`. `0.5349` = moderate agreement; `0.7823` = strong agreement (`0.8757` on `correctness`).

### Human Calibration Comparison (`N=20` Stratified Blind Items / `60` Ordinal Pairs)
| Dimension | Baseline (`qwen2.5:7b`) Exact % | Candidate (`qwen3.5:9b`) Exact % | Delta | Baseline $\kappa_w$ | Candidate $\kappa_w$ | Candidate MAE |
|---|---:|---:|---:|---:|---:|---:|
| `correctness` | 60.0% | **85.0%** | **+25.0 pp** | 0.4236 | **0.8757** | 0.3 |
| `faithfulness` | 85.0% | **90.0%** | **+5.0 pp** | -0.0465 | -0.0526 | 0.4 |
| `citation_accuracy` | 95.0% | **95.0%** | **+0.0 pp** | 0.0 | 0.0 | 0.2 |
| **Pooled (`N=60`)** | **80.0%** (`95% CI [68.33%, 90.0%]`) | **90.0%** (`95% CI [81.67%, 96.67%]`) | **+10.0 pp** (`95% CI [1.67%, 20.0%]`) | **0.5349** | **0.7823** | **0.300** |

*Note: Override exact match (`N=4`) remained **100.0%** for both judges.*

### Benchmark Score Shift (`N=120` Frozen `llama3.2:3b` Outputs)
- **Zero-Shot**: Overall correctness `2.200 -> 1.733` (`-0.467`), faithfulness `4.933 -> 4.533` (`-0.400`), citation accuracy `4.983 -> 4.800` (`-0.183`). Slice shifts: `static` `1.45 -> 1.15`, `live` `1.00 -> 1.00`, `temporal_override` `3.30 -> 1.10`, `unanswerable` `5.00 -> 5.00`.
- **RAG (`k=4`)**: Overall correctness `3.417 -> 3.650` (`+0.233`), faithfulness `4.850 -> 4.917` (`+0.067`), citation accuracy `4.983 -> 5.000` (`+0.017`), override obedience `0.700 -> 0.500` (`-0.200`). Slice shifts: `static` `1.80 -> 1.80`, `live` `3.85 -> 4.85` (`+1.00`), `temporal_override` `4.20 -> 3.60` (`-0.60`), `unanswerable` `5.00 -> 5.00`.

### Inter-Judge Agreement & Discordance (`N=120`, `360` Ordinal Pairs)
- **Pooled Metrics**: Exact Match: **`86.11%`**, Within-1: **`90.0%`**, Bucket Match: **`89.72%`**, Pooled $\kappa_w$: **`0.7647`**, Mean Signed Diff: **`-0.122`**.
- **Discordance (`34/120` items)**: `rag_completeness_strictness` (`16`), `false_abstention_fix` (`8`), `hedged_zero_shot_ungrounded` (`6`), `rubric_boundary_shift` (`4`).

### Speed, Median Latency & Token Volume Comparison (`N=120` Sequential Warm Runs)
| Metric | Baseline (`qwen2.5:7b` GGUF `Q4_K_M`) | Candidate (`qwen3.5:9b` MLX `nvfp4`) | Delta / Comparison |
|---|---:|---:|---:|
| **Median Throughput (`tok/s`)** | 25.01 | **36.86** | **+11.85 tok/s** (`1.47×`) |
| **Mean Throughput (`tok/s`)** | 23.44 | **45.19** | **+21.75 tok/s** (`1.93×`) |
| **Median Latency / Item** | 7,754.56 ms (`7.75s`) | **3,758.85 ms** (`3.76s`) | **-3,995.71 ms** (`2.06×` faster) |
| **Mean Latency / Item** | 16,442.07 ms (`1,973.05s` total) | **8,820.11 ms** (`1,058.41s` total) | **-7,621.96 ms** (`1.86×` faster) |
| **Median Output Tokens / Item** | 63.0 tokens | **119.5 tokens** | **+56.5 tokens** (`1.90×` longer rationale) |
| **Mean Output Tokens / Item** | 67.92 tokens (`8,150` total) | **122.83 tokens** (`14,740` total) | **+54.91 tokens** |
| **Model Footprint / Size** | **4.68 GB** | 7.64 GB | +2.96 GB |

---

## Project Layout

```text
corpus/
  static.jsonl                    # 300 pre-T cs.CL abstracts (< 2025-01-01)
  live.jsonl                      # 200 post-T cs.CL abstracts (>= 2025-01-01)
  split_metadata.json             # Cutoff provenance + date/overlap checks
  env_manifest.json               # Mac mini + Ollama model digests
data/index/
  live.faiss                      # 384-d FAISS IndexFlatIP over live.jsonl
  live_meta.json                  # Chunk metadata including published_date
data/qlora/
  train.jsonl                     # 1,060 leakage-free QLoRA SFT chat examples
  valid.jsonl                     # 120 QLoRA validation examples
  dataset_manifest.json           # Split counts, date bounds, SHA-256 checksums
  lora_config.yaml                # mlx_lm LoRA config (rank=16, scale=2.0, 28 layers)
  adapter_config.json             # MLX LoRA adapter configuration snapshot
  train_manifest.json             # Training telemetry, val loss curve & iter 400 selection
  training_log.jsonl              # Parsed step-by-step train/val loss log
eval/
  rubric.md                       # 1-5 evaluation rubric + abstention calibration rules
  golden.jsonl                    # 60 prompts (20 static, 20 live, 10 override, 10 unanswerable)
  slices/*.jsonl                  # Per-slice golden sets
  outputs_zero_shot.jsonl         # Base llama3.2:3b zero-shot outputs
  outputs_rag.jsonl               # Base llama3.2:3b RAG (k=4) outputs
  outputs_zero_shot_ft.jsonl      # Fine-tuned llama3.2-qlora:3b zero-shot outputs
  outputs_rag_ft.jsonl            # Fine-tuned llama3.2-qlora:3b RAG (k=4) outputs
  blind_sample_20.jsonl           # Stratified 20-item blind audit sample
  human_scores_20.jsonl           # Blinded human grades
  judge_scores.jsonl              # Base model scored by Qwen2.5-7B (120 items)
  judge_scores_qwen35_9b.jsonl    # Base model scored by Qwen3.5-9B (120 items)
  judge_scores_ft_qwen25_7b.jsonl # Fine-tuned model scored by Qwen2.5-7B (120 items)
  judge_scores_ft_qwen35_9b.jsonl # Fine-tuned model scored by Qwen3.5-9B (120 items)
  results_summary.{json,md}
  results_summary_qwen35_9b.{json,md}
  results_summary_ft.{json,md}
  results_summary_ft_qwen35_9b.{json,md}
  judge_comparison.{json,md}      # Judge vs judge agreement & human calibration report
  model_comparison.{json,md}      # Base vs QLoRA model comparison, CIs & hybrid routing
src/
  fetch_corpus.py                 # Rate-limited arXiv API date-split fetcher
  ingest.py                       # bge-small-en-v1.5 embedding + FAISS builder
  retriever.py                    # retrieve(query, k, after=None) with date filtering
  baselines.py                    # Sequential zero-shot + RAG (k=4) runner
  judge.py                        # Structured LLM-as-a-Judge runner
  compare_judges.py               # Qwen3.5-9B MLX judge vs Qwen2.5-7B comparator
  prepare_qlora_data.py           # Pre-T static distillation & QLoRA split builder
  train_qlora.py                  # MLX QLoRA trainer, adapter fuser & Modelfile builder
  compare_models.py               # Base vs QLoRA comparator, bootstrap CIs & hybrid sim
  metrics.py                      # Summary tables + human-agreement evaluator
  build_book.py                   # Generates interactive Tailwind/shadcn/Plotly experiment book (12 pages)
tests/
  test_pipeline.py                # Corpus, retrieval, golden-set, blind-sample, agreement tests
  test_compare_models.py          # QLoRA comparison, bootstrap CI & hybrid routing tests
main.py                           # Unified CLI entrypoint
experiment_book.html              # Self-contained 12-page interactive book (Chapters 1-8 + Appendices A-C)
notes.md                          # Error analysis & empirical QLoRA results
```

## Quickstart (on Mac mini with `uv` + Ollama)

```bash
uv sync
uv run python main.py ingest
uv run python main.py retrieve --query "continual learning ASR" --k 4 --after 2025-07-01
uv run python main.py baselines
uv run python main.py judge --model qwen2.5:7b
uv run python main.py judge --model qwen3.5:9b
uv run python main.py compare-judges
uv run python main.py prepare-qlora
uv run python main.py train-qlora --fuse --build-modelfile
uv run python main.py baselines --model llama3.2-qlora:3b --zs-out eval/outputs_zero_shot_ft.jsonl --rag-out eval/outputs_rag_ft.jsonl
uv run python main.py judge --model qwen3.5:9b --zs-path eval/outputs_zero_shot_ft.jsonl --rag-path eval/outputs_rag_ft.jsonl --out-path eval/judge_scores_ft_qwen35_9b.jsonl
uv run python main.py compare-models
uv run python main.py metrics
uv run python main.py book
uv run pytest -v
```

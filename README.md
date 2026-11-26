# arXiv Hybrid LLM — Evening 1: Date-Split Corpus, Golden Set & Baselines

Leakage-safe temporal RAG benchmark on arXiv `cs.CL` running locally on an **Apple Mac mini (M5 Pro, 48 GB)** via **Ollama** (`llama3.2:3b` base model + `qwen2.5:7b` LLM-as-a-Judge) and **FAISS + `BAAI/bge-small-en-v1.5`**.

---

## Start Here: Interactive 10-Page Experiment Book (`experiment_book.html`)

> [!IMPORTANT]
> **Best way to review this experiment and all results:** open **[`experiment_book.html`](experiment_book.html)**.
> It compiles the full methodology, multi-agent orchestration, slice definitions with real positive/negative failure transcripts, all 18 interactive Plotly charts, judge-vs-human audit tables, and 3 QLoRA Evening 2+ research continuations into a single self-contained, Tailwind + shadcn styled book.

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

### Book Structure (10 Pages + Clickable Agenda)

| Page | Section | What You Will Find |
| :--- | :--- | :--- |
| **Page 1** | **Prologue: Abstract & Clickable Agenda** | Executive KPIs, experimental design thesis, and direct chapter navigation |
| **Page 2** | **Ch. 1: Slice Definitions, Metrics & Agents** | Static/Live/Override/Unanswerable specs, 1–5 rubric + Override Obedience, 9-agent pipeline, and real `+`/`-` output examples per slice |
| **Page 3** | **Ch. 2: Date-Split Corpus & Cutoff T** | `2024-09-25` Llama-3.2 cutoff provenance, 300 static + 200 live `cs.CL` papers, zero-overlap check |
| **Page 4** | **Ch. 3: Vector Ingestion & Retriever** | `BAAI/bge-small-en-v1.5` 384-d embeddings, FAISS `IndexFlatIP`, `retrieve(query, k, after=...)` |
| **Page 5** | **Ch. 4: 60-Prompt Golden Benchmark** | Slice composition, temporal override pairs, unanswerable trap construction |
| **Page 6** | **Ch. 5: Evaluation Rubric & Calibration** | Cross-family `qwen2.5:7b` judge schema and calibration rules |
| **Page 7** | **Ch. 6: Baselines & Judge Agreement** | Zero-shot vs RAG (`k=4`) scores, latency breakdown, 20-sample blind human agreement (`95.0%`) |
| **Page 8** | **Ch. 7: Failure Modes & Evening 2 QLoRA** | Citation leakage, override resistance, and QLoRA mitigation targets |
| **Page 9** | **Appendix A: Hardware & Sub-Agent Trace** | Mac mini M5 Pro runtime telemetry, Ollama throughput, and 9-agent execution trace |
| **Page 10** | **Appendix B: 3 Future Research Directions** | Temporal DPO/ORPO, Abstention-Calibrated QLoRA, and Cross-Family Judge Distillation |

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

## Project Layout

```text
corpus/
  static.jsonl          # 300 pre-T cs.CL abstracts (< 2025-01-01)
  live.jsonl            # 200 post-T cs.CL abstracts (>= 2025-01-01)
  split_metadata.json   # Cutoff provenance + date/overlap checks
  env_manifest.json     # Mac mini + Ollama model digests
data/index/
  live.faiss            # 384-d FAISS IndexFlatIP over live.jsonl
  live_meta.json        # Chunk metadata including published_date
eval/
  rubric.md             # 1-5 evaluation rubric + abstention calibration rules
  golden.jsonl          # 60 prompts (20 static, 20 live, 10 override, 10 unanswerable)
  slices/*.jsonl        # Per-slice golden sets
  outputs_zero_shot.jsonl
  outputs_rag.jsonl
  blind_sample_20.jsonl # Stratified 20-item blind audit sample
  human_scores_20.jsonl # Blinded human grades
  judge_scores.jsonl    # Calibrated Qwen2.5-7B scores (120 items)
  results_summary.{json,md}
src/
  fetch_corpus.py       # Rate-limited arXiv API date-split fetcher
  ingest.py             # bge-small-en-v1.5 embedding + FAISS builder
  retriever.py          # retrieve(query, k, after=None) with date filtering
  baselines.py          # Sequential zero-shot + RAG (k=4) runner
  judge.py              # Structured Qwen2.5-7B judge
  metrics.py            # Summary tables + human-agreement evaluator
  build_book.py         # Generates interactive Tailwind/shadcn/Plotly experiment book
tests/
  test_pipeline.py      # Corpus, retrieval, golden-set, blind-sample, agreement tests
main.py                 # Unified CLI entrypoint
experiment_book.html    # Self-contained 10-page interactive book (Chapters 1-7 + Appendices A-B)
notes.md                # Error analysis & Evening 2 QLoRA expectations
```

## Quickstart (on Mac mini with `uv` + Ollama)

```bash
uv sync
uv run python main.py ingest
uv run python main.py retrieve --query "continual learning ASR" --k 4 --after 2025-07-01
uv run python main.py baselines
uv run python main.py judge
uv run python main.py metrics
uv run python main.py book
uv run pytest -v
```

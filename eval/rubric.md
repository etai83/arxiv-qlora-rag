# Evaluation Rubric — arXiv QLoRA & Temporal RAG Benchmark

## 1. Experimental Cutoffs & Models
- **Temporal Cutoff (`T`)**: `2025-01-01`
  - **Pre-T (`static`) corpus**: `corpus/static.jsonl` — papers published strictly before `2025-01-01` (`published_date < "2025-01-01"`).
  - **Post-T (`live`) corpus**: `corpus/live.jsonl` — papers published on or after `2025-01-01` (`published_date >= "2025-01-01"`).
- **Generator / Base & Fine-Tuned Models**:
  - Baseline Generator: `Llama-3.2-3B-Instruct` (`llama3.2:3b` via Ollama, GGUF `Q4_K_M`), knowledge cutoff **December 2023** per the Meta Llama 3.2 Model Card (well prior to `T = 2025-01-01`).
  - Fine-Tuned Candidate: `llama3.2-qlora:3b` (`mlx-community/Llama-3.2-3B-Instruct-4bit` QLoRA fine-tuned via `mlx_lm 0.32.2` strictly on pre-T `corpus/static.jsonl` distilled splits, `iter 400` checkpoint fused with `--dequantize` and quantized to `nvfp4` via Ollama's native Apple Silicon `MLX` runner).
- **Judge Models**:
  - Baseline Judge: `Qwen2.5-7B-Instruct` (`qwen2.5:7b` via Ollama), GGUF `Q4_K_M` quantization, structured JSON verdict with `temperature = 0.0` and `num_ctx = 4096`.
  - Stronger Judge: `Qwen3.5-9B` (`qwen3.5:9b` via Ollama), `9.7B` parameters, native Apple Silicon `MLX` runner (`nvfp4`), configured with `think=False`, `temperature = 0.0`, and `num_ctx = 4096`.
- **Embedding Model**: `BAAI/bge-small-en-v1.5` (L2-normalized embeddings, inner-product / cosine similarity via FAISS `IndexFlatIP`, `k = 4`).

---

## 2. Evaluation Slices (`eval/golden.jsonl`, 60 items)
| Slice | Count | Purpose | Expected Behavior |
|---|---:|---|---|
| `static` | 20 | Pre-cutoff (`< 2025-01-01`) cs.CL facts | Answerable via parametric memory in zero-shot; in RAG over `live.jsonl` only, may require `"Not in the sources."` unless covered by a post-T paper. |
| `live` | 20 | Post-cutoff (`>= 2025-01-01`) cs.CL papers | Unseen by base model; RAG must retrieve from `live.jsonl`, answer accurately, and cite `[arxiv_id]`. |
| `temporal_override` | 10 | Post-T findings superseding/updating pre-T claims | Must adopt/report the newer post-T finding/method (`newer_paper_id`) as the updated/current conclusion rather than reverting to pre-T priors or abstaining. |
| `unanswerable` | 10 | False-premise or absent questions not in corpus | Must cleanly abstain with `"Not in the sources."` without fabricating facts or citations. |

---

## 3. Scoring Dimensions & Exact Anchors

### 3.1 `correctness` (Integer `1–5`, slice-aware)
Measures factual accuracy relative to `reference_answer` (and slice expectations):
- **`5`**: Fully accurate core claim, method, or result relative to `reference_answer` (or, for `unanswerable`, clean abstention `"Not in the sources."` without fabricated claims).
- **`4`**: Accurate core claim with minor secondary omission or slight imprecision.
- **`3`**: Partially right, overly vague, or covers only half of a multi-part answer.
- **`2`**: Mostly wrong or misses core finding, with only minor tangential overlap.
- **`1`**: Completely wrong, contradicts `reference_answer`, fabricates an answer on `unanswerable`, OR falsely abstains (`"Not in the sources."`) on an answerable `static`, `live`, or `temporal_override` question.

### 3.2 `faithfulness` (Integer `1–5`)
Measures groundedness strictly in `retrieved_context`:
- **`5`**: Every factual claim is directly supported by `retrieved_context`, OR valid context-bound abstention (`"Not in the sources."`) when `retrieved_context` lacks the answer (including when `retrieved_context` is `NONE` / empty).
- **`4`**: Core claims supported by `retrieved_context`, with minor harmless connective or phrasing synthesis.
- **`3`**: Mix of context-supported claims and unsupported or parametric claims.
- **`2`**: Mostly unsupported by `retrieved_context`.
- **`1`**: Completely ungrounded in `retrieved_context` — including asserting factual claims in zero-shot mode when `retrieved_context` is `NONE` / empty, or inventing facts not present in `retrieved_context`.

### 3.3 `citation_accuracy` (Integer `1–5`)
Measures whether cited `[arxiv_id]` tags exist in `retrieved_context` AND genuinely support the claims made:
- **`5`**: Every key factual claim cites a valid `[arxiv_id]` from `retrieved_context` that genuinely supports that claim, OR clean abstention (`"Not in the sources."`) with zero fabricated citations when the answer is absent from `retrieved_context`.
- **`4`**: Supporting `[arxiv_id]` cited accurately, with minor uncited secondary detail or one extra mildly-related real chunk ID from `retrieved_context`.
- **`3`**: Cites a real `[arxiv_id]` from `retrieved_context` that only partially or weakly supports the claim.
- **`2`**: Cites a real `[arxiv_id]` from `retrieved_context` that does NOT support the claim.
- **`1`**: States factual claims with ZERO citations, OR cites a hallucinated ID not present in `retrieved_context`, OR abstains while citing an unrelated paper as support.

### 3.4 `override_obedience` (`0`, `1`, or `null`)
Binary obedience score applicable **only** to `slice == "temporal_override"` (`null` on `static`, `live`, and `unanswerable`):
- **`1`**: The answer adopts/reports the newer post-T finding or method (`newer_paper_id` / post-T reference) as the current best or updated conclusion rather than relying solely on the older pre-T approach or abstaining.
- **`0`**: Otherwise (e.g., answers using pre-T priors, denies the newer finding, hedges without adopting the post-T update, or abstains).
- **`null`**: Mandatory for all non-`temporal_override` slices (`static`, `live`, `unanswerable`).

### 3.5 Abstention Decision Rules (Calibration Clarifications)
Any response that refuses or declines to answer without making substantive factual claims (either `"Not in the sources."` or natural-language zero-shot refusals like `"I don't have information..."` / `"My knowledge cutoff is..."`) is a **clean abstention** and MUST be scored deterministically. In contrast, a **hedged refusal + substantive overview** (e.g., `"I don't have the exact numbers, however X works by..."`) is NOT a clean abstention — it makes factual claims and must be scored under Sections 3.1–3.4 (receiving `faithfulness = 1`, `citation_accuracy = 1` in zero-shot when `retrieved_context` is `NONE`):
1. **`correctness`**:
   - If `slice == "unanswerable"`: clean abstention = **`5`** (never penalize natural-language refusal phrasing when no answer claims are fabricated).
   - If `slice in {"static", "live", "temporal_override"}`: clean abstention = **`1`** (false abstention on an answerable question — never `3`).
   - Note: if the model partially answers and then hedges or appends `"Not in the sources."` despite useful facts, score `2–4` based on how much of `reference_answer` was accurately covered.
2. **`faithfulness`**:
   - Clean abstention with zero unsupported claims = **`5`** in **both** zero-shot (`retrieved_context == NONE`) and RAG modes (never `1` — abstaining is 100% faithful to having no supporting context).
   - Only score **`1`** when the answer **asserts factual claims** without support in `retrieved_context` (including zero-shot factual answers when `retrieved_context` is `NONE`).
3. **`citation_accuracy`**:
   - Clean abstention with zero citations (`extracted_cited_ids == []`) = **`5`** (never `1` — abstaining makes no factual claims requiring citation).
   - Only score **`1`** when the answer **asserts factual claims** with zero citations, cites a hallucinated ID not in `valid_context_ids`, or abstains while citing an unrelated paper as support.
4. **`override_obedience`** (`temporal_override` only): clean abstention = **`0`**.

---

## 4. Agreement & Pooling Protocol (Human Audit vs. Judge)
- **Blind Sample**: `eval/blind_sample_20.jsonl` — 20 items stratified across modes and slices (`10` zero-shot + `10` RAG: `3 static`, `3 live`, `2 temporal_override`, `2 unanswerable` per mode), stripped of all judge outputs.
- **Agreement Metrics** (reported per dimension and pooled across the `20 × 3 = 60` ordinal ratings):
  - `exact_match_pct`: `%` where `judge == human`
  - `within_1_pct`: `%` where `abs(judge - human) <= 1`
  - `bucket_match_pct`: `%` sharing coarse quality bucket (`low = {1, 2}`, `mid = {3}`, `high = {4, 5}`)
  - `override_exact_pct`: `%` where `judge.override_obedience == human.override_obedience` on `temporal_override` items.

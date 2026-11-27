# Error Analysis & QLoRA Fine-Tuning Outlook

## Setup & Temporal Integrity
We chose **`T = 2025-01-01`** for arXiv `cs.CL`, strictly after **`Llama-3.2-3B-Instruct`** (`llama3.2:3b`)'s published pretraining cutoff of **December 2023** (and September 25, 2024 release date, per the [Meta Llama 3.2 Model Card](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)). `corpus/static.jsonl` contains **300** pre-T papers (`2020-05-22` to `2024-01-06`) and `corpus/live.jsonl` contains **200** post-T papers (`2025-01-01` to `2026-07-01`), with **0 overlapping arXiv IDs**.

## Rubric Calibration (48.3% -> 80.0% Agreement)
On the first pass, `qwen2.5:7b` achieved only **48.33% exact** (**55.0% within-1**) agreement against 20 blinded human grades. Inspecting rationales revealed two systematic failure modes: (1) emitting numeric scores *before* `rationale` caused the judge to output `faithfulness=1, citation_accuracy=1` on clean abstentions even when its own rationale praised them, and (2) `llama3.2:3b` zero-shot frequently used natural-language refusals (`"I don't have information..."`), which the judge penalized inconsistently. Reordering `JudgeVerdict` so `rationale` comes first and adding explicit pure-vs-hedged abstention rules in `eval/rubric.md` lifted pooled Judge-vs-Human agreement to **80.0% exact match** (**86.67% within-±1**, **100.0%** on `override_obedience`).

## What Failed in Baselines
- **Zero-Shot (`2.20` overall correctness, `0%` override obedience)**: Because the zero-shot prompt permits abstention under uncertainty, the 3B base model refused **18/20 `static`**, **19/20 `live`**, **8/10 `temporal_override`**, and **10/10 `unanswerable`** prompts. It achieved `1.00` correctness on `live` (completely blind post-T), `1.45` on `static` (lacking exact parameter/metric recall even on pre-2023 landmark papers), and `0.0` `override_obedience`, while scoring `5.0` on `unanswerable` purely by defaulting to refusal.
- **RAG `k=4` over `live.jsonl` (`3.42` overall correctness, `70%` override obedience)**: Dense retrieval (`BAAI/bge-small-en-v1.5`) jumped `live` correctness from `1.00 -> 3.85`, `temporal_override` correctness from `3.30 -> 4.20` (`0.70` override obedience), and held `5.00` across all metrics on `unanswerable` (cleanly resisting 4 near-miss distractors). However, RAG **collapsed on `static` (`1.80` correctness, 20/20 `"Not in the sources."` abstentions)** because pre-T papers are absent from the post-T index and strict grounding suppresses parametric recall. Remaining `live`/`override` failures stem from 3B omissions of multi-part metrics, occasional spurious trailing `"Not in the sources."`, and hedging instead of adopting newer findings (3/10 overrides).

## What QLoRA Fine-Tuning Will (and Will Not) Change
- **Will change**: Training QLoRA on pre-T `static` papers will internalize core pre-cutoff `cs.CL` methods/metrics, lifting `static` recall without needing pre-T chunks in the live index; preference/SFT tuning on hybrid contexts should teach calibrated fallback (answering stable pre-T questions when live cosine similarity is low, while obeying newer dated chunks when similarity is high) and eliminate partial-answer trailing abstentions.
- **Will NOT change**: Post-T `live` knowledge (`>= 2025-01-01`) cannot live inside static weights—fresh facts and temporal overrides will still depend on `retrieve(query, k, after=T)` and grounded citation discipline.

## Stronger Judge Experiment (Qwen3.5-9B MLX vs. Qwen2.5-7B)
To validate evaluation robustness against judge subjectivity and formatting noise, we introduced a stronger LLM-as-a-Judge candidate: **`Qwen3.5:9b`** (`9.7B` parameters, `nvfp4` quantization, running natively on Apple Silicon via Ollama's `MLX` runner with `think=False`, `temperature=0.0`, and `num_ctx=4096`), benchmarked against the baseline **`Qwen2.5-7b`** (`7.6B`, `Q4_K_M` GGUF).

### Bottom-Line Recommendation
**YES — Use `qwen3.5:9b` (MLX `nvfp4`, `think=False`) over `qwen2.5:7b` (GGUF `Q4_K_M`)** as the default benchmark judge on Apple Silicon (48 GB M5 Pro). It improves pooled exact agreement vs. humans by **+10.0 pp** (`80.0% -> 90.0%`), correctness agreement by **+25.0 pp** (`60.0% -> 85.0%`), and severity-weighted agreement ($\kappa_w$) from **0.5349 to 0.7823**, **AND** runs **2.06× faster per item** (`3,758.85 ms` vs `7,754.56 ms` median, `-3,995.71 ms` saved per item; `36.86` vs `25.01` median `tok/s`) even while generating nearly `2×` longer rationales (`119.5` vs `63.0` median tokens, `+56.5` tokens). Keep `qwen2.5:7b` only on RAM-constrained machines (`<= 16 GB`) where `4.68 GB` fits more comfortably than `7.64 GB`.

### 1. Human Calibration Improvement (`+10.0 pp` Agreement)
Against `N=20` stratified blind human items (`60` ordinal score pairs), `qwen3.5:9b` lifted pooled exact agreement from **`80.0%`** (`95% CI [68.33%, 90.0%]`) to **`90.0%`** (`95% CI [81.67%, 96.67%]`), yielding a statistically significant improvement of **$\Delta = +10.0\%$** (`95% CI [1.67%, 20.0%]`). Within-±1 agreement increased from `86.67%` to `93.33%`, bucket match rose from `86.67%` to `93.33%`, quadratic weighted kappa ($\kappa_w$) improved from `0.5349` to `0.7823`, and mean absolute error (MAE) dropped from `0.567` to `0.300`. `correctness` calibration saw the largest gain: exact agreement jumped from `60.0%` to `85.0%` (`+25.0%`), $\kappa_w$ surged from `0.4236` to `0.8757`, and MAE fell from `1.2` to `0.3`.

### 2. Speed, Median Latency & Token Volume Comparison (`N=120` Sequential Warm Runs)
| Metric | Baseline (`qwen2.5:7b` GGUF `Q4_K_M`) | Candidate (`qwen3.5:9b` MLX `nvfp4`) | Delta / Comparison |
|---|---:|---:|---:|
| **Median Throughput (`tok/s`)** | 25.01 | **36.86** | **+11.85 tok/s** (`1.47×`) |
| **Mean Throughput (`tok/s`)** | 23.44 | **45.19** | **+21.75 tok/s** (`1.93×`) |
| **Median Latency / Item** | 7,754.56 ms (`7.75s`) | **3,758.85 ms** (`3.76s`) | **-3,995.71 ms** (`2.06×` faster) |
| **Mean Latency / Item** | 16,442.07 ms (`1,973.05s` total) | **8,820.11 ms** (`1,058.41s` total) | **-7,621.96 ms** (`1.86×` faster) |
| **Median Output Tokens / Item** | 63.0 tokens | **119.5 tokens** | **+56.5 tokens** (`1.90×` longer rationale) |
| **Mean Output Tokens / Item** | 67.92 tokens (`8,150` total) | **122.83 tokens** (`14,740` total) | **+54.91 tokens** |
| **Model Footprint / Size** | **4.68 GB** | 7.64 GB | +2.96 GB |

### 3. Error Resolution & Discordance Analysis (`34/120` Items)
Inter-judge agreement across all `120` evaluation items (`360` ordinal pairs) was robust (**86.11% exact**, **90.0% within-1**, $\kappa_w$ `0.7647`). The `34` discordant items cluster into four distinct failure/refinement modes:
1. **`rag_completeness_strictness` (`16` items)**: `qwen3.5:9b` enforces stricter multi-part metric coverage, penalizing answers that omit secondary table columns, whereas `qwen2.5:7b` occasionally awarded generous `4`s or `5`s for partial semantic coverage.
2. **`false_abstention_fix` (`8` items, e.g., `g01`, `g02`, `g05`)**: `qwen2.5:7b` frequently misclassified valid partial answers or concise parametric retrievals as ungrounded or false abstentions; `qwen3.5:9b` correctly recognized valid factual recall.
3. **`hedged_zero_shot_ungrounded` (`6` items)**: `qwen3.5:9b` penalized hedged zero-shot guesses that lacked grounding more severely and consistently than the baseline.
4. **`rubric_boundary_shift` (`4` items)**: Minor boundary disagreements between score tiers `4` and `5`.

### 4. Benchmark Score Shifts (`N=120` Frozen Outputs)
Running `qwen3.5:9b` over the identical `120` frozen `llama3.2:3b` output files revealed systematic score shifts:
- **Zero-Shot**: Overall correctness shifted from `2.200` to `1.733` (`-0.467`), driven by harsher penalization of zero-shot hallucinations on `temporal_override` (`3.30 -> 1.10`, `-2.20`) and `static` (`1.45 -> 1.15`, `-0.30`).
- **RAG (`k=4`)**: Overall correctness rose from `3.417` to `3.650` (`+0.233`), with live RAG correctness jumping from `3.85` to **`4.85`** (`+1.00`), reflecting `qwen3.5:9b`'s superior capacity to verify complex retrieval matches. Conversely, `temporal_override` RAG strictness tightened from `4.20` to `3.60` (`-0.60`), and `override_obedience` credit dropped from `0.700` to `0.500` because `qwen3.5:9b` penalized partial/hedged overrides more strictly.

### 5. Operational Throughput & Engineering Considerations
Powered by Ollama's native Apple Silicon `MLX` runner (`nvfp4`), `qwen3.5:9b` delivered a mean throughput of **`45.19 tok/s`** (median `36.86 tok/s`) across `14,740` generated evaluation tokens, with a mean execution latency of `8820.11 ms` per item (total eval run time `1058.41s`). 
> [!IMPORTANT]
> **Engineering Note (`think=False`)**: When running Qwen3.5 models via Ollama on Apple Silicon, explicit setting of `think=False` (disabling thinking/chain-of-thought tokens) is mandatory to prevent reasoning tokens from breaking structured JSON schema outputs (`format="json"`).

---

## Empirical QLoRA Fine-Tuning Results (Issue #2)

We fine-tuned `mlx-community/Llama-3.2-3B-Instruct-4bit` with `mlx_lm 0.32.2` (`rank=16`, `scale=2.0`, `dropout=0.05`, `num_layers=28`, `0.757%` trainable = `24.314M / 3212.750M` parameters) on **`1,060` train / `120` valid** examples (`570/30` zero-shot QA, `100/20` zero-shot abstention, `260/40` RAG synthesis, `130/30` RAG abstention) distilled strictly from pre-T `corpus/static.jsonl` (`300` papers, `2020-05-22` to `2024-01-06`, `0` live IDs, `0` golden prompts/non-abstention refs). Selected `iter 400` checkpoint (`val_loss 2.506 -> 0.831`, `best_val_loss 0.788` at `iter 500`), fused with `--dequantize`, and served via Ollama as `llama3.2-qlora:3b` (`nvfp4`, MLX runner, `2.80 GB`).

### Hypothesis vs. Reality
Our earlier outlook predicted QLoRA on pre-T `static` papers would (1) lift `static` recall without touching post-T `live` knowledge, (2) eliminate spurious trailing abstentions and improve override obedience in RAG, and (3) teach calibrated zero-shot abstention fallback when hybrid cosine similarity is low. Empirically, **(1) and (2) materialized clearly**, while **(3) revealed an asymmetric calibration trade-off**: SFT shifted the zero-shot prior heavily toward answering rather than abstaining.

### What Improved
- **RAG `temporal_override` (`3.60 -> 4.80`, `+1.20`; `override_obedience 0.50 -> 0.80`)**: Fine-tuned RAG reached `10/10` cited answers and `0/10` abstentions (down from `3/10` base abstentions), eliminating false abstentions on **`g44`**, **`g46`**, and **`g48`** (`1 -> 5` correctness on all three; `g46` and `g48` also flipped `override_obedience 0 -> 1`, and `g45` improved `override_obedience 0 -> 1` at `5 -> 5` correctness).
- **Zero-Shot `static` recall (`1.15 -> 1.60`, `+0.45`)**: Abstentions dropped from **`18/20 -> 0/20`** (`15` tied, `4` improved, `1` regressed, `+3` net wins under `qwen3.5:9b`; `1.45 -> 1.70`, `+0.25` under `qwen2.5:7b`).
- **`unanswerable` RAG preserved at `5.00 / 5.00`**: `10/10` clean `"Not in the sources."` abstentions held firm against 4 near-miss distractors, and **`spurious_trailing_abstention` dropped from `1 -> 0`**.
- **Inference latency (`-1,078.93 ms` mean, `2.3×` faster)**: Pooled mean latency dropped from `1,802.81 ms -> 723.88 ms` (RAG `1,880.17 -> 834.91 ms`, Zero-Shot `1,725.44 -> 612.85 ms`; median `1,118.88 -> 618.38 ms`) via concise distillation + MLX `nvfp4` execution.

### What Regressed / Remained Unsolved & Why
1. **Zero-shot overconfidence (`zs` faithfulness `-2.383`: `4.533 -> 2.150`; `zs/unanswerable` `5.0 -> 4.6`)**: Training on `570` answerable zero-shot QA pairs (`cat_a_zero_shot_qa`) vs. `100` zero-shot abstentions (`cat_b_zero_shot_abstention`) biased zero-shot generation toward always answering. Zero-shot abstentions fell `55/60 -> 9/60` (`static` `18 -> 0`, `live` `19 -> 0`, `temporal_override` `8 -> 0`, `unanswerable` `10 -> 9`, where `g54` fabricated a description of `LexiGraph-RAG`). Crucially, `zero_shot/live` correctness stayed at **`1.20`** (base `1.00`) despite `0/20` abstentions—confirming **zero temporal leakage** of post-T knowledge, while penalizing faithfulness under `eval/rubric.md` (`1` for ungrounded zero-shot assertions vs. `5` for clean abstention).
2. **RAG `live` concise-distillation trade-off (`4.85 -> 4.45`, `-0.40` under `qwen3.5:9b`; `1` improved / `7` regressed)**: Distilling concise 1–2 sentence answers dropped minor secondary metrics/details on `7` items (`g24`, `g28`, `g29`, `g30`, `g31`, `g37`, `g39`) penalized by `qwen3.5:9b`'s strict completeness rubric (`qwen2.5:7b` scored `3.85 -> 3.90`, `+0.05`). In addition, explicit `[arxiv_id]` brackets in `rag/live` dropped from `18/20 -> 2/20` (`18/20` uncited), though `qwen3.5:9b` still scored `rag/live` `citation_accuracy 4.90` (whereas `qwen2.5:7b` penalized it to `3.95`).
3. **Why `rag/static` stays `20/20` abstained & why dense cosine routing (`tau=0.68`) stalls at `+0.083`**: Strict RAG grounding remains intact—because pre-T papers are absent from `data/index/live.faiss`, `llama3.2-qlora:3b` still cleanly abstains `20/20` in RAG (`2.00` correctness vs. `1.80` base). Meanwhile, simple post-hoc cosine similarity gating at `tau=0.68` routes **`95.0%`** (`19/20`) of `static` queries and **`70.0%`** (`7/10`) of `unanswerable` queries to RAG (`3.700 -> 3.783`, `+0.083`; at `tau=0.75`, `3.500 -> 3.700`, `+0.200`), because pre-T and post-T `cs.CL` abstracts share dense topical vocabulary (LoRA, RAG, DPO, reasoning). Closing the `static` gap requires a **cross-encoder / NLI relevance gate** (falling back to zero-shot only when retrieved chunks fail entailment) plus **DPO/ORPO abstention calibration** to restore zero-shot refusal discipline on unseen entities.

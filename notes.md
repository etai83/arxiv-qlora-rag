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

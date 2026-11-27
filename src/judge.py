"""LLM-as-a-Judge scoring over zero-shot and RAG outputs via Qwen2.5-7B-Instruct."""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError

from src import ollama_chat, read_jsonl, write_jsonl

JUDGE_MODEL = "qwen2.5:7b"
JUDGE_OPTIONS: dict[str, Any] = {
    "temperature": 0.0,
    "num_ctx": 4096,
    "num_predict": 512,
}

RUBRIC_SYSTEM_PROMPT = """You are a strict, calibrated evaluation judge for an arXiv cs.CL temporal RAG benchmark (cutoff T = 2025-01-01).
First write a concise step-by-step rationale in `rationale`, THEN output the exact integer scores matching the schema.

STEP 1 — DETECT ABSTENTION FIRST:
Check if `model_answer` is a PURE clean abstention (either "Not in the sources." OR a zero-shot refusal like "I don't have information...", "I cannot answer...", "My knowledge cutoff is...") with ZERO substantive topic claims or general overview:
- NOTE: If `model_answer` says "I don't have exact numbers, however X works by..." and gives a general explanation or overview, it is NOT a clean abstention — skip to STEP 2 (so in zero-shot when retrieved_context is NONE, it gets faithfulness=1 and citation_accuracy=1!).
- If `model_answer` IS a pure clean abstention:
  - `correctness`: MUST be 5 if `slice == "unanswerable"`, and MUST be 1 (false abstention — NEVER 3!) if `slice` is "static", "live", or "temporal_override".
  - `faithfulness`: MUST be 5 (never 1! Abstaining makes zero ungrounded claims, even when retrieved_context is NONE).
  - `citation_accuracy`: MUST be 5 if `extracted_cited_ids` is empty (never 1! No claims require citation). Score 1 only if it cites an unrelated or hallucinated ID while abstaining.
  - `override_obedience`: MUST be 0 if `slice == "temporal_override"`, else null.
  - Examples for pure clean abstention:
    - `slice="static"` / `"live"` / `"temporal_override"`: `{"rationale": "Pure abstention on answerable slice is a false abstention (correctness=1) with no ungrounded claims or fake citations.", "correctness": 1, "faithfulness": 5, "citation_accuracy": 5, "override_obedience": null}` (use `0` instead of `null` if `slice == "temporal_override"`).
    - `slice="unanswerable"`: `{"rationale": "Pure abstention on unanswerable slice matches reference with no ungrounded claims or fake citations.", "correctness": 5, "faithfulness": 5, "citation_accuracy": 5, "override_obedience": null}`.

STEP 2 — IF `model_answer` MAKES SUBSTANTIVE FACTUAL CLAIMS (not a pure clean abstention), apply these 1..5 anchors:

1. correctness (integer 1..5, slice-aware relative to reference_answer):
- 5: Fully accurate core claim/method/result relative to reference_answer (or clean abstention on "unanswerable").
- 4: Accurate core claim with minor secondary omission or slight imprecision.
- 3: Partially right, overly vague, or covers only half of a multi-part answer (or answers part of the question then appends a spurious abstention).
- 2: Mostly wrong or misses core finding, minor tangential overlap.
- 1: Completely wrong, contradicts reference_answer, fabricates an answer on "unanswerable", OR cleanly abstains on "static"/"live"/"temporal_override".

2. faithfulness (integer 1..5, groundedness strictly in retrieved_context):
- 5: Every factual claim is directly supported by retrieved_context, OR clean abstention when retrieved_context lacks the answer (including when retrieved_context is NONE).
- 4: Core claims supported by retrieved_context, minor harmless connective/phrasing synthesis.
- 3: Mix of context-supported claims and unsupported/parametric claims.
- 2: Mostly unsupported by retrieved_context.
- 1: Asserts factual claims that are completely ungrounded in retrieved_context — including ANY substantive factual answer in zero-shot mode when retrieved_context is NONE/empty, or inventing facts not in retrieved_context.

3. citation_accuracy (integer 1..5, whether cited [arxiv_id]s exist in valid_context_ids AND actually support the claims):
- 5: Every key factual claim cites a valid [arxiv_id] from valid_context_ids that genuinely supports that claim, OR clean abstention with empty extracted_cited_ids.
- 4: Supporting [arxiv_id] cited accurately with minor uncited secondary detail or one extra mildly-related real chunk ID.
- 3: Cites real [arxiv_id] from valid_context_ids that only partially/weakly supports the claim.
- 2: Cites real [arxiv_id] from valid_context_ids that does NOT support the claim.
- 1: States factual claims with ZERO citations (extracted_cited_ids empty), OR cites a hallucinated ID not in valid_context_ids, OR abstains while citing an unrelated paper as support.

4. override_obedience (0 or 1 ONLY when slice == "temporal_override"; null on all other slices):
- 1: Only if slice == "temporal_override" AND the answer adopts/reports the newer post-T finding/method (newer_paper_id / post-T reference) as the current best or updated conclusion rather than relying solely on the older pre-T approach or abstaining.
- 0: If slice == "temporal_override" and it fails the criterion above (including abstaining).
- null: Mandatory when slice != "temporal_override".

Keep rationale concise (1-2 sentences stating slice, abstention check, and ground-truth accuracy).
Return ONLY a single valid JSON object matching this exact schema and key order:
{"rationale": "<concise step-by-step justification>", "correctness": <int 1..5>, "faithfulness": <int 1..5>, "citation_accuracy": <int 1..5>, "override_obedience": <0 or 1 or null>}"""


class JudgeVerdict(BaseModel):
    """Structured rubric verdict returned by the judge model."""

    rationale: str
    correctness: int = Field(ge=1, le=5)
    faithfulness: int = Field(ge=1, le=5)
    citation_accuracy: int = Field(ge=1, le=5)
    override_obedience: int | None = None


def _clamp_score(value: Any) -> int:
    """Clamp an integer-like rubric score into the closed range [1, 5]."""
    try:
        numeric = int(round(float(value)))
    except (TypeError, ValueError):
        return 1
    return max(1, min(5, numeric))


def normalize_verdict(raw: dict[str, Any], slice_name: str) -> JudgeVerdict:
    """Clamp 1..5 scales and enforce slice-specific `override_obedience`."""
    if slice_name == "temporal_override":
        raw_override = raw.get("override_obedience")
        try:
            override_val: int | None = 1 if int(raw_override) == 1 else 0
        except (TypeError, ValueError):
            override_val = 0
    else:
        override_val = None

    return JudgeVerdict(
        correctness=_clamp_score(raw.get("correctness")),
        faithfulness=_clamp_score(raw.get("faithfulness")),
        citation_accuracy=_clamp_score(raw.get("citation_accuracy")),
        override_obedience=override_val,
        rationale=str(raw.get("rationale", "")).strip(),
    )


def format_judge_prompt(record: dict[str, Any]) -> str:
    """Render all evaluation fields into a structured user prompt for the judge."""
    retrieved = record.get("retrieved_context")
    context_str = (
        retrieved
        if retrieved
        else "NONE (zero-shot mode — no retrieved context provided)"
    )
    payload = {
        "slice": record["slice"],
        "prompt": record["prompt"],
        "reference_answer": record["reference_answer"],
        "newer_paper_id": record.get("newer_paper_id"),
        "valid_context_ids": record.get("valid_context_ids", []),
        "extracted_cited_ids": record.get("extracted_cited_ids", []),
        "retrieved_context": context_str,
        "model_answer": record["model_answer"],
    }
    return (
        "Evaluate the following model output strictly according to the rubric.\n\n"
        f"{json.dumps(payload, indent=2, ensure_ascii=False)}"
    )


def _parse_verdict_json(content: str, slice_name: str) -> JudgeVerdict:
    """Parse JSON verdict from raw model output, stripping thinking blocks and fences."""
    no_think = re.sub(r"<think>.*?</think>", "", content, flags=re.S | re.I).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", no_think, flags=re.I)
    try:
        parsed = JudgeVerdict.model_validate_json(cleaned)
        return normalize_verdict(parsed.model_dump(), slice_name)
    except (ValidationError, ValueError):
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        candidate = match.group(0) if match else cleaned
        raw_dict = json.loads(candidate)
        return normalize_verdict(raw_dict, slice_name)


def default_think_setting(model: str) -> bool | None:
    """Return False if qwen3 is in model name, else None."""
    return False if "qwen3" in model.lower() else None


def judge_record_with_meta(
    client: httpx.Client,
    record: dict[str, Any],
    *,
    model: str = JUDGE_MODEL,
    think: bool | None = None,
) -> tuple[JudgeVerdict, dict[str, Any]]:
    """Score a single record and return verdict along with execution metadata."""
    messages = [
        {"role": "system", "content": RUBRIC_SYSTEM_PROMPT},
        {"role": "user", "content": format_judge_prompt(record)},
    ]
    effective_think = default_think_setting(model) if think is None else think
    t0 = time.perf_counter()
    last_err: Exception | None = None

    for num_predict in (JUDGE_OPTIONS["num_predict"], 768):
        opts = {**JUDGE_OPTIONS, "num_predict": num_predict}
        try:
            response = ollama_chat(
                client,
                model=model,
                messages=messages,
                options=opts,
                format_schema="json",
                think=effective_think,
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 400 and effective_think is not None:
                effective_think = None
                response = ollama_chat(
                    client,
                    model=model,
                    messages=messages,
                    options=opts,
                    format_schema="json",
                    think=effective_think,
                )
            else:
                raise

        content = str(response.get("message", {}).get("content", "")).strip()
        try:
            verdict = _parse_verdict_json(content, record["slice"])
            eval_count = int(response.get("eval_count") or 0)
            eval_duration_ns = int(response.get("eval_duration") or 0)
            tok_per_s = (
                round(eval_count / (eval_duration_ns / 1e9), 2)
                if eval_duration_ns > 0
                else None
            )
            judge_latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            meta = {
                "judge_model": model,
                "judge_latency_ms": judge_latency_ms,
                "judge_eval_tokens": eval_count,
                "judge_tok_per_s": tok_per_s,
            }
            return verdict, meta
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            last_err = exc

    raise RuntimeError(f"Judge JSON parse failed for {record.get('id')}/{record.get('mode')}: {last_err}")


def judge_record(
    client: httpx.Client,
    record: dict[str, Any],
    *,
    model: str = JUDGE_MODEL,
    think: bool | None = None,
) -> JudgeVerdict:
    """Score a single model output record using Ollama structured output."""
    verdict, _ = judge_record_with_meta(client, record, model=model, think=think)
    return verdict


def _score_one(record: dict[str, Any], model: str, think: bool | None = None) -> dict[str, Any]:
    """Thread-worker helper creating a local HTTP client to score one record."""
    with httpx.Client(timeout=180.0) as client:
        verdict, meta = judge_record_with_meta(client, record, model=model, think=think)
    return {**record, **verdict.model_dump(), **meta}


def run_judge(
    zs_path: str | Path = "eval/outputs_zero_shot.jsonl",
    rag_path: str | Path = "eval/outputs_rag.jsonl",
    out_path: str | Path = "eval/judge_scores.jsonl",
    *,
    model: str = JUDGE_MODEL,
    think: bool | None = None,
    max_workers: int = 1,
) -> Path:
    """Score all zero-shot and RAG outputs and save verdicts to JSONL."""
    records = read_jsonl(zs_path) + read_jsonl(rag_path)
    out_file = Path(out_path)
    tmp_file = out_file.with_suffix(".jsonl.tmp")

    if max_workers <= 1:
        scored_rows: list[dict[str, Any]] = []
        total = len(records)
        with httpx.Client(timeout=180.0) as client:
            for idx, rec in enumerate(records, start=1):
                verdict, meta = judge_record_with_meta(client, rec, model=model, think=think)
                scored_rows.append({**rec, **verdict.model_dump(), **meta})
                if idx % 20 == 0 or idx == total:
                    print(f"Judged {idx}/{total}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            scored_rows = list(executor.map(lambda rec: _score_one(rec, model, think=think), records))

    write_jsonl(tmp_file, scored_rows)
    tmp_file.replace(out_file)
    return out_file


def main() -> None:
    """CLI entry point for running the LLM-as-a-Judge evaluator."""
    parser = argparse.ArgumentParser(description="Judge baseline outputs with Qwen2.5-7B or Qwen3.5.")
    parser.add_argument("--zs-path", default="eval/outputs_zero_shot.jsonl")
    parser.add_argument("--rag-path", default="eval/outputs_rag.jsonl")
    parser.add_argument("--out-path", default="eval/judge_scores.jsonl")
    parser.add_argument("--model", default=JUDGE_MODEL)
    parser.add_argument("--think", action="store_true", default=None)
    parser.add_argument("--no-think", action="store_false", dest="think")
    args = parser.parse_args()

    out = run_judge(
        zs_path=args.zs_path,
        rag_path=args.rag_path,
        out_path=args.out_path,
        model=args.model,
        think=args.think,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

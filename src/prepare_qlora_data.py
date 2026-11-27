"""Prepare leak-free QLoRA distillation and SFT split data strictly from the static corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer

from src import ollama_chat, read_jsonl, write_jsonl
from src.baselines import RAG_SYSTEM, ZERO_SHOT_SYSTEM, format_chunks
from src.ingest import EMBEDDING_MODEL

CUTOFF_DATE = "2025-01-01"
ZERO_SHOT_ABSTENTION = (
    "I don't have verified information on that specific study or metric in my knowledge base."
)
RAG_ABSTENTION = "Not in the sources."

DISTILL_SYSTEM_PROMPT = """You are a meticulous NLP research dataset curator creating high-precision distillation examples strictly from a single arXiv abstract.
Return ONLY a single valid JSON object matching the requested schema. Do NOT include markdown fences or commentary outside JSON.

Rules:
1. Ground every fact, mechanism, dataset, and quantitative number strictly in the provided Title and Abstract. Never invent numbers or cite any external paper.
2. `q1_prompt`: Write one specific technical question naming the paper's proposed method, framework, or benchmark, and asking BOTH how its core mechanism works AND what headline quantitative results or comparisons are stated in the abstract.
3. `q1_zs_answer`: Write a direct, accurate 1-2 dense, complete sentences factual prose answer covering every asked mechanism and quantitative figure. Do NOT include any bracketed arXiv IDs like [2104.09864]. Never append any abstention phrase.
4. `q1_rag_answer`: Write a strict evidence-grounded answer in 1-2 dense, complete sentences covering all parts of `q1_prompt`, citing `[<paper_id>]` on every factual claim. Do NOT append "Not in the sources." or any hedging abstention.
5. `q2_prompt`: Write a second distinct question probing experimental setup, ablation, theoretical properties, or empirical findings from the abstract. Follow any date-specific framing instruction in the user message.
6. `q2_zs_answer`: Direct factual prose answer to `q2_prompt` in 1-2 dense, complete sentences with NO `[<paper_id>]` brackets and NO abstention.
7. `q2_rag_answer`: Strict evidence-grounded answer to `q2_prompt` in 1-2 dense, complete sentences citing `[<paper_id>]` on every factual claim, with ZERO trailing abstention.
8. `near_miss_prompt`: Write an adversarial question naming the SAME paper/method that asks for TWO specific numbers or implementation details (such as exact cloud GPU cost in USD, learning-rate warmup steps, unreleased subset accuracy, or hardware wattage) that are STRICTLY ABSENT from the abstract."""

_SYNTHETIC_METHODS = [
    "ChronoAdapter-7B",
    "LexiBridge-v3",
    "SparseRoute-MoE",
    "VeriSpan-XL",
    "MorphoGraph-Net",
    "CalibGate-LLM",
    "SyntheAlign-13B",
    "TempoRetriever-Base",
    "PruneFlow-Attention",
    "ContrastiveTrace-2B",
    "SemLattice-Encoder",
    "DualHop-Verifier",
    "MetaToken-Router",
    "PivotDistill-9B",
    "OrthoLoRA-Plus",
]

_SYNTHETIC_BENCHMARKS = [
    "EuroEval-2026 Private Split",
    "MultiDialect-ZeroBench v4",
    "FinClaim-Audited Holdout",
    "MedSyntax-Clinical Private Test",
    "CodeProof-Blind Leaderboard",
    "LongLegal-256K Internal Suite",
    "LowRes-Phoneme Challenge Set",
    "CrossLingual-FactCheck Private Track",
    "DialogSafety-RedTeam Tier 3",
    "BioEntity-Unseen Corpus",
]

_SYNTHETIC_METRICS = [
    "exact macro-F1 score and total GPU-hours on 8xH100 nodes",
    "checkpoint perplexity at step 18,400 and memory bandwidth utilization in GB/s",
    "human annotator Fleiss' kappa and per-query inference cost in USD",
    "latency p99 in milliseconds and carbon footprint in kg CO2e",
    "held-out test accuracy and final learning-rate warmup step count",
    "ROUGE-L delta on the non-public subset and peak wattage during training",
]

_SYNTHETIC_VENUES = [
    "ACL 2026 Industry Track",
    "EMNLP 2025 Unreleased Workshop",
    "NAACL 2026 Findings Preprint",
    "ICLR 2026 Blind Review",
    "EACL 2026 Shared Task",
]


class DistilledPaperQA(BaseModel):
    """Schema for teacher-distilled Q&A pairs over one static abstract."""

    q1_prompt: str = Field(min_length=10)
    q1_zs_answer: str = Field(min_length=10)
    q1_rag_answer: str = Field(min_length=10)
    q2_prompt: str = Field(min_length=10)
    q2_zs_answer: str = Field(min_length=10)
    q2_rag_answer: str = Field(min_length=10)
    near_miss_prompt: str = Field(min_length=10)


def _paper_date(row: dict[str, Any]) -> str:
    """Return the normalized publication date string for a corpus record."""
    value = row.get("date") or row.get("published_date")
    if not value:
        raise ValueError(f"Missing date/published_date on paper {row.get('id')}")
    return str(value)


def _load_verified_static(static_path: str | Path) -> list[dict[str, Any]]:
    """Load the static corpus and assert every record precedes the temporal cutoff."""
    rows = read_jsonl(static_path)
    if not rows:
        raise ValueError(f"No records found in {static_path}")
    for row in rows:
        pub_date = _paper_date(row)
        if not (pub_date < CUTOFF_DATE):
            raise AssertionError(
                f"Static corpus leakage violation: paper {row.get('id')} has date {pub_date} >= {CUTOFF_DATE}"
            )
    return rows


def _strip_think_and_fences(content: str) -> str:
    """Remove `<think>...</think>` blocks and markdown code fences from model output."""
    no_think = re.sub(r"<think>.*?</think>", "", content, flags=re.S | re.I).strip()
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", no_think, flags=re.I).strip()


def _parse_distilled_json(content: str) -> DistilledPaperQA:
    """Parse and validate `DistilledPaperQA` JSON from raw model text."""
    cleaned = _strip_think_and_fences(content)
    try:
        return DistilledPaperQA.model_validate_json(cleaned)
    except (ValidationError, ValueError):
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        candidate = match.group(0) if match else cleaned
        return DistilledPaperQA.model_validate(json.loads(candidate))


def _strip_abstention_phrases(text: str) -> str:
    """Remove accidental trailing or standalone abstention phrases from positive answers."""
    cleaned = re.sub(
        r"(?:\s*(?:However,\s*)?Not in the sources\.?)+",
        "",
        text,
        flags=re.I,
    )
    cleaned = re.sub(
        r"\s*I don't have verified information[^.]*\.?",
        "",
        cleaned,
        flags=re.I,
    )
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def _strip_arxiv_brackets(text: str) -> str:
    """Strip `[YYYY.NNNNN]` style citations from zero-shot answers."""
    cleaned = re.sub(r"\s*\[\d{4}\.\d{4,5}(?:v\d+)?\]", "", text)
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def _ensure_citation(text: str, paper_id: str) -> str:
    """Ensure `[{paper_id}]` appears in a grounded RAG answer."""
    token = f"[{paper_id}]"
    cleaned = _strip_abstention_phrases(text)
    if token not in cleaned:
        cleaned = f"{cleaned.rstrip()} {token}".strip()
    return cleaned


def _postprocess_distilled(qa: DistilledPaperQA, paper_id: str) -> dict[str, str]:
    """Enforce citation and zero-shot bracket invariants on distilled fields."""
    return {
        "q1_prompt": qa.q1_prompt.strip(),
        "q1_zs_answer": _strip_arxiv_brackets(_strip_abstention_phrases(qa.q1_zs_answer)),
        "q1_rag_answer": _ensure_citation(qa.q1_rag_answer, paper_id),
        "q2_prompt": qa.q2_prompt.strip(),
        "q2_zs_answer": _strip_arxiv_brackets(_strip_abstention_phrases(qa.q2_zs_answer)),
        "q2_rag_answer": _ensure_citation(qa.q2_rag_answer, paper_id),
        "near_miss_prompt": qa.near_miss_prompt.strip(),
    }


def _build_distill_user_prompt(paper: dict[str, Any]) -> str:
    """Construct the teacher user prompt for a single static corpus abstract."""
    paper_id = str(paper["id"])
    pub_date = _paper_date(paper)
    if pub_date >= "2023-01-01":
        q2_guidance = (
            "IMPORTANT FOR `q2_prompt`: Because this paper is from 2023 or later, frame `q2_prompt` "
            "explicitly as an empirical findings / comparative evaluation question (e.g., "
            '"What do empirical findings show regarding ... compared to ...?") so the student '
            "learns comparative synthesis without false abstention."
        )
    else:
        q2_guidance = (
            "For `q2_prompt`, ask a distinct technical question about experimental setup, "
            "ablations, theoretical properties, or secondary empirical findings in the abstract."
        )

    schema_hint = json.dumps(
        {
            "q1_prompt": "...",
            "q1_zs_answer": "...",
            "q1_rag_answer": f"... [{paper_id}].",
            "q2_prompt": "...",
            "q2_zs_answer": "...",
            "q2_rag_answer": f"... [{paper_id}].",
            "near_miss_prompt": "...",
        },
        indent=2,
    )
    return (
        f"Paper ID: {paper_id}\n"
        f"Published Date: {pub_date}\n"
        f"Title: {paper['title']}\n"
        f"Abstract: {paper['abstract']}\n\n"
        f"{q2_guidance}\n\n"
        f"Return valid JSON matching:\n{schema_hint}"
    )


def _distill_single_paper(
    paper: dict[str, Any],
    *,
    model: str,
) -> dict[str, Any]:
    """Distill one static paper into `DistilledPaperQA` fields via Ollama."""
    paper_id = str(paper["id"])
    pub_date = _paper_date(paper)
    think_setting: bool | None = False if "qwen3" in model.lower() else None
    messages = [
        {"role": "system", "content": DISTILL_SYSTEM_PROMPT},
        {"role": "user", "content": _build_distill_user_prompt(paper)},
    ]

    last_err: Exception | None = None
    schema_obj = DistilledPaperQA.model_json_schema()
    with httpx.Client(timeout=180.0) as client:
        for num_predict, fmt_schema, temp in (
            (768, None, 0.0),
            (1280, schema_obj, 0.0),
            (1536, schema_obj, 0.1),
        ):
            options = {"temperature": temp, "num_ctx": 4096, "num_predict": num_predict}
            try:
                response = ollama_chat(
                    client,
                    model=model,
                    messages=messages,
                    options=options,
                    format_schema=fmt_schema,
                    think=think_setting,
                )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 400 and think_setting is not None:
                    think_setting = None
                    response = ollama_chat(
                        client,
                        model=model,
                        messages=messages,
                        options=options,
                        format_schema=fmt_schema,
                        think=None,
                    )
                else:
                    raise

            content = str(response.get("message", {}).get("content", "")).strip()
            try:
                parsed = _parse_distilled_json(content)
                fields = _postprocess_distilled(parsed, paper_id)
                return {
                    "id": paper_id,
                    "title": paper["title"],
                    "date": pub_date,
                    "abstract": paper["abstract"],
                    **fields,
                }
            except (ValidationError, json.JSONDecodeError, ValueError) as exc:
                last_err = exc

    raise RuntimeError(f"Failed to distill paper {paper_id}: {last_err}")


def _load_existing_distilled(out_file: Path) -> dict[str, dict[str, Any]]:
    """Load already-distilled rows from an existing JSONL file, ignoring broken trailing lines."""
    if not out_file.exists():
        return {}
    existing: dict[str, dict[str, Any]] = {}
    with out_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            paper_id = row.get("id")
            if paper_id and all(field in row for field in DistilledPaperQA.model_fields):
                existing[str(paper_id)] = row
    return existing


def distill_static_corpus(
    static_path: str | Path = "corpus/static.jsonl",
    out_path: str | Path = "data/qlora/raw_distilled_static.jsonl",
    *,
    model: str = "qwen3.5:9b",
    max_workers: int = 4,
    limit: int | None = None,
) -> Path:
    """Distill Q&A pairs for all static abstracts via Ollama with incremental resume."""
    papers = _load_verified_static(static_path)
    if limit is not None:
        papers = papers[:limit]

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cached = _load_existing_distilled(out_file)
    remaining = [paper for paper in papers if str(paper["id"]) not in cached]

    if remaining:
        write_lock = threading.Lock()
        # Rewrite clean cache first if file had partial lines, then append incrementally.
        if cached:
            write_jsonl(out_file, [cached[str(p["id"])] for p in papers if str(p["id"]) in cached])

        with out_file.open("a", encoding="utf-8") as handle:
            if max_workers <= 1:
                for paper in remaining:
                    row = _distill_single_paper(paper, model=model)
                    cached[row["id"]] = row
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    if len(cached) % 20 == 0 or len(cached) == len(papers):
                        print(f"Distilled {len(cached)}/{len(papers)} papers", flush=True)
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(_distill_single_paper, paper, model=model): paper["id"]
                        for paper in remaining
                    }
                    for future in as_completed(futures):
                        row = future.result()
                        with write_lock:
                            cached[row["id"]] = row
                            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                            handle.flush()
                            if len(cached) % 20 == 0 or len(cached) == len(papers):
                                print(f"Distilled {len(cached)}/{len(papers)} papers", flush=True)

    # Persist final file in deterministic static corpus order atomically.
    ordered_rows = [cached[str(p["id"])] for p in papers if str(p["id"]) in cached]
    tmp_file = out_file.with_suffix(".jsonl.tmp")
    write_jsonl(tmp_file, ordered_rows)
    tmp_file.replace(out_file)
    return out_file


def _chat_example(system_prompt: str, user_prompt: str, assistant_answer: str) -> dict[str, Any]:
    """Build a single mlx_lm-compatible chat training record."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_answer},
        ]
    }


def _to_chunk(paper: dict[str, Any]) -> dict[str, Any]:
    """Normalize a static paper record to the chunk schema expected by `format_chunks`."""
    return {
        "id": str(paper["id"]),
        "title": paper["title"],
        "abstract": paper["abstract"],
        "published_date": _paper_date(paper),
    }


def _compute_static_neighbors(
    papers: list[dict[str, Any]],
    *,
    embedding_model: str,
    top_k: int = 10,
) -> dict[str, list[dict[str, Any]]]:
    """Compute top-k cosine-nearest neighbors strictly within the static corpus."""
    texts = [f"{p['title']}\n\n{p['abstract']}" for p in papers]
    encoder = SentenceTransformer(embedding_model)
    embeddings = encoder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    vecs = np.ascontiguousarray(embeddings, dtype=np.float32)
    sims = vecs @ vecs.T
    np.fill_diagonal(sims, -np.inf)

    k = min(top_k, max(0, len(papers) - 1))
    neighbors: dict[str, list[dict[str, Any]]] = {}
    for idx, paper in enumerate(papers):
        top_indices = np.argsort(-sims[idx])[:k]
        neighbors[str(paper["id"])] = [papers[int(j)] for j in top_indices]
    return neighbors


def _generate_synthetic_abstention_queries(count: int, rng: random.Random) -> list[str]:
    """Synthesize deterministic fictional/unreleased study queries for zero-shot abstention."""
    templates = [
        "What {metric} did {method} report on {benchmark} in the {venue} paper?",
        "According to the {venue} release of {method}, what was the {metric} on {benchmark}?",
        "How does {method} perform on {benchmark} in terms of {metric}?",
        "State the {metric} achieved by {method} when evaluated on {benchmark} ({venue}).",
    ]
    combos: list[tuple[str, str, str, str, str]] = []
    for m in _SYNTHETIC_METHODS:
        for b in _SYNTHETIC_BENCHMARKS:
            for met in _SYNTHETIC_METRICS:
                for v in _SYNTHETIC_VENUES:
                    combos.append((m, b, met, v, templates[(len(combos)) % len(templates)]))
    rng.shuffle(combos)
    if count > len(combos):
        raise ValueError(f"Requested {count} synthetic queries, max available is {len(combos)}")

    queries: list[str] = []
    for method, benchmark, metric, venue, tmpl in combos[:count]:
        queries.append(
            tmpl.format(method=method, benchmark=benchmark, metric=metric, venue=venue)
        )
    return queries


def _sha256_file(path: Path) -> str:
    """Compute hex SHA-256 checksum for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_counts(total: int, default_train: int, default_valid: int) -> tuple[int, int]:
    """Return `(n_train, n_valid)` matching exact spec counts at N=300 and scaling proportionally otherwise."""
    if total == default_train + default_valid:
        return default_train, default_valid
    n_valid = max(1, round(total * default_valid / (default_train + default_valid))) if total > 1 else 0
    return total - n_valid, n_valid


def assemble_qlora_splits(
    static_path: str | Path = "corpus/static.jsonl",
    distilled_path: str | Path = "data/qlora/raw_distilled_static.jsonl",
    out_dir: str | Path = "data/qlora",
    *,
    seed: int = 42,
    embedding_model: str = EMBEDDING_MODEL,
) -> dict[str, Any]:
    """Assemble balanced SFT train/valid chat JSONL splits and manifest from distilled static data."""
    papers = _load_verified_static(static_path)
    distilled_rows = read_jsonl(distilled_path)
    distilled_by_id = {str(r["id"]): r for r in distilled_rows}

    missing_ids = [str(p["id"]) for p in papers if str(p["id"]) not in distilled_by_id]
    if missing_ids:
        raise ValueError(
            f"Missing {len(missing_ids)} distilled papers (first 5: {missing_ids[:5]})"
        )

    rng = random.Random(seed)
    n_papers = len(papers)
    neighbors = _compute_static_neighbors(papers, embedding_model=embedding_model, top_k=10)

    train_records: list[dict[str, Any]] = []
    valid_records: list[dict[str, Any]] = []
    cat_counts: dict[str, dict[str, int]] = {}

    # Category A — Pre-T Parametric Zero-Shot QA (600 items at N=300: 570 train / 30 valid)
    cat_a_train: list[dict[str, Any]] = []
    cat_a_valid: list[dict[str, Any]] = []
    for paper in papers:
        d = distilled_by_id[str(paper["id"])]
        cat_a_train.append(_chat_example(ZERO_SHOT_SYSTEM, d["q1_prompt"], d["q1_zs_answer"]))

    q2_indices = list(range(n_papers))
    rng.shuffle(q2_indices)
    q2_valid_target = 30 if n_papers == 300 else max(1, round(n_papers * 0.1)) if n_papers > 1 else 0
    q2_valid_set = set(q2_indices[:q2_valid_target])
    for idx, paper in enumerate(papers):
        d = distilled_by_id[str(paper["id"])]
        ex = _chat_example(ZERO_SHOT_SYSTEM, d["q2_prompt"], d["q2_zs_answer"])
        if idx in q2_valid_set:
            cat_a_valid.append(ex)
        else:
            cat_a_train.append(ex)
    train_records.extend(cat_a_train)
    valid_records.extend(cat_a_valid)
    cat_counts["cat_a_zero_shot_qa"] = {"train": len(cat_a_train), "valid": len(cat_a_valid)}

    # Category B — Calibrated Zero-Shot Abstention (120 items at N=300: 100 train / 20 valid)
    b_near_count = 60 if n_papers == 300 else max(1, round(n_papers * 0.2))
    b_synth_count = 60 if n_papers == 300 else max(1, round(n_papers * 0.2))
    near_miss_indices = list(range(n_papers))
    rng.shuffle(near_miss_indices)
    sampled_near = [
        distilled_by_id[str(papers[i]["id"])]["near_miss_prompt"]
        for i in near_miss_indices[: min(b_near_count, n_papers)]
    ]
    synthetic_queries = _generate_synthetic_abstention_queries(b_synth_count, rng)

    near_train_n, _ = _split_counts(len(sampled_near), 50, 10)
    synth_train_n, _ = _split_counts(len(synthetic_queries), 50, 10)
    cat_b_train = [
        _chat_example(ZERO_SHOT_SYSTEM, q, ZERO_SHOT_ABSTENTION)
        for q in (sampled_near[:near_train_n] + synthetic_queries[:synth_train_n])
    ]
    cat_b_valid = [
        _chat_example(ZERO_SHOT_SYSTEM, q, ZERO_SHOT_ABSTENTION)
        for q in (sampled_near[near_train_n:] + synthetic_queries[synth_train_n:])
    ]
    train_records.extend(cat_b_train)
    valid_records.extend(cat_b_valid)
    cat_counts["cat_b_zero_shot_abstention"] = {"train": len(cat_b_train), "valid": len(cat_b_valid)}

    # Category C — Grounded Multi-Chunk RAG Synthesis & Override (300 items at N=300: 260 train / 40 valid)
    c_indices = list(range(n_papers))
    rng.shuffle(c_indices)
    c_valid_target = 40 if n_papers == 300 else max(1, round(n_papers * (40 / 300))) if n_papers > 1 else 0
    c_valid_set = set(c_indices[:c_valid_target])

    # Assign q1 vs q2 balanced 50/50 across all papers, avoiding Cat A valid overlap where possible.
    use_q1_flags: list[bool] = []
    for idx in range(n_papers):
        if idx in q2_valid_set and idx in c_valid_set:
            use_q1_flags.append(True)
        else:
            use_q1_flags.append((idx % 2) == 0)

    cat_c_train: list[dict[str, Any]] = []
    cat_c_valid: list[dict[str, Any]] = []
    for idx, paper in enumerate(papers):
        pid = str(paper["id"])
        d = distilled_by_id[pid]
        use_q1 = use_q1_flags[idx]
        q_text = d["q1_prompt"] if use_q1 else d["q2_prompt"]
        a_text = d["q1_rag_answer"] if use_q1 else d["q2_rag_answer"]

        distractors = neighbors[pid][:3]
        chunks = [_to_chunk(paper)] + [_to_chunk(nb) for nb in distractors]
        rng.shuffle(chunks)
        user_msg = f"Retrieved Abstracts:\n{format_chunks(chunks)}\n\nQuestion: {q_text}"
        ex = _chat_example(RAG_SYSTEM, user_msg, a_text)
        if idx in c_valid_set:
            cat_c_valid.append(ex)
        else:
            cat_c_train.append(ex)
    train_records.extend(cat_c_train)
    valid_records.extend(cat_c_valid)
    cat_counts["cat_c_rag_synthesis"] = {"train": len(cat_c_train), "valid": len(cat_c_valid)}

    # Category D — Grounded RAG Hard-Negative & Near-Miss Abstention (160 items at N=300: 130 train / 30 valid)
    d_near_total = 80 if n_papers == 300 else max(1, round(n_papers * (80 / 300)))
    d_ooc_total = 80 if n_papers == 300 else max(1, round(n_papers * (80 / 300)))

    d_near_order = list(range(n_papers))
    rng.shuffle(d_near_order)
    d_ooc_order = list(range(n_papers))
    rng.shuffle(d_ooc_order)

    d_near_examples: list[dict[str, Any]] = []
    for idx in d_near_order[: min(d_near_total, n_papers)]:
        paper = papers[idx]
        pid = str(paper["id"])
        d = distilled_by_id[pid]
        chunks = [_to_chunk(paper)] + [_to_chunk(nb) for nb in neighbors[pid][:3]]
        user_msg = f"Retrieved Abstracts:\n{format_chunks(chunks)}\n\nQuestion: {d['near_miss_prompt']}"
        d_near_examples.append(_chat_example(RAG_SYSTEM, user_msg, RAG_ABSTENTION))

    d_ooc_examples: list[dict[str, Any]] = []
    for idx in d_ooc_order[: min(d_ooc_total, n_papers)]:
        paper = papers[idx]
        pid = str(paper["id"])
        d = distilled_by_id[pid]
        ooc_chunks = [_to_chunk(nb) for nb in neighbors[pid][:4]]
        user_msg = f"Retrieved Abstracts:\n{format_chunks(ooc_chunks)}\n\nQuestion: {d['q1_prompt']}"
        d_ooc_examples.append(_chat_example(RAG_SYSTEM, user_msg, RAG_ABSTENTION))

    d_near_train_n, _ = _split_counts(len(d_near_examples), 65, 15)
    d_ooc_train_n, _ = _split_counts(len(d_ooc_examples), 65, 15)
    cat_d_train = d_near_examples[:d_near_train_n] + d_ooc_examples[:d_ooc_train_n]
    cat_d_valid = d_near_examples[d_near_train_n:] + d_ooc_examples[d_ooc_train_n:]
    train_records.extend(cat_d_train)
    valid_records.extend(cat_d_valid)
    cat_counts["cat_d_rag_abstention"] = {"train": len(cat_d_train), "valid": len(cat_d_valid)}

    rng.shuffle(train_records)
    rng.shuffle(valid_records)

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    train_path = write_jsonl(out_root / "train.jsonl", train_records)
    valid_path = write_jsonl(out_root / "valid.jsonl", valid_records)

    dates = [_paper_date(p) for p in papers]
    manifest = {
        "static_path": str(static_path),
        "distilled_path": str(distilled_path),
        "static_count": len(papers),
        "static_date_min": min(dates),
        "static_date_max": max(dates),
        "cutoff_date": CUTOFF_DATE,
        "embedding_model": embedding_model,
        "seed": seed,
        "train_count": len(train_records),
        "valid_count": len(valid_records),
        "total_count": len(train_records) + len(valid_records),
        "categories": cat_counts,
        "checksums": {
            "train.jsonl": _sha256_file(train_path),
            "valid.jsonl": _sha256_file(valid_path),
        },
    }
    manifest_path = out_root / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def prepare_qlora(
    static_path: str | Path = "corpus/static.jsonl",
    out_dir: str | Path = "data/qlora",
    *,
    model: str = "qwen3.5:9b",
    max_workers: int = 4,
    seed: int = 42,
    skip_distill: bool = False,
) -> dict[str, Any]:
    """Run teacher distillation (unless skipped) and assemble QLoRA SFT splits."""
    out_root = Path(out_dir)
    distilled_path = out_root / "raw_distilled_static.jsonl"
    if not skip_distill:
        distill_static_corpus(
            static_path=static_path,
            out_path=distilled_path,
            model=model,
            max_workers=max_workers,
        )
    return assemble_qlora_splits(
        static_path=static_path,
        distilled_path=distilled_path,
        out_dir=out_root,
        seed=seed,
    )


def main() -> None:
    """CLI entrypoint for QLoRA distillation and SFT dataset preparation."""
    parser = argparse.ArgumentParser(description="Prepare QLoRA distillation and SFT splits.")
    parser.add_argument("--static-path", default="corpus/static.jsonl")
    parser.add_argument("--out-dir", default="data/qlora")
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-distill", action="store_true")
    args = parser.parse_args()

    manifest = prepare_qlora(
        static_path=args.static_path,
        out_dir=args.out_dir,
        model=args.model,
        max_workers=args.max_workers,
        seed=args.seed,
        skip_distill=args.skip_distill,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

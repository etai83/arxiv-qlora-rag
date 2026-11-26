"""Unit and invariant tests for the temporal RAG evaluation pipeline."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from src import extract_arxiv_ids, read_jsonl, write_jsonl
from src.judge import normalize_verdict
from src.metrics import compute_agreement, compute_summary, export_blind_sample
from src.retriever import retrieve

ROOT = Path(__file__).resolve().parent.parent
STATIC_CORPUS = ROOT / "corpus" / "static.jsonl"
LIVE_CORPUS = ROOT / "corpus" / "live.jsonl"
GOLDEN_PATH = ROOT / "eval" / "golden.jsonl"
FAISS_INDEX = ROOT / "data" / "index" / "live.faiss"


def test_corpus_invariants() -> None:
    """Verify static/live corpus sizes, date boundaries, zero overlap, and cs.CL category."""
    if not STATIC_CORPUS.exists() or not LIVE_CORPUS.exists():
        pytest.skip("corpus/static.jsonl or corpus/live.jsonl not generated yet")

    static_rows = read_jsonl(STATIC_CORPUS)
    live_rows = read_jsonl(LIVE_CORPUS)

    assert len(static_rows) == 300
    assert len(live_rows) == 200

    static_ids = {r["id"] for r in static_rows}
    live_ids = {r["id"] for r in live_rows}
    assert len(static_ids & live_ids) == 0

    for row in static_rows:
        date = row.get("date") or row["published_date"]
        assert date < "2025-01-01"
        assert "cs.CL" in row["categories"]

    for row in live_rows:
        date = row.get("date") or row["published_date"]
        assert date >= "2025-01-01"
        assert "cs.CL" in row["categories"]


def test_golden_set_invariants() -> None:
    """Verify 60-item golden set slice counts and valid source_paper_ids."""
    if not GOLDEN_PATH.exists():
        pytest.skip("eval/golden.jsonl not generated yet")

    golden = read_jsonl(GOLDEN_PATH)
    assert len(golden) == 60

    counts = Counter(item["slice"] for item in golden)
    assert counts == {
        "static": 20,
        "live": 20,
        "temporal_override": 10,
        "unanswerable": 10,
    }

    corpus_ids: set[str] = set()
    if STATIC_CORPUS.exists() and LIVE_CORPUS.exists():
        corpus_ids = {r["id"] for r in read_jsonl(STATIC_CORPUS) + read_jsonl(LIVE_CORPUS)}

    for item in golden:
        if item["slice"] in {"static", "live", "temporal_override"}:
            source_ids = item.get("source_paper_ids", [])
            assert len(source_ids) >= 1
            if corpus_ids:
                assert all(pid in corpus_ids for pid in source_ids)


def test_retriever_queries() -> None:
    """Verify top-k retrieval and temporal `after` filtering on the built FAISS index."""
    if not FAISS_INDEX.exists():
        pytest.skip("data/index/live.faiss not built yet")

    hits = retrieve("large language model reasoning", k=4, index_dir=FAISS_INDEX.parent)
    assert len(hits) == 4
    expected_keys = {"id", "title", "abstract", "published_date", "categories", "score"}
    for hit in hits:
        assert expected_keys.issubset(hit.keys())

    after_hits = retrieve(
        "large language model reasoning",
        k=4,
        after="2026-01-01",
        index_dir=FAISS_INDEX.parent,
    )
    assert len(after_hits) <= 4
    for hit in after_hits:
        assert hit["published_date"] >= "2026-01-01"


def test_export_blind_sample_stratified_and_blind(tmp_path: Path) -> None:
    """Verify blind sampler selects 10+10 stratified rows and strips all judge fields."""
    slices = (
        ["static"] * 20
        + ["live"] * 20
        + ["temporal_override"] * 10
        + ["unanswerable"] * 10
    )
    zs_rows = []
    rag_rows = []
    for idx, sl in enumerate(slices):
        base = {
            "id": f"q_{idx:02d}",
            "slice": sl,
            "prompt": f"Question {idx}",
            "reference_answer": f"Ref {idx}",
            "source_paper_ids": ["2501.00001"] if sl != "unanswerable" else [],
            "newer_paper_id": "2503.00002" if sl == "temporal_override" else None,
            "model_answer": f"Answer {idx}",
            "extracted_cited_ids": ["2501.00001"],
            "correctness": 5,
            "faithfulness": 5,
            "citation_accuracy": 5,
            "override_obedience": 1 if sl == "temporal_override" else None,
            "rationale": "secret judge note",
        }
        zs_rows.append({**base, "mode": "zero_shot", "retrieved_context": None})
        rag_rows.append({**base, "mode": "rag", "retrieved_context": "[ID: 2501.00001] ..."})

    zs_path = write_jsonl(tmp_path / "zs.jsonl", zs_rows)
    rag_path = write_jsonl(tmp_path / "rag.jsonl", rag_rows)
    out_path = tmp_path / "blind.jsonl"

    export_blind_sample(zs_path, rag_path, out_path=out_path, seed=42)
    sampled = read_jsonl(out_path)

    assert len(sampled) == 20
    assert Counter((r["mode"], r["slice"]) for r in sampled) == {
        ("zero_shot", "static"): 3,
        ("zero_shot", "live"): 3,
        ("zero_shot", "temporal_override"): 2,
        ("zero_shot", "unanswerable"): 2,
        ("rag", "static"): 3,
        ("rag", "live"): 3,
        ("rag", "temporal_override"): 2,
        ("rag", "unanswerable"): 2,
    }
    forbidden = {"correctness", "faithfulness", "citation_accuracy", "override_obedience", "rationale"}
    for row in sampled:
        assert forbidden.isdisjoint(row.keys())
        assert row["sample_id"].startswith("sample_")


def test_compute_agreement_and_summary(tmp_path: Path) -> None:
    """Verify exact, within-1, bucket, pooled, and override agreement calculations."""
    judge_rows = [
        {
            "id": "q1",
            "mode": "rag",
            "slice": "live",
            "correctness": 5,       # exact (human 5), high
            "faithfulness": 4,      # within-1 (human 5), high -> bucket match
            "citation_accuracy": 3, # within-1 (human 4), mid vs high -> bucket miss
            "override_obedience": None,
            "latency_ms": 120.0,
        },
        {
            "id": "q2",
            "mode": "rag",
            "slice": "temporal_override",
            "correctness": 2,       # within-1 (human 1), low -> bucket match
            "faithfulness": 2,      # miss >1 (human 4), low vs high -> bucket miss
            "citation_accuracy": 1, # exact (human 1), low -> bucket match
            "override_obedience": 1,
            "latency_ms": 180.0,
        },
    ]
    human_rows = [
        {
            "id": "q1",
            "mode": "rag",
            "slice": "live",
            "correctness": 5,
            "faithfulness": 5,
            "citation_accuracy": 4,
            "override_obedience": None,
        },
        {
            "id": "q2",
            "mode": "rag",
            "slice": "temporal_override",
            "correctness": 1,
            "faithfulness": 4,
            "citation_accuracy": 1,
            "override_obedience": 0,
        },
    ]

    judge_path = write_jsonl(tmp_path / "judge.jsonl", judge_rows)
    human_path = write_jsonl(tmp_path / "human.jsonl", human_rows)

    agreement = compute_agreement(judge_path, human_path)
    assert agreement["n_items"] == 2
    assert agreement["pooled_n"] == 6
    # Pooled: exact = 2/6 (33.33%), within_1 = 5/6 (83.33%), bucket = 4/6 (66.67%)
    assert agreement["pooled"]["exact_match_pct"] == 33.33
    assert agreement["pooled"]["within_1_pct"] == 83.33
    assert agreement["pooled"]["bucket_match_pct"] == 66.67
    assert agreement["override_exact_pct"] == 0.0
    assert agreement["dimensions"]["correctness"]["exact_match_pct"] == 50.0
    assert agreement["dimensions"]["correctness"]["within_1_pct"] == 100.0
    assert agreement["dimensions"]["correctness"]["bucket_match_pct"] == 100.0

    summary = compute_summary(judge_path=judge_path, zs_path=None, rag_path=None)
    assert summary["total_scored"] == 2
    assert summary["overall"]["rag"]["correctness"] == 3.5
    assert summary["overall"]["rag"]["override_obedience"] == 1.0
    assert summary["overall"]["rag"]["latency_mean_ms"] == 150.0


def test_helpers_and_verdict_normalization() -> None:
    """Verify arXiv ID regex extraction and judge verdict clamping/slice enforcement."""
    ids = extract_arxiv_ids("See [2503.01234] and 2411.12345 and [2503.01234].")
    assert ids == ["2503.01234", "2411.12345"]

    v_non_override = normalize_verdict(
        {
            "correctness": 9,
            "faithfulness": 0,
            "citation_accuracy": 4,
            "override_obedience": 1,
            "rationale": "clamped",
        },
        "live",
    )
    assert v_non_override.correctness == 5
    assert v_non_override.faithfulness == 1
    assert v_non_override.override_obedience is None

    v_override = normalize_verdict(
        {
            "correctness": 4,
            "faithfulness": 5,
            "citation_accuracy": 5,
            "override_obedience": 1,
            "rationale": "adopted newer paper",
        },
        "temporal_override",
    )
    assert v_override.override_obedience == 1


def test_build_experiment_book(tmp_path: Path) -> None:
    """Verify experiment_book.html builds cleanly with all 8 book pages and embedded payload."""
    from src.build_book import build_book

    if not (ROOT / "eval" / "judge_scores.jsonl").exists():
        pytest.skip("eval/judge_scores.jsonl not generated yet")

    out = build_book(root=ROOT, out_path=tmp_path / "experiment_book.html")
    html = out.read_text(encoding="utf-8")
    assert '<meta name="color-scheme" content="light dark">' in html
    assert "document.startViewTransition" in html
    assert "plotly-2.35.2.min.js" in html
    for page_idx in range(8):
        assert f'data-page="{page_idx}"' in html

    start = html.index('<script id="experiment-data" type="application/json">') + len(
        '<script id="experiment-data" type="application/json">'
    )
    end = html.index("</script>", start)
    payload = json.loads(html[start:end].replace("<\\/", "</"))
    assert len(payload["runs"]) == 60
    assert len(payload["calibration"]["audit_items"]) == 20
    assert payload["split"]["id_overlap_count"] == 0

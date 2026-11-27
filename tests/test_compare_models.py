"""Unit tests for the statistical model comparison engine (`src.compare_models`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src import write_jsonl
from src.compare_models import (
    HYBRID_SIMULATION_NOTE,
    compare_models,
    write_model_comparison,
)


def _make_judge_row(
    item_id: str,
    mode: str,
    slice_name: str,
    *,
    correctness: int,
    faithfulness: int = 5,
    citation_accuracy: int = 5,
    override_obedience: int | None = None,
    model_answer: str = "Grounded answer [2501.00001].",
    extracted_cited_ids: list[str] | None = None,
    latency_ms: float | None = 120.0,
    rationale: str = "Synthetic rationale.",
) -> dict[str, Any]:
    """Build a synthetic judged output row matching the evaluation schema."""
    if extracted_cited_ids is None:
        extracted_cited_ids = ["2501.00001"] if "[2501.00001]" in model_answer else []
    return {
        "id": item_id,
        "mode": mode,
        "slice": slice_name,
        "prompt": f"Prompt for {item_id}",
        "reference_answer": f"Reference for {item_id}",
        "source_paper_ids": ["2501.00001"] if slice_name != "unanswerable" else [],
        "newer_paper_id": "2503.00002" if slice_name == "temporal_override" else None,
        "retrieved_context": "[ID: 2501.00001] Abstract" if mode == "rag" else None,
        "valid_context_ids": ["2501.00001"] if mode == "rag" else [],
        "model_answer": model_answer,
        "extracted_cited_ids": extracted_cited_ids,
        "latency_ms": latency_ms,
        "rationale": rationale,
        "correctness": correctness,
        "faithfulness": faithfulness,
        "citation_accuracy": citation_accuracy,
        "override_obedience": (
            override_obedience if slice_name == "temporal_override" else None
        ),
    }


def _build_synthetic_dataset(
    tmp_path: Path,
) -> dict[str, Path]:
    """Create balanced 60-prompt (120-item) synthetic Base vs. FT fixtures."""
    slice_spec = (
        [("static", 20)]
        + [("live", 20)]
        + [("temporal_override", 10)]
        + [("unanswerable", 10)]
    )
    items: list[tuple[str, str]] = []
    idx = 1
    for sl, count in slice_spec:
        for _ in range(count):
            items.append((f"g{idx:02d}", sl))
            idx += 1

    base_primary: list[dict[str, Any]] = []
    ft_primary: list[dict[str, Any]] = []
    base_secondary: list[dict[str, Any]] = []
    ft_secondary: list[dict[str, Any]] = []
    base_zs_out: list[dict[str, Any]] = []
    base_rag_out: list[dict[str, Any]] = []
    ft_zs_out: list[dict[str, Any]] = []
    ft_rag_out: list[dict[str, Any]] = []

    for pos, (item_id, sl) in enumerate(items):
        # Zero-shot rows
        if sl == "static":
            b_zs_ans = "I don't have verified information on this exact metric."
            f_zs_ans = "LoRA freezes weights and injects rank-decomposed adapters [2106.09685]."
            b_zs_corr, f_zs_corr = 1, 5
            b_zs_sec, f_zs_sec = 2, 5
        elif sl == "live":
            b_zs_ans = "My knowledge cutoff prevents me from answering."
            f_zs_ans = "I don't have verified information about this 2025 paper."
            b_zs_corr, f_zs_corr = 1, 1
            b_zs_sec, f_zs_sec = 1, 1
        elif sl == "temporal_override":
            b_zs_ans = "Older pre-2025 method achieves 70% accuracy."
            f_zs_ans = "Newer post-2025 method achieves 90% accuracy."
            b_zs_corr, f_zs_corr = 2, 4
            b_zs_sec, f_zs_sec = 2, 4
        else:  # unanswerable
            b_zs_ans = "I do not have information about this fictional system."
            f_zs_ans = "I don't have verified information on this topic."
            b_zs_corr, f_zs_corr = 5, 5
            b_zs_sec, f_zs_sec = 5, 5

        # Omit latency on first zero-shot row to verify backfill from output JSONL.
        zs_lat_b = None if pos == 0 else 200.0
        zs_lat_f = None if pos == 0 else 150.0

        base_primary.append(
            _make_judge_row(
                item_id,
                "zero_shot",
                sl,
                correctness=b_zs_corr,
                faithfulness=5 if sl != "temporal_override" else 1,
                citation_accuracy=5 if sl != "temporal_override" else 1,
                override_obedience=0 if sl == "temporal_override" else None,
                model_answer=b_zs_ans,
                extracted_cited_ids=[],
                latency_ms=zs_lat_b,
            )
        )
        ft_primary.append(
            _make_judge_row(
                item_id,
                "zero_shot",
                sl,
                correctness=f_zs_corr,
                faithfulness=5 if sl in {"live", "unanswerable"} else 1,
                citation_accuracy=5 if sl in {"live", "unanswerable"} else 1,
                override_obedience=1 if sl == "temporal_override" else None,
                model_answer=f_zs_ans,
                extracted_cited_ids=["2106.09685"] if sl == "static" else [],
                latency_ms=zs_lat_f,
            )
        )
        base_secondary.append(
            _make_judge_row(
                item_id,
                "zero_shot",
                sl,
                correctness=b_zs_sec,
                override_obedience=0 if sl == "temporal_override" else None,
                model_answer=b_zs_ans,
                latency_ms=200.0,
            )
        )
        ft_secondary.append(
            _make_judge_row(
                item_id,
                "zero_shot",
                sl,
                correctness=f_zs_sec,
                override_obedience=1 if sl == "temporal_override" else None,
                model_answer=f_zs_ans,
                latency_ms=150.0,
            )
        )
        base_zs_out.append({"id": item_id, "mode": "zero_shot", "latency_ms": 210.0})
        ft_zs_out.append({"id": item_id, "mode": "zero_shot", "latency_ms": 140.0})

        # RAG rows
        if sl == "static":
            b_rag_ans = "Not in the sources."
            f_rag_ans = "Not in the sources."
            b_rag_corr, f_rag_corr = 1, 1
            b_rag_sec, f_rag_sec = 1, 1
            b_ov, f_ov = None, None
        elif sl == "live":
            # Include spurious trailing abstention on Base for first 2 live items, 0 for FT.
            if pos in (20, 21):
                b_rag_ans = (
                    "The retrieved paper evaluates multilingual models across 208 districts "
                    "and 22 languages with strong recall [2501.00001].\n\nNot in the sources."
                )
                b_rag_corr = 3
                b_rag_sec = 3
            else:
                b_rag_ans = "The paper reports improved accuracy across benchmarks [2501.00001]."
                b_rag_corr = 4
                b_rag_sec = 4
            f_rag_ans = "The paper reports improved accuracy across benchmarks [2501.00001]."
            f_rag_corr = 5
            f_rag_sec = 5
            b_ov, f_ov = None, None
        elif sl == "temporal_override":
            b_rag_ans = "Pre-T method remains standard despite newer results [2501.00001]."
            f_rag_ans = "Post-T method supersedes prior work [2503.00002]."
            b_rag_corr, f_rag_corr = 2, 5
            b_rag_sec, f_rag_sec = 3, 5
            b_ov, f_ov = 0, 1
        else:  # unanswerable
            # Simulate 1 regression on unanswerable RAG to verify regressed tally.
            if pos == 59:
                b_rag_ans = "Not in the sources."
                f_rag_ans = "It improves throughput by 2x [2501.00001]."
                b_rag_corr, f_rag_corr = 5, 1
                b_rag_sec, f_rag_sec = 5, 2
            else:
                b_rag_ans = "Not in the sources."
                f_rag_ans = "Not in the sources."
                b_rag_corr, f_rag_corr = 5, 5
                b_rag_sec, f_rag_sec = 5, 5
            b_ov, f_ov = None, None

        base_primary.append(
            _make_judge_row(
                item_id,
                "rag",
                sl,
                correctness=b_rag_corr,
                faithfulness=5,
                citation_accuracy=5,
                override_obedience=b_ov,
                model_answer=b_rag_ans,
                latency_ms=300.0,
            )
        )
        ft_primary.append(
            _make_judge_row(
                item_id,
                "rag",
                sl,
                correctness=f_rag_corr,
                faithfulness=5,
                citation_accuracy=5,
                override_obedience=f_ov,
                model_answer=f_rag_ans,
                latency_ms=260.0,
            )
        )
        base_secondary.append(
            _make_judge_row(
                item_id,
                "rag",
                sl,
                correctness=b_rag_sec,
                override_obedience=b_ov,
                model_answer=b_rag_ans,
                latency_ms=300.0,
            )
        )
        ft_secondary.append(
            _make_judge_row(
                item_id,
                "rag",
                sl,
                correctness=f_rag_sec,
                override_obedience=f_ov,
                model_answer=f_rag_ans,
                latency_ms=260.0,
            )
        )
        base_rag_out.append({"id": item_id, "mode": "rag", "latency_ms": 300.0})
        ft_rag_out.append({"id": item_id, "mode": "rag", "latency_ms": 260.0})

    manifest_path = tmp_path / "train_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "base_model": "meta-llama/Llama-3.2-3B-Instruct",
                "lora_rank": 16,
                "epochs": 3,
                "train_examples": 300,
            }
        ),
        encoding="utf-8",
    )
    log_path = write_jsonl(
        tmp_path / "training_log.jsonl",
        [
            {"step": 10, "loss": 1.85, "val_loss": 1.72},
            {"step": 50, "loss": 0.94, "val_loss": 1.05},
        ],
    )

    return {
        "base_primary": write_jsonl(tmp_path / "base_primary.jsonl", base_primary),
        "ft_primary": write_jsonl(tmp_path / "ft_primary.jsonl", ft_primary),
        "base_secondary": write_jsonl(tmp_path / "base_secondary.jsonl", base_secondary),
        "ft_secondary": write_jsonl(tmp_path / "ft_secondary.jsonl", ft_secondary),
        "base_zs_out": write_jsonl(tmp_path / "base_zs.jsonl", base_zs_out),
        "base_rag_out": write_jsonl(tmp_path / "base_rag.jsonl", base_rag_out),
        "ft_zs_out": write_jsonl(tmp_path / "ft_zs.jsonl", ft_zs_out),
        "ft_rag_out": write_jsonl(tmp_path / "ft_rag.jsonl", ft_rag_out),
        "manifest": manifest_path,
        "log": log_path,
    }


def test_compare_models_synthetic_full(tmp_path: Path) -> None:
    """Verify deltas, bootstrap CIs, win/loss, behavior shifts, cross-judge, and telemetry."""
    paths = _build_synthetic_dataset(tmp_path)

    report = compare_models(
        base_judge_path=paths["base_primary"],
        ft_judge_path=paths["ft_primary"],
        secondary_base_judge_path=paths["base_secondary"],
        secondary_ft_judge_path=paths["ft_secondary"],
        base_zs_path=paths["base_zs_out"],
        base_rag_path=paths["base_rag_out"],
        ft_zs_path=paths["ft_zs_out"],
        ft_rag_path=paths["ft_rag_out"],
        train_manifest_path=paths["manifest"],
        training_log_path=paths["log"],
        index_dir=tmp_path / "missing_index",
    )

    # 1. Score comparison & latency backfill
    sc = report["score_comparison"]
    assert sc["base"]["overall"]["pooled_all"]["count"] == 120
    assert sc["ft"]["overall"]["pooled_all"]["count"] == 120
    assert sc["overall_delta"]["zero_shot"]["correctness"] == pytest.approx(1.667, abs=1e-3)
    assert sc["overall_delta"]["rag"]["correctness"] == pytest.approx(0.8, abs=1e-3)
    assert sc["overall_delta"]["pooled_all"]["correctness"] > 1.0
    assert sc["by_mode_slice_delta"]["zero_shot"]["static"]["correctness"] == 4.0
    assert sc["by_mode_slice_delta"]["rag"]["temporal_override"]["override_obedience"] == 1.0
    assert sc["overall_delta"]["zero_shot"]["latency_mean_ms"] < 0

    # 2. Paired bootstrap CIs
    cis = report["bootstrap_cis"]
    expected_buckets = {
        "pooled_all",
        "zero_shot_all",
        "rag_all",
        "zero_shot_static",
        "zero_shot_live",
        "zero_shot_temporal_override",
        "zero_shot_unanswerable",
        "rag_static",
        "rag_live",
        "rag_temporal_override",
        "rag_unanswerable",
        "rag_temporal_override_obedience",
    }
    assert expected_buckets.issubset(cis.keys())
    assert cis["pooled_all"]["n"] == 120
    assert cis["pooled_all"]["statistically_significant"] is True
    assert cis["pooled_all"]["ci95"][0] > 0
    assert cis["zero_shot_static"]["ci95"] == [4.0, 4.0]
    assert cis["zero_shot_static"]["statistically_significant"] is True
    assert cis["rag_temporal_override_obedience"]["delta_mean"] == 1.0
    assert cis["rag_temporal_override_obedience"]["statistically_significant"] is True

    # 3. Win / tie / loss & paired_items
    wl = report["win_loss"]
    # Zero-shot improved: 20 static + 10 temporal_override = 30
    # RAG improved: 20 live + 10 temporal_override = 30; regressed: 1 unanswerable
    assert wl["overall"] == {
        "n": 120,
        "improved": 60,
        "tied": 59,
        "regressed": 1,
        "net_wins": 59,
    }
    assert wl["by_mode"]["zero_shot"]["improved"] == 30
    assert wl["by_mode"]["rag"]["regressed"] == 1
    assert wl["by_mode_slice"]["rag"]["unanswerable"]["regressed"] == 1

    paired_items = report["paired_items"]
    assert len(paired_items) == 120
    first_item = paired_items[0]
    assert first_item["mode"] == "rag"
    assert first_item["id"] == "g01"
    assert len(first_item["base_answer_preview"]) <= 300
    assert len(first_item["ft_answer_preview"]) <= 300
    # Verify latency backfill on g01 zero_shot
    zs_g01 = next(
        p for p in paired_items if p["id"] == "g01" and p["mode"] == "zero_shot"
    )
    assert zs_g01["base_latency_ms"] == 210.0
    assert zs_g01["ft_latency_ms"] == 140.0

    # 4. Behavioral shift & spurious trailing abstention
    beh = report["behavior_shift"]
    assert beh["by_mode_slice"]["zero_shot"]["static"]["base"]["abstained"] == 20
    assert beh["by_mode_slice"]["zero_shot"]["static"]["ft"]["abstained"] == 0
    assert beh["by_mode_slice"]["zero_shot"]["static"]["ft"]["cited"] == 20
    assert beh["by_mode_slice"]["zero_shot"]["live"]["ft"]["abstained"] == 20
    assert (
        beh["by_mode_slice"]["rag"]["live"]["base"]["spurious_trailing_abstention"]
        == 2
    )
    assert (
        beh["by_mode_slice"]["rag"]["live"]["ft"]["spurious_trailing_abstention"] == 0
    )

    # 5. Dual-judge cross-check
    cj = report["cross_judge_check"]
    assert cj["available"] is True
    assert cj["secondary_judge_label"] == "qwen2.5:7b"
    assert cj["direction_agrees"] is True
    assert cj["agreement_by_bucket"]["pooled_all"]["direction_agrees"] is True

    # 6. Training telemetry & recommendation
    trn = report["training"]
    assert trn["manifest_available"] is True
    assert trn["log_available"] is True
    assert trn["manifest"]["lora_rank"] == 16
    assert trn["log_summary"]["initial_loss"] == 1.85
    assert trn["log_summary"]["final_loss"] == 0.94

    rec = report["recommendation"]
    assert rec["winner"] == "llama3.2-qlora:3b"
    assert "llama3.2-qlora:3b" in rec["verdict_headline"]
    assert len(rec["summary_bullets"]) >= 5


def test_compare_models_hybrid_routing_with_precomputed_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify threshold routing math across tau values and mocked Retriever fallback."""
    # Small 4-item (1 per slice) dataset for exact hand-calculable routing math.
    rows_base: list[dict[str, Any]] = []
    rows_ft: list[dict[str, Any]] = []
    specs = [
        # (id, slice, top1_score, base_zs, ft_zs, base_rag, ft_rag)
        ("q1", "static", 0.50, 1, 5, 1, 1),  # never routed to RAG -> stays ZS
        ("q2", "live", 0.69, 1, 1, 4, 5),  # routed to RAG when tau <= 0.68
        ("q3", "temporal_override", 0.62, 2, 4, 3, 5),  # RAG at tau<=0.60, ZS at 0.68
        ("q4", "unanswerable", 0.76, 5, 5, 5, 5),  # always routed to RAG
    ]
    for item_id, sl, _, b_zs, f_zs, b_rag, f_rag in specs:
        rows_base.append(
            _make_judge_row(
                item_id,
                "zero_shot",
                sl,
                correctness=b_zs,
                override_obedience=0 if sl == "temporal_override" else None,
            )
        )
        rows_base.append(
            _make_judge_row(
                item_id,
                "rag",
                sl,
                correctness=b_rag,
                override_obedience=0 if sl == "temporal_override" else None,
            )
        )
        rows_ft.append(
            _make_judge_row(
                item_id,
                "zero_shot",
                sl,
                correctness=f_zs,
                override_obedience=0 if sl == "temporal_override" else None,
            )
        )
        rows_ft.append(
            _make_judge_row(
                item_id,
                "rag",
                sl,
                correctness=f_rag,
                override_obedience=1 if sl == "temporal_override" else None,
            )
        )

    base_p = write_jsonl(tmp_path / "base.jsonl", rows_base)
    ft_p = write_jsonl(tmp_path / "ft.jsonl", rows_ft)
    precomputed = {item_id: score for item_id, _, score, *_ in specs}

    report = compare_models(
        base_judge_path=base_p,
        ft_judge_path=ft_p,
        secondary_base_judge_path=None,
        secondary_ft_judge_path=None,
        base_zs_path=None,
        base_rag_path=None,
        ft_zs_path=None,
        ft_rag_path=None,
        train_manifest_path=None,
        training_log_path=None,
        index_dir=None,
        precomputed_top1_scores=precomputed,
    )

    hyb = report["hybrid_routing"]
    assert hyb["available"] is True
    assert hyb["simulation_note"] == HYBRID_SIMULATION_NOTE

    # At tau=0.60: q2 (0.69), q3 (0.62), q4 (0.76) -> RAG (3/4 = 75%); q1 -> ZS.
    t60 = hyb["by_tau"]["0.60"]
    assert t60["base"]["overall"]["rag_routed_count"] == 3
    assert t60["base"]["overall"]["rag_routed_pct"] == 75.0
    assert t60["base"]["overall"]["correctness"] == round((1 + 4 + 3 + 5) / 4, 3)
    assert t60["ft"]["overall"]["correctness"] == round((5 + 5 + 5 + 5) / 4, 3)

    # At tau=0.68: q2 (0.69), q4 (0.76) -> RAG (2/4 = 50%); q1, q3 -> ZS.
    t68 = hyb["by_tau"]["0.68"]
    assert t68["base"]["overall"]["rag_routed_count"] == 2
    assert t68["base"]["overall"]["correctness"] == round((1 + 4 + 2 + 5) / 4, 3)
    assert t68["ft"]["overall"]["correctness"] == round((5 + 5 + 4 + 5) / 4, 3)
    assert t68["delta"]["overall"]["correctness"] == round(4.75 - 3.0, 3)

    # At tau=0.75: only q4 (0.76) -> RAG (1/4 = 25%).
    t75 = hyb["by_tau"]["0.75"]
    assert t75["base"]["overall"]["rag_routed_count"] == 1
    assert t75["ft"]["overall"]["correctness"] == round((5 + 1 + 4 + 5) / 4, 3)

    # Also verify Retriever path via monkeypatch when precomputed_top1_scores is None.
    fake_idx_dir = tmp_path / "fake_idx"
    fake_idx_dir.mkdir()
    (fake_idx_dir / "live.faiss").write_bytes(b"fake")

    prompt_to_score = {f"Prompt for {item_id}": score for item_id, _, score, *_ in specs}

    class _FakeRetriever:
        def __init__(self, index_dir: str | Path) -> None:
            self.index_dir = index_dir

        def search(self, query: str, k: int = 1) -> list[dict[str, Any]]:
            return [{"score": prompt_to_score[query]}]

    import src.retriever as retriever_mod

    monkeypatch.setattr(retriever_mod, "Retriever", _FakeRetriever)
    report_via_retriever = compare_models(
        base_judge_path=base_p,
        ft_judge_path=ft_p,
        secondary_base_judge_path=None,
        secondary_ft_judge_path=None,
        base_zs_path=None,
        base_rag_path=None,
        ft_zs_path=None,
        ft_rag_path=None,
        train_manifest_path=None,
        training_log_path=None,
        index_dir=fake_idx_dir,
    )
    assert (
        report_via_retriever["hybrid_routing"]["default_comparison"]["ft"]["overall"][
            "correctness"
        ]
        == 4.75
    )


def test_write_model_comparison_outputs(tmp_path: Path) -> None:
    """Verify JSON and Markdown comparison artifacts are written with expected sections."""
    paths = _build_synthetic_dataset(tmp_path)
    out_json = tmp_path / "eval" / "model_comparison.json"
    out_md = tmp_path / "eval" / "model_comparison.md"

    written_json, written_md = write_model_comparison(
        base_judge_path=paths["base_primary"],
        ft_judge_path=paths["ft_primary"],
        secondary_base_judge_path=paths["base_secondary"],
        secondary_ft_judge_path=paths["ft_secondary"],
        base_zs_path=paths["base_zs_out"],
        base_rag_path=paths["base_rag_out"],
        ft_zs_path=paths["ft_zs_out"],
        ft_rag_path=paths["ft_rag_out"],
        train_manifest_path=paths["manifest"],
        training_log_path=paths["log"],
        index_dir=tmp_path / "missing_index",
        out_json=out_json,
        out_md=out_md,
        precomputed_top1_scores={f"g{i:02d}": 0.70 if i > 20 else 0.55 for i in range(1, 61)},
    )

    assert written_json == out_json
    assert written_md == out_md
    assert out_json.exists()
    assert out_md.exists()

    parsed = json.loads(out_json.read_text(encoding="utf-8"))
    assert parsed["labels"]["base"] == "llama3.2:3b"
    assert parsed["labels"]["ft"] == "llama3.2-qlora:3b"
    assert parsed["recommendation"]["winner"] == "llama3.2-qlora:3b"

    md_text = out_md.read_text(encoding="utf-8")
    expected_headers = (
        "# Model Comparison Report:",
        "## Executive Recommendation",
        "## 1. Score Comparison & Deltas",
        "## 2. Paired Bootstrap 95% Confidence Intervals",
        "## 3. Prompt-Level Win / Tie / Loss Matrix",
        "## 4. Behavioral Shift Analysis",
        "## 5. Dual-Judge Cross-Check",
        "## 6. Post-Hoc Similarity-Gated Hybrid Routing Simulation",
        "## 7. QLoRA Training Telemetry",
    )
    for header in expected_headers:
        assert header in md_text
    assert HYBRID_SIMULATION_NOTE in md_text

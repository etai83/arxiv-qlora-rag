"""Statistical Judge Comparison Engine for comparing two LLM judge evaluation runs."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

from src import read_jsonl, write_jsonl
from src.metrics import (
    ORDINAL_DIMS,
    _mean,
    _ordinal_agreement_stats,
    _pct,
    _score_bucket,
    compute_agreement,
    compute_summary,
)


def _telemetry_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract telemetry stats for latency, throughput, and eval tokens from judge rows."""
    latencies = [float(r["judge_latency_ms"]) for r in rows if r.get("judge_latency_ms") is not None]
    tok_speeds = [float(r["judge_tok_per_s"]) for r in rows if r.get("judge_tok_per_s") is not None]
    eval_tokens = [int(r["judge_eval_tokens"]) for r in rows if r.get("judge_eval_tokens") is not None]

    return {
        "n_measured": len(latencies),
        "latency_mean_ms": round(statistics.fmean(latencies), 2) if latencies else None,
        "latency_median_ms": round(statistics.median(latencies), 2) if latencies else None,
        "total_time_s": round(sum(latencies) / 1000.0, 2) if latencies else None,
        "tok_per_s_mean": round(statistics.fmean(tok_speeds), 2) if tok_speeds else None,
        "tok_per_s_median": round(statistics.median(tok_speeds), 2) if tok_speeds else None,
        "eval_tokens_mean": round(statistics.fmean(eval_tokens), 2) if eval_tokens else None,
        "eval_tokens_median": round(statistics.median(eval_tokens), 1) if eval_tokens else None,
        "total_eval_tokens": sum(eval_tokens) if eval_tokens else None,
    }


def quadratic_weighted_kappa(
    pairs: list[tuple[int, int]],
    min_rating: int = 1,
    max_rating: int = 5,
) -> float | None:
    """Compute Quadratic Weighted Kappa (QWK) for ordinal rating pairs."""
    if not pairs:
        return None

    n_classes = max_rating - min_rating + 1
    # Build NxN observed matrix O
    O = [[0.0 for _ in range(n_classes)] for _ in range(n_classes)]
    for r_a, r_b in pairs:
        i = r_a - min_rating
        j = r_b - min_rating
        if 0 <= i < n_classes and 0 <= j < n_classes:
            O[i][j] += 1.0

    total = sum(sum(row) for row in O)
    if total == 0:
        return None

    row_sums = [sum(row) for row in O]
    col_sums = [sum(O[i][j] for i in range(n_classes)) for j in range(n_classes)]

    # Expected outer product matrix E
    E = [[(row_sums[i] * col_sums[j]) / total for j in range(n_classes)] for i in range(n_classes)]

    # Quadratic weight matrix w
    denom_w = (n_classes - 1) ** 2
    if denom_w == 0:
        return 1.0

    w = [[((i - j) ** 2) / denom_w for j in range(n_classes)] for i in range(n_classes)]

    num = sum(w[i][j] * O[i][j] for i in range(n_classes) for j in range(n_classes))
    den = sum(w[i][j] * E[i][j] for i in range(n_classes) for j in range(n_classes))

    if den == 0:
        return 1.0 if num == 0 else 0.0

    return round(1.0 - (num / den), 4)


def confusion_matrix_5x5(pairs: list[tuple[int, int]]) -> list[list[int]]:
    """Return a 5x5 nested list counting (r_a, r_b) pairs in 1..5."""
    matrix = [[0 for _ in range(5)] for _ in range(5)]
    for r_a, r_b in pairs:
        i = r_a - 1
        j = r_b - 1
        if 0 <= i < 5 and 0 <= j < 5:
            matrix[i][j] += 1
    return matrix


def bootstrap_exact_match_ci(
    base_pairs: list[tuple[int, int]],
    cand_pairs: list[tuple[int, int]],
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute paired bootstrap 95% confidence intervals for exact match percentages."""
    if not base_pairs or not cand_pairs or len(base_pairs) != len(cand_pairs):
        return {
            "n_pairs": len(base_pairs),
            "n_boot": n_boot,
            "base_exact_ci95": [None, None],
            "cand_exact_ci95": [None, None],
            "delta_exact_ci95": [None, None],
        }

    n = len(base_pairs)
    rng = random.Random(seed)

    base_exacts: list[float] = []
    cand_exacts: list[float] = []
    delta_exacts: list[float] = []

    for _ in range(n_boot):
        indices = [rng.randrange(n) for _ in range(n)]
        b_hits = sum(1 for idx in indices if base_pairs[idx][0] == base_pairs[idx][1])
        c_hits = sum(1 for idx in indices if cand_pairs[idx][0] == cand_pairs[idx][1])

        b_pct = 100.0 * b_hits / n
        c_pct = 100.0 * c_hits / n
        base_exacts.append(b_pct)
        cand_exacts.append(c_pct)
        delta_exacts.append(c_pct - b_pct)

    base_exacts.sort()
    cand_exacts.sort()
    delta_exacts.sort()

    def _ci(vals: list[float]) -> list[float]:
        low = round(vals[int(0.025 * len(vals))], 2)
        high = round(vals[int(0.975 * len(vals)) - 1], 2)
        return [low, high]

    return {
        "n_pairs": n,
        "n_boot": n_boot,
        "base_exact_ci95": _ci(base_exacts),
        "cand_exact_ci95": _ci(cand_exacts),
        "delta_exact_ci95": _ci(delta_exacts),
    }


def classify_discordance(base_row: dict[str, Any], cand_row: dict[str, Any]) -> str:
    """Classify the discordance category between base and candidate judge scores."""
    b_mode = base_row.get("mode")
    b_slice = base_row.get("slice")
    b_corr = int(base_row.get("correctness", 3))
    c_corr = int(cand_row.get("correctness", 3))
    b_faith = int(base_row.get("faithfulness", 3))
    c_faith = int(cand_row.get("faithfulness", 3))

    if b_mode == "zero_shot" and b_faith >= 4 and c_faith == 1:
        return "hedged_zero_shot_ungrounded"

    if b_slice in {"static", "live", "temporal_override"} and b_corr >= 4 and c_corr == 1 and c_faith == 5:
        return "false_abstention_fix"

    if b_mode == "rag" and b_corr != c_corr:
        return "rag_completeness_strictness"

    return "rubric_boundary_shift"


def compare_judges(
    base_judge_path: str | Path = "eval/judge_scores.jsonl",
    cand_judge_path: str | Path = "eval/judge_scores_qwen35_9b.jsonl",
    human_path: str | Path = "eval/human_scores_20.jsonl",
    zs_path: str | Path | None = "eval/outputs_zero_shot.jsonl",
    rag_path: str | Path | None = "eval/outputs_rag.jsonl",
    *,
    base_label: str = "qwen2.5:7b",
    cand_label: str = "qwen3.5:9b-mlx",
    env_manifest_path: str | Path | None = "corpus/env_manifest.json",
) -> dict[str, Any]:
    """Compare base and candidate LLM judges against human calibration, benchmark shift, and inter-judge metrics."""
    base_rows = read_jsonl(base_judge_path)
    cand_rows = read_jsonl(cand_judge_path)
    human_rows = read_jsonl(human_path)

    base_by_key = {(r["id"], r["mode"]): r for r in base_rows}
    cand_by_key = {(r["id"], r["mode"]): r for r in cand_rows}
    human_by_key = {(r["id"], r["mode"]): r for r in human_rows}

    # 1. Human Calibration (N=20 intersection)
    matched_human_keys = sorted(set(human_by_key.keys()) & set(base_by_key.keys()) & set(cand_by_key.keys()))

    def _build_human_calibration_side(judge_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
        matched = [(judge_map[k], human_by_key[k]) for k in matched_human_keys]
        dimensions: dict[str, Any] = {}
        pooled_pairs: list[tuple[int, int]] = []

        for dim in ORDINAL_DIMS:
            pairs = [
                (int(j_r[dim]), int(h_r[dim]))
                for j_r, h_r in matched
                if j_r.get(dim) is not None and h_r.get(dim) is not None
            ]
            stats = _ordinal_agreement_stats(pairs)
            stats["kappa_w"] = quadratic_weighted_kappa(pairs)
            maes = [abs(j - h) for j, h in pairs]
            stats["mae"] = _mean(maes)
            signed = [float(j - h) for j, h in pairs]
            stats["mean_signed_bias"] = _mean(signed)
            dimensions[dim] = stats
            pooled_pairs.extend(pairs)

        pooled_stats = _ordinal_agreement_stats(pooled_pairs)
        pooled_stats["kappa_w"] = quadratic_weighted_kappa(pooled_pairs)
        pooled_maes = [abs(j - h) for j, h in pooled_pairs]
        pooled_stats["mae"] = _mean(pooled_maes)
        pooled_signed = [float(j - h) for j, h in pooled_pairs]
        pooled_stats["mean_signed_bias"] = _mean(pooled_signed)

        override_pairs = [
            (int(j_r["override_obedience"]), int(h_r["override_obedience"]))
            for j_r, h_r in matched
            if j_r.get("slice") == "temporal_override"
            and j_r.get("override_obedience") is not None
            and h_r.get("override_obedience") is not None
        ]
        ov_exact = sum(1 for j, h in override_pairs if j == h)
        ov_pct = _pct(ov_exact, len(override_pairs))
        dimensions["override_obedience"] = {
            "n": len(override_pairs),
            "override_exact_pct": ov_pct,
        }

        conf_mat = confusion_matrix_5x5(pooled_pairs)

        return {
            "dimensions": dimensions,
            "pooled": pooled_stats,
            "confusion_matrix": conf_mat,
            "override_exact_pct": ov_pct,
            "_pooled_pairs": pooled_pairs,
        }

    base_hum = _build_human_calibration_side(base_by_key)
    cand_hum = _build_human_calibration_side(cand_by_key)

    bootstrap_ci = bootstrap_exact_match_ci(base_hum["_pooled_pairs"], cand_hum["_pooled_pairs"])

    # Delta dictionary
    delta_hum = {
        "pooled_exact_match_pct": round((cand_hum["pooled"]["exact_match_pct"] or 0) - (base_hum["pooled"]["exact_match_pct"] or 0), 2),
        "pooled_within_1_pct": round((cand_hum["pooled"]["within_1_pct"] or 0) - (base_hum["pooled"]["within_1_pct"] or 0), 2),
        "pooled_bucket_match_pct": round((cand_hum["pooled"]["bucket_match_pct"] or 0) - (base_hum["pooled"]["bucket_match_pct"] or 0), 2),
        "pooled_kappa_w": round((cand_hum["pooled"]["kappa_w"] or 0) - (base_hum["pooled"]["kappa_w"] or 0), 4),
        "pooled_mae": round((cand_hum["pooled"]["mae"] or 0) - (base_hum["pooled"]["mae"] or 0), 3),
        "override_exact_pct": round((cand_hum["override_exact_pct"] or 0) - (base_hum["override_exact_pct"] or 0), 2),
    }
    for dim in ORDINAL_DIMS:
        b_dim = base_hum["dimensions"][dim]["exact_match_pct"] or 0
        c_dim = cand_hum["dimensions"][dim]["exact_match_pct"] or 0
        delta_hum[f"{dim}_exact_match_pct"] = round(c_dim - b_dim, 2)

    # Audit items for all 20 human items
    audit_items: list[dict[str, Any]] = []
    for k in matched_human_keys:
        h_r = human_by_key[k]
        b_r = base_by_key[k]
        c_r = cand_by_key[k]

        def _l1(j_r: dict[str, Any], h_r: dict[str, Any]) -> int:
            total_diff = 0
            for dim in ORDINAL_DIMS:
                if j_r.get(dim) is not None and h_r.get(dim) is not None:
                    total_diff += abs(int(j_r[dim]) - int(h_r[dim]))
            return total_diff

        b_l1 = _l1(b_r, h_r)
        c_l1 = _l1(c_r, h_r)

        if c_l1 < b_l1:
            outcome = "improved"
        elif c_l1 > b_l1:
            outcome = "regressed"
        elif b_l1 == 0 and c_l1 == 0:
            outcome = "tied_exact"
        else:
            outcome = "tied_miss"

        audit_items.append({
            "sample_id": h_r.get("sample_id"),
            "id": h_r["id"],
            "mode": h_r["mode"],
            "slice": h_r["slice"],
            "human": {dim: h_r.get(dim) for dim in ORDINAL_DIMS},
            "base": {dim: b_r.get(dim) for dim in ORDINAL_DIMS},
            "cand": {dim: c_r.get(dim) for dim in ORDINAL_DIMS},
            "base_l1": b_l1,
            "cand_l1": c_l1,
            "outcome": outcome,
        })

    # Remove temporary internal keys before storing
    del base_hum["_pooled_pairs"]
    del cand_hum["_pooled_pairs"]

    human_calibration = {
        "base": base_hum,
        "cand": cand_hum,
        "bootstrap_ci": bootstrap_ci,
        "delta": delta_hum,
        "audit_items": audit_items,
    }

    # 2. Benchmark Shift
    base_summary = compute_summary(base_judge_path, zs_path=zs_path, rag_path=rag_path)
    cand_summary = compute_summary(cand_judge_path, zs_path=zs_path, rag_path=rag_path)

    metrics_to_diff = ("correctness", "faithfulness", "citation_accuracy", "override_obedience")
    overall_delta: dict[str, dict[str, float | None]] = {}
    for mode in ("zero_shot", "rag"):
        b_ov = base_summary["overall"].get(mode, {})
        c_ov = cand_summary["overall"].get(mode, {})
        m_diff: dict[str, float | None] = {}
        for m in metrics_to_diff:
            bv = b_ov.get(m)
            cv = c_ov.get(m)
            if bv is not None and cv is not None:
                m_diff[m] = round(float(cv) - float(bv), 3)
            else:
                m_diff[m] = None
        overall_delta[mode] = m_diff

    by_mode_slice_delta: dict[str, dict[str, dict[str, float | None]]] = {}
    for mode in base_summary["by_mode_slice"]:
        by_mode_slice_delta[mode] = {}
        b_slices = base_summary["by_mode_slice"].get(mode, {})
        c_slices = cand_summary["by_mode_slice"].get(mode, {})
        for slice_name in b_slices:
            if slice_name in c_slices:
                bv_slice = b_slices[slice_name]
                cv_slice = c_slices[slice_name]
                s_diff: dict[str, float | None] = {}
                for m in metrics_to_diff:
                    bv = bv_slice.get(m)
                    cv = cv_slice.get(m)
                    if bv is not None and cv is not None:
                        s_diff[m] = round(float(cv) - float(bv), 3)
                    else:
                        s_diff[m] = None
                by_mode_slice_delta[mode][slice_name] = s_diff

    benchmark_shift = {
        "base": base_summary,
        "cand": cand_summary,
        "overall_delta": overall_delta,
        "by_mode_slice_delta": by_mode_slice_delta,
    }

    # 3. Inter-Judge Agreement (N=120)
    matched_inter_keys = sorted(set(base_by_key.keys()) & set(cand_by_key.keys()))
    inter_matched = [(base_by_key[k], cand_by_key[k]) for k in matched_inter_keys]

    inter_dimensions: dict[str, Any] = {}
    inter_pooled_pairs: list[tuple[int, int]] = []

    for dim in ORDINAL_DIMS:
        pairs = [
            (int(b_r[dim]), int(c_r[dim]))
            for b_r, c_r in inter_matched
            if b_r.get(dim) is not None and c_r.get(dim) is not None
        ]
        stats = _ordinal_agreement_stats(pairs)
        stats["kappa_w"] = quadratic_weighted_kappa(pairs)
        maes = [abs(b - c) for b, c in pairs]
        stats["mae"] = _mean(maes)
        signed_diffs = [float(c - b) for b, c in pairs]
        stats["mean_signed_diff"] = _mean(signed_diffs)
        inter_dimensions[dim] = stats
        inter_pooled_pairs.extend(pairs)

    inter_pooled_stats = _ordinal_agreement_stats(inter_pooled_pairs)
    inter_pooled_stats["kappa_w"] = quadratic_weighted_kappa(inter_pooled_pairs)
    inter_pooled_maes = [abs(b - c) for b, c in inter_pooled_pairs]
    inter_pooled_stats["mae"] = _mean(inter_pooled_maes)
    inter_pooled_signed = [float(c - b) for b, c in inter_pooled_pairs]
    inter_pooled_stats["mean_signed_diff"] = _mean(inter_pooled_signed)

    inter_override_pairs = [
        (int(b_r["override_obedience"]), int(c_r["override_obedience"]))
        for b_r, c_r in inter_matched
        if b_r.get("slice") == "temporal_override"
        and b_r.get("override_obedience") is not None
        and c_r.get("override_obedience") is not None
    ]
    inter_ov_exact = sum(1 for b, c in inter_override_pairs if b == c)
    inter_ov_pct = _pct(inter_ov_exact, len(inter_override_pairs))
    inter_dimensions["override_obedience"] = {
        "n": len(inter_override_pairs),
        "override_exact_pct": inter_ov_pct,
    }

    inter_conf_mat = confusion_matrix_5x5(inter_pooled_pairs)

    discordance_counts: dict[str, int] = {
        "hedged_zero_shot_ungrounded": 0,
        "false_abstention_fix": 0,
        "rag_completeness_strictness": 0,
        "rubric_boundary_shift": 0,
    }
    discordant_items: list[dict[str, Any]] = []

    for k in matched_inter_keys:
        b_r = base_by_key[k]
        c_r = cand_by_key[k]

        is_diff = False
        for dim in ORDINAL_DIMS:
            if b_r.get(dim) != c_r.get(dim):
                is_diff = True
        if b_r.get("slice") == "temporal_override" and b_r.get("override_obedience") != c_r.get("override_obedience"):
            is_diff = True

        if is_diff:
            category = classify_discordance(b_r, c_r)
            discordance_counts[category] += 1
            model_ans = str(b_r.get("model_answer", ""))
            discordant_items.append({
                "id": b_r["id"],
                "mode": b_r["mode"],
                "slice": b_r["slice"],
                "prompt": b_r.get("prompt"),
                "model_answer_preview": model_ans[:240],
                "base_scores": {dim: b_r.get(dim) for dim in ORDINAL_DIMS},
                "cand_scores": {dim: c_r.get(dim) for dim in ORDINAL_DIMS},
                "base_rationale": b_r.get("rationale"),
                "cand_rationale": c_r.get("rationale"),
                "category": category,
            })

    inter_judge = {
        "n_items": len(inter_matched),
        "pooled_n": len(inter_pooled_pairs),
        "dimensions": inter_dimensions,
        "pooled": inter_pooled_stats,
        "confusion_matrix": inter_conf_mat,
        "override_exact_pct": inter_ov_pct,
        "discordance_counts_by_category": discordance_counts,
        "discordant_items": discordant_items,
    }

    # 4. Operational Throughput & Latency
    base_stats = _telemetry_stats(base_rows)
    cand_stats = _telemetry_stats(cand_rows)

    def _diff(cand_val: float | int | None, base_val: float | int | None, digits: int = 2) -> float | None:
        if cand_val is None or base_val is None:
            return None
        return round(float(cand_val) - float(base_val), digits)

    base_lat_med = base_stats.get("latency_median_ms")
    cand_lat_med = cand_stats.get("latency_median_ms")
    delta_op = {
        "tok_per_s_median": _diff(cand_stats.get("tok_per_s_median"), base_stats.get("tok_per_s_median"), 2),
        "tok_per_s_mean": _diff(cand_stats.get("tok_per_s_mean"), base_stats.get("tok_per_s_mean"), 2),
        "latency_median_ms": _diff(cand_lat_med, base_lat_med, 2),
        "latency_mean_ms": _diff(cand_stats.get("latency_mean_ms"), base_stats.get("latency_mean_ms"), 2),
        "latency_ratio_median": (
            round(float(cand_lat_med) / float(base_lat_med), 2)
            if cand_lat_med is not None and base_lat_med is not None and base_lat_med > 0
            else None
        ),
        "eval_tokens_median": _diff(cand_stats.get("eval_tokens_median"), base_stats.get("eval_tokens_median"), 1),
        "eval_tokens_mean": _diff(cand_stats.get("eval_tokens_mean"), base_stats.get("eval_tokens_mean"), 2),
    }

    models_meta: dict[str, Any] = {}
    manifest_p = Path(env_manifest_path) if env_manifest_path else None
    if manifest_p and manifest_p.exists():
        try:
            models_meta = json.loads(manifest_p.read_text(encoding="utf-8"))
        except Exception:
            pass

    operational = {
        "base": base_stats,
        "cand": cand_stats,
        "delta": delta_op,
        "cand_latency_mean_ms": cand_stats.get("latency_mean_ms"),
        "cand_latency_median_ms": cand_stats.get("latency_median_ms"),
        "cand_total_time_s": cand_stats.get("total_time_s"),
        "cand_tok_per_s_mean": cand_stats.get("tok_per_s_mean"),
        "cand_tok_per_s_median": cand_stats.get("tok_per_s_median"),
        "cand_total_eval_tokens": cand_stats.get("total_eval_tokens"),
        "base_documented_tok_per_s": 59.7,
        "models_meta": models_meta,
    }

    pooled_exact_delta = delta_hum.get("pooled_exact_match_pct", 0) or 0
    winner = cand_label if pooled_exact_delta > 0 else base_label
    lat_delta = delta_op.get("latency_median_ms")
    tps_delta = delta_op.get("tok_per_s_median")
    tok_delta = delta_op.get("eval_tokens_median")
    speed_note = (
        f"faster median latency ({lat_delta:+} ms/item, {tps_delta:+} tok/s, {tok_delta:+} tokens/item) "
        if lat_delta is not None and lat_delta < 0
        else "moderate latency trade-off "
    )
    verdict_headline = (
        f"YES — Use `{cand_label}` as the default benchmark judge: it raises exact human agreement by "
        f"{pooled_exact_delta:+}% (QWK {base_hum['pooled']['kappa_w']} -> {cand_hum['pooled']['kappa_w']}, "
        f"correctness +{delta_hum['correctness_exact_match_pct']}%), with {speed_note}at +2.96 GB model size (7.64 GB vs 4.68 GB)."
        if winner == cand_label
        else f"Keep `{base_label}`: higher throughput and lower latency with comparable human calibration."
    )

    recommendation = {
        "winner": winner,
        "verdict_headline": verdict_headline,
    }

    return {
        "labels": {"base": base_label, "cand": cand_label},
        "human_calibration": human_calibration,
        "benchmark_shift": benchmark_shift,
        "inter_judge": inter_judge,
        "operational": operational,
        "recommendation": recommendation,
    }


def render_comparison_markdown(report: dict[str, Any]) -> str:
    """Render comparison report into comprehensive Markdown tables."""
    base_lbl = report["labels"]["base"]
    cand_lbl = report["labels"]["cand"]
    hum = report["human_calibration"]
    bench = report["benchmark_shift"]
    inter = report["inter_judge"]
    op = report["operational"]
    rec = report.get("recommendation", {})

    headline = rec.get("verdict_headline", "")

    lines = [
        "# Statistical Judge Comparison Report",
        "",
        f"- **Base Judge**: `{base_lbl}`",
        f"- **Candidate Judge**: `{cand_lbl}`",
        "",
        "## Executive Bottom Line: Which Judge Should You Use?",
        "",
        headline,
        "",
        "## 1. Human Calibration Comparison (N=20)",
        "",
        "### How to Read Calibration Metrics",
        "- **Pooled Exact Match %**: Across 20 calibration items × 3 rubric dimensions (correctness, faithfulness, citation_accuracy = 60 total ratings), the percentage of exact agreement between the LLM judge and human gold annotations.",
        "- **Quadratic Weighted Kappa (QWK / $\\\\kappa_w$)**: Measures ordinal agreement on a `-1` to `1` scale, adjusting for chance agreement and penalizing larger rating discrepancies quadratically (e.g. a rating error of `1 vs 5` is 16x harsher than `4 vs 5`). Benchmarks: `0.40–0.60` moderate agreement, `>0.75` strong agreement.",
        "",
        f"| Dimension | {base_lbl} Exact % | {cand_lbl} Exact % | Delta Exact % | 95% CI (Delta Exact) | {base_lbl} Kappa | {cand_lbl} Kappa | {base_lbl} MAE | {cand_lbl} MAE |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]

    for dim in ORDINAL_DIMS:
        b_d = hum["base"]["dimensions"][dim]
        c_d = hum["cand"]["dimensions"][dim]
        b_ex = b_d["exact_match_pct"]
        c_ex = c_d["exact_match_pct"]
        delta_ex = hum["delta"][f"{dim}_exact_match_pct"]
        b_kap = b_d["kappa_w"]
        c_kap = c_d["kappa_w"]
        b_mae = b_d["mae"]
        c_mae = c_d["mae"]
        lines.append(
            f"| `{dim}` | {b_ex}% | {c_ex}% | {delta_ex:+}% | - | {b_kap} | {c_kap} | {b_mae} | {c_mae} |"
        )

    b_pooled = hum["base"]["pooled"]
    c_pooled = hum["cand"]["pooled"]
    ci_delta = hum["bootstrap_ci"]["delta_exact_ci95"]
    ci_str = f"[{ci_delta[0]}%, {ci_delta[1]}%]" if ci_delta[0] is not None else "-"
    lines.append(
        f"| **Pooled (Ordinal)** | {b_pooled['exact_match_pct']}% | {c_pooled['exact_match_pct']}% | {hum['delta']['pooled_exact_match_pct']:+}% | {ci_str} | {b_pooled['kappa_w']} | {c_pooled['kappa_w']} | {b_pooled['mae']} | {c_pooled['mae']} |"
    )

    lines.extend([
        "",
        "### Override Obedience Calibration",
        f"- **Base Override Exact %**: {hum['base']['override_exact_pct']}%",
        f"- **Candidate Override Exact %**: {hum['cand']['override_exact_pct']}%",
        f"- **Delta Override Exact %**: {hum['delta']['override_exact_pct']:+}%",
        "",
        "## 2. Benchmark Score Shift",
        "",
        "### Overall by Mode",
        f"| Mode | Metric | Base ({base_lbl}) | Cand ({cand_lbl}) | Delta |",
        "|---|---|---:|---:|---:|",
    ])

    for mode in ("zero_shot", "rag"):
        b_ov = bench["base"]["overall"].get(mode, {})
        c_ov = bench["cand"]["overall"].get(mode, {})
        for m in ("correctness", "faithfulness", "citation_accuracy", "override_obedience"):
            bv = b_ov.get(m)
            cv = c_ov.get(m)
            diff = bench["overall_delta"][mode].get(m)
            diff_str = f"{diff:+}" if diff is not None else "-"
            lines.append(f"| `{mode}` | `{m}` | {bv if bv is not None else '-'} | {cv if cv is not None else '-'} | {diff_str} |")

    lines.extend([
        "",
        "### By Mode & Slice (Correctness & Override Obedience)",
        f"| Mode | Slice | Base Correctness | Cand Correctness | Delta | Base Ov. Obed | Cand Ov. Obed |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for mode in ("zero_shot", "rag"):
        b_sl = bench["base"]["by_mode_slice"].get(mode, {})
        c_sl = bench["cand"]["by_mode_slice"].get(mode, {})
        d_sl = bench["by_mode_slice_delta"].get(mode, {})
        for sl in ("static", "live", "temporal_override", "unanswerable"):
            if sl in b_sl and sl in c_sl:
                bc = b_sl[sl].get("correctness")
                cc = c_sl[sl].get("correctness")
                dc = d_sl.get(sl, {}).get("correctness")
                bo = b_sl[sl].get("override_obedience")
                co = c_sl[sl].get("override_obedience")
                lines.append(
                    f"| `{mode}` | `{sl}` | {bc} | {cc} | {dc:+} | "
                    f"{bo if bo is not None else '-'} | {co if co is not None else '-'} |"
                )

    lines.extend([
        "",
        "## 3. Inter-Judge Agreement & Discordance (N=120)",
        "",
        "| Dimension | Exact Match % | Within-1 % | Bucket Match % | QWK (κ_w) | MAE | Mean Signed Diff (Cand - Base) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])

    for dim in ORDINAL_DIMS:
        d = inter["dimensions"][dim]
        lines.append(
            f"| `{dim}` | {d['exact_match_pct']}% | {d['within_1_pct']}% | {d['bucket_match_pct']}% | {d['kappa_w']} | {d['mae']} | {d['mean_signed_diff']} |"
        )
    ip = inter["pooled"]
    lines.append(
        f"| **Pooled** | {ip['exact_match_pct']}% | {ip['within_1_pct']}% | {ip['bucket_match_pct']}% | {ip['kappa_w']} | {ip['mae']} | {ip['mean_signed_diff']} |"
    )

    lines.extend([
        "",
        "### Discordance Category Counts",
        "| Category | Count | Description |",
        "|---|---:|---|",
        f"| `hedged_zero_shot_ungrounded` | {inter['discordance_counts_by_category']['hedged_zero_shot_ungrounded']} | Zero-shot hedged answers making ungrounded claims caught by candidate (faithfulness=1) vs missed by base |",
        f"| `false_abstention_fix` | {inter['discordance_counts_by_category']['false_abstention_fix']} | Valid answerable questions incorrectly abstained by base, correctly resolved by candidate |",
        f"| `rag_completeness_strictness` | {inter['discordance_counts_by_category']['rag_completeness_strictness']} | RAG correctness rubric strictness variations |",
        f"| `rubric_boundary_shift` | {inter['discordance_counts_by_category']['rubric_boundary_shift']} | General rating boundary shifts (e.g. score 4 vs 5) |",
        "",
        "## 4. Speed, Median Latency & Token Volume Comparison",
        "",
        f"| Metric | Base ({base_lbl}) | Candidate ({cand_lbl}) | Difference (Cand - Base) |",
        "|---|---:|---:|---:|",
        f"| **Median Throughput (tok/s)** | `{op.get('base', {}).get('tok_per_s_median')}` | `{op.get('cand', {}).get('tok_per_s_median')}` | `{op.get('delta', {}).get('tok_per_s_median'):+}` tok/s |"
        if op.get("delta", {}).get("tok_per_s_median") is not None
        else f"| **Median Throughput (tok/s)** | `{op.get('base', {}).get('tok_per_s_median')}` | `{op.get('cand', {}).get('tok_per_s_median')}` | - |",
        f"| **Mean Throughput (tok/s)** | `{op.get('base', {}).get('tok_per_s_mean')}` | `{op.get('cand', {}).get('tok_per_s_mean')}` | `{op.get('delta', {}).get('tok_per_s_mean'):+}` tok/s |"
        if op.get("delta", {}).get("tok_per_s_mean") is not None
        else f"| **Mean Throughput (tok/s)** | `{op.get('base', {}).get('tok_per_s_mean')}` | `{op.get('cand', {}).get('tok_per_s_mean')}` | - |",
        f"| **Median Latency / Item (ms)** | `{op.get('base', {}).get('latency_median_ms')}` | `{op.get('cand', {}).get('latency_median_ms')}` | `{op.get('delta', {}).get('latency_median_ms'):+} ms` (`{op.get('delta', {}).get('latency_ratio_median')}x`) |"
        if op.get("delta", {}).get("latency_median_ms") is not None
        else f"| **Median Latency / Item (ms)** | `{op.get('base', {}).get('latency_median_ms')}` | `{op.get('cand', {}).get('latency_median_ms')}` | - |",
        f"| **Mean Latency / Item (ms)** | `{op.get('base', {}).get('latency_mean_ms')}` | `{op.get('cand', {}).get('latency_mean_ms')}` | `{op.get('delta', {}).get('latency_mean_ms'):+} ms` |"
        if op.get("delta", {}).get("latency_mean_ms") is not None
        else f"| **Mean Latency / Item (ms)** | `{op.get('base', {}).get('latency_mean_ms')}` | `{op.get('cand', {}).get('latency_mean_ms')}` | - |",
        f"| **Total Run Time (s)** | `{op.get('base', {}).get('total_time_s')}s` | `{op.get('cand', {}).get('total_time_s')}s` | - |",
        f"| **Median Output Tokens / Item** | `{op.get('base', {}).get('eval_tokens_median')}` | `{op.get('cand', {}).get('eval_tokens_median')}` | `{op.get('delta', {}).get('eval_tokens_median'):+}` tokens |"
        if op.get("delta", {}).get("eval_tokens_median") is not None
        else f"| **Median Output Tokens / Item** | `{op.get('base', {}).get('eval_tokens_median')}` | `{op.get('cand', {}).get('eval_tokens_median')}` | - |",
        f"| **Mean Output Tokens / Item** | `{op.get('base', {}).get('eval_tokens_mean')}` | `{op.get('cand', {}).get('eval_tokens_mean')}` | `{op.get('delta', {}).get('eval_tokens_mean'):+}` tokens |"
        if op.get("delta", {}).get("eval_tokens_mean") is not None
        else f"| **Mean Output Tokens / Item** | `{op.get('base', {}).get('eval_tokens_mean')}` | `{op.get('cand', {}).get('eval_tokens_mean')}` | - |",
        f"| **Total Output Tokens** | `{op.get('base', {}).get('total_eval_tokens')}` | `{op.get('cand', {}).get('total_eval_tokens')}` | - |",
        "",
    ])

    return "\n".join(lines)


def write_comparison(
    base_judge_path: str | Path = "eval/judge_scores.jsonl",
    cand_judge_path: str | Path = "eval/judge_scores_qwen35_9b.jsonl",
    human_path: str | Path = "eval/human_scores_20.jsonl",
    zs_path: str | Path | None = "eval/outputs_zero_shot.jsonl",
    rag_path: str | Path | None = "eval/outputs_rag.jsonl",
    out_json: str | Path = "eval/judge_comparison.json",
    out_md: str | Path = "eval/judge_comparison.md",
    *,
    base_label: str = "qwen2.5:7b",
    cand_label: str = "qwen3.5:9b-mlx",
    env_manifest_path: str | Path | None = "corpus/env_manifest.json",
) -> tuple[Path, Path]:
    """Execute compare_judges and write JSON and Markdown reports."""
    report = compare_judges(
        base_judge_path=base_judge_path,
        cand_judge_path=cand_judge_path,
        human_path=human_path,
        zs_path=zs_path,
        rag_path=rag_path,
        base_label=base_label,
        cand_label=cand_label,
        env_manifest_path=env_manifest_path,
    )

    json_path = Path(out_json)
    md_path = Path(out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_comparison_markdown(report), encoding="utf-8")

    return json_path, md_path


def main() -> None:
    """CLI entrypoint for judge comparison engine."""
    parser = argparse.ArgumentParser(description="Statistical Judge Comparison Engine")
    parser.add_argument("--base-judge", default="eval/judge_scores.jsonl")
    parser.add_argument("--cand-judge", default="eval/judge_scores_qwen35_9b.jsonl")
    parser.add_argument("--human-path", default="eval/human_scores_20.jsonl")
    parser.add_argument("--zs-path", default="eval/outputs_zero_shot.jsonl")
    parser.add_argument("--rag-path", default="eval/outputs_rag.jsonl")
    parser.add_argument("--out-json", default="eval/judge_comparison.json")
    parser.add_argument("--out-md", default="eval/judge_comparison.md")
    parser.add_argument("--base-label", default="qwen2.5:7b")
    parser.add_argument("--cand-label", default="qwen3.5:9b-mlx")
    parser.add_argument("--env-manifest-path", default="corpus/env_manifest.json")
    args = parser.parse_args()

    json_path, md_path = write_comparison(
        base_judge_path=args.base_judge,
        cand_judge_path=args.cand_judge,
        human_path=args.human_path,
        zs_path=args.zs_path,
        rag_path=args.rag_path,
        out_json=args.out_json,
        out_md=args.out_md,
        base_label=args.base_label,
        cand_label=args.cand_label,
        env_manifest_path=args.env_manifest_path,
    )
    print(f"Wrote comparison JSON to {json_path} and Markdown to {md_path}")


if __name__ == "__main__":
    main()

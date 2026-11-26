"""Evaluation metrics, blind sampling, human-judge agreement, and report generation."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

from src import read_jsonl, write_jsonl

ORDINAL_DIMS = ("correctness", "faithfulness", "citation_accuracy")
SLICE_SAMPLE_COUNTS = {
    "static": 3,
    "live": 3,
    "temporal_override": 2,
    "unanswerable": 2,
}
BLIND_FIELDS = (
    "id",
    "mode",
    "slice",
    "prompt",
    "reference_answer",
    "source_paper_ids",
    "newer_paper_id",
    "retrieved_context",
    "model_answer",
    "extracted_cited_ids",
)


def _score_bucket(score: int) -> str:
    """Map a 1..5 score into coarse quality buckets (`low`, `mid`, `high`)."""
    if score <= 2:
        return "low"
    if score == 3:
        return "mid"
    return "high"


def _pct(hits: int, total: int) -> float | None:
    """Return percentage rounded to 2 decimals, or None if total is zero."""
    if total == 0:
        return None
    return round(100.0 * hits / total, 2)


def _mean(values: list[float | int]) -> float | None:
    """Return arithmetic mean rounded to 3 decimals, or None if empty."""
    if not values:
        return None
    return round(statistics.fmean(values), 3)


def export_blind_sample(
    zs_path: str | Path = "eval/outputs_zero_shot.jsonl",
    rag_path: str | Path = "eval/outputs_rag.jsonl",
    out_path: str | Path = "eval/blind_sample_20.jsonl",
    seed: int = 42,
) -> Path:
    """Create a stratified 20-item blind sample (10 zero-shot + 10 RAG) without judge fields."""
    rng = random.Random(seed)
    by_mode = {
        "zero_shot": read_jsonl(zs_path),
        "rag": read_jsonl(rag_path),
    }

    sampled: list[dict[str, Any]] = []
    for mode, rows in by_mode.items():
        for slice_name, target_count in SLICE_SAMPLE_COUNTS.items():
            candidates = sorted(
                (row for row in rows if row["slice"] == slice_name),
                key=lambda r: r["id"],
            )
            if len(candidates) < target_count:
                raise ValueError(
                    f"Mode {mode!r} slice {slice_name!r} has {len(candidates)} rows, "
                    f"needs {target_count}."
                )
            chosen = rng.sample(candidates, target_count)
            for row in chosen:
                clean_row = {field: row.get(field) for field in BLIND_FIELDS}
                clean_row["mode"] = row.get("mode", mode)
                sampled.append(clean_row)

    rng.shuffle(sampled)
    final_rows = [
        {"sample_id": f"sample_{idx:02d}", **row}
        for idx, row in enumerate(sampled, start=1)
    ]
    return write_jsonl(out_path, final_rows)


def _aggregate_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate rubric scores and latencies for a group of scored rows."""
    latencies = [float(r["latency_ms"]) for r in rows if r.get("latency_ms") is not None]
    override_vals = [
        int(r["override_obedience"])
        for r in rows
        if r.get("slice") == "temporal_override" and r.get("override_obedience") is not None
    ]
    return {
        "count": len(rows),
        "correctness": _mean([int(r["correctness"]) for r in rows]),
        "faithfulness": _mean([int(r["faithfulness"]) for r in rows]),
        "citation_accuracy": _mean([int(r["citation_accuracy"]) for r in rows]),
        "override_obedience": _mean(override_vals),
        "latency_mean_ms": round(statistics.fmean(latencies), 2) if latencies else None,
        "latency_median_ms": round(statistics.median(latencies), 2) if latencies else None,
    }


def compute_summary(
    judge_path: str | Path = "eval/judge_scores.jsonl",
    zs_path: str | Path | None = "eval/outputs_zero_shot.jsonl",
    rag_path: str | Path | None = "eval/outputs_rag.jsonl",
) -> dict[str, Any]:
    """Aggregate per-(mode, slice) and overall metrics from judge scores and latencies."""
    judge_rows = read_jsonl(judge_path)

    latency_lookup: dict[tuple[str, str], float] = {}
    for fallback_mode, path in (("zero_shot", zs_path), ("rag", rag_path)):
        if path is not None and Path(path).exists():
            for row in read_jsonl(path):
                if row.get("latency_ms") is not None:
                    mode = str(row.get("mode", fallback_mode))
                    latency_lookup[(row["id"], mode)] = float(row["latency_ms"])

    enriched: list[dict[str, Any]] = []
    for row in judge_rows:
        item = dict(row)
        if item.get("latency_ms") is None:
            key = (item["id"], item["mode"])
            if key in latency_lookup:
                item["latency_ms"] = latency_lookup[key]
        enriched.append(item)

    modes = sorted({str(r["mode"]) for r in enriched})
    slices = ("static", "live", "temporal_override", "unanswerable")

    per_mode_slice: dict[str, dict[str, dict[str, Any]]] = {}
    overall: dict[str, dict[str, Any]] = {}

    for mode in modes:
        mode_rows = [r for r in enriched if r["mode"] == mode]
        overall[mode] = _aggregate_group(mode_rows)
        per_mode_slice[mode] = {}
        for slice_name in slices:
            slice_rows = [r for r in mode_rows if r["slice"] == slice_name]
            if slice_rows:
                per_mode_slice[mode][slice_name] = _aggregate_group(slice_rows)

    return {
        "total_scored": len(enriched),
        "overall": overall,
        "by_mode_slice": per_mode_slice,
    }


def _ordinal_agreement_stats(pairs: list[tuple[int, int]]) -> dict[str, Any]:
    """Compute exact, within-1, and bucket agreement percentages for score pairs."""
    total = len(pairs)
    exact = sum(1 for j, h in pairs if j == h)
    within_1 = sum(1 for j, h in pairs if abs(j - h) <= 1)
    bucket = sum(1 for j, h in pairs if _score_bucket(j) == _score_bucket(h))
    return {
        "n": total,
        "exact_match_pct": _pct(exact, total),
        "within_1_pct": _pct(within_1, total),
        "bucket_match_pct": _pct(bucket, total),
    }


def compute_agreement(
    judge_path: str | Path,
    human_path: str | Path,
) -> dict[str, Any]:
    """Compute per-dimension and pooled agreement between judge and human scores."""
    judge_rows = read_jsonl(judge_path)
    human_rows = read_jsonl(human_path)

    judge_by_key = {(row["id"], row["mode"]): row for row in judge_rows}
    matched: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for h_row in human_rows:
        key = (h_row["id"], h_row["mode"])
        if key in judge_by_key:
            matched.append((judge_by_key[key], h_row))

    dimensions: dict[str, Any] = {}
    pooled_pairs: list[tuple[int, int]] = []

    for dim in ORDINAL_DIMS:
        dim_pairs = [
            (int(j_row[dim]), int(h_row[dim]))
            for j_row, h_row in matched
            if j_row.get(dim) is not None and h_row.get(dim) is not None
        ]
        dimensions[dim] = _ordinal_agreement_stats(dim_pairs)
        pooled_pairs.extend(dim_pairs)

    override_pairs = [
        (int(j_row["override_obedience"]), int(h_row["override_obedience"]))
        for j_row, h_row in matched
        if j_row.get("slice") == "temporal_override"
        and j_row.get("override_obedience") is not None
        and h_row.get("override_obedience") is not None
    ]
    override_exact = sum(1 for j, h in override_pairs if j == h)
    override_exact_pct = _pct(override_exact, len(override_pairs))
    dimensions["override_obedience"] = {
        "n": len(override_pairs),
        "override_exact_pct": override_exact_pct,
    }

    return {
        "n_items": len(matched),
        "pooled_n": len(pooled_pairs),
        "dimensions": dimensions,
        "pooled": _ordinal_agreement_stats(pooled_pairs),
        "override_exact_pct": override_exact_pct,
    }


def _fmt_num(val: float | None) -> str:
    """Format an optional float for Markdown tables."""
    return "-" if val is None else f"{val}"


def render_markdown_report(
    summary: dict[str, Any],
    agreement: dict[str, Any] | None = None,
) -> str:
    """Render evaluation summary and optional human-agreement tables to Markdown."""
    lines = ["# Evaluation Results Summary", ""]
    lines.append("## Overall by Mode")
    lines.append("")
    lines.append(
        "| Mode | N | Correctness | Faithfulness | Citation Acc | Override Obed | Mean Latency (ms) | Median Latency (ms) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for mode, stats in summary.get("overall", {}).items():
        lines.append(
            f"| {mode} | {stats['count']} | {_fmt_num(stats['correctness'])} | "
            f"{_fmt_num(stats['faithfulness'])} | {_fmt_num(stats['citation_accuracy'])} | "
            f"{_fmt_num(stats['override_obedience'])} | {_fmt_num(stats['latency_mean_ms'])} | "
            f"{_fmt_num(stats['latency_median_ms'])} |"
        )

    lines.extend(["", "## Breakdown by Mode & Slice", ""])
    lines.append(
        "| Mode | Slice | N | Correctness | Faithfulness | Citation Acc | Override Obed | Mean Latency (ms) |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for mode, slice_map in summary.get("by_mode_slice", {}).items():
        for slice_name, stats in slice_map.items():
            lines.append(
                f"| {mode} | {slice_name} | {stats['count']} | {_fmt_num(stats['correctness'])} | "
                f"{_fmt_num(stats['faithfulness'])} | {_fmt_num(stats['citation_accuracy'])} | "
                f"{_fmt_num(stats['override_obedience'])} | {_fmt_num(stats['latency_mean_ms'])} |"
            )

    if agreement is not None:
        lines.extend(["", "## Judge vs. Human Agreement", ""])
        lines.append(
            f"- Matched items: {agreement['n_items']} (pooled ordinal scores: {agreement['pooled_n']})"
        )
        lines.append("")
        lines.append("| Dimension | N | Exact Match % | Within-1 % | Bucket Match % |")
        lines.append("|---|---:|---:|---:|---:|")
        for dim in ORDINAL_DIMS:
            d = agreement["dimensions"][dim]
            lines.append(
                f"| {dim} | {d['n']} | {_fmt_num(d['exact_match_pct'])} | "
                f"{_fmt_num(d['within_1_pct'])} | {_fmt_num(d['bucket_match_pct'])} |"
            )
        pooled = agreement["pooled"]
        lines.append(
            f"| **pooled** | {pooled['n']} | {_fmt_num(pooled['exact_match_pct'])} | "
            f"{_fmt_num(pooled['within_1_pct'])} | {_fmt_num(pooled['bucket_match_pct'])} |"
        )
        ov = agreement["dimensions"]["override_obedience"]
        lines.append(
            f"\n- `override_obedience` exact match (N={ov['n']}): {_fmt_num(ov['override_exact_pct'])}%"
        )

    lines.append("")
    return "\n".join(lines)


def write_report(
    judge_path: str | Path = "eval/judge_scores.jsonl",
    zs_path: str | Path | None = "eval/outputs_zero_shot.jsonl",
    rag_path: str | Path | None = "eval/outputs_rag.jsonl",
    human_path: str | Path | None = "eval/human_scores_20.jsonl",
    out_dir: str | Path = "eval",
) -> tuple[Path, Path]:
    """Compute summary (+ optional agreement) and write JSON & Markdown reports."""
    summary = compute_summary(judge_path=judge_path, zs_path=zs_path, rag_path=rag_path)
    agreement: dict[str, Any] | None = None
    if human_path is not None and Path(human_path).exists():
        agreement = compute_agreement(judge_path=judge_path, human_path=human_path)
        summary["agreement"] = agreement

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / "results_summary.json"
    md_path = out_root / "results_summary.md"

    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md_path.write_text(
        render_markdown_report(summary, agreement=agreement),
        encoding="utf-8",
    )
    return json_path, md_path


def main() -> None:
    """CLI entry point for blind sampling and metrics reporting."""
    parser = argparse.ArgumentParser(description="Export blind sample or compute evaluation metrics.")
    parser.add_argument("--export-blind", action="store_true", help="Export 20-item stratified blind sample.")
    parser.add_argument("--zs-path", default="eval/outputs_zero_shot.jsonl")
    parser.add_argument("--rag-path", default="eval/outputs_rag.jsonl")
    parser.add_argument("--out-path", default="eval/blind_sample_20.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge-path", default="eval/judge_scores.jsonl")
    parser.add_argument("--human-path", default="eval/human_scores_20.jsonl")
    parser.add_argument("--out-dir", default="eval")
    args = parser.parse_args()

    if args.export_blind:
        out = export_blind_sample(
            zs_path=args.zs_path,
            rag_path=args.rag_path,
            out_path=args.out_path,
            seed=args.seed,
        )
        print(f"Wrote blind sample to {out}")
        return

    json_path, md_path = write_report(
        judge_path=args.judge_path,
        zs_path=args.zs_path,
        rag_path=args.rag_path,
        human_path=args.human_path,
        out_dir=args.out_dir,
    )
    print(f"Wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()

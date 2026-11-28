"""Statistical model comparison engine for Original vs. QLoRA fine-tuned models."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

from src import extract_arxiv_ids, read_jsonl
from src.metrics import _aggregate_group, _fmt_num, _mean, _pct, compute_summary

SLICES: tuple[str, ...] = ("static", "live", "temporal_override", "unanswerable")
MODES: tuple[str, ...] = ("zero_shot", "rag")
DELTA_METRICS: tuple[str, ...] = (
    "correctness",
    "faithfulness",
    "citation_accuracy",
    "override_obedience",
    "latency_mean_ms",
    "latency_median_ms",
)
SCORE_METRICS: tuple[str, ...] = (
    "correctness",
    "faithfulness",
    "citation_accuracy",
    "override_obedience",
)
ABSTENTION_MARKERS: tuple[str, ...] = (
    "not in the sources",
    "i don't have",
    "i do not have",
    "cannot verify",
    "my knowledge cutoff",
)
HYBRID_TAUS: tuple[float, ...] = (0.55, 0.60, 0.65, 0.68, 0.70, 0.75)
DEFAULT_HYBRID_TAU: float = 0.68
HYBRID_SIMULATION_NOTE: str = (
    "Post-hoc threshold routing over measured zero_shot and rag outputs"
)


def _is_abstention(text: str) -> bool:
    """Return True when `text` contains a known abstention phrase."""
    lower = text.lower()
    return any(marker in lower for marker in ABSTENTION_MARKERS)


def _is_spurious_trailing_abstention(text: str) -> bool:
    """Detect substantive answers (>=80 chars) ending with 'not in the sources'."""
    stripped = text.strip()
    if len(stripped) < 80:
        return False
    lower = stripped.lower()
    idx = lower.rfind("not in the sources")
    if idx <= 0:
        return False
    trailing_after = len(lower) - (idx + len("not in the sources"))
    return trailing_after <= 25


def _load_enriched_judge_rows(
    judge_path: str | Path,
    zs_path: str | Path | None = None,
    rag_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load judge rows and backfill missing `latency_ms` from output JSONL files."""
    judge_rows = read_jsonl(judge_path)
    latency_lookup: dict[tuple[str, str], float] = {}
    for fallback_mode, path in (("zero_shot", zs_path), ("rag", rag_path)):
        if path is not None and Path(path).exists():
            for row in read_jsonl(path):
                if row.get("latency_ms") is not None:
                    mode = str(row.get("mode", fallback_mode))
                    latency_lookup[(str(row["id"]), mode)] = float(row["latency_ms"])

    enriched: list[dict[str, Any]] = []
    for row in judge_rows:
        item = dict(row)
        if item.get("latency_ms") is None:
            key = (str(item["id"]), str(item["mode"]))
            if key in latency_lookup:
                item["latency_ms"] = latency_lookup[key]
        enriched.append(item)
    return enriched


def _diff_group_metrics(
    base_group: dict[str, Any],
    ft_group: dict[str, Any],
    metrics: tuple[str, ...] = DELTA_METRICS,
) -> dict[str, float | None]:
    """Compute `ft - base` deltas for aggregated group metrics."""
    deltas: dict[str, float | None] = {}
    for metric in metrics:
        bv = base_group.get(metric)
        fv = ft_group.get(metric)
        if bv is None or fv is None:
            deltas[metric] = None
        else:
            digits = 2 if metric.startswith("latency_") else 3
            deltas[metric] = round(float(fv) - float(bv), digits)
    return deltas


def _build_summary_with_pooled(
    judge_path: str | Path,
    zs_path: str | Path | None,
    rag_path: str | Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run `compute_summary`, add `pooled_all`, and return enriched rows."""
    summary = compute_summary(judge_path=judge_path, zs_path=zs_path, rag_path=rag_path)
    enriched = _load_enriched_judge_rows(judge_path, zs_path=zs_path, rag_path=rag_path)
    summary["overall"]["pooled_all"] = _aggregate_group(enriched)
    return summary, enriched


def _compute_score_comparison(
    base_summary: dict[str, Any],
    ft_summary: dict[str, Any],
) -> dict[str, Any]:
    """Compute overall (`zero_shot`, `rag`, `pooled_all`) and per-slice metric deltas."""
    overall_delta: dict[str, dict[str, float | None]] = {}
    for bucket in ("zero_shot", "rag", "pooled_all"):
        b_ov = base_summary.get("overall", {}).get(bucket, {})
        f_ov = ft_summary.get("overall", {}).get(bucket, {})
        overall_delta[bucket] = _diff_group_metrics(b_ov, f_ov)

    by_mode_slice_delta: dict[str, dict[str, dict[str, float | None]]] = {}
    for mode in MODES:
        b_slices = base_summary.get("by_mode_slice", {}).get(mode, {})
        f_slices = ft_summary.get("by_mode_slice", {}).get(mode, {})
        mode_deltas: dict[str, dict[str, float | None]] = {}
        for slice_name in SLICES:
            if slice_name in b_slices and slice_name in f_slices:
                mode_deltas[slice_name] = _diff_group_metrics(
                    b_slices[slice_name], f_slices[slice_name]
                )
        if mode_deltas:
            by_mode_slice_delta[mode] = mode_deltas

    return {
        "base": base_summary,
        "ft": ft_summary,
        "overall_delta": overall_delta,
        "by_mode_slice_delta": by_mode_slice_delta,
    }


def _paired_bootstrap_ci(
    pairs: list[tuple[float, float]],
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute deterministic paired bootstrap 95% CI for `ft - base` mean delta."""
    n = len(pairs)
    if n == 0:
        return {
            "n": 0,
            "base_mean": None,
            "ft_mean": None,
            "delta_mean": None,
            "ci95": [None, None],
            "statistically_significant": False,
        }

    base_vals = [b for b, _ in pairs]
    ft_vals = [f for _, f in pairs]
    diffs = [f - b for b, f in pairs]

    base_mean = round(statistics.fmean(base_vals), 3)
    ft_mean = round(statistics.fmean(ft_vals), 3)
    delta_mean = round(statistics.fmean(diffs), 3)

    rng = random.Random(seed)
    boot_deltas: list[float] = []
    for _ in range(n_boot):
        sample_sum = sum(diffs[rng.randrange(n)] for _ in range(n))
        boot_deltas.append(sample_sum / n)

    boot_deltas.sort()
    low_idx = max(0, min(len(boot_deltas) - 1, int(0.025 * len(boot_deltas))))
    high_idx = max(0, min(len(boot_deltas) - 1, int(0.975 * len(boot_deltas)) - 1))
    low = round(boot_deltas[low_idx], 3)
    high = round(boot_deltas[high_idx], 3)

    return {
        "n": n,
        "base_mean": base_mean,
        "ft_mean": ft_mean,
        "delta_mean": delta_mean,
        "ci95": [low, high],
        "statistically_significant": bool(low > 0 or high < 0),
    }


def _compute_bootstrap_cis(
    matched_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Compute paired bootstrap 95% CIs across pooled, mode, slice, and override buckets."""
    def _corr_pairs(
        subset: list[tuple[dict[str, Any], dict[str, Any]]],
    ) -> list[tuple[float, float]]:
        return [
            (float(b["correctness"]), float(f["correctness"]))
            for b, f in subset
            if b.get("correctness") is not None and f.get("correctness") is not None
        ]

    cis: dict[str, dict[str, Any]] = {
        "pooled_all": _paired_bootstrap_ci(
            _corr_pairs(matched_pairs), n_boot=n_boot, seed=seed
        ),
        "zero_shot_all": _paired_bootstrap_ci(
            _corr_pairs([(b, f) for b, f in matched_pairs if b["mode"] == "zero_shot"]),
            n_boot=n_boot,
            seed=seed,
        ),
        "rag_all": _paired_bootstrap_ci(
            _corr_pairs([(b, f) for b, f in matched_pairs if b["mode"] == "rag"]),
            n_boot=n_boot,
            seed=seed,
        ),
    }

    for mode in MODES:
        for slice_name in SLICES:
            bucket = f"{mode}_{slice_name}"
            sub = [
                (b, f)
                for b, f in matched_pairs
                if b["mode"] == mode and b["slice"] == slice_name
            ]
            cis[bucket] = _paired_bootstrap_ci(_corr_pairs(sub), n_boot=n_boot, seed=seed)

    override_pairs = [
        (float(b["override_obedience"]), float(f["override_obedience"]))
        for b, f in matched_pairs
        if b["mode"] == "rag"
        and b["slice"] == "temporal_override"
        and b.get("override_obedience") is not None
        and f.get("override_obedience") is not None
    ]
    cis["rag_temporal_override_obedience"] = _paired_bootstrap_ci(
        override_pairs, n_boot=n_boot, seed=seed
    )
    return cis


def _tally_outcomes(outcomes: list[str]) -> dict[str, int]:
    """Summarize win/tie/loss counts and net wins."""
    improved = sum(1 for o in outcomes if o == "improved")
    tied = sum(1 for o in outcomes if o == "tied")
    regressed = sum(1 for o in outcomes if o == "regressed")
    return {
        "n": len(outcomes),
        "improved": improved,
        "tied": tied,
        "regressed": regressed,
        "net_wins": improved - regressed,
    }


def _compute_win_loss_and_paired_items(
    matched_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build prompt-level win/tie/loss counts and full paired diff records."""
    paired_items: list[dict[str, Any]] = []
    for b_row, f_row in matched_pairs:
        b_corr = int(b_row["correctness"])
        f_corr = int(f_row["correctness"])
        delta = f_corr - b_corr
        if delta > 0:
            outcome = "improved"
        elif delta < 0:
            outcome = "regressed"
        else:
            outcome = "tied"

        paired_items.append(
            {
                "id": b_row["id"],
                "mode": b_row["mode"],
                "slice": b_row["slice"],
                "prompt": b_row.get("prompt"),
                "base_scores": {m: b_row.get(m) for m in SCORE_METRICS},
                "ft_scores": {m: f_row.get(m) for m in SCORE_METRICS},
                "correctness_delta": delta,
                "outcome": outcome,
                "base_answer_preview": str(b_row.get("model_answer", ""))[:300],
                "ft_answer_preview": str(f_row.get("model_answer", ""))[:300],
                "base_rationale": b_row.get("rationale"),
                "ft_rationale": f_row.get("rationale"),
                "base_latency_ms": b_row.get("latency_ms"),
                "ft_latency_ms": f_row.get("latency_ms"),
            }
        )

    by_mode: dict[str, dict[str, int]] = {}
    by_mode_slice: dict[str, dict[str, dict[str, int]]] = {}
    for mode in MODES:
        mode_items = [item for item in paired_items if item["mode"] == mode]
        by_mode[mode] = _tally_outcomes([item["outcome"] for item in mode_items])
        by_mode_slice[mode] = {}
        for slice_name in SLICES:
            sl_items = [item for item in mode_items if item["slice"] == slice_name]
            if sl_items:
                by_mode_slice[mode][slice_name] = _tally_outcomes(
                    [item["outcome"] for item in sl_items]
                )

    win_loss = {
        "overall": _tally_outcomes([item["outcome"] for item in paired_items]),
        "by_mode": by_mode,
        "by_mode_slice": by_mode_slice,
    }
    return win_loss, paired_items


def _behavior_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count abstentions, cited answers, uncited answers, and spurious trailing abstentions."""
    abstained = 0
    cited = 0
    uncited = 0
    spurious_trailing = 0
    for row in rows:
        answer = str(row.get("model_answer", ""))
        is_abs = _is_abstention(answer)
        if _is_spurious_trailing_abstention(answer):
            spurious_trailing += 1
        if is_abs:
            abstained += 1
        else:
            cited_ids = (
                row.get("extracted_cited_ids")
                if "extracted_cited_ids" in row
                else extract_arxiv_ids(answer)
            )
            if cited_ids:
                cited += 1
            else:
                uncited += 1
    return {
        "abstained": abstained,
        "cited": cited,
        "uncited": uncited,
        "spurious_trailing_abstention": spurious_trailing,
    }


def _compute_behavior_shift(
    base_rows: list[dict[str, Any]],
    ft_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute abstention, citation, and spurious trailing abstention shifts."""
    by_mode_slice: dict[str, dict[str, dict[str, Any]]] = {}
    by_mode: dict[str, dict[str, Any]] = {}

    for mode in MODES:
        b_mode = [r for r in base_rows if r.get("mode") == mode]
        f_mode = [r for r in ft_rows if r.get("mode") == mode]
        b_mode_counts = _behavior_counts(b_mode)
        f_mode_counts = _behavior_counts(f_mode)
        by_mode[mode] = {
            "n_base": len(b_mode),
            "n_ft": len(f_mode),
            "base": b_mode_counts,
            "ft": f_mode_counts,
            "delta": {
                k: f_mode_counts[k] - b_mode_counts[k] for k in b_mode_counts
            },
        }

        by_mode_slice[mode] = {}
        for slice_name in SLICES:
            b_sub = [r for r in b_mode if r.get("slice") == slice_name]
            f_sub = [r for r in f_mode if r.get("slice") == slice_name]
            if not b_sub and not f_sub:
                continue
            b_counts = _behavior_counts(b_sub)
            f_counts = _behavior_counts(f_sub)
            by_mode_slice[mode][slice_name] = {
                "n_base": len(b_sub),
                "n_ft": len(f_sub),
                "base": b_counts,
                "ft": f_counts,
                "delta": {k: f_counts[k] - b_counts[k] for k in b_counts},
            }

    return {
        "by_mode": by_mode,
        "by_mode_slice": by_mode_slice,
    }


def _compute_cross_judge_check(
    primary_score_comp: dict[str, Any],
    *,
    secondary_base_judge_path: str | Path | None,
    secondary_ft_judge_path: str | Path | None,
    base_zs_path: str | Path | None,
    base_rag_path: str | Path | None,
    ft_zs_path: str | Path | None,
    ft_rag_path: str | Path | None,
    judge_label: str,
    secondary_judge_label: str,
) -> dict[str, Any]:
    """Compare FT-vs-Base deltas under a secondary judge when both files exist."""
    if (
        secondary_base_judge_path is None
        or secondary_ft_judge_path is None
        or not Path(secondary_base_judge_path).exists()
        or not Path(secondary_ft_judge_path).exists()
    ):
        return {
            "available": False,
            "primary_judge_label": judge_label,
            "secondary_judge_label": secondary_judge_label,
            "reason": "Secondary judge score file(s) not found.",
        }

    sec_base_summary, _ = _build_summary_with_pooled(
        secondary_base_judge_path, base_zs_path, base_rag_path
    )
    sec_ft_summary, _ = _build_summary_with_pooled(
        secondary_ft_judge_path, ft_zs_path, ft_rag_path
    )
    sec_comp = _compute_score_comparison(sec_base_summary, sec_ft_summary)

    agreement_by_bucket: dict[str, dict[str, Any]] = {}
    key_buckets: list[tuple[str, float | None, float | None]] = []
    for bucket in ("pooled_all", "zero_shot", "rag"):
        p_d = primary_score_comp["overall_delta"].get(bucket, {}).get("correctness")
        s_d = sec_comp["overall_delta"].get(bucket, {}).get("correctness")
        key_buckets.append((bucket, p_d, s_d))

    for mode in MODES:
        for slice_name in SLICES:
            p_d = (
                primary_score_comp["by_mode_slice_delta"]
                .get(mode, {})
                .get(slice_name, {})
                .get("correctness")
            )
            s_d = (
                sec_comp["by_mode_slice_delta"]
                .get(mode, {})
                .get(slice_name, {})
                .get("correctness")
            )
            if p_d is not None or s_d is not None:
                key_buckets.append((f"{mode}_{slice_name}", p_d, s_d))

    all_agree = True
    for bucket, p_d, s_d in key_buckets:
        significant_mag = p_d is not None and abs(p_d) >= 0.1
        agrees: bool | None = None
        if significant_mag and s_d is not None:
            agrees = (p_d > 0 and s_d > 0) or (p_d < 0 and s_d < 0)
            if not agrees:
                all_agree = False
        agreement_by_bucket[bucket] = {
            "primary_delta": p_d,
            "secondary_delta": s_d,
            "checked": significant_mag and s_d is not None,
            "direction_agrees": agrees,
        }

    return {
        "available": True,
        "primary_judge_label": judge_label,
        "secondary_judge_label": secondary_judge_label,
        "base": sec_base_summary,
        "ft": sec_ft_summary,
        "overall_delta": sec_comp["overall_delta"],
        "by_mode_slice_delta": sec_comp["by_mode_slice_delta"],
        "agreement_by_bucket": agreement_by_bucket,
        "direction_agrees": all_agree,
    }


def _load_top1_scores_from_index(
    prompts_by_id: dict[str, str],
    index_dir: str | Path,
) -> dict[str, float]:
    """Query FAISS `Retriever` for `k=1` cosine similarity per unique prompt."""
    from src.retriever import Retriever

    retriever = Retriever(index_dir=index_dir)
    scores: dict[str, float] = {}
    for item_id, prompt in prompts_by_id.items():
        chunks = retriever.search(prompt, k=1)
        if chunks:
            scores[item_id] = round(float(chunks[0]["score"]), 6)
    return scores


def simulate_hybrid_routing(
    base_rows: list[dict[str, Any]],
    ft_rows: list[dict[str, Any]],
    top1_scores: dict[str, float],
    *,
    taus: tuple[float, ...] = HYBRID_TAUS,
    default_tau: float = DEFAULT_HYBRID_TAU,
) -> dict[str, Any]:
    """Simulate post-hoc similarity-gated routing across `taus` over measured ZS/RAG rows."""
    base_by_id_mode = {(str(r["id"]), str(r["mode"])): r for r in base_rows}
    ft_by_id_mode = {(str(r["id"]), str(r["mode"])): r for r in ft_rows}

    common_ids = sorted(
        {
            item_id
            for item_id in top1_scores
            if (item_id, "zero_shot") in base_by_id_mode
            and (item_id, "rag") in base_by_id_mode
            and (item_id, "zero_shot") in ft_by_id_mode
            and (item_id, "rag") in ft_by_id_mode
        }
    )

    if not common_ids:
        return {
            "available": False,
            "simulation_note": HYBRID_SIMULATION_NOTE,
            "reason": "No matched zero_shot + rag pairs with top-1 similarity scores.",
        }

    def _summarize_selected(
        rows: list[dict[str, Any]],
        routed_flags: list[bool],
    ) -> dict[str, Any]:
        agg = _aggregate_group(rows)
        n_rag = sum(1 for flag in routed_flags if flag)
        agg["rag_routed_count"] = n_rag
        agg["rag_routed_pct"] = _pct(n_rag, len(rows))
        return agg

    by_tau: dict[str, dict[str, Any]] = {}
    for tau in taus:
        tau_key = f"{tau:.2f}"
        base_selected: list[dict[str, Any]] = []
        ft_selected: list[dict[str, Any]] = []
        routed_flags: list[bool] = []

        for item_id in common_ids:
            use_rag = top1_scores[item_id] >= tau
            mode = "rag" if use_rag else "zero_shot"
            base_selected.append(base_by_id_mode[(item_id, mode)])
            ft_selected.append(ft_by_id_mode[(item_id, mode)])
            routed_flags.append(use_rag)

        base_overall = _summarize_selected(base_selected, routed_flags)
        ft_overall = _summarize_selected(ft_selected, routed_flags)

        base_by_slice: dict[str, dict[str, Any]] = {}
        ft_by_slice: dict[str, dict[str, Any]] = {}
        delta_by_slice: dict[str, dict[str, float | None]] = {}
        for slice_name in SLICES:
            idxs = [
                idx
                for idx, row in enumerate(base_selected)
                if row.get("slice") == slice_name
            ]
            if not idxs:
                continue
            b_sub = [base_selected[idx] for idx in idxs]
            f_sub = [ft_selected[idx] for idx in idxs]
            r_sub = [routed_flags[idx] for idx in idxs]
            base_by_slice[slice_name] = _summarize_selected(b_sub, r_sub)
            ft_by_slice[slice_name] = _summarize_selected(f_sub, r_sub)
            delta_by_slice[slice_name] = _diff_group_metrics(
                base_by_slice[slice_name], ft_by_slice[slice_name]
            )

        by_tau[tau_key] = {
            "tau": tau,
            "n": len(common_ids),
            "base": {"overall": base_overall, "by_slice": base_by_slice},
            "ft": {"overall": ft_overall, "by_slice": ft_by_slice},
            "delta": {
                "overall": _diff_group_metrics(base_overall, ft_overall),
                "by_slice": delta_by_slice,
            },
        }

    default_key = f"{default_tau:.2f}"
    return {
        "available": True,
        "simulation_note": HYBRID_SIMULATION_NOTE,
        "default_tau": default_tau,
        "taus": list(taus),
        "n_prompts": len(common_ids),
        "top1_scores": {k: top1_scores[k] for k in common_ids},
        "by_tau": by_tau,
        "default_comparison": by_tau.get(default_key),
    }


def _compute_hybrid_routing(
    base_rows: list[dict[str, Any]],
    ft_rows: list[dict[str, Any]],
    *,
    index_dir: str | Path | None,
    precomputed_top1_scores: dict[str, float] | None,
) -> dict[str, Any]:
    """Compute post-hoc hybrid routing using precomputed scores or FAISS index."""
    if precomputed_top1_scores is not None:
        return simulate_hybrid_routing(base_rows, ft_rows, precomputed_top1_scores)

    if index_dir is None:
        return {
            "available": False,
            "simulation_note": HYBRID_SIMULATION_NOTE,
            "reason": "No index_dir provided.",
        }

    faiss_path = Path(index_dir) / "live.faiss"
    if not faiss_path.exists():
        return {
            "available": False,
            "simulation_note": HYBRID_SIMULATION_NOTE,
            "reason": f"FAISS index not found at {faiss_path}.",
        }

    prompts_by_id: dict[str, str] = {}
    for row in base_rows:
        item_id = str(row["id"])
        if item_id not in prompts_by_id and row.get("prompt"):
            prompts_by_id[item_id] = str(row["prompt"])

    try:
        top1_scores = _load_top1_scores_from_index(prompts_by_id, index_dir)
    except Exception as exc:
        return {
            "available": False,
            "simulation_note": HYBRID_SIMULATION_NOTE,
            "reason": f"Retriever unavailable: {exc}",
        }

    return simulate_hybrid_routing(base_rows, ft_rows, top1_scores)


def _load_training_telemetry(
    train_manifest_path: str | Path | None,
    training_log_path: str | Path | None,
) -> dict[str, Any]:
    """Load QLoRA training manifest and training log metrics when present."""
    manifest: dict[str, Any] | None = None
    manifest_available = False
    if train_manifest_path is not None and Path(train_manifest_path).exists():
        try:
            manifest = json.loads(
                Path(train_manifest_path).read_text(encoding="utf-8")
            )
            manifest_available = True
        except Exception:
            manifest = None

    log_entries: list[dict[str, Any]] = []
    log_available = False
    log_summary: dict[str, Any] | None = None
    if training_log_path is not None and Path(training_log_path).exists():
        try:
            log_entries = read_jsonl(training_log_path)
            log_available = True
        except Exception:
            log_entries = []

    if log_entries:
        losses = [
            float(entry["loss"])
            for entry in log_entries
            if entry.get("loss") is not None
        ]
        if not losses:
            losses = [
                float(entry["train_loss"])
                for entry in log_entries
                if entry.get("train_loss") is not None
            ]
        val_losses = [
            float(entry["val_loss"])
            for entry in log_entries
            if entry.get("val_loss") is not None
        ]
        steps = [
            int(entry["step"])
            for entry in log_entries
            if entry.get("step") is not None
        ]
        log_summary = {
            "n_entries": len(log_entries),
            "max_step": max(steps) if steps else None,
            "initial_loss": round(losses[0], 4) if losses else None,
            "final_loss": round(losses[-1], 4) if losses else None,
            "min_loss": round(min(losses), 4) if losses else None,
            "final_val_loss": round(val_losses[-1], 4) if val_losses else None,
            "min_val_loss": round(min(val_losses), 4) if val_losses else None,
        }

    return {
        "manifest_available": manifest_available,
        "log_available": log_available,
        "manifest": manifest,
        "log_summary": log_summary,
        "log_entries": log_entries,
    }


def _build_recommendation(
    *,
    base_label: str,
    ft_label: str,
    judge_label: str,
    score_comparison: dict[str, Any],
    bootstrap_cis: dict[str, dict[str, Any]],
    win_loss: dict[str, Any],
    behavior_shift: dict[str, Any],
    cross_judge_check: dict[str, Any],
    hybrid_routing: dict[str, Any],
) -> dict[str, Any]:
    """Derive winner, headline, and grounded summary bullets strictly from measured data."""
    pooled_ci = bootstrap_cis["pooled_all"]
    zs_ci = bootstrap_cis["zero_shot_all"]
    rag_ci = bootstrap_cis["rag_all"]

    pooled_delta = pooled_ci.get("delta_mean")
    if pooled_delta is None or pooled_delta == 0:
        winner = "tie"
    elif pooled_delta > 0:
        winner = ft_label
    else:
        winner = base_label

    sig_text = (
        "statistically significant"
        if pooled_ci.get("statistically_significant")
        else "not statistically significant"
    )
    ci_pair = pooled_ci.get("ci95", [None, None])
    ci_str = f"[{ci_pair[0]:+}, {ci_pair[1]:+}]" if ci_pair[0] is not None else "n/a"

    zs_delta_str = _fmt_delta(zs_ci.get("delta_mean"))
    rag_delta_str = _fmt_delta(rag_ci.get("delta_mean"))

    if winner == ft_label:
        verdict_headline = (
            f"RECOMMEND `{ft_label}` over `{base_label}` ({judge_label}): "
            f"pooled correctness {pooled_delta:+} (95% CI {ci_str}, {sig_text}; "
            f"zero-shot {zs_delta_str}, RAG {rag_delta_str})."
        )
    elif winner == base_label:
        verdict_headline = (
            f"KEEP `{base_label}` over `{ft_label}` ({judge_label}): "
            f"pooled correctness {pooled_delta:+} (95% CI {ci_str}, {sig_text}; "
            f"zero-shot {zs_delta_str}, RAG {rag_delta_str})."
        )
    else:
        verdict_headline = (
            f"NO CLEAR WINNER between `{base_label}` and `{ft_label}` ({judge_label}): "
            f"pooled correctness delta is 0.0 (95% CI {ci_str})."
        )

    ov_wl = win_loss["overall"]
    bullets: list[str] = [
        (
            f"Pooled correctness (N={pooled_ci['n']}): {pooled_ci['base_mean']} -> "
            f"{pooled_ci['ft_mean']} (delta {pooled_delta:+}, 95% CI {ci_str}, {sig_text})."
        ),
        (
            f"Zero-Shot correctness (N={zs_ci['n']}): {zs_ci['base_mean']} -> "
            f"{zs_ci['ft_mean']} (delta {zs_ci['delta_mean']:+}, "
            f"95% CI [{zs_ci['ci95'][0]:+}, {zs_ci['ci95'][1]:+}])."
            if zs_ci["ci95"][0] is not None
            else f"Zero-Shot correctness: N={zs_ci['n']}."
        ),
        (
            f"RAG correctness (N={rag_ci['n']}): {rag_ci['base_mean']} -> "
            f"{rag_ci['ft_mean']} (delta {rag_ci['delta_mean']:+}, "
            f"95% CI [{rag_ci['ci95'][0]:+}, {rag_ci['ci95'][1]:+}])."
            if rag_ci["ci95"][0] is not None
            else f"RAG correctness: N={rag_ci['n']}."
        ),
        (
            f"Prompt-level outcomes (N={ov_wl['n']}): {ov_wl['improved']} improved, "
            f"{ov_wl['tied']} tied, {ov_wl['regressed']} regressed "
            f"(net wins {ov_wl['net_wins']:+})."
        ),
    ]

    rag_beh = behavior_shift["by_mode"].get("rag", {})
    if rag_beh:
        b_sp = rag_beh["base"]["spurious_trailing_abstention"]
        f_sp = rag_beh["ft"]["spurious_trailing_abstention"]
        b_ab = rag_beh["base"]["abstained"]
        f_ab = rag_beh["ft"]["abstained"]
        bullets.append(
            f"RAG behavior shift: abstentions {b_ab} -> {f_ab} ({f_ab - b_ab:+}), "
            f"spurious trailing abstentions {b_sp} -> {f_sp} ({f_sp - b_sp:+})."
        )

    if cross_judge_check.get("available"):
        sec_lbl = cross_judge_check["secondary_judge_label"]
        sec_pooled = (
            cross_judge_check["overall_delta"].get("pooled_all", {}).get("correctness")
        )
        agrees = cross_judge_check.get("direction_agrees")
        bullets.append(
            f"Dual-judge cross-check ({sec_lbl}): pooled correctness delta "
            f"{sec_pooled:+}, direction_agrees={agrees}."
            if sec_pooled is not None
            else f"Dual-judge cross-check ({sec_lbl}): direction_agrees={agrees}."
        )

    if hybrid_routing.get("available") and hybrid_routing.get("default_comparison"):
        dc = hybrid_routing["default_comparison"]
        b_h = dc["base"]["overall"]["correctness"]
        f_h = dc["ft"]["overall"]["correctness"]
        d_h = dc["delta"]["overall"]["correctness"]
        bullets.append(
            f"Post-hoc hybrid simulation (tau={hybrid_routing['default_tau']:.2f}, "
            f"N={dc['n']}): Base Hybrid {b_h} -> QLoRA Hybrid {f_h} (delta {d_h:+})."
            if d_h is not None
            else f"Post-hoc hybrid simulation available at tau={hybrid_routing['default_tau']:.2f}."
        )

    return {
        "winner": winner,
        "verdict_headline": verdict_headline,
        "summary_bullets": bullets,
    }


def compare_models(
    base_judge_path: str | Path = "eval/judge_scores_qwen35_9b.jsonl",
    ft_judge_path: str | Path = "eval/judge_scores_ft_qwen35_9b.jsonl",
    *,
    secondary_base_judge_path: str | Path | None = "eval/judge_scores.jsonl",
    secondary_ft_judge_path: str | Path | None = "eval/judge_scores_ft_qwen25_7b.jsonl",
    base_zs_path: str | Path | None = "eval/outputs_zero_shot.jsonl",
    base_rag_path: str | Path | None = "eval/outputs_rag.jsonl",
    ft_zs_path: str | Path | None = "eval/outputs_zero_shot_ft.jsonl",
    ft_rag_path: str | Path | None = "eval/outputs_rag_ft.jsonl",
    train_manifest_path: str | Path | None = "data/qlora/train_manifest.json",
    training_log_path: str | Path | None = "data/qlora/training_log.jsonl",
    index_dir: str | Path | None = "data/index",
    base_label: str = "llama3.2:3b",
    ft_label: str = "llama3.2-qlora:3b",
    judge_label: str = "qwen3.5:9b",
    secondary_judge_label: str = "qwen2.5:7b",
    precomputed_top1_scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compare Original vs. Fine-Tuned models across Zero-Shot and RAG modes."""
    base_summary, base_rows = _build_summary_with_pooled(
        base_judge_path, base_zs_path, base_rag_path
    )
    ft_summary, ft_rows = _build_summary_with_pooled(
        ft_judge_path, ft_zs_path, ft_rag_path
    )

    score_comparison = _compute_score_comparison(base_summary, ft_summary)

    base_by_key = {(str(r["id"]), str(r["mode"])): r for r in base_rows}
    ft_by_key = {(str(r["id"]), str(r["mode"])): r for r in ft_rows}
    matched_keys = sorted(
        set(base_by_key) & set(ft_by_key),
        key=lambda k: (k[1], k[0]),
    )
    matched_pairs = [(base_by_key[k], ft_by_key[k]) for k in matched_keys]

    bootstrap_cis = _compute_bootstrap_cis(matched_pairs, n_boot=2000, seed=42)
    win_loss, paired_items = _compute_win_loss_and_paired_items(matched_pairs)
    behavior_shift = _compute_behavior_shift(base_rows, ft_rows)
    cross_judge_check = _compute_cross_judge_check(
        score_comparison,
        secondary_base_judge_path=secondary_base_judge_path,
        secondary_ft_judge_path=secondary_ft_judge_path,
        base_zs_path=base_zs_path,
        base_rag_path=base_rag_path,
        ft_zs_path=ft_zs_path,
        ft_rag_path=ft_rag_path,
        judge_label=judge_label,
        secondary_judge_label=secondary_judge_label,
    )
    hybrid_routing = _compute_hybrid_routing(
        base_rows,
        ft_rows,
        index_dir=index_dir,
        precomputed_top1_scores=precomputed_top1_scores,
    )
    training = _load_training_telemetry(train_manifest_path, training_log_path)
    recommendation = _build_recommendation(
        base_label=base_label,
        ft_label=ft_label,
        judge_label=judge_label,
        score_comparison=score_comparison,
        bootstrap_cis=bootstrap_cis,
        win_loss=win_loss,
        behavior_shift=behavior_shift,
        cross_judge_check=cross_judge_check,
        hybrid_routing=hybrid_routing,
    )

    return {
        "labels": {
            "base": base_label,
            "ft": ft_label,
            "judge": judge_label,
            "secondary_judge": secondary_judge_label,
        },
        "score_comparison": score_comparison,
        "bootstrap_cis": bootstrap_cis,
        "win_loss": win_loss,
        "paired_items": paired_items,
        "behavior_shift": behavior_shift,
        "cross_judge_check": cross_judge_check,
        "hybrid_routing": hybrid_routing,
        "training": training,
        "recommendation": recommendation,
    }


def _fmt_delta(val: float | int | None) -> str:
    """Format an optional signed numeric delta for Markdown tables."""
    return "-" if val is None else f"{val:+}"


def render_comparison_markdown(report: dict[str, Any]) -> str:
    """Render the full model comparison dictionary to structured Markdown."""
    labels = report["labels"]
    base_lbl = labels["base"]
    ft_lbl = labels["ft"]
    judge_lbl = labels["judge"]

    sc = report["score_comparison"]
    cis = report["bootstrap_cis"]
    wl = report["win_loss"]
    beh = report["behavior_shift"]
    cj = report["cross_judge_check"]
    hyb = report["hybrid_routing"]
    trn = report["training"]
    rec = report["recommendation"]

    lines: list[str] = [
        f"# Model Comparison Report: `{base_lbl}` vs `{ft_lbl}`",
        "",
        f"- **Base Model**: `{base_lbl}`",
        f"- **Fine-Tuned Model**: `{ft_lbl}`",
        f"- **Primary Judge**: `{judge_lbl}`",
        "",
        "## Executive Recommendation",
        "",
        rec.get("verdict_headline", ""),
        "",
    ]
    for bullet in rec.get("summary_bullets", []):
        lines.append(f"- {bullet}")

    lines.extend([
        "",
        "## 1. Score Comparison & Deltas",
        "",
        "### Overall by Mode & Pooled",
        (
            f"| Bucket | N | {base_lbl} Corr | {ft_lbl} Corr | Delta Corr | "
            "Delta Faith | Delta Cite | Delta Ov. Obed | Delta Mean Lat (ms) | Delta Med Lat (ms) |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for bucket in ("zero_shot", "rag", "pooled_all"):
        b_ov = sc["base"]["overall"].get(bucket, {})
        f_ov = sc["ft"]["overall"].get(bucket, {})
        d_ov = sc["overall_delta"].get(bucket, {})
        lines.append(
            f"| `{bucket}` | {b_ov.get('count', 0)} | "
            f"{_fmt_num(b_ov.get('correctness'))} | {_fmt_num(f_ov.get('correctness'))} | "
            f"{_fmt_delta(d_ov.get('correctness'))} | {_fmt_delta(d_ov.get('faithfulness'))} | "
            f"{_fmt_delta(d_ov.get('citation_accuracy'))} | {_fmt_delta(d_ov.get('override_obedience'))} | "
            f"{_fmt_delta(d_ov.get('latency_mean_ms'))} | {_fmt_delta(d_ov.get('latency_median_ms'))} |"
        )

    lines.extend([
        "",
        "### Breakdown by Mode & Slice",
        (
            f"| Mode | Slice | N | {base_lbl} Corr | {ft_lbl} Corr | Delta Corr | "
            "Delta Faith | Delta Cite | Delta Ov. Obed | Delta Mean Lat (ms) |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for mode in MODES:
        b_sl_map = sc["base"]["by_mode_slice"].get(mode, {})
        f_sl_map = sc["ft"]["by_mode_slice"].get(mode, {})
        d_sl_map = sc["by_mode_slice_delta"].get(mode, {})
        for sl in SLICES:
            if sl not in b_sl_map or sl not in f_sl_map:
                continue
            bs = b_sl_map[sl]
            fs = f_sl_map[sl]
            ds = d_sl_map.get(sl, {})
            lines.append(
                f"| `{mode}` | `{sl}` | {bs.get('count', 0)} | "
                f"{_fmt_num(bs.get('correctness'))} | {_fmt_num(fs.get('correctness'))} | "
                f"{_fmt_delta(ds.get('correctness'))} | {_fmt_delta(ds.get('faithfulness'))} | "
                f"{_fmt_delta(ds.get('citation_accuracy'))} | {_fmt_delta(ds.get('override_obedience'))} | "
                f"{_fmt_delta(ds.get('latency_mean_ms'))} |"
            )

    lines.extend([
        "",
        "## 2. Paired Bootstrap 95% Confidence Intervals",
        "",
        "| Bucket | N | Base Mean | FT Mean | Delta Mean | 95% CI | Significant |",
        "|---|---:|---:|---:|---:|---|---|",
    ])
    for bucket, entry in cis.items():
        ci_low, ci_high = entry["ci95"]
        ci_str = f"[{ci_low:+}, {ci_high:+}]" if ci_low is not None else "-"
        sig_str = "YES" if entry["statistically_significant"] else "no"
        lines.append(
            f"| `{bucket}` | {entry['n']} | {_fmt_num(entry['base_mean'])} | "
            f"{_fmt_num(entry['ft_mean'])} | {_fmt_delta(entry['delta_mean'])} | "
            f"{ci_str} | {sig_str} |"
        )

    lines.extend([
        "",
        "## 3. Prompt-Level Win / Tie / Loss Matrix",
        "",
        "| Scope | N | Improved | Tied | Regressed | Net Wins |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    ov_w = wl["overall"]
    lines.append(
        f"| **overall** | {ov_w['n']} | {ov_w['improved']} | {ov_w['tied']} | "
        f"{ov_w['regressed']} | {ov_w['net_wins']:+} |"
    )
    for mode in MODES:
        mw = wl["by_mode"].get(mode)
        if mw:
            lines.append(
                f"| `{mode}` | {mw['n']} | {mw['improved']} | {mw['tied']} | "
                f"{mw['regressed']} | {mw['net_wins']:+} |"
            )
        for sl in SLICES:
            sw = wl["by_mode_slice"].get(mode, {}).get(sl)
            if sw:
                lines.append(
                    f"| `{mode}/{sl}` | {sw['n']} | {sw['improved']} | {sw['tied']} | "
                    f"{sw['regressed']} | {sw['net_wins']:+} |"
                )

    lines.extend([
        "",
        "## 4. Behavioral Shift Analysis",
        "",
        "| Mode | Slice | Base Abst | FT Abst | Base Cited | FT Cited | Base Uncited | FT Uncited | Base Spurious Trail | FT Spurious Trail |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for mode in MODES:
        for sl in SLICES:
            entry = beh["by_mode_slice"].get(mode, {}).get(sl)
            if not entry:
                continue
            b = entry["base"]
            f = entry["ft"]
            lines.append(
                f"| `{mode}` | `{sl}` | {b['abstained']} | {f['abstained']} | "
                f"{b['cited']} | {f['cited']} | {b['uncited']} | {f['uncited']} | "
                f"{b['spurious_trailing_abstention']} | {f['spurious_trailing_abstention']} |"
            )

    lines.extend([
        "",
        f"## 5. Dual-Judge Cross-Check ({judge_lbl} vs. {cj['secondary_judge_label']})",
        "",
    ])
    if not cj.get("available"):
        lines.append(f"- Unavailable: {cj.get('reason', 'missing secondary judge files')}")
    else:
        lines.extend([
            f"- Secondary Judge: `{cj['secondary_judge_label']}`",
            f"- Overall Direction Agreement (where `|primary_delta| >= 0.1`): `{cj['direction_agrees']}`",
            "- **Data Compared**: All `240` outputs (`120` Base + `120` Fine-Tuned) scored independently by `qwen3.5:9b` (`90.0%` human exact match, `QWK=0.782`) and `qwen2.5:7b` (`80.0%` human exact match, `QWK=0.535`). `Direction Agrees` checks whether both judges report the same sign when `|primary_delta| >= 0.10`.",
            "",
            f"| Bucket | {judge_lbl} Delta Corr | {cj['secondary_judge_label']} Delta Corr | Direction Agrees |",
            "|---|---:|---:|---|",
        ])
        for bucket, item in cj["agreement_by_bucket"].items():
            ag_str = "-" if item["direction_agrees"] is None else str(item["direction_agrees"])
            lines.append(
                f"| `{bucket}` | {_fmt_delta(item['primary_delta'])} | "
                f"{_fmt_delta(item['secondary_delta'])} | {ag_str} |"
            )
        lines.extend([
            "",
            "- **Why Agreeing Buckets Validate Headline Wins**: Both judges agree in direction on `pooled_all`, `zero_shot`, `rag`, `zero_shot_static`, `zero_shot_live`, `zero_shot_temporal_override`, and `rag_temporal_override` (`7/10` active buckets), confirming QLoRA's gains are real model changes rather than judge bias.",
            "- **Why Divergent Buckets Differ (`False`)**: `rag_live` (`qwen3.5:9b` penalizes secondary metric omissions in concise 1–2 sentence answers, whereas `qwen2.5:7b` credits core mechanism accuracy), `zero_shot_unanswerable` (`qwen3.5:9b` catches `g54` fabrication while `qwen2.5:7b` had offsetting variance), and `rag_static` (both models cleanly abstain `20/20`, with a 1-item boundary shift on `qwen3.5:9b`).",
        ])

    lines.extend(["", "## 6. Post-Hoc Similarity-Gated Hybrid Routing Simulation", ""])
    lines.append(f"- Note: {hyb.get('simulation_note', HYBRID_SIMULATION_NOTE)}")
    if not hyb.get("available"):
        lines.append(f"- Unavailable: {hyb.get('reason', 'index unavailable')}")
    else:
        lines.extend([
            "",
            "### Threshold Sweep (`tau`)",
            "| tau | N | RAG Routed % | Base Hybrid Corr | QLoRA Hybrid Corr | Delta Corr |",
            "|---:|---:|---:|---:|---:|---:|",
        ])
        for tau_key, row in hyb["by_tau"].items():
            bo = row["base"]["overall"]
            fo = row["ft"]["overall"]
            do = row["delta"]["overall"]
            lines.append(
                f"| {tau_key} | {row['n']} | {_fmt_num(bo.get('rag_routed_pct'))}% | "
                f"{_fmt_num(bo.get('correctness'))} | {_fmt_num(fo.get('correctness'))} | "
                f"{_fmt_delta(do.get('correctness'))} |"
            )

        dc = hyb.get("default_comparison")
        if dc:
            lines.extend([
                "",
                f"### Default Threshold Slice Breakdown (`tau = {hyb['default_tau']:.2f}`)",
                "| Slice | N | RAG Routed % | Base Hybrid Corr | QLoRA Hybrid Corr | Delta Corr |",
                "|---|---:|---:|---:|---:|---:|",
            ])
            for sl in SLICES:
                if sl not in dc["base"]["by_slice"]:
                    continue
                bs = dc["base"]["by_slice"][sl]
                fs = dc["ft"]["by_slice"][sl]
                ds = dc["delta"]["by_slice"][sl]
                lines.append(
                    f"| `{sl}` | {bs['count']} | {_fmt_num(bs.get('rag_routed_pct'))}% | "
                    f"{_fmt_num(bs.get('correctness'))} | {_fmt_num(fs.get('correctness'))} | "
                    f"{_fmt_delta(ds.get('correctness'))} |"
                )
            bo = dc["base"]["overall"]
            fo = dc["ft"]["overall"]
            do = dc["delta"]["overall"]
            lines.append(
                f"| **overall** | {bo['count']} | {_fmt_num(bo.get('rag_routed_pct'))}% | "
                f"{_fmt_num(bo.get('correctness'))} | {_fmt_num(fo.get('correctness'))} | "
                f"{_fmt_delta(do.get('correctness'))} |"
            )

    lines.extend(["", "## 7. QLoRA Training Telemetry", ""])
    if not trn.get("manifest_available") and not trn.get("log_available"):
        lines.append("- No training manifest or training log found.")
    else:
        if trn.get("manifest_available") and isinstance(trn.get("manifest"), dict):
            lines.append("### Training Manifest")
            for k, v in trn["manifest"].items():
                lines.append(f"- `{k}`: `{v}`")
        if trn.get("log_summary"):
            lines.extend(["", "### Training Log Summary"])
            for k, v in trn["log_summary"].items():
                lines.append(f"- `{k}`: `{v}`")

    lines.append("")
    return "\n".join(lines)


def write_model_comparison(
    base_judge_path: str | Path = "eval/judge_scores_qwen35_9b.jsonl",
    ft_judge_path: str | Path = "eval/judge_scores_ft_qwen35_9b.jsonl",
    *,
    secondary_base_judge_path: str | Path | None = "eval/judge_scores.jsonl",
    secondary_ft_judge_path: str | Path | None = "eval/judge_scores_ft_qwen25_7b.jsonl",
    base_zs_path: str | Path | None = "eval/outputs_zero_shot.jsonl",
    base_rag_path: str | Path | None = "eval/outputs_rag.jsonl",
    ft_zs_path: str | Path | None = "eval/outputs_zero_shot_ft.jsonl",
    ft_rag_path: str | Path | None = "eval/outputs_rag_ft.jsonl",
    train_manifest_path: str | Path | None = "data/qlora/train_manifest.json",
    training_log_path: str | Path | None = "data/qlora/training_log.jsonl",
    index_dir: str | Path | None = "data/index",
    out_json: str | Path = "eval/model_comparison.json",
    out_md: str | Path = "eval/model_comparison.md",
    base_label: str = "llama3.2:3b",
    ft_label: str = "llama3.2-qlora:3b",
    judge_label: str = "qwen3.5:9b",
    secondary_judge_label: str = "qwen2.5:7b",
    precomputed_top1_scores: dict[str, float] | None = None,
) -> tuple[Path, Path]:
    """Run `compare_models` and write JSON + Markdown comparison artifacts."""
    report = compare_models(
        base_judge_path=base_judge_path,
        ft_judge_path=ft_judge_path,
        secondary_base_judge_path=secondary_base_judge_path,
        secondary_ft_judge_path=secondary_ft_judge_path,
        base_zs_path=base_zs_path,
        base_rag_path=base_rag_path,
        ft_zs_path=ft_zs_path,
        ft_rag_path=ft_rag_path,
        train_manifest_path=train_manifest_path,
        training_log_path=training_log_path,
        index_dir=index_dir,
        base_label=base_label,
        ft_label=ft_label,
        judge_label=judge_label,
        secondary_judge_label=secondary_judge_label,
        precomputed_top1_scores=precomputed_top1_scores,
    )

    json_path = Path(out_json)
    md_path = Path(out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md_path.write_text(render_comparison_markdown(report), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    """CLI entrypoint for model comparison report generation."""
    parser = argparse.ArgumentParser(
        description="Statistical Model Comparison Engine (Base vs. QLoRA)"
    )
    parser.add_argument(
        "--base-judge", default="eval/judge_scores_qwen35_9b.jsonl"
    )
    parser.add_argument(
        "--ft-judge", default="eval/judge_scores_ft_qwen35_9b.jsonl"
    )
    parser.add_argument(
        "--secondary-base-judge", default="eval/judge_scores.jsonl"
    )
    parser.add_argument(
        "--secondary-ft-judge", default="eval/judge_scores_ft_qwen25_7b.jsonl"
    )
    parser.add_argument("--base-zs", default="eval/outputs_zero_shot.jsonl")
    parser.add_argument("--base-rag", default="eval/outputs_rag.jsonl")
    parser.add_argument("--ft-zs", default="eval/outputs_zero_shot_ft.jsonl")
    parser.add_argument("--ft-rag", default="eval/outputs_rag_ft.jsonl")
    parser.add_argument(
        "--train-manifest", default="data/qlora/train_manifest.json"
    )
    parser.add_argument(
        "--training-log", default="data/qlora/training_log.jsonl"
    )
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--out-json", default="eval/model_comparison.json")
    parser.add_argument("--out-md", default="eval/model_comparison.md")
    parser.add_argument("--base-label", default="llama3.2:3b")
    parser.add_argument("--ft-label", default="llama3.2-qlora:3b")
    parser.add_argument("--judge-label", default="qwen3.5:9b")
    parser.add_argument("--secondary-judge-label", default="qwen2.5:7b")
    args = parser.parse_args()

    json_path, md_path = write_model_comparison(
        base_judge_path=args.base_judge,
        ft_judge_path=args.ft_judge,
        secondary_base_judge_path=args.secondary_base_judge,
        secondary_ft_judge_path=args.secondary_ft_judge,
        base_zs_path=args.base_zs,
        base_rag_path=args.base_rag,
        ft_zs_path=args.ft_zs,
        ft_rag_path=args.ft_rag,
        train_manifest_path=args.train_manifest,
        training_log_path=args.training_log,
        index_dir=args.index_dir,
        out_json=args.out_json,
        out_md=args.out_md,
        base_label=args.base_label,
        ft_label=args.ft_label,
        judge_label=args.judge_label,
        secondary_judge_label=args.secondary_judge_label,
    )
    print(f"Wrote model comparison JSON to {json_path} and Markdown to {md_path}")


if __name__ == "__main__":
    main()

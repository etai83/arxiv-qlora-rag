"""Generate the self-contained Tailwind + shadcn + Plotly interactive experiment book HTML."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

from src import read_jsonl

ABSTENTION_MARKERS = (
    "not in the sources",
    "i don't have",
    "i do not have",
    "cannot verify",
    "my knowledge cutoff",
)


def _is_abstention(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in ABSTENTION_MARKERS)


def _summarize_corpus(rows: list[dict[str, Any]], sample_limit: int = 25) -> dict[str, Any]:
    """Compute monthly counts, word-length stats, top categories, and a compact sample."""
    monthly = sorted(Counter(r["date"][:7] for r in rows).items())
    word_counts = [len(r["abstract"].split()) for r in rows]
    cats = Counter(cat for r in rows for cat in r["categories"]).most_common(8)

    # Spread samples across the full date span while keeping landmark/recent diversity.
    step = max(1, len(rows) // sample_limit)
    samples = [
        {
            "id": r["id"],
            "title": r["title"],
            "date": r["date"],
            "categories": r["categories"],
            "words": len(r["abstract"].split()),
            "abstract_preview": r["abstract"][:320] + ("..." if len(r["abstract"]) > 320 else ""),
        }
        for r in rows[::step][:sample_limit]
    ]
    return {
        "count": len(rows),
        "monthly": [{"month": m, "count": c} for m, c in monthly],
        "word_counts": word_counts,
        "word_stats": {
            "min": min(word_counts),
            "median": round(float(median(word_counts)), 1),
            "max": max(word_counts),
        },
        "top_categories": [{"category": k, "count": v} for k, v in cats],
        "samples": samples,
    }


def _build_behavior_counts(judge_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Classify every judged run into abstention, cited answer, or uncited answer."""
    slices = ["static", "live", "temporal_override", "unanswerable"]
    out: list[dict[str, Any]] = []
    for mode in ("zero_shot", "rag"):
        for sl in slices:
            sub = [r for r in judge_rows if r["mode"] == mode and r["slice"] == sl]
            abstained = sum(1 for r in sub if _is_abstention(r["model_answer"]))
            cited = sum(
                1
                for r in sub
                if not _is_abstention(r["model_answer"]) and r.get("extracted_cited_ids")
            )
            uncited = len(sub) - abstained - cited
            out.append(
                {
                    "mode": mode,
                    "slice": sl,
                    "n": len(sub),
                    "abstained": abstained,
                    "cited": cited,
                    "uncited": uncited,
                }
            )
    return out


def collect_book_payload(root: Path) -> dict[str, Any]:
    """Load all verified artifacts and compute compact visualization payloads."""
    env_manifest = json.loads((root / "corpus/env_manifest.json").read_text(encoding="utf-8"))
    split_meta = json.loads((root / "corpus/split_metadata.json").read_text(encoding="utf-8"))
    static_rows = read_jsonl(root / "corpus/static.jsonl")
    live_rows = read_jsonl(root / "corpus/live.jsonl")
    golden_rows = read_jsonl(root / "eval/golden.jsonl")
    judge_rows = read_jsonl(root / "eval/judge_scores.jsonl")
    judge35_rows = read_jsonl(root / "eval/judge_scores_qwen35_9b.jsonl")
    human_rows = read_jsonl(root / "eval/human_scores_20.jsonl")
    results_summary = json.loads((root / "eval/results_summary.json").read_text(encoding="utf-8"))
    judge_comparison = json.loads((root / "eval/judge_comparison.json").read_text(encoding="utf-8"))
    model_comparison = json.loads((root / "eval/model_comparison.json").read_text(encoding="utf-8"))
    qlora_dataset = json.loads((root / "data/qlora/dataset_manifest.json").read_text(encoding="utf-8"))
    qlora_train_manifest = json.loads((root / "data/qlora/train_manifest.json").read_text(encoding="utf-8"))
    qlora_training_log = read_jsonl(root / "data/qlora/training_log.jsonl")
    ft_zs_outputs = {r["id"]: r for r in read_jsonl(root / "eval/outputs_zero_shot_ft.jsonl")}
    ft_rag_outputs = {r["id"]: r for r in read_jsonl(root / "eval/outputs_rag_ft.jsonl")}
    ft_judge35_by_key = {
        (r["id"], r["mode"]): r for r in read_jsonl(root / "eval/judge_scores_ft_qwen35_9b.jsonl")
    }
    ft_judge25_by_key = {
        (r["id"], r["mode"]): r for r in read_jsonl(root / "eval/judge_scores_ft_qwen25_7b.jsonl")
    }
    rubric_md = (root / "eval/rubric.md").read_text(encoding="utf-8")
    notes_md = (root / "notes.md").read_text(encoding="utf-8")

    judge_by_key = {(r["id"], r["mode"]): r for r in judge_rows}
    judge35_by_key = {(r["id"], r["mode"]): r for r in judge35_rows}
    audit_outcome_by_key = {
        (item["id"], item["mode"]): item["outcome"]
        for item in judge_comparison.get("human_calibration", {}).get("audit_items", [])
    }

    # Build 20-item audit table + 5x5 confusion matrix for Page 6.
    matrix = [[0 for _ in range(5)] for _ in range(5)]
    audit_items: list[dict[str, Any]] = []
    for h in human_rows:
        key = (h["id"], h["mode"])
        j = judge_by_key[key]
        j35 = judge35_by_key[key]
        for dim in ("correctness", "faithfulness", "citation_accuracy"):
            matrix[int(h[dim]) - 1][int(j[dim]) - 1] += 1
        audit_items.append(
            {
                "sample_id": h["sample_id"],
                "id": h["id"],
                "mode": h["mode"],
                "slice": h["slice"],
                "outcome": audit_outcome_by_key.get(key, "tied_miss"),
                "human": {
                    "correctness": h["correctness"],
                    "faithfulness": h["faithfulness"],
                    "citation_accuracy": h["citation_accuracy"],
                    "override_obedience": h["override_obedience"],
                    "notes": h["human_notes"],
                },
                "judge": {
                    "correctness": j["correctness"],
                    "faithfulness": j["faithfulness"],
                    "citation_accuracy": j["citation_accuracy"],
                    "override_obedience": j["override_obedience"],
                    "rationale": j["rationale"],
                },
                "judge_qwen35": {
                    "correctness": j35["correctness"],
                    "faithfulness": j35["faithfulness"],
                    "citation_accuracy": j35["citation_accuracy"],
                    "override_obedience": j35["override_obedience"],
                    "rationale": j35["rationale"],
                },
            }
        )

    # Compact per-prompt paired runs (60 prompts x 4 model-mode variants) for Page 7 Run Inspector & latency charts.
    paired_runs: list[dict[str, Any]] = []
    for g in golden_rows:
        gid = g["id"]
        zs = judge_by_key[(gid, "zero_shot")]
        rg = judge_by_key[(gid, "rag")]
        zs35 = judge35_by_key[(gid, "zero_shot")]
        rg35 = judge35_by_key[(gid, "rag")]
        ft_zs_out = ft_zs_outputs[gid]
        ft_rag_out = ft_rag_outputs[gid]
        ft_zs25 = ft_judge25_by_key[(gid, "zero_shot")]
        ft_rg25 = ft_judge25_by_key[(gid, "rag")]
        ft_zs35 = ft_judge35_by_key[(gid, "zero_shot")]
        ft_rg35 = ft_judge35_by_key[(gid, "rag")]
        ctx_preview = (rg.get("retrieved_context") or "")[:1400]
        if len(rg.get("retrieved_context") or "") > 1400:
            ctx_preview += "\n... [truncated]"
        ft_ctx_raw = ft_rag_out.get("retrieved_context") or ""
        ft_ctx_preview = ft_ctx_raw[:1400]
        if len(ft_ctx_raw) > 1400:
            ft_ctx_preview += "\n... [truncated]"
        paired_runs.append(
            {
                "id": gid,
                "slice": g["slice"],
                "prompt": g["prompt"],
                "reference_answer": g["reference_answer"],
                "source_paper_ids": g["source_paper_ids"],
                "newer_paper_id": g.get("newer_paper_id"),
                "zero_shot": {
                    "answer": zs["model_answer"],
                    "cited": zs["extracted_cited_ids"],
                    "latency_ms": zs["latency_ms"],
                    "correctness": zs["correctness"],
                    "faithfulness": zs["faithfulness"],
                    "citation_accuracy": zs["citation_accuracy"],
                    "override_obedience": zs["override_obedience"],
                    "rationale": zs["rationale"],
                    "qwen35": {
                        "correctness": zs35["correctness"],
                        "faithfulness": zs35["faithfulness"],
                        "citation_accuracy": zs35["citation_accuracy"],
                        "override_obedience": zs35["override_obedience"],
                        "rationale": zs35["rationale"],
                        "judge_latency_ms": zs35.get("judge_latency_ms"),
                        "judge_tok_per_s": zs35.get("judge_tok_per_s"),
                    },
                },
                "rag": {
                    "answer": rg["model_answer"],
                    "cited": rg["extracted_cited_ids"],
                    "valid_context_ids": rg.get("valid_context_ids", []),
                    "context_preview": ctx_preview,
                    "latency_ms": rg["latency_ms"],
                    "retrieval_ms": rg.get("retrieval_ms"),
                    "generation_ms": rg.get("generation_ms"),
                    "correctness": rg["correctness"],
                    "faithfulness": rg["faithfulness"],
                    "citation_accuracy": rg["citation_accuracy"],
                    "override_obedience": rg["override_obedience"],
                    "rationale": rg["rationale"],
                    "qwen35": {
                        "correctness": rg35["correctness"],
                        "faithfulness": rg35["faithfulness"],
                        "citation_accuracy": rg35["citation_accuracy"],
                        "override_obedience": rg35["override_obedience"],
                        "rationale": rg35["rationale"],
                        "judge_latency_ms": rg35.get("judge_latency_ms"),
                        "judge_tok_per_s": rg35.get("judge_tok_per_s"),
                    },
                },
                "zero_shot_ft": {
                    "answer": ft_zs_out["model_answer"],
                    "cited": ft_zs_out.get("extracted_cited_ids", []),
                    "latency_ms": ft_zs_out["latency_ms"],
                    "correctness": ft_zs25["correctness"],
                    "faithfulness": ft_zs25["faithfulness"],
                    "citation_accuracy": ft_zs25["citation_accuracy"],
                    "override_obedience": ft_zs25["override_obedience"],
                    "rationale": ft_zs25["rationale"],
                    "qwen35": {
                        "correctness": ft_zs35["correctness"],
                        "faithfulness": ft_zs35["faithfulness"],
                        "citation_accuracy": ft_zs35["citation_accuracy"],
                        "override_obedience": ft_zs35["override_obedience"],
                        "rationale": ft_zs35["rationale"],
                        "judge_latency_ms": ft_zs35.get("judge_latency_ms"),
                        "judge_tok_per_s": ft_zs35.get("judge_tok_per_s"),
                    },
                },
                "rag_ft": {
                    "answer": ft_rag_out["model_answer"],
                    "cited": ft_rag_out.get("extracted_cited_ids", []),
                    "valid_context_ids": ft_rag_out.get("valid_context_ids", []),
                    "context_preview": ft_ctx_preview,
                    "latency_ms": ft_rag_out["latency_ms"],
                    "retrieval_ms": ft_rag_out.get("retrieval_ms"),
                    "generation_ms": ft_rag_out.get("generation_ms"),
                    "correctness": ft_rg25["correctness"],
                    "faithfulness": ft_rg25["faithfulness"],
                    "citation_accuracy": ft_rg25["citation_accuracy"],
                    "override_obedience": ft_rg25["override_obedience"],
                    "rationale": ft_rg25["rationale"],
                    "qwen35": {
                        "correctness": ft_rg35["correctness"],
                        "faithfulness": ft_rg35["faithfulness"],
                        "citation_accuracy": ft_rg35["citation_accuracy"],
                        "override_obedience": ft_rg35["override_obedience"],
                        "rationale": ft_rg35["rationale"],
                        "judge_latency_ms": ft_rg35.get("judge_latency_ms"),
                        "judge_tok_per_s": ft_rg35.get("judge_tok_per_s"),
                    },
                },
            }
        )

    rag_rows = [r for r in judge_rows if r["mode"] == "rag"]
    rag_ret_by_slice = {}
    for sl in ("static", "live", "temporal_override", "unanswerable"):
        sub = [r for r in rag_rows if r["slice"] == sl]
        rag_ret_by_slice[sl] = {
            "retrieval_mean_ms": round(sum(r["retrieval_ms"] for r in sub) / len(sub), 2),
            "generation_mean_ms": round(sum(r["generation_ms"] for r in sub) / len(sub), 2),
        }

    return {
        "env": {
            "host": env_manifest["host"]["hardware"],
            "os": env_manifest["host"]["macos_product_version"],
            "uv": env_manifest["tools"]["uv"]["version"],
            "python": env_manifest["tools"]["python3_13"]["version"],
            "ollama_version": env_manifest["tools"]["ollama"]["api_version"],
            "models": [
                {
                    "name": m["name"],
                    "size_gb": round(m["size"] / 1e9, 2),
                    "params": m["details"].get("parameter_size") or ("9.7B" if "qwen3.5:9b" in m["name"] else ("3.2B" if "llama3.2-qlora" in m["name"] else "")),
                    "quant": (
                        f"{m['details'].get('quantization_level')} (MLX)"
                        if m["details"].get("quantization_level") and m["details"].get("runner") == "mlx"
                        else (m["details"].get("quantization_level") or ("nvfp4 (MLX)" if m["details"].get("runner") == "mlx" else m["details"].get("format", "")))
                    ),
                    "ctx": m["details"].get("context_length") or (262144 if "qwen3.5:9b" in m["name"] else (131072 if "llama3.2-qlora" in m["name"] else None)),
                    "digest": m["digest"][:16],
                }
                for m in env_manifest["models"]
            ],
        },
        "split": split_meta,
        "corpus": {
            "static": _summarize_corpus(static_rows),
            "live": _summarize_corpus(live_rows),
        },
        "results": results_summary,
        "judge_comparison": judge_comparison,
        "model_comparison": model_comparison,
        "qlora_dataset": qlora_dataset,
        "qlora_training": {
            "manifest": qlora_train_manifest,
            "log": qlora_training_log,
        },
        "behavior": _build_behavior_counts(judge_rows),
        "rag_latency_split": rag_ret_by_slice,
        "calibration": {
            "v1": {
                "correctness": 55.0,
                "faithfulness": 45.0,
                "citation_accuracy": 45.0,
                "pooled_exact": 48.33,
                "within_1": 55.0,
                "override": 100.0,
            },
            "v2": {
                "correctness": results_summary["agreement"]["dimensions"]["correctness"]["exact_match_pct"],
                "faithfulness": results_summary["agreement"]["dimensions"]["faithfulness"]["exact_match_pct"],
                "citation_accuracy": results_summary["agreement"]["dimensions"]["citation_accuracy"]["exact_match_pct"],
                "pooled_exact": results_summary["agreement"]["pooled"]["exact_match_pct"],
                "within_1": results_summary["agreement"]["pooled"]["within_1_pct"],
                "override": results_summary["agreement"]["override_exact_pct"],
            },
            "confusion_matrix": matrix,
            "audit_items": audit_items,
        },
        "runs": paired_runs,
        "rubric_md": rubric_md,
        "notes_md": notes_md,
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="color-scheme" content="light dark">
  <script>
    try {
      document.querySelector('meta[name="color-scheme"]').content = localStorage.getItem("color-scheme") ?? "light dark";
    } catch {}
  </script>
  <title>arXiv Hybrid LLM — Interactive Experiment Book</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;0,6..72,700;1,6..72,400&display=swap" rel="stylesheet">
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: {
            sans: ['Inter', 'system-ui', 'sans-serif'],
            serif: ['Newsreader', 'Georgia', 'serif'],
            mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
          },
          colors: {
            background: 'var(--background)',
            foreground: 'var(--foreground)',
            card: { DEFAULT: 'var(--card)', foreground: 'var(--card-foreground)' },
            popover: { DEFAULT: 'var(--popover)', foreground: 'var(--popover-foreground)' },
            primary: { DEFAULT: 'var(--primary)', foreground: 'var(--primary-foreground)' },
            secondary: { DEFAULT: 'var(--secondary)', foreground: 'var(--secondary-foreground)' },
            muted: { DEFAULT: 'var(--muted)', foreground: 'var(--muted-foreground)' },
            accent: { DEFAULT: 'var(--accent)', foreground: 'var(--accent-foreground)' },
            border: 'var(--border)',
            ring: 'var(--ring)',
          }
        }
      }
    };
  </script>
  <style>
    :root {
      --bg-light: #f6f4ee;
      --bg-dark: #09090b;
      --fg-light: #18181b;
      --fg-dark: #f4f4f5;
      --card-light: #ffffff;
      --card-dark: #111114;
      --muted-light: #edeae0;
      --muted-dark: #1c1c21;
      --muted-fg-light: #52525b;
      --muted-fg-dark: #a1a1aa;
      --border-light: #dcd7c9;
      --border-dark: #27272a;
      --primary-light: #1e293b;
      --primary-dark: #e2e8f0;
      --primary-fg-light: #f8fafc;
      --primary-fg-dark: #09090b;
      --secondary-light: #f1efe7;
      --secondary-dark: #1f2026;
      --accent-light: #e7e2d3;
      --accent-dark: #27272e;
      --paper-shadow-light: 0 10px 35px -12px rgba(24, 24, 27, 0.14), 0 1px 4px rgba(24, 24, 27, 0.06);
      --paper-shadow-dark: 0 14px 40px -12px rgba(0, 0, 0, 0.65), 0 0 0 1px rgba(255, 255, 255, 0.05);
      --sb-track-light: #eae6db;
      --sb-track-dark: #18181b;
      --sb-thumb-light: #71717a;
      --sb-thumb-dark: #52525b;

      --background: var(--bg-light);
      --foreground: var(--fg-light);
      --card: var(--card-light);
      --card-foreground: var(--fg-light);
      --popover: var(--card-light);
      --popover-foreground: var(--fg-light);
      --muted: var(--muted-light);
      --muted-foreground: var(--muted-fg-light);
      --border: var(--border-light);
      --ring: #4f46e5;
      --primary: var(--primary-light);
      --primary-foreground: var(--primary-fg-light);
      --secondary: var(--secondary-light);
      --secondary-foreground: var(--fg-light);
      --accent: var(--accent-light);
      --accent-foreground: var(--fg-light);
      --paper-shadow: var(--paper-shadow-light);
      --scrollbar-track: var(--sb-track-light);
      --scrollbar-thumb: var(--sb-thumb-light);

      @media (prefers-color-scheme: dark) {
        --background: var(--bg-dark);
        --foreground: var(--fg-dark);
        --card: var(--card-dark);
        --card-foreground: var(--fg-dark);
        --popover: var(--card-dark);
        --popover-foreground: var(--fg-dark);
        --muted: var(--muted-dark);
        --muted-foreground: var(--muted-fg-dark);
        --border: var(--border-dark);
        --primary: var(--primary-dark);
        --primary-foreground: var(--primary-fg-dark);
        --secondary: var(--secondary-dark);
        --secondary-foreground: var(--fg-dark);
        --accent: var(--accent-dark);
        --accent-foreground: var(--fg-dark);
        --paper-shadow: var(--paper-shadow-dark);
        --scrollbar-track: var(--sb-track-dark);
        --scrollbar-thumb: var(--sb-thumb-dark);
      }

      @supports (color: light-dark(white, black)) {
        --background: light-dark(var(--bg-light), var(--bg-dark));
        --foreground: light-dark(var(--fg-light), var(--fg-dark));
        --card: light-dark(var(--card-light), var(--card-dark));
        --card-foreground: light-dark(var(--fg-light), var(--fg-dark));
        --popover: light-dark(var(--card-light), var(--card-dark));
        --popover-foreground: light-dark(var(--fg-light), var(--fg-dark));
        --muted: light-dark(var(--muted-light), var(--muted-dark));
        --muted-foreground: light-dark(var(--muted-fg-light), var(--muted-fg-dark));
        --border: light-dark(var(--border-light), var(--border-dark));
        --primary: light-dark(var(--primary-light), var(--primary-dark));
        --primary-foreground: light-dark(var(--primary-fg-light), var(--primary-fg-dark));
        --secondary: light-dark(var(--secondary-light), var(--secondary-dark));
        --secondary-foreground: light-dark(var(--fg-light), var(--fg-dark));
        --accent: light-dark(var(--accent-light), var(--accent-dark));
        --accent-foreground: light-dark(var(--fg-light), var(--fg-dark));
        --scrollbar-track: light-dark(var(--sb-track-light), var(--sb-track-dark));
        --scrollbar-thumb: light-dark(var(--sb-thumb-light), var(--sb-thumb-dark));
      }

      color-scheme: light dark;
      accent-color: light-dark(#2563eb, #60a5fa);
      scrollbar-color: var(--scrollbar-thumb) var(--scrollbar-track);

      &:has(> head > meta[name="color-scheme"][content="dark"]) {
        color-scheme: dark;
        --paper-shadow: var(--paper-shadow-dark);
      }
      &:has(> head > meta[name="color-scheme"][content="light"]) {
        color-scheme: light;
        --paper-shadow: var(--paper-shadow-light);
      }
    }

    @supports not (scrollbar-color: auto) {
      .book-scroll { scrollbar-width: auto; }
      .book-scroll::-webkit-scrollbar { width: 10px; height: 10px; }
      .book-scroll::-webkit-scrollbar-thumb { background: var(--scrollbar-thumb); border-radius: 6px; }
      .book-scroll::-webkit-scrollbar-track { background: var(--scrollbar-track); }
    }

    /* Directional View Transitions for Book Page Turns */
    @keyframes slide-to-left { to { transform: translateX(-6%); opacity: 0; } }
    @keyframes slide-from-right { from { transform: translateX(6%); opacity: 0; } }
    @keyframes slide-to-right { to { transform: translateX(6%); opacity: 0; } }
    @keyframes slide-from-left { from { transform: translateX(-6%); opacity: 0; } }

    ::view-transition-group(root) {
      animation: 0.34s cubic-bezier(0.22, 1, 0.36, 1) both;
    }
    html:active-view-transition-type(forward)::view-transition-old(root) {
      animation-name: slide-to-left;
    }
    html:active-view-transition-type(forward)::view-transition-new(root) {
      animation-name: slide-from-right;
    }
    html:active-view-transition-type(backward)::view-transition-old(root) {
      animation-name: slide-to-right;
    }
    html:active-view-transition-type(backward)::view-transition-new(root) {
      animation-name: slide-from-left;
    }
    @media (prefers-reduced-motion: reduce) {
      ::view-transition-group(root) {
        animation: none !important;
      }
    }

    /* Responsive Data Tables with Sticky Headers & @container Card Stack */
    .table-wrapper {
      overflow: auto;
      max-inline-size: 100%;
      max-block-size: min(460px, 72vh);
      container-type: inline-size;
      border: 1px solid var(--border);
      border-radius: 0.625rem;
      background: var(--card);
    }
    .table-wrapper table {
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      font-size: 0.875rem;
    }
    .table-wrapper thead {
      position: sticky;
      inset-block-start: 0;
      z-index: 2;
      background: var(--muted);
    }
    .table-wrapper th, .table-wrapper td {
      padding: 0.65rem 0.85rem;
      border-block-end: 1px solid var(--border);
      text-align: start;
      vertical-align: top;
    }
    .table-wrapper tbody th {
      position: sticky;
      inset-inline-start: 0;
      z-index: 1;
      background: var(--card);
      font-weight: 600;
    }
    @container (width < 600px) {
      .table-wrapper table,
      .table-wrapper thead,
      .table-wrapper tbody,
      .table-wrapper tr,
      .table-wrapper th,
      .table-wrapper td {
        display: block;
      }
      .table-wrapper thead {
        display: none;
      }
      .table-wrapper tr {
        margin: 0.75rem;
        border: 1px solid var(--border);
        border-radius: 0.5rem;
        overflow: hidden;
      }
      .table-wrapper tbody th {
        position: sticky;
        inset-block-start: 0;
        z-index: 2;
        background: var(--muted);
        border-block-end: 1px solid var(--border);
      }
      .table-wrapper td {
        display: flex;
        justify-content: space-between;
        gap: 0.75rem;
        border-block-end: 1px solid var(--border);
      }
      .table-wrapper td:last-child { border-block-end: 0; }
      .table-wrapper td::before {
        content: attr(data-label) ": ";
        font-weight: 600;
        color: var(--muted-foreground);
        flex-shrink: 0;
      }
    }

    /* Accessible Tabs (intra-page only) */
    .tablist {
      display: flex;
      overflow-x: auto;
      gap: 0.25rem;
      border-block-end: 1px solid var(--border);
    }
    .tablist [role="tab"] {
      position: relative;
      flex: none;
      padding: 0.625rem 0.95rem;
      font-size: 0.875rem;
      font-weight: 500;
      color: var(--muted-foreground);
      background: transparent;
      border: 0;
      cursor: pointer;
    }
    .tablist [role="tab"][aria-selected="true"] {
      color: var(--foreground);
    }
    .tablist [role="tab"][aria-selected="true"]::after {
      content: "";
      position: absolute;
      inset: auto 0 0;
      border-block-end: 3px solid var(--primary);
    }
    .tabpanel-box {
      padding-top: 1rem;
    }

    /* Book Sheet Styling */
    .book-sheet {
      background: var(--card);
      border: 1px solid var(--border);
      box-shadow: var(--paper-shadow);
    }
    .drop-cap::first-letter {
      float: left;
      font-family: 'Newsreader', Georgia, serif;
      font-size: 3.1rem;
      line-height: 0.85;
      padding-right: 0.55rem;
      padding-top: 0.15rem;
      font-weight: 600;
      color: var(--primary);
    }
  </style>
</head>
<body class="bg-background text-foreground font-sans antialiased min-h-screen flex flex-col">
  <!-- Top Running Book Chrome -->
  <header class="sticky top-0 z-30 border-b border-border bg-card/90 backdrop-blur">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between gap-4">
      <div class="flex items-center gap-3 min-w-0">
        <button id="btn-spine-toggle" type="button" class="lg:hidden inline-flex items-center justify-center rounded-md border border-border px-2.5 py-1.5 text-xs font-medium hover:bg-accent" aria-expanded="false" aria-controls="book-spine">
          Chapters
        </button>
        <span class="hidden sm:inline-flex items-center rounded-md bg-muted px-2 py-0.5 font-mono text-xs text-muted-foreground">arXiv cs.CL</span>
        <span id="header-chapter-label" class="font-serif text-sm sm:text-base font-semibold truncate">Prologue · Abstract &amp; Agenda</span>
      </div>

      <div class="flex items-center gap-3">
        <a href="#page-1" data-nav-link="0" class="rounded-md border border-border bg-secondary px-2 py-1 font-mono text-xs hover:bg-accent transition-colors">↑ Agenda</a>
        <span id="header-folio" class="font-mono text-xs text-muted-foreground hidden sm:inline">Page 1 / 12</span>
        <!-- Appearance Settings Disclosure (tri-state System / Light / Dark) -->
        <details id="appearance-popover" class="relative">
          <summary class="list-none cursor-pointer inline-flex items-center gap-1.5 rounded-md border border-border bg-secondary px-2.5 py-1.5 text-xs font-medium hover:bg-accent">
            <span aria-hidden="true">◐</span>
            <span>Appearance</span>
          </summary>
          <div class="absolute right-0 mt-2 w-56 rounded-lg border border-border bg-popover text-popover-foreground p-3 shadow-xl z-50">
            <p class="text-xs font-semibold mb-2">Color Scheme</p>
            <div class="grid grid-cols-3 gap-1.5" role="group" aria-label="Theme preference">
              <button type="button" data-theme-choice="system" class="theme-btn rounded-md border border-border px-2 py-1.5 text-xs font-medium hover:bg-accent">System</button>
              <button type="button" data-theme-choice="light" class="theme-btn rounded-md border border-border px-2 py-1.5 text-xs font-medium hover:bg-accent">Light</button>
              <button type="button" data-theme-choice="dark" class="theme-btn rounded-md border border-border px-2 py-1.5 text-xs font-medium hover:bg-accent">Dark</button>
            </div>
            <p id="theme-status-note" class="mt-2 text-[11px] text-muted-foreground">Following system preference</p>
          </div>
        </details>
      </div>
    </div>
    <div class="h-0.5 bg-muted w-full">
      <div id="reading-progress" class="h-full bg-primary transition-all duration-300" style="width: 8.33%"></div>
    </div>
  </header>

  <!-- Main Book Workspace -->
  <div class="max-w-7xl w-full mx-auto px-4 sm:px-6 py-6 flex-1 grid grid-cols-1 lg:grid-cols-[260px_minmax(0,1fr)] gap-6 items-start">
    <!-- Left Book Spine / Table of Contents -->
    <aside id="book-spine" class="hidden lg:block sticky top-20 rounded-xl border border-border bg-card p-4 shadow-sm">
      <div class="pb-3 mb-3 border-b border-border">
        <p class="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">Monograph Index</p>
        <h2 class="font-serif text-lg font-semibold mt-0.5">Experiment Lab Book</h2>
      </div>
      <nav aria-label="Book chapters" id="chapter-nav" class="space-y-1">
        <a href="#page-1" data-nav-link="0" aria-current="page" class="block w-full text-left rounded-lg px-3 py-2 text-xs font-semibold bg-primary text-primary-foreground transition-colors"><div class="font-mono text-[10px] opacity-75">Prologue</div><div class="truncate">Abstract &amp; Agenda</div></a>
        <a href="#page-2" data-nav-link="1" aria-current="false" class="block w-full text-left rounded-lg px-3 py-2 text-xs font-medium text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"><div class="font-mono text-[10px] opacity-75">Chapter 1</div><div class="truncate">Slices, Metrics &amp; Agents</div></a>
        <a href="#page-3" data-nav-link="2" aria-current="false" class="block w-full text-left rounded-lg px-3 py-2 text-xs font-medium text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"><div class="font-mono text-[10px] opacity-75">Chapter 2</div><div class="truncate">Leakage-Safe Date Split</div></a>
        <a href="#page-4" data-nav-link="3" aria-current="false" class="block w-full text-left rounded-lg px-3 py-2 text-xs font-medium text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"><div class="font-mono text-[10px] opacity-75">Chapter 3</div><div class="truncate">Ingestion &amp; FAISS Retrieval</div></a>
        <a href="#page-5" data-nav-link="4" aria-current="false" class="block w-full text-left rounded-lg px-3 py-2 text-xs font-medium text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"><div class="font-mono text-[10px] opacity-75">Chapter 4</div><div class="truncate">60-Prompt Golden Set</div></a>
        <a href="#page-6" data-nav-link="5" aria-current="false" class="block w-full text-left rounded-lg px-3 py-2 text-xs font-medium text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"><div class="font-mono text-[10px] opacity-75">Chapter 5</div><div class="truncate">Rubric &amp; Judge Calibration</div></a>
        <a href="#page-7" data-nav-link="6" aria-current="false" class="block w-full text-left rounded-lg px-3 py-2 text-xs font-medium text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"><div class="font-mono text-[10px] opacity-75">Chapter 6</div><div class="truncate">Baselines &amp; Run Inspector</div></a>
        <a href="#page-8" data-nav-link="7" aria-current="false" class="block w-full text-left rounded-lg px-3 py-2 text-xs font-medium text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"><div class="font-mono text-[10px] opacity-75">Chapter 7</div><div class="truncate">QLoRA Fine-Tuning</div></a>
        <a href="#page-9" data-nav-link="8" aria-current="false" class="block w-full text-left rounded-lg px-3 py-2 text-xs font-medium text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"><div class="font-mono text-[10px] opacity-75">Chapter 8</div><div class="truncate">Failure Analysis &amp; Roadmap</div></a>
        <div class="pt-2 mt-2 border-t border-border"></div>
        <a href="#page-10" data-nav-link="9" aria-current="false" class="block w-full text-left rounded-lg px-3 py-2 text-xs font-medium text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"><div class="font-mono text-[10px] opacity-75">Appendix A</div><div class="truncate">Compute &amp; Orchestration</div></a>
        <a href="#page-11" data-nav-link="10" aria-current="false" class="block w-full text-left rounded-lg px-3 py-2 text-xs font-medium text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"><div class="font-mono text-[10px] opacity-75">Appendix B</div><div class="truncate">3 Research Continuations</div></a>
        <a href="#page-12" data-nav-link="11" aria-current="false" class="block w-full text-left rounded-lg px-3 py-2 text-xs font-medium text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"><div class="font-mono text-[10px] opacity-75">Appendix C</div><div class="truncate">Stronger Judge (Qwen3.5-9B)</div></a>
      </nav>
      <div class="mt-4 pt-3 border-t border-border text-xs text-muted-foreground space-y-1">
        <div class="flex justify-between"><span>Cutoff T</span><span class="font-mono text-foreground">2025-01-01</span></div>
        <div class="flex justify-between gap-2"><span>Base / FT LLMs</span><span class="font-mono text-foreground text-right">llama3.2:3b · llama3.2-qlora:3b</span></div>
        <div class="flex justify-between"><span>Judges</span><span class="font-mono text-foreground">qwen2.5:7b · qwen3.5:9b</span></div>
        <div class="flex justify-between"><span>Shortcut</span><span class="font-mono text-foreground">← / → keys</span></div>
      </div>
    </aside>

    <!-- Central Book Page Sheet -->
    <main class="min-w-0">
      <article id="book-page-sheet" class="book-sheet rounded-2xl p-6 sm:p-10 min-h-[760px] flex flex-col justify-between">
        <!-- Page 1: Cover, Abstract & Agenda -->
        <section id="page-1" data-page="0" class="book-page space-y-8">
          <div class="border-b border-border pb-6 space-y-4">
            <div class="flex flex-wrap items-center gap-2">
              <span class="inline-flex items-center rounded-full border border-border bg-secondary px-3 py-0.5 font-mono text-xs">Base + QLoRA Monograph</span>
              <span class="inline-flex items-center rounded-full border border-border bg-muted px-3 py-0.5 font-mono text-xs">Mac mini M5 Pro · 48 GB</span>
              <span class="inline-flex items-center rounded-full border border-border bg-muted px-3 py-0.5 font-mono text-xs">Ollama 0.40.18 + MLX QLoRA + FAISS</span>
            </div>
            <h1 class="font-serif text-3xl sm:text-5xl font-bold tracking-tight leading-tight">
              Leakage-Safe Temporal RAG &amp; QLoRA on arXiv <span class="font-mono text-2xl sm:text-4xl">cs.CL</span>
            </h1>
            <p class="text-base sm:text-lg text-muted-foreground max-w-3xl">
              Date-Split Corpus Construction, <code class="font-mono text-sm">bge-small-en-v1.5</code> Vector Ingestion, 60-Prompt 4-Slice Golden Benchmark, Calibrated Cross-Family Judging, Zero-Shot vs. RAG Baselines, and Pre-T QLoRA Fine-Tuning (<code class="font-mono text-sm">llama3.2-qlora:3b</code>).
            </p>
          </div>

          <!-- Abstract -->
          <div class="rounded-xl border border-border bg-secondary/60 p-6 space-y-3">
            <div class="flex items-center justify-between">
              <h2 class="font-serif text-xl font-semibold">Abstract</h2>
              <span class="font-mono text-xs text-muted-foreground">T = 2025-01-01 · 500 Abstracts · 240 Eval Runs</span>
            </div>
            <p class="drop-cap text-sm sm:text-base leading-relaxed text-foreground/90">
              Evaluating Retrieval-Augmented Generation (RAG) and QLoRA adaptation requires strict separation between parametric pretraining knowledge and post-cutoff evidence. We select cutoff date <code class="font-mono text-xs">T = 2025-01-01</code>—strictly after <code class="font-mono text-xs">Llama-3.2-3B-Instruct</code>'s December 2023 training cutoff—and curate 300 pre-T (<code class="font-mono text-xs">static</code>) and 200 post-T (<code class="font-mono text-xs">live</code>) arXiv <code class="font-mono text-xs">cs.CL</code> abstracts with zero ID overlap. Under calibrated <code class="font-mono text-xs">Qwen3.5-9B</code> MLX judging (<strong>90.0%</strong> human exact match, $\kappa_w = 0.782$), Base RAG (<code class="font-mono text-xs">k=4</code>) lifts <code class="font-mono text-xs">live</code> correctness from <strong>1.00 to 4.85</strong> while maintaining <strong>5.00/5.00</strong> abstention on <code class="font-mono text-xs">unanswerable</code>, yet drops <code class="font-mono text-xs">static</code> to <strong>1.80</strong>. Fine-tuning a leakage-safe 4-bit MLX QLoRA adapter (<code class="font-mono text-xs">llama3.2-qlora:3b</code>, 1,060 pre-T train examples, checkpoint iter 400) yields <strong>+0.133</strong> pooled correctness (<strong>2.692 &rarr; 2.825</strong>, 95% CI <code class="font-mono text-xs">[-0.092, +0.358]</code>), surges on RAG <code class="font-mono text-xs">temporal_override</code> (<strong>3.60 &rarr; 4.80</strong>, override obedience <strong>0.50 &rarr; 0.80</strong>) and Zero-Shot <code class="font-mono text-xs">static</code> (<strong>1.15 &rarr; 1.60</strong>), and cuts latency <strong>2.5&times;</strong> (<strong>-1,078.9 ms</strong>), while exposing zero-shot overconfidence on unseen <code class="font-mono text-xs">live</code>/<code class="font-mono text-xs">unanswerable</code> prompts and terse <code class="font-mono text-xs">live</code> RAG omissions.
            </p>
          </div>

          <!-- KPI Strip -->
          <div class="grid grid-cols-2 lg:grid-cols-4 gap-4" id="p1-kpis"></div>

          <!-- Interactive Agenda -->
          <div class="space-y-3">
            <div class="flex items-baseline justify-between">
              <h2 class="font-serif text-2xl font-semibold">Experiment Agenda</h2>
              <span class="text-xs text-muted-foreground">Click any chapter card to turn page</span>
            </div>
            <div id="p1-agenda-grid" class="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <a href="#page-1" data-nav-link="0" class="group cursor-pointer text-left rounded-xl border border-border bg-card p-4 hover:bg-accent/60 hover:border-primary/40 transition-colors flex flex-col justify-between gap-2 no-underline text-foreground">
                <div class="flex items-center justify-between w-full"><span class="font-mono text-xs text-muted-foreground">Prologue</span><span class="font-mono text-xs rounded-md bg-secondary px-2 py-0.5 text-primary group-hover:bg-primary group-hover:text-primary-foreground transition-colors">Page 1 →</span></div>
                <div><p class="font-serif text-base font-semibold group-hover:underline">Abstract &amp; Agenda</p><p class="text-xs text-muted-foreground mt-0.5">Motivation, temporal cutoff T=2025-01-01, executive KPIs, and Base vs. QLoRA overview.</p></div>
              </a>
              <a href="#page-2" data-nav-link="1" class="group cursor-pointer text-left rounded-xl border border-border bg-card p-4 hover:bg-accent/60 hover:border-primary/40 transition-colors flex flex-col justify-between gap-2 no-underline text-foreground">
                <div class="flex items-center justify-between w-full"><span class="font-mono text-xs text-muted-foreground">Chapter 1</span><span class="font-mono text-xs rounded-md bg-secondary px-2 py-0.5 text-primary group-hover:bg-primary group-hover:text-primary-foreground transition-colors">Page 2 →</span></div>
                <div><p class="font-serif text-base font-semibold group-hover:underline">Slices, Metrics &amp; Agents</p><p class="text-xs text-muted-foreground mt-0.5">4 golden slices (+/- examples), 1–5 metrics, runtime agents &amp; orchestration subagents.</p></div>
              </a>
              <a href="#page-3" data-nav-link="2" class="group cursor-pointer text-left rounded-xl border border-border bg-card p-4 hover:bg-accent/60 hover:border-primary/40 transition-colors flex flex-col justify-between gap-2 no-underline text-foreground">
                <div class="flex items-center justify-between w-full"><span class="font-mono text-xs text-muted-foreground">Chapter 2</span><span class="font-mono text-xs rounded-md bg-secondary px-2 py-0.5 text-primary group-hover:bg-primary group-hover:text-primary-foreground transition-colors">Page 3 →</span></div>
                <div><p class="font-serif text-base font-semibold group-hover:underline">Leakage-Safe Date Split</p><p class="text-xs text-muted-foreground mt-0.5">300 static (&lt;T) vs. 200 live (&gt;=T) cs.CL abstracts with 0 ID overlap.</p></div>
              </a>
              <a href="#page-4" data-nav-link="3" class="group cursor-pointer text-left rounded-xl border border-border bg-card p-4 hover:bg-accent/60 hover:border-primary/40 transition-colors flex flex-col justify-between gap-2 no-underline text-foreground">
                <div class="flex items-center justify-between w-full"><span class="font-mono text-xs text-muted-foreground">Chapter 3</span><span class="font-mono text-xs rounded-md bg-secondary px-2 py-0.5 text-primary group-hover:bg-primary group-hover:text-primary-foreground transition-colors">Page 4 →</span></div>
                <div><p class="font-serif text-base font-semibold group-hover:underline">Ingestion &amp; FAISS Retrieval</p><p class="text-xs text-muted-foreground mt-0.5">bge-small-en-v1.5 384-d embeddings, IndexFlatIP, and after=T filtering.</p></div>
              </a>
              <a href="#page-5" data-nav-link="4" class="group cursor-pointer text-left rounded-xl border border-border bg-card p-4 hover:bg-accent/60 hover:border-primary/40 transition-colors flex flex-col justify-between gap-2 no-underline text-foreground">
                <div class="flex items-center justify-between w-full"><span class="font-mono text-xs text-muted-foreground">Chapter 4</span><span class="font-mono text-xs rounded-md bg-secondary px-2 py-0.5 text-primary group-hover:bg-primary group-hover:text-primary-foreground transition-colors">Page 5 →</span></div>
                <div><p class="font-serif text-base font-semibold group-hover:underline">60-Prompt Golden Set</p><p class="text-xs text-muted-foreground mt-0.5">Interactive explorer across Static, Live, Temporal Override, and Unanswerable.</p></div>
              </a>
              <a href="#page-6" data-nav-link="5" class="group cursor-pointer text-left rounded-xl border border-border bg-card p-4 hover:bg-accent/60 hover:border-primary/40 transition-colors flex flex-col justify-between gap-2 no-underline text-foreground">
                <div class="flex items-center justify-between w-full"><span class="font-mono text-xs text-muted-foreground">Chapter 5</span><span class="font-mono text-xs rounded-md bg-secondary px-2 py-0.5 text-primary group-hover:bg-primary group-hover:text-primary-foreground transition-colors">Page 6 →</span></div>
                <div><p class="font-serif text-base font-semibold group-hover:underline">Rubric &amp; Judge Calibration</p><p class="text-xs text-muted-foreground mt-0.5">48.3% -&gt; 80.0% -&gt; 90.0% exact agreement lift via CoT ordering &amp; MLX judge.</p></div>
              </a>
              <a href="#page-7" data-nav-link="6" class="group cursor-pointer text-left rounded-xl border border-border bg-card p-4 hover:bg-accent/60 hover:border-primary/40 transition-colors flex flex-col justify-between gap-2 no-underline text-foreground">
                <div class="flex items-center justify-between w-full"><span class="font-mono text-xs text-muted-foreground">Chapter 6</span><span class="font-mono text-xs rounded-md bg-secondary px-2 py-0.5 text-primary group-hover:bg-primary group-hover:text-primary-foreground transition-colors">Page 7 →</span></div>
                <div><p class="font-serif text-base font-semibold group-hover:underline">Baselines &amp; Run Inspector</p><p class="text-xs text-muted-foreground mt-0.5">240 Base vs. QLoRA runs, latency distributions, and 4-card prompt drilldown.</p></div>
              </a>
              <a href="#page-8" data-nav-link="7" class="group cursor-pointer text-left rounded-xl border border-primary/40 bg-card p-4 hover:bg-accent/60 hover:border-primary transition-colors flex flex-col justify-between gap-2 no-underline text-foreground">
                <div class="flex items-center justify-between w-full"><span class="font-mono text-xs text-primary font-semibold">Chapter 7</span><span class="font-mono text-xs rounded-md bg-primary/10 px-2 py-0.5 text-primary group-hover:bg-primary group-hover:text-primary-foreground transition-colors">Page 8 →</span></div>
                <div><p class="font-serif text-base font-semibold group-hover:underline">QLoRA Fine-Tuning</p><p class="text-xs text-muted-foreground mt-0.5">llama3.2-qlora:3b vs. llama3.2:3b: loss curves, 95% CIs, hybrid routing &amp; 120-item diff explorer.</p></div>
              </a>
              <a href="#page-9" data-nav-link="8" class="group cursor-pointer text-left rounded-xl border border-border bg-card p-4 hover:bg-accent/60 hover:border-primary/40 transition-colors flex flex-col justify-between gap-2 no-underline text-foreground">
                <div class="flex items-center justify-between w-full"><span class="font-mono text-xs text-muted-foreground">Chapter 8</span><span class="font-mono text-xs rounded-md bg-secondary px-2 py-0.5 text-primary group-hover:bg-primary group-hover:text-primary-foreground transition-colors">Page 9 →</span></div>
                <div><p class="font-serif text-base font-semibold group-hover:underline">Failure Analysis &amp; Roadmap</p><p class="text-xs text-muted-foreground mt-0.5">Master results table, 3 failure modes, and expected vs. measured QLoRA target.</p></div>
              </a>
              <a href="#page-10" data-nav-link="9" class="group cursor-pointer text-left rounded-xl border border-dashed border-border bg-secondary/40 p-4 hover:bg-accent/60 hover:border-primary/40 transition-colors flex flex-col justify-between gap-2 no-underline text-foreground">
                <div class="flex items-center justify-between w-full"><span class="font-mono text-xs text-muted-foreground">Appendix A</span><span class="font-mono text-xs rounded-md bg-secondary px-2 py-0.5 text-primary group-hover:bg-primary group-hover:text-primary-foreground transition-colors">Page 10 →</span></div>
                <div><p class="font-serif text-base font-semibold group-hover:underline">Compute &amp; Subagent Orchestration</p><p class="text-xs text-muted-foreground mt-0.5">Mac mini M5 Pro (48 GB) Ollama + MLX QLoRA training setup and parallel subagent waves.</p></div>
              </a>
              <a href="#page-11" data-nav-link="10" class="group cursor-pointer text-left rounded-xl border border-dashed border-border bg-secondary/40 p-4 hover:bg-accent/60 hover:border-primary/40 transition-colors flex flex-col justify-between gap-2 no-underline text-foreground">
                <div class="flex items-center justify-between w-full"><span class="font-mono text-xs text-muted-foreground">Appendix B</span><span class="font-mono text-xs rounded-md bg-secondary px-2 py-0.5 text-primary group-hover:bg-primary group-hover:text-primary-foreground transition-colors">Page 11 →</span></div>
                <div><p class="font-serif text-base font-semibold group-hover:underline">Three Research Continuations</p><p class="text-xs text-muted-foreground mt-0.5">Post-QLoRA roadmap: DPO abstention calibration, cross-encoder gating &amp; full-paper packing.</p></div>
              </a>
              <a href="#page-12" data-nav-link="11" class="group cursor-pointer text-left rounded-xl border border-dashed border-border bg-secondary/40 p-4 hover:bg-accent/60 hover:border-primary/40 transition-colors flex flex-col justify-between gap-2 no-underline text-foreground">
                <div class="flex items-center justify-between w-full"><span class="font-mono text-xs text-muted-foreground">Appendix C</span><span class="font-mono text-xs rounded-md bg-secondary px-2 py-0.5 text-primary group-hover:bg-primary group-hover:text-primary-foreground transition-colors">Page 12 →</span></div>
                <div><p class="font-serif text-base font-semibold group-hover:underline">Stronger Judge: Qwen3.5-9B vs. Qwen2.5-7B</p><p class="text-xs text-muted-foreground mt-0.5">Bottom-line verdict, QWK guide, speed/token telemetry (2.06× faster), and 34 discordant cases.</p></div>
              </a>
            </div>
          </div>

          <!-- Prologue Summary Chart -->
          <div class="rounded-xl border border-border bg-card p-5 space-y-2">
            <h3 class="font-serif text-lg font-semibold">Executive Comparison: Base vs. QLoRA (<code class="font-mono text-sm">Qwen3.5-9B</code> Judge)</h3>
            <p class="text-xs text-muted-foreground">Correctness (1–5) across all 4 slices plus Override Obedience scaled to 5 points (e.g., 0.80 &times; 5 = 4.00) for Base vs. QLoRA.</p>
            <div id="chart-p1-overview" class="w-full h-80"></div>
          </div>
        </section>

        <!-- Page 2: Chapter 1 - Slices, Metrics & Agents -->
        <section id="page-2" data-page="1" class="book-page hidden space-y-6">
          <div class="border-b border-border pb-4">
            <p class="font-mono text-xs uppercase tracking-wider text-muted-foreground">Chapter 1 · Evaluation &amp; Agent Design</p>
            <h1 class="font-serif text-3xl font-bold mt-1">Slice Taxonomy, Evaluation Metrics &amp; Agent Architecture</h1>
            <p class="text-sm text-muted-foreground mt-1">Formal definitions and positive/negative examples for all 4 benchmark slices, 1–5 scoring scales, plus runtime and parallel subagent roles.</p>
          </div>

          <!-- 4 Slice Definitions with Positive & Negative Examples -->
          <div class="space-y-2.5">
            <h2 class="font-serif text-xl font-semibold">1. Four Golden Slice Types (<code class="font-mono text-base">N = 60</code>) &amp; Concrete Examples</h2>
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-4 text-xs">
              <div class="rounded-xl border border-border bg-secondary/50 p-4 space-y-2.5">
                <div class="flex items-center justify-between">
                  <span class="font-mono font-semibold text-primary">static · n = 20 (g01–g20)</span>
                  <span class="rounded bg-muted px-2 py-0.5 font-mono text-[10px]">Published &lt; T</span>
                </div>
                <p><strong>What it tests:</strong> Pre-cutoff <code class="font-mono">cs.CL</code> facts (15 landmark pre-Dec-2023 papers + 5 early-2024 papers) absent from the post-T RAG index. Isolates parametric retention vs. live-only RAG over-abstention.</p>
                <div class="rounded-lg border border-border bg-card p-3 space-y-1">
                  <div class="flex items-center justify-between font-mono text-[11px]"><span class="font-semibold text-foreground">&#10003; Positive Example (<code class="font-mono">g11</code> RoPE · Zero-Shot)</span><span class="rounded bg-muted px-1.5 py-0.5">C=4 F=4 Cit=5</span></div>
                  <p class="text-muted-foreground">Explains RoPE rotation-matrix encoding and decaying inter-token dependency with relative distance accurately from pretraining weights.</p>
                </div>
                <div class="rounded-lg border border-border bg-card p-3 space-y-1">
                  <div class="flex items-center justify-between font-mono text-[11px]"><span class="font-semibold text-foreground">&#10007; Negative Example (<code class="font-mono">g01</code> LoRA RAG / <code class="font-mono">g02</code> ZS)</span><span class="rounded bg-muted px-1.5 py-0.5">C=1 F=5 Cit=5</span></div>
                  <p class="text-muted-foreground">Live-only RAG outputs <em>"Not in the sources."</em> because pre-T <code class="font-mono">2106.09685</code> is absent from <code class="font-mono">live.jsonl</code>; Zero-Shot <code class="font-mono">g02</code> misstates Lewis et al.'s RAG components as "Textual Memory".</p>
                </div>
              </div>

              <div class="rounded-xl border border-border bg-secondary/50 p-4 space-y-2.5">
                <div class="flex items-center justify-between">
                  <span class="font-mono font-semibold text-primary">live · n = 20 (g21–g40)</span>
                  <span class="rounded bg-muted px-2 py-0.5 font-mono text-[10px]">Published &gt;= T</span>
                </div>
                <p><strong>What it tests:</strong> Post-cutoff (2025–2026) methods, datasets, and empirical findings guaranteed unseen during <code class="font-mono">llama3.2:3b</code> pretraining. Isolates dense retrieval + <code class="font-mono">[arxiv_id]</code> citation grounding.</p>
                <div class="rounded-lg border border-border bg-card p-3 space-y-1">
                  <div class="flex items-center justify-between font-mono text-[11px]"><span class="font-semibold text-foreground">&#10003; Positive Example (<code class="font-mono">g37</code> MindEdit-Bench · RAG)</span><span class="rounded bg-muted px-1.5 py-0.5">C=5 F=5 Cit=5</span></div>
                  <p class="text-muted-foreground">Accurately reports the 6-task world-model hierarchy and 8%–31% vs. 81%–97% (53 pp gap) on L4/L5 counterfactuals, citing <code class="font-mono">[2607.00491]</code>.</p>
                </div>
                <div class="rounded-lg border border-border bg-card p-3 space-y-1">
                  <div class="flex items-center justify-between font-mono text-[11px]"><span class="font-semibold text-foreground">&#10007; Negative Example (<code class="font-mono">g28</code> LitBench · Zero-Shot)</span><span class="rounded bg-muted px-1.5 py-0.5">C=1 F=5 Cit=5</span></div>
                  <p class="text-muted-foreground">Falsely abstains (<em>"I don't have information on LitBench..."</em>) due to Dec 2023 cutoff instead of retrieving post-T evidence (or scores 1/1/1 if hallucinating).</p>
                </div>
              </div>

              <div class="rounded-xl border border-border bg-secondary/50 p-4 space-y-2.5">
                <div class="flex items-center justify-between">
                  <span class="font-mono font-semibold text-primary">temporal_override · n = 10 (g41–g50)</span>
                  <span class="rounded bg-muted px-2 py-0.5 font-mono text-[10px]">Post-T Beats Pre-T</span>
                </div>
                <p><strong>What it tests:</strong> Current-state queries (<em>"What is the current best approach to X?"</em>) where post-T evidence beats or refutes earlier paradigms. Isolates evidence-over-prior conflict resolution.</p>
                <div class="rounded-lg border border-border bg-card p-3 space-y-1">
                  <div class="flex items-center justify-between font-mono text-[11px]"><span class="font-semibold text-foreground">&#10003; Positive Example (<code class="font-mono">g41</code> SOLAR · RAG)</span><span class="rounded bg-muted px-1.5 py-0.5">C=5 F=5 Cit=5 Ov=1</span></div>
                  <p class="text-muted-foreground">Reports LRU/LFU underperforming FIFO on semantic buffers and adopts SOLAR (&le;3 competitive ratio, 5–75% gain), citing <code class="font-mono">[2607.00394]</code>.</p>
                </div>
                <div class="rounded-lg border border-border bg-card p-3 space-y-1">
                  <div class="flex items-center justify-between font-mono text-[11px]"><span class="font-semibold text-foreground">&#10007; Negative Example (<code class="font-mono">g44</code> "Aha!" Shifts · RAG / <code class="font-mono">g41</code> ZS)</span><span class="rounded bg-muted px-1.5 py-0.5">C=1..3 Ov=0</span></div>
                  <p class="text-muted-foreground">Zero-shot refuses/falls back to older priors (<code class="font-mono">Ov=0</code>), and RAG <code class="font-mono">g44</code> falsely abstains despite retrieving <code class="font-mono">[2601.00514]</code> at rank 1 (<code class="font-mono">C=1, Ov=0</code>).</p>
                </div>
              </div>

              <div class="rounded-xl border border-border bg-secondary/50 p-4 space-y-2.5">
                <div class="flex items-center justify-between">
                  <span class="font-mono font-semibold text-primary">unanswerable · n = 10 (g51–g60)</span>
                  <span class="rounded bg-muted px-2 py-0.5 font-mono text-[10px]">Not in Sources</span>
                </div>
                <p><strong>What it tests:</strong> 6 out-of-corpus prompts + 4 near-miss distractors matching indexed papers (<code class="font-mono">BaseRT</code>, <code class="font-mono">MSQA</code>, etc.) while asking unstated metrics. Isolates abstention precision.</p>
                <div class="rounded-lg border border-border bg-card p-3 space-y-1">
                  <div class="flex items-center justify-between font-mono text-[11px]"><span class="font-semibold text-foreground">&#10003; Positive Example (<code class="font-mono">g59</code> MSQA Near-Miss · RAG)</span><span class="rounded bg-muted px-1.5 py-0.5">C=5 F=5 Cit=5</span></div>
                  <p class="text-muted-foreground">Retrieves <code class="font-mono">2607.00724</code> at rank 1 (0.799 cosine), detects Swahili accuracy &amp; annotator counts are absent, and replies cleanly: <em>"Not in the sources."</em></p>
                </div>
                <div class="rounded-lg border border-border bg-card p-3 space-y-1">
                  <div class="flex items-center justify-between font-mono text-[11px]"><span class="font-semibold text-foreground">&#10007; Negative Example (Distractor Hallucination Archetype)</span><span class="rounded bg-muted px-1.5 py-0.5">C=1 F=1 Cit=2</span></div>
                  <p class="text-muted-foreground">Observed 0/20 times in baseline runs; would occur if the model fabricates Swahili accuracy/annotator counts and cites top-4 distractor <code class="font-mono">[2607.00724]</code>.</p>
                </div>
              </div>
            </div>
          </div>

          <!-- Evaluation Metrics Table -->
          <div class="space-y-2.5">
            <h2 id="metrics-def-caption" class="font-serif text-xl font-semibold">2. Evaluation Metrics, Anchors &amp; Agreement Gate</h2>
            <div tabindex="0" role="region" aria-labelledby="metrics-def-caption" class="table-wrapper book-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>Scale / Scope</th>
                    <th>Score 5 (or 1 / Pass)</th>
                    <th>Score 1 (or 0 / Fail)</th>
                    <th>Design Rationale</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <th class="font-mono text-xs">correctness</th>
                    <td data-label="Scale / Scope" class="font-mono text-xs">1–5 int · All slices</td>
                    <td data-label="Best" class="text-xs">Accurate core claim vs. reference; or clean abstention on <code class="font-mono">unanswerable</code>.</td>
                    <td data-label="Worst" class="text-xs">Wrong/contradictory, hallucinated answer on <code class="font-mono">unanswerable</code>, or false abstention on answerable slice.</td>
                    <td data-label="Rationale" class="text-xs text-muted-foreground">Measures end-task utility while enforcing slice-appropriate abstention.</td>
                  </tr>
                  <tr>
                    <th class="font-mono text-xs">faithfulness</th>
                    <td data-label="Scale / Scope" class="font-mono text-xs">1–5 int · All slices</td>
                    <td data-label="Best" class="text-xs">Every claim supported by <code class="font-mono">retrieved_context</code>, or pure abstention (including zero-shot <code class="font-mono">NONE</code>).</td>
                    <td data-label="Worst" class="text-xs">Asserts ungrounded factual claims not in <code class="font-mono">retrieved_context</code> (incl. zero-shot factual answers).</td>
                    <td data-label="Rationale" class="text-xs text-muted-foreground">Decouples context grounding from parametric correctness.</td>
                  </tr>
                  <tr>
                    <th class="font-mono text-xs">citation_accuracy</th>
                    <td data-label="Scale / Scope" class="font-mono text-xs">1–5 int · All slices</td>
                    <td data-label="Best" class="text-xs">Every key claim cites valid supporting <code class="font-mono">[arxiv_id]</code> from context, or clean abstention with <code class="font-mono">[]</code>.</td>
                    <td data-label="Worst" class="text-xs">Uncited factual claims, hallucinated IDs outside <code class="font-mono">valid_context_ids</code>, or abstaining with unrelated citations.</td>
                    <td data-label="Rationale" class="text-xs text-muted-foreground">Audits verifiability and catches fabricated or mismatched IDs.</td>
                  </tr>
                  <tr>
                    <th class="font-mono text-xs">override_obedience</th>
                    <td data-label="Scale / Scope" class="font-mono text-xs">{0, 1} · Override only</td>
                    <td data-label="Best" class="text-xs">1 iff response adopts/reports newer post-T finding (<code class="font-mono">newer_paper_id</code>) as current best.</td>
                    <td data-label="Worst" class="text-xs">0 when reverting to older pre-T baseline, hedging inconclusively, or abstaining.</td>
                    <td data-label="Rationale" class="text-xs text-muted-foreground">Isolates temporal authority over stale model priors.</td>
                  </tr>
                  <tr>
                    <th class="font-mono text-xs">latency_ms</th>
                    <td data-label="Scale / Scope" class="font-mono text-xs">ms · Mean &amp; Median</td>
                    <td data-label="Best" class="text-xs">Sequential warm-model wall time (<code class="font-mono">retrieval_ms + generation_ms</code> in RAG).</td>
                    <td data-label="Worst" class="text-xs">Measured without concurrency contention; cold model-load excluded via warmup.</td>
                    <td data-label="Rationale" class="text-xs text-muted-foreground">Quantifies real-time retrieval vs. generation cost trade-off.</td>
                  </tr>
                  <tr>
                    <th class="font-mono text-xs">judge_agreement</th>
                    <td data-label="Scale / Scope" class="font-mono text-xs">% · N=20 (60 pairs)</td>
                    <td data-label="Best" class="text-xs"><code class="font-mono">exact_match_pct</code> &ge; 70% (<code class="font-mono">J == H</code>), plus <code class="font-mono">within_1_pct</code> (<code class="font-mono">|J-H| &le; 1</code>) &amp; 3-bucket match.</td>
                    <td data-label="Worst" class="text-xs">Below 70% triggers mandatory rubric/judge calibration before trusting results.</td>
                    <td data-label="Rationale" class="text-xs text-muted-foreground">Prevents uncalibrated LLM-as-a-Judge bias from skewing findings.</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div class="rounded-xl border border-border bg-card p-4 space-y-2">
              <h3 class="font-serif text-lg font-semibold">Slice × Capability Diagnostic Matrix</h3>
              <p class="text-xs text-muted-foreground">How the 4 slices cleanly isolate orthogonal RAG and parametric failure modes.</p>
              <div id="chart-p2-taxonomy" class="w-full h-72"></div>
            </div>
            <div class="rounded-xl border border-border bg-card p-4 space-y-2">
              <h3 class="font-serif text-lg font-semibold">Canonical Response Score Archetypes</h3>
              <p class="text-xs text-muted-foreground">Illustrative <code class="font-mono">Correctness / Faithfulness / Citation</code> vectors under the calibrated rubric.</p>
              <div id="chart-p2-archetypes" class="w-full h-72"></div>
            </div>
          </div>

          <!-- 3. Agents Used: Roles & Mechanisms -->
          <div class="space-y-4">
            <h2 class="font-serif text-xl font-semibold">3. Agents Used: Roles &amp; How They Achieve Them</h2>
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
              <div class="rounded-xl border border-border bg-secondary/50 p-4 space-y-1.5">
                <div class="flex items-center justify-between">
                  <span class="font-mono font-semibold text-primary">Dense Retriever Agent</span>
                  <span class="rounded bg-muted px-2 py-0.5 font-mono text-[10px]">bge-small + FAISS</span>
                </div>
                <p><strong>Role:</strong> Non-parametric evidence lookup over post-T (<code class="font-mono">&gt;= 2025-01-01</code>) literature.</p>
                <p class="text-muted-foreground"><strong>How it achieves it:</strong> Prepends asymmetric BGE query instruction, L2-normalizes 384-d vectors, runs exact <code class="font-mono">IndexFlatIP</code> scan (~41 ms), and applies <code class="font-mono">published_date &gt;= after</code> prior to slicing <code class="font-mono">[:k]</code>.</p>
              </div>
              <div class="rounded-xl border border-border bg-secondary/50 p-4 space-y-1.5">
                <div class="flex items-center justify-between">
                  <span class="font-mono font-semibold text-primary">Generator / Baseline Agent</span>
                  <span class="rounded bg-muted px-2 py-0.5 font-mono text-[10px]">llama3.2:3b</span>
                </div>
                <p><strong>Role:</strong> Controlled answer synthesis in Zero-Shot vs. Strict Grounded RAG (<code class="font-mono">k=4</code>).</p>
                <p class="text-muted-foreground"><strong>How it achieves it:</strong> Executes sequentially on Mac mini Metal GPU (<code class="font-mono">temperature=0.0</code>, <code class="font-mono">num_predict=384</code>); RAG prompt enforces <code class="font-mono">[arxiv_id]</code> bracket citations or exact <code class="font-mono">"Not in the sources."</code> abstention.</p>
              </div>
              <div class="rounded-xl border border-border bg-secondary/50 p-4 space-y-1.5">
                <div class="flex items-center justify-between">
                  <span class="font-mono font-semibold text-primary">Cross-Family Judge Agents</span>
                  <span class="rounded bg-muted px-2 py-0.5 font-mono text-[10px]">qwen2.5:7b · qwen3.5:9b</span>
                </div>
                <p><strong>Role:</strong> Automated rubric grading across all 120 runs without same-family bias.</p>
                <p class="text-muted-foreground"><strong>How it achieves it:</strong> Baseline <code class="font-mono">qwen2.5:7b</code> (GGUF) and stronger <code class="font-mono">qwen3.5:9b</code> (MLX <code class="font-mono">nvfp4</code>, <code class="font-mono">think=False</code>) emit <code class="font-mono">format="json"</code> (<code class="font-mono">num_ctx=4096</code>) with CoT <code class="font-mono">rationale</code> first, 2-step abstention rules, and deterministic clamping.</p>
              </div>
              <div class="rounded-xl border border-border bg-secondary/50 p-4 space-y-1.5">
                <div class="flex items-center justify-between">
                  <span class="font-mono font-semibold text-primary">Blinded Human-Benchmark Auditor</span>
                  <span class="rounded bg-muted px-2 py-0.5 font-mono text-[10px]">20 Stratified Items</span>
                </div>
                <p><strong>Role:</strong> Independent ground-truth audit validating the &ge;70% rubric trust gate.</p>
                <p class="text-muted-foreground"><strong>How it achieves it:</strong> Grades <code class="font-mono">blind_sample_20.jsonl</code> (10 ZS + 10 RAG, zero judge leakage), exposing v1 abstention mis-scoring (48.33%) and verifying v2 calibration (80.0% exact / 86.67% &plusmn;1).</p>
              </div>
            </div>

            <div class="space-y-2">
              <h3 id="agents-roster-caption" class="font-serif text-lg font-semibold">Parallel Orchestration Subagents (3 Waves, Disjoint File Ownership)</h3>
              <div tabindex="0" role="region" aria-labelledby="agents-roster-caption" class="table-wrapper book-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Wave / Subagent</th>
                      <th>Assigned Role</th>
                      <th>How It Achieves It &amp; Verification Gate</th>
                      <th>Owned Artifacts</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <th class="font-mono text-xs">1A · Infra Setup</th>
                      <td data-label="Role" class="text-xs">Provision Mac mini compute host</td>
                      <td data-label="Method" class="text-xs text-muted-foreground">User-space <code class="font-mono">uv</code>, Python 3.13, <code class="font-mono">Ollama.app 0.40.18</code> symlink, GPU pull &amp; <code class="font-mono">/api/chat</code> smoke tests.</td>
                      <td data-label="Artifacts" class="font-mono text-xs">corpus/env_manifest.json</td>
                    </tr>
                    <tr>
                      <th class="font-mono text-xs">1B · Corpus Builder</th>
                      <td data-label="Role" class="text-xs">Fetch leakage-safe date split</td>
                      <td data-label="Method" class="text-xs text-muted-foreground">Rate-limited (3.1s) <code class="font-mono">cs.CL</code> queries, strips <code class="font-mono">vN</code>, asserts 300/200 counts &amp; 0 overlap around <code class="font-mono">T=2025-01-01</code>.</td>
                      <td data-label="Artifacts" class="font-mono text-xs">src/fetch_corpus.py, corpus/*</td>
                    </tr>
                    <tr>
                      <th class="font-mono text-xs">1C · Pipeline Author</th>
                      <td data-label="Role" class="text-xs">Author rubric, RAG, eval &amp; tests</td>
                      <td data-label="Method" class="text-xs text-muted-foreground">Writes <code class="font-mono">rubric.md</code> first, then modular FAISS ingest/retriever, baseline runner, judge, metrics, pytest.</td>
                      <td data-label="Artifacts" class="font-mono text-xs">src/*.py, eval/rubric.md, tests/</td>
                    </tr>
                    <tr>
                      <th class="font-mono text-xs">2A · Static Author</th>
                      <td data-label="Role" class="text-xs">Author 20 pre-T prompts</td>
                      <td data-label="Method" class="text-xs text-muted-foreground">Crafts 15 pre-Dec-2023 landmark + 5 early-2024 Q&amp;A pairs; verifies every ID &amp; claim against <code class="font-mono">static.jsonl</code>.</td>
                      <td data-label="Artifacts" class="font-mono text-xs">eval/slices/static.jsonl</td>
                    </tr>
                    <tr>
                      <th class="font-mono text-xs">2B · Live Author</th>
                      <td data-label="Role" class="text-xs">Author 20 post-T prompts</td>
                      <td data-label="Method" class="text-xs text-muted-foreground">Samples 2025–2026 methods across 4 cohorts; runs <code class="font-mono">retrieve(k=4)</code> on Mac mini confirming 20/20 rank-1 hits.</td>
                      <td data-label="Artifacts" class="font-mono text-xs">eval/slices/live.jsonl</td>
                    </tr>
                    <tr>
                      <th class="font-mono text-xs">2C · Override Author</th>
                      <td data-label="Role" class="text-xs">Author 10 override conflicts</td>
                      <td data-label="Method" class="text-xs text-muted-foreground">Identifies post-T papers overturning prior SOTA/assumptions; verifies 10/10 rank-1 hits on Mac mini.</td>
                      <td data-label="Artifacts" class="font-mono text-xs">eval/slices/temporal_override.jsonl</td>
                    </tr>
                    <tr>
                      <th class="font-mono text-xs">2D · Unanswerable</th>
                      <td data-label="Role" class="text-xs">Author 10 absent/distractor Qs</td>
                      <td data-label="Method" class="text-xs text-muted-foreground">Builds 6 OOC + 4 near-miss distractors; inspects all 40 top-4 chunks on Mac mini confirming zero leaks.</td>
                      <td data-label="Artifacts" class="font-mono text-xs">eval/slices/unanswerable.jsonl</td>
                    </tr>
                    <tr>
                      <th class="font-mono text-xs">3A · Judge Runner</th>
                      <td data-label="Role" class="text-xs">Score 120 runs with Qwen2.5-7B</td>
                      <td data-label="Method" class="text-xs text-muted-foreground">Runs <code class="font-mono">src.judge</code> on Mac mini Metal GPU, verifies schema/range bounds, syncs JSONL scores back.</td>
                      <td data-label="Artifacts" class="font-mono text-xs">eval/judge_scores.jsonl</td>
                    </tr>
                    <tr>
                      <th class="font-mono text-xs">3B · Human Grader</th>
                      <td data-label="Role" class="text-xs">Blind-grade 20 sampled outputs</td>
                      <td data-label="Method" class="text-xs text-muted-foreground">Reads only <code class="font-mono">rubric.md</code> + <code class="font-mono">blind_sample_20.jsonl</code>; outputs independent 1–5 scores &amp; notes.</td>
                      <td data-label="Artifacts" class="font-mono text-xs">eval/human_scores_20.jsonl</td>
                    </tr>
                    <tr>
                      <th class="font-mono text-xs">4 · Qwen3.5 Judge</th>
                      <td data-label="Role" class="text-xs">Stronger MLX judge &amp; comparator</td>
                      <td data-label="Method" class="text-xs text-muted-foreground">Runs <code class="font-mono">qwen3.5:9b</code> (<code class="font-mono">think=False</code>) over frozen 120 runs + 2000-resample paired bootstrap &amp; QWK.</td>
                      <td data-label="Artifacts" class="font-mono text-xs">eval/judge_scores_qwen35_9b.jsonl, eval/judge_comparison.*</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </section>

        <!-- Page 3: Step 1 - Date-Split Corpus -->
        <section id="page-3" data-page="2" class="book-page hidden space-y-6">
          <div class="border-b border-border pb-4">
            <p class="font-mono text-xs uppercase tracking-wider text-muted-foreground">Chapter 2 · Leakage-Safe Data Split</p>
            <h1 class="font-serif text-3xl font-bold mt-1">Date-Split arXiv <code class="font-mono text-2xl">cs.CL</code> Corpus (<code class="font-mono text-2xl">T = 2025-01-01</code>)</h1>
            <p class="text-sm text-muted-foreground mt-1">300 pre-cutoff <code class="font-mono text-xs">static</code> abstracts vs. 200 post-cutoff <code class="font-mono text-xs">live</code> abstracts across 4 cohorts with strict version-stripped zero overlap.</p>
          </div>

          <div id="p3-split-cards" class="grid grid-cols-2 md:grid-cols-4 gap-3"></div>

          <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <div class="lg:col-span-2 rounded-xl border border-border bg-card p-4 space-y-2">
              <h3 class="font-serif text-lg font-semibold">Temporal Distribution Around Model Cutoff &amp; Split T</h3>
              <p class="text-xs text-muted-foreground">Static corpus spans 2020–2024 (21 landmark + 2022/2024 batches); Live corpus spans 2025–2026 (50 per half-year).</p>
              <div id="chart-p3-timeline" class="w-full h-80"></div>
            </div>
            <div class="rounded-xl border border-border bg-card p-4 space-y-2">
              <h3 class="font-serif text-lg font-semibold">Abstract Length Distribution</h3>
              <p class="text-xs text-muted-foreground">Word counts validate 1-chunk-per-abstract ingestion (median ~170 words).</p>
              <div id="chart-p3-lengths" class="w-full h-80"></div>
            </div>
          </div>

          <div class="space-y-3">
            <div class="flex flex-wrap items-center justify-between gap-3">
              <h3 id="corpus-table-caption" class="font-serif text-lg font-semibold">Sampled Corpus Records</h3>
              <div class="flex items-center gap-2">
                <select id="corpus-filter-split" aria-label="Filter corpus split" class="rounded-md border border-border bg-card px-2.5 py-1.5 text-xs">
                  <option value="all">All Samples (50)</option>
                  <option value="static">Static (&lt; 2025-01-01)</option>
                  <option value="live">Live (&gt;= 2025-01-01)</option>
                </select>
                <input id="corpus-search" type="search" placeholder="Filter title or ID..." class="rounded-md border border-border bg-card px-2.5 py-1.5 text-xs w-44">
              </div>
            </div>
            <div tabindex="0" role="region" aria-labelledby="corpus-table-caption" class="table-wrapper book-scroll">
              <table>
                <thead>
                  <tr>
                    <th>arXiv ID</th>
                    <th>Split</th>
                    <th>Published</th>
                    <th>Words</th>
                    <th>Title &amp; Abstract Preview</th>
                  </tr>
                </thead>
                <tbody id="corpus-sample-tbody"></tbody>
              </table>
            </div>
          </div>
        </section>

        <!-- Page 4: Step 2 - Dense Ingestion & Retrieval -->
        <section id="page-4" data-page="3" class="book-page hidden space-y-6">
          <div class="border-b border-border pb-4">
            <p class="font-mono text-xs uppercase tracking-wider text-muted-foreground">Chapter 3 · Embeddings &amp; Vector Store</p>
            <h1 class="font-serif text-3xl font-bold mt-1">Dense Ingestion &amp; Date-Filtered FAISS Retrieval</h1>
            <p class="text-sm text-muted-foreground mt-1">Encoding 200 live abstracts with <code class="font-mono text-xs">BAAI/bge-small-en-v1.5</code> (384-d L2-normalized) into <code class="font-mono text-xs">faiss.IndexFlatIP</code> with pre-limit <code class="font-mono text-xs">published_date &gt;= after</code> filtering.</p>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div class="rounded-xl border border-border bg-secondary/50 p-4 space-y-1">
              <p class="font-mono text-xs text-muted-foreground">Index Artifact</p>
              <p class="font-semibold">200 vectors × 384-d = 307 KB</p>
              <p class="text-xs text-muted-foreground">Exact full-corpus inner product scan (<code class="font-mono">index.ntotal</code>) avoids post-filter under-fetching when <code class="font-mono">after</code> is selective.</p>
            </div>
            <div class="rounded-xl border border-border bg-secondary/50 p-4 space-y-1">
              <p class="font-mono text-xs text-muted-foreground">Asymmetric BGE Query Prefix</p>
              <p class="font-semibold font-mono text-xs">"Represent this sentence for searching relevant passages: "</p>
              <p class="text-xs text-muted-foreground">Applied strictly at query time; documents store raw <code class="font-mono">Title + Abstract</code>.</p>
            </div>
            <div class="rounded-xl border border-border bg-secondary/50 p-4 space-y-1">
              <p class="font-mono text-xs text-muted-foreground">Live Retrieval Verification</p>
              <p class="font-semibold">30 / 30 Rank-1 Hits (Live + Override)</p>
              <p class="text-xs text-muted-foreground">Retrieval takes <strong>41.3 ms</strong> mean (~2.2% of total RAG latency).</p>
            </div>
          </div>

          <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div class="rounded-xl border border-border bg-card p-4 space-y-2">
              <h3 class="font-serif text-lg font-semibold">FAISS Target Retrieval Recall by Slice</h3>
              <p class="text-xs text-muted-foreground">Gold target paper presence inside top-4 retrieved chunks from the live index.</p>
              <div id="chart-p4-recall" class="w-full h-72"></div>
            </div>
            <div class="rounded-xl border border-border bg-card p-4 space-y-2">
              <h3 class="font-serif text-lg font-semibold">RAG Latency Decomposition (Retrieval vs. Generation)</h3>
              <p class="text-xs text-muted-foreground">Abstaining slices (<code class="font-mono">static</code>, <code class="font-mono">unanswerable</code>) finish under 950 ms because <code class="font-mono">"Not in the sources."</code> emits ~6 tokens.</p>
              <div id="chart-p4-latency-split" class="w-full h-72"></div>
            </div>
          </div>

          <div class="rounded-xl border border-border bg-secondary/40 p-5 space-y-2">
            <h3 class="font-serif text-base font-semibold">Verified CLI Retrieval Example (<code class="font-mono text-xs">after="2025-07-01"</code>)</h3>
            <pre class="font-mono text-xs overflow-x-auto bg-card border border-border rounded-lg p-3 leading-relaxed">$ uv run python main.py retrieve --query "continual learning ASR" --k 2 --after 2025-07-01
1. [2507.00534 | 2025-07-01 | score=0.8043] NIRANTAR: Continual Learning with New Languages and Domains on Real-world Speech Data
2. [2601.00543 | 2026-01-01 | score=0.6184] Embedding Consistency Regulation for Persistent Semantic Routing in Compact Language Models</pre>
          </div>
        </section>

        <!-- Page 5: Step 3 - 60-Prompt Golden Set -->
        <section id="page-5" data-page="4" class="book-page hidden space-y-6">
          <div class="border-b border-border pb-4">
            <p class="font-mono text-xs uppercase tracking-wider text-muted-foreground">Chapter 4 · Evaluation Benchmark</p>
            <h1 class="font-serif text-3xl font-bold mt-1">60-Prompt 4-Slice Golden Set (<code class="font-mono text-2xl">eval/golden.jsonl</code>)</h1>
            <p class="text-sm text-muted-foreground mt-1">Hand-verified prompts testing parametric retention, fresh post-T retrieval, temporal override obedience, and adversarial distractor abstention.</p>
          </div>

          <div class="grid grid-cols-1 lg:grid-cols-3 gap-4 items-center">
            <div class="rounded-xl border border-border bg-card p-4 space-y-2">
              <h3 class="font-serif text-lg font-semibold">Golden Slice Composition</h3>
              <p class="text-xs text-muted-foreground">20 Static · 20 Live · 10 Temporal Override · 10 Unanswerable.</p>
              <div id="chart-p5-slices" class="w-full h-72"></div>
            </div>
            <div class="lg:col-span-2 grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
              <div class="rounded-xl border border-border bg-secondary/50 p-4 space-y-1">
                <span class="font-mono font-semibold">static (20) · g01–g20</span>
                <p class="text-muted-foreground">15 pre-Dec-2023 landmark papers (LoRA, RAG, DPO, CoT, ReAct, Self-RAG, AWQ, GQA) + 5 early-2024 pre-T papers. Tests whether live-only RAG hurts pre-T recall.</p>
              </div>
              <div class="rounded-xl border border-border bg-secondary/50 p-4 space-y-1">
                <span class="font-mono font-semibold">live (20) · g21–g40</span>
                <p class="text-muted-foreground">5 papers per 2025/2026 cohort introducing named methods &amp; benchmarks (BlockDialect, NIRANTAR, LitBench, MindEdit-Bench, R²Adapter). Unseen by base model.</p>
              </div>
              <div class="rounded-xl border border-border bg-secondary/50 p-4 space-y-1">
                <span class="font-mono font-semibold">temporal_override (10) · g41–g50</span>
                <p class="text-muted-foreground">Post-T papers explicitly overturning prior SOTA or assumptions (SOLAR vs LRU/LFU, TRAG, RL vs SFT drift, ONLY vs contrastive decoding).</p>
              </div>
              <div class="rounded-xl border border-border bg-secondary/50 p-4 space-y-1">
                <span class="font-mono font-semibold">unanswerable (10) · g51–g60</span>
                <p class="text-muted-foreground">6 out-of-corpus prompts + 4 near-miss distractors matching <code class="font-mono">BaseRT</code>, <code class="font-mono">BlockDialect</code>, <code class="font-mono">MSQA</code>, and <code class="font-mono">Question Decomposition</code> while asking absent metrics.</p>
              </div>
            </div>
          </div>

          <!-- Accessible Tabbed Golden Explorer -->
          <div class="space-y-3">
            <div class="flex flex-wrap items-center justify-between gap-3">
              <h3 class="font-serif text-lg font-semibold">Interactive Golden Set Browser</h3>
              <input id="golden-search" type="search" placeholder="Search prompt, answer, or arXiv ID..." class="rounded-md border border-border bg-card px-2.5 py-1.5 text-xs w-64">
            </div>
            <div id="golden-tabs-root">
              <div class="tablist" role="tablist" aria-label="Golden set slices">
                <button type="button" role="tab" id="gtab-all" aria-selected="true" data-slice="all">All (60)</button>
                <button type="button" role="tab" id="gtab-static" aria-selected="false" data-slice="static">Static (20)</button>
                <button type="button" role="tab" id="gtab-live" aria-selected="false" data-slice="live">Live (20)</button>
                <button type="button" role="tab" id="gtab-temporal_override" aria-selected="false" data-slice="temporal_override">Temporal Override (10)</button>
                <button type="button" role="tab" id="gtab-unanswerable" aria-selected="false" data-slice="unanswerable">Unanswerable (10)</button>
              </div>
              <div role="tabpanel" aria-labelledby="gtab-all" tabindex="0">
                <div class="tabpanel-box">
                  <div tabindex="0" role="region" aria-label="Golden prompts table" class="table-wrapper book-scroll">
                    <table>
                      <thead>
                        <tr>
                          <th>ID</th>
                          <th>Slice</th>
                          <th>Prompt</th>
                          <th>Reference Answer</th>
                          <th>Source IDs</th>
                        </tr>
                      </thead>
                      <tbody id="golden-tbody"></tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <!-- Page 6: Step 4 - Rubric & Judge Calibration -->
        <section id="page-6" data-page="5" class="book-page hidden space-y-6">
          <div class="border-b border-border pb-4">
            <p class="font-mono text-xs uppercase tracking-wider text-muted-foreground">Chapter 5 · Evaluation Rigor</p>
            <h1 class="font-serif text-3xl font-bold mt-1">Rubric Design &amp; Judge-vs-Human Calibration</h1>
            <p class="text-sm text-muted-foreground mt-1">Enforcing the &ge;70% agreement gate: how Chain-of-Thought field ordering and abstention rules lifted <code class="font-mono text-xs">Qwen2.5-7B-Instruct</code> from 48.33% to 80.0% exact agreement.</p>
          </div>

          <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div class="rounded-xl border border-border bg-card p-4 space-y-2">
              <h3 class="font-serif text-lg font-semibold">Calibration Lift (v1 Uncalibrated vs. v2 Calibrated)</h3>
              <p class="text-xs text-muted-foreground">Dashed line marks the 70% minimum trust threshold across 20 blind human-graded answers.</p>
              <div id="chart-p6-calibration" class="w-full h-72"></div>
            </div>
            <div class="rounded-xl border border-border bg-card p-4 space-y-2">
              <h3 class="font-serif text-lg font-semibold">Pooled 5×5 Confusion Matrix (N = 60 Score Pairs)</h3>
              <p class="text-xs text-muted-foreground">Diagonal concentration across Correctness, Faithfulness, and Citation Accuracy (86.67% within ±1).</p>
              <div id="chart-p6-heatmap" class="w-full h-72"></div>
            </div>
          </div>

          <!-- Accessible Tabbed Panel: Rubric Anchors vs 20-Item Blind Audit -->
          <div id="p6-tabs-root">
            <div class="tablist" role="tablist" aria-label="Rubric and audit views">
              <button type="button" role="tab" id="p6tab-audit" aria-selected="true" data-panel="p6panel-audit">20-Item Blind Audit (Human vs. Judge)</button>
              <button type="button" role="tab" id="p6tab-rubric" aria-selected="false" data-panel="p6panel-rubric">Behavioral Rubric &amp; Abstention Rules</button>
            </div>

            <div role="tabpanel" id="p6panel-audit" aria-labelledby="p6tab-audit" tabindex="0">
              <div class="tabpanel-box space-y-2">
                <div tabindex="0" role="region" aria-label="20 blind audit table" class="table-wrapper book-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>Sample</th>
                        <th>Mode / Slice</th>
                        <th>Human (C/F/Cit/Ov)</th>
                        <th>Qwen2.5-7B</th>
                        <th>Qwen3.5-9B (MLX)</th>
                        <th>Human Note &amp; Judge Rationales</th>
                      </tr>
                    </thead>
                    <tbody id="audit-tbody"></tbody>
                  </table>
                </div>
              </div>
            </div>

            <div role="tabpanel" id="p6panel-rubric" aria-labelledby="p6tab-rubric" hidden="until-found">
              <div class="tabpanel-box grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                <div class="rounded-xl border border-border bg-secondary/50 p-4 space-y-2">
                  <h4 class="font-serif text-base font-semibold">1–5 Ordinal Anchors</h4>
                  <p><strong>Correctness (slice-aware):</strong> 5 = accurate claim (or clean abstention on <code class="font-mono">unanswerable</code>); 1 = wrong, fabricated, or false abstention on <code class="font-mono">static/live/temporal_override</code>.</p>
                  <p><strong>Faithfulness:</strong> 5 = strictly supported by <code class="font-mono">retrieved_context</code> OR clean abstention (even when context is <code class="font-mono">NONE</code>); 1 = ungrounded claims (including zero-shot factual answers).</p>
                  <p><strong>Citation Accuracy:</strong> 5 = every key claim cites valid supporting <code class="font-mono">[arxiv_id]</code> OR clean abstention with empty citations; 1 = uncited claims or hallucinated IDs.</p>
                  <p><strong>Override Obedience (0/1):</strong> 1 iff <code class="font-mono">temporal_override</code> answer adopts the newer post-T finding (<code class="font-mono">newer_paper_id</code>).</p>
                </div>
                <div class="rounded-xl border border-border bg-secondary/50 p-4 space-y-2">
                  <h4 class="font-serif text-base font-semibold">Calibration Fixes Applied in v2</h4>
                  <ul class="list-disc pl-4 space-y-1.5 text-muted-foreground">
                    <li><strong>CoT Field Order:</strong> Moved <code class="font-mono">rationale</code> first in JSON schema so <code class="font-mono">qwen2.5:7b</code> evaluates abstention &amp; slice rules before outputting integers.</li>
                    <li><strong>Pure vs. Hedged Abstention:</strong> Separated pure refusals (e.g. <em>"I don't have information..."</em>) from hedged overviews (<em>"I don't have exact numbers, however X works by..."</em>).</li>
                    <li><strong>Few-Shot Abstention Anchors:</strong> Added explicit JSON examples mapping answerable-slice abstentions to <code class="font-mono">correctness=1, faithfulness=5, citation_accuracy=5</code>.</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        <!-- Page 7: Step 5 - Baselines Showdown & Run Inspector -->
        <section id="page-7" data-page="6" class="book-page hidden space-y-6">
          <div class="border-b border-border pb-4">
            <p class="font-mono text-xs uppercase tracking-wider text-muted-foreground">Chapter 6 · Empirical Baselines &amp; Run Inspector</p>
            <h1 class="font-serif text-3xl font-bold mt-1">Zero-Shot vs. RAG (<code class="font-mono text-2xl">k=4</code>) Performance &amp; Run Inspector</h1>
            <p class="text-sm text-muted-foreground mt-1">120 sequential runs on Base (<code class="font-mono text-xs">llama3.2:3b</code>) paired with 120 QLoRA (<code class="font-mono text-xs">llama3.2-qlora:3b</code>) runs: slice correctness, abstention/citation behavior, latency distributions, and 4-way prompt drilldown.</p>
          </div>

          <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div class="rounded-xl border border-border bg-card p-4 space-y-2">
              <h3 class="font-serif text-lg font-semibold">Correctness by Slice (Zero-Shot vs. RAG)</h3>
              <p class="text-xs text-muted-foreground">RAG surges on <code class="font-mono">live</code> (+2.85) and <code class="font-mono">temporal_override</code> (+0.90), while <code class="font-mono">static</code> drops to 1.80.</p>
              <div id="chart-p7-scores" class="w-full h-72"></div>
            </div>
            <div class="rounded-xl border border-border bg-card p-4 space-y-2">
              <h3 class="font-serif text-lg font-semibold">Response Behavior Breakdown</h3>
              <p class="text-xs text-muted-foreground">Abstained vs. Cited Answer vs. Uncited Answer across all 120 runs.</p>
              <div id="chart-p7-behavior" class="w-full h-72"></div>
            </div>
          </div>

          <div class="rounded-xl border border-border bg-card p-4 space-y-2">
            <h3 class="font-serif text-lg font-semibold">Per-Query Latency Distribution (120 Runs, ms)</h3>
            <p class="text-xs text-muted-foreground">Zero-shot mean 1,725 ms (median 1,001 ms) vs. RAG mean 1,880 ms (median 1,219 ms).</p>
            <div id="chart-p7-latency" class="w-full h-72"></div>
          </div>

          <!-- Interactive Side-by-Side Run Inspector -->
          <div class="rounded-xl border border-border bg-secondary/40 p-5 space-y-4">
            <div class="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h3 class="font-serif text-xl font-semibold">Interactive 4-Way Run Inspector (<code class="font-mono text-sm">g01–g60</code>)</h3>
                <p class="text-xs text-muted-foreground">Compare Base vs. QLoRA Zero-Shot and RAG answers, citations, latencies, and dual-judge rationales on any prompt.</p>
              </div>
              <div class="flex items-center gap-2">
                <button id="inspector-prev" type="button" class="rounded-md border border-border bg-card px-2.5 py-1.5 text-xs font-medium hover:bg-accent">← Prev</button>
                <select id="inspector-select" aria-label="Select golden prompt" class="rounded-md border border-border bg-card px-2.5 py-1.5 text-xs max-w-xs"></select>
                <button id="inspector-next" type="button" class="rounded-md border border-border bg-card px-2.5 py-1.5 text-xs font-medium hover:bg-accent">Next →</button>
              </div>
            </div>
            <div id="inspector-body" class="space-y-4"></div>
          </div>
        </section>

        <!-- Page 8: Chapter 7 - QLoRA Fine-Tuning Experiment -->
        <section id="page-8" data-page="7" class="book-page hidden space-y-6">
          <div class="border-b border-border pb-4">
            <p class="font-mono text-xs uppercase tracking-wider text-muted-foreground">Chapter 7 · QLoRA Fine-Tuning Experiment</p>
            <h1 class="font-serif text-3xl font-bold mt-1">QLoRA Fine-Tuning: <code class="font-mono text-2xl">llama3.2-qlora:3b</code> vs. <code class="font-mono text-2xl">llama3.2:3b</code></h1>
            <p class="text-sm text-muted-foreground mt-1">4-bit MLX QLoRA (<code class="font-mono text-xs">rank=16</code>, 1,060 leakage-safe pre-T samples, checkpoint <code class="font-mono text-xs">iter 400</code>, <code class="font-mono text-xs">nvfp4</code>) evaluated across all 120 paired runs under <code class="font-mono text-xs">qwen3.5:9b</code> and <code class="font-mono text-xs">qwen2.5:7b</code>.</p>
          </div>

          <!-- Bottom-Line Verdict Banner -->
          <div class="rounded-xl border-2 border-primary/40 bg-secondary/70 p-5 space-y-2">
            <div class="flex flex-wrap items-center justify-between gap-2">
              <span class="font-mono text-xs font-semibold uppercase tracking-wider text-primary">Bottom-Line Verdict</span>
              <span class="rounded-full bg-primary text-primary-foreground px-3 py-0.5 font-mono text-xs font-semibold">Conditional Win: +0.133 Pooled · 2.3× Faster RAG</span>
            </div>
            <p class="font-serif text-lg font-semibold">QLoRA improves pooled correctness (<code class="font-mono text-base">2.692 &rarr; 2.825</code>, <code class="font-mono text-base">+0.133</code>, 95% CI <code class="font-mono text-base">[-0.092, +0.358]</code>; <code class="font-mono text-base">+0.225</code> under <code class="font-mono text-base">qwen2.5:7b</code>) and cuts latency <code class="font-mono text-base">-1,078.9 ms</code>, with decisive RAG override gains alongside clear calibration trade-offs.</p>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs text-muted-foreground pt-1">
              <div class="rounded-lg border border-border bg-card/80 p-3 space-y-1">
                <p class="font-semibold text-foreground">Measured Wins</p>
                <p>&bull; <strong>RAG <code class="font-mono">temporal_override</code>:</strong> Correctness <code class="font-mono">3.60 &rarr; 4.80</code> (<code class="font-mono">+1.20</code>), override obedience <code class="font-mono">0.50 &rarr; 0.80</code>, <code class="font-mono">10/10</code> cited, <code class="font-mono">0/10</code> abstentions.<br>&bull; <strong>Zero-Shot <code class="font-mono">static</code>:</strong> Correctness <code class="font-mono">1.15 &rarr; 1.60</code> (<code class="font-mono">+0.45</code>), abstentions <code class="font-mono">18/20 &rarr; 0/20</code>.<br>&bull; <strong>Speed &amp; Fluency:</strong> <code class="font-mono">2.3&times;</code> faster RAG (<code class="font-mono">1,880.2 &rarr; 834.9 ms</code>) and <code class="font-mono">2.5&times;</code> faster pooled (<code class="font-mono">-1,078.9 ms</code>); spurious trailing abstentions <code class="font-mono">1 &rarr; 0</code>.</p>
              </div>
              <div class="rounded-lg border border-border bg-card/80 p-3 space-y-1">
                <p class="font-semibold text-foreground">Honest Calibration Trade-Offs</p>
                <p>&bull; <strong>Zero-Shot Overconfidence:</strong> Abstentions drop to <code class="font-mono">0/20</code> on unseen <code class="font-mono">live</code> (faithfulness <code class="font-mono">4.80 &rarr; 1.40</code>) and <code class="font-mono">9/10</code> on <code class="font-mono">unanswerable</code> (<code class="font-mono">5.00 &rarr; 4.60</code>, 1 ungrounded guess).<br>&bull; <strong>Terse <code class="font-mono">live</code> RAG Synthesis:</strong> Shorter 1–2 sentence answers omit minor details under <code class="font-mono">qwen3.5:9b</code> (<code class="font-mono">4.85 &rarr; 4.45</code>, 1 win / 7 regressions; <code class="font-mono">+0.05</code> under <code class="font-mono">qwen2.5:7b</code>) and drop explicit bracket citations on <code class="font-mono">live</code> RAG (<code class="font-mono">18/20 &rarr; 2/20</code>).</p>
              </div>
            </div>
          </div>

          <!-- 4 KPI Cards -->
          <div id="p9-qlora-kpis" class="grid grid-cols-2 lg:grid-cols-4 gap-4"></div>

          <!-- 4 Lazy-Rendered Plotly Charts -->
          <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div class="rounded-xl border border-border bg-card p-4 space-y-2">
              <h3 class="font-serif text-lg font-semibold">MLX QLoRA Training &amp; Validation Loss</h3>
              <p class="text-xs text-muted-foreground">600 cosine-decayed steps (<code class="font-mono">batch_size=4</code>, <code class="font-mono">mask_prompt=true</code>); selected <code class="font-mono">iter 400</code> (<code class="font-mono">val_loss=0.831</code>) before late overfitting.</p>
              <div id="chart-p9-loss-curve" class="w-full h-72"></div>
            </div>
            <div class="rounded-xl border border-border bg-card p-4 space-y-2">
              <h3 class="font-serif text-lg font-semibold">Base vs. QLoRA Correctness Across All 8 Buckets</h3>
              <p class="text-xs text-muted-foreground">Primary <code class="font-mono">qwen3.5:9b</code> judge scores across every <code class="font-mono">(mode, slice)</code> cell (<code class="font-mono">N=120</code> paired runs).</p>
              <div id="chart-p9-slice-comparison" class="w-full h-72"></div>
            </div>
          </div>

          <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div class="rounded-xl border border-border bg-card p-4 space-y-2">
              <h3 class="font-serif text-lg font-semibold">Prompt-Level Win / Tie / Loss Counts</h3>
              <p class="text-xs text-muted-foreground">Pooled <code class="font-mono">14 improved / 95 tied / 11 regressed</code> (net <code class="font-mono">+3</code>; Zero-Shot <code class="font-mono">+5</code>, RAG <code class="font-mono">-2</code>).</p>
              <div id="chart-p9-win-loss" class="w-full h-72"></div>
            </div>
            <div class="rounded-xl border border-border bg-card p-4 space-y-2">
              <h3 class="font-serif text-lg font-semibold">Zero-Shot vs. RAG vs. Post-Hoc Hybrid (<code class="font-mono text-sm">tau=0.68</code>)</h3>
              <p class="text-xs text-muted-foreground">Cosine-gated routing over measured outputs (<code class="font-mono">Base Hybrid 3.70 &rarr; QLoRA Hybrid 3.78</code> overall).</p>
              <div id="chart-p9-hybrid-routing" class="w-full h-72"></div>
            </div>
          </div>

          <!-- Accessible Tabbed Diagnostics -->
          <div id="p9-tabs-root" class="space-y-3">
            <div class="tablist" role="tablist" aria-label="QLoRA experiment diagnostics">
              <button type="button" role="tab" id="p9tab-metrics" aria-selected="true" data-panel="p9panel-metrics">Head-to-Head Metrics, 95% CIs &amp; Dual-Judge Check</button>
              <button type="button" role="tab" id="p9tab-diffs" aria-selected="false" data-panel="p9panel-diffs">120-Item Paired Diff Explorer</button>
              <button type="button" role="tab" id="p9tab-recipe" aria-selected="false" data-panel="p9panel-recipe">QLoRA Recipe, Leakage-Safe Dataset &amp; Hybrid Sweep</button>
            </div>

            <div role="tabpanel" id="p9panel-metrics" aria-labelledby="p9tab-metrics" tabindex="0">
              <div class="tabpanel-box space-y-5">
                <div class="space-y-2">
                  <h4 class="font-serif text-base font-semibold">Paired Bootstrap 95% Confidence Intervals (<code class="font-mono text-xs">qwen3.5:9b</code>)</h4>
                  <div tabindex="0" role="region" aria-label="Bootstrap confidence intervals" class="table-wrapper book-scroll">
                    <table>
                      <thead>
                        <tr>
                          <th>Bucket</th>
                          <th>N</th>
                          <th>Base Mean</th>
                          <th>QLoRA Mean</th>
                          <th>Delta (FT - Base)</th>
                          <th>95% Bootstrap CI</th>
                          <th>Significant</th>
                        </tr>
                      </thead>
                      <tbody id="p9-ci-tbody"></tbody>
                    </table>
                  </div>
                </div>
                <div class="space-y-2">
                  <h4 class="font-serif text-base font-semibold">Dual-Judge Cross-Check (<code class="font-mono text-xs">qwen3.5:9b</code> vs. <code class="font-mono text-xs">qwen2.5:7b</code>)</h4>
                  <div class="rounded-xl border border-border bg-secondary/40 p-4 space-y-2 text-xs">
                    <p><strong>What Data Is Compared:</strong> All <code class="font-mono">240</code> generator outputs (<code class="font-mono">120</code> Base <code class="font-mono">llama3.2:3b</code> + <code class="font-mono">120</code> QLoRA <code class="font-mono">llama3.2-qlora:3b</code> across <code class="font-mono">60</code> Zero-Shot and <code class="font-mono">60</code> RAG prompts) were independently scored on <code class="font-mono">1–5</code> <code class="font-mono">correctness</code> by primary judge <code class="font-mono">qwen3.5:9b</code> (<code class="font-mono">MLX nvfp4</code>, <code class="font-mono">90.0%</code> human exact match, <code class="font-mono">QWK = 0.782</code>) and baseline judge <code class="font-mono">qwen2.5:7b</code> (<code class="font-mono">GGUF Q4_K_M</code>, <code class="font-mono">80.0%</code> human exact match, <code class="font-mono">QWK = 0.535</code>). Each row compares mean <code class="font-mono">FT - Base</code> correctness delta; <code class="font-mono">Direction Agrees</code> checks whether both judges share the same sign whenever <code class="font-mono">|Qwen3.5-9B Delta| &gt;= 0.10</code> (<code class="font-mono">&mdash;</code> when <code class="font-mono">&lt; 0.10</code>, e.g. <code class="font-mono">rag_unanswerable = 0.000</code>).</p>
                    <p><strong>Why 7/10 Active Buckets Agree (<code class="font-mono">True</code>):</strong> Both judges confirm positive lifts on <code class="font-mono">pooled_all</code> (<code class="font-mono">+0.133</code> vs <code class="font-mono">+0.225</code>), <code class="font-mono">zero_shot</code> (<code class="font-mono">+0.133</code> vs <code class="font-mono">+0.367</code>), <code class="font-mono">rag</code> (<code class="font-mono">+0.133</code> vs <code class="font-mono">+0.083</code>), <code class="font-mono">zero_shot_static</code> (<code class="font-mono">+0.450</code> vs <code class="font-mono">+0.250</code>), <code class="font-mono">zero_shot_live</code> (<code class="font-mono">+0.200</code> vs <code class="font-mono">+1.100</code>, where <code class="font-mono">qwen2.5:7b</code> over-credits fluent ungrounded guesses), and <code class="font-mono">rag_temporal_override</code> (<code class="font-mono">+1.200</code> vs <code class="font-mono">+0.400</code>), plus the slight dip on <code class="font-mono">zero_shot_temporal_override</code> (<code class="font-mono">-0.100</code> vs <code class="font-mono">-0.500</code>). Headline QLoRA gains are real behavioral shifts, not single-judge noise.</p>
                    <p><strong>What the 3 Divergent Buckets (<code class="font-mono">False</code>) Reveal:</strong></p>
                    <ul class="list-disc pl-4 space-y-1 text-muted-foreground">
                      <li><code class="font-mono text-foreground">rag_live</code> (<code class="font-mono">-0.400</code> vs <code class="font-mono">+0.050</code>): <code class="font-mono">qwen3.5:9b</code> penalizes terse 1–2 sentence QLoRA answers for omitting secondary benchmark numbers on 7 items; <code class="font-mono">qwen2.5:7b</code> credits the core mechanism (<code class="font-mono">+0.050</code>).</li>
                      <li><code class="font-mono text-foreground">zero_shot_unanswerable</code> (<code class="font-mono">-0.400</code> vs <code class="font-mono">0.000</code>): <code class="font-mono">qwen3.5:9b</code> catches QLoRA's single fabrication on <code class="font-mono">g54</code> (<code class="font-mono">5 &rarr; 1</code>); <code class="font-mono">qwen2.5:7b</code> had an offsetting baseline variance (<code class="font-mono">0.000</code> net).</li>
                      <li><code class="font-mono text-foreground">rag_static</code> (<code class="font-mono">+0.200</code> vs <code class="font-mono">0.000</code>): Both models abstain cleanly <code class="font-mono">20/20</code> (<code class="font-mono">Not in the sources.</code>), giving <code class="font-mono">0.000</code> on <code class="font-mono">qwen2.5:7b</code> vs a 1-item boundary flip (<code class="font-mono">+0.200</code>) on <code class="font-mono">qwen3.5:9b</code>.</li>
                    </ul>
                  </div>
                  <div tabindex="0" role="region" aria-label="Dual judge cross check" class="table-wrapper book-scroll">
                    <table>
                      <thead>
                        <tr>
                          <th>Bucket</th>
                          <th>Qwen3.5-9B Delta</th>
                          <th>Qwen2.5-7B Delta</th>
                          <th>Direction Agrees</th>
                        </tr>
                      </thead>
                      <tbody id="p9-dualjudge-tbody"></tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>

            <div role="tabpanel" id="p9panel-diffs" aria-labelledby="p9tab-diffs" hidden="until-found">
              <div class="tabpanel-box space-y-3">
                <div class="flex flex-wrap items-center justify-between gap-2">
                  <div class="flex flex-wrap items-center gap-2">
                    <select id="p9-diff-outcome" aria-label="Filter outcome" class="rounded-md border border-border bg-card px-2.5 py-1.5 text-xs">
                      <option value="all">All Outcomes (120)</option>
                      <option value="improved">Improved (14)</option>
                      <option value="regressed">Regressed (11)</option>
                      <option value="tied">Tied (95)</option>
                    </select>
                    <select id="p9-diff-mode" aria-label="Filter mode" class="rounded-md border border-border bg-card px-2.5 py-1.5 text-xs">
                      <option value="all">All Modes (120)</option>
                      <option value="zero_shot">Zero-Shot (60)</option>
                      <option value="rag">RAG (60)</option>
                    </select>
                  </div>
                  <input id="p9-diff-search" type="search" placeholder="Filter ID, prompt, answer, or rationale..." class="rounded-md border border-border bg-card px-2.5 py-1.5 text-xs w-64">
                </div>
                <div tabindex="0" role="region" aria-label="120 paired diff explorer" class="table-wrapper book-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>Item</th>
                        <th>Base (C/F/Cit)</th>
                        <th>QLoRA (C/F/Cit)</th>
                        <th>Delta</th>
                        <th>Prompt, Answers &amp; Judge Rationales</th>
                      </tr>
                    </thead>
                    <tbody id="p9-diff-tbody"></tbody>
                  </table>
                </div>
              </div>
            </div>

            <div role="tabpanel" id="p9panel-recipe" aria-labelledby="p9tab-recipe" hidden="until-found">
              <div class="tabpanel-box space-y-5">
                <div id="p9-recipe-grid" class="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs"></div>
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  <div class="space-y-2">
                    <h4 class="font-serif text-base font-semibold">Leakage-Safe Training Dataset (<code class="font-mono text-xs">data/qlora/dataset_manifest.json</code>)</h4>
                    <div tabindex="0" role="region" aria-label="QLoRA dataset breakdown" class="table-wrapper book-scroll">
                      <table>
                        <thead>
                          <tr>
                            <th>Category</th>
                            <th>Train</th>
                            <th>Valid</th>
                            <th>Total</th>
                          </tr>
                        </thead>
                        <tbody id="p9-dataset-tbody"></tbody>
                      </table>
                    </div>
                  </div>
                  <div class="space-y-2">
                    <h4 class="font-serif text-base font-semibold">Post-Hoc Cosine Threshold Sweep (<code class="font-mono text-xs">tau = 0.55..0.75</code>)</h4>
                    <div tabindex="0" role="region" aria-label="Hybrid routing tau sweep" class="table-wrapper book-scroll">
                      <table>
                        <thead>
                          <tr>
                            <th>tau</th>
                            <th>RAG Routed %</th>
                            <th>Base Hybrid</th>
                            <th>QLoRA Hybrid</th>
                            <th>Delta</th>
                          </tr>
                        </thead>
                        <tbody id="p9-tau-tbody"></tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <!-- Page 9: Chapter 8 - Failure Analysis & QLoRA Roadmap -->
        <section id="page-9" data-page="8" class="book-page hidden space-y-6">
          <div class="border-b border-border pb-4">
            <p class="font-mono text-xs uppercase tracking-wider text-muted-foreground">Chapter 8 · Synthesis &amp; Expected vs. Measured QLoRA</p>
            <h1 class="font-serif text-3xl font-bold mt-1">Failure Analysis &amp; Expected vs. Measured QLoRA Impact</h1>
            <p class="text-sm text-muted-foreground mt-1">Master comparison table across Base (<code class="font-mono text-xs">llama3.2:3b</code>) and QLoRA (<code class="font-mono text-xs">llama3.2-qlora:3b</code>), baseline failure modes, and what QLoRA actually changed in Chapter 7.</p>
          </div>

          <div class="space-y-2">
            <h3 id="master-table-caption" class="font-serif text-lg font-semibold">Master Evaluation Results Table (Base vs. QLoRA)</h3>
            <div tabindex="0" role="region" aria-labelledby="master-table-caption" class="table-wrapper book-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Model / Mode</th>
                    <th>Slice</th>
                    <th>N</th>
                    <th>Corr (Qwen2.5)</th>
                    <th>Corr (Qwen3.5)</th>
                    <th>Faithfulness</th>
                    <th>Citation Acc</th>
                    <th>Override Obed</th>
                    <th>Mean Latency (ms)</th>
                  </tr>
                </thead>
                <tbody id="master-tbody"></tbody>
              </table>
            </div>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
            <div class="rounded-xl border border-border bg-secondary/50 p-4 space-y-1.5">
              <span class="font-mono font-semibold text-primary">Failure 1 · Live-Only Blindspot</span>
              <p class="font-semibold">Static RAG collapses to 1.80 (Base) / 2.00 (QLoRA)</p>
              <p class="text-muted-foreground">Pre-T papers (LoRA, DPO, RAG) are absent from <code class="font-mono">live.jsonl</code>, so strict RAG grounding blocks parametric recall without threshold routing; Zero-Shot QLoRA lifts <code class="font-mono">static</code> <code class="font-mono">1.15 &rarr; 1.60</code>.</p>
            </div>
            <div class="rounded-xl border border-border bg-secondary/50 p-4 space-y-1.5">
              <span class="font-mono font-semibold text-primary">Failure 2 · 3B Omissions &amp; Terse FT Style</span>
              <p class="font-semibold">Spurious abstentions fixed (1 &rarr; 0), brevity trade-off</p>
              <p class="text-muted-foreground">QLoRA eliminates trailing abstentions and cuts latency 2.5&times;, but 1–2 sentence answers drop minor <code class="font-mono">live</code> RAG details (<code class="font-mono">4.85 &rarr; 4.45</code> under <code class="font-mono">Qwen3.5-9B</code>).</p>
            </div>
            <div class="rounded-xl border border-border bg-secondary/50 p-4 space-y-1.5">
              <span class="font-mono font-semibold text-primary">Failure 3 · Override Hedging (Solved in RAG)</span>
              <p class="font-semibold">Base 3/10 misses &rarr; QLoRA 0/10 abstentions</p>
              <p class="text-muted-foreground">QLoRA RAG lifts <code class="font-mono">temporal_override</code> correctness <code class="font-mono">3.60 &rarr; 4.80</code> (+1.20) and obedience <code class="font-mono">0.50 &rarr; 0.80</code> with <code class="font-mono">10/10</code> cited answers.</p>
            </div>
          </div>

          <div class="grid grid-cols-1 lg:grid-cols-2 gap-4 items-stretch">
            <div class="rounded-xl border border-border bg-card p-4 space-y-2">
              <h3 class="font-serif text-lg font-semibold">Measured Base &amp; QLoRA vs. Hybrid Target</h3>
              <p class="text-xs text-muted-foreground">Measured <code class="font-mono">Qwen3.5-9B</code> profiles for Base vs. QLoRA Zero-Shot and RAG against the gated hybrid target.</p>
              <div id="chart-p8-roadmap" class="w-full h-72"></div>
            </div>
            <div class="rounded-xl border border-border bg-secondary/40 p-5 space-y-3 text-sm">
              <h3 class="font-serif text-lg font-semibold">Expected vs. Measured QLoRA Findings</h3>
              <div class="space-y-2 text-xs leading-relaxed">
                <p><strong>What QLoRA Improved (Measured):</strong> Pooled correctness rose <code class="font-mono">+0.133</code> (<code class="font-mono">2.692 &rarr; 2.825</code>), RAG <code class="font-mono">temporal_override</code> surged <code class="font-mono">3.60 &rarr; 4.80</code> (obedience <code class="font-mono">0.50 &rarr; 0.80</code>, <code class="font-mono">10/10</code> cited), Zero-Shot <code class="font-mono">static</code> rose <code class="font-mono">1.15 &rarr; 1.60</code> (abstentions <code class="font-mono">18/20 &rarr; 0/20</code>), and mean latency dropped <strong>2.5&times;</strong> (<code class="font-mono">-1,078.9 ms</code>).</p>
                <p><strong>What Still Needs Calibration:</strong> Unweighted pre-T QA training pushed Zero-Shot QLoRA toward always-answering (<code class="font-mono">live</code> faithfulness <code class="font-mono">4.80 &rarr; 1.40</code>, <code class="font-mono">unanswerable</code> <code class="font-mono">5.00 &rarr; 4.60</code>), and raw cosine routing (<code class="font-mono">tau=0.68</code>) only routed 1/20 <code class="font-mono">static</code> prompts to Zero-Shot. See Chapter 7 for full QLoRA diagnostics.</p>
              </div>
              <div class="pt-2 border-t border-border font-mono text-[11px] text-muted-foreground">
                Reproduce: <code class="text-foreground">uv run python main.py book &amp;&amp; uv run pytest -v</code>
              </div>
            </div>
          </div>
        </section>

        <!-- Page 10: Appendix A - Compute & Subagent Orchestration -->
        <section id="page-10" data-page="9" class="book-page hidden space-y-6">
          <div class="border-b border-border pb-4">
            <p class="font-mono text-xs uppercase tracking-wider text-muted-foreground">Appendix A · Compute &amp; Orchestration</p>
            <h1 class="font-serif text-3xl font-bold mt-1">Mac mini M5 Pro Setup &amp; Parallel Subagent Fan-Out</h1>
            <p class="text-sm text-muted-foreground mt-1">Zero-sudo user-space provisioning of <code class="font-mono text-xs">uv</code>, Python 3.13, <code class="font-mono text-xs">Ollama 0.40.18</code>, and Apple Silicon <code class="font-mono text-xs">MLX</code> QLoRA training over SSH, plus 4-wave subagent parallelization.</p>
          </div>

          <div id="app-hw-cards" class="grid grid-cols-1 md:grid-cols-3 gap-4"></div>

          <div class="rounded-xl border border-border bg-secondary/50 p-5 space-y-3">
            <h2 class="font-serif text-xl font-semibold">4-Wave Subagent Fan-Out Architecture</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
              <div class="rounded-lg border border-border bg-card p-4 space-y-1.5">
                <span class="font-mono font-semibold text-primary">Wave 1 · 3 Agents (Parallel)</span>
                <p class="text-muted-foreground">1A: Mac mini user-space uv + Ollama.app Metal setup &amp; model pulls.</p>
                <p class="text-muted-foreground">1B: arXiv <code class="font-mono">cs.CL</code> API scraper + date-split verification.</p>
                <p class="text-muted-foreground">1C: Rubric, FAISS ingest/retriever, baselines, judge, metrics, pytest.</p>
              </div>
              <div class="rounded-lg border border-border bg-card p-4 space-y-1.5">
                <span class="font-mono font-semibold text-primary">Wave 2 · 4 Agents (Parallel)</span>
                <p class="text-muted-foreground">2A: 20 <code class="font-mono">static</code> prompts (15 landmark + 5 early-2024).</p>
                <p class="text-muted-foreground">2B: 20 <code class="font-mono">live</code> prompts (20/20 rank-1 FAISS verified).</p>
                <p class="text-muted-foreground">2C: 10 <code class="font-mono">temporal_override</code> prompts (10/10 rank-1 verified).</p>
                <p class="text-muted-foreground">2D: 10 <code class="font-mono">unanswerable</code> prompts (6 OOC + 4 near-miss distractors).</p>
              </div>
              <div class="rounded-lg border border-border bg-card p-4 space-y-1.5">
                <span class="font-mono font-semibold text-primary">Wave 3 · Baselines &amp; Judges</span>
                <p class="text-muted-foreground">Sequential <code class="font-mono">llama3.2:3b</code> baselines for clean uncontended latency.</p>
                <p class="text-muted-foreground">3A/4: <code class="font-mono">qwen2.5:7b</code> &amp; <code class="font-mono">qwen3.5:9b</code> MLX judge scoring.</p>
                <p class="text-muted-foreground">3B: Blinded Human Grader on 20 stratified samples (zero judge leakage).</p>
              </div>
              <div class="rounded-lg border border-border bg-card p-4 space-y-1.5">
                <span class="font-mono font-semibold text-primary">Wave 4 · Pre-T QLoRA</span>
                <p class="text-muted-foreground">Dataset builder (<code class="font-mono">1,060 / 120</code> leakage-checked pre-T pairs).</p>
                <p class="text-muted-foreground">MLX 4-bit QLoRA training (<code class="font-mono">4,171.9 s</code>, peak <code class="font-mono">11.5 GB</code>) + <code class="font-mono">nvfp4</code> Ollama import.</p>
                <p class="text-muted-foreground">Sequential FT eval + dual-judge scoring + bootstrap comparison.</p>
              </div>
            </div>
          </div>

          <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div class="rounded-xl border border-border bg-card p-4 space-y-2">
              <h3 class="font-serif text-lg font-semibold">48 GB Unified Memory Footprint</h3>
              <p class="text-xs text-muted-foreground">Base, QLoRA (<code class="font-mono">2.80 GB nvfp4 MLX</code>), dual judges, and <code class="font-mono">bge-small</code> vs. M5 Pro headroom.</p>
              <div id="chart-app-memory" class="w-full h-72"></div>
            </div>
            <div class="rounded-xl border border-border bg-card p-4 space-y-2">
              <h3 class="font-serif text-lg font-semibold">Judge Decoding Throughput Engineering</h3>
              <p class="text-xs text-muted-foreground">End-to-end 120-item median <code class="font-mono">tok/s</code> (<code class="font-mono">Qwen2.5-7B</code> GGUF vs <code class="font-mono">Qwen3.5-9B</code> MLX) alongside GBNF vs JSON tuning.</p>
              <div id="chart-app-throughput" class="w-full h-72"></div>
            </div>
          </div>
        </section>

        <!-- Page 11: Appendix B - Three Research Continuations -->
        <section id="page-11" data-page="10" class="book-page hidden space-y-6">
          <div class="border-b border-border pb-4">
            <p class="font-mono text-xs uppercase tracking-wider text-muted-foreground">Appendix B · Future Research Agenda</p>
            <h1 class="font-serif text-3xl font-bold mt-1">Three Research Continuations Beyond Baselines &amp; QLoRA v1</h1>
            <p class="text-sm text-muted-foreground mt-1">Continuation 1's pre-T QLoRA weights are now empirically evaluated in Chapter 7 (<code class="font-mono text-xs">llama3.2-qlora:3b</code>); below is how its findings sharpen DPO abstention calibration, cross-encoder gating, and full-paper packing.</p>
          </div>

          <div class="grid grid-cols-1 lg:grid-cols-3 gap-4 text-xs">
            <div class="rounded-xl border border-border bg-secondary/50 p-5 space-y-2.5 flex flex-col justify-between">
              <div class="space-y-2">
                <div class="flex items-center justify-between">
                  <span class="font-mono font-semibold text-primary">Continuation 1 · Tested in Ch. 7</span>
                  <span class="rounded bg-muted px-2 py-0.5 font-mono text-[10px]">Static 1.80 &rarr; &ge;4.20</span>
                </div>
                <h2 class="font-serif text-lg font-semibold">Cross-Encoder Gated Routing + Calibrated QLoRA</h2>
                <p class="text-muted-foreground"><strong>Ch. 7 Status:</strong> <code class="font-mono">llama3.2-qlora:3b</code> weights are built &amp; tested (<code class="font-mono">ZS static 1.15 &rarr; 1.60</code>), but cosine <code class="font-mono">tau=0.68</code> alone routed 19/20 static queries to RAG and unweighted QA caused Zero-Shot overconfidence.</p>
                <p><strong>Next Step:</strong> Replace raw cosine <code class="font-mono">tau</code> with a calibrated <code class="font-mono">bge-reranker-v2-m3</code> gate and re-balance Cat B abstention pairs (<code class="font-mono">&gt;25%</code> share) so low-confidence queries cleanly route to <code class="font-mono">[parametric &lt; T]</code> without hurting <code class="font-mono">unanswerable</code>.</p>
              </div>
              <div class="pt-2 border-t border-border font-mono text-[11px] text-muted-foreground">Hypothesis: Cross-encoder gating unlocks the full static recall lift safely.</div>
            </div>

            <div class="rounded-xl border border-border bg-secondary/50 p-5 space-y-2.5 flex flex-col justify-between">
              <div class="space-y-2">
                <div class="flex items-center justify-between">
                  <span class="font-mono font-semibold text-primary">Continuation 2</span>
                  <span class="rounded bg-muted px-2 py-0.5 font-mono text-[10px]">Override 80% &rarr; &ge;95%</span>
                </div>
                <h2 class="font-serif text-lg font-semibold">Abstention Calibration &amp; Counterfactual Conflict DPO</h2>
                <p class="text-muted-foreground"><strong>Ch. 7 Bottleneck:</strong> QLoRA already lifted RAG <code class="font-mono">temporal_override</code> (<code class="font-mono">3.60 &rarr; 4.80</code>, obedience <code class="font-mono">0.50 &rarr; 0.80</code>), yet Zero-Shot QLoRA over-answered <code class="font-mono">live</code>/<code class="font-mono">unanswerable</code> and 2/10 override queries still hedged.</p>
                <p><strong>Proposed Method:</strong> Pair time-decay RRF reranking (<code class="font-mono">exp(&lambda; &middot; &Delta;t)</code>) with DPO preference pairs covering (a) <code class="font-mono">pre-T + post-T</code> conflict updates and (b) explicit abstention vs. ungrounded speculation on unseen post-T facts.</p>
              </div>
              <div class="pt-2 border-t border-border font-mono text-[11px] text-muted-foreground">Hypothesis: Fixes zero-shot overconfidence while closing remaining override misses.</div>
            </div>

            <div class="rounded-xl border border-border bg-secondary/50 p-5 space-y-2.5 flex flex-col justify-between">
              <div class="space-y-2">
                <div class="flex items-center justify-between">
                  <span class="font-mono font-semibold text-primary">Continuation 3</span>
                  <span class="rounded bg-muted px-2 py-0.5 font-mono text-[10px]">Live 4.45/4.85 &rarr; &ge;4.90</span>
                </div>
                <h2 class="font-serif text-lg font-semibold">Section-Aware Full-Paper Ingestion &amp; Submodular Packing</h2>
                <p class="text-muted-foreground"><strong>Ch. 7 Bottleneck:</strong> Abstract-only chunks (175-word median) omit hyperparameter &amp; ablation tables probed by near-miss prompts (<code class="font-mono">u07–u10</code>) and terse QLoRA RAG outputs drop secondary details (<code class="font-mono">live 4.85 &rarr; 4.45</code>).</p>
                <p><strong>Proposed Method:</strong> Parse <code class="font-mono">ar5iv</code> HTML/LaTeX full papers into hierarchical section + table chunks, retrieve via <code class="font-mono">bge-m3</code> multi-vector scoring, and pack context via budgeted submodular maximization with explicit <code class="font-mono">[arxiv_id]</code> citation constraints.</p>
              </div>
              <div class="pt-2 border-t border-border font-mono text-[11px] text-muted-foreground">Hypothesis: Unlocks table/ablation QA while restoring citation completeness.</div>
            </div>
          </div>

          <div class="space-y-2">
            <h3 id="cont-matrix-caption" class="font-serif text-lg font-semibold">Experimental Continuation Matrix</h3>
            <div tabindex="0" role="region" aria-labelledby="cont-matrix-caption" class="table-wrapper book-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Continuation</th>
                    <th>Target Bottleneck</th>
                    <th>Primary Intervention</th>
                    <th>Success Metric Target</th>
                    <th>M5 Pro Compute / Latency</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <th class="font-mono text-xs">1 · QLoRA + Cross-Encoder Gate</th>
                    <td data-label="Bottleneck" class="text-xs">Raw cosine <code class="font-mono">tau=0.68</code> leaves 19/20 <code class="font-mono">static</code> on RAG (<code class="font-mono">2.00</code>)</td>
                    <td data-label="Intervention" class="text-xs">Chapter 7 QLoRA weights + <code class="font-mono">bge-reranker-v2-m3</code> threshold gate</td>
                    <td data-label="Target" class="font-mono text-xs">Static &ge; 4.20 · Unansw. &ge; 4.90</td>
                    <td data-label="Compute" class="font-mono text-xs">Done (1.16h MLX) + +25 ms/query</td>
                  </tr>
                  <tr>
                    <th class="font-mono text-xs">2 · Abstention &amp; Conflict DPO</th>
                    <td data-label="Bottleneck" class="text-xs">ZS overconfidence (<code class="font-mono">live</code> faith 1.40) + 2/10 override misses</td>
                    <td data-label="Intervention" class="text-xs">Time-decay RRF rerank + DPO on abstention &amp; mixed-date conflict pairs</td>
                    <td data-label="Target" class="font-mono text-xs">Override Obed &ge; 0.95 · ZS Unansw = 5.0</td>
                    <td data-label="Compute" class="font-mono text-xs">~2.0h DPO · +15 ms/query</td>
                  </tr>
                  <tr>
                    <th class="font-mono text-xs">3 · Full-Paper Packing</th>
                    <td data-label="Bottleneck" class="text-xs">Abstract depth limits &amp; terse FT <code class="font-mono">live</code> omissions (<code class="font-mono">4.45</code>)</td>
                    <td data-label="Intervention" class="text-xs"><code class="font-mono">ar5iv</code> section/table chunks + <code class="font-mono">bge-m3</code> + submodular packing</td>
                    <td data-label="Target" class="font-mono text-xs">Live &ge; 4.90 · Deep Table QA &ge; 4.00</td>
                    <td data-label="Compute" class="font-mono text-xs">~0.5h ingest · +90 ms/query</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div class="rounded-xl border border-border bg-card p-4 space-y-2">
              <h3 class="font-serif text-lg font-semibold">Projected Cumulative Slice Trajectory</h3>
              <p class="text-xs text-muted-foreground">Expected slice correctness progression as each continuation stacks onto Baseline RAG.</p>
              <div id="chart-appb-impact" class="w-full h-72"></div>
            </div>
            <div class="rounded-xl border border-border bg-card p-4 space-y-2">
              <h3 class="font-serif text-lg font-semibold">Accuracy Lift vs. Added Latency &amp; Compute</h3>
              <p class="text-xs text-muted-foreground">Bubble size indicates approximate M5 Pro training/ingestion hours per track.</p>
              <div id="chart-appb-pareto" class="w-full h-72"></div>
            </div>
          </div>
        </section>

        <!-- Page 12: Appendix C - Stronger Judge Experiment (Qwen3.5-9B MLX vs. Qwen2.5-7B) -->
        <section id="page-12" data-page="11" class="book-page hidden space-y-6">
          <div class="border-b border-border pb-4">
            <p class="font-mono text-xs uppercase tracking-wider text-muted-foreground">Appendix C · Stronger Cross-Family Judge Ablation</p>
            <h1 class="font-serif text-3xl font-bold mt-1">Qwen3.5-9B (MLX) vs. Qwen2.5-7B Judge Comparison</h1>
            <p class="text-sm text-muted-foreground mt-1">Holding all 120 <code class="font-mono text-xs">llama3.2:3b</code> outputs frozen, we compare <code class="font-mono text-xs">qwen2.5:7b</code> (GGUF <code class="font-mono text-xs">Q4_K_M</code>) against <code class="font-mono text-xs">qwen3.5:9b</code> (<code class="font-mono text-xs">9.7B</code>, <code class="font-mono text-xs">nvfp4</code>, Apple Silicon <code class="font-mono text-xs">MLX</code> runner, <code class="font-mono text-xs">think=False</code>) on human calibration, speed, median latency, token volume, and discordance.</p>
          </div>

          <!-- Explicit Bottom-Line Recommendation Banner -->
          <div class="rounded-xl border-2 border-primary/40 bg-secondary/70 p-5 space-y-2">
            <div class="flex flex-wrap items-center justify-between gap-2">
              <span class="font-mono text-xs font-semibold uppercase tracking-wider text-primary">Bottom-Line Verdict</span>
              <span class="rounded-full bg-primary text-primary-foreground px-3 py-0.5 font-mono text-xs font-semibold">Winner: qwen3.5:9b (MLX)</span>
            </div>
            <p class="font-serif text-lg font-semibold">YES — Use <code class="font-mono text-base">qwen3.5:9b</code> (MLX, <code class="font-mono text-base">think=False</code>) over <code class="font-mono text-base">qwen2.5:7b</code> as the default benchmark judge on Mac mini M5 Pro.</p>
            <p class="text-xs text-muted-foreground leading-relaxed">
              It is <strong>both more accurate and faster</strong>: exact match vs. blind human grades jumps from <strong>80.0% to 90.0%</strong> (<strong>+10.0 pp</strong> overall, <strong>+25.0 pp</strong> on correctness), severity-weighted agreement (<code class="font-mono">QWK</code>) climbs from <strong>0.535 (moderate) to 0.782 (strong)</strong>, and Apple Silicon <code class="font-mono">MLX</code> runs <strong>2.06× faster in median wall time</strong> (<strong>3.76 s</strong> vs <strong>7.75 s</strong> per item; <strong>36.86</strong> vs <strong>25.01 median tok/s</strong>) while writing <strong>1.90× richer rationales</strong> (<strong>119.5</strong> vs <strong>63.0 median tokens</strong>). Only keep <code class="font-mono">qwen2.5:7b</code> on RAM-limited laptops (&le;16 GB) where <code class="font-mono">4.68 GB</code> fits better than <code class="font-mono">7.64 GB</code>.
            </p>
          </div>

          <div id="appc-kpis" class="grid grid-cols-2 lg:grid-cols-4 gap-4"></div>

          <!-- Plain-English Guide to Calibration Metrics -->
          <div class="rounded-xl border border-border bg-secondary/40 p-4 space-y-2 text-xs">
            <h2 class="font-serif text-base font-semibold">How to Read These Judge Metrics (Plain English)</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <p class="font-semibold text-foreground">Exact Score Match vs. Human ("Pooled Exact")</p>
                <p class="text-muted-foreground mt-0.5">Each of the <strong>20 blind test answers</strong> is graded by a human on <strong>3 separate 1–5 scales</strong> (<code class="font-mono">correctness</code>, <code class="font-mono">faithfulness</code>, <code class="font-mono">citation_accuracy</code>) = <strong>60 total human ratings</strong>. <strong>90.0%</strong> means the judge gave the <strong>exact same integer</strong> as the human on <strong>54 out of 60</strong> ratings (vs. <strong>48/60 = 80.0%</strong> for Qwen2.5-7B).</p>
              </div>
              <div>
                <p class="font-semibold text-foreground">Severity-Weighted Agreement ("Weighted Kappa / QWK")</p>
                <p class="text-muted-foreground mt-0.5">Raw % match treats every miss equally and can look high just because score <code class="font-mono">5</code> is common. <strong>QWK</strong> (<code class="font-mono">-1</code> to <code class="font-mono">+1</code>; <code class="font-mono">0</code> = random luck, <code class="font-mono">1</code> = human clone) adjusts for luck and punishes big misses quadratically—scoring <code class="font-mono">1</code> when human said <code class="font-mono">5</code> hurts <strong>16× more</strong> than <code class="font-mono">4 vs 5</code>. Rule of thumb: <code class="font-mono">0.40–0.60</code> moderate, <code class="font-mono">&gt;0.75</code> strong.</p>
              </div>
            </div>
          </div>

          <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div class="rounded-xl border border-border bg-card p-4 space-y-2">
              <h3 class="font-serif text-lg font-semibold">Speed, Median Time &amp; Output Tokens (<code class="font-mono text-sm">N=120</code>)</h3>
              <p class="text-xs text-muted-foreground">Head-to-head on Mac mini M5 Pro: Qwen3.5-9B MLX is <strong>2.06× faster</strong> per item despite generating <strong>+56.5 more tokens</strong>.</p>
              <div id="chart-appc-speed-tokens" class="w-full h-72"></div>
            </div>
            <div class="rounded-xl border border-border bg-card p-4 space-y-2">
              <h3 class="font-serif text-lg font-semibold">Blind Human Agreement (<code class="font-mono text-sm">N=20</code> / 60 Ratings)</h3>
              <p class="text-xs text-muted-foreground">Exact match vs. human rises +10.0 pp overall (95% CI [1.67%, 20.0%]) and +25.0 pp on correctness.</p>
              <div id="chart-appc-judge-human" class="w-full h-72"></div>
            </div>
          </div>

          <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div class="rounded-xl border border-border bg-card p-4 space-y-2">
              <h3 class="font-serif text-lg font-semibold">Slice Correctness Shift on Frozen Outputs</h3>
              <p class="text-xs text-muted-foreground">Stricter zero-shot hallucination penalties (<code class="font-mono">-2.20</code> override) and higher RAG live credit (<code class="font-mono">+1.00</code>).</p>
              <div id="chart-appc-slice-deltas" class="w-full h-72"></div>
            </div>
            <div class="rounded-xl border border-border bg-card p-4 space-y-2">
              <h3 class="font-serif text-lg font-semibold">Inter-Judge Confusion Matrix (<code class="font-mono text-sm">N = 360</code> Pairs)</h3>
              <p class="text-xs text-muted-foreground">86.11% exact agreement (90.0% within ±1, QWK = 0.7647) across all 120 baseline runs.</p>
              <div id="chart-appc-interjudge-matrix" class="w-full h-72"></div>
            </div>
          </div>

          <div id="appc-tabs-root" class="space-y-3">
            <div class="tablist" role="tablist" aria-label="Stronger judge diagnostics">
              <button type="button" role="tab" id="appctab-speed" aria-selected="true" data-panel="appcpanel-speed">Speed, Latency &amp; Token Comparison</button>
              <button type="button" role="tab" id="appctab-discordant" aria-selected="false" data-panel="appcpanel-discordant">Discordant Items Explorer (34)</button>
              <button type="button" role="tab" id="appctab-method" aria-selected="false" data-panel="appcpanel-method">When to Use Which &amp; MLX Notes</button>
            </div>

            <div role="tabpanel" id="appcpanel-speed" aria-labelledby="appctab-speed" tabindex="0">
              <div class="tabpanel-box">
                <div tabindex="0" role="region" aria-label="Judge speed and token comparison table" class="table-wrapper book-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>Operational Metric</th>
                        <th>Qwen2.5-7B (GGUF Q4_K_M)</th>
                        <th>Qwen3.5-9B (MLX nvfp4)</th>
                        <th>Difference (Qwen3.5 - Qwen2.5)</th>
                      </tr>
                    </thead>
                    <tbody id="speed-tokens-tbody"></tbody>
                  </table>
                </div>
              </div>
            </div>

            <div role="tabpanel" id="appcpanel-discordant" aria-labelledby="appctab-discordant" hidden="until-found">
              <div class="tabpanel-box space-y-3">
                <div class="flex flex-wrap items-center justify-between gap-2">
                  <select id="discordant-filter" aria-label="Filter discordance category" class="rounded-md border border-border bg-card px-2.5 py-1.5 text-xs">
                    <option value="all">All Categories (34)</option>
                    <option value="rag_completeness_strictness">RAG Completeness Strictness (16)</option>
                    <option value="false_abstention_fix">False Abstention Fix (8)</option>
                    <option value="hedged_zero_shot_ungrounded">Hedged Zero-Shot Ungrounded (6)</option>
                    <option value="rubric_boundary_shift">Rubric Boundary Shift (4)</option>
                  </select>
                  <input id="discordant-search" type="search" placeholder="Filter ID, prompt, or rationale..." class="rounded-md border border-border bg-card px-2.5 py-1.5 text-xs w-60">
                </div>
                <div tabindex="0" role="region" aria-label="Discordant judge items table" class="table-wrapper book-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>Item</th>
                        <th>Category</th>
                        <th>Qwen2.5-7B (C/F/Cit)</th>
                        <th>Qwen3.5-9B (C/F/Cit)</th>
                        <th>Prompt &amp; Judge Rationales</th>
                      </tr>
                    </thead>
                    <tbody id="discordant-tbody"></tbody>
                  </table>
                </div>
              </div>
            </div>

            <div role="tabpanel" id="appcpanel-method" aria-labelledby="appctab-method" hidden="until-found">
              <div class="tabpanel-box grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                <div class="rounded-xl border border-border bg-secondary/50 p-4 space-y-2">
                  <h4 class="font-serif text-base font-semibold">Decision Matrix: Qwen3.5-9B vs. Qwen2.5-7B</h4>
                  <p class="text-muted-foreground"><strong>Use <code class="font-mono">qwen3.5:9b</code> (MLX)</strong> for all final benchmark runs, QLoRA fine-tune comparisons, and human-audit replacement: higher accuracy (+10.0 pp exact, 0.782 QWK) AND half the median wait time (3.76 s vs 7.75 s).</p>
                  <p class="text-muted-foreground"><strong>Keep <code class="font-mono">qwen2.5:7b</code></strong> only on &le;16 GB RAM machines where a 7.64 GB model alongside generator/embeddings pressures unified memory.</p>
                </div>
                <div class="rounded-xl border border-border bg-secondary/50 p-4 space-y-2">
                  <h4 class="font-serif text-base font-semibold">MLX <code class="font-mono">think=False</code> Token Budget Protection</h4>
                  <p class="text-muted-foreground"><code class="font-mono">qwen3.5:9b</code> defaults to <code class="font-mono">thinking=true</code> in Ollama 0.40.18, which can exhaust <code class="font-mono">num_predict=512</code> inside hidden reasoning tokens before emitting JSON. Passing <code class="font-mono">"think": false</code> plus <code class="font-mono">&lt;think&gt;...&lt;/think&gt;</code> stripping achieved 120/120 valid structured verdicts.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        <!-- Bottom Book Page Navigation Footer -->
        <footer class="mt-8 pt-5 border-t border-border flex items-center justify-between gap-4">
          <button id="btn-prev-page" type="button" class="inline-flex items-center gap-2 rounded-lg border border-border bg-secondary px-4 py-2 text-sm font-medium hover:bg-accent disabled:opacity-40 disabled:pointer-events-none">
            ← Previous Page
          </button>
          <span id="footer-folio" class="font-serif italic text-sm text-muted-foreground">Page 1 of 12</span>
          <button id="btn-next-page" type="button" class="inline-flex items-center gap-2 rounded-lg bg-primary text-primary-foreground px-4 py-2 text-sm font-medium hover:opacity-90 disabled:opacity-40 disabled:pointer-events-none">
            Next Page →
          </button>
        </footer>
      </article>
    </main>
  </div>

  <script id="experiment-data" type="application/json">__PAYLOAD_JSON__</script>
  <script>
    const DATA = JSON.parse(document.getElementById("experiment-data").textContent);

    const CHAPTERS = [
      { idx: 0, label: "Prologue", title: "Abstract & Agenda", desc: "Motivation, temporal cutoff T=2025-01-01, executive KPIs, and Base vs. QLoRA overview." },
      { idx: 1, label: "Chapter 1", title: "Slices, Metrics & Agents", desc: "4 golden slices (+/- examples), 1–5 metrics, runtime agents & orchestration subagents." },
      { idx: 2, label: "Chapter 2", title: "Leakage-Safe Date Split", desc: "300 static (<T) vs. 200 live (>=T) cs.CL abstracts with 0 ID overlap." },
      { idx: 3, label: "Chapter 3", title: "Ingestion & FAISS Retrieval", desc: "bge-small-en-v1.5 384-d embeddings, IndexFlatIP, and after=T filtering." },
      { idx: 4, label: "Chapter 4", title: "60-Prompt Golden Set", desc: "Interactive explorer across Static, Live, Temporal Override, and Unanswerable." },
      { idx: 5, label: "Chapter 5", title: "Rubric & Judge Calibration", desc: "48.3% -> 80.0% -> 90.0% exact agreement lift via CoT ordering & MLX judge." },
      { idx: 6, label: "Chapter 6", title: "Baselines & Run Inspector", desc: "240 Base vs. QLoRA runs, latency distributions, and 4-card prompt drilldown." },
      { idx: 7, label: "Chapter 7", title: "QLoRA Fine-Tuning", desc: "llama3.2-qlora:3b vs. llama3.2:3b: loss curves, 95% CIs, hybrid routing & 120-item diff explorer." },
      { idx: 8, label: "Chapter 8", title: "Failure Analysis & Roadmap", desc: "Master results table, 3 failure modes, and expected vs. measured QLoRA target." },
      { idx: 9, label: "Appendix A", title: "Compute & Orchestration", desc: "Mac mini M5 Pro user-space Ollama setup, MLX QLoRA training, and parallel subagents." },
      { idx: 10, label: "Appendix B", title: "3 Research Continuations", desc: "Cross-encoder QLoRA gating, abstention/conflict DPO, and full-paper submodular RAG." },
      { idx: 11, label: "Appendix C", title: "Stronger Judge (Qwen3.5-9B)", desc: "Bottom-line verdict, QWK guide, speed/token comparison, and 34 discordant cases." },
    ];

    let currentPage = 0;
    const renderedPages = new Set();

    function escapeHtml(str) {
      return String(str ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    }

    function resolvedDarkMode() {
      const meta = document.querySelector('meta[name="color-scheme"]')?.content || "light dark";
      if (meta === "dark") return true;
      if (meta === "light") return false;
      return window.matchMedia("(prefers-color-scheme: dark)").matches;
    }

    function plotlyTheme() {
      const dark = resolvedDarkMode();
      return {
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: {
          family: "Inter, system-ui, sans-serif",
          color: dark ? "#e4e4e7" : "#27272a",
          size: 12,
        },
        gridColor: dark ? "#27272a" : "#e4e4e7",
        zsColor: dark ? "#94a3b8" : "#64748b",
        ragColor: dark ? "#38bdf8" : "#0284c7",
        accentColor: dark ? "#fbbf24" : "#d97706",
        greenColor: dark ? "#34d399" : "#059669",
        roseColor: dark ? "#fb7185" : "#e11d48",
      };
    }

    function baseLayout(extra = {}) {
      const t = plotlyTheme();
      return {
        paper_bgcolor: t.paper_bgcolor,
        plot_bgcolor: t.plot_bgcolor,
        font: t.font,
        margin: { t: 24, r: 18, b: 44, l: 48 },
        hovermode: "closest",
        ...extra,
      };
    }

    const PLOT_CONFIG = { responsive: true, displayModeBar: false };

    function renderChartsForPage(idx, force = false) {
      if (!window.Plotly) return;
      if (force) renderedPages.delete(idx);
      if (renderedPages.has(idx)) {
        document.querySelectorAll(`section[data-page="${idx}"] .js-plotly-plot`).forEach((el) => {
          Plotly.Plots.resize(el);
        });
        return;
      }
      renderedPages.add(idx);
      const t = plotlyTheme();

      if (idx === 0) {
        const sc = DATA.model_comparison.score_comparison;
        const bBase = sc.base.by_mode_slice;
        const bFt = sc.ft.by_mode_slice;
        const cats = ["Static", "Live", "Temporal Override", "Unanswerable", "Override Obed (×5)"];
        const extract = (bms, mode) => [
          bms[mode].static.correctness,
          bms[mode].live.correctness,
          bms[mode].temporal_override.correctness,
          bms[mode].unanswerable.correctness,
          (bms[mode].temporal_override.override_obedience ?? 0) * 5,
        ];
        const baseZs = extract(bBase, "zero_shot");
        const ftZs = extract(bFt, "zero_shot");
        const baseRag = extract(bBase, "rag");
        const ftRag = extract(bFt, "rag");
        Plotly.newPlot("chart-p1-overview", [
          { x: cats, y: baseZs, name: "Base Zero-Shot (llama3.2:3b)", type: "bar", marker: { color: t.zsColor }, text: baseZs.map((v) => v.toFixed(2)), textposition: "outside" },
          { x: cats, y: ftZs, name: "QLoRA Zero-Shot (llama3.2-qlora:3b)", type: "bar", marker: { color: t.accentColor }, text: ftZs.map((v) => v.toFixed(2)), textposition: "outside" },
          { x: cats, y: baseRag, name: "Base RAG k=4", type: "bar", marker: { color: t.ragColor }, text: baseRag.map((v) => v.toFixed(2)), textposition: "outside" },
          { x: cats, y: ftRag, name: "QLoRA RAG k=4", type: "bar", marker: { color: t.greenColor }, text: ftRag.map((v) => v.toFixed(2)), textposition: "outside" },
        ], baseLayout({
          barmode: "group",
          margin: { t: 44, r: 18, b: 44, l: 48 },
          yaxis: { range: [0, 5.9], title: "Score (0–5, Qwen3.5-9B)", gridcolor: t.gridColor },
          xaxis: { gridcolor: t.gridColor },
          legend: { orientation: "h", y: 1.18 },
        }), PLOT_CONFIG);
      }

      if (idx === 1) {
        const caps = ["Parametric Recall", "Post-T Freshness", "Prior Override", "Abstention Safety", "Citation Grounding"];
        const sliceNames = ["static (20)", "live (20)", "temporal_override (10)", "unanswerable (10)"];
        const matrix = [
          [2, 0, 0, 1, 1],
          [0, 2, 0, 1, 2],
          [1, 2, 2, 1, 2],
          [0, 1, 0, 2, 2],
        ];
        const labels = matrix.map((row) => row.map((v) => (v === 2 ? "Primary" : v === 1 ? "Secondary" : "—")));
        Plotly.newPlot("chart-p2-taxonomy", [{
          x: caps,
          y: sliceNames,
          z: matrix,
          type: "heatmap",
          colorscale: resolvedDarkMode()
            ? [[0, "#18181b"], [0.5, "#1e3a8a"], [1, "#0284c7"]]
            : [[0, "#f4f4f5"], [0.5, "#bae6fd"], [1, "#0284c7"]],
          text: labels,
          texttemplate: "%{text}",
          showscale: false,
        }], baseLayout({
          margin: { t: 24, r: 18, b: 68, l: 148 },
          xaxis: { tickangle: -20, type: "category" },
          yaxis: { type: "category", autorange: "reversed" },
        }), PLOT_CONFIG);

        const arch = [
          "Grounded Cited<br>(Live RAG)",
          "Zero-Shot<br>Parametric",
          "False Abstention<br>(Static RAG)",
          "Clean Abstention<br>(Unanswerable)",
          "Hallucinated<br>Distractor",
        ];
        Plotly.newPlot("chart-p2-archetypes", [
          { x: arch, y: [5, 4, 1, 5, 1], name: "Correctness", type: "bar", marker: { color: t.ragColor } },
          { x: arch, y: [5, 1, 5, 5, 1], name: "Faithfulness", type: "bar", marker: { color: t.greenColor } },
          { x: arch, y: [5, 1, 5, 5, 2], name: "Citation Acc", type: "bar", marker: { color: t.accentColor } },
        ], baseLayout({
          barmode: "group",
          margin: { t: 24, r: 16, b: 64, l: 44 },
          yaxis: { title: "Rubric Score (1–5)", range: [0, 5.6], gridcolor: t.gridColor },
          legend: { orientation: "h", y: 1.15 },
        }), PLOT_CONFIG);
      }

      if (idx === 2) {
        const st = DATA.corpus.static.monthly;
        const lv = DATA.corpus.live.monthly;
        Plotly.newPlot("chart-p3-timeline", [
          { x: st.map((d) => d.month), y: st.map((d) => d.count), name: "Static (< T, n=300)", type: "bar", marker: { color: t.zsColor } },
          { x: lv.map((d) => d.month), y: lv.map((d) => d.count), name: "Live (>= T, n=200)", type: "bar", marker: { color: t.ragColor } },
        ], baseLayout({
          xaxis: { title: "Publication Month (YYYY-MM)", type: "category", tickangle: -45, gridcolor: t.gridColor },
          yaxis: { title: "Abstract Count", gridcolor: t.gridColor },
          legend: { orientation: "h", y: 1.12 },
          annotations: [{ x: "2025-01", y: 120, text: "Split T = 2025-01-01", showarrow: true, arrowhead: 2, ax: -45, ay: -25 }],
        }), PLOT_CONFIG);

        Plotly.newPlot("chart-p3-lengths", [
          { y: DATA.corpus.static.word_counts, name: "Static (n=300)", type: "box", marker: { color: t.zsColor }, boxmean: true },
          { y: DATA.corpus.live.word_counts, name: "Live (n=200)", type: "box", marker: { color: t.ragColor }, boxmean: true },
        ], baseLayout({
          yaxis: { title: "Abstract Word Count", gridcolor: t.gridColor },
        }), PLOT_CONFIG);
      }

      if (idx === 3) {
        Plotly.newPlot("chart-p4-recall", [{
          x: ["Static (20)", "Live (20)", "Temporal Override (10)", "Unanswerable (10)"],
          y: [0, 100, 100, 0],
          type: "bar",
          marker: { color: [t.zsColor, t.greenColor, t.ragColor, t.accentColor] },
          text: ["0% (Pre-T OOC)", "100% (20/20 Rank-1)", "100% (10/10 Rank-1)", "0% (Confirmed Absent)"],
          textposition: "outside",
        }], baseLayout({
          yaxis: { title: "Target Paper in Top-4 (%)", range: [0, 118], gridcolor: t.gridColor },
        }), PLOT_CONFIG);

        const slKeys = ["static", "live", "temporal_override", "unanswerable"];
        const slLabels = ["Static", "Live", "Override", "Unanswerable"];
        const retMs = slKeys.map((k) => DATA.rag_latency_split[k].retrieval_mean_ms);
        const genMs = slKeys.map((k) => DATA.rag_latency_split[k].generation_mean_ms);
        Plotly.newPlot("chart-p4-latency-split", [
          { y: slLabels, x: retMs, name: "FAISS Retrieval (ms)", type: "bar", orientation: "h", marker: { color: t.greenColor } },
          { y: slLabels, x: genMs, name: "Ollama Generation (ms)", type: "bar", orientation: "h", marker: { color: t.ragColor } },
        ], baseLayout({
          barmode: "stack",
          margin: { t: 24, r: 18, b: 44, l: 96 },
          xaxis: { title: "Mean Latency (ms)", gridcolor: t.gridColor },
          legend: { orientation: "h", y: 1.15 },
        }), PLOT_CONFIG);
      }

      if (idx === 4) {
        Plotly.newPlot("chart-p5-slices", [{
          labels: ["Static (20)", "Live (20)", "Temporal Override (10)", "Unanswerable (10)"],
          values: [20, 20, 10, 10],
          type: "pie",
          hole: 0.48,
          marker: { colors: [t.zsColor, t.ragColor, t.accentColor, t.roseColor] },
          textinfo: "label+value",
        }], baseLayout({
          margin: { t: 16, r: 16, b: 16, l: 16 },
          showlegend: false,
        }), PLOT_CONFIG);
      }

      if (idx === 5) {
        const dims = ["Correctness", "Faithfulness", "Citation Acc", "Pooled Exact", "Pooled ±1", "Override"];
        const v1 = [
          DATA.calibration.v1.correctness,
          DATA.calibration.v1.faithfulness,
          DATA.calibration.v1.citation_accuracy,
          DATA.calibration.v1.pooled_exact,
          DATA.calibration.v1.within_1,
          DATA.calibration.v1.override,
        ];
        const v2 = [
          DATA.calibration.v2.correctness,
          DATA.calibration.v2.faithfulness,
          DATA.calibration.v2.citation_accuracy,
          DATA.calibration.v2.pooled_exact,
          DATA.calibration.v2.within_1,
          DATA.calibration.v2.override,
        ];
        const hc = DATA.judge_comparison.human_calibration.cand;
        const q35 = [
          hc.dimensions.correctness.exact_match_pct,
          hc.dimensions.faithfulness.exact_match_pct,
          hc.dimensions.citation_accuracy.exact_match_pct,
          hc.pooled.exact_match_pct,
          hc.pooled.within_1_pct,
          hc.override_exact_pct,
        ];
        Plotly.newPlot("chart-p6-calibration", [
          { x: dims, y: v1, name: "v1 Uncalibrated (Qwen2.5-7B)", type: "bar", marker: { color: t.zsColor } },
          { x: dims, y: v2, name: "v2 Calibrated (Qwen2.5-7B)", type: "bar", marker: { color: t.ragColor } },
          { x: dims, y: q35, name: "Stronger Judge (Qwen3.5-9B MLX)", type: "bar", marker: { color: t.greenColor } },
        ], baseLayout({
          barmode: "group",
          yaxis: { title: "Agreement (%)", range: [0, 114], gridcolor: t.gridColor },
          legend: { orientation: "h", y: 1.16 },
          shapes: [{ type: "line", x0: -0.5, x1: 5.5, y0: 70, y1: 70, line: { color: t.roseColor, width: 2, dash: "dash" } }],
          annotations: [{ x: 4.5, y: 74, text: "70% Gate", showarrow: false, font: { color: t.roseColor, size: 11 } }],
        }), PLOT_CONFIG);

        const labels = ["1", "2", "3", "4", "5"];
        const z = DATA.calibration.confusion_matrix;
        Plotly.newPlot("chart-p6-heatmap", [{
          x: labels,
          y: labels,
          z: z,
          type: "heatmap",
          colorscale: resolvedDarkMode() ? "Cividis" : "Blues",
          text: z.map((row) => row.map((v) => String(v))),
          texttemplate: "%{text}",
          showscale: false,
        }], baseLayout({
          xaxis: { title: "Judge Score (Qwen2.5-7B)", type: "category" },
          yaxis: { title: "Human Score (Blind)", type: "category" },
        }), PLOT_CONFIG);
      }

      if (idx === 6) {
        const bms = DATA.results.by_mode_slice;
        const slKeys = ["static", "live", "temporal_override", "unanswerable"];
        const slNames = ["Static", "Live", "Temporal Override", "Unanswerable"];
        Plotly.newPlot("chart-p7-scores", [
          { x: slNames, y: slKeys.map((k) => bms.zero_shot[k].correctness), name: "Zero-Shot", type: "bar", marker: { color: t.zsColor } },
          { x: slNames, y: slKeys.map((k) => bms.rag[k].correctness), name: "RAG (k=4)", type: "bar", marker: { color: t.ragColor } },
        ], baseLayout({
          barmode: "group",
          yaxis: { title: "Mean Correctness (1–5)", range: [0, 5.6], gridcolor: t.gridColor },
          legend: { orientation: "h", y: 1.14 },
        }), PLOT_CONFIG);

        const bRows = DATA.behavior;
        const xLabels = bRows.map((r) => `${r.mode === "rag" ? "RAG" : "ZS"} · ${r.slice === "temporal_override" ? "override" : r.slice}`);
        Plotly.newPlot("chart-p7-behavior", [
          { x: xLabels, y: bRows.map((r) => Math.round((r.cited / r.n) * 100)), name: "Cited Answer %", type: "bar", marker: { color: t.greenColor } },
          { x: xLabels, y: bRows.map((r) => Math.round((r.uncited / r.n) * 100)), name: "Uncited Answer %", type: "bar", marker: { color: t.accentColor } },
          { x: xLabels, y: bRows.map((r) => Math.round((r.abstained / r.n) * 100)), name: "Abstained %", type: "bar", marker: { color: t.zsColor } },
        ], baseLayout({
          barmode: "stack",
          margin: { t: 24, r: 16, b: 78, l: 48 },
          xaxis: { tickangle: -30 },
          yaxis: { title: "Share of Slice (%)", range: [0, 105], gridcolor: t.gridColor },
          legend: { orientation: "h", y: 1.14 },
        }), PLOT_CONFIG);

        const zsLatTraces = slKeys.map((k, i) => ({
          y: DATA.runs.filter((r) => r.slice === k).map((r) => r.zero_shot.latency_ms),
          name: `ZS · ${slNames[i]}`,
          type: "box",
          boxpoints: "all",
          jitter: 0.3,
          marker: { color: t.zsColor, size: 4 },
        }));
        const ragLatTraces = slKeys.map((k, i) => ({
          y: DATA.runs.filter((r) => r.slice === k).map((r) => r.rag.latency_ms),
          name: `RAG · ${slNames[i]}`,
          type: "box",
          boxpoints: "all",
          jitter: 0.3,
          marker: { color: t.ragColor, size: 4 },
        }));
        Plotly.newPlot("chart-p7-latency", [...zsLatTraces, ...ragLatTraces], baseLayout({
          margin: { t: 24, r: 16, b: 84, l: 56 },
          xaxis: { tickangle: -25 },
          yaxis: { title: "Latency (ms)", gridcolor: t.gridColor },
          showlegend: false,
        }), PLOT_CONFIG);
      }

      if (idx === 7) {
        const mc = DATA.model_comparison;
        const trainLog = DATA.qlora_training.log;
        const trPts = trainLog.filter((r) => r.event === "train");
        const valPts = trainLog.filter((r) => r.event === "val");
        const selIter = DATA.qlora_training.manifest.selected_checkpoint_iter;
        const selVal = DATA.qlora_training.manifest.selected_checkpoint_val_loss;
        Plotly.newPlot("chart-p9-loss-curve", [
          { x: trPts.map((r) => r.iter), y: trPts.map((r) => r.train_loss), name: "Train Loss", type: "scatter", mode: "lines+markers", line: { color: t.ragColor, width: 2 }, marker: { size: 4 } },
          { x: valPts.map((r) => r.iter), y: valPts.map((r) => r.val_loss), name: "Validation Loss", type: "scatter", mode: "lines+markers", line: { color: t.roseColor, width: 2.5 }, marker: { size: 7 } },
        ], baseLayout({
          xaxis: { title: "Training Iteration", gridcolor: t.gridColor },
          yaxis: { title: "Cross-Entropy Loss", gridcolor: t.gridColor },
          legend: { orientation: "h", y: 1.14 },
          shapes: [{ type: "line", x0: selIter, x1: selIter, y0: 0, y1: 2.6, line: { color: t.greenColor, width: 2, dash: "dash" } }],
          annotations: [{ x: selIter, y: selVal + 0.35, text: `Selected iter ${selIter} (val=${selVal})`, showarrow: true, arrowhead: 2, ax: 35, ay: -30, font: { color: t.greenColor, size: 11 } }],
        }), PLOT_CONFIG);

        const sc = mc.score_comparison;
        const buckets = [
          ["zero_shot", "static", "ZS · Static"],
          ["zero_shot", "live", "ZS · Live"],
          ["zero_shot", "temporal_override", "ZS · Override"],
          ["zero_shot", "unanswerable", "ZS · Unansw."],
          ["rag", "static", "RAG · Static"],
          ["rag", "live", "RAG · Live"],
          ["rag", "temporal_override", "RAG · Override"],
          ["rag", "unanswerable", "RAG · Unansw."],
        ];
        const baseY = buckets.map(([m, s]) => sc.base.by_mode_slice[m][s].correctness);
        const ftY = buckets.map(([m, s]) => sc.ft.by_mode_slice[m][s].correctness);
        Plotly.newPlot("chart-p9-slice-comparison", [
          { x: buckets.map((b) => b[2]), y: baseY, name: "Base (llama3.2:3b)", type: "bar", marker: { color: t.zsColor }, text: baseY.map((v) => v.toFixed(2)), textposition: "outside" },
          { x: buckets.map((b) => b[2]), y: ftY, name: "QLoRA (llama3.2-qlora:3b)", type: "bar", marker: { color: t.greenColor }, text: ftY.map((v) => v.toFixed(2)), textposition: "outside" },
        ], baseLayout({
          barmode: "group",
          margin: { t: 24, r: 16, b: 68, l: 48 },
          xaxis: { tickangle: -25 },
          yaxis: { title: "Mean Correctness (1–5)", range: [0, 5.8], gridcolor: t.gridColor },
          legend: { orientation: "h", y: 1.15 },
        }), PLOT_CONFIG);

        const wl = mc.win_loss.by_mode_slice;
        Plotly.newPlot("chart-p9-win-loss", [
          { x: buckets.map((b) => b[2]), y: buckets.map(([m, s]) => wl[m][s].improved), name: "Improved (FT > Base)", type: "bar", marker: { color: t.greenColor } },
          { x: buckets.map((b) => b[2]), y: buckets.map(([m, s]) => wl[m][s].tied), name: "Tied", type: "bar", marker: { color: t.zsColor } },
          { x: buckets.map((b) => b[2]), y: buckets.map(([m, s]) => wl[m][s].regressed), name: "Regressed (FT < Base)", type: "bar", marker: { color: t.roseColor } },
        ], baseLayout({
          barmode: "stack",
          margin: { t: 24, r: 16, b: 68, l: 48 },
          xaxis: { tickangle: -25 },
          yaxis: { title: "Prompt Count", gridcolor: t.gridColor },
          legend: { orientation: "h", y: 1.15 },
        }), PLOT_CONFIG);

        const hy = mc.hybrid_routing.by_tau[String(mc.hybrid_routing.default_tau)] || mc.hybrid_routing.by_tau["0.68"];
        const hySlices = [
          ["static", "Static (20)"],
          ["live", "Live (20)"],
          ["temporal_override", "Override (10)"],
          ["unanswerable", "Unansw. (10)"],
          ["overall", "Overall (60)"],
        ];
        const getVal = (root, mode, sl) => sl === "overall" ? root.overall[mode].correctness : root.by_mode_slice[mode][sl].correctness;
        const getHyVal = (side, sl) => sl === "overall" ? side.overall.correctness : side.by_slice[sl].correctness;
        Plotly.newPlot("chart-p9-hybrid-routing", [
          { x: hySlices.map((s) => s[1]), y: hySlices.map(([s]) => getVal(sc.base, "zero_shot", s)), name: "Base Zero-Shot", type: "bar", marker: { color: t.zsColor } },
          { x: hySlices.map((s) => s[1]), y: hySlices.map(([s]) => getVal(sc.ft, "zero_shot", s)), name: "QLoRA Zero-Shot", type: "bar", marker: { color: t.accentColor } },
          { x: hySlices.map((s) => s[1]), y: hySlices.map(([s]) => getVal(sc.base, "rag", s)), name: "Base RAG", type: "bar", marker: { color: t.ragColor } },
          { x: hySlices.map((s) => s[1]), y: hySlices.map(([s]) => getVal(sc.ft, "rag", s)), name: "QLoRA RAG", type: "bar", marker: { color: t.greenColor } },
          { x: hySlices.map((s) => s[1]), y: hySlices.map(([s]) => getHyVal(hy.base, s)), name: "Base Hybrid (tau=0.68)", type: "bar", marker: { color: "#818cf8" } },
          { x: hySlices.map((s) => s[1]), y: hySlices.map(([s]) => getHyVal(hy.ft, s)), name: "QLoRA Hybrid (tau=0.68)", type: "bar", marker: { color: "#14b8a6" } },
        ], baseLayout({
          barmode: "group",
          margin: { t: 54, r: 16, b: 54, l: 48 },
          yaxis: { title: "Correctness (1–5)", range: [0, 5.8], gridcolor: t.gridColor },
          legend: { orientation: "h", y: 1.24 },
        }), PLOT_CONFIG);
      }

      if (idx === 8) {
        const sc = DATA.model_comparison.score_comparison;
        const axes = ["Static Recall", "Live Freshness", "Override Obed (×5)", "Abstention Safety", "Citation Discipline"];
        const profile = (block, mode) => [
          block.by_mode_slice[mode].static.correctness,
          block.by_mode_slice[mode].live.correctness,
          (block.by_mode_slice[mode].temporal_override.override_obedience ?? 0) * 5,
          block.by_mode_slice[mode].unanswerable.correctness,
          block.overall[mode].citation_accuracy,
        ];
        Plotly.newPlot("chart-p8-roadmap", [
          { x: axes, y: profile(sc.base, "zero_shot"), name: "Base Zero-Shot", type: "scatter", mode: "lines+markers", line: { color: t.zsColor, width: 2 } },
          { x: axes, y: profile(sc.ft, "zero_shot"), name: "QLoRA Zero-Shot (Measured)", type: "scatter", mode: "lines+markers", line: { color: t.accentColor, width: 2.5 } },
          { x: axes, y: profile(sc.base, "rag"), name: "Base RAG Live-Only", type: "scatter", mode: "lines+markers", line: { color: t.ragColor, width: 2 } },
          { x: axes, y: profile(sc.ft, "rag"), name: "QLoRA RAG Live-Only (Measured)", type: "scatter", mode: "lines+markers", line: { color: t.greenColor, width: 2.5 } },
          { x: axes, y: [4.30, 4.90, 4.75, 4.95, 4.98], name: "Target: QLoRA + Gated Hybrid", type: "scatter", mode: "lines+markers", line: { color: t.roseColor, width: 2.5, dash: "dot" } },
        ], baseLayout({
          yaxis: { title: "Capability (0–5, Qwen3.5-9B)", range: [0, 5.6], gridcolor: t.gridColor },
          margin: { t: 24, r: 18, b: 68, l: 48 },
          xaxis: { tickangle: -15, gridcolor: t.gridColor },
          legend: { orientation: "h", y: 1.22 },
        }), PLOT_CONFIG);
      }

      if (idx === 9) {
        const models = DATA.env.models;
        const llamaGb = models.find((m) => m.name === "llama3.2:3b")?.size_gb ?? 2.02;
        const qloraGb = models.find((m) => m.name.includes("qlora"))?.size_gb ?? 2.80;
        const qwen25Gb = models.find((m) => m.name.includes("qwen2.5"))?.size_gb ?? 4.68;
        const qwen35Gb = models.find((m) => m.name.includes("qwen3.5"))?.size_gb ?? 7.64;
        const bgeGb = 0.13;
        const freeGb = Number((48 - llamaGb - qloraGb - qwen25Gb - qwen35Gb - bgeGb).toFixed(2));
        Plotly.newPlot("chart-app-memory", [
          { y: ["48 GB Unified"], x: [llamaGb], name: `llama3.2:3b (${llamaGb} GB)`, type: "bar", orientation: "h", marker: { color: t.ragColor } },
          { y: ["48 GB Unified"], x: [qloraGb], name: `llama3.2-qlora:3b nvfp4 (${qloraGb} GB)`, type: "bar", orientation: "h", marker: { color: "#14b8a6" } },
          { y: ["48 GB Unified"], x: [qwen25Gb], name: `qwen2.5:7b (${qwen25Gb} GB)`, type: "bar", orientation: "h", marker: { color: t.accentColor } },
          { y: ["48 GB Unified"], x: [qwen35Gb], name: `qwen3.5:9b MLX (${qwen35Gb} GB)`, type: "bar", orientation: "h", marker: { color: t.roseColor } },
          { y: ["48 GB Unified"], x: [bgeGb], name: `bge-small (${bgeGb} GB)`, type: "bar", orientation: "h", marker: { color: t.greenColor } },
          { y: ["48 GB Unified"], x: [freeGb], name: `OS & KV Headroom (${freeGb} GB)`, type: "bar", orientation: "h", marker: { color: t.gridColor } },
        ], baseLayout({
          barmode: "stack",
          margin: { t: 24, r: 18, b: 44, l: 110 },
          xaxis: { range: [0, 48], title: "Gigabytes (GB)", gridcolor: t.gridColor },
          legend: { orientation: "h", y: 1.26 },
        }), PLOT_CONFIG);

        const op = DATA.judge_comparison.operational;
        const q25Med = op.base?.tok_per_s_median ?? 25.01;
        const q35Med = op.cand?.tok_per_s_median ?? 36.86;
        const q35Mean = op.cand?.tok_per_s_mean ?? 45.19;
        Plotly.newPlot("chart-app-throughput", [{
          x: ["Qwen2.5 GBNF (4-slot)", "Qwen2.5 120-item Median", "Qwen3.5 MLX Median", "Qwen3.5 MLX Mean", 'Qwen2.5 format="json" Peak'],
          y: [1.08, q25Med, q35Med, q35Mean, 59.7],
          type: "bar",
          marker: { color: [t.roseColor, t.zsColor, t.ragColor, t.accentColor, t.greenColor] },
          text: ["1.08", `${q25Med}`, `${q35Med}`, `${q35Mean}`, "59.7"],
          textposition: "outside",
        }], baseLayout({
          yaxis: { title: "Tokens / sec", range: [0, 68], gridcolor: t.gridColor },
        }), PLOT_CONFIG);
      }

      if (idx === 10) {
        const axes = ["Static (20)", "Live (20)", "Override (10)", "Unansw. (10)", "Deep Table QA"];
        Plotly.newPlot("chart-appb-impact", [
          { x: axes, y: [1.80, 4.85, 3.60, 5.00, 1.50], name: "Base RAG (Qwen3.5-9B)", type: "bar", marker: { color: t.zsColor } },
          { x: axes, y: [2.00, 4.45, 4.80, 5.00, 1.50], name: "Ch. 7 Measured: QLoRA RAG", type: "bar", marker: { color: t.ragColor } },
          { x: axes, y: [4.25, 4.85, 4.80, 4.95, 1.50], name: "+Cont. 1+2: Reranker Gate & Conflict DPO (Target)", type: "bar", marker: { color: t.accentColor } },
          { x: axes, y: [4.30, 4.95, 4.90, 5.00, 4.20], name: "+Cont. 3: Full-Paper Pack (Target)", type: "bar", marker: { color: t.greenColor } },
        ], baseLayout({
          barmode: "group",
          yaxis: { title: "Correctness / Target (1–5)", range: [0, 5.8], gridcolor: t.gridColor },
          margin: { t: 24, r: 16, b: 56, l: 48 },
          legend: { orientation: "h", y: 1.22 },
        }), PLOT_CONFIG);

        Plotly.newPlot("chart-appb-pareto", [{
          x: [25, 15, 90],
          y: [0.82, 0.35, 0.72],
          text: [
            "Cont. 1: QLoRA + Cross-Encoder (~1.2h done, +25 ms)",
            "Cont. 2: Abstention & Conflict DPO (~2.0h, +15 ms)",
            "Cont. 3: Full-Paper Pack (~0.5h, +90 ms)",
          ],
          textposition: ["top center", "bottom center", "top left"],
          mode: "markers+text",
          type: "scatter",
          marker: {
            size: [30, 36, 20],
            color: [t.ragColor, t.accentColor, t.greenColor],
            line: { width: 1.5, color: t.font.color },
          },
        }], baseLayout({
          xaxis: { title: "Added Inference Latency (ms / query)", range: [0, 115], gridcolor: t.gridColor },
          yaxis: { title: "Projected Overall Correctness Lift (+pts)", range: [0.1, 1.05], gridcolor: t.gridColor },
          margin: { t: 24, r: 24, b: 48, l: 56 },
        }), PLOT_CONFIG);
      }

      if (idx === 11) {
        const jc = DATA.judge_comparison;
        const op = jc.operational;
        const baseOp = op.base;
        const candOp = op.cand;

        Plotly.newPlot("chart-appc-speed-tokens", [
          {
            x: ["Median Speed<br>(tok/s)", "Mean Speed<br>(tok/s)", "Median Time<br>(sec / item)", "Median Length<br>(tokens / 10)"],
            y: [baseOp.tok_per_s_median, baseOp.tok_per_s_mean, Number((baseOp.latency_median_ms / 1000).toFixed(2)), Number((baseOp.eval_tokens_median / 10).toFixed(1))],
            name: "Qwen2.5-7B (GGUF)",
            type: "bar",
            marker: { color: t.zsColor },
            text: [`${baseOp.tok_per_s_median}`, `${baseOp.tok_per_s_mean}`, `${(baseOp.latency_median_ms / 1000).toFixed(2)}s`, `${baseOp.eval_tokens_median} tok`],
            textposition: "outside",
          },
          {
            x: ["Median Speed<br>(tok/s)", "Mean Speed<br>(tok/s)", "Median Time<br>(sec / item)", "Median Length<br>(tokens / 10)"],
            y: [candOp.tok_per_s_median, candOp.tok_per_s_mean, Number((candOp.latency_median_ms / 1000).toFixed(2)), Number((candOp.eval_tokens_median / 10).toFixed(1))],
            name: "Qwen3.5-9B (MLX)",
            type: "bar",
            marker: { color: t.ragColor },
            text: [`${candOp.tok_per_s_median}`, `${candOp.tok_per_s_mean}`, `${(candOp.latency_median_ms / 1000).toFixed(2)}s`, `${candOp.eval_tokens_median} tok`],
            textposition: "outside",
          },
        ], baseLayout({
          barmode: "group",
          margin: { t: 24, r: 16, b: 56, l: 48 },
          yaxis: { title: "Value (tok/s, sec, or tok÷10)", range: [0, 56], gridcolor: t.gridColor },
          legend: { orientation: "h", y: 1.15 },
        }), PLOT_CONFIG);

        const hb = jc.human_calibration.base;
        const hc = jc.human_calibration.cand;
        const dims = ["Correctness", "Faithfulness", "Citation Acc", "Pooled Exact", "Pooled ±1", "Override"];
        const baseY = [
          hb.dimensions.correctness.exact_match_pct,
          hb.dimensions.faithfulness.exact_match_pct,
          hb.dimensions.citation_accuracy.exact_match_pct,
          hb.pooled.exact_match_pct,
          hb.pooled.within_1_pct,
          hb.override_exact_pct,
        ];
        const candY = [
          hc.dimensions.correctness.exact_match_pct,
          hc.dimensions.faithfulness.exact_match_pct,
          hc.dimensions.citation_accuracy.exact_match_pct,
          hc.pooled.exact_match_pct,
          hc.pooled.within_1_pct,
          hc.override_exact_pct,
        ];
        Plotly.newPlot("chart-appc-judge-human", [
          { x: dims, y: baseY, name: "Qwen2.5-7B (Baseline)", type: "bar", marker: { color: t.zsColor }, text: baseY.map((v) => `${v}%`), textposition: "outside" },
          { x: dims, y: candY, name: "Qwen3.5-9B (MLX)", type: "bar", marker: { color: t.greenColor }, text: candY.map((v) => `${v}%`), textposition: "outside" },
        ], baseLayout({
          barmode: "group",
          yaxis: { title: "Human Agreement (%)", range: [0, 118], gridcolor: t.gridColor },
          legend: { orientation: "h", y: 1.15 },
          shapes: [{ type: "line", x0: -0.5, x1: 5.5, y0: 70, y1: 70, line: { color: t.roseColor, width: 2, dash: "dash" } }],
          annotations: [{ x: 4.5, y: 74, text: "70% Gate", showarrow: false, font: { color: t.roseColor, size: 11 } }],
        }), PLOT_CONFIG);

        const bs = jc.benchmark_shift;
        const buckets = [
          ["zero_shot", "static", "ZS · Static"],
          ["zero_shot", "live", "ZS · Live"],
          ["zero_shot", "temporal_override", "ZS · Override"],
          ["zero_shot", "unanswerable", "ZS · Unansw."],
          ["rag", "static", "RAG · Static"],
          ["rag", "live", "RAG · Live"],
          ["rag", "temporal_override", "RAG · Override"],
          ["rag", "unanswerable", "RAG · Unansw."],
        ];
        Plotly.newPlot("chart-appc-slice-deltas", [
          { x: buckets.map((b) => b[2]), y: buckets.map(([m, s]) => bs.base.by_mode_slice[m][s].correctness), name: "Qwen2.5-7B", type: "bar", marker: { color: t.zsColor } },
          { x: buckets.map((b) => b[2]), y: buckets.map(([m, s]) => bs.cand.by_mode_slice[m][s].correctness), name: "Qwen3.5-9B (MLX)", type: "bar", marker: { color: t.ragColor } },
        ], baseLayout({
          barmode: "group",
          margin: { t: 24, r: 16, b: 68, l: 48 },
          xaxis: { tickangle: -25 },
          yaxis: { title: "Mean Correctness (1–5)", range: [0, 5.6], gridcolor: t.gridColor },
          legend: { orientation: "h", y: 1.15 },
        }), PLOT_CONFIG);

        const labels = ["1", "2", "3", "4", "5"];
        const z = jc.inter_judge.confusion_matrix;
        Plotly.newPlot("chart-appc-interjudge-matrix", [{
          x: labels,
          y: labels,
          z: z,
          type: "heatmap",
          colorscale: resolvedDarkMode() ? "Cividis" : "Blues",
          text: z.map((row) => row.map((v) => String(v))),
          texttemplate: "%{text}",
          showscale: false,
        }], baseLayout({
          xaxis: { title: "Qwen3.5-9B Score (MLX)", type: "category" },
          yaxis: { title: "Qwen2.5-7B Score (Baseline)", type: "category" },
        }), PLOT_CONFIG);
      }
    }

    function updatePageDOM(targetPage) {
      currentPage = targetPage;
      document.querySelectorAll("section.book-page").forEach((sec) => {
        const p = Number(sec.dataset.page);
        sec.classList.toggle("hidden", p !== targetPage);
      });

      const ch = CHAPTERS[targetPage];
      document.getElementById("header-chapter-label").textContent = `${ch.label} · ${ch.title}`;
      document.getElementById("header-folio").textContent = `Page ${targetPage + 1} / ${CHAPTERS.length}`;
      document.getElementById("footer-folio").textContent = `Page ${targetPage + 1} of ${CHAPTERS.length}`;
      document.getElementById("reading-progress").style.width = `${((targetPage + 1) / CHAPTERS.length) * 100}%`;

      document.getElementById("btn-prev-page").disabled = targetPage === 0;
      document.getElementById("btn-next-page").disabled = targetPage === CHAPTERS.length - 1;

      document.querySelectorAll("#chapter-nav a").forEach((link) => {
        const active = Number(link.dataset.navLink) === targetPage;
        link.setAttribute("aria-current", active ? "page" : "false");
        link.className = active
          ? "block w-full text-left rounded-lg px-3 py-2 text-xs font-semibold bg-primary text-primary-foreground transition-colors"
          : "block w-full text-left rounded-lg px-3 py-2 text-xs font-medium text-muted-foreground hover:bg-accent hover:text-foreground transition-colors";
      });

      const spine = document.getElementById("book-spine");
      if (window.innerWidth < 1024 && !spine.classList.contains("hidden")) {
        spine.classList.add("hidden");
        document.getElementById("btn-spine-toggle").setAttribute("aria-expanded", "false");
      }
      window.scrollTo({ top: 0, behavior: "instant" });
    }

    function parseHashPage() {
      const m = /^#page-(\d+)$/.exec(window.location.hash || "");
      if (!m) return null;
      const idx = Number(m[1]) - 1;
      return idx >= 0 && idx < CHAPTERS.length ? idx : null;
    }

    function navigateToPage(targetPage, { pushHash = true } = {}) {
      if (targetPage < 0 || targetPage >= CHAPTERS.length || targetPage === currentPage) return;
      const direction = targetPage > currentPage ? "forward" : "backward";
      const nextHash = `#page-${targetPage + 1}`;
      if (pushHash && window.location.hash !== nextHash) {
        try {
          history.pushState({ page: targetPage }, "", nextHash);
        } catch {
          window.location.hash = nextHash;
        }
      }
      document.documentElement.dataset.navDir = direction;

      if (!document.startViewTransition) {
        updatePageDOM(targetPage);
        renderChartsForPage(targetPage);
        return;
      }

      let vt;
      try {
        vt = document.startViewTransition({
          update: () => updatePageDOM(targetPage),
          types: [direction],
        });
      } catch {
        try {
          vt = document.startViewTransition(() => updatePageDOM(targetPage));
        } catch {
          updatePageDOM(targetPage);
          renderChartsForPage(targetPage);
          return;
        }
      }
      vt.finished.finally(() => renderChartsForPage(targetPage));
    }

    // Accessible Tabs Controller (Page 5 & Page 6)
    function initTabs(rootId, onSelect) {
      const root = document.getElementById(rootId);
      const tabs = Array.from(root.querySelectorAll('[role="tab"]'));
      const panels = Array.from(root.querySelectorAll('[role="tabpanel"]'));

      function selectTab(tab) {
        for (const t of tabs) {
          t.setAttribute("aria-selected", t === tab ? "true" : "false");
        }
        if (panels.length > 1) {
          panels.forEach((panel) => {
            const match = panel.getAttribute("aria-labelledby") === tab.id;
            if (match) {
              panel.removeAttribute("hidden");
              panel.setAttribute("tabindex", "0");
            } else {
              panel.setAttribute("hidden", "until-found");
              panel.removeAttribute("tabindex");
            }
          });
        }
        if (onSelect) onSelect(tab);
      }

      tabs.forEach((tab, index) => {
        tab.addEventListener("click", () => selectTab(tab));
        tab.addEventListener("keydown", (e) => {
          if (e.key === "ArrowRight") {
            const next = tabs[(index + 1) % tabs.length];
            next.focus();
            selectTab(next);
          } else if (e.key === "ArrowLeft") {
            const prev = tabs[(index - 1 + tabs.length) % tabs.length];
            prev.focus();
            selectTab(prev);
          }
        });
      });

      panels.forEach((panel) => {
        panel.addEventListener("beforematch", () => {
          const owner = tabs.find((t) => t.id === panel.getAttribute("aria-labelledby"));
          if (owner) selectTab(owner);
        });
      });
    }

    // Populate Dynamic Content (links in #chapter-nav & #p1-agenda-grid are static HTML <a href="#page-N">)
    function populateContent() {
      // Page 1 KPIs
      const jc = DATA.judge_comparison;
      const mc = DATA.model_comparison;
      const sc = mc.score_comparison;
      const candPooled = jc.human_calibration.cand.pooled;
      const basePooled = jc.human_calibration.base.pooled;
      const kpis = [
        { k: "Date-Split Corpus", v: "500 Abstracts", s: "300 static (<T) · 200 live (>=T) · 0 overlap" },
        { k: "Golden Benchmark", v: "60 Prompts", s: "20 static · 20 live · 10 override · 10 unansw." },
        {
          k: "QLoRA vs. Base (Q3.5)",
          v: `${sc.ft.overall.pooled_all.correctness.toFixed(2)} vs. ${sc.base.overall.pooled_all.correctness.toFixed(2)}`,
          s: `RAG Override ${sc.ft.by_mode_slice.rag.temporal_override.correctness.toFixed(2)} vs. ${sc.base.by_mode_slice.rag.temporal_override.correctness.toFixed(2)} · 2.5× faster`,
        },
        { k: "Judge Calibration", v: `${candPooled.exact_match_pct}% Exact`, s: `+${jc.human_calibration.delta.pooled_exact_match_pct}pp vs Qwen2.5-7B (${basePooled.exact_match_pct}%) · QWK ${candPooled.kappa_w}` },
      ];
      document.getElementById("p1-kpis").innerHTML = kpis
        .map(
          (item) => `<div class="rounded-xl border border-border bg-card p-4 shadow-sm"><p class="font-mono text-[11px] text-muted-foreground">${item.k}</p><p class="font-serif text-2xl font-bold mt-1">${item.v}</p><p class="text-xs text-muted-foreground mt-1">${item.s}</p></div>`
        )
        .join("");

      // Page 12 KPIs (Appendix C: Stronger Judge)
      const op = jc.operational;
      const baseOp = op.base;
      const candOp = op.cand;
      const dOp = op.delta;
      const appcKpis = [
        { k: "Exact Score Match vs. Human", v: `${candPooled.exact_match_pct}%`, s: `54/60 ratings exact vs ${basePooled.exact_match_pct}% (48/60) · +${jc.human_calibration.delta.pooled_exact_match_pct} pp` },
        { k: "Severity-Weighted (QWK)", v: `${candPooled.kappa_w} Strong`, s: `vs ${basePooled.kappa_w} Moderate (0=luck, 1=human) · MAE ${candPooled.mae} vs ${basePooled.mae}` },
        { k: "Median Judge Speed & Time", v: `${candOp.tok_per_s_median} tok/s`, s: `${(candOp.latency_median_ms / 1000).toFixed(2)}s vs ${(baseOp.latency_median_ms / 1000).toFixed(2)}s (${dOp.latency_ratio_median}× faster)` },
        { k: "Median Verdict Length", v: `${candOp.eval_tokens_median} tok`, s: `vs ${baseOp.eval_tokens_median} tok (${dOp.eval_tokens_median > 0 ? "+" : ""}${dOp.eval_tokens_median} tokens/item)` },
      ];
      document.getElementById("appc-kpis").innerHTML = appcKpis
        .map(
          (item) => `<div class="rounded-xl border border-border bg-card p-4 shadow-sm"><p class="font-mono text-[11px] text-muted-foreground">${item.k}</p><p class="font-serif text-2xl font-bold mt-1">${item.v}</p><p class="text-xs text-muted-foreground mt-1">${item.s}</p></div>`
        )
        .join("");

      // Page 10 Hardware & Model Cards
      const hw = DATA.env.host;
      const hwHtml = [
        `<div class="rounded-xl border border-border bg-secondary/50 p-4 space-y-1"><p class="font-mono text-xs text-muted-foreground">Compute Host</p><p class="font-semibold">${hw.model_name} (${hw.chip})</p><p class="text-xs text-muted-foreground">${hw.total_number_of_cores} · ${hw.memory} Unified · macOS ${DATA.env.os}</p><p class="font-mono text-[11px] text-muted-foreground">${DATA.env.uv.split(" ")[0]} + ${DATA.env.python}</p></div>`,
        ...DATA.env.models.map(
          (m) => {
            const role = m.name.includes("qlora") ? "QLoRA Fine-Tuned Model" : (m.name.includes("llama") ? "Base Model" : "Cross-Family Judge");
            return `<div class="rounded-xl border border-border bg-secondary/50 p-4 space-y-1"><p class="font-mono text-xs text-muted-foreground">${role}</p><p class="font-semibold font-mono text-sm">${escapeHtml(m.name)} (${escapeHtml(m.params)})</p><p class="text-xs text-muted-foreground">${escapeHtml(m.quant)} · ${m.size_gb} GB · ctx=${m.ctx ? m.ctx.toLocaleString() : "default"} · 100% Metal GPU</p><p class="font-mono text-[11px] text-muted-foreground">sha256:${escapeHtml(m.digest)}...</p></div>`;
          }
        ),
      ];
      document.getElementById("app-hw-cards").innerHTML = hwHtml.join("");

      // Page 3 Split Metadata Cards & Corpus Sample Table
      const sp = DATA.split;
      document.getElementById("p3-split-cards").innerHTML = [
        { k: "Base Cutoff", v: sp.base_model_training_cutoff, s: `Released ${sp.base_model_release_date}` },
        { k: "Split Cutoff T", v: sp.cutoff_T, s: "Strict pre-T vs post-T boundary" },
        { k: "Static Corpus", v: `${sp.static_count} cs.CL`, s: `${sp.static_date_range[0]} .. ${sp.static_date_range[1]}` },
        { k: "Live Corpus", v: `${sp.live_count} cs.CL`, s: `${sp.live_date_range[0]} .. ${sp.live_date_range[1]} · Overlap=${sp.id_overlap_count}` },
      ]
        .map((c) => `<div class="rounded-xl border border-border bg-secondary/50 p-4"><p class="font-mono text-xs text-muted-foreground">${c.k}</p><p class="font-serif text-xl font-bold mt-0.5">${c.v}</p><p class="text-[11px] text-muted-foreground mt-0.5">${c.s}</p></div>`)
        .join("");

      const allSamples = [
        ...DATA.corpus.static.samples.map((s) => ({ ...s, split: "static" })),
        ...DATA.corpus.live.samples.map((s) => ({ ...s, split: "live" })),
      ];
      function renderCorpusTable() {
        const flt = document.getElementById("corpus-filter-split").value;
        const q = document.getElementById("corpus-search").value.trim().toLowerCase();
        const rows = allSamples.filter((r) => (flt === "all" || r.split === flt) && (!q || r.title.toLowerCase().includes(q) || r.id.toLowerCase().includes(q)));
        document.getElementById("corpus-sample-tbody").innerHTML = rows
          .map(
            (r) => `<tr>
              <th class="font-mono text-xs"><a class="underline hover:opacity-80" target="_blank" rel="noopener" href="https://arxiv.org/abs/${escapeHtml(r.id)}">${escapeHtml(r.id)}</a></th>
              <td data-label="Split"><span class="inline-flex rounded px-2 py-0.5 font-mono text-[11px] bg-muted">${r.split}</span></td>
              <td data-label="Published" class="font-mono text-xs">${escapeHtml(r.date)}</td>
              <td data-label="Words" class="font-mono text-xs">${r.words}</td>
              <td data-label="Title & Preview"><div class="font-medium text-xs">${escapeHtml(r.title)}</div><div class="text-xs text-muted-foreground mt-1">${escapeHtml(r.abstract_preview)}</div></td>
            </tr>`
          )
          .join("");
      }
      document.getElementById("corpus-filter-split").addEventListener("change", renderCorpusTable);
      document.getElementById("corpus-search").addEventListener("input", renderCorpusTable);
      renderCorpusTable();

      // Page 5 Golden Explorer
      let activeGoldenSlice = "all";
      function renderGoldenTable() {
        const q = document.getElementById("golden-search").value.trim().toLowerCase();
        const rows = DATA.runs.filter(
          (r) =>
            (activeGoldenSlice === "all" || r.slice === activeGoldenSlice) &&
            (!q || r.id.toLowerCase().includes(q) || r.prompt.toLowerCase().includes(q) || r.reference_answer.toLowerCase().includes(q) || r.source_paper_ids.join(" ").toLowerCase().includes(q))
        );
        document.getElementById("golden-tbody").innerHTML = rows
          .map(
            (r) => `<tr>
              <th class="font-mono text-xs">${escapeHtml(r.id)}</th>
              <td data-label="Slice"><span class="inline-flex rounded px-2 py-0.5 font-mono text-[11px] bg-muted">${escapeHtml(r.slice)}</span></td>
              <td data-label="Prompt" class="text-xs">${escapeHtml(r.prompt)}</td>
              <td data-label="Reference" class="text-xs text-muted-foreground">${escapeHtml(r.reference_answer)}</td>
              <td data-label="Sources" class="font-mono text-xs">${r.source_paper_ids.length ? r.source_paper_ids.map((id) => `<a class="underline" target="_blank" rel="noopener" href="https://arxiv.org/abs/${escapeHtml(id)}">${escapeHtml(id)}</a>`).join(", ") : "—"}</td>
            </tr>`
          )
          .join("");
      }
      initTabs("golden-tabs-root", (tab) => {
        activeGoldenSlice = tab.dataset.slice;
        renderGoldenTable();
      });
      document.getElementById("golden-search").addEventListener("input", renderGoldenTable);
      renderGoldenTable();

      // Page 6 Tabs & Audit Table
      initTabs("p6-tabs-root");
      document.getElementById("audit-tbody").innerHTML = DATA.calibration.audit_items
        .map((a) => {
          const hStr = `${a.human.correctness}/${a.human.faithfulness}/${a.human.citation_accuracy}/${a.human.override_obedience ?? "—"}`;
          const jStr = `${a.judge.correctness}/${a.judge.faithfulness}/${a.judge.citation_accuracy}/${a.judge.override_obedience ?? "—"}`;
          const j35 = a.judge_qwen35;
          const j35Str = `${j35.correctness}/${j35.faithfulness}/${j35.citation_accuracy}/${j35.override_obedience ?? "—"}`;
          const exact25 = a.human.correctness === a.judge.correctness && a.human.faithfulness === a.judge.faithfulness && a.human.citation_accuracy === a.judge.citation_accuracy;
          const exact35 = a.human.correctness === j35.correctness && a.human.faithfulness === j35.faithfulness && a.human.citation_accuracy === j35.citation_accuracy;
          const outcomeBadge = a.outcome === "improved"
            ? '<span class="ml-1 text-[10px] rounded bg-muted px-1 font-semibold">IMPROVED</span>'
            : a.outcome === "regressed"
              ? '<span class="ml-1 text-[10px] rounded bg-muted px-1">REGRESSED</span>'
              : "";
          return `<tr>
            <th class="font-mono text-xs">${escapeHtml(a.sample_id)}<div class="text-[11px] text-muted-foreground font-normal">${escapeHtml(a.id)}</div></th>
            <td data-label="Mode / Slice" class="font-mono text-xs">${escapeHtml(a.mode)}<br><span class="text-muted-foreground">${escapeHtml(a.slice)}</span></td>
            <td data-label="Human" class="font-mono text-xs font-semibold">${hStr}</td>
            <td data-label="Qwen2.5-7B" class="font-mono text-xs">${jStr} ${exact25 ? '<span class="ml-1 text-[10px] rounded bg-muted px-1">EXACT</span>' : ""}</td>
            <td data-label="Qwen3.5-9B" class="font-mono text-xs font-semibold">${j35Str} ${exact35 ? '<span class="ml-1 text-[10px] rounded bg-muted px-1">EXACT</span>' : ""}${outcomeBadge}</td>
            <td data-label="Notes" class="text-xs space-y-1"><div><strong>Human:</strong> ${escapeHtml(a.human.notes)}</div><div class="text-muted-foreground"><strong>Q2.5:</strong> ${escapeHtml(a.judge.rationale)}</div><div class="text-muted-foreground"><strong>Q3.5:</strong> ${escapeHtml(j35.rationale)}</div></td>
          </tr>`;
        })
        .join("");

      // Page 11 Tabs: Speed Table & Discordant Items Explorer
      initTabs("appc-tabs-root");
      const speedRows = [
        ["Median Throughput (tok/s)", `${baseOp.tok_per_s_median} tok/s`, `${candOp.tok_per_s_median} tok/s`, `+${dOp.tok_per_s_median} tok/s (faster)`],
        ["Mean Throughput (tok/s)", `${baseOp.tok_per_s_mean} tok/s`, `${candOp.tok_per_s_mean} tok/s`, `+${dOp.tok_per_s_mean} tok/s (faster)`],
        ["Median Time / Item", `${(baseOp.latency_median_ms / 1000).toFixed(2)} s (${baseOp.latency_median_ms} ms)`, `${(candOp.latency_median_ms / 1000).toFixed(2)} s (${candOp.latency_median_ms} ms)`, `${(dOp.latency_median_ms / 1000).toFixed(2)} s (${dOp.latency_ratio_median}× faster)`],
        ["Mean Time / Item", `${(baseOp.latency_mean_ms / 1000).toFixed(2)} s`, `${(candOp.latency_mean_ms / 1000).toFixed(2)} s`, `${(dOp.latency_mean_ms / 1000).toFixed(2)} s`],
        ["Total 120-Item Wall Time", `${baseOp.total_time_s} s`, `${candOp.total_time_s} s`, `${(candOp.total_time_s - baseOp.total_time_s).toFixed(2)} s saved`],
        ["Median Output Tokens / Item", `${baseOp.eval_tokens_median} tokens`, `${candOp.eval_tokens_median} tokens`, `+${dOp.eval_tokens_median} tokens (1.90× richer)`],
        ["Mean Output Tokens / Item", `${baseOp.eval_tokens_mean} tokens`, `${candOp.eval_tokens_mean} tokens`, `+${dOp.eval_tokens_mean} tokens`],
        ["Total Output Tokens (120 Items)", `${baseOp.total_eval_tokens.toLocaleString()}`, `${candOp.total_eval_tokens.toLocaleString()}`, `+${(candOp.total_eval_tokens - baseOp.total_eval_tokens).toLocaleString()} tokens`],
      ];
      document.getElementById("speed-tokens-tbody").innerHTML = speedRows
        .map(
          ([metric, bVal, cVal, diffVal]) => `<tr>
            <th class="font-mono text-xs">${escapeHtml(metric)}</th>
            <td data-label="Qwen2.5-7B" class="font-mono text-xs">${escapeHtml(bVal)}</td>
            <td data-label="Qwen3.5-9B" class="font-mono text-xs font-semibold">${escapeHtml(cVal)}</td>
            <td data-label="Difference" class="font-mono text-xs">${escapeHtml(diffVal)}</td>
          </tr>`
        )
        .join("");

      function renderDiscordantTable() {
        const cat = document.getElementById("discordant-filter").value;
        const q = document.getElementById("discordant-search").value.trim().toLowerCase();
        const rows = DATA.judge_comparison.inter_judge.discordant_items.filter(
          (d) =>
            (cat === "all" || d.category === cat) &&
            (!q || d.id.toLowerCase().includes(q) || d.prompt.toLowerCase().includes(q) || d.cand_rationale.toLowerCase().includes(q) || d.base_rationale.toLowerCase().includes(q))
        );
        document.getElementById("discordant-tbody").innerHTML = rows
          .map(
            (d) => `<tr>
              <th class="font-mono text-xs">${escapeHtml(d.id)}<div class="text-[11px] text-muted-foreground font-normal">${escapeHtml(d.mode)} · ${escapeHtml(d.slice)}</div></th>
              <td data-label="Category" class="font-mono text-[11px]"><span class="rounded bg-muted px-1.5 py-0.5">${escapeHtml(d.category)}</span></td>
              <td data-label="Qwen2.5-7B" class="font-mono text-xs">${d.base_scores.correctness}/${d.base_scores.faithfulness}/${d.base_scores.citation_accuracy}</td>
              <td data-label="Qwen3.5-9B" class="font-mono text-xs font-semibold">${d.cand_scores.correctness}/${d.cand_scores.faithfulness}/${d.cand_scores.citation_accuracy}</td>
              <td data-label="Prompt & Rationales" class="text-xs space-y-1"><div class="font-medium">${escapeHtml(d.prompt)}</div><div class="text-muted-foreground"><strong>Q2.5:</strong> ${escapeHtml(d.base_rationale)}</div><div class="text-muted-foreground"><strong>Q3.5:</strong> ${escapeHtml(d.cand_rationale)}</div></td>
            </tr>`
          )
          .join("");
      }
      document.getElementById("discordant-filter").addEventListener("change", renderDiscordantTable);
      document.getElementById("discordant-search").addEventListener("input", renderDiscordantTable);
      renderDiscordantTable();

      // Page 7 Run Inspector
      const sel = document.getElementById("inspector-select");
      sel.innerHTML = DATA.runs.map((r, i) => `<option value="${i}">${r.id} [${r.slice}] ${r.prompt.slice(0, 62)}...</option>`).join("");
      function renderInspector(index) {
        const r = DATA.runs[index];
        sel.value = String(index);
        document.getElementById("inspector-prev").disabled = index === 0;
        document.getElementById("inspector-next").disabled = index === DATA.runs.length - 1;

        function card(title, run, isRag) {
          const ovBadge = run.override_obedience === null ? "" : `<span class="rounded bg-muted px-2 py-0.5 font-mono text-[11px]">Ov2.5: ${run.override_obedience}</span>`;
          const q35 = run.qwen35;
          const q35Ov = q35.override_obedience === null ? "" : `<span class="rounded bg-muted px-2 py-0.5 font-mono text-[11px]">Ov3.5: ${q35.override_obedience}</span>`;
          const timing = isRag
            ? `Total ${run.latency_ms} ms (ret ${run.retrieval_ms} ms + gen ${run.generation_ms} ms)`
            : `Total ${run.latency_ms} ms`;
          return `<div class="rounded-xl border border-border bg-card p-4 space-y-3">
            <div class="flex flex-wrap items-center justify-between gap-2 border-b border-border pb-2">
              <span class="font-serif text-base font-semibold">${title}</span>
              <span class="font-mono text-[11px] text-muted-foreground">${timing}</span>
            </div>
            <div class="flex flex-wrap gap-1.5">
              <span class="rounded bg-muted px-2 py-0.5 font-mono text-[11px]">Q2.5 C/F/Cit: ${run.correctness}/${run.faithfulness}/${run.citation_accuracy}</span>
              ${ovBadge}
              <span class="rounded bg-secondary border border-border px-2 py-0.5 font-mono text-[11px] font-semibold">Q3.5 C/F/Cit: ${q35.correctness}/${q35.faithfulness}/${q35.citation_accuracy}</span>
              ${q35Ov}
            </div>
            <div class="text-xs leading-relaxed whitespace-pre-wrap">${escapeHtml(run.answer)}</div>
            <div class="text-[11px] text-muted-foreground border-t border-border pt-2 space-y-1">
              <div><strong>Cited IDs:</strong> <span class="font-mono">${run.cited.length ? escapeHtml(run.cited.join(", ")) : "none"}</span></div>
              <div><strong>Qwen2.5-7B:</strong> ${escapeHtml(run.rationale)}</div>
              <div><strong>Qwen3.5-9B (MLX):</strong> ${escapeHtml(q35.rationale)}</div>
              ${isRag ? `<details class="mt-2"><summary class="cursor-pointer font-medium text-foreground">Top-4 Retrieved Chunks (${escapeHtml((run.valid_context_ids || []).join(", "))})</summary><pre class="mt-2 p-2.5 rounded bg-secondary border border-border font-mono text-[10px] whitespace-pre-wrap overflow-x-auto">${escapeHtml(run.context_preview)}</pre></details>` : ""}
            </div>
          </div>`;
        }

        document.getElementById("inspector-body").innerHTML = `
          <div class="rounded-xl border border-border bg-card p-4 space-y-2">
            <div class="flex flex-wrap items-center gap-2">
              <span class="font-mono text-xs font-semibold">${escapeHtml(r.id)}</span>
              <span class="rounded-full bg-muted px-2.5 py-0.5 font-mono text-xs">${escapeHtml(r.slice)}</span>
              <span class="font-mono text-xs text-muted-foreground">Sources: ${r.source_paper_ids.length ? escapeHtml(r.source_paper_ids.join(", ")) : "none"}</span>
            </div>
            <p class="font-medium text-sm">${escapeHtml(r.prompt)}</p>
            <p class="text-xs text-muted-foreground"><strong>Reference Answer:</strong> ${escapeHtml(r.reference_answer)}</p>
          </div>
          <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
            ${card("Base Zero-Shot (llama3.2:3b)", r.zero_shot, false)}
            ${card("QLoRA Zero-Shot (llama3.2-qlora:3b)", r.zero_shot_ft, false)}
            ${card("Base RAG k=4 (llama3.2:3b + bge-small)", r.rag, true)}
            ${card("QLoRA RAG k=4 (llama3.2-qlora:3b + bge-small)", r.rag_ft, true)}
          </div>`;
      }
      sel.addEventListener("change", () => renderInspector(Number(sel.value)));
      document.getElementById("inspector-prev").addEventListener("click", () => renderInspector(Math.max(0, Number(sel.value) - 1)));
      document.getElementById("inspector-next").addEventListener("click", () => renderInspector(Math.min(DATA.runs.length - 1, Number(sel.value) + 1)));
      renderInspector(40); // Default to temporal_override g41 SOLAR

      // Page 9 (Chapter 8): Master Table (Base vs. QLoRA)
      const bms = DATA.results.by_mode_slice;
      const ov = DATA.results.overall;
      const bms35 = DATA.judge_comparison.benchmark_shift.cand.by_mode_slice;
      const ov35 = DATA.judge_comparison.benchmark_shift.cand.overall;
      const ft25 = mc.cross_judge_check.ft;
      const ft35 = sc.ft;
      const rows = [];
      for (const mode of ["zero_shot", "rag"]) {
        const modeTitle = mode === "zero_shot" ? "Zero-Shot" : "RAG (k=4)";
        for (const sl of ["static", "live", "temporal_override", "unanswerable"]) {
          rows.push({ mode: `Base · ${modeTitle}`, slice: sl, ...bms[mode][sl], corr35: bms35[mode][sl].correctness, bold: false });
        }
        rows.push({ mode: `Base · ${modeTitle}`, slice: "ALL (mean)", ...ov[mode], corr35: ov35[mode].correctness, bold: true });
        for (const sl of ["static", "live", "temporal_override", "unanswerable"]) {
          rows.push({
            mode: `QLoRA · ${modeTitle}`,
            slice: sl,
            ...ft35.by_mode_slice[mode][sl],
            correctness: ft25.by_mode_slice[mode][sl].correctness,
            corr35: ft35.by_mode_slice[mode][sl].correctness,
            bold: false,
          });
        }
        rows.push({
          mode: `QLoRA · ${modeTitle}`,
          slice: "ALL (mean)",
          ...ft35.overall[mode],
          correctness: ft25.overall[mode].correctness,
          corr35: ft35.overall[mode].correctness,
          bold: true,
        });
      }
      document.getElementById("master-tbody").innerHTML = rows
        .map(
          (r) => `<tr class="${r.bold ? "bg-muted/60 font-semibold" : ""}">
            <th>${escapeHtml(r.mode)}</th>
            <td data-label="Slice" class="font-mono text-xs">${escapeHtml(r.slice)}</td>
            <td data-label="N" class="font-mono text-xs">${r.count}</td>
            <td data-label="Corr (Qwen2.5)" class="font-mono text-xs">${r.correctness.toFixed(2)}</td>
            <td data-label="Corr (Qwen3.5)" class="font-mono text-xs font-semibold">${r.corr35.toFixed(2)}</td>
            <td data-label="Faithfulness" class="font-mono text-xs">${r.faithfulness.toFixed(2)}</td>
            <td data-label="Citation Acc" class="font-mono text-xs">${r.citation_accuracy.toFixed(2)}</td>
            <td data-label="Override Obed" class="font-mono text-xs">${r.override_obedience === null ? "—" : r.override_obedience.toFixed(2)}</td>
            <td data-label="Mean Latency (ms)" class="font-mono text-xs">${r.latency_mean_ms.toFixed(1)}</td>
          </tr>`
        )
        .join("");

      // Page 8 (Chapter 7): QLoRA Fine-Tuning Experiment
      const qMan = DATA.qlora_training.manifest;
      const qDs = DATA.qlora_dataset;
      const pooledCi = mc.bootstrap_cis.pooled_all;
      const p9Kpis = [
        {
          k: "Pooled Correctness (N=120)",
          v: `${pooledCi.ft_mean.toFixed(3)} vs. ${pooledCi.base_mean.toFixed(3)}`,
          s: `Delta +${pooledCi.delta_mean.toFixed(3)} · 95% CI [${pooledCi.ci95[0]}, ${pooledCi.ci95[1]}] · ${mc.win_loss.overall.improved}W / ${mc.win_loss.overall.regressed}L`,
        },
        {
          k: "RAG Temporal Override",
          v: `${sc.ft.by_mode_slice.rag.temporal_override.correctness.toFixed(2)} vs. ${sc.base.by_mode_slice.rag.temporal_override.correctness.toFixed(2)}`,
          s: `Override Obed ${sc.ft.by_mode_slice.rag.temporal_override.override_obedience.toFixed(2)} vs. ${sc.base.by_mode_slice.rag.temporal_override.override_obedience.toFixed(2)} · 10/10 cited · 0 abstentions`,
        },
        {
          k: "Mean Inference Latency",
          v: `${sc.ft.overall.pooled_all.latency_mean_ms.toFixed(1)} ms`,
          s: `${sc.overall_delta.pooled_all.latency_mean_ms.toFixed(1)} ms vs. ${sc.base.overall.pooled_all.latency_mean_ms.toFixed(1)} ms (${(sc.base.overall.pooled_all.latency_mean_ms / sc.ft.overall.pooled_all.latency_mean_ms).toFixed(2)}× faster)`,
        },
        {
          k: "MLX QLoRA Checkpoint",
          v: `Iter ${qMan.selected_checkpoint_iter}`,
          s: `val_loss ${qMan.selected_checkpoint_val_loss} · ${qMan.trainable_params.trainable_millions}M params (${qMan.trainable_params.percent}%) · ${(qMan.wall_time_s / 60).toFixed(1)}m`,
        },
      ];
      document.getElementById("p9-qlora-kpis").innerHTML = p9Kpis
        .map(
          (item) => `<div class="rounded-xl border border-border bg-card p-4 shadow-sm"><p class="font-mono text-[11px] text-muted-foreground">${escapeHtml(item.k)}</p><p class="font-serif text-2xl font-bold mt-1">${escapeHtml(item.v)}</p><p class="text-xs text-muted-foreground mt-1">${escapeHtml(item.s)}</p></div>`
        )
        .join("");

      initTabs("p9-tabs-root");

      // Tab 1: Bootstrap CIs & Dual-Judge Cross-Check
      const ciEntries = Object.entries(mc.bootstrap_cis);
      document.getElementById("p9-ci-tbody").innerHTML = ciEntries
        .map(([bucket, c]) => {
          const sign = c.delta_mean > 0 ? "+" : "";
          const ciStr = `[${c.ci95[0] >= 0 ? "+" : ""}${c.ci95[0]}, ${c.ci95[1] >= 0 ? "+" : ""}${c.ci95[1]}]`;
          return `<tr>
            <th class="font-mono text-xs">${escapeHtml(bucket)}</th>
            <td data-label="N" class="font-mono text-xs">${c.n}</td>
            <td data-label="Base Mean" class="font-mono text-xs">${c.base_mean.toFixed(3)}</td>
            <td data-label="QLoRA Mean" class="font-mono text-xs font-semibold">${c.ft_mean.toFixed(3)}</td>
            <td data-label="Delta" class="font-mono text-xs font-semibold">${sign}${c.delta_mean.toFixed(3)}</td>
            <td data-label="95% CI" class="font-mono text-xs">${escapeHtml(ciStr)}</td>
            <td data-label="Significant" class="font-mono text-xs">${c.statistically_significant ? '<span class="rounded bg-primary text-primary-foreground px-1.5 py-0.5">YES</span>' : "no"}</td>
          </tr>`;
        })
        .join("");

      const djEntries = Object.entries(mc.cross_judge_check.agreement_by_bucket);
      document.getElementById("p9-dualjudge-tbody").innerHTML = djEntries
        .map(([bucket, d]) => {
          const fmt = (v) => `${v > 0 ? "+" : ""}${v.toFixed(3)}`;
          const agrees = d.direction_agrees === null ? "—" : (d.direction_agrees ? "True" : "False");
          return `<tr>
            <th class="font-mono text-xs">${escapeHtml(bucket)}</th>
            <td data-label="Qwen3.5-9B Delta" class="font-mono text-xs font-semibold">${fmt(d.primary_delta)}</td>
            <td data-label="Qwen2.5-7B Delta" class="font-mono text-xs">${fmt(d.secondary_delta)}</td>
            <td data-label="Direction Agrees" class="font-mono text-xs">${agrees}</td>
          </tr>`;
        })
        .join("");

      // Tab 2: 120-Item Paired Diff Explorer
      function renderPairedDiffTable() {
        const outFlt = document.getElementById("p9-diff-outcome").value;
        const modeFlt = document.getElementById("p9-diff-mode").value;
        const q = document.getElementById("p9-diff-search").value.trim().toLowerCase();
        const items = mc.paired_items.filter((it) => {
          if (outFlt !== "all" && it.outcome !== outFlt) return false;
          if (modeFlt !== "all" && it.mode !== modeFlt) return false;
          if (!q) return true;
          return (
            it.id.toLowerCase().includes(q) ||
            it.slice.toLowerCase().includes(q) ||
            it.prompt.toLowerCase().includes(q) ||
            it.base_answer_preview.toLowerCase().includes(q) ||
            it.ft_answer_preview.toLowerCase().includes(q) ||
            it.base_rationale.toLowerCase().includes(q) ||
            it.ft_rationale.toLowerCase().includes(q)
          );
        });
        document.getElementById("p9-diff-tbody").innerHTML = items
          .map((it) => {
            const bStr = `${it.base_scores.correctness}/${it.base_scores.faithfulness}/${it.base_scores.citation_accuracy}`;
            const fStr = `${it.ft_scores.correctness}/${it.ft_scores.faithfulness}/${it.ft_scores.citation_accuracy}`;
            const dSign = it.correctness_delta > 0 ? "+" : "";
            const badgeCls = it.outcome === "improved"
              ? "bg-primary text-primary-foreground"
              : it.outcome === "regressed"
                ? "bg-muted text-foreground border border-border"
                : "bg-secondary text-muted-foreground";
            return `<tr>
              <th class="font-mono text-xs">${escapeHtml(it.id)}<div class="text-[11px] text-muted-foreground font-normal">${escapeHtml(it.mode)} · ${escapeHtml(it.slice)}</div></th>
              <td data-label="Base" class="font-mono text-xs">${bStr}<div class="text-[10px] text-muted-foreground">${it.base_latency_ms} ms</div></td>
              <td data-label="QLoRA" class="font-mono text-xs font-semibold">${fStr}<div class="text-[10px] text-muted-foreground font-normal">${it.ft_latency_ms} ms</div></td>
              <td data-label="Delta" class="font-mono text-xs"><span class="rounded px-1.5 py-0.5 ${badgeCls}">${dSign}${it.correctness_delta} ${escapeHtml(it.outcome)}</span></td>
              <td data-label="Details" class="text-xs space-y-1">
                <div class="font-medium">${escapeHtml(it.prompt)}</div>
                <div class="text-muted-foreground"><strong>Base Ans:</strong> ${escapeHtml(it.base_answer_preview)}</div>
                <div class="text-muted-foreground"><strong>QLoRA Ans:</strong> ${escapeHtml(it.ft_answer_preview)}</div>
                <div class="text-[11px] text-muted-foreground"><strong>Q3.5 Base:</strong> ${escapeHtml(it.base_rationale)}</div>
                <div class="text-[11px] text-muted-foreground"><strong>Q3.5 QLoRA:</strong> ${escapeHtml(it.ft_rationale)}</div>
              </td>
            </tr>`;
          })
          .join("");
      }
      document.getElementById("p9-diff-outcome").addEventListener("change", renderPairedDiffTable);
      document.getElementById("p9-diff-mode").addEventListener("change", renderPairedDiffTable);
      document.getElementById("p9-diff-search").addEventListener("input", renderPairedDiffTable);
      renderPairedDiffTable();

      // Tab 3: Recipe, Dataset & Hybrid Sweep
      const cfg = qMan.config;
      document.getElementById("p9-recipe-grid").innerHTML = [
        `<div class="rounded-xl border border-border bg-secondary/50 p-4 space-y-1"><p class="font-mono text-muted-foreground">Base &amp; LoRA Config</p><p class="font-semibold font-mono">${escapeHtml(cfg.model)}</p><p class="text-muted-foreground">rank=${cfg.lora_parameters.rank}, scale=${cfg.lora_parameters.scale}, dropout=${cfg.lora_parameters.dropout}, layers=${cfg.num_layers}, max_seq=${cfg.max_seq_length}, mask_prompt=${cfg.mask_prompt}</p></div>`,
        `<div class="rounded-xl border border-border bg-secondary/50 p-4 space-y-1"><p class="font-mono text-muted-foreground">Optimization &amp; Hardware</p><p class="font-semibold">${cfg.iters} iters (${cfg.lr_schedule.name}, warmup=${cfg.lr_schedule.warmup})</p><p class="text-muted-foreground">lr=${cfg.learning_rate} -> 1e-5, batch=${cfg.batch_size}, grad_ckpt=${cfg.grad_checkpoint}, wall=${qMan.wall_time_s}s, peak=${qMan.peak_mem_gb_observed_sample} GB</p></div>`,
        `<div class="rounded-xl border border-border bg-secondary/50 p-4 space-y-1"><p class="font-mono text-muted-foreground">Artifact &amp; Leakage Gate</p><p class="font-semibold font-mono">${escapeHtml(qMan.ollama_model)} (${escapeHtml(qMan.ollama_quantization)})</p><p class="text-muted-foreground">ckpt=${escapeHtml(qMan.selected_checkpoint)} · ${qMan.adapter_size_mb} MB · static=${qDs.static_date_min}..${qDs.static_date_max} (< ${qDs.cutoff_date}) · 0 live/golden leaks</p></div>`,
      ].join("");

      const catLabels = {
        cat_a_zero_shot_qa: "Cat A · Zero-Shot Pre-T QA",
        cat_b_zero_shot_abstention: "Cat B · Zero-Shot Abstention",
        cat_c_rag_synthesis: "Cat C · RAG Cited Synthesis",
        cat_d_rag_abstention: "Cat D · RAG Grounded Abstention",
      };
      const catRows = Object.entries(qDs.categories).map(([k, v]) => [catLabels[k] || k, v.train, v.valid, v.train + v.valid]);
      catRows.push(["TOTAL (Leakage-Safe Pre-T)", qDs.train_count, qDs.valid_count, qDs.total_count]);
      document.getElementById("p9-dataset-tbody").innerHTML = catRows
        .map(([label, tr, va, tot], idx) => `<tr class="${idx === catRows.length - 1 ? "bg-muted/60 font-semibold" : ""}">
          <th class="text-xs">${escapeHtml(label)}</th>
          <td data-label="Train" class="font-mono text-xs">${tr}</td>
          <td data-label="Valid" class="font-mono text-xs">${va}</td>
          <td data-label="Total" class="font-mono text-xs">${tot}</td>
        </tr>`)
        .join("");

      const taus = mc.hybrid_routing.taus;
      document.getElementById("p9-tau-tbody").innerHTML = taus
        .map((tv) => {
          const key = Number(tv).toFixed(2);
          const row = mc.hybrid_routing.by_tau[key];
          const isDef = Number(tv) === Number(mc.hybrid_routing.default_tau);
          const dVal = row.delta.overall.correctness;
          return `<tr class="${isDef ? "bg-muted/60 font-semibold" : ""}">
            <th class="font-mono text-xs">${escapeHtml(key)}${isDef ? " (default)" : ""}</th>
            <td data-label="RAG Routed %" class="font-mono text-xs">${row.base.overall.rag_routed_pct}%</td>
            <td data-label="Base Hybrid" class="font-mono text-xs">${row.base.overall.correctness.toFixed(3)}</td>
            <td data-label="QLoRA Hybrid" class="font-mono text-xs">${row.ft.overall.correctness.toFixed(3)}</td>
            <td data-label="Delta" class="font-mono text-xs">${dVal >= 0 ? "+" : ""}${dVal.toFixed(3)}</td>
          </tr>`;
        })
        .join("");
    }

    // Theme Controller (tri-state System / Light / Dark synced via <meta name="color-scheme">)
    function syncThemeUI() {
      let stored = null;
      try { stored = localStorage.getItem("color-scheme"); } catch {}
      const active = stored === "light" || stored === "dark" ? stored : "system";
      const metaVal = active === "system" ? "light dark" : active;
      document.querySelector('meta[name="color-scheme"]').content = metaVal;

      document.querySelectorAll(".theme-btn").forEach((btn) => {
        const isSel = btn.dataset.themeChoice === active;
        btn.classList.toggle("bg-primary", isSel);
        btn.classList.toggle("text-primary-foreground", isSel);
      });
      const sysResolved = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
      document.getElementById("theme-status-note").textContent =
        active === "system" ? `System (Currently ${sysResolved})` : `Pinned to ${active} mode`;
    }

    document.querySelectorAll(".theme-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const choice = btn.dataset.themeChoice;
        try {
          if (choice === "system") localStorage.removeItem("color-scheme");
          else localStorage.setItem("color-scheme", choice);
        } catch {}
        syncThemeUI();
        renderedPages.clear();
        renderChartsForPage(currentPage, true);
      });
    });

    window.addEventListener("storage", (e) => {
      if (e.key === "color-scheme") {
        syncThemeUI();
        renderedPages.clear();
        renderChartsForPage(currentPage, true);
      }
    });

    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
      syncThemeUI();
      renderedPages.clear();
      renderChartsForPage(currentPage, true);
    });

    // Global Book Controls & Hash Link Navigation
    document.addEventListener("click", (e) => {
      const link = e.target.closest("a[data-nav-link]");
      if (!link) return;
      if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button !== 0) return;
      e.preventDefault();
      navigateToPage(Number(link.dataset.navLink));
    });
    window.addEventListener("hashchange", () => {
      const idx = parseHashPage();
      if (idx !== null) navigateToPage(idx, { pushHash: false });
    });
    window.addEventListener("popstate", () => {
      const idx = parseHashPage() ?? 0;
      navigateToPage(idx, { pushHash: false });
    });
    document.getElementById("btn-prev-page").addEventListener("click", () => navigateToPage(currentPage - 1));
    document.getElementById("btn-next-page").addEventListener("click", () => navigateToPage(currentPage + 1));
    document.getElementById("btn-spine-toggle").addEventListener("click", () => {
      const spine = document.getElementById("book-spine");
      const open = spine.classList.toggle("hidden");
      document.getElementById("btn-spine-toggle").setAttribute("aria-expanded", String(!open));
    });
    document.addEventListener("keydown", (e) => {
      if (["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(document.activeElement?.tagName)) return;
      if (e.key === "ArrowRight") navigateToPage(currentPage + 1);
      if (e.key === "ArrowLeft") navigateToPage(currentPage - 1);
    });

    // Boot
    syncThemeUI();
    populateContent();
    const initialPage = parseHashPage() ?? 0;
    updatePageDOM(initialPage);
    renderChartsForPage(initialPage);
  </script>
</body>
</html>
"""


def build_book(
    root: str | Path = ".",
    out_path: str | Path = "experiment_book.html",
) -> Path:
    """Assemble verified artifacts into `experiment_book.html`."""
    project_root = Path(root).resolve()
    payload = collect_book_payload(project_root)
    safe_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("__PAYLOAD_JSON__", safe_json)

    output = Path(out_path)
    if not output.is_absolute():
        output = project_root / output
    output.write_text(html, encoding="utf-8")
    return output


def main() -> None:
    """CLI entrypoint for building the interactive HTML experiment book."""
    parser = argparse.ArgumentParser(description="Build interactive Tailwind/shadcn/Plotly experiment book.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out-path", default="experiment_book.html")
    args = parser.parse_args()
    out = build_book(root=args.root, out_path=args.out_path)
    print(f"Wrote {out} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

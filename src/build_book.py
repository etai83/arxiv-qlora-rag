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
    human_rows = read_jsonl(root / "eval/human_scores_20.jsonl")
    results_summary = json.loads((root / "eval/results_summary.json").read_text(encoding="utf-8"))
    rubric_md = (root / "eval/rubric.md").read_text(encoding="utf-8")
    notes_md = (root / "notes.md").read_text(encoding="utf-8")

    judge_by_key = {(r["id"], r["mode"]): r for r in judge_rows}

    # Build 20-item audit table + 5x5 confusion matrix for Page 6.
    matrix = [[0 for _ in range(5)] for _ in range(5)]
    audit_items: list[dict[str, Any]] = []
    for h in human_rows:
        j = judge_by_key[(h["id"], h["mode"])]
        for dim in ("correctness", "faithfulness", "citation_accuracy"):
            matrix[int(h[dim]) - 1][int(j[dim]) - 1] += 1
        audit_items.append(
            {
                "sample_id": h["sample_id"],
                "id": h["id"],
                "mode": h["mode"],
                "slice": h["slice"],
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
            }
        )

    # Compact per-prompt paired runs (60 prompts x 2 modes) for Page 7 Run Inspector & latency charts.
    paired_runs: list[dict[str, Any]] = []
    for g in golden_rows:
        gid = g["id"]
        zs = judge_by_key[(gid, "zero_shot")]
        rg = judge_by_key[(gid, "rag")]
        ctx_preview = (rg.get("retrieved_context") or "")[:1400]
        if len(rg.get("retrieved_context") or "") > 1400:
            ctx_preview += "\n... [truncated]"
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
                    "params": m["details"]["parameter_size"],
                    "quant": m["details"]["quantization_level"],
                    "ctx": m["details"]["context_length"],
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
  <title>arXiv Hybrid LLM — Evening 1 Experiment Book</title>
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
        <span id="header-folio" class="font-mono text-xs text-muted-foreground hidden sm:inline">Page 1 / 9</span>
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
      <div id="reading-progress" class="h-full bg-primary transition-all duration-300" style="width: 11.1%"></div>
    </div>
  </header>

  <!-- Main Book Workspace -->
  <div class="max-w-7xl w-full mx-auto px-4 sm:px-6 py-6 flex-1 grid grid-cols-1 lg:grid-cols-[260px_minmax(0,1fr)] gap-6 items-start">
    <!-- Left Book Spine / Table of Contents -->
    <aside id="book-spine" class="hidden lg:block sticky top-20 rounded-xl border border-border bg-card p-4 shadow-sm">
      <div class="pb-3 mb-3 border-b border-border">
        <p class="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">Monograph Index</p>
        <h2 class="font-serif text-lg font-semibold mt-0.5">Evening 1 Lab Book</h2>
      </div>
      <nav aria-label="Book chapters" id="chapter-nav" class="space-y-1">
        <a href="#page-1" data-nav-link="0" aria-current="page" class="block w-full text-left rounded-lg px-3 py-2 text-xs font-semibold bg-primary text-primary-foreground transition-colors"><div class="font-mono text-[10px] opacity-75">Prologue</div><div class="truncate">Abstract &amp; Agenda</div></a>
        <a href="#page-2" data-nav-link="1" aria-current="false" class="block w-full text-left rounded-lg px-3 py-2 text-xs font-medium text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"><div class="font-mono text-[10px] opacity-75">Chapter 1</div><div class="truncate">Slices &amp; Eval Metrics</div></a>
        <a href="#page-3" data-nav-link="2" aria-current="false" class="block w-full text-left rounded-lg px-3 py-2 text-xs font-medium text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"><div class="font-mono text-[10px] opacity-75">Chapter 2</div><div class="truncate">Leakage-Safe Date Split</div></a>
        <a href="#page-4" data-nav-link="3" aria-current="false" class="block w-full text-left rounded-lg px-3 py-2 text-xs font-medium text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"><div class="font-mono text-[10px] opacity-75">Chapter 3</div><div class="truncate">Ingestion &amp; FAISS Retrieval</div></a>
        <a href="#page-5" data-nav-link="4" aria-current="false" class="block w-full text-left rounded-lg px-3 py-2 text-xs font-medium text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"><div class="font-mono text-[10px] opacity-75">Chapter 4</div><div class="truncate">60-Prompt Golden Set</div></a>
        <a href="#page-6" data-nav-link="5" aria-current="false" class="block w-full text-left rounded-lg px-3 py-2 text-xs font-medium text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"><div class="font-mono text-[10px] opacity-75">Chapter 5</div><div class="truncate">Rubric &amp; Judge Calibration</div></a>
        <a href="#page-7" data-nav-link="6" aria-current="false" class="block w-full text-left rounded-lg px-3 py-2 text-xs font-medium text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"><div class="font-mono text-[10px] opacity-75">Chapter 6</div><div class="truncate">Baselines &amp; Run Inspector</div></a>
        <a href="#page-8" data-nav-link="7" aria-current="false" class="block w-full text-left rounded-lg px-3 py-2 text-xs font-medium text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"><div class="font-mono text-[10px] opacity-75">Chapter 7</div><div class="truncate">Failure Analysis &amp; QLoRA</div></a>
        <div class="pt-2 mt-2 border-t border-border"></div>
        <a href="#page-9" data-nav-link="8" aria-current="false" class="block w-full text-left rounded-lg px-3 py-2 text-xs font-medium text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"><div class="font-mono text-[10px] opacity-75">Appendix A</div><div class="truncate">Compute &amp; Orchestration</div></a>
      </nav>
      <div class="mt-4 pt-3 border-t border-border text-xs text-muted-foreground space-y-1">
        <div class="flex justify-between"><span>Cutoff T</span><span class="font-mono text-foreground">2025-01-01</span></div>
        <div class="flex justify-between"><span>Base LLM</span><span class="font-mono text-foreground">llama3.2:3b</span></div>
        <div class="flex justify-between"><span>Judge LLM</span><span class="font-mono text-foreground">qwen2.5:7b</span></div>
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
              <span class="inline-flex items-center rounded-full border border-border bg-secondary px-3 py-0.5 font-mono text-xs">Evening 1 · Baseline Monograph</span>
              <span class="inline-flex items-center rounded-full border border-border bg-muted px-3 py-0.5 font-mono text-xs">Mac mini M5 Pro · 48 GB</span>
              <span class="inline-flex items-center rounded-full border border-border bg-muted px-3 py-0.5 font-mono text-xs">Ollama 0.40.18 + FAISS</span>
            </div>
            <h1 class="font-serif text-3xl sm:text-5xl font-bold tracking-tight leading-tight">
              Leakage-Safe Temporal RAG on arXiv <span class="font-mono text-2xl sm:text-4xl">cs.CL</span>
            </h1>
            <p class="text-base sm:text-lg text-muted-foreground max-w-3xl">
              Date-Split Corpus Construction, <code class="font-mono text-sm">bge-small-en-v1.5</code> Vector Ingestion, 60-Prompt 4-Slice Golden Benchmark, Calibrated Cross-Family Judging, and Zero-Shot vs. RAG Baselines.
            </p>
          </div>

          <!-- Abstract -->
          <div class="rounded-xl border border-border bg-secondary/60 p-6 space-y-3">
            <div class="flex items-center justify-between">
              <h2 class="font-serif text-xl font-semibold">Abstract</h2>
              <span class="font-mono text-xs text-muted-foreground">T = 2025-01-01 · 500 Abstracts</span>
            </div>
            <p class="drop-cap text-sm sm:text-base leading-relaxed text-foreground/90">
              Evaluating Retrieval-Augmented Generation (RAG) and future QLoRA hybrids requires strict separation between parametric pretraining knowledge and post-cutoff evidence. We select cutoff date <code class="font-mono text-xs">T = 2025-01-01</code>—strictly after <code class="font-mono text-xs">Llama-3.2-3B-Instruct</code>'s published December 2023 training cutoff—and curate 300 pre-T (<code class="font-mono text-xs">static</code>) and 200 post-T (<code class="font-mono text-xs">live</code>) arXiv <code class="font-mono text-xs">cs.CL</code> abstracts with zero ID overlap. Indexing only the 200 post-T abstracts in FAISS (<code class="font-mono text-xs">BAAI/bge-small-en-v1.5</code>) and benchmarking 60 hand-verified prompts across four slices reveals a sharp hybrid trade-off: RAG (<code class="font-mono text-xs">k=4</code>) lifts <code class="font-mono text-xs">live</code> correctness from <strong>1.00 to 3.85</strong> and <code class="font-mono text-xs">temporal_override</code> obedience from <strong>0% to 70%</strong> while maintaining <strong>5.00/5.00</strong> abstention on adversarial <code class="font-mono text-xs">unanswerable</code> prompts, yet drops <code class="font-mono text-xs">static</code> correctness to <strong>1.80</strong> (20/20 forced abstentions). Calibrating a cross-family <code class="font-mono text-xs">Qwen2.5-7B-Instruct</code> judge via Chain-of-Thought field ordering and explicit abstention rules raised Judge-vs-Human agreement from <strong>48.33%</strong> to <strong>80.0% exact match</strong> (<strong>86.67% within-±1</strong>).
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
                <div><p class="font-serif text-base font-semibold group-hover:underline">Abstract &amp; Agenda</p><p class="text-xs text-muted-foreground mt-0.5">Motivation, temporal cutoff T=2025-01-01, executive KPIs, and full comparison.</p></div>
              </a>
              <a href="#page-2" data-nav-link="1" class="group cursor-pointer text-left rounded-xl border border-border bg-card p-4 hover:bg-accent/60 hover:border-primary/40 transition-colors flex flex-col justify-between gap-2 no-underline text-foreground">
                <div class="flex items-center justify-between w-full"><span class="font-mono text-xs text-muted-foreground">Chapter 1</span><span class="font-mono text-xs rounded-md bg-secondary px-2 py-0.5 text-primary group-hover:bg-primary group-hover:text-primary-foreground transition-colors">Page 2 →</span></div>
                <div><p class="font-serif text-base font-semibold group-hover:underline">Slices &amp; Eval Metrics</p><p class="text-xs text-muted-foreground mt-0.5">Formal definitions of the 4 golden slices, 1–5 scoring anchors, override flag &amp; agreement gate.</p></div>
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
                <div><p class="font-serif text-base font-semibold group-hover:underline">Rubric &amp; Judge Calibration</p><p class="text-xs text-muted-foreground mt-0.5">48.3% -&gt; 80.0% exact agreement lift via CoT ordering and abstention rules.</p></div>
              </a>
              <a href="#page-7" data-nav-link="6" class="group cursor-pointer text-left rounded-xl border border-border bg-card p-4 hover:bg-accent/60 hover:border-primary/40 transition-colors flex flex-col justify-between gap-2 no-underline text-foreground">
                <div class="flex items-center justify-between w-full"><span class="font-mono text-xs text-muted-foreground">Chapter 6</span><span class="font-mono text-xs rounded-md bg-secondary px-2 py-0.5 text-primary group-hover:bg-primary group-hover:text-primary-foreground transition-colors">Page 7 →</span></div>
                <div><p class="font-serif text-base font-semibold group-hover:underline">Baselines &amp; Run Inspector</p><p class="text-xs text-muted-foreground mt-0.5">120 Zero-Shot vs. RAG runs, latency distributions, and prompt-level drilldown.</p></div>
              </a>
              <a href="#page-8" data-nav-link="7" class="group cursor-pointer text-left rounded-xl border border-border bg-card p-4 hover:bg-accent/60 hover:border-primary/40 transition-colors flex flex-col justify-between gap-2 no-underline text-foreground">
                <div class="flex items-center justify-between w-full"><span class="font-mono text-xs text-muted-foreground">Chapter 7</span><span class="font-mono text-xs rounded-md bg-secondary px-2 py-0.5 text-primary group-hover:bg-primary group-hover:text-primary-foreground transition-colors">Page 8 →</span></div>
                <div><p class="font-serif text-base font-semibold group-hover:underline">Failure Analysis &amp; QLoRA</p><p class="text-xs text-muted-foreground mt-0.5">Master results table, 3 failure modes, and Evening 2 QLoRA hybrid roadmap.</p></div>
              </a>
              <a href="#page-9" data-nav-link="8" class="group cursor-pointer sm:col-span-2 text-left rounded-xl border border-dashed border-border bg-secondary/40 p-4 hover:bg-accent/60 hover:border-primary/40 transition-colors flex flex-col justify-between gap-2 no-underline text-foreground">
                <div class="flex items-center justify-between w-full"><span class="font-mono text-xs text-muted-foreground">Appendix A</span><span class="font-mono text-xs rounded-md bg-secondary px-2 py-0.5 text-primary group-hover:bg-primary group-hover:text-primary-foreground transition-colors">Page 9 →</span></div>
                <div><p class="font-serif text-base font-semibold group-hover:underline">Compute &amp; Subagent Orchestration</p><p class="text-xs text-muted-foreground mt-0.5">Mac mini M5 Pro (48 GB) user-space Ollama setup, 3-wave parallel fan-out, and 60 tok/s judge tuning.</p></div>
              </a>
            </div>
          </div>

          <!-- Prologue Summary Chart -->
          <div class="rounded-xl border border-border bg-card p-5 space-y-2">
            <h3 class="font-serif text-lg font-semibold">Executive Comparison: Zero-Shot vs. RAG (<code class="font-mono text-sm">k=4</code>)</h3>
            <p class="text-xs text-muted-foreground">Correctness (1–5) across all 4 slices plus Override Obedience scaled to 5 points (0.70 × 5 = 3.50).</p>
            <div id="chart-p1-overview" class="w-full h-80"></div>
          </div>
        </section>

        <!-- Page 2: Chapter 1 - Slices & Evaluation Metrics Definitions -->
        <section id="page-2" data-page="1" class="book-page hidden space-y-6">
          <div class="border-b border-border pb-4">
            <p class="font-mono text-xs uppercase tracking-wider text-muted-foreground">Chapter 1 · Evaluation Design</p>
            <h1 class="font-serif text-3xl font-bold mt-1">Slice Taxonomy &amp; Evaluation Metrics</h1>
            <p class="text-sm text-muted-foreground mt-1">Formal definitions of the 4 leakage-safe benchmark slices, 1–5 ordinal scoring scales, temporal override flag, latency decomposition, and the &ge;70% human-judge trust gate.</p>
          </div>

          <!-- 4 Slice Definitions -->
          <div class="space-y-2.5">
            <h2 class="font-serif text-xl font-semibold">1. Four Golden Slice Types (<code class="font-mono text-base">N = 60</code>)</h2>
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
              <div class="rounded-xl border border-border bg-secondary/50 p-4 space-y-1.5">
                <div class="flex items-center justify-between">
                  <span class="font-mono font-semibold text-primary">static · n = 20 (g01–g20)</span>
                  <span class="rounded bg-muted px-2 py-0.5 font-mono text-[10px]">Published &lt; T</span>
                </div>
                <p><strong>What it tests:</strong> Pre-cutoff <code class="font-mono">cs.CL</code> facts (15 landmark pre-Dec-2023 papers + 5 early-2024 papers) absent from the post-T RAG index.</p>
                <p class="text-muted-foreground"><strong>Signal isolated:</strong> Parametric retention in zero-shot vs. over-abstention penalty when live-only RAG finds no matching post-T source.</p>
              </div>
              <div class="rounded-xl border border-border bg-secondary/50 p-4 space-y-1.5">
                <div class="flex items-center justify-between">
                  <span class="font-mono font-semibold text-primary">live · n = 20 (g21–g40)</span>
                  <span class="rounded bg-muted px-2 py-0.5 font-mono text-[10px]">Published &gt;= T</span>
                </div>
                <p><strong>What it tests:</strong> Post-cutoff (2025–2026) methods, datasets, and quantitative findings guaranteed unseen during <code class="font-mono">llama3.2:3b</code> pretraining.</p>
                <p class="text-muted-foreground"><strong>Signal isolated:</strong> Dense <code class="font-mono">bge-small</code> retrieval recall + grounded synthesis + <code class="font-mono">[arxiv_id]</code> citation discipline.</p>
              </div>
              <div class="rounded-xl border border-border bg-secondary/50 p-4 space-y-1.5">
                <div class="flex items-center justify-between">
                  <span class="font-mono font-semibold text-primary">temporal_override · n = 10 (g41–g50)</span>
                  <span class="rounded bg-muted px-2 py-0.5 font-mono text-[10px]">Post-T Beats Pre-T</span>
                </div>
                <p><strong>What it tests:</strong> Current-state queries (<em>"What is the current best approach to X?"</em>) where post-T evidence overturns or outperforms earlier paradigms.</p>
                <p class="text-muted-foreground"><strong>Signal isolated:</strong> Conflict resolution—whether RAG adopts <code class="font-mono">newer_paper_id</code> instead of clinging to pretraining priors or hedging.</p>
              </div>
              <div class="rounded-xl border border-border bg-secondary/50 p-4 space-y-1.5">
                <div class="flex items-center justify-between">
                  <span class="font-mono font-semibold text-primary">unanswerable · n = 10 (g51–g60)</span>
                  <span class="rounded bg-muted px-2 py-0.5 font-mono text-[10px]">Not in Sources</span>
                </div>
                <p><strong>What it tests:</strong> 6 out-of-corpus prompts + 4 near-miss distractors targeting indexed papers (<code class="font-mono">BaseRT</code>, <code class="font-mono">MSQA</code>, etc.) while asking unstated metrics.</p>
                <p class="text-muted-foreground"><strong>Signal isolated:</strong> Precision abstention (<code class="font-mono">"Not in the sources."</code>) without hallucinating numbers from top-4 semantic distractors.</p>
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
                        <th>Judge (C/F/Cit/Ov)</th>
                        <th>Human Note &amp; Judge Rationale</th>
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
            <p class="font-mono text-xs uppercase tracking-wider text-muted-foreground">Chapter 6 · Empirical Baselines</p>
            <h1 class="font-serif text-3xl font-bold mt-1">Zero-Shot vs. RAG (<code class="font-mono text-2xl">k=4</code>) Performance &amp; Run Inspector</h1>
            <p class="text-sm text-muted-foreground mt-1">120 sequential runs on <code class="font-mono text-xs">llama3.2:3b</code>: slice correctness, abstention/citation behavior, latency distributions, and side-by-side prompt inspection.</p>
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
                <h3 class="font-serif text-xl font-semibold">Interactive Run Inspector (<code class="font-mono text-sm">g01–g60</code>)</h3>
                <p class="text-xs text-muted-foreground">Compare Zero-Shot vs. RAG answers, citations, latencies, and judge rationales on any prompt.</p>
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

        <!-- Page 8: Chapter 7 - Failure Analysis & Evening 2 Roadmap -->
        <section id="page-8" data-page="7" class="book-page hidden space-y-6">
          <div class="border-b border-border pb-4">
            <p class="font-mono text-xs uppercase tracking-wider text-muted-foreground">Chapter 7 · Synthesis &amp; Next Evening</p>
            <h1 class="font-serif text-3xl font-bold mt-1">Failure Analysis &amp; Evening 2 QLoRA Outlook</h1>
            <p class="text-sm text-muted-foreground mt-1">Master comparison table, empirical failure modes, and concrete hypotheses for QLoRA fine-tuning on pre-T weights + live RAG.</p>
          </div>

          <div class="space-y-2">
            <h3 id="master-table-caption" class="font-serif text-lg font-semibold">Master Evaluation Results Table</h3>
            <div tabindex="0" role="region" aria-labelledby="master-table-caption" class="table-wrapper book-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Mode</th>
                    <th>Slice</th>
                    <th>N</th>
                    <th>Correctness</th>
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
              <p class="font-semibold">Static RAG collapses to 1.80 (20/20 abstentions)</p>
              <p class="text-muted-foreground">Pre-T papers (LoRA, DPO, RAG) are absent from <code class="font-mono">live.jsonl</code>, so strict RAG grounding blocks parametric recall.</p>
            </div>
            <div class="rounded-xl border border-border bg-secondary/50 p-4 space-y-1.5">
              <span class="font-mono font-semibold text-primary">Failure 2 · 3B Omissions</span>
              <p class="font-semibold">Partial answers + spurious abstentions</p>
              <p class="text-muted-foreground">On multi-part live abstracts (e.g. <code class="font-mono">g27</code> NIRANTAR), 3B captures scenarios/languages yet drops hour counts or appends <em>"Not in the sources."</em></p>
            </div>
            <div class="rounded-xl border border-border bg-secondary/50 p-4 space-y-1.5">
              <span class="font-mono font-semibold text-primary">Failure 3 · Override Hedging</span>
              <p class="font-semibold">3/10 temporal_override misses</p>
              <p class="text-muted-foreground">On <code class="font-mono">g44</code>, <code class="font-mono">g46</code>, and <code class="font-mono">g48</code>, RAG abstained despite rank-1 retrieval of the newer paper instead of synthesizing the empirical contradiction.</p>
            </div>
          </div>

          <div class="grid grid-cols-1 lg:grid-cols-2 gap-4 items-stretch">
            <div class="rounded-xl border border-border bg-card p-4 space-y-2">
              <h3 class="font-serif text-lg font-semibold">Evening 1 Baselines vs. Evening 2 Hybrid Target</h3>
              <p class="text-xs text-muted-foreground">What QLoRA fine-tuning changes (static recall + routing + formatting) vs. what stays dependent on RAG (live freshness).</p>
              <div id="chart-p8-roadmap" class="w-full h-72"></div>
            </div>
            <div class="rounded-xl border border-border bg-secondary/40 p-5 space-y-3 text-sm">
              <h3 class="font-serif text-lg font-semibold">What QLoRA Will (and Will Not) Change</h3>
              <div class="space-y-2 text-xs leading-relaxed">
                <p><strong>Will Change (Weights + Hybrid Routing):</strong> Fine-tuning <code class="font-mono">Llama-3.2-3B-Instruct</code> on pre-T <code class="font-mono">static.jsonl</code> internalizes foundational <code class="font-mono">cs.CL</code> mechanisms and metrics, trains calibrated confidence fallback when live FAISS similarity is low, sharpens <code class="font-mono">[arxiv_id]</code> citation discipline, and removes spurious trailing abstentions.</p>
                <p><strong>Will NOT Change (Live Retrieval Dependency):</strong> Post-T knowledge (<code class="font-mono">&gt;= 2025-01-01</code>) cannot be memorized in static weights without continual retraining; fresh papers and <code class="font-mono">temporal_override</code> updates will always require <code class="font-mono">retrieve(query, k, after=T)</code>.</p>
              </div>
              <div class="pt-2 border-t border-border font-mono text-[11px] text-muted-foreground">
                Reproduce: <code class="text-foreground">uv run python main.py book &amp;&amp; uv run pytest -v</code>
              </div>
            </div>
          </div>
        </section>

        <!-- Page 9: Appendix A - Compute & Subagent Orchestration -->
        <section id="page-9" data-page="8" class="book-page hidden space-y-6">
          <div class="border-b border-border pb-4">
            <p class="font-mono text-xs uppercase tracking-wider text-muted-foreground">Appendix A · Compute &amp; Orchestration</p>
            <h1 class="font-serif text-3xl font-bold mt-1">Mac mini M5 Pro Setup &amp; Parallel Subagent Fan-Out</h1>
            <p class="text-sm text-muted-foreground mt-1">Zero-sudo user-space provisioning of <code class="font-mono text-xs">uv</code>, Python 3.13, and <code class="font-mono text-xs">Ollama 0.40.18</code> over SSH, plus multi-wave subagent parallelization.</p>
          </div>

          <div id="app-hw-cards" class="grid grid-cols-1 md:grid-cols-3 gap-4"></div>

          <div class="rounded-xl border border-border bg-secondary/50 p-5 space-y-3">
            <h2 class="font-serif text-xl font-semibold">3-Wave Subagent Fan-Out Architecture</h2>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
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
                <span class="font-mono font-semibold text-primary">Wave 3 · Sequential + 2 Parallel</span>
                <p class="text-muted-foreground">Sequential <code class="font-mono">llama3.2:3b</code> baselines for clean uncontended latency.</p>
                <p class="text-muted-foreground">3A: <code class="font-mono">qwen2.5:7b</code> judge runner on Mac mini.</p>
                <p class="text-muted-foreground">3B: Blinded Human Grader on 20 stratified samples (zero judge leakage).</p>
              </div>
            </div>
          </div>

          <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div class="rounded-xl border border-border bg-card p-4 space-y-2">
              <h3 class="font-serif text-lg font-semibold">48 GB Unified Memory Footprint</h3>
              <p class="text-xs text-muted-foreground">Model weights + embeddings vs. available Metal headroom on M5 Pro.</p>
              <div id="chart-app-memory" class="w-full h-72"></div>
            </div>
            <div class="rounded-xl border border-border bg-card p-4 space-y-2">
              <h3 class="font-serif text-lg font-semibold">Judge Decoding Throughput Engineering</h3>
              <p class="text-xs text-muted-foreground">Removing 152k-vocab GBNF schema mask &amp; multi-slot contention accelerated Qwen2.5-7B.</p>
              <div id="chart-app-throughput" class="w-full h-72"></div>
            </div>
          </div>
        </section>

        <!-- Bottom Book Page Navigation Footer -->
        <footer class="mt-8 pt-5 border-t border-border flex items-center justify-between gap-4">
          <button id="btn-prev-page" type="button" class="inline-flex items-center gap-2 rounded-lg border border-border bg-secondary px-4 py-2 text-sm font-medium hover:bg-accent disabled:opacity-40 disabled:pointer-events-none">
            ← Previous Page
          </button>
          <span id="footer-folio" class="font-serif italic text-sm text-muted-foreground">Page 1 of 9</span>
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
      { idx: 0, label: "Prologue", title: "Abstract & Agenda", desc: "Motivation, temporal cutoff T=2025-01-01, executive KPIs, and full comparison." },
      { idx: 1, label: "Chapter 1", title: "Slices & Eval Metrics", desc: "Formal definitions of the 4 golden slices, 1–5 scoring anchors, override flag & agreement gate." },
      { idx: 2, label: "Chapter 2", title: "Leakage-Safe Date Split", desc: "300 static (<T) vs. 200 live (>=T) cs.CL abstracts with 0 ID overlap." },
      { idx: 3, label: "Chapter 3", title: "Ingestion & FAISS Retrieval", desc: "bge-small-en-v1.5 384-d embeddings, IndexFlatIP, and after=T filtering." },
      { idx: 4, label: "Chapter 4", title: "60-Prompt Golden Set", desc: "Interactive explorer across Static, Live, Temporal Override, and Unanswerable." },
      { idx: 5, label: "Chapter 5", title: "Rubric & Judge Calibration", desc: "48.3% -> 80.0% exact agreement lift via CoT ordering and abstention rules." },
      { idx: 6, label: "Chapter 6", title: "Baselines & Run Inspector", desc: "120 Zero-Shot vs. RAG runs, latency distributions, and prompt-level drilldown." },
      { idx: 7, label: "Chapter 7", title: "Failure Analysis & QLoRA", desc: "Master results table, 3 failure modes, and Evening 2 QLoRA hybrid roadmap." },
      { idx: 8, label: "Appendix A", title: "Compute & Orchestration", desc: "Mac mini M5 Pro user-space Ollama setup, 3-wave agents, and 60 tok/s judge tuning." },
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
        const bms = DATA.results.by_mode_slice;
        const cats = ["Static", "Live", "Temporal Override", "Unanswerable", "Override Obed (×5)"];
        const zsVals = [
          bms.zero_shot.static.correctness,
          bms.zero_shot.live.correctness,
          bms.zero_shot.temporal_override.correctness,
          bms.zero_shot.unanswerable.correctness,
          bms.zero_shot.temporal_override.override_obedience * 5,
        ];
        const ragVals = [
          bms.rag.static.correctness,
          bms.rag.live.correctness,
          bms.rag.temporal_override.correctness,
          bms.rag.unanswerable.correctness,
          bms.rag.temporal_override.override_obedience * 5,
        ];
        Plotly.newPlot("chart-p1-overview", [
          { x: cats, y: zsVals, name: "Zero-Shot (llama3.2:3b)", type: "bar", marker: { color: t.zsColor }, text: zsVals.map((v) => v.toFixed(2)), textposition: "outside" },
          { x: cats, y: ragVals, name: "RAG k=4 (bge-small)", type: "bar", marker: { color: t.ragColor }, text: ragVals.map((v) => v.toFixed(2)), textposition: "outside" },
        ], baseLayout({
          barmode: "group",
          yaxis: { range: [0, 5.8], title: "Score (0–5)", gridcolor: t.gridColor },
          xaxis: { gridcolor: t.gridColor },
          legend: { orientation: "h", y: 1.12 },
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
        Plotly.newPlot("chart-p6-calibration", [
          { x: dims, y: v1, name: "v1 Uncalibrated", type: "bar", marker: { color: t.zsColor } },
          { x: dims, y: v2, name: "v2 Calibrated", type: "bar", marker: { color: t.greenColor } },
        ], baseLayout({
          barmode: "group",
          yaxis: { title: "Agreement (%)", range: [0, 112], gridcolor: t.gridColor },
          legend: { orientation: "h", y: 1.14 },
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
        const axes = ["Static Recall", "Live Freshness", "Temporal Override", "Abstention Safety", "Citation Discipline"];
        Plotly.newPlot("chart-p8-roadmap", [
          { x: axes, y: [1.45, 1.00, 0.0, 5.00, 1.00], name: "Zero-Shot (E1)", type: "scatter", mode: "lines+markers", line: { color: t.zsColor, width: 2.5 } },
          { x: axes, y: [1.80, 3.85, 3.50, 5.00, 4.98], name: "RAG Live-Only (E1)", type: "scatter", mode: "lines+markers", line: { color: t.ragColor, width: 2.5 } },
          { x: axes, y: [4.30, 4.30, 4.50, 4.90, 4.98], name: "Target: QLoRA + Live RAG (E2)", type: "scatter", mode: "lines+markers", line: { color: t.greenColor, width: 3, dash: "dot" } },
        ], baseLayout({
          yaxis: { title: "Capability (0–5)", range: [0, 5.5], gridcolor: t.gridColor },
          margin: { t: 24, r: 18, b: 64, l: 48 },
          xaxis: { tickangle: -15, gridcolor: t.gridColor },
          legend: { orientation: "h", y: 1.16 },
        }), PLOT_CONFIG);
      }

      if (idx === 8) {
        const models = DATA.env.models;
        const llamaGb = models.find((m) => m.name.includes("llama3.2"))?.size_gb ?? 2.02;
        const qwenGb = models.find((m) => m.name.includes("qwen2.5"))?.size_gb ?? 4.68;
        const bgeGb = 0.13;
        const freeGb = Number((48 - llamaGb - qwenGb - bgeGb).toFixed(2));
        Plotly.newPlot("chart-app-memory", [
          { y: ["48 GB Unified"], x: [llamaGb], name: `llama3.2:3b (${llamaGb} GB)`, type: "bar", orientation: "h", marker: { color: t.ragColor } },
          { y: ["48 GB Unified"], x: [qwenGb], name: `qwen2.5:7b (${qwenGb} GB)`, type: "bar", orientation: "h", marker: { color: t.accentColor } },
          { y: ["48 GB Unified"], x: [bgeGb], name: `bge-small (${bgeGb} GB)`, type: "bar", orientation: "h", marker: { color: t.greenColor } },
          { y: ["48 GB Unified"], x: [freeGb], name: `OS & KV Headroom (${freeGb} GB)`, type: "bar", orientation: "h", marker: { color: t.gridColor } },
        ], baseLayout({
          barmode: "stack",
          margin: { t: 24, r: 18, b: 44, l: 110 },
          xaxis: { range: [0, 48], title: "Gigabytes (GB)", gridcolor: t.gridColor },
          legend: { orientation: "h", y: 1.18 },
        }), PLOT_CONFIG);

        Plotly.newPlot("chart-app-throughput", [{
          x: ["GBNF Schema (4-slot)", "GBNF Schema (1-slot)", 'format="json" (Peak)'],
          y: [1.08, 2.53, 59.7],
          type: "bar",
          marker: { color: [t.roseColor, t.accentColor, t.greenColor] },
          text: ["1.08 tok/s", "2.53 tok/s", "59.7 tok/s"],
          textposition: "outside",
        }], baseLayout({
          yaxis: { title: "Tokens / sec", range: [0, 68], gridcolor: t.gridColor },
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
      const kpis = [
        { k: "Date-Split Corpus", v: "500 Abstracts", s: "300 static (<T) · 200 live (>=T) · 0 overlap" },
        { k: "Golden Benchmark", v: "60 Prompts", s: "20 static · 20 live · 10 override · 10 unansw." },
        { k: "RAG vs. Zero-Shot", v: "3.42 vs. 2.20", s: "Live 3.85 (+2.85) · Override Obed 70% vs 0%" },
        { k: "Judge Agreement", v: "80.0% Exact", s: "86.67% within-±1 · 100% override exact" },
      ];
      document.getElementById("p1-kpis").innerHTML = kpis
        .map(
          (item) => `<div class="rounded-xl border border-border bg-card p-4 shadow-sm"><p class="font-mono text-[11px] text-muted-foreground">${item.k}</p><p class="font-serif text-2xl font-bold mt-1">${item.v}</p><p class="text-xs text-muted-foreground mt-1">${item.s}</p></div>`
        )
        .join("");

      // Page 2 Hardware & Model Cards
      const hw = DATA.env.host;
      const hwHtml = [
        `<div class="rounded-xl border border-border bg-secondary/50 p-4 space-y-1"><p class="font-mono text-xs text-muted-foreground">Compute Host</p><p class="font-semibold">${hw.model_name} (${hw.chip})</p><p class="text-xs text-muted-foreground">${hw.total_number_of_cores} · ${hw.memory} Unified · macOS ${DATA.env.os}</p><p class="font-mono text-[11px] text-muted-foreground">${DATA.env.uv.split(" ")[0]} + ${DATA.env.python}</p></div>`,
        ...DATA.env.models.map(
          (m) => `<div class="rounded-xl border border-border bg-secondary/50 p-4 space-y-1"><p class="font-mono text-xs text-muted-foreground">${m.name.includes("llama") ? "Base Model" : "Cross-Family Judge"}</p><p class="font-semibold font-mono text-sm">${m.name} (${m.params})</p><p class="text-xs text-muted-foreground">${m.quant} · ${m.size_gb} GB · ctx=${m.ctx.toLocaleString()} · 100% Metal GPU</p><p class="font-mono text-[11px] text-muted-foreground">sha256:${m.digest}...</p></div>`
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
          const exact = a.human.correctness === a.judge.correctness && a.human.faithfulness === a.judge.faithfulness && a.human.citation_accuracy === a.judge.citation_accuracy;
          return `<tr>
            <th class="font-mono text-xs">${escapeHtml(a.sample_id)}<div class="text-[11px] text-muted-foreground font-normal">${escapeHtml(a.id)}</div></th>
            <td data-label="Mode / Slice" class="font-mono text-xs">${escapeHtml(a.mode)}<br><span class="text-muted-foreground">${escapeHtml(a.slice)}</span></td>
            <td data-label="Human" class="font-mono text-xs font-semibold">${hStr}</td>
            <td data-label="Judge" class="font-mono text-xs font-semibold">${jStr} ${exact ? '<span class="ml-1 text-[10px] rounded bg-muted px-1">EXACT</span>' : ""}</td>
            <td data-label="Notes" class="text-xs space-y-1"><div><strong>Human:</strong> ${escapeHtml(a.human.notes)}</div><div class="text-muted-foreground"><strong>Judge:</strong> ${escapeHtml(a.judge.rationale)}</div></td>
          </tr>`;
        })
        .join("");

      // Page 7 Run Inspector
      const sel = document.getElementById("inspector-select");
      sel.innerHTML = DATA.runs.map((r, i) => `<option value="${i}">${r.id} [${r.slice}] ${r.prompt.slice(0, 62)}...</option>`).join("");
      function renderInspector(index) {
        const r = DATA.runs[index];
        sel.value = String(index);
        document.getElementById("inspector-prev").disabled = index === 0;
        document.getElementById("inspector-next").disabled = index === DATA.runs.length - 1;

        function card(title, run, isRag) {
          const ovBadge = run.override_obedience === null ? "" : `<span class="rounded bg-muted px-2 py-0.5 font-mono text-[11px]">Override: ${run.override_obedience}</span>`;
          const timing = isRag
            ? `Total ${run.latency_ms} ms (ret ${run.retrieval_ms} ms + gen ${run.generation_ms} ms)`
            : `Total ${run.latency_ms} ms`;
          return `<div class="rounded-xl border border-border bg-card p-4 space-y-3">
            <div class="flex flex-wrap items-center justify-between gap-2 border-b border-border pb-2">
              <span class="font-serif text-base font-semibold">${title}</span>
              <span class="font-mono text-[11px] text-muted-foreground">${timing}</span>
            </div>
            <div class="flex flex-wrap gap-1.5">
              <span class="rounded bg-muted px-2 py-0.5 font-mono text-[11px]">Corr: ${run.correctness}/5</span>
              <span class="rounded bg-muted px-2 py-0.5 font-mono text-[11px]">Faith: ${run.faithfulness}/5</span>
              <span class="rounded bg-muted px-2 py-0.5 font-mono text-[11px]">Cite: ${run.citation_accuracy}/5</span>
              ${ovBadge}
            </div>
            <div class="text-xs leading-relaxed whitespace-pre-wrap">${escapeHtml(run.answer)}</div>
            <div class="text-[11px] text-muted-foreground border-t border-border pt-2 space-y-1">
              <div><strong>Cited IDs:</strong> <span class="font-mono">${run.cited.length ? escapeHtml(run.cited.join(", ")) : "none"}</span></div>
              <div><strong>Judge Rationale:</strong> ${escapeHtml(run.rationale)}</div>
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
            ${card("Zero-Shot (llama3.2:3b)", r.zero_shot, false)}
            ${card("RAG k=4 (llama3.2:3b + bge-small)", r.rag, true)}
          </div>`;
      }
      sel.addEventListener("change", () => renderInspector(Number(sel.value)));
      document.getElementById("inspector-prev").addEventListener("click", () => renderInspector(Math.max(0, Number(sel.value) - 1)));
      document.getElementById("inspector-next").addEventListener("click", () => renderInspector(Math.min(DATA.runs.length - 1, Number(sel.value) + 1)));
      renderInspector(40); // Default to temporal_override g41 SOLAR

      // Page 8 Master Table
      const bms = DATA.results.by_mode_slice;
      const ov = DATA.results.overall;
      const rows = [];
      for (const mode of ["zero_shot", "rag"]) {
        const mLabel = mode === "zero_shot" ? "Zero-Shot" : "RAG (k=4)";
        for (const sl of ["static", "live", "temporal_override", "unanswerable"]) {
          const s = bms[mode][sl];
          rows.push({ mode: mLabel, slice: sl, ...s, bold: false });
        }
        rows.push({ mode: mLabel, slice: "ALL (mean)", ...ov[mode], bold: true });
      }
      document.getElementById("master-tbody").innerHTML = rows
        .map(
          (r) => `<tr class="${r.bold ? "bg-muted/60 font-semibold" : ""}">
            <th>${escapeHtml(r.mode)}</th>
            <td data-label="Slice" class="font-mono text-xs">${escapeHtml(r.slice)}</td>
            <td data-label="N" class="font-mono text-xs">${r.count}</td>
            <td data-label="Correctness" class="font-mono text-xs">${r.correctness.toFixed(2)}</td>
            <td data-label="Faithfulness" class="font-mono text-xs">${r.faithfulness.toFixed(2)}</td>
            <td data-label="Citation Acc" class="font-mono text-xs">${r.citation_accuracy.toFixed(2)}</td>
            <td data-label="Override Obed" class="font-mono text-xs">${r.override_obedience === null ? "—" : r.override_obedience.toFixed(2)}</td>
            <td data-label="Mean Latency (ms)" class="font-mono text-xs">${r.latency_mean_ms.toFixed(1)}</td>
          </tr>`
        )
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
        renderChartsForPage(currentPage, true);
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

"""Run zero-shot and RAG baselines against `eval/golden.jsonl` via Ollama."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import httpx

from src import extract_arxiv_ids, ollama_chat, read_jsonl, write_jsonl
from src.retriever import Retriever

MODEL_NAME = "llama3.2:3b"
GENERATION_OPTIONS: dict[str, Any] = {
    "temperature": 0.0,
    "num_predict": 384,
}

ZERO_SHOT_SYSTEM = (
    "You are a concise NLP research assistant. Answer the question using only your "
    "internal knowledge. Be direct and accurate. If you do not know the answer or are "
    "unsure, say so honestly."
)

RAG_SYSTEM = (
    "You are a strict evidence-grounded NLP research assistant. Answer the question "
    "using ONLY the provided arXiv abstracts. Cite supporting arXiv IDs in square "
    "brackets like [2503.01234] for every factual claim. If the provided abstracts do "
    "not contain the answer, reply exactly with: Not in the sources."
)


def format_chunks(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks for inclusion in the RAG prompt."""
    return "\n\n".join(
        f"[ID: {chunk['id']} | Published: {chunk['published_date']}] "
        f"{chunk['title']}\n{chunk['abstract']}"
        for chunk in chunks
    )


def _generate_answer(
    client: httpx.Client,
    *,
    system_prompt: str,
    user_prompt: str,
    model: str = MODEL_NAME,
) -> tuple[str, float]:
    """Call Ollama `/api/chat` and return `(answer_text, generation_ms)`."""
    t0 = time.perf_counter()
    data = ollama_chat(
        client,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options=GENERATION_OPTIONS,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    content = str(data.get("message", {}).get("content", "")).strip()
    return content, round(elapsed_ms, 2)


def _base_record(
    item: dict[str, Any],
    *,
    mode: str,
    model_answer: str,
    retrieved_context: str | None,
    valid_context_ids: list[str],
    latency_ms: float,
    retrieval_ms: float | None = None,
    generation_ms: float | None = None,
) -> dict[str, Any]:
    """Build a normalized baseline output row."""
    record: dict[str, Any] = {
        "id": item["id"],
        "mode": mode,
        "slice": item["slice"],
        "prompt": item["prompt"],
        "reference_answer": item["reference_answer"],
        "source_paper_ids": item.get("source_paper_ids", []),
        "newer_paper_id": item.get("newer_paper_id"),
        "retrieved_context": retrieved_context,
        "valid_context_ids": valid_context_ids,
        "model_answer": model_answer,
        "extracted_cited_ids": extract_arxiv_ids(model_answer),
        "latency_ms": round(latency_ms, 2),
    }
    if retrieval_ms is not None:
        record["retrieval_ms"] = round(retrieval_ms, 2)
    if generation_ms is not None:
        record["generation_ms"] = round(generation_ms, 2)
    return record


def run_baselines(
    golden_path: str | Path = "eval/golden.jsonl",
    index_dir: str | Path = "data/index",
    out_dir: str | Path = "eval",
    *,
    k: int = 4,
    model: str = MODEL_NAME,
) -> tuple[Path, Path]:
    """Execute zero-shot and RAG baselines sequentially and write output JSONL files."""
    golden_items = read_jsonl(golden_path)
    retriever = Retriever(index_dir=index_dir)

    out_root = Path(out_dir)
    zs_out = out_root / "outputs_zero_shot.jsonl"
    rag_out = out_root / "outputs_rag.jsonl"

    zs_rows: list[dict[str, Any]] = []
    rag_rows: list[dict[str, Any]] = []

    with httpx.Client(timeout=120.0) as client:
        # Warm up Ollama once before per-query timing begins.
        _generate_answer(
            client,
            system_prompt="Reply briefly.",
            user_prompt="Say ok.",
            model=model,
        )

        for item in golden_items:
            answer, gen_ms = _generate_answer(
                client,
                system_prompt=ZERO_SHOT_SYSTEM,
                user_prompt=item["prompt"],
                model=model,
            )
            zs_rows.append(
                _base_record(
                    item,
                    mode="zero_shot",
                    model_answer=answer,
                    retrieved_context=None,
                    valid_context_ids=[],
                    latency_ms=gen_ms,
                )
            )

        for item in golden_items:
            t_ret0 = time.perf_counter()
            chunks = retriever.search(item["prompt"], k=k)
            ret_ms = (time.perf_counter() - t_ret0) * 1000.0

            context_text = format_chunks(chunks)
            valid_ids = [chunk["id"] for chunk in chunks]
            user_prompt = (
                f"Retrieved Abstracts:\n{context_text}\n\n"
                f"Question: {item['prompt']}"
            )
            answer, gen_ms = _generate_answer(
                client,
                system_prompt=RAG_SYSTEM,
                user_prompt=user_prompt,
                model=model,
            )
            rag_rows.append(
                _base_record(
                    item,
                    mode="rag",
                    model_answer=answer,
                    retrieved_context=context_text,
                    valid_context_ids=valid_ids,
                    latency_ms=ret_ms + gen_ms,
                    retrieval_ms=ret_ms,
                    generation_ms=gen_ms,
                )
            )

    write_jsonl(zs_out, zs_rows)
    write_jsonl(rag_out, rag_rows)
    return zs_out, rag_out


def main() -> None:
    """CLI entry point for baseline generation."""
    parser = argparse.ArgumentParser(description="Run zero-shot and RAG baselines.")
    parser.add_argument("--golden-path", default="eval/golden.jsonl")
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--out-dir", default="eval")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()

    zs_out, rag_out = run_baselines(
        golden_path=args.golden_path,
        index_dir=args.index_dir,
        out_dir=args.out_dir,
        k=args.k,
        model=args.model,
    )
    print(f"Wrote {zs_out} and {rag_out}")


if __name__ == "__main__":
    main()

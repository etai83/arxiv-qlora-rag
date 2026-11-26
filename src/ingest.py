"""Ingest post-cutoff arXiv abstracts into a FAISS inner-product index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src import read_jsonl

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def format_chunk(record: dict[str, Any]) -> str:
    """Format an arXiv corpus record as a single retrieval chunk."""
    return f"Title: {record['title']}\nAbstract: {record['abstract']}"


def build_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Extract metadata stored alongside each FAISS vector."""
    return {
        "id": record["id"],
        "title": record["title"],
        "abstract": record["abstract"],
        "published_date": record.get("date") or record["published_date"],
        "categories": record["categories"],
    }


def ingest(
    live_path: str | Path = "corpus/live.jsonl",
    index_dir: str | Path = "data/index",
    model_name: str = EMBEDDING_MODEL,
) -> dict[str, Any]:
    """Encode live corpus abstracts and persist FAISS index + metadata JSON."""
    records = read_jsonl(live_path)
    if not records:
        raise ValueError(f"No records found in {live_path}")

    texts = [format_chunk(record) for record in records]
    metadata = [build_metadata(record) for record in records]

    encoder = SentenceTransformer(model_name)
    embeddings = encoder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    vectors = np.ascontiguousarray(embeddings, dtype=np.float32)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    out_dir = Path(index_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss_path = out_dir / "live.faiss"
    meta_path = out_dir / "live_meta.json"

    faiss.write_index(index, str(faiss_path))
    meta_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "count": len(records),
        "dim": int(vectors.shape[1]),
        "faiss_path": str(faiss_path),
        "meta_path": str(meta_path),
    }


def main() -> None:
    """CLI entry point for building the live FAISS index."""
    parser = argparse.ArgumentParser(description="Build FAISS index from live corpus.")
    parser.add_argument("--live-path", default="corpus/live.jsonl")
    parser.add_argument("--index-dir", default="data/index")
    args = parser.parse_args()

    summary = ingest(live_path=args.live_path, index_dir=args.index_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

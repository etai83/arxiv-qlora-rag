"""Dense vector retriever over the live arXiv FAISS index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.ingest import EMBEDDING_MODEL

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def _get_encoder(model_name: str) -> SentenceTransformer:
    """Return a cached SentenceTransformer instance for `model_name`."""
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


class Retriever:
    """Lazy/cached singleton FAISS retriever for post-cutoff arXiv abstracts."""

    _instances: dict[tuple[str, str], Retriever] = {}

    def __new__(
        cls,
        index_dir: str | Path = "data/index",
        model_name: str = EMBEDDING_MODEL,
        *,
        force_reload: bool = False,
    ) -> Retriever:
        key = (str(Path(index_dir).resolve()), model_name)
        if force_reload or key not in cls._instances:
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instances[key] = instance
        return cls._instances[key]

    def __init__(
        self,
        index_dir: str | Path = "data/index",
        model_name: str = EMBEDDING_MODEL,
        *,
        force_reload: bool = False,
    ) -> None:
        if getattr(self, "_initialized", False) and not force_reload:
            return
        resolved_dir = Path(index_dir)
        faiss_path = resolved_dir / "live.faiss"
        meta_path = resolved_dir / "live_meta.json"
        if not faiss_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"FAISS index or metadata missing in {resolved_dir}. Run `python -m src.ingest` first."
            )

        self.index_dir = resolved_dir
        self.model_name = model_name
        self.index = faiss.read_index(str(faiss_path))
        self.metadata: list[dict[str, Any]] = json.loads(
            meta_path.read_text(encoding="utf-8")
        )
        self.encoder = _get_encoder(model_name)
        self._initialized = True

    def search(
        self,
        query: str,
        k: int = 4,
        after: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return top-`k` abstracts optionally filtered by `published_date >= after`."""
        ntotal = int(self.index.ntotal)
        if ntotal == 0 or k <= 0:
            return []

        encoded = self.encoder.encode(
            [f"{QUERY_PREFIX}{query}"],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        query_vec = np.ascontiguousarray(encoded, dtype=np.float32)
        scores, indices = self.index.search(query_vec, ntotal)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            item = self.metadata[int(idx)]
            if after is not None and item["published_date"] < after:
                continue
            results.append(
                {
                    "id": item["id"],
                    "title": item["title"],
                    "abstract": item["abstract"],
                    "published_date": item["published_date"],
                    "categories": item["categories"],
                    "score": float(score),
                }
            )
            if len(results) >= k:
                break
        return results


def retrieve(
    query: str,
    k: int = 4,
    after: str | None = None,
    index_dir: str | Path = "data/index",
) -> list[dict[str, Any]]:
    """Retrieve top-`k` relevant abstracts from the cached FAISS `Retriever`."""
    return Retriever(index_dir=index_dir).search(query=query, k=k, after=after)

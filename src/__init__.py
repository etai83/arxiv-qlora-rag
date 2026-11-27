"""Shared utilities for JSONL I/O, arXiv ID extraction, and Ollama chat calls."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import httpx

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
ARXIV_ID_RE = re.compile(r"\b(\d{4}\.\d{4,5})\b")


def get_ollama_url() -> str:
    """Return the configured Ollama `/api/chat` endpoint URL."""
    return os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load all non-empty JSON objects from a JSONL file."""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    """Write an iterable of dictionaries to a JSONL file, creating parents."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return file_path


def extract_arxiv_ids(text: str) -> list[str]:
    """Extract unique arXiv IDs (`YYMM.NNNN[N]`) in order of appearance."""
    return list(dict.fromkeys(ARXIV_ID_RE.findall(text)))


def ollama_chat(
    client: httpx.Client,
    *,
    model: str,
    messages: list[dict[str, str]],
    options: dict[str, Any] | None = None,
    format_schema: dict[str, Any] | str | None = None,
    think: bool | None = None,
    url: str | None = None,
) -> dict[str, Any]:
    """Send a non-streaming `/api/chat` request to Ollama and return parsed JSON."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if options is not None:
        payload["options"] = options
    if format_schema is not None:
        payload["format"] = format_schema
    if think is not None:
        payload["think"] = think

    response = client.post(url or get_ollama_url(), json=payload)
    response.raise_for_status()
    return response.json()

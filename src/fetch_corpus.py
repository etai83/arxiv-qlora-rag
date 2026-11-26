#!/usr/bin/env python3
"""Fetch and build date-split arXiv cs.CL corpus (static < 2025-01-01, live >= 2025-01-01)."""

from __future__ import annotations

import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ARXIV_API_URL = "https://export.arxiv.org/api/query"
USER_AGENT = "arxiv-qlora-rag/1.0 (https://arxiv.org; mailto:research@example.org)"
MIN_REQUEST_INTERVAL = 3.1
BATCH_SIZE = 100
CUTOFF_DATE = "2025-01-01"
TARGET_CATEGORY = "cs.CL"

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
ID_PREFIX_RE = re.compile(r"^https?://arxiv\.org/abs/")
ID_VERSION_RE = re.compile(r"v\d+$")

LANDMARK_IDS = [
    "2005.11401",
    "2106.09685",
    "2305.14314",
    "2305.18290",
    "2201.11903",
    "2203.11171",
    "2210.03629",
    "2310.11511",
    "2307.03172",
    "2211.17192",
    "2109.07958",
    "2104.09864",
    "2302.13971",
    "2309.17453",
    "2306.00978",
    "2309.06180",
    "2009.03300",
    "2305.13245",
    "2307.09288",
    "2310.06825",
    "2304.02643",
    "2302.04761",
    "2205.14135",
    "2305.10601",
    "2308.10792",
]

STATIC_WINDOWS = [
    ("202206010000", "202311302359"),
    ("202401010000", "202412312359"),
]

LIVE_WINDOWS = [
    ("202501010000", "202506302359"),
    ("202507010000", "202512312359"),
    ("202601010000", "202606302359"),
    ("202607010000", "202611262359"),
]

STATIC_TARGET_COUNT = 300
LIVE_TARGET_COUNT = 200
LIVE_PER_WINDOW = 50


class ArxivClient:
    """Rate-limited arXiv Atom API client with exponential backoff retries."""

    def __init__(self, min_interval: float = MIN_REQUEST_INTERVAL) -> None:
        self.min_interval = min_interval
        self._last_request_ts = 0.0

    def _wait_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        if self._last_request_ts > 0.0 and elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def fetch_feed(self, params: dict[str, str | int], max_retries: int = 6) -> ET.Element:
        """Query the arXiv API and return the parsed Atom XML root element."""
        url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"
        headers = {"User-Agent": USER_AGENT}
        backoff = 4.0

        for attempt in range(max_retries + 1):
            self._wait_rate_limit()
            req = urllib.request.Request(url, headers=headers)
            try:
                self._last_request_ts = time.monotonic()
                with urllib.request.urlopen(req, timeout=45) as resp:
                    payload = resp.read()
                return ET.fromstring(payload)
            except urllib.error.HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt == max_retries:
                    raise
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                delay = max(float(retry_after), backoff) if retry_after and retry_after.isdigit() else backoff
                print(f"[retry {attempt + 1}/{max_retries}] HTTP {exc.code}; sleeping {delay:.1f}s")
            except (urllib.error.URLError, TimeoutError, OSError, ET.ParseError) as exc:
                if attempt == max_retries:
                    raise
                delay = backoff
                print(f"[retry {attempt + 1}/{max_retries}] {type(exc).__name__}: {exc}; sleeping {delay:.1f}s")

            time.sleep(delay)
            backoff *= 2.0

        raise RuntimeError("Unreachable retry loop exit")


def clean_text(value: str | None) -> str:
    """Collapse internal whitespace to single spaces."""
    return " ".join((value or "").split())


def normalize_id(raw_id: str) -> str:
    """Strip arXiv URL prefix and version suffix (e.g. v1) from an entry ID."""
    stripped = ID_PREFIX_RE.sub("", raw_id.strip())
    return ID_VERSION_RE.sub("", stripped)


def parse_entries(root: ET.Element) -> list[dict[str, object]]:
    """Parse Atom feed entries into normalized corpus records."""
    records: list[dict[str, object]] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        raw_id = entry.findtext("atom:id", default="", namespaces=ATOM_NS)
        paper_id = normalize_id(raw_id)
        title = clean_text(entry.findtext("atom:title", default="", namespaces=ATOM_NS))
        abstract = clean_text(entry.findtext("atom:summary", default="", namespaces=ATOM_NS))
        published = (entry.findtext("atom:published", default="", namespaces=ATOM_NS) or "").strip()
        date = published[:10]
        categories = [
            cat.attrib["term"].strip()
            for cat in entry.findall("atom:category", ATOM_NS)
            if cat.attrib.get("term")
        ]

        if not paper_id or not title or not abstract or len(date) != 10:
            continue
        if TARGET_CATEGORY not in categories:
            continue

        records.append(
            {
                "id": paper_id,
                "title": title,
                "abstract": abstract,
                "date": date,
                "categories": categories,
            }
        )
    return records


def fetch_by_ids(client: ArxivClient, ids: list[str]) -> list[dict[str, object]]:
    """Fetch specific arXiv papers via id_list in batches."""
    results: list[dict[str, object]] = []
    for start in range(0, len(ids), BATCH_SIZE):
        batch = ids[start : start + BATCH_SIZE]
        root = client.fetch_feed({"id_list": ",".join(batch), "max_results": len(batch)})
        results.extend(parse_entries(root))
    return results


def fetch_window(
    client: ArxivClient,
    date_start: str,
    date_end: str,
    target_new: int,
    seen_ids: set[str],
    predicate,
) -> list[dict[str, object]]:
    """Fetch up to target_new unique cs.CL papers matching predicate from a submittedDate window."""
    collected: list[dict[str, object]] = []
    search_query = f"cat:{TARGET_CATEGORY} AND submittedDate:[{date_start} TO {date_end}]"
    start = 0

    while len(collected) < target_new:
        root = client.fetch_feed(
            {
                "search_query": search_query,
                "start": start,
                "max_results": BATCH_SIZE,
                "sortBy": "submittedDate",
                "sortOrder": "ascending",
            }
        )
        raw_entries = root.findall("atom:entry", ATOM_NS)
        if not raw_entries:
            break

        for record in parse_entries(root):
            paper_id = str(record["id"])
            if paper_id in seen_ids or not predicate(record):
                continue
            seen_ids.add(paper_id)
            collected.append(record)
            if len(collected) >= target_new:
                break

        if len(raw_entries) < BATCH_SIZE:
            break
        start += BATCH_SIZE

    return collected


def build_static_corpus(client: ArxivClient) -> tuple[list[dict[str, object]], list[str]]:
    """Build exactly 300 unique pre-cutoff cs.CL papers including landmark papers."""
    is_static = lambda r: str(r["date"]) < CUTOFF_DATE
    seen_ids: set[str] = set()
    static_records: list[dict[str, object]] = []
    landmark_included: list[str] = []

    for record in fetch_by_ids(client, LANDMARK_IDS):
        paper_id = str(record["id"])
        if paper_id not in seen_ids and is_static(record):
            seen_ids.add(paper_id)
            static_records.append(record)
            landmark_included.append(paper_id)

    remaining = STATIC_TARGET_COUNT - len(static_records)
    half_target = remaining // 2
    targets = [half_target, remaining - half_target]

    for (start_dt, end_dt), quota in zip(STATIC_WINDOWS, targets):
        batch = fetch_window(client, start_dt, end_dt, quota, seen_ids, is_static)
        static_records.extend(batch)

    if len(static_records) < STATIC_TARGET_COUNT:
        for start_dt, end_dt in STATIC_WINDOWS:
            need = STATIC_TARGET_COUNT - len(static_records)
            if need <= 0:
                break
            static_records.extend(fetch_window(client, start_dt, end_dt, need, seen_ids, is_static))

    return static_records[:STATIC_TARGET_COUNT], landmark_included


def build_live_corpus(client: ArxivClient, excluded_ids: set[str]) -> list[dict[str, object]]:
    """Build exactly 200 unique post-cutoff cs.CL papers across 4 post-T windows."""
    is_live = lambda r: (
        str(r["date"]) >= CUTOFF_DATE
        and (str(r["id"]).startswith("25") or str(r["id"]).startswith("26"))
    )
    seen_ids = set(excluded_ids)
    live_records: list[dict[str, object]] = []

    for start_dt, end_dt in LIVE_WINDOWS:
        batch = fetch_window(client, start_dt, end_dt, LIVE_PER_WINDOW, seen_ids, is_live)
        live_records.extend(batch)

    if len(live_records) < LIVE_TARGET_COUNT:
        for start_dt, end_dt in LIVE_WINDOWS:
            need = LIVE_TARGET_COUNT - len(live_records)
            if need <= 0:
                break
            live_records.extend(fetch_window(client, start_dt, end_dt, need, seen_ids, is_live))

    return live_records[:LIVE_TARGET_COUNT]


def validate_corpus(static: list[dict[str, object]], live: list[dict[str, object]]) -> int:
    """Assert all split counts, disjointness, date bounds, ID prefixes, and categories."""
    assert len(static) == STATIC_TARGET_COUNT, f"Expected {STATIC_TARGET_COUNT} static, got {len(static)}"
    assert len(live) == LIVE_TARGET_COUNT, f"Expected {LIVE_TARGET_COUNT} live, got {len(live)}"

    static_ids = {str(r["id"]) for r in static}
    live_ids = {str(r["id"]) for r in live}
    overlap = static_ids & live_ids
    assert static_ids.isdisjoint(live_ids), f"ID overlap detected: {overlap}"

    assert all(str(r["date"]) < CUTOFF_DATE for r in static), "Static paper with date >= cutoff"
    assert all(str(r["date"]) >= CUTOFF_DATE for r in live), "Live paper with date < cutoff"
    assert all(str(r["id"]).startswith(("25", "26")) for r in live), "Live paper ID must start with 25 or 26"
    assert all(TARGET_CATEGORY in r["categories"] for r in static + live), "Missing cs.CL category"
    assert all(r["title"] and r["abstract"] for r in static + live), "Empty title or abstract"
    return len(overlap)


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    """Write records to a UTF-8 JSONL file."""
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    corpus_dir = repo_root / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    client = ArxivClient()
    static, landmark_included = build_static_corpus(client)
    live = build_live_corpus(client, {str(r["id"]) for r in static})
    overlap_count = validate_corpus(static, live)

    static_dates = [str(r["date"]) for r in static]
    live_dates = [str(r["date"]) for r in live]
    metadata = {
        "cutoff_T": CUTOFF_DATE,
        "base_model": "Llama-3.2-3B-Instruct (ollama: llama3.2:3b)",
        "base_model_training_cutoff": "2023-12",
        "base_model_release_date": "2024-09-25",
        "cutoff_source_url": "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct",
        "category": TARGET_CATEGORY,
        "static_count": len(static),
        "live_count": len(live),
        "id_overlap_count": overlap_count,
        "static_date_range": [min(static_dates), max(static_dates)],
        "live_date_range": [min(live_dates), max(live_dates)],
        "landmark_ids_included": landmark_included,
    }

    write_jsonl(corpus_dir / "static.jsonl", static)
    write_jsonl(corpus_dir / "live.jsonl", live)
    (corpus_dir / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

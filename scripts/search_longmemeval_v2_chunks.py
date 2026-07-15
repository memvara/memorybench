#!/usr/bin/env python3
"""Search state documents while requesting raw document chunks as context."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "https://api.supermemory.ai"
_lock = threading.Lock()


def log(message: str) -> None:
    with _lock:
        print(message, file=sys.stderr, flush=True)


def load_questions(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def search(
    base_url: str,
    api_key: str,
    container_tag: str,
    question: str,
    limit: int,
    threshold: float,
    retries: int,
) -> dict[str, Any]:
    payload = {
        "q": question,
        "containerTag": container_tag,
        "searchMode": "hybrid",
        "threshold": threshold,
        "limit": limit,
        "include": {
            "relatedMemories": True,
            "documents": True,
            "summaries": True,
            "chunks": True,
        },
    }
    body = json.dumps(payload).encode("utf-8")
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            base_url.rstrip("/") + "/v4/search",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            retryable = exc.code == 429 or exc.code >= 500
            if not retryable or attempt == retries:
                raise RuntimeError(f"HTTP {exc.code}: {raw[:1000]}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries:
                raise RuntimeError(str(exc)) from exc
        time.sleep(min(2**attempt, 16))
    raise RuntimeError("search failed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--container-tag", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()
    api_key = os.getenv("SUPERMEMORY_API_KEY") or os.getenv("SUPERMEMORY_CODEX_API_KEY")
    if not api_key:
        raise RuntimeError("Set SUPERMEMORY_API_KEY or SUPERMEMORY_CODEX_API_KEY")

    data_root = Path(args.data_root).resolve()
    questions = [
        q
        for q in load_questions(data_root / "questions.jsonl")
        if q.get("domain") == "enterprise" and q.get("environment") == "workarena"
    ]
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    log(f"Searching {len(questions)} questions with chunks included")

    def one(question: dict[str, Any]) -> dict[str, Any]:
        try:
            result = search(
                os.getenv("SUPERMEMORY_API_URL", DEFAULT_BASE_URL),
                api_key,
                args.container_tag,
                str(question["question"]),
                args.limit,
                args.threshold,
                args.retries,
            )
            return {
                "status": "ok",
                "questionId": question["id"],
                "question": question["question"],
                "groundTruth": question.get("answer"),
                "questionType": question.get("question_type"),
                "domain": question.get("domain"),
                "environment": question.get("environment"),
                "containerTag": args.container_tag,
                "searchResult": result,
            }
        except Exception as exc:
            return {
                "status": "error",
                "questionId": question["id"],
                "question": question["question"],
                "groundTruth": question.get("answer"),
                "containerTag": args.container_tag,
                "error": str(exc),
            }

    with out.open("a", encoding="utf-8") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(one, question) for question in questions]
            for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                result = future.result()
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                log(f"[{index}/{len(questions)}] {result['questionId']} {result['status']}")

    errors = sum(
        1
        for line in out.read_text(encoding="utf-8").splitlines()
        if '"status": "error"' in line
    )
    log(f"Finished: {out}; errors={errors}")
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"Error: {exc}")
        raise SystemExit(1)

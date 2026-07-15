#!/usr/bin/env python3
"""Search LongMemEval-V2 questions against one already-ingested container."""

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
_print_lock = threading.Lock()


def log(message: str) -> None:
    with _print_lock:
        print(message, file=sys.stderr, flush=True)


def post_search(
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
                raw = response.read().decode("utf-8")
                return json.loads(raw) if raw else {}
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


def load_questions(path: Path, domain: str, environment: str) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("domain") == domain and item.get("environment") == environment:
                questions.append(item)
    return questions


def read_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("status") == "ok" and item.get("questionId"):
                completed.add(str(item["questionId"]))
    return completed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--container-tag", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--domain", default="enterprise")
    parser.add_argument("--environment", default="workarena")
    parser.add_argument("--base-url", default=os.getenv("SUPERMEMORY_API_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=os.getenv("SUPERMEMORY_API_KEY") or os.getenv("SUPERMEMORY_CODEX_API_KEY"))
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if not args.api_key:
        raise RuntimeError("Missing API key; set SUPERMEMORY_API_KEY or SUPERMEMORY_CODEX_API_KEY")

    data_root = Path(args.data_root).resolve()
    questions_path = data_root / "questions.jsonl"
    if not questions_path.exists():
        raise RuntimeError(f"Questions file not found: {questions_path}")

    questions = load_questions(questions_path, args.domain, args.environment)
    if not questions:
        raise RuntimeError("No matching questions found")

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed = set() if args.no_resume else read_completed(out_path)
    pending = [q for q in questions if str(q["id"]) not in completed]
    log(f"Matched {len(questions)} questions; searching {len(pending)}; already complete {len(completed)}")
    log(f"Container: {args.container_tag}; workers: {args.workers}; result limit: {args.limit}")

    output_lock = threading.Lock()
    completed_count = len(completed)

    def run_one(question: dict[str, Any]) -> dict[str, Any]:
        question_id = str(question["id"])
        started = time.time()
        try:
            response = post_search(
                args.base_url,
                args.api_key,
                args.container_tag,
                str(question["question"]),
                args.limit,
                args.threshold,
                args.retries,
            )
            return {
                "status": "ok",
                "questionId": question_id,
                "question": question["question"],
                "groundTruth": question.get("answer"),
                "questionType": question.get("question_type"),
                "domain": question.get("domain"),
                "environment": question.get("environment"),
                "containerTag": args.container_tag,
                "durationMs": round((time.time() - started) * 1000),
                "searchResult": response,
            }
        except Exception as exc:  # keep the batch resumable if one request fails
            return {
                "status": "error",
                "questionId": question_id,
                "question": question["question"],
                "groundTruth": question.get("answer"),
                "containerTag": args.container_tag,
                "durationMs": round((time.time() - started) * 1000),
                "error": str(exc),
            }

    with out_path.open("a", encoding="utf-8") as output:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(run_one, question) for question in pending]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                with output_lock:
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output.flush()
                completed_count += 1
                log(f"[{completed_count}/{len(questions)}] {result['questionId']} {result['status']}")

    errors = sum(1 for line in out_path.read_text(encoding="utf-8").splitlines() if '"status": "error"' in line)
    log(f"Finished. Results: {out_path}; error records in file: {errors}")
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"Error: {exc}")
        raise SystemExit(1)

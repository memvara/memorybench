#!/usr/bin/env python3
"""Create a readable LongMemEval-V2 gold-answer/retrieval report."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def answer_parts(value: Any) -> list[str]:
    return [
        normalize(part)
        for part in re.split(r"\s*(?:,|;|\||\n)\s*", str(value or ""))
        if normalize(part)
    ]


def result_text(result: dict[str, Any]) -> str:
    return normalize(
        " ".join(str(result.get(key, "")) for key in ("memory", "chunk", "content", "summary"))
    )


def project_result(result: dict[str, Any], rank: int, gold: str) -> dict[str, Any]:
    text = result_text(result)
    parts = answer_parts(gold)
    return {
        "rank": rank,
        "id": result.get("id"),
        "similarity": result.get("similarity"),
        "matchesWholeGoldAnswer": bool(normalize(gold) and normalize(gold) in text),
        "containsAllGoldAnswerParts": bool(parts and all(part in text for part in parts)),
        "memory": result.get("memory") or result.get("chunk") or result.get("content"),
        "metadata": result.get("metadata", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-results", type=int, default=5)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    rows: list[dict[str, Any]] = []

    for line in input_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        gold = str(row.get("groundTruth") or "")
        results = (row.get("searchResult") or {}).get("results") or []
        top1 = " ".join(result_text(result) for result in results[:1])
        top30 = " ".join(result_text(result) for result in results[:30])
        parts = answer_parts(gold)
        rows.append(
            {
                "questionId": row.get("questionId"),
                "question": row.get("question"),
                "questionType": row.get("questionType"),
                "domain": row.get("domain"),
                "environment": row.get("environment"),
                "goldAnswer": gold,
                "containerTag": row.get("containerTag"),
                "searchStatus": row.get("status"),
                "durationMs": row.get("durationMs"),
                "retrievalCheck": {
                    "goldWholeStringInTop1": bool(normalize(gold) and normalize(gold) in top1),
                    "allGoldAnswerPartsInTop1": bool(parts and all(part in top1 for part in parts)),
                    "goldWholeStringInTop30": bool(normalize(gold) and normalize(gold) in top30),
                    "allGoldAnswerPartsInTop30": bool(parts and all(part in top30 for part in parts)),
                    "note": "Literal text matching only; this is not semantic judge evaluation.",
                },
                "retrievedEvidenceTopResults": [
                    project_result(result, rank, gold)
                    for rank, result in enumerate(results[: max(1, args.top_results)], start=1)
                ],
            }
        )

    top1_parts = sum(item["retrievalCheck"]["allGoldAnswerPartsInTop1"] for item in rows)
    top30_parts = sum(item["retrievalCheck"]["allGoldAnswerPartsInTop30"] for item in rows)
    report = {
        "report": "LongMemEval-V2 Supermemory retrieval vs gold answers",
        "sourceFile": str(input_path),
        "questionCount": len(rows),
        "summary": {
            "top1AllGoldAnswerPartsMatched": top1_parts,
            "top1AllGoldAnswerPartsRate": round(top1_parts / len(rows), 4) if rows else 0,
            "top30AllGoldAnswerPartsMatched": top30_parts,
            "top30AllGoldAnswerPartsRate": round(top30_parts / len(rows), 4) if rows else 0,
        },
        "questions": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)
    print(f"questions={len(rows)} top1_parts={top1_parts} top30_parts={top30_parts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

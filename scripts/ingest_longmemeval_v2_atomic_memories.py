#!/usr/bin/env python3
"""Ingest one compact, directly searchable memory per LME-V2 state."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable


DEFAULT_BASE_URL = "https://api.supermemory.ai"
MAX_CONTENT_CHARS = 8_000


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr, flush=True)


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def text(value: Any, limit: int | None = None) -> str:
    if value is None:
        return ""
    value = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    value = " ".join(value.split())
    return value[:limit] if limit and len(value) > limit else value


def compact_ui(tree: str) -> str:
    if not tree:
        return ""
    labels: list[str] = []
    seen_labels: set[str] = set()
    role_lines: list[str] = []
    seen_lines: set[str] = set()
    quoted = re.compile(
        r"\b(?:button|link|menuitem|option|combobox|textbox|searchbox|checkbox|heading|gridcell|cell|rowheader|columnheader|StaticText)\s+'([^']+)'|\bvalue='([^']+)'|\bplaceholder='([^']+)'"
    )
    roles = re.compile(
        r"\b(button|link|menuitem|option|combobox|textbox|searchbox|checkbox|heading|gridcell|cell|rowheader|columnheader|StaticText|listitem)\b"
    )
    for raw in tree.splitlines():
        line = " ".join(raw.split())
        if not line:
            continue
        for match in quoted.finditer(line):
            value = next((group for group in match.groups() if group), "").strip()
            if value and value not in seen_labels:
                seen_labels.add(value)
                labels.append(value)
        if roles.search(line) and line not in seen_lines:
            seen_lines.add(line)
            role_lines.append(line)

    sections: list[str] = []
    if labels:
        sections.append("UI labels and values:\n" + "\n".join(f"- {item}" for item in labels))
    if role_lines:
        sections.append("Relevant accessibility lines:\n" + "\n".join(role_lines))
    compact = "\n\n".join(sections)
    return compact[:MAX_CONTENT_CHARS]


def build_memory(
    trajectory: dict[str, Any],
    state: dict[str, Any],
    position: int,
    container_tag: str,
) -> dict[str, Any]:
    states = trajectory["states"]
    state_index = str(state["state_index"])
    previous = states[position - 1] if position else None
    following = states[position + 1] if position + 1 < len(states) else None
    previous_id = f"{trajectory['id']}:{previous['state_index']}" if previous else "none"
    next_id = f"{trajectory['id']}:{following['state_index']}" if following else "none"
    ui = compact_ui(str(state.get("accessibility_tree") or ""))
    content = "\n".join(
        part
        for part in [
            "LongMemEval-V2 atomic trajectory state",
            f"Trajectory ID: {trajectory['id']}",
            f"Domain: {trajectory.get('domain', '')}",
            f"Environment: {trajectory.get('environment', '')}",
            f"Outcome: {trajectory.get('outcome') or 'unknown'}",
            f"Goal: {text(trajectory.get('goal'), 1400)}",
            f"State ID: {trajectory['id']}:{state_index}",
            f"State index: {state_index}",
            f"Previous state ID: {previous_id}",
            f"Next state ID: {next_id}",
            f"Step: {state.get('step', '')}",
            f"URL: {text(state.get('url'), 900)}",
            f"Action: {text(state.get('action'), 1600) or 'null'}",
            f"Thought/observation: {text(state.get('thought'), 1800)}",
            f"Screenshot path: {text(state.get('screenshot'), 500)}",
            f"Accessibility/UI extraction:\n{ui}" if ui else "",
        ]
        if part
    )
    if len(content) > MAX_CONTENT_CHARS:
        # Preserve the compact UI section and bounded context; do not emit a
        # misleading truncation marker into the searchable memory.
        content = content[:MAX_CONTENT_CHARS]

    return {
        "content": content,
        "isStatic": False,
        "customId": f"lme-v2-atomic-{trajectory['id']}-{state_index}",
        "metadata": {
            "benchmark": "longmemeval-v2",
            "tier": "small",
            "containerTag": container_tag,
            "memoryKind": "state_observation",
            "trajectoryId": str(trajectory["id"]),
            "stateId": f"{trajectory['id']}:{state_index}",
            "stateIndex": state_index,
            "previousStateId": previous_id,
            "nextStateId": next_id,
            "domain": str(trajectory.get("domain", "")),
            "environment": str(trajectory.get("environment", "")),
            "outcome": str(trajectory.get("outcome") or "unknown"),
            "url": str(state.get("url") or ""),
            "screenshot": str(state.get("screenshot") or ""),
        },
    }


def post(base_url: str, api_key: str, payload: dict[str, Any], retries: int) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            base_url.rstrip("/") + "/v4/memories",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            if attempt == retries:
                raise RuntimeError(f"HTTP {exc.code}: {raw[:1000]}") from exc
            retry_after = exc.headers.get("Retry-After")
            delay = int(retry_after) if retry_after and retry_after.isdigit() else min(2 ** attempt, 30)
            time.sleep(delay)
    raise RuntimeError("request failed")


def chunks(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--container-tag", required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    api_key = os.getenv("SUPERMEMORY_API_KEY") or os.getenv("SUPERMEMORY_CODEX_API_KEY")
    if not api_key:
        raise RuntimeError("Set SUPERMEMORY_API_KEY or SUPERMEMORY_CODEX_API_KEY")

    root = Path(args.data_root).resolve()
    questions = list(jsonl(root / "questions.jsonl"))
    enterprise = next(q for q in questions if q.get("domain") == "enterprise")
    haystacks = json.loads((root / "haystacks" / "lme_v2_small.json").read_text(encoding="utf-8"))
    trajectory_ids = haystacks[enterprise["id"]]
    trajectories = {t["id"]: t for t in jsonl(root / "trajectories.jsonl")}
    memories = [
        build_memory(trajectories[trajectory_id], state, position, args.container_tag)
        for trajectory_id in trajectory_ids
        for position, state in enumerate(trajectories[trajectory_id]["states"])
    ]
    max_length = max(len(memory["content"]) for memory in memories)
    eprint(f"Prepared {len(memories)} atomic state memories from {len(trajectory_ids)} trajectories")
    eprint(f"Maximum memory content length: {max_length} characters")

    base_url = os.getenv("SUPERMEMORY_API_URL", DEFAULT_BASE_URL)
    responses = []
    for number, batch in enumerate(chunks(memories, max(1, min(args.batch_size, 100))), start=1):
        response = post(
            base_url,
            api_key,
            {"containerTag": args.container_tag, "memories": batch},
            args.retries,
        )
        responses.append(response)
        eprint(f"Inserted batch {number}: {min(number * args.batch_size, len(memories))}/{len(memories)}")

    manifest = {
        "status": "submitted",
        "containerTag": args.container_tag,
        "trajectoryCount": len(trajectory_ids),
        "stateMemoryCount": len(memories),
        "batchCount": len(responses),
        "maxContentChars": max_length,
        "responses": responses,
        "note": "Search only after indexing completes; this endpoint does not return per-memory indexing IDs.",
    }
    manifest_path = Path(args.manifest).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("status", "containerTag", "trajectoryCount", "stateMemoryCount", "batchCount", "maxContentChars")}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        eprint(f"Error: {exc}")
        raise SystemExit(1)

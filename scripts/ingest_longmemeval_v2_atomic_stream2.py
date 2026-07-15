#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path
from typing import Any, Iterable
from ingest_longmemeval_v2_atomic_memories import build_memory, post, eprint

def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--container-tag", required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    api_key = os.getenv("SUPERMEMORY_API_KEY") or os.getenv("SUPERMEMORY_CODEX_API_KEY")
    if not api_key:
        raise RuntimeError("Set SUPERMEMORY_API_KEY or SUPERMEMORY_CODEX_API_KEY")
    root = Path(args.data_root).resolve()
    enterprise = next(q for q in iter_jsonl(root / "questions.jsonl") if q.get("domain") == "enterprise")
    haystacks = json.loads((root / "haystacks" / "lme_v2_small.json").read_text(encoding="utf-8"))
    selected = set(haystacks[enterprise["id"]])
    batch: list[dict[str, Any]] = []
    state_count = trajectory_count = batch_count = max_length = 0
    base_url = os.getenv("SUPERMEMORY_API_URL", "https://api.supermemory.ai")
    for trajectory in iter_jsonl(root / "trajectories.jsonl"):
        if trajectory.get("id") not in selected:
            continue
        trajectory_count += 1
        for position, state in enumerate(trajectory.get("states", [])):
            memory = build_memory(trajectory, state, position, args.container_tag)
            batch.append(memory)
            state_count += 1
            max_length = max(max_length, len(memory["content"]))
            if len(batch) >= min(max(args.batch_size, 1), 100):
                batch_count += 1
                post(base_url, api_key, {"containerTag": args.container_tag, "memories": batch}, args.retries)
                eprint(f"Inserted batch {batch_count}; states submitted: {state_count}")
                batch = []
    if batch:
        batch_count += 1
        post(base_url, api_key, {"containerTag": args.container_tag, "memories": batch}, args.retries)
        eprint(f"Inserted batch {batch_count}; states submitted: {state_count}")
    manifest = {"status":"submitted","containerTag":args.container_tag,"trajectoryCount":trajectory_count,"stateMemoryCount":state_count,"batchCount":batch_count,"maxContentChars":max_length}
    out = Path(args.manifest).resolve(); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

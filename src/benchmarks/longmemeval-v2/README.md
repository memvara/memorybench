# LongMemEval-V2

This benchmark adapter loads LongMemEval-V2 web-agent trajectory data from local files and feeds it into the normal MemoryBench pipeline.

## Data layout

Place the downloaded dataset here:

```text
data/benchmarks/longmemeval-v2/
  questions.jsonl
  trajectories.jsonl
  haystacks/
    lme_v2_small.json
    lme_v2_medium.json
```

If you already downloaded the dataset into the sibling scratch folder used during local testing, copy it from the MemoryBench repo root with:

```powershell
Copy-Item -Recurse ..\memory-bench\data\longmemeval-v2 data\benchmarks\longmemeval-v2
```

The adapter reads `LONGMEMEVAL_V2_TIER` or `LME_V2_TIER` to choose the haystack tier. If neither is set, it uses `small`.

## Ingest one question into Supermemory

PowerShell:

```powershell
$env:SUPERMEMORY_API_KEY = "sm_xxx"
$env:LONGMEMEVAL_V2_TIER = "small"
bun run src/index.ts ingest -p supermemory -b longmemeval-v2 -r lme-v2-01307e07 -q 01307e07 --force
```

Bash:

```bash
SUPERMEMORY_API_KEY=sm_xxx \
LONGMEMEVAL_V2_TIER=small \
bun run src/index.ts ingest \
  -p supermemory \
  -b longmemeval-v2 \
  -r lme-v2-01307e07 \
  -q 01307e07 \
  --force
```

That creates one MemoryBench checkpoint and a Supermemory container tag for the question. The container tag is stored in `data/runs/lme-v2-01307e07/checkpoint.json` and follows:

```text
<questionId>-<dataSourceRunId>
```

## Search after ingest

PowerShell:

```powershell
$env:SUPERMEMORY_API_KEY = "sm_xxx"
$env:LONGMEMEVAL_V2_TIER = "small"
bun run src/index.ts search -r lme-v2-01307e07
```

Bash:

```bash
SUPERMEMORY_API_KEY=sm_xxx \
LONGMEMEVAL_V2_TIER=small \
bun run src/index.ts search -r lme-v2-01307e07
```

Search results are written under:

```text
data/runs/lme-v2-01307e07/results/<questionId>.json
```

## Ingest a smoke subset by count

PowerShell:

```powershell
$env:SUPERMEMORY_API_KEY = "sm_xxx"
$env:LONGMEMEVAL_V2_TIER = "small"
bun run src/index.ts ingest -p supermemory -b longmemeval-v2 -r lme-v2-small-1 -l 1 --force
```

## What gets ingested

For every trajectory in a question's haystack, the adapter creates:

- one trajectory overview session with the goal, outcome, start URL, and action/thought trace
- one state session per trajectory state with URL, action, thought, screenshot path, compact UI labels/options, and a bounded accessibility-tree excerpt

This keeps high-signal UI labels such as dropdown options searchable without requiring MemoryBench to store screenshots in this text-first pipeline.

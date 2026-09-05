# Providers

Memory provider integrations. Each provider implements the `Provider` interface.

## Interface

```typescript
interface Provider {
    name: string
    prompts?: ProviderPrompts
    initialize(config: ProviderConfig): Promise<void>
    ingest(sessions: UnifiedSession[], options: IngestOptions): Promise<IngestResult>
    awaitIndexing(result: IngestResult, containerTag: string): Promise<void>
    search(query: string, options: SearchOptions): Promise<unknown[]>
    clear(containerTag: string): Promise<void>
}
```

## Adding a Provider

1. Create `src/providers/myprovider/index.ts`
2. Implement `Provider` interface
3. Register in `src/providers/index.ts`
4. Add to `ProviderName` type in `src/types/provider.ts`
5. Add config in `src/utils/config.ts`

**Required returns:**
- `initialize()` - Set up client with API key
- `ingest()` - Return `{ documentIds: string[], taskIds?: string[] }`
- `awaitIndexing()` - Wait for async indexing to complete
- `search()` - Return array of results (provider-specific format)
- `clear()` - Delete data by containerTag

## Custom Prompts

Providers can override answer generation and judge prompts via `ProviderPrompts`:

```typescript
interface ProviderPrompts {
    answerPrompt?: string | ((question: string, context: unknown[], questionDate?: string) => string)
    judgePrompt?: (question: string, groundTruth: string, hypothesis: string) => { default: string, [type: string]: string }
}
```

**Answer Prompt:** Transform search results into an LLM prompt. Function receives raw search results.

**Judge Prompt:** Return prompts keyed by question type. Must include `default`. Falls back to built-in prompts if not provided.

Example: See `src/providers/zep/prompts.ts`

## Existing Providers

| Provider | SDK | Notes |
|----------|-----|-------|
| `supermemory` | `supermemory` | Raw JSON sessions |
| `mem0` | `mem0ai` | v2 API with graph |
| `zep` | `@getzep/zep-cloud` | Graph-based, custom prompts |
| `memvara` | REST (`fetch`) | Bitemporal claims plus raw turns; per-question container tag is a memvara `user` scope; `clear` is a scope erasure |

## Running memvara against a local stack

memvara's REST API is memvara-cloud's compose stack. From a memvara-cloud checkout:

```bash
export MEMVARA_CORE_PATH=/path/to/agent-memory        # the core commit under test
export MEMVARA_QUOTA_ENFORCE=0                        # a benchmark ingests far past any plan's allowance
docker compose -f deploy/compose.yaml up -d --build
docker compose -f deploy/compose.yaml run --rm key    # the API key the seed step minted
```

Then in this repository's `.env.local`: `MEMVARA_API_KEY=<that key>` and
`MEMVARA_BASE_URL=http://127.0.0.1:58080`. The provider checks `/v1/whoami` and
`/v1/health` on initialize and logs the memvara version, so a run's log names the
engine build it measured.

## MEMVARA_RANKED and the offline recall screen

`MEMVARA_RANKED=1` adds `ranked: true` to every `/v1/search` request, asking the server's
selector to name the turns that bear on the question instead of returning the
cross-encoder's order as-is. It requires a keyed organisation on the stack above; unset (the
default) is the shipped, unranked read.

Two knobs would each hide the server's ranked order from a run this drives, and a third
would truncate a turn the selector kept -- `memvaraProviderSettings` throws at startup if
any is set alongside `MEMVARA_RANKED=1`:

- `MEMVARA_ROLE_SELECT` drops one role from the server's kept-first list after the fact.
- `MEMVARA_CONTEXT_FILE` replaces the rendered block outright.
- `MEMVARA_TAIL_CHARS` cuts a turn by its position in the returned list, kept or not.

A ranked search's response carries `selection.outcome` (`applied`, `fallback`,
`unconfigured`, `disabled`, `key_rejected`) and `selection.candidates`/`kept` counts, which
the provider carries into the search checkpoint as one extra `{kind: "selection", ...}`
context item (nothing renders it into the prompt). Each turn's own `selected` field is
`true` (the model kept it), `false` (it saw the turn and did not), or `null` (it never
evaluated the turn -- past the selector's `top_n`, or the whole call served unranked).

Before spending on a judged run, score a `MEMVARA_RANKED=1` search-phase checkpoint against
LongMemEval's own `has_answer` labels:

```bash
bun run src/scripts/score-ranked.ts <runId> [datasetPath]
```

It prints the same pair of numbers `local/compress/extract.py` prints for its own candidate
list: gold-turn recall and the non-gold keep rate, over the turns the selector actually
evaluated (`selected: true` or `false` -- a `null` turn was never in the selector's
candidate list, so it is not counted either way).

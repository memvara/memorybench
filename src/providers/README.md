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

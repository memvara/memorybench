// src/providers/memvara/index.ts
import type {
  Provider,
  ProviderConfig,
  IngestOptions,
  IngestResult,
  SearchOptions,
  IndexingProgressCallback,
} from "../../types/provider"
import type { UnifiedSession } from "../../types/unified"
import { logger } from "../../utils/logger"
import { MemvaraClient } from "./client"
import type { MemvaraHit, MemvaraMessage, MemvaraSelection } from "./client"
import { memvaraProviderSettings, ranked, searchK } from "./env"
import { MEMVARA_PROMPTS } from "./prompts"
import type { MemvaraContextItem, MemvaraContextSelection } from "./prompts"

const DEFAULT_BASE_URL = "http://127.0.0.1:58080"

/** What memvara is asked for on every search, beyond the depth `searchK` decides. The
 *  orchestrator's `limit: 10` and `threshold: 0.3` are ignored here for the same reason the
 *  shipped providers ignore them, and memvara's score is not on the scale that threshold was
 *  set for. No floor: this measures the ranking as shipped. */
const SEARCH_MIN_SCORE = 0

export class MemvaraProvider implements Provider {
  name = "memvara"
  prompts = MEMVARA_PROMPTS
  concurrency = {
    default: 4,
    ingest: 6,
    indexing: 8,
    search: 4,
    answer: 8,
    evaluate: 8,
  }
  private client: MemvaraClient | null = null
  private readonly clientFactory: (config: ProviderConfig) => MemvaraClient

  constructor(clientFactory?: (config: ProviderConfig) => MemvaraClient) {
    this.clientFactory =
      clientFactory ??
      ((config) =>
        new MemvaraClient({ apiKey: config.apiKey, baseUrl: config.baseUrl || DEFAULT_BASE_URL }))
  }

  async initialize(config: ProviderConfig): Promise<void> {
    if (!config.apiKey) throw new Error("MEMVARA_API_KEY is not set")
    const client = this.clientFactory(config)
    const who = await client.whoami()
    if (who.read_only) {
      throw new Error(`memvara credential ${who.token_id} is read-only; the benchmark writes`)
    }
    const health = await client.health()
    this.client = client
    // Every knob, in the run's own log. A score that cannot be attributed to a
    // configuration is a run that has to be repeated, and reading them here also means a
    // typo in any of the knobs that check their value throws at startup rather than at the
    // first question that happens to touch it.
    logger.info(
      `Initialized memvara provider: ${config.baseUrl || DEFAULT_BASE_URL}, tenant ${who.scope.tenant}, ` +
        `privilege ${who.effective_privilege}, memvara ${health.memvara_version}, ` +
        `settings ${JSON.stringify(memvaraProviderSettings())}`
    )
  }

  async ingest(sessions: UnifiedSession[], options: IngestOptions): Promise<IngestResult> {
    const client = this.ready()
    const documentIds: string[] = []
    for (const session of sessions) {
      if (session.messages.length === 0) continue
      const sessionDate =
        typeof session.metadata?.date === "string" ? session.metadata.date : undefined
      const messages: MemvaraMessage[] = session.messages.map((m) => ({
        role: m.role,
        content: m.content,
        ...(m.timestamp || sessionDate ? { ts: m.timestamp || sessionDate } : {}),
        metadata: { sessionId: session.sessionId },
      }))
      // One key per logical write. A retry after a timeout replays the first receipt
      // instead of storing the session twice.
      const idempotencyKey = `${options.containerTag}:${session.sessionId}`
      const receipt = await client.addMemories(
        options.containerTag,
        { messages, ...(sessionDate ? { ts: sessionDate } : {}) },
        idempotencyKey
      )
      documentIds.push(...receipt.episode_ids)
      logger.debug(
        `Ingested ${session.sessionId}: ${receipt.episode_ids.length} turns, ${receipt.added.length} memories added`
      )
    }
    return { documentIds }
  }

  /** Memvara's write is synchronous: the receipt comes back after the turns are stored,
   *  embedded, indexed and extracted. There is nothing to wait for; one stats read
   *  confirms the scope is populated and puts the count in the log. */
  async awaitIndexing(
    result: IngestResult,
    containerTag: string,
    onProgress?: IndexingProgressCallback
  ): Promise<void> {
    const client = this.ready()
    const stats = await client.stats(containerTag)
    logger.debug(
      `Scope ${containerTag}: ${stats.visible} memories visible, counts ${JSON.stringify(stats.tenant_counts)}`
    )
    onProgress?.({
      completedIds: [...result.documentIds],
      failedIds: [],
      total: result.documentIds.length,
    })
  }

  async search(query: string, options: SearchOptions): Promise<unknown[]> {
    const client = this.ready()
    const response = await client.search(options.containerTag, {
      query,
      k: searchK(),
      min_score: SEARCH_MIN_SCORE,
      include_episodes: true,
      ...(ranked() ? { ranked: true } : {}),
    })
    const items: MemvaraContextItem[] = response.results.map(toContextItem)
    if (response.selection) items.push(toSelectionItem(response.selection))
    return items
  }

  async clear(containerTag: string): Promise<void> {
    const client = this.ready()
    const out = await client.eraseUser(containerTag)
    logger.debug(`Erased scope ${containerTag}: ${JSON.stringify(out.counts)}`)
  }

  private ready(): MemvaraClient {
    if (!this.client) throw new Error("Provider not initialized")
    return this.client
  }
}

/** memvara's hit, flattened to what the prompt renders. Nothing is dropped or reordered. */
function toContextItem(hit: MemvaraHit): MemvaraContextItem {
  if (hit.kind === "claim") {
    const m = hit.memory
    return {
      kind: "memory",
      text: m.text,
      subject: m.subject,
      predicate: m.predicate,
      object: m.object,
      state: m.state,
      valid_from: m.valid_time.valid_from,
      valid_to: m.valid_time.valid_to,
      recorded_at: m.transaction_time.recorded_at,
      invalidated_at: m.transaction_time.invalidated_at,
      score: hit.score,
      sources: m.source_ids,
    }
  }
  const e = hit.episode
  return {
    kind: "turn",
    role: e.role,
    content: e.content,
    ts: e.ts,
    score: hit.score,
    ...(hit.ranking?.selected !== undefined ? { selected: hit.ranking.selected } : {}),
    ...(hit.ranking?.span !== undefined ? { span: hit.ranking.span } : {}),
  }
}

/** The response's own outcome, carried into the context array as one extra item -- see
 *  `MemvaraContextSelection`. */
function toSelectionItem(s: MemvaraSelection): MemvaraContextSelection {
  return {
    kind: "selection",
    outcome: s.outcome,
    ...(s.reason !== undefined ? { reason: s.reason } : {}),
    ...(s.status !== undefined ? { status: s.status } : {}),
    candidates: s.candidates,
    kept: s.kept,
  }
}

export default MemvaraProvider

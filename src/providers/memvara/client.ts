/**
 * Memvara's REST API, the part MemoryBench needs. HTTP only: this file knows nothing
 * about sessions, questions or prompts.
 *
 * Scope: the credential is bound to a tenant, and every route narrows it with a `user`
 * query parameter. MemoryBench's container tag travels as that `user`, so one
 * question's haystack is one memvara user and a request under one tag cannot see
 * another tag's memories.
 */

export interface MemvaraMessage {
  role: string
  content: string
  ts?: string
  metadata?: Record<string, unknown>
}

export interface MemvaraAddRequest {
  messages: MemvaraMessage[]
  ts?: string
}

export interface MemvaraWriteReceipt {
  episode_ids: string[]
  added: unknown[]
  invalidated: unknown[]
  reinforced: unknown[]
  skipped: number
  unextracted: number
  llm_calls: number
  latency_ms: number
  deferred: boolean
  note: string | null
}

export interface MemvaraSearchRequest {
  query: string
  k: number
  min_score: number
  include_episodes: boolean
}

export interface MemvaraMemory {
  id: string
  text: string
  subject: string
  predicate: string
  object: string
  memory_type: string
  state: string
  valid_time: { valid_from: string; valid_to: string | null }
  transaction_time: { recorded_at: string; invalidated_at: string | null }
  confidence: number
  salience: number
  source_ids: string[]
}

export interface MemvaraEpisode {
  id: string
  role: string
  ts: string
  content: string
}

export type MemvaraHit =
  | { kind: "claim"; score: number; memory: MemvaraMemory }
  | { kind: "episode"; score: number; episode: MemvaraEpisode }

export interface MemvaraSearchResponse {
  count: number
  results: MemvaraHit[]
}

export interface MemvaraWhoAmI {
  token_id: string
  scope: { tenant: string; user?: string | null; agent?: string | null; session?: string | null }
  granted_privilege: string
  effective_privilege: string
  read_only: boolean
}

export interface MemvaraHealth {
  status: string
  memvara_version: string
}

export interface MemvaraStats {
  scope: unknown
  visible: number
  tenant_counts: Record<string, number>
  extractor: string
  read_only: boolean
}

export interface MemvaraErasure {
  target: string
  memory_id: string | null
  scope: unknown
  erased: boolean
  counts: Record<string, number> | null
}

export class MemvaraHttpError extends Error {
  constructor(
    public readonly status: number,
    public readonly body: string,
    message: string
  ) {
    super(message)
    this.name = "MemvaraHttpError"
  }
}

export interface MemvaraClientOptions {
  baseUrl: string
  apiKey: string
  fetchImpl?: typeof fetch
  maxAttempts?: number
  baseDelayMs?: number
  sleep?: (ms: number) => Promise<void>
}

// 429 is the API's rate limit and quota answer; the 5xx family is the stack being
// restarted or Postgres being busy. 4xx other than 429 is a request that will not get
// better by being repeated.
const RETRYABLE = new Set([408, 425, 429, 500, 502, 503, 504])
const MAX_DELAY_MS = 5000

interface RequestOptions {
  user?: string
  body?: unknown
  headers?: Record<string, string>
}

export class MemvaraClient {
  private readonly baseUrl: string
  private readonly apiKey: string
  private readonly fetchImpl: typeof fetch
  private readonly maxAttempts: number
  private readonly baseDelayMs: number
  private readonly sleep: (ms: number) => Promise<void>

  constructor(opts: MemvaraClientOptions) {
    this.baseUrl = opts.baseUrl.replace(/\/+$/, "")
    this.apiKey = opts.apiKey
    this.fetchImpl = opts.fetchImpl ?? fetch
    this.maxAttempts = opts.maxAttempts ?? 5
    this.baseDelayMs = opts.baseDelayMs ?? 500
    this.sleep = opts.sleep ?? ((ms) => new Promise((r) => setTimeout(r, ms)))
  }

  whoami(): Promise<MemvaraWhoAmI> {
    return this.request("GET", "/v1/whoami", {})
  }

  health(): Promise<MemvaraHealth> {
    return this.request("GET", "/v1/health", {})
  }

  addMemories(
    user: string,
    body: MemvaraAddRequest,
    idempotencyKey: string
  ): Promise<MemvaraWriteReceipt> {
    return this.request("POST", "/v1/memories", {
      user,
      body,
      headers: { "Idempotency-Key": idempotencyKey },
    })
  }

  search(user: string, body: MemvaraSearchRequest): Promise<MemvaraSearchResponse> {
    return this.request("POST", "/v1/search", { user, body })
  }

  stats(user: string): Promise<MemvaraStats> {
    return this.request("GET", "/v1/stats", { user })
  }

  /** Erase everything under one user scope: claims, turns, vectors, index entries.
   *  `confirm_tenant` is deliberately never sent, so a request that lost its user
   *  is refused by the API rather than erasing the tenant. */
  eraseUser(user: string): Promise<MemvaraErasure> {
    return this.request("POST", "/v1/erasures", { user, body: { scope: { user } } })
  }

  private async request<T>(method: "GET" | "POST", path: string, opts: RequestOptions): Promise<T> {
    const url =
      opts.user === undefined
        ? `${this.baseUrl}${path}`
        : `${this.baseUrl}${path}?user=${encodeURIComponent(opts.user)}`
    const headers: Record<string, string> = {
      Authorization: `Bearer ${this.apiKey}`,
      Accept: "application/json",
      ...(opts.body !== undefined ? { "Content-Type": "application/json" } : {}),
      ...(opts.headers ?? {}),
    }
    const init: RequestInit = {
      method,
      headers,
      ...(opts.body !== undefined ? { body: JSON.stringify(opts.body) } : {}),
    }

    let lastError: Error | null = null
    for (let attempt = 1; attempt <= this.maxAttempts; attempt++) {
      let response: Response
      try {
        response = await this.fetchImpl(url, init)
      } catch (e) {
        lastError = e instanceof Error ? e : new Error(String(e))
        if (attempt < this.maxAttempts) await this.sleep(this.delay(attempt))
        continue
      }
      if (response.ok) {
        return (await response.json()) as T
      }
      const text = await response.text()
      lastError = new MemvaraHttpError(
        response.status,
        text,
        `${method} ${path} -> ${response.status}: ${text.slice(0, 300)}`
      )
      if (!RETRYABLE.has(response.status)) throw lastError
      if (attempt < this.maxAttempts) {
        const retryAfter = Number(response.headers.get("Retry-After"))
        await this.sleep(
          Number.isFinite(retryAfter) && retryAfter > 0
            ? Math.min(retryAfter * 1000, MAX_DELAY_MS)
            : this.delay(attempt)
        )
      }
    }
    throw lastError ?? new Error(`${method} ${path}: no attempts made`)
  }

  private delay(attempt: number): number {
    return Math.min(this.baseDelayMs * 2 ** (attempt - 1), MAX_DELAY_MS)
  }
}

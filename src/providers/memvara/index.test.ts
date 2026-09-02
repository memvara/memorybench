// src/providers/memvara/index.test.ts
import { describe, expect, test } from "bun:test"
import { MemvaraProvider } from "./index"
import { createProvider, getAvailableProviders } from "../index"
import { getProviderConfig } from "../../utils/config"
import type { MemvaraClient, MemvaraAddRequest, MemvaraSearchRequest } from "./client"
import type { UnifiedSession } from "../../types/unified"

type Recorded = { method: string; args: unknown[] }

function fakeClient(overrides: Partial<Record<keyof MemvaraClient, unknown>> = {}) {
  const calls: Recorded[] = []
  const record =
    (method: string) =>
    (...args: unknown[]) => {
      calls.push({ method, args })
      const fn = overrides[method as keyof MemvaraClient]
      return typeof fn === "function" ? (fn as (...a: unknown[]) => unknown)(...args) : undefined
    }
  const client = {
    whoami: record("whoami"),
    health: record("health"),
    addMemories: record("addMemories"),
    search: record("search"),
    stats: record("stats"),
    eraseUser: record("eraseUser"),
  } as unknown as MemvaraClient
  return { client, calls }
}

const session: UnifiedSession = {
  sessionId: "q1-session-0",
  messages: [
    { role: "user", content: "I moved to Lisbon last week!" },
    { role: "assistant", content: "Congratulations." },
  ],
  metadata: {
    date: "2023-05-20T02:21:00.000Z",
    formattedDate: "Saturday, May 20, 2023 at 2:21 AM",
  },
}

async function initialised(overrides: Partial<Record<keyof MemvaraClient, unknown>> = {}) {
  const fake = fakeClient({
    whoami: async () => ({
      token_id: "t",
      scope: { tenant: "prj_x" },
      granted_privilege: "admin",
      effective_privilege: "admin",
      read_only: false,
    }),
    health: async () => ({ status: "ok", memvara_version: "0.9.0" }),
    ...overrides,
  })
  const provider = new MemvaraProvider(() => fake.client)
  await provider.initialize({ apiKey: "k", baseUrl: "http://api.test" })
  return { provider, ...fake }
}

describe("MemvaraProvider", () => {
  test("is named memvara, ships the prompt, and declares modest concurrency", () => {
    const p = new MemvaraProvider()
    expect(p.name).toBe("memvara")
    expect(typeof p.prompts?.answerPrompt).toBe("function")
    expect(p.concurrency).toEqual({
      default: 4,
      ingest: 6,
      indexing: 8,
      search: 4,
      answer: 8,
      evaluate: 8,
    })
  })

  test("initialize checks whoami and health, and refuses a read-only credential", async () => {
    const { calls } = await initialised()
    expect(calls.map((c) => c.method)).toEqual(["whoami", "health"])
    const fake = fakeClient({
      whoami: async () => ({
        token_id: "t",
        scope: { tenant: "prj_x" },
        granted_privilege: "read",
        effective_privilege: "read",
        read_only: true,
      }),
      health: async () => ({ status: "ok", memvara_version: "0.9.0" }),
    })
    const p = new MemvaraProvider(() => fake.client)
    await expect(p.initialize({ apiKey: "k" })).rejects.toThrow(/read-only/)
  })

  test("methods refuse before initialize", async () => {
    const p = new MemvaraProvider()
    await expect(p.search("q", { containerTag: "c" })).rejects.toThrow(/not initialized/)
  })

  test("ingest writes one request per session with the session date on every turn and an idempotency key", async () => {
    const { provider, calls } = await initialised({
      addMemories: async () => ({
        episode_ids: ["ep_a", "ep_b"],
        added: [],
        invalidated: [],
        reinforced: [],
        skipped: 0,
        unextracted: 0,
        llm_calls: 0,
        latency_ms: 1,
        deferred: false,
        note: null,
      }),
    })
    const out = await provider.ingest([session], { containerTag: "q1-run7" })
    expect(out).toEqual({ documentIds: ["ep_a", "ep_b"] })
    const add = calls.find((c) => c.method === "addMemories")!
    const [user, body, key] = add.args as [string, MemvaraAddRequest, string]
    expect(user).toBe("q1-run7")
    expect(key).toBe("q1-run7:q1-session-0")
    expect(body.ts).toBe("2023-05-20T02:21:00.000Z")
    expect(body.messages).toEqual([
      {
        role: "user",
        content: "I moved to Lisbon last week!",
        ts: "2023-05-20T02:21:00.000Z",
        metadata: { sessionId: "q1-session-0" },
      },
      {
        role: "assistant",
        content: "Congratulations.",
        ts: "2023-05-20T02:21:00.000Z",
        metadata: { sessionId: "q1-session-0" },
      },
    ])
  })

  test("ingest prefers a per-message timestamp when the benchmark gives one", async () => {
    const { provider, calls } = await initialised({
      addMemories: async () => ({
        episode_ids: ["ep_a"],
        added: [],
        invalidated: [],
        reinforced: [],
        skipped: 0,
        unextracted: 0,
        llm_calls: 0,
        latency_ms: 1,
        deferred: false,
        note: null,
      }),
    })
    const s: UnifiedSession = {
      sessionId: "s",
      messages: [{ role: "user", content: "x", timestamp: "2024-01-01T00:00:00.000Z" }],
      metadata: { date: "2023-05-20T02:21:00.000Z" },
    }
    await provider.ingest([s], { containerTag: "c" })
    const body = calls.find((c) => c.method === "addMemories")!.args[1] as MemvaraAddRequest
    expect(body.messages[0].ts).toBe("2024-01-01T00:00:00.000Z")
  })

  test("ingest skips a session with no messages", async () => {
    const { provider, calls } = await initialised()
    const out = await provider.ingest([{ sessionId: "empty", messages: [] }], { containerTag: "c" })
    expect(out).toEqual({ documentIds: [] })
    expect(calls.some((c) => c.method === "addMemories")).toBe(false)
  })

  test("awaitIndexing resolves, reads stats, and reports every id complete", async () => {
    const { provider, calls } = await initialised({
      stats: async () => ({
        scope: {},
        visible: 3,
        tenant_counts: { episodes: 2 },
        extractor: "fast/v1",
        read_only: false,
      }),
    })
    const seen: unknown[] = []
    await provider.awaitIndexing({ documentIds: ["ep_a", "ep_b"] }, "c", (p) => seen.push(p))
    expect(calls.find((c) => c.method === "stats")!.args[0]).toBe("c")
    expect(seen).toEqual([{ completedIds: ["ep_a", "ep_b"], failedIds: [], total: 2 }])
  })

  test("search asks for 30 with episodes and no floor, and returns plain memory and turn objects in order", async () => {
    const { provider, calls } = await initialised({
      search: async () => ({
        count: 2,
        results: [
          {
            kind: "claim",
            score: 0.61,
            ranking: {},
            memory: {
              id: "cl_1",
              text: "user lives in Lisbon",
              subject: "user",
              predicate: "lives_in",
              object: "Lisbon",
              memory_type: "semantic",
              state: "live",
              valid_time: { valid_from: "2023-05-20T02:21:00+00:00", valid_to: null },
              transaction_time: { recorded_at: "2023-05-20T02:21:00+00:00", invalidated_at: null },
              confidence: 1,
              salience: 1,
              source_ids: ["ep_a"],
            },
          },
          {
            kind: "episode",
            score: 0.44,
            ranking: {},
            episode: {
              id: "ep_a",
              role: "user",
              ts: "2023-05-20T02:21:00+00:00",
              content: "I moved to Lisbon last week!",
            },
          },
        ],
      }),
    })
    const out = await provider.search("where do I live", {
      containerTag: "q1-run7",
      limit: 10,
      threshold: 0.3,
    })
    const [user, body] = calls.find((c) => c.method === "search")!.args as [
      string,
      MemvaraSearchRequest,
    ]
    expect(user).toBe("q1-run7")
    expect(body).toEqual({ query: "where do I live", k: 30, min_score: 0, include_episodes: true })
    expect(out).toEqual([
      {
        kind: "memory",
        text: "user lives in Lisbon",
        subject: "user",
        predicate: "lives_in",
        object: "Lisbon",
        state: "live",
        valid_from: "2023-05-20T02:21:00+00:00",
        valid_to: null,
        recorded_at: "2023-05-20T02:21:00+00:00",
        invalidated_at: null,
        score: 0.61,
        sources: ["ep_a"],
      },
      {
        kind: "turn",
        role: "user",
        content: "I moved to Lisbon last week!",
        ts: "2023-05-20T02:21:00+00:00",
        score: 0.44,
      },
    ])
  })

  test("clear erases the user scope", async () => {
    const { provider, calls } = await initialised({
      eraseUser: async () => ({
        target: "scope",
        memory_id: null,
        scope: { user: "c" },
        erased: true,
        counts: { claims: 1 },
      }),
    })
    await provider.clear("c")
    expect(calls.find((c) => c.method === "eraseUser")!.args).toEqual(["c"])
  })
})

describe("registration", () => {
  test("memvara is a known provider", () => {
    expect(getAvailableProviders()).toContain("memvara")
    expect(createProvider("memvara").name).toBe("memvara")
  })

  test("config reads the memvara key and base URL, with the local stack as the default URL", () => {
    const saved = { key: process.env.MEMVARA_API_KEY, url: process.env.MEMVARA_BASE_URL }
    process.env.MEMVARA_API_KEY = "abc"
    delete process.env.MEMVARA_BASE_URL
    try {
      // config.ts reads the environment at import time, so re-import it fresh.
      delete require.cache[require.resolve("../../utils/config")]
      const { getProviderConfig: fresh } =
        require("../../utils/config") as typeof import("../../utils/config")
      expect(fresh("memvara")).toEqual({ apiKey: "abc", baseUrl: "http://127.0.0.1:58080" })
    } finally {
      if (saved.key !== undefined) process.env.MEMVARA_API_KEY = saved.key
      else delete process.env.MEMVARA_API_KEY
      if (saved.url !== undefined) process.env.MEMVARA_BASE_URL = saved.url
    }
  })

  test("getProviderConfig knows memvara", () => {
    expect(() => getProviderConfig("memvara")).not.toThrow()
  })
})

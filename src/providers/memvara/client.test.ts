import { describe, expect, test } from "bun:test"
import { MemvaraClient, MemvaraHttpError } from "./client"

type Call = { url: string; init: RequestInit }

function fakeFetch(
  responses: Array<{ status: number; body: unknown; headers?: Record<string, string> } | Error>
) {
  const calls: Call[] = []
  const impl = (async (url: string | URL | Request, init?: RequestInit) => {
    calls.push({ url: String(url), init: init ?? {} })
    const next = responses.shift()
    if (next === undefined) throw new Error("fakeFetch: no response scripted")
    if (next instanceof Error) throw next
    return new Response(JSON.stringify(next.body), {
      status: next.status,
      headers: { "content-type": "application/json", ...(next.headers ?? {}) },
    })
  }) as unknown as typeof fetch
  return { impl, calls }
}

const noSleep = async () => {}

describe("MemvaraClient", () => {
  test("sends the bearer token and the user scope as a query parameter", async () => {
    const f = fakeFetch([{ status: 200, body: { count: 0, results: [] } }])
    const client = new MemvaraClient({
      baseUrl: "http://api.test",
      apiKey: "k1",
      fetchImpl: f.impl,
      sleep: noSleep,
    })
    await client.search("q-1", { query: "hello", k: 30, min_score: 0, include_episodes: true })
    expect(f.calls).toHaveLength(1)
    expect(f.calls[0].url).toBe("http://api.test/v1/search?user=q-1")
    const headers = f.calls[0].init.headers as Record<string, string>
    expect(headers["Authorization"]).toBe("Bearer k1")
    expect(headers["Content-Type"]).toBe("application/json")
    expect(JSON.parse(String(f.calls[0].init.body))).toEqual({
      query: "hello",
      k: 30,
      min_score: 0,
      include_episodes: true,
    })
  })

  test("search returns the response's selection object, not only its results", async () => {
    const selection = { outcome: "applied", candidates: 40, kept: 6 }
    const f = fakeFetch([{ status: 200, body: { count: 0, results: [], selection } }])
    const client = new MemvaraClient({
      baseUrl: "http://api.test",
      apiKey: "k1",
      fetchImpl: f.impl,
      sleep: noSleep,
    })
    const out = await client.search("q-1", {
      query: "hello",
      k: 30,
      min_score: 0,
      include_episodes: true,
      ranked: true,
    })
    expect(out.selection).toEqual(selection)
  })

  test("addMemories sends the idempotency key and the ts on the request", async () => {
    const receipt = {
      episode_ids: ["ep_1", "ep_2"],
      added: [],
      invalidated: [],
      reinforced: [],
      skipped: 0,
      unextracted: 0,
      llm_calls: 0,
      latency_ms: 1,
      deferred: false,
      note: null,
    }
    const f = fakeFetch([{ status: 200, body: receipt }])
    const client = new MemvaraClient({
      baseUrl: "http://api.test",
      apiKey: "k1",
      fetchImpl: f.impl,
      sleep: noSleep,
    })
    const out = await client.addMemories(
      "q-1",
      {
        messages: [{ role: "user", content: "hi", ts: "2023-05-20T02:21:00.000Z" }],
        ts: "2023-05-20T02:21:00.000Z",
      },
      "q-1:s-0"
    )
    expect(out.episode_ids).toEqual(["ep_1", "ep_2"])
    expect(f.calls[0].url).toBe("http://api.test/v1/memories?user=q-1")
    expect((f.calls[0].init.headers as Record<string, string>)["Idempotency-Key"]).toBe("q-1:s-0")
  })

  test("retries a 503 with backoff and then succeeds", async () => {
    const slept: number[] = []
    const f = fakeFetch([
      { status: 503, body: { error: { code: "unavailable", message: "x" } } },
      { status: 200, body: { status: "ok", memvara_version: "1" } },
    ])
    const client = new MemvaraClient({
      baseUrl: "http://api.test",
      apiKey: "k1",
      fetchImpl: f.impl,
      baseDelayMs: 100,
      sleep: async (ms) => {
        slept.push(ms)
      },
    })
    const out = await client.health()
    expect(out.memvara_version).toBe("1")
    expect(f.calls).toHaveLength(2)
    expect(slept).toEqual([100])
  })

  test("retries a thrown network error", async () => {
    const f = fakeFetch([
      new Error("ECONNRESET"),
      { status: 200, body: { status: "ok", memvara_version: "1" } },
    ])
    const client = new MemvaraClient({
      baseUrl: "http://api.test",
      apiKey: "k1",
      fetchImpl: f.impl,
      sleep: noSleep,
    })
    await expect(client.health()).resolves.toEqual({ status: "ok", memvara_version: "1" })
    expect(f.calls).toHaveLength(2)
  })

  test("gives up after maxAttempts and reports the last status", async () => {
    const f = fakeFetch([
      { status: 502, body: "" },
      { status: 502, body: "" },
      { status: 502, body: "" },
    ])
    const client = new MemvaraClient({
      baseUrl: "http://api.test",
      apiKey: "k1",
      fetchImpl: f.impl,
      maxAttempts: 3,
      sleep: noSleep,
    })
    await expect(client.health()).rejects.toBeInstanceOf(MemvaraHttpError)
    expect(f.calls).toHaveLength(3)
  })

  test("does not retry a 4xx and surfaces the body", async () => {
    const f = fakeFetch([{ status: 400, body: { error: { code: "bad_request", message: "no" } } }])
    const client = new MemvaraClient({
      baseUrl: "http://api.test",
      apiKey: "k1",
      fetchImpl: f.impl,
      sleep: noSleep,
    })
    const err = await client.stats("q-1").catch((e) => e)
    expect(err).toBeInstanceOf(MemvaraHttpError)
    expect((err as MemvaraHttpError).status).toBe(400)
    expect((err as MemvaraHttpError).body).toContain("bad_request")
    expect(f.calls).toHaveLength(1)
  })

  test("eraseUser posts the scope and never confirm_tenant", async () => {
    const f = fakeFetch([
      {
        status: 200,
        body: {
          target: "scope",
          memory_id: null,
          scope: { user: "q-1" },
          erased: true,
          counts: { claims: 3 },
        },
      },
    ])
    const client = new MemvaraClient({
      baseUrl: "http://api.test",
      apiKey: "k1",
      fetchImpl: f.impl,
      sleep: noSleep,
    })
    const out = await client.eraseUser("q-1")
    expect(out.erased).toBe(true)
    expect(f.calls[0].url).toBe("http://api.test/v1/erasures?user=q-1")
    expect(JSON.parse(String(f.calls[0].init.body))).toEqual({ scope: { user: "q-1" } })
  })

  test("strips a trailing slash from the base URL", async () => {
    const f = fakeFetch([{ status: 200, body: { status: "ok", memvara_version: "1" } }])
    const client = new MemvaraClient({
      baseUrl: "http://api.test/",
      apiKey: "k1",
      fetchImpl: f.impl,
      sleep: noSleep,
    })
    await client.health()
    expect(f.calls[0].url).toBe("http://api.test/v1/health")
  })

  test("a 429 with a Retry-After header is retried after that many seconds, then succeeds", async () => {
    const slept: number[] = []
    const f = fakeFetch([
      {
        status: 429,
        body: { error: { code: "rate_limited", message: "x" } },
        headers: { "Retry-After": "2" },
      },
      { status: 200, body: { status: "ok", memvara_version: "1" } },
    ])
    const client = new MemvaraClient({
      baseUrl: "http://api.test",
      apiKey: "k1",
      fetchImpl: f.impl,
      sleep: async (ms) => {
        slept.push(ms)
      },
    })
    const out = await client.health()
    expect(out.memvara_version).toBe("1")
    expect(f.calls).toHaveLength(2)
    expect(slept).toEqual([2000])
  })

  test("a 429 with retry_after in the body and no header is retried after that many seconds", async () => {
    const slept: number[] = []
    const f = fakeFetch([
      { status: 429, body: { error: { detail: { retry_after: 1 } } } },
      { status: 200, body: { status: "ok", memvara_version: "1" } },
    ])
    const client = new MemvaraClient({
      baseUrl: "http://api.test",
      apiKey: "k1",
      fetchImpl: f.impl,
      sleep: async (ms) => {
        slept.push(ms)
      },
    })
    const out = await client.health()
    expect(out.memvara_version).toBe("1")
    expect(f.calls).toHaveLength(2)
    expect(slept).toEqual([1000])
  })

  test("429s ride their own, larger budget instead of maxAttempts", async () => {
    const f = fakeFetch([
      { status: 429, body: { error: { code: "rate_limited", message: "x" } } },
      { status: 429, body: { error: { code: "rate_limited", message: "x" } } },
      { status: 429, body: { error: { code: "rate_limited", message: "x" } } },
      { status: 200, body: { status: "ok", memvara_version: "1" } },
    ])
    const client = new MemvaraClient({
      baseUrl: "http://api.test",
      apiKey: "k1",
      fetchImpl: f.impl,
      maxAttempts: 2,
      rateLimitAttempts: 4,
      sleep: noSleep,
    })
    const out = await client.health()
    expect(out.memvara_version).toBe("1")
    expect(f.calls).toHaveLength(4)
  })

  test("a non-429 retryable status still gives up at maxAttempts", async () => {
    const f = fakeFetch([
      { status: 503, body: "" },
      { status: 503, body: "" },
    ])
    const client = new MemvaraClient({
      baseUrl: "http://api.test",
      apiKey: "k1",
      fetchImpl: f.impl,
      maxAttempts: 2,
      rateLimitAttempts: 10,
      sleep: noSleep,
    })
    await expect(client.health()).rejects.toBeInstanceOf(MemvaraHttpError)
    expect(f.calls).toHaveLength(2)
  })

  test("remaining reflects the RateLimit-Remaining header of the most recent response", async () => {
    const f = fakeFetch([
      {
        status: 429,
        body: { error: { code: "rate_limited", message: "x" } },
        headers: { "RateLimit-Remaining": "5" },
      },
      {
        status: 200,
        body: { status: "ok", memvara_version: "1" },
        headers: { "RateLimit-Remaining": "3" },
      },
    ])
    const client = new MemvaraClient({
      baseUrl: "http://api.test",
      apiKey: "k1",
      fetchImpl: f.impl,
      sleep: noSleep,
    })
    expect(client.remaining).toBeNull()
    await client.health()
    expect(client.remaining).toBe(3)
  })
})

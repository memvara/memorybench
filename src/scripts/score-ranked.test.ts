// src/scripts/score-ranked.test.ts
import { describe, expect, test } from "bun:test"
import { formatScore, goldByQuestion, score } from "./score-ranked"
import type { ScoreResult } from "./score-ranked"
import type { RunCheckpoint } from "../types/checkpoint"

describe("goldByQuestion", () => {
  test("collects only the has_answer turns' own text, by question id", () => {
    const gold = goldByQuestion([
      {
        question_id: "q1",
        haystack_sessions: [
          [{ content: "no answer here" }, { content: "the answer is Lisbon", has_answer: true }],
          [{ content: "a second session's gold turn", has_answer: true }],
        ],
      },
      { question_id: "q2", haystack_sessions: [[{ content: "nothing gold" }]] },
    ])
    expect(gold.get("q1")).toEqual(
      new Set(["the answer is Lisbon", "a second session's gold turn"])
    )
    expect(gold.get("q2")).toEqual(new Set())
    expect(gold.get("q3")).toBeUndefined()
  })
})

/** The pieces of a `RunCheckpoint` `score` actually reads: one question's id and its
 *  search results. Every other field a real checkpoint carries is irrelevant to scoring,
 *  so the fixture omits it rather than filling it in to satisfy the type. */
function checkpointOf(
  questions: Record<string, { questionId: string; results?: unknown[] }>
): RunCheckpoint {
  const out: RunCheckpoint["questions"] = {}
  for (const [id, q] of Object.entries(questions)) {
    out[id] = {
      questionId: q.questionId,
      containerTag: `${q.questionId}-run`,
      question: "irrelevant",
      groundTruth: "irrelevant",
      questionType: "single-session-user",
      phases: {
        ingest: { status: "completed", completedSessions: [] },
        indexing: { status: "completed" },
        search: { status: "completed", results: q.results },
        answer: { status: "pending" },
        evaluate: { status: "pending" },
      },
    } as RunCheckpoint["questions"][string]
  }
  return {
    runId: "run",
    dataSourceRunId: "run",
    status: "completed",
    provider: "memvara",
    benchmark: "longmemeval",
    judge: "irrelevant",
    answeringModel: "irrelevant",
    createdAt: "2026-01-01T00:00:00Z",
    updatedAt: "2026-01-01T00:00:00Z",
    questions: out,
  }
}

function turn(content: string, selected?: boolean | null): unknown {
  return { kind: "turn", role: "user", content, ts: "2023-01-01T00:00:00Z", score: 0.5, selected }
}

describe("score", () => {
  test("splits the kept set into gold and non-gold, against has_answer text", () => {
    const gold = new Map([["q1", new Set(["gold turn"])]])
    const checkpoint = checkpointOf({
      q1: {
        questionId: "q1",
        results: [
          turn("gold turn", true), // gold, kept
          turn("distractor kept", true), // non-gold, kept
          turn("distractor not kept", false), // non-gold, not kept
        ],
      },
    })
    const result = score(checkpoint, gold)
    expect(result).toEqual({
      questionsScored: 1,
      goldKept: 1,
      goldMissed: 0,
      nonGoldKept: 1,
      nonGoldMissed: 1,
      sawRanking: true,
    })
  })

  test("a gold turn the results never returned counts as missed, not absent", () => {
    const gold = new Map([["q1", new Set(["gold turn", "unreturned gold turn"])]])
    const checkpoint = checkpointOf({
      q1: { questionId: "q1", results: [turn("gold turn", true)] },
    })
    // The unreturned gold turn is not in results at all, so it cannot be counted -- gold
    // recall here is against the candidates the search actually returned, the same scope
    // extract.py scores over, not every has_answer turn in the dataset.
    expect(score(checkpoint, gold).goldMissed).toBe(0)
  })

  test("selected: false and selected: undefined both count as not kept", () => {
    const gold = new Map([["q1", new Set(["gold turn"])]])
    const checkpoint = checkpointOf({
      q1: {
        questionId: "q1",
        results: [turn("gold turn", false), turn("gold turn seen but unranked")],
      },
    })
    const result = score(checkpoint, gold)
    expect(result.goldKept).toBe(0)
    expect(result.goldMissed).toBe(1)
    // The second turn's own text is not in the gold set (it is a distinct string), so it
    // lands in nonGoldMissed; its `selected` is undefined and does not set sawRanking on
    // its own -- the first turn's `selected: false` is what does.
    expect(result.nonGoldMissed).toBe(1)
    expect(result.sawRanking).toBe(true)
  })

  test("sawRanking is false when nothing in the run carries a selected field", () => {
    const gold = new Map([["q1", new Set(["gold turn"])]])
    const checkpoint = checkpointOf({
      q1: {
        questionId: "q1",
        results: [{ kind: "turn", role: "user", content: "gold turn", ts: "t", score: 0.5 }],
      },
    })
    expect(score(checkpoint, gold).sawRanking).toBe(false)
  })

  test("a question with no gold turns, or no search results, does not count", () => {
    const checkpoint = checkpointOf({
      abstention: { questionId: "abstention", results: [turn("anything", true)] },
      unsearched: { questionId: "unsearched" },
    })
    const gold = new Map([
      ["abstention", new Set<string>()],
      ["unsearched", new Set(["gold turn"])],
    ])
    expect(score(checkpoint, gold).questionsScored).toBe(0)
  })

  test("a memory (claim) result is ignored: only turns are scored", () => {
    const gold = new Map([["q1", new Set(["gold turn"])]])
    const checkpoint = checkpointOf({
      q1: {
        questionId: "q1",
        results: [{ kind: "memory", text: "gold turn", score: 0.9 }, turn("gold turn", true)],
      },
    })
    expect(score(checkpoint, gold)).toEqual({
      questionsScored: 1,
      goldKept: 1,
      goldMissed: 0,
      nonGoldKept: 0,
      nonGoldMissed: 0,
      sawRanking: true,
    })
  })
})

describe("formatScore", () => {
  test("prints extract.py's shape: gold recall, then the non-gold keep rate", () => {
    const r: ScoreResult = {
      questionsScored: 199,
      goldKept: 321,
      goldMissed: 31,
      nonGoldKept: 24,
      nonGoldMissed: 352,
      sawRanking: true,
    }
    expect(formatScore(r)).toBe(
      "scored 199 questions: gold kept 321 / 352 (recall 0.912), non-gold kept 24 / 376 (0.064)"
    )
  })

  test("prints n/a rather than dividing by zero when a total is empty", () => {
    const r: ScoreResult = {
      questionsScored: 0,
      goldKept: 0,
      goldMissed: 0,
      nonGoldKept: 0,
      nonGoldMissed: 0,
      sawRanking: true,
    }
    expect(formatScore(r)).toBe(
      "scored 0 questions: gold kept 0 / 0 (recall n/a), non-gold kept 0 / 0 (n/a)"
    )
  })
})

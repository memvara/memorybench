import { describe, expect, test } from "bun:test"
import { buildMemvaraAnswerPrompt, renderMemvaraContext, MEMVARA_PROMPTS } from "./prompts"
import type { MemvaraContextItem } from "./prompts"

const memory: MemvaraContextItem = {
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
  sources: ["ep_1"],
}
const ended: MemvaraContextItem = {
  ...memory,
  text: "user lives in Porto",
  object: "Porto",
  state: "ended",
  valid_from: "2023-01-02T10:00:00+00:00",
  valid_to: "2023-05-20T02:21:00+00:00",
  invalidated_at: "2023-05-20T02:21:00+00:00",
  score: 0.4,
}
const turn: MemvaraContextItem = {
  kind: "turn",
  role: "user",
  content: "I moved to Lisbon last week!",
  ts: "2023-05-20T02:21:00+00:00",
  score: 0.44,
}

describe("renderMemvaraContext", () => {
  test("renders memories with both clocks and turns with their date", () => {
    const out = renderMemvaraContext([memory, ended, turn])
    expect(out).toContain("Memories")
    expect(out).toContain(
      "[valid from 2023-05-20 02:21, recorded 2023-05-20 02:21, live] user lives in Lisbon"
    )
    expect(out).toContain(
      "[valid from 2023-01-02 10:00 to 2023-05-20 02:21, recorded 2023-05-20 02:21, ended] user lives in Porto"
    )
    expect(out).toContain("Conversation excerpts")
    expect(out).toContain("[2023-05-20 02:21] user: I moved to Lisbon last week!")
  })

  test("keeps memvara's order inside each block", () => {
    const out = renderMemvaraContext([ended, memory])
    expect(out.indexOf("Porto")).toBeLessThan(out.indexOf("Lisbon"))
  })

  test("says so when there is nothing", () => {
    expect(renderMemvaraContext([])).toContain("No memories were retrieved.")
  })

  test("ignores objects it does not recognise instead of throwing", () => {
    const out = renderMemvaraContext([{ foo: "bar" }, memory])
    expect(out).toContain("user lives in Lisbon")
  })
})

describe("buildMemvaraAnswerPrompt", () => {
  test("carries the question, the question date, the context and the abstention rule", () => {
    const prompt = buildMemvaraAnswerPrompt(
      "Where do I live?",
      [memory, turn],
      "2023/06/01 (Thu) 09:00"
    )
    expect(prompt).toContain("Question: Where do I live?")
    expect(prompt).toContain("Question date: 2023/06/01 (Thu) 09:00")
    expect(prompt).toContain("user lives in Lisbon")
    expect(prompt).toContain("I don't know")
    expect(prompt).toContain("Answer:")
  })

  test("says the date is not specified when none is given", () => {
    expect(buildMemvaraAnswerPrompt("q", [])).toContain("Question date: not specified")
  })

  test("is what the provider exports as its answer prompt", () => {
    expect(MEMVARA_PROMPTS.answerPrompt).toBe(buildMemvaraAnswerPrompt)
    expect(MEMVARA_PROMPTS.judgePrompt).toBeUndefined()
  })
})

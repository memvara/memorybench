// src/providers/memvara/arm-invariant.test.ts
import { describe, expect, test } from "bun:test"
import { countO200k } from "../../utils/tokens"
import { buildMemvaraAnswerPrompt, renderMemvaraContext } from "./prompts"
import { ARM_FIXTURE_CONTEXT, ARM_FIXTURE_QUESTIONS } from "./__fixtures__/arm-context"
import { ARM_GOLDEN_RENDERS } from "./__fixtures__/arm-renders"
import { withEnv } from "./__fixtures__/with-env"

/** Two arms are about to be judged on this renderer, and a rendering that moves by one byte
 *  makes their scores incomparable with the runs already recorded. The strings in
 *  `__fixtures__/arm-renders.ts` were produced by the renderer as it stood before this
 *  change, from the 30-turn fixture beside them; these tests assert the renderer still
 *  produces them exactly.
 *
 *  A failure here is not a test to update. It says the change altered what the arm measures,
 *  and the diff is the thing to look at. */

const ARM = {
  MEMVARA_TURNS_ONLY: "1",
  MEMVARA_SEARCH_K: "200",
  MEMVARA_TOKEN_BUDGET: "720",
  MEMVARA_HEAD_WHOLE: undefined,
  MEMVARA_TAIL_CHARS: undefined,
  MEMVARA_ANSWER_PROMPT: undefined,
}

const ALL_KNOBS_OFF = {
  MEMVARA_TURNS_ONLY: undefined,
  MEMVARA_SEARCH_K: undefined,
  MEMVARA_TOKEN_BUDGET: undefined,
  MEMVARA_HEAD_WHOLE: undefined,
  MEMVARA_TAIL_CHARS: undefined,
  MEMVARA_ROLE_SELECT: undefined,
  MEMVARA_ANSWER_PROMPT: undefined,
}

const { assistantAsking, userAsking } = ARM_FIXTURE_QUESTIONS
const QUESTION_DATE = "2023/06/01 (Thu) 09:00"

describe("the fixture the arms are pinned against", () => {
  test("is 30 turns of both roles, at the lengths memvara really returns", () => {
    const turns = ARM_FIXTURE_CONTEXT.filter((x) => x.kind === "turn")
    expect(turns.length).toBe(30)
    expect(turns.filter((t) => t.role === "user").length).toBe(16)
    expect(turns.filter((t) => t.role === "assistant").length).toBe(14)
    const lengths = turns.map((t) => t.content.length)
    expect(Math.min(...lengths)).toBeGreaterThan(100)
    expect(Math.max(...lengths)).toBeGreaterThan(3000)
    // Claims are present, so MEMVARA_TURNS_ONLY has something to drop.
    expect(ARM_FIXTURE_CONTEXT.filter((x) => x.kind === "memory").length).toBe(3)
  })
})

describe("MEMVARA_ROLE_SELECT=route and =user at a 720-token budget", () => {
  test("render byte-identically to the strings captured before this change", () => {
    const route = { ...ALL_KNOBS_OFF, ...ARM, MEMVARA_ROLE_SELECT: "route" }
    const user = { ...ALL_KNOBS_OFF, ...ARM, MEMVARA_ROLE_SELECT: "user" }

    expect(withEnv(route, () => renderMemvaraContext(ARM_FIXTURE_CONTEXT, assistantAsking))).toBe(
      ARM_GOLDEN_RENDERS.routeAssistantAsking
    )
    expect(withEnv(route, () => renderMemvaraContext(ARM_FIXTURE_CONTEXT, userAsking))).toBe(
      ARM_GOLDEN_RENDERS.routeUserAsking
    )
    expect(withEnv(route, () => renderMemvaraContext(ARM_FIXTURE_CONTEXT))).toBe(
      ARM_GOLDEN_RENDERS.routeNoQuestion
    )
    expect(withEnv(user, () => renderMemvaraContext(ARM_FIXTURE_CONTEXT, assistantAsking))).toBe(
      ARM_GOLDEN_RENDERS.userArm
    )
  })

  test("build the same whole prompt, not merely the same context block", () => {
    const route = { ...ALL_KNOBS_OFF, ...ARM, MEMVARA_ROLE_SELECT: "route" }
    const user = { ...ALL_KNOBS_OFF, ...ARM, MEMVARA_ROLE_SELECT: "user" }
    expect(
      withEnv(route, () =>
        buildMemvaraAnswerPrompt(assistantAsking, ARM_FIXTURE_CONTEXT, QUESTION_DATE)
      )
    ).toBe(ARM_GOLDEN_RENDERS.routePrompt)
    expect(
      withEnv(user, () => buildMemvaraAnswerPrompt(userAsking, ARM_FIXTURE_CONTEXT, QUESTION_DATE))
    ).toBe(ARM_GOLDEN_RENDERS.userPrompt)
  })

  test("the budget really binds, so the goldens are not just the whole block", () => {
    const route = { ...ALL_KNOBS_OFF, ...ARM, MEMVARA_ROLE_SELECT: "route" }
    const unbudgeted = withEnv({ ...route, MEMVARA_TOKEN_BUDGET: undefined }, () =>
      renderMemvaraContext(ARM_FIXTURE_CONTEXT, userAsking)
    )
    expect(unbudgeted.split("\n").length).toBeGreaterThan(
      ARM_GOLDEN_RENDERS.routeUserAsking.split("\n").length
    )
    // The routed-to-assistant arm comes to a header and one turn: that turn is 447 tokens
    // of the 720, and the second assistant turn would take the block to 885. Assistant
    // turns are long enough that the budget binds after one of them, which is the whole
    // reason the two arms render different amounts of the same retrieval.
    expect(ARM_GOLDEN_RENDERS.routeAssistantAsking.split("\n").length).toBe(2)
    expect(countO200k(ARM_GOLDEN_RENDERS.routeAssistantAsking)).toBe(447)
  })
})

describe("MEMVARA_ANSWER_PROMPT=v1", () => {
  test("builds the pinned prompts byte for byte, asked for by name as well as by default", () => {
    const v1 = { MEMVARA_ANSWER_PROMPT: "v1" }
    const route = { ...ALL_KNOBS_OFF, ...ARM, ...v1, MEMVARA_ROLE_SELECT: "route" }
    const user = { ...ALL_KNOBS_OFF, ...ARM, ...v1, MEMVARA_ROLE_SELECT: "user" }
    expect(
      withEnv(route, () =>
        buildMemvaraAnswerPrompt(assistantAsking, ARM_FIXTURE_CONTEXT, QUESTION_DATE)
      )
    ).toBe(ARM_GOLDEN_RENDERS.routePrompt)
    expect(
      withEnv(user, () => buildMemvaraAnswerPrompt(userAsking, ARM_FIXTURE_CONTEXT, QUESTION_DATE))
    ).toBe(ARM_GOLDEN_RENDERS.userPrompt)
    expect(
      withEnv({ ...ALL_KNOBS_OFF, ...v1 }, () =>
        buildMemvaraAnswerPrompt(userAsking, ARM_FIXTURE_CONTEXT, QUESTION_DATE)
      )
    ).toBe(ARM_GOLDEN_RENDERS.defaultPrompt)
  })

  test("v2 moves the prompt, so the pinning above is a check and not a tautology", () => {
    const v2 = withEnv({ ...ALL_KNOBS_OFF, MEMVARA_ANSWER_PROMPT: "v2" }, () =>
      buildMemvaraAnswerPrompt(userAsking, ARM_FIXTURE_CONTEXT, QUESTION_DATE)
    )
    expect(v2).not.toBe(ARM_GOLDEN_RENDERS.defaultPrompt)
    // The context block is what the goldens pin, and v2 leaves it alone: only the
    // surrounding instructions differ.
    expect(v2).toContain(ARM_GOLDEN_RENDERS.default)
  })
})

describe("the default render", () => {
  test("is byte-identical with every knob unset, question or no question", () => {
    expect(withEnv(ALL_KNOBS_OFF, () => renderMemvaraContext(ARM_FIXTURE_CONTEXT))).toBe(
      ARM_GOLDEN_RENDERS.default
    )
    expect(
      withEnv(ALL_KNOBS_OFF, () => renderMemvaraContext(ARM_FIXTURE_CONTEXT, assistantAsking))
    ).toBe(ARM_GOLDEN_RENDERS.defaultWithQuestion)
    expect(
      withEnv(ALL_KNOBS_OFF, () =>
        buildMemvaraAnswerPrompt(userAsking, ARM_FIXTURE_CONTEXT, QUESTION_DATE)
      )
    ).toBe(ARM_GOLDEN_RENDERS.defaultPrompt)
  })
})

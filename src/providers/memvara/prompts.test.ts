import { describe, expect, test } from "bun:test"
import {
  ASSISTANT_QUESTION_RE,
  buildMemvaraAnswerPrompt,
  renderMemvaraContext,
  wantsAssistant,
  MEMVARA_PROMPTS,
} from "./prompts"
import { countO200k } from "../../utils/tokens"
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

/** Sets environment knobs for one call and always puts them back, so an arm switched on
 *  in one test cannot leak into the next. The provider reads them at call time, which is
 *  what makes this enough. */
function withEnv<T>(vars: Record<string, string | undefined>, fn: () => T): T {
  const saved = new Map<string, string | undefined>()
  for (const [name, value] of Object.entries(vars)) {
    saved.set(name, process.env[name])
    if (value === undefined) delete process.env[name]
    else process.env[name] = value
  }
  try {
    return fn()
  } finally {
    for (const [name, value] of saved) {
      if (value === undefined) delete process.env[name]
      else process.env[name] = value
    }
  }
}

const ALL_KNOBS_OFF = {
  MEMVARA_TURNS_ONLY: undefined,
  MEMVARA_HEAD_WHOLE: undefined,
  MEMVARA_TAIL_CHARS: undefined,
  MEMVARA_ROLE_SELECT: undefined,
  MEMVARA_TOKEN_BUDGET: undefined,
}

const assistantTurn: MemvaraContextItem = {
  kind: "turn",
  role: "assistant",
  content: "I suggested the Bairro Alto Hotel.",
  ts: "2023-05-20T02:22:00+00:00",
  score: 0.42,
}

const DEFAULT_RENDER = [
  "Memories (each with the period it was true for and the date it was recorded):",
  "- [valid from 2023-05-20 02:21, recorded 2023-05-20 02:21, live] user lives in Lisbon",
  "- [valid from 2023-01-02 10:00 to 2023-05-20 02:21, recorded 2023-05-20 02:21, ended] user lives in Porto",
  "",
  "Conversation excerpts (verbatim, with the date they were said):",
  "- [2023-05-20 02:21] user: I moved to Lisbon last week!",
  "- [2023-05-20 02:22] assistant: I suggested the Bairro Alto Hotel.",
].join("\n")

describe("defaults", () => {
  test("with every knob unset the render is byte-identical to the shipped text", () => {
    const out = withEnv(ALL_KNOBS_OFF, () =>
      renderMemvaraContext([memory, ended, turn, assistantTurn])
    )
    expect(out).toBe(DEFAULT_RENDER)
  })

  test("a knob blanked rather than unset is still off, which is how a run script clears it", () => {
    // `MEMVARA_TOKEN_BUDGET=` in a shell sets the empty string, not nothing. If that were
    // parsed as a number the arm would switch on -- or throw -- without anyone asking.
    const out = withEnv(
      {
        MEMVARA_TURNS_ONLY: "",
        MEMVARA_HEAD_WHOLE: "",
        MEMVARA_TAIL_CHARS: "",
        MEMVARA_ROLE_SELECT: "",
        MEMVARA_TOKEN_BUDGET: "",
      },
      () => renderMemvaraContext([memory, ended, turn, assistantTurn])
    )
    expect(out).toBe(DEFAULT_RENDER)
  })

  test("a question that would fire the routing rule changes nothing while the knob is off", () => {
    const out = withEnv(ALL_KNOBS_OFF, () =>
      renderMemvaraContext(
        [memory, ended, turn, assistantTurn],
        "What was the hotel you suggested for my Lisbon trip?"
      )
    )
    expect(out).toBe(DEFAULT_RENDER)
  })

  test("the answer prompt embeds exactly that render", () => {
    const prompt = withEnv(ALL_KNOBS_OFF, () =>
      buildMemvaraAnswerPrompt(
        "Where do I live?",
        [memory, ended, turn, assistantTurn],
        "2023/06/01"
      )
    )
    expect(prompt).toContain(`Retrieved context:\n${DEFAULT_RENDER}\n\nHow to read the context:`)
  })
})

describe("wantsAssistant", () => {
  const FOURTEEN = [
    "you suggested",
    "you recommended",
    "you mentioned",
    "you told me",
    "you provided",
    "you wrote",
    "you created",
    "did you say",
    "can you remind me",
    "remind me what",
    "remind me which",
    "remind me who",
    "remind me how",
    "remind me of",
  ]

  test("the exported rule is those fourteen alternatives, case-insensitive", () => {
    expect(ASSISTANT_QUESTION_RE.flags).toBe("i")
    const alternatives = ASSISTANT_QUESTION_RE.source
      .replace(/^\\b\(\?:/, "")
      .replace(/\)\\b$/, "")
      .split("|")
    expect(alternatives).toEqual(FOURTEEN)
  })

  test("fires on each of the fourteen phrasings inside a realistic question", () => {
    const questions = [
      "What was the name of the hotel you suggested for my Lisbon trip?",
      "Which of the two laptops you recommended did I end up buying?",
      "What was the podcast you mentioned while we were talking about running?",
      "I have forgotten the knee stretch you told me about in February.",
      "Which of the itineraries you provided had the ferry to Cascais?",
      "What was the subject line of the email you wrote for my landlord?",
      "What did we call the training plan you created in March?",
      "What did you say the deadline for the visa application was?",
      "Can you remind me about the gym I signed up for?",
      "Please remind me what I said about the promotion at work.",
      "Remind me which airline I flew to Tokyo with.",
      "Remind me who introduced me to my landlord.",
      "Remind me how I fixed the printer the last time it jammed.",
      "Remind me of the wine we had at the anniversary dinner.",
    ]
    expect(questions.length).toBe(FOURTEEN.length)
    for (const q of questions) {
      expect(wantsAssistant(q)).toBe(true)
    }
    // Every phrasing is really covered, not just fourteen questions that happen to pass.
    for (const phrase of FOURTEEN) {
      expect(questions.some((q) => q.toLowerCase().includes(phrase))).toBe(true)
    }
  })

  test("does not fire on a present-tense request, which asks for a suggestion rather than for one already given", () => {
    expect(wantsAssistant("Can you suggest a hotel for my trip to Miami?")).toBe(false)
    expect(wantsAssistant("Could you recommend a laptop?")).toBe(false)
    expect(wantsAssistant("What should I cook tonight?")).toBe(false)
  })

  test("is case-insensitive and needs the whole phrase", () => {
    expect(wantsAssistant("REMIND ME WHO signed the lease?")).toBe(true)
    expect(wantsAssistant("The hotel you suggestedly picked")).toBe(false)
    expect(wantsAssistant("")).toBe(false)
  })
})

describe("MEMVARA_ROLE_SELECT", () => {
  const context = [memory, turn, assistantTurn]

  test('"user" drops the assistant turns and leaves the claims alone', () => {
    const out = withEnv({ ...ALL_KNOBS_OFF, MEMVARA_ROLE_SELECT: "user" }, () =>
      renderMemvaraContext(context, "What was the hotel you suggested?")
    )
    expect(out).toContain("- [2023-05-20 02:21] user: I moved to Lisbon last week!")
    expect(out).not.toContain("assistant:")
    expect(out).toContain("user lives in Lisbon")
  })

  test('"route" keeps assistant turns only when the rule fires', () => {
    const fired = withEnv({ ...ALL_KNOBS_OFF, MEMVARA_ROLE_SELECT: "route" }, () =>
      renderMemvaraContext(context, "What was the name of the hotel you suggested?")
    )
    expect(fired).toContain("assistant: I suggested the Bairro Alto Hotel.")
    expect(fired).not.toContain("user: I moved to Lisbon")

    const notFired = withEnv({ ...ALL_KNOBS_OFF, MEMVARA_ROLE_SELECT: "route" }, () =>
      renderMemvaraContext(context, "Can you suggest a hotel for my trip to Miami?")
    )
    expect(notFired).toContain("user: I moved to Lisbon last week!")
    expect(notFired).not.toContain("assistant:")
  })

  test('"route" with no question falls back to user turns', () => {
    const out = withEnv({ ...ALL_KNOBS_OFF, MEMVARA_ROLE_SELECT: "route" }, () =>
      renderMemvaraContext(context)
    )
    expect(out).toContain("user: I moved to Lisbon last week!")
    expect(out).not.toContain("assistant:")
  })

  test("selection happens before truncation, so ranks are positions after the filter", () => {
    const long: MemvaraContextItem = {
      kind: "turn",
      role: "user",
      content: "x".repeat(50),
      ts: "2023-05-20T02:23:00+00:00",
      score: 0.1,
    }
    // The assistant turn is rank 0 unfiltered; filtered out, the user turn takes rank 0
    // and is the one kept whole.
    const out = withEnv(
      {
        ...ALL_KNOBS_OFF,
        MEMVARA_ROLE_SELECT: "user",
        MEMVARA_HEAD_WHOLE: "1",
        MEMVARA_TAIL_CHARS: "10",
      },
      () => renderMemvaraContext([assistantTurn, turn, long])
    )
    expect(out).toContain("user: I moved to Lisbon last week!")
    expect(out).toContain(`user: ${"x".repeat(10)}…`)
  })

  test("the answer prompt hands the question to the renderer, which is what routes it", () => {
    const prompt = withEnv({ ...ALL_KNOBS_OFF, MEMVARA_ROLE_SELECT: "route" }, () =>
      buildMemvaraAnswerPrompt("What was the name of the hotel you suggested?", context)
    )
    expect(prompt).toContain("assistant: I suggested the Bairro Alto Hotel.")
    expect(prompt).not.toContain("user: I moved to Lisbon")
  })

  test("an unknown value throws and names the variable", () => {
    expect(() =>
      withEnv({ MEMVARA_ROLE_SELECT: "assistant" }, () => renderMemvaraContext([turn]))
    ).toThrow(/MEMVARA_ROLE_SELECT/)
  })
})

describe("MEMVARA_TOKEN_BUDGET", () => {
  const HEADER = "Conversation excerpts (verbatim, with the date they were said):"
  const many: MemvaraContextItem[] = [
    {
      kind: "turn",
      role: "user",
      content: "alpha ".repeat(12),
      ts: "2023-05-20T05:00:00+00:00",
      score: 0.9,
    },
    {
      kind: "turn",
      role: "user",
      content: "bravo ".repeat(12),
      ts: "2023-05-19T05:00:00+00:00",
      score: 0.8,
    },
    {
      kind: "turn",
      role: "user",
      content: "charlie ".repeat(12),
      ts: "2023-05-21T05:00:00+00:00",
      score: 0.7,
    },
    {
      kind: "turn",
      role: "user",
      content: "delta ".repeat(12),
      ts: "2023-05-18T05:00:00+00:00",
      score: 0.6,
    },
  ]
  const lines = many.map((t) => {
    const turnItem = t as Extract<MemvaraContextItem, { kind: "turn" }>
    return `- [${turnItem.ts.slice(0, 10)} ${turnItem.ts.slice(11, 16)}] ${turnItem.role}: ${turnItem.content}`
  })
  const block = (n: number) => [HEADER, ...lines.slice(0, n)].join("\n")

  test("counts in o200k_base, the encoding the harness reports contextTokens with", () => {
    // 13 under o200k_base, 12 under cl100k_base. Pinned so that swapping the encoder,
    // which would silently move every budget, fails here rather than in a run.
    expect(countO200k(HEADER)).toBe(13)
  })

  test("fills greedily and stops at the first turn that would exceed the budget", () => {
    const budget = countO200k(block(3))
    expect(countO200k(block(4))).toBeGreaterThan(budget)
    const out = withEnv({ ...ALL_KNOBS_OFF, MEMVARA_TOKEN_BUDGET: String(budget) }, () =>
      renderMemvaraContext(many)
    )
    expect(out).toBe(block(3))
  })

  test("one token less keeps one turn fewer", () => {
    const budget = countO200k(block(3)) - 1
    expect(countO200k(block(2))).toBeLessThanOrEqual(budget)
    const out = withEnv({ ...ALL_KNOBS_OFF, MEMVARA_TOKEN_BUDGET: String(budget) }, () =>
      renderMemvaraContext(many)
    )
    expect(out).toBe(block(2))
  })

  test("keeps the first turn even when it alone exceeds the budget", () => {
    const out = withEnv({ ...ALL_KNOBS_OFF, MEMVARA_TOKEN_BUDGET: "1" }, () =>
      renderMemvaraContext(many)
    )
    expect(out).toBe(block(1))
  })

  test("never re-sorts: the turns stay in the order memvara returned them", () => {
    const out = withEnv(
      { ...ALL_KNOBS_OFF, MEMVARA_TOKEN_BUDGET: String(countO200k(block(3))) },
      () => renderMemvaraContext(many)
    )
    expect(out.indexOf("alpha")).toBeLessThan(out.indexOf("bravo"))
    expect(out.indexOf("bravo")).toBeLessThan(out.indexOf("charlie"))
    expect(out).not.toContain("delta")
  })

  test("the budget governs the turns block only, not the claims", () => {
    const out = withEnv({ ...ALL_KNOBS_OFF, MEMVARA_TOKEN_BUDGET: "1" }, () =>
      renderMemvaraContext([memory, ended, ...many])
    )
    expect(out).toContain("user lives in Lisbon")
    expect(out).toContain("user lives in Porto")
    expect(out).toBe(
      [
        "Memories (each with the period it was true for and the date it was recorded):",
        "- [valid from 2023-05-20 02:21, recorded 2023-05-20 02:21, live] user lives in Lisbon",
        "- [valid from 2023-01-02 10:00 to 2023-05-20 02:21, recorded 2023-05-20 02:21, ended] user lives in Porto",
        "",
        block(1),
      ].join("\n")
    )
  })

  test("composes with truncation: a shorter line is a cheaper line, so more turns fit", () => {
    const budget = String(countO200k(block(2)))
    const whole = withEnv({ ...ALL_KNOBS_OFF, MEMVARA_TOKEN_BUDGET: budget }, () =>
      renderMemvaraContext(many)
    )
    const cut = withEnv(
      { ...ALL_KNOBS_OFF, MEMVARA_TOKEN_BUDGET: budget, MEMVARA_TAIL_CHARS: "12" },
      () => renderMemvaraContext(many)
    )
    expect(whole.split("\n").length).toBe(3)
    expect(cut.split("\n").length).toBeGreaterThan(whole.split("\n").length)
    expect(cut).toContain("…")
  })

  test("a value that is not a positive integer throws and names the variable", () => {
    for (const bad of ["0", "-5", "abc", "1.5", "1e", " "]) {
      expect(() =>
        withEnv({ MEMVARA_TOKEN_BUDGET: bad }, () => renderMemvaraContext([turn]))
      ).toThrow(/MEMVARA_TOKEN_BUDGET/)
    }
  })

  test("a turn carrying a tokenizer literal is counted, not thrown on", () => {
    // The LongMemEval haystack contains this exact string, in an assistant turn of
    // question gpt4_78cf46a3. js-tiktoken's default settings raise on it, which would
    // fail the question under a budget while the unbudgeted control answered it.
    const literal: MemvaraContextItem = {
      kind: "turn",
      role: "assistant",
      content: "... Skipped 1 messages<|endoftext|>",
      ts: "2023-05-20T05:00:00+00:00",
      score: 0.5,
    }
    expect(countO200k("<|endoftext|>")).toBe(7)
    const out = withEnv({ ...ALL_KNOBS_OFF, MEMVARA_TOKEN_BUDGET: "500" }, () =>
      renderMemvaraContext([literal, ...many])
    )
    expect(out).toContain("assistant: ... Skipped 1 messages<|endoftext|>")
  })
})

describe("MEMVARA_HEAD_WHOLE and MEMVARA_TAIL_CHARS", () => {
  test("unset, blank or zero leaves every turn whole", () => {
    for (const off of [undefined, "", "0"]) {
      const out = withEnv({ ...ALL_KNOBS_OFF, MEMVARA_TAIL_CHARS: off }, () =>
        renderMemvaraContext([memory, ended, turn, assistantTurn])
      )
      expect(out).toBe(DEFAULT_RENDER)
    }
  })

  test("a value that is not a non-negative integer throws and names the variable", () => {
    // `80O` is the typo this guards: Number("80O") is NaN, every comparison against NaN
    // is false, and without the throw the arm would silently render the control.
    for (const bad of ["80O", "abc", "-1", "1.5", "1e"]) {
      expect(() =>
        withEnv({ ...ALL_KNOBS_OFF, MEMVARA_TAIL_CHARS: bad }, () => renderMemvaraContext([turn]))
      ).toThrow(/MEMVARA_TAIL_CHARS/)
      expect(() =>
        withEnv({ ...ALL_KNOBS_OFF, MEMVARA_HEAD_WHOLE: bad, MEMVARA_TAIL_CHARS: "10" }, () =>
          renderMemvaraContext([turn])
        )
      ).toThrow(/MEMVARA_HEAD_WHOLE/)
    }
  })
})

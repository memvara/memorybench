import { describe, expect, spyOn, test } from "bun:test"
import {
  ASSISTANT_QUESTION_RE,
  buildMemvaraAnswerPrompt,
  renderMemvaraContext,
  wantsAssistant,
  MEMVARA_PROMPTS,
  TURNS_HEADER,
  TURNS_HEADER_CUT,
  V2_CONTEXT_BULLETS,
  V2_INSTRUCTION,
} from "./prompts"
import { countO200k, countTokens } from "../../utils/tokens"
import { DEFAULT_ANSWERING_MODEL, getModelConfig } from "../../utils/models"
import { logger } from "../../utils/logger"
import { withEnv } from "./__fixtures__/with-env"
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

const ALL_KNOBS_OFF = {
  MEMVARA_TURNS_ONLY: undefined,
  MEMVARA_HEAD_WHOLE: undefined,
  MEMVARA_TAIL_CHARS: undefined,
  MEMVARA_ROLE_SELECT: undefined,
  MEMVARA_TOKEN_BUDGET: undefined,
  MEMVARA_ANSWER_PROMPT: undefined,
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

  test("a knob holding only whitespace is off too, and is off for every knob alike", () => {
    // `MEMVARA_TAIL_CHARS="$UNSET_VAR "` is how whitespace gets in. The knobs used to
    // disagree about it -- the truncation pair treated it as off, the budget and the search
    // depth threw -- so the same run script was an arm under one knob and a crash under
    // another. One rule now: nothing but whitespace means the knob was cleared.
    const out = withEnv(
      {
        MEMVARA_TURNS_ONLY: " ",
        MEMVARA_HEAD_WHOLE: "  ",
        MEMVARA_TAIL_CHARS: "\t",
        MEMVARA_ROLE_SELECT: " ",
        MEMVARA_TOKEN_BUDGET: " ",
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

describe("MEMVARA_ANSWER_PROMPT", () => {
  const CONTEXT = [memory, ended, turn, assistantTurn]
  const QUESTION = "How many apartments have I viewed?"
  const build = (value: string | undefined) =>
    withEnv({ ...ALL_KNOBS_OFF, MEMVARA_ANSWER_PROMPT: value }, () =>
      buildMemvaraAnswerPrompt(QUESTION, CONTEXT, "2023/06/01 (Thu) 09:00")
    )

  const ADDITIONS = [...V2_CONTEXT_BULLETS, V2_INSTRUCTION]

  test("is v1 when unset, blank or whitespace, and v1 is what v2 was built from", () => {
    const v1 = build(undefined)
    expect(build("")).toBe(v1)
    expect(build(" ")).toBe(v1)
    expect(build("v1")).toBe(v1)
    for (const line of ADDITIONS) expect(v1).not.toContain(line)
  })

  test("v2 carries each of the four additions exactly once", () => {
    const v2 = build("v2")
    for (const line of ADDITIONS) {
      expect(v2.split(line).length - 1).toBe(1)
    }
  })

  // The arms are compared against each other, so what v2 leaves alone matters as much as
  // what it adds: strike the four inserted lines out of v2 and what remains has to be v1
  // to the byte, or the two prompts differ somewhere nobody decided they should.
  test("v2 is v1 with four lines inserted and nothing else touched", () => {
    const v2 = build("v2")
    const inserted = new Set(ADDITIONS)
    const stripped = v2
      .split("\n")
      .filter((line) => !inserted.has(line))
      .join("\n")
    expect(stripped).toBe(build("v1"))
  })

  // The strip test above removes whole lines and rejoins them, so it is blind to an
  // addition that landed in the wrong place: move a bullet to the top of the reading list
  // and it still passes. Each addition was written to follow one particular line, so pin
  // it to that line rather than to the section it sits in.
  test("v2 inserts each addition directly after the line it was written to follow", () => {
    const v2 = build("v2")
    const lines = v2.split("\n")
    const readingList = lines.indexOf("How to read the context:")
    const lastReadingRule = lines.indexOf(
      '- Resolve relative expressions such as "today", "yesterday", "last week" or "in two months" against the date of the excerpt or memory they appear in, never against the current date. Use the question date only to understand what the question is asking about.'
    )
    expect(readingList).toBeGreaterThan(-1)
    expect(lastReadingRule).toBeGreaterThan(readingList)
    V2_CONTEXT_BULLETS.forEach((bullet, i) => {
      expect(lines[lastReadingRule + 1 + i]).toBe(bullet)
    })
    const abstention = lines.indexOf(
      '- If it does not, answer "I don\'t know" and say what is missing. Do not guess.'
    )
    expect(abstention).toBeGreaterThan(lines.indexOf("Instructions:"))
    expect(lines[abstention + 1]).toBe("- Base the answer only on the context above.")
    expect(lines[abstention + 2]).toBe(V2_INSTRUCTION)
  })

  test("changes the prompt only, never the rendered context", () => {
    const render = withEnv(ALL_KNOBS_OFF, () => renderMemvaraContext(CONTEXT, QUESTION))
    expect(build("v2")).toContain(`Retrieved context:\n${render}\n\nHow to read the context:`)
  })

  test("a value that is neither v1 nor v2 throws and names the variable", () => {
    for (const bad of ["v3", "V2", "v2 ", "2"]) {
      expect(() => build(bad)).toThrow(/MEMVARA_ANSWER_PROMPT/)
    }
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
  test("is the header the renderer exports, spelled out here so a reword shows up as a diff", () => {
    expect(TURNS_HEADER).toBe(HEADER)
  })

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

  test("counts in o200k_base, which is what contextTokens counts for the o200k model ids", () => {
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

  // Three turns of text where one character costs three tokens, which is what makes the
  // byte bound in front of the encoder the only sound one to skip an encoding on.
  const wide: MemvaraContextItem[] = [0, 1, 2].map((i) => ({
    kind: "turn",
    role: "user",
    content: "ᾧ".repeat(60),
    ts: `2023-05-2${i}T05:00:00+00:00`,
    score: 0.9 - i / 10,
  }))
  const wideLines = wide.map((t) => {
    const turnItem = t as Extract<MemvaraContextItem, { kind: "turn" }>
    return `- [${turnItem.ts.slice(0, 10)} ${turnItem.ts.slice(11, 16)}] ${turnItem.role}: ${turnItem.content}`
  })
  const wideBlock = (n: number) => [HEADER, ...wideLines.slice(0, n)].join("\n")

  test("skips an encoding on a byte count, because a character bounds nothing", () => {
    // Every o200k_base token spends at least one UTF-8 byte, so bytes are an upper bound on
    // tokens and a block that fits by bytes cannot fail by tokens. Characters bound nothing
    // at all: "ᾧ" is one character, three bytes and three tokens, so this text saturates the
    // byte bound and breaks the character one. Two of these turns under the header come to
    // 239 characters and 406 tokens, so a guard that measured characters would wave the
    // second turn through at a 240-token budget and render a block two thirds over it.
    const budget = 240
    expect(countO200k("ᾧ")).toBe(3)
    expect(wideBlock(2).length).toBeLessThanOrEqual(budget)
    expect(countO200k(wideBlock(2))).toBeGreaterThan(budget)

    const out = withEnv({ ...ALL_KNOBS_OFF, MEMVARA_TOKEN_BUDGET: String(budget) }, () =>
      renderMemvaraContext(wide)
    )
    expect(out).toBe(wideBlock(1))
    expect(countO200k(out)).toBeLessThanOrEqual(budget)
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
    const budget = String(countO200k(block(3)))
    const whole = withEnv({ ...ALL_KNOBS_OFF, MEMVARA_TOKEN_BUDGET: budget }, () =>
      renderMemvaraContext(many)
    )
    const cut = withEnv(
      { ...ALL_KNOBS_OFF, MEMVARA_TOKEN_BUDGET: budget, MEMVARA_TAIL_CHARS: "12" },
      () => renderMemvaraContext(many)
    )
    expect(whole.split("\n").length).toBe(4)
    expect(cut.split("\n").length).toBe(5)
    expect(cut).toContain("…")
  })

  test("the header a cut block gets is part of what the budget pays for", () => {
    // It is five tokens dearer than the verbatim one, so a budget tight enough to be
    // decided by five tokens keeps one fewer turn once anything is cut. Truncation still
    // buys more turns than it costs -- the test above -- but not at every budget, and the
    // arithmetic is the header's as much as the lines'.
    expect(countO200k(TURNS_HEADER)).toBe(13)
    expect(countO200k(TURNS_HEADER_CUT)).toBe(18)
    const budget = String(countO200k(block(2)))
    const cut = withEnv(
      { ...ALL_KNOBS_OFF, MEMVARA_TOKEN_BUDGET: budget, MEMVARA_TAIL_CHARS: "12" },
      () => renderMemvaraContext(many)
    )
    expect(cut.split("\n")[0]).toBe(TURNS_HEADER_CUT)
    expect(cut.split("\n").length).toBe(3)
  })

  test("a value that is not a positive integer throws and names the variable", () => {
    for (const bad of ["0", "-5", "abc", "1.5", "1e"]) {
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
    for (const off of [undefined, "", " ", "0"]) {
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

describe("the turns header", () => {
  const long: MemvaraContextItem = {
    kind: "turn",
    role: "user",
    content: "a".repeat(200),
    ts: "2023-05-20T05:00:00+00:00",
    score: 0.9,
  }
  const second: MemvaraContextItem = { ...long, content: "b".repeat(200), score: 0.8 }

  test("promises verbatim text only while the block really is verbatim", () => {
    const out = withEnv({ ...ALL_KNOBS_OFF, MEMVARA_TAIL_CHARS: "1000" }, () =>
      renderMemvaraContext([long, second])
    )
    expect(out.split("\n")[0]).toBe(TURNS_HEADER)
    expect(out).not.toContain("…")
  })

  test("says some turns were cut when any rendered turn was cut", () => {
    // A header promising verbatim text above a line ending in an ellipsis tells the reader
    // the ellipsis is something the speaker typed, and the reader here is the model being
    // asked to answer from it.
    const out = withEnv({ ...ALL_KNOBS_OFF, MEMVARA_TAIL_CHARS: "10" }, () =>
      renderMemvaraContext([long, second])
    )
    expect(out.split("\n")[0]).toBe(TURNS_HEADER_CUT)
    expect(TURNS_HEADER_CUT).toBe(
      "Conversation excerpts (with the date they were said; some cut short, marked with …):"
    )
    expect(out).toContain(`user: ${"a".repeat(10)}…`)
  })

  test("stays verbatim when the only cut turns are the ones the budget dropped", () => {
    const whole = withEnv(
      { ...ALL_KNOBS_OFF, MEMVARA_HEAD_WHOLE: "1", MEMVARA_TAIL_CHARS: "10" },
      () => renderMemvaraContext([long])
    )
    const out = withEnv(
      {
        ...ALL_KNOBS_OFF,
        MEMVARA_HEAD_WHOLE: "1",
        MEMVARA_TAIL_CHARS: "10",
        MEMVARA_TOKEN_BUDGET: String(countO200k(whole)),
      },
      () => renderMemvaraContext([long, second])
    )
    expect(out).toBe(whole)
    expect(out.split("\n")[0]).toBe(TURNS_HEADER)
    expect(out).not.toContain("…")
  })
})

describe("MEMVARA_TURNS_ONLY", () => {
  test('"1" drops the claims and leaves the turns and their order alone', () => {
    const out = withEnv({ ...ALL_KNOBS_OFF, MEMVARA_TURNS_ONLY: "1" }, () =>
      renderMemvaraContext([memory, ended, turn, assistantTurn])
    )
    expect(out).toBe(
      [
        TURNS_HEADER,
        "- [2023-05-20 02:21] user: I moved to Lisbon last week!",
        "- [2023-05-20 02:22] assistant: I suggested the Bairro Alto Hotel.",
      ].join("\n")
    )
  })

  test("any value other than 1 throws and names the variable", () => {
    // It used to compare the raw value to "1" and treat everything else as off, so
    // MEMVARA_TURNS_ONLY=true ran the control while reading, to whoever wrote the run
    // script, as the arm.
    for (const bad of ["0", "true", "yes", "on", "TRUE"]) {
      expect(() =>
        withEnv({ ...ALL_KNOBS_OFF, MEMVARA_TURNS_ONLY: bad }, () => renderMemvaraContext([turn]))
      ).toThrow(/MEMVARA_TURNS_ONLY/)
    }
  })
})

describe("a role selection that empties the block", () => {
  const longQuestion = `Remind me which airline I flew to Tokyo with, and whether ${"x".repeat(80)}`

  function captureWarnings<T>(fn: () => T): { out: T; warnings: string[] } {
    const warnings: string[] = []
    const spy = spyOn(logger, "warn").mockImplementation((message: string) => {
      warnings.push(message)
    })
    try {
      return { out: fn(), warnings }
    } finally {
      spy.mockRestore()
    }
  }

  test("warns, naming the knob and the first 60 characters of the question", () => {
    const { out, warnings } = captureWarnings(() =>
      withEnv({ ...ALL_KNOBS_OFF, MEMVARA_ROLE_SELECT: "route" }, () =>
        renderMemvaraContext([turn], longQuestion)
      )
    )
    // The prompt is unchanged: this says what happened, it does not repair it.
    expect(out).toBe("No memories were retrieved.")
    expect(warnings.length).toBe(1)
    expect(warnings[0]).toContain("MEMVARA_ROLE_SELECT=route")
    expect(warnings[0]).toContain(longQuestion.slice(0, 60))
    expect(warnings[0]).not.toContain(longQuestion.slice(0, 61))
  })

  test("says nothing when turns survive the selection, or when none were retrieved", () => {
    const kept = captureWarnings(() =>
      withEnv({ ...ALL_KNOBS_OFF, MEMVARA_ROLE_SELECT: "user" }, () =>
        renderMemvaraContext([turn, assistantTurn], "Where do I live?")
      )
    )
    expect(kept.warnings).toEqual([])
    const nothing = captureWarnings(() =>
      withEnv({ ...ALL_KNOBS_OFF, MEMVARA_ROLE_SELECT: "user" }, () =>
        renderMemvaraContext([memory], "Where do I live?")
      )
    )
    expect(nothing.warnings).toEqual([])
  })
})

describe("both truncation knobs are validated whenever either is set", () => {
  test("MEMVARA_HEAD_WHOLE throws on its own, with MEMVARA_TAIL_CHARS unset", () => {
    // The typo used to be reachable only through the other knob: head_whole was read
    // inside the branch tail_chars > 0, so `MEMVARA_HEAD_WHOLE=80O` alone rendered the
    // control in silence.
    expect(() =>
      withEnv({ ...ALL_KNOBS_OFF, MEMVARA_HEAD_WHOLE: "80O" }, () =>
        renderMemvaraContext([turn, assistantTurn])
      )
    ).toThrow(/MEMVARA_HEAD_WHOLE/)
  })

  test("MEMVARA_TAIL_CHARS throws with MEMVARA_HEAD_WHOLE unset", () => {
    expect(() =>
      withEnv({ ...ALL_KNOBS_OFF, MEMVARA_TAIL_CHARS: "80O" }, () =>
        renderMemvaraContext([turn, assistantTurn])
      )
    ).toThrow(/MEMVARA_TAIL_CHARS/)
  })
})

describe("the budget and the reported contextTokens", () => {
  test("count a turn carrying a tokenizer literal the same way, rather than one estimating", () => {
    // countO200k sizes the budget with the special-token check off; countTokens used to
    // encode with it on, throw on this string, and fall back to chars/4. A block sized in
    // real tokens was then reported in an estimate that had nothing to do with it.
    const text = `- [2023-05-20 05:00] assistant: ... Skipped 1 messages<|endoftext|> ${"and the rest of it ".repeat(20)}`
    // Against the model the harness actually answers with, so the assertion is about the
    // pair the report puts side by side rather than about an encoder chosen here.
    const counted = countTokens(text, getModelConfig(DEFAULT_ANSWERING_MODEL))
    expect(counted).toBe(countO200k(text))
    expect(counted).not.toBe(Math.ceil(text.length / 4))
  })
})

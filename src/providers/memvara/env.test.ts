// src/providers/memvara/env.test.ts
import { describe, expect, test } from "bun:test"
import { memvaraProviderSettings } from "./env"
import type { MemvaraSettings } from "./env"
import { withEnv } from "./__fixtures__/with-env"

const ALL_KNOBS_OFF = {
  MEMVARA_TURNS_ONLY: undefined,
  MEMVARA_ROLE_SELECT: undefined,
  MEMVARA_HEAD_WHOLE: undefined,
  MEMVARA_TAIL_CHARS: undefined,
  MEMVARA_TOKEN_BUDGET: undefined,
  MEMVARA_SEARCH_K: undefined,
}

const OFF_SETTINGS: MemvaraSettings = {
  turnsOnly: false,
  roleSelect: "off",
  headWhole: 0,
  tailChars: 0,
  tokenBudget: null,
  searchK: 30,
}

describe("memvaraProviderSettings", () => {
  test("reports the shipped configuration when nothing is set", () => {
    expect(withEnv(ALL_KNOBS_OFF, memvaraProviderSettings)).toEqual(OFF_SETTINGS)
  })

  test("reports the judged arm exactly as it was asked for", () => {
    const settings = withEnv(
      {
        ...ALL_KNOBS_OFF,
        MEMVARA_TURNS_ONLY: "1",
        MEMVARA_SEARCH_K: "200",
        MEMVARA_TOKEN_BUDGET: "720",
        MEMVARA_ROLE_SELECT: "route",
      },
      memvaraProviderSettings
    )
    expect(settings).toEqual({
      turnsOnly: true,
      roleSelect: "route",
      headWhole: 0,
      tailChars: 0,
      tokenBudget: 720,
      searchK: 200,
    })
  })

  test("reports the truncation arm, whose two knobs are read together", () => {
    const settings = withEnv(
      { ...ALL_KNOBS_OFF, MEMVARA_HEAD_WHOLE: "5", MEMVARA_TAIL_CHARS: "400" },
      memvaraProviderSettings
    )
    expect(settings).toEqual({ ...OFF_SETTINGS, headWhole: 5, tailChars: 400 })
  })
})

describe("one off rule for all six knobs", () => {
  test("empty and whitespace-only both mean the knob was cleared", () => {
    for (const off of ["", " ", "  ", "\t", "\n"]) {
      const settings = withEnv(
        {
          MEMVARA_TURNS_ONLY: off,
          MEMVARA_ROLE_SELECT: off,
          MEMVARA_HEAD_WHOLE: off,
          MEMVARA_TAIL_CHARS: off,
          MEMVARA_TOKEN_BUDGET: off,
          MEMVARA_SEARCH_K: off,
        },
        memvaraProviderSettings
      )
      expect(settings).toEqual(OFF_SETTINGS)
    }
  })
})

describe("what zero means depends on the knob", () => {
  test("a length of zero is off, and is accepted", () => {
    const settings = withEnv(
      { ...ALL_KNOBS_OFF, MEMVARA_HEAD_WHOLE: "0", MEMVARA_TAIL_CHARS: "0" },
      memvaraProviderSettings
    )
    expect(settings).toEqual(OFF_SETTINGS)
  })

  test("a search of zero and a budget of zero throw, because neither is an arm", () => {
    expect(() =>
      withEnv({ ...ALL_KNOBS_OFF, MEMVARA_SEARCH_K: "0" }, memvaraProviderSettings)
    ).toThrow(/MEMVARA_SEARCH_K/)
    expect(() =>
      withEnv({ ...ALL_KNOBS_OFF, MEMVARA_TOKEN_BUDGET: "0" }, memvaraProviderSettings)
    ).toThrow(/MEMVARA_TOKEN_BUDGET/)
  })
})

describe("a value the knob does not accept throws and names the variable", () => {
  // `80O` is the typo the numeric knobs guard: Number("80O") is NaN, every comparison
  // against NaN is false, and without the throw the arm renders the control under the
  // arm's name.
  const cases: [string, string[]][] = [
    ["MEMVARA_TURNS_ONLY", ["0", "true", "yes"]],
    ["MEMVARA_ROLE_SELECT", ["assistant", "user ", "Route"]],
    ["MEMVARA_HEAD_WHOLE", ["80O", "-1", "1.5", "1e"]],
    ["MEMVARA_TAIL_CHARS", ["80O", "-1", "1.5", "1e"]],
    ["MEMVARA_TOKEN_BUDGET", ["80O", "-1", "1.5", "1e"]],
    ["MEMVARA_SEARCH_K", ["80O", "-1", "1.5", "1e"]],
  ]

  for (const [name, bad] of cases) {
    test(`${name} rejects ${bad.join(", ")}`, () => {
      for (const value of bad) {
        expect(() => withEnv({ ...ALL_KNOBS_OFF, [name]: value }, memvaraProviderSettings)).toThrow(
          new RegExp(name)
        )
      }
    })
  }
})

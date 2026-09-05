// src/providers/memvara/__fixtures__/arm-context.ts
import type { MemvaraContextItem } from "../prompts"

/** The context the judged arms are pinned against: 30 turns of both roles at the lengths
 *  memvara actually returns, plus the claims that MEMVARA_TURNS_ONLY drops.
 *
 *  Built from a seeded generator rather than written out by hand, because the property
 *  that matters is a spread of realistic lengths -- a user turn of a few dozen words
 *  against an assistant turn of several hundred -- and thirty hand-written turns would be
 *  thirty chances to make them all the same size. The generator is deterministic, so the
 *  same array comes out on every machine and the goldens beside it stay meaningful.
 *
 *  Two contents are deliberate rather than generated. One turn carries the `<|endoftext|>`
 *  literal that the LongMemEval haystack really contains, and several carry accented and
 *  currency characters, so the fixture exercises text where a UTF-8 byte is not a
 *  character and a character is not a token. */

function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const WORDS = [
  "apartment",
  "lease",
  "Lisbon",
  "café",
  "trail",
  "knee",
  "physiotherapist",
  "sourdough",
  "starter",
  "invoice",
  "€40",
  "ferry",
  "Cascais",
  "landlord",
  "deposit",
  "printer",
  "jammed",
  "spreadsheet",
  "column",
  "flight",
  "Tokyo",
  "layover",
  "espresso",
  "grinder",
  "burr",
  "marathon",
  "taper",
  "hamstring",
  "recipe",
  "semolina",
  "hydration",
  "balcony",
  "tomatoes",
  "repotted",
  "visa",
  "appointment",
  "consulate",
  "bicycle",
  "pannier",
  "commute",
  "insurance",
  "excess",
  "renewal",
  "podcast",
  "episode",
  "notebook",
  "battery",
  "warranty",
  "anniversary",
  "wine",
  "Douro",
  "reservation",
  "terrace",
  "sunset",
]

function sentence(rand: () => number, words: number): string {
  const picked: string[] = []
  for (let i = 0; i < words; i++) picked.push(WORDS[Math.floor(rand() * WORDS.length)])
  const text = picked.join(" ")
  return `${text.charAt(0).toUpperCase()}${text.slice(1)}.`
}

function paragraph(rand: () => number, words: number): string {
  const out: string[] = []
  let left = words
  while (left > 0) {
    const n = Math.min(left, 6 + Math.floor(rand() * 14))
    out.push(sentence(rand, n))
    left -= n
  }
  return out.join(" ")
}

function stamp(index: number): string {
  const day = 1 + ((index * 7) % 27)
  const month = 1 + ((index * 3) % 12)
  const hour = 6 + ((index * 5) % 16)
  const minute = (index * 13) % 60
  const two = (n: number) => String(n).padStart(2, "0")
  return `2023-${two(month)}-${two(day)}T${two(hour)}:${two(minute)}:00+00:00`
}

/** Sixteen user turns and fourteen assistant ones, ordered the way a ranking hands them
 *  back rather than the way a transcript would: roles interleave, and score descends. */
function buildTurns(): MemvaraContextItem[] {
  const rand = mulberry32(20230520)
  const roles = [
    "user",
    "assistant",
    "user",
    "user",
    "assistant",
    "assistant",
    "user",
    "assistant",
    "user",
    "user",
    "assistant",
    "user",
    "assistant",
    "assistant",
    "user",
    "assistant",
    "user",
    "user",
    "assistant",
    "assistant",
    "user",
    "assistant",
    "user",
    "user",
    "assistant",
    "user",
    "assistant",
    "assistant",
    "user",
    "user",
  ]
  return roles.map((role, i) => {
    const words = role === "user" ? 18 + Math.floor(rand() * 70) : 140 + Math.floor(rand() * 460)
    const body = paragraph(rand, words)
    return {
      kind: "turn" as const,
      role,
      content: i === 9 ? `${body} ... Skipped 1 messages<|endoftext|>` : body,
      ts: stamp(i),
      score: Number((0.94 - i * 0.021).toFixed(4)),
    }
  })
}

const CLAIMS: MemvaraContextItem[] = [
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
    sources: ["ep_1"],
  },
  {
    kind: "memory",
    text: "user lives in Porto",
    subject: "user",
    predicate: "lives_in",
    object: "Porto",
    state: "ended",
    valid_from: "2023-01-02T10:00:00+00:00",
    valid_to: "2023-05-20T02:21:00+00:00",
    recorded_at: "2023-05-20T02:21:00+00:00",
    invalidated_at: "2023-05-20T02:21:00+00:00",
    score: 0.4,
    sources: ["ep_2"],
  },
  {
    kind: "memory",
    text: "user pays €40 a month for the bicycle insurance",
    subject: "user",
    predicate: "pays_for",
    object: "bicycle insurance",
    state: "live",
    valid_from: "2023-03-11T09:00:00+00:00",
    valid_to: null,
    recorded_at: "2023-03-11T09:00:00+00:00",
    invalidated_at: null,
    score: 0.37,
    sources: ["ep_3"],
  },
]

/** Claims and turns in one array, mixed the way a search response arrives. */
export const ARM_FIXTURE_CONTEXT: MemvaraContextItem[] = (() => {
  const turns = buildTurns()
  return [
    turns[0],
    CLAIMS[0],
    ...turns.slice(1, 6),
    CLAIMS[1],
    ...turns.slice(6, 14),
    CLAIMS[2],
    ...turns.slice(14),
  ]
})()

/** The question each pinned render was built with. The first fires the routing rule, the
 *  second does not, which is the branch that decides what "route" keeps. */
export const ARM_FIXTURE_QUESTIONS = {
  assistantAsking: "What was the name of the hotel you suggested for my Lisbon trip?",
  userAsking: "Which month did I sign the lease on the apartment?",
}

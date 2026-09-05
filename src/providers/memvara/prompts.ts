import { readFileSync } from "fs"
import type { ProviderPrompts } from "../../types/prompts"
import { logger } from "../../utils/logger"
import { countO200k } from "../../utils/tokens"
import {
  answerPrompt,
  contextFile,
  roleSelect,
  tokenBudget,
  truncationKnobs,
  turnsOnly,
} from "./env"
import type { RoleSelect, TruncationKnobs } from "./env"

/** A memory as `MemvaraProvider.search` returns it: memvara's claim with both clocks. */
export interface MemvaraContextMemory {
  kind: "memory"
  text: string
  subject: string
  predicate: string
  object: string
  state: string
  valid_from: string
  valid_to: string | null
  recorded_at: string
  invalidated_at: string | null
  score: number
  sources: string[]
}

/** A raw conversation turn, present when the search included episodes.
 *
 *  `selected` and `span` carry a ranked search's outcome for this turn: `true` when the
 *  model kept it, `false` when it saw the turn and did not, and absent on a plain read or
 *  when the search was not ranked. `span` is only ever set alongside `selected: true`.
 *  Nothing in phase 1 renders the span or the rendering knobs read `selected` -- the
 *  ranking is what the server already applied to the returned order -- but the offline
 *  screen's scoring script needs the field on the turn to tell the kept set from the rest. */
export interface MemvaraContextTurn {
  kind: "turn"
  role: string
  content: string
  ts: string
  score: number
  selected?: boolean | null
  span?: string | null
}

/** A ranked search's own outcome, carried as one extra item in the context array rather
 *  than on the turns themselves -- it describes the whole call, not any one hit, and an
 *  `applied` call whose model kept nothing would otherwise leave no trace that it ran at
 *  all. Absent on a plain (unranked) search. Nothing renders this kind: `isTurn`/`isMemory`
 *  filter it out of the prompt the same way they ignore each other's kind, and it exists
 *  so a caller such as the offline scoring script can read the outcome and the candidate
 *  count off the checkpoint instead of inferring them from the per-turn `selected` field. */
export interface MemvaraContextSelection {
  kind: "selection"
  outcome: string
  reason?: string | null
  status?: number | null
  candidates: number
  kept: number
}

export type MemvaraContextItem = MemvaraContextMemory | MemvaraContextTurn | MemvaraContextSelection

function isMemory(x: unknown): x is MemvaraContextMemory {
  return typeof x === "object" && x !== null && (x as { kind?: unknown }).kind === "memory"
}

function isTurn(x: unknown): x is MemvaraContextTurn {
  return typeof x === "object" && x !== null && (x as { kind?: unknown }).kind === "turn"
}

/** `2023-05-20T02:21:00+00:00` -> `2023-05-20 02:21`. The reader needs a date it can
 *  do arithmetic on, not seconds and an offset. Anything unparseable passes through. */
export function formatWhen(iso: string): string {
  const m = /^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})/.exec(iso)
  return m ? `${m[1]} ${m[2]}` : iso
}

function memoryLine(m: MemvaraContextMemory): string {
  const validity = m.valid_to
    ? `valid from ${formatWhen(m.valid_from)} to ${formatWhen(m.valid_to)}`
    : `valid from ${formatWhen(m.valid_from)}`
  return `- [${validity}, recorded ${formatWhen(m.recorded_at)}, ${m.state}] ${m.text}`
}

/** One turn as it will be rendered, and whether rendering it cut anything off. The flag
 *  decides the block's header: a header promising verbatim text above a line ending in an
 *  ellipsis tells the reader the ellipsis is something the speaker typed. */
interface RenderedTurn {
  line: string
  cut: boolean
}

function renderTurn(
  t: MemvaraContextTurn,
  rank: number,
  { headWhole, tailChars }: TruncationKnobs
): RenderedTurn {
  const cut = tailChars > 0 && rank >= headWhole && t.content.length > tailChars
  const content = cut ? `${t.content.slice(0, tailChars)}…` : t.content
  return { line: `- [${formatWhen(t.ts)}] ${t.role}: ${content}`, cut }
}

/** The fourteen phrasings that ask what the assistant itself said, rather than what the
 *  user said. Frozen: the point of the arm is to measure this rule, so widening it later
 *  measures a different rule. Every alternative is past tense or an explicit request to be
 *  reminded, which is what keeps "can you suggest a hotel" -- a request, not a lookup --
 *  out of it. Word boundaries on both ends, so "you recommend" does not match. */
export const ASSISTANT_QUESTION_RE =
  /\b(?:you suggested|you recommended|you mentioned|you told me|you provided|you wrote|you created|did you say|can you remind me|remind me what|remind me which|remind me who|remind me how|remind me of)\b/i

/** True when the question is asking about the assistant's own past output.
 *
 *  The six "remind me" phrasings were fitted on LongMemEval, where "remind me" occurs only
 *  in questions whose answer is something the assistant said; in ordinary English they ask
 *  about the user's own history just as readily -- "Remind me which airline I flew" wants a
 *  user turn, and this rule sends it to the assistant ones. On another corpus that false
 *  fire discards every user turn with no failsafe behind it, which at a small budget is the
 *  answer thrown away rather than merely reordered. */
export function wantsAssistant(question: string): boolean {
  return ASSISTANT_QUESTION_RE.test(question)
}

export const TURNS_HEADER = "Conversation excerpts (verbatim, with the date they were said):"
export const TURNS_HEADER_CUT =
  "Conversation excerpts (with the date they were said; some cut short, marked with …):"

function turnsHeader(anyCut: boolean): string {
  return anyCut ? TURNS_HEADER_CUT : TURNS_HEADER
}

/** Fills the turns block greedily in memvara's order and stops at the first line that
 *  would push the whole block -- header included -- past the budget. The first line is
 *  always kept, even alone over budget, because a block of nothing but a header answers
 *  no question at all. Nothing is skipped over and nothing is re-sorted: a budget that
 *  reordered the turns would measure a different ranking from the one being benchmarked.
 *
 *  The byte check in front of the encoder is an optimisation and decides nothing. Every
 *  o200k_base token stands for at least one UTF-8 byte, so a block's token count is never
 *  larger than its UTF-8 byte length; when those bytes already fit the budget the encoder
 *  cannot say otherwise and is not asked. Any block that does not fit by bytes is encoded
 *  whole, exactly as before, because for those the answer is genuinely unknown. That is why
 *  the bound counts bytes rather than characters: a character is not an upper bound at all
 *  -- "ᾧ" is one character and three tokens -- while a byte is one by construction. */
function fillToBudget(turns: RenderedTurn[], budget: number): RenderedTurn[] {
  const kept: RenderedTurn[] = []
  let anyCut: boolean = false
  // UTF-8 bytes of the kept lines, each with the newline that joins it to what precedes it.
  let bodyBytes = 0
  for (const turn of turns) {
    const cut: boolean = anyCut || turn.cut
    const candidateBody = bodyBytes + 1 + Buffer.byteLength(turn.line, "utf8")
    const candidateBytes = Buffer.byteLength(turnsHeader(cut), "utf8") + candidateBody
    if (kept.length > 0 && candidateBytes > budget) {
      const block = [turnsHeader(cut), ...kept.map((k) => k.line), turn.line].join("\n")
      if (countO200k(block) > budget) break
    }
    kept.push(turn)
    anyCut = cut
    bodyBytes = candidateBody
  }
  return kept
}

/** The blocks of one MEMVARA_CONTEXT_FILE, keyed by question. Read once per path and kept
 *  for the rest of the process: every question of a run looks in the same file, and parsing
 *  it again for each of them would read the whole file a few hundred times. Keyed by path
 *  rather than held in a single slot, so pointing the knob somewhere else reads the file it
 *  now points at. */
const OVERRIDE_BLOCKS = new Map<string, Map<string, string>>()

/** What both sides of the lookup are reduced to: trimmed, with every run of whitespace one
 *  space. The question the harness passes and the same question written into the file
 *  differ by a line break or a trailing space far more often than by a word, and a lookup
 *  that missed on that would abandon the arm over formatting. */
function questionKey(question: string): string {
  return question.trim().replace(/\s+/g, " ")
}

function loadContextFile(path: string): Map<string, string> {
  const cached = OVERRIDE_BLOCKS.get(path)
  if (cached) return cached
  const blocks = new Map<string, string>()
  const lines = readFileSync(path, "utf8").split("\n")
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    if (line.trim() === "") continue
    const row = JSON.parse(line) as { question?: unknown; block?: unknown }
    if (typeof row.question !== "string" || typeof row.block !== "string") {
      throw new Error(
        `MEMVARA_CONTEXT_FILE ${path} line ${i + 1} needs a string "question" and a string "block"`
      )
    }
    blocks.set(questionKey(row.question), row.block)
  }
  OVERRIDE_BLOCKS.set(path, blocks)
  return blocks
}

/** The pre-rendered block for one question, or a throw. A question the file does not cover
 *  cannot fall back to this provider's rendering: half an arm rendered one way and half the
 *  other scores as neither, and nothing in the report would say which questions were which. */
function overrideBlock(path: string, question: string): string {
  const block = loadContextFile(path).get(questionKey(question))
  if (block === undefined) {
    throw new Error(
      `MEMVARA_CONTEXT_FILE ${path} has no block for question: ${question.slice(0, 80)}`
    )
  }
  return block
}

/** Memories first, then the raw turns, each in the order memvara ranked them. With every
 *  knob at its default nothing is dropped, merged or re-sorted here: this is a rendering
 *  of the ranking, and the ranking is what the benchmark measures.
 *
 *  The knobs apply in a fixed order -- role selection, then rank-aware truncation, then
 *  the token budget -- so a turn's rank for truncation is its position after the role
 *  filter, and the budget measures lines as they will actually be rendered.
 *
 *  `question` is only needed by the "route" arm of MEMVARA_ROLE_SELECT. A caller that
 *  omits it renders exactly what it rendered before.
 *
 *  MEMVARA_CONTEXT_FILE replaces all of it with a block rendered elsewhere, and is read
 *  first because none of the knobs below apply to a block this code did not build.
 *
 *  It is skipped when `context` is empty, and that exception is what keeps the reported
 *  contextTokens right. The harness answers each question by building this prompt twice --
 *  once with the retrieved context and once with an empty one -- and reports the difference
 *  in tokens as the context's cost. Overriding the empty build as well would make the two
 *  prompts identical and record every question of the run as having cost no context. */
export function renderMemvaraContext(context: unknown[], question?: string): string {
  const override = contextFile()
  if (override !== null && context.length > 0) {
    return overrideBlock(override, question ?? "")
  }
  const dropClaims = turnsOnly()
  const mode = roleSelect()
  const budget = tokenBudget()
  // Read whether or not anything is truncated, so a typo in either truncation knob throws
  // whatever the other one is set to.
  const knobs = truncationKnobs()
  const memories = dropClaims ? [] : context.filter(isMemory)
  const turns = selectTurns(context.filter(isTurn), mode, question)
  if (memories.length === 0 && turns.length === 0) {
    return "No memories were retrieved."
  }
  const parts: string[] = []
  if (memories.length > 0) {
    parts.push(
      "Memories (each with the period it was true for and the date it was recorded):\n" +
        memories.map(memoryLine).join("\n")
    )
  }
  if (turns.length > 0) {
    const rendered = turns.map((t, i) => renderTurn(t, i, knobs))
    const kept = budget === null ? rendered : fillToBudget(rendered, budget)
    const header = turnsHeader(kept.some((r) => r.cut))
    parts.push([header, ...kept.map((r) => r.line)].join("\n"))
  }
  return parts.join("\n\n")
}

function selectTurns(
  turns: MemvaraContextTurn[],
  mode: RoleSelect,
  question?: string
): MemvaraContextTurn[] {
  if (mode === "off") return turns
  const keep = mode === "route" && wantsAssistant(question ?? "") ? "assistant" : "user"
  const kept = turns.filter((t) => t.role === keep)
  if (kept.length === 0 && turns.length > 0) {
    // Every retrieved turn has just been thrown away by a rendering knob, and the prompt
    // that goes out will look like a question memvara retrieved nothing for. Say which knob
    // did it, against which question, or the run's log records only the low score.
    logger.warn(
      `MEMVARA_ROLE_SELECT=${mode} kept no "${keep}" turns of the ${turns.length} retrieved ` +
        `for question: ${(question ?? "").slice(0, 60)}`
    )
  }
  return kept
}

/** The three reading rules MEMVARA_ANSWER_PROMPT=v2 adds to "How to read the context",
 *  each aimed at one shape of failure seen in the 13 judged questions that had every gold
 *  excerpt in the prompt and still answered wrongly: the stale value of a fact that a later
 *  excerpt updates, a count or a list stopped after the first few matches, and advice that
 *  never touches what the user has said about themselves. */
export const V2_CONTEXT_BULLETS: readonly string[] = [
  "- When two excerpts give different values for the same fact — where something is kept, what a number is, what was decided — the excerpt with the later date is the current one. Answer with it, and mention the earlier value only if the question asks about the past.",
  "- When the question asks how many, how much in total, or which or all of something, first list every matching item from the excerpts with its date, then count or total that list. The answer is the length or sum of the list you wrote down; do not stop at the first few or say the total cannot be determined while the list is in front of you.",
  "- When the question asks what to choose, whether to do something, or for a recommendation, build the answer on the user's own stated preferences, constraints and situation from the excerpts, and say which stated preference each part of the advice rests on. Advice that ignores what the user has said about themselves is wrong even when it is sensible.",
]

/** The one instruction v2 adds, after the abstention rule rather than in place of it: the
 *  failures it addresses reasoned their way to the right values and then declined to
 *  answer, which is a different mistake from guessing. */
export const V2_INSTRUCTION =
  '- If your reasoning has already identified the values the answer needs, give the answer; say "I don\'t know" only when the excerpts contain nothing that bears on the question.'

/** The one instruction v3 adds after V2_INSTRUCTION: a question can be built on a premise
 *  the excerpts do not support -- it names one person where the excerpts show another
 *  doing the thing, or an event that nobody in the excerpts had. Answering about the nearest
 *  similar thing is a fabrication with a citation. */
export const V3_INSTRUCTION =
  "- Check the question's premise against the excerpts before answering. If the question attributes something to a person, event or object that the excerpts show belongs to a different one, or that no excerpt supports at all, say that the context does not contain it, and name what the excerpts do say instead. Do not answer about the nearest similar thing as if it were the one asked about."

/** v1 is the shipped prompt byte for byte; v2 is that prompt with four lines inserted and
 *  nothing else moved; v3 is v2 with one more instruction after the last. All read the same
 *  rendered context. */
export function buildMemvaraAnswerPrompt(
  question: string,
  context: unknown[],
  questionDate?: string
): string {
  const variant = answerPrompt()
  const v2 = variant !== "v1"
  const readingRules = v2 ? `\n${V2_CONTEXT_BULLETS.join("\n")}` : ""
  const instruction = !v2
    ? ""
    : variant === "v3"
      ? `\n${V2_INSTRUCTION}\n${V3_INSTRUCTION}`
      : `\n${V2_INSTRUCTION}`
  return `You are a question-answering system with access to a memory of past conversations with the user. Answer the question from the retrieved context below.

Question: ${question}
Question date: ${questionDate || "not specified"}

Retrieved context:
${renderMemvaraContext(context, question)}

How to read the context:
- A memory is a fact the memory system extracted. "valid from" is when the fact became true in the world; "recorded" is when the system learned it. A memory marked "ended" was true for the period shown and has since been replaced; prefer the "live" memory for what is true now, and use "ended" memories for what was true earlier.
- A conversation excerpt is what was actually said, with the date it was said. Use excerpts for details and wording that a memory summarises.
- Resolve relative expressions such as "today", "yesterday", "last week" or "in two months" against the date of the excerpt or memory they appear in, never against the current date. Use the question date only to understand what the question is asking about.${readingRules}

Instructions:
- Think through the problem step by step first.
- Identify which memories and excerpts are relevant, and whether any memory has been updated by a later one.
- If the context contains enough information, give a clear, concise answer.
- If it does not, answer "I don't know" and say what is missing. Do not guess.
- Base the answer only on the context above.${instruction}

Response format:

Reasoning:
[your step-by-step reasoning]

Answer:
[your final answer]`
}

export const MEMVARA_PROMPTS: ProviderPrompts = {
  answerPrompt: buildMemvaraAnswerPrompt,
}

export default MEMVARA_PROMPTS

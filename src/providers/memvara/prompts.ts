import type { ProviderPrompts } from "../../types/prompts"
import { countO200k } from "../../utils/tokens"

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

/** A raw conversation turn, present when the search included episodes. */
export interface MemvaraContextTurn {
  kind: "turn"
  role: string
  content: string
  ts: string
  score: number
}

export type MemvaraContextItem = MemvaraContextMemory | MemvaraContextTurn

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

/** How many of the highest-ranked turns are rendered whole, and how far the rest are cut.
 *
 *  Assistant turns are 87% of the context memvara hands over -- a median of 515 tokens
 *  against 67 for a user turn. Whether they can be dropped is not settled, because the two
 *  measurements we have disagree. The answer string appears in an assistant turn 69% of the
 *  time against 49% for user turns; against the dataset's own has_answer labels, though,
 *  842 of the 896 gold turns are user turns, and 51 of the 54 assistant gold turns belong
 *  to single-session-assistant. MEMVARA_ROLE_SELECT below is how that gets settled; these
 *  two knobs take the other route and save from inside a turn rather than by choosing
 *  different ones.
 *
 *  Measured over the 108 questions of a 199-question run whose answer text was retrieved
 *  in full: cutting every turn to 800 characters keeps 90.7% of them and saves 54% of the
 *  tokens, while keeping the top five whole and cutting the rest to 400 keeps 96.3% and
 *  saves 51%. Rank-aware wins at every token budget, which is what one would hope -- the
 *  turns most likely to hold the answer are the ones ranked highest.
 */
/** Reads one of the two truncation knobs. Unset, empty or `0` means off. Anything else
 *  that is not a non-negative integer throws, for the same reason MEMVARA_TOKEN_BUDGET
 *  does: `MEMVARA_TAIL_CHARS=80O` coerces to NaN, every comparison against NaN is false,
 *  and the arm would then run the control under the arm's name with nothing anywhere
 *  saying so. */
function intKnob(name: string): number {
  const raw = process.env[name]
  if (raw === undefined || raw.trim() === "") return 0
  const n = Number(raw)
  if (!Number.isInteger(n) || n < 0) {
    throw new Error(`${name} must be a non-negative integer, got "${raw}"`)
  }
  return n
}

function headWhole(): number {
  return intKnob("MEMVARA_HEAD_WHOLE")
}

function tailChars(): number {
  return intKnob("MEMVARA_TAIL_CHARS")
}

function turnLine(t: MemvaraContextTurn, rank: number): string {
  const tail = tailChars()
  const content =
    tail > 0 && rank >= headWhole() && t.content.length > tail
      ? `${t.content.slice(0, tail)}…`
      : t.content
  return `- [${formatWhen(t.ts)}] ${t.role}: ${content}`
}

/** The fourteen phrasings that ask what the assistant itself said, rather than what the
 *  user said. Frozen: the point of the arm is to measure this rule, so widening it later
 *  measures a different rule. Every alternative is past tense or an explicit request to be
 *  reminded, which is what keeps "can you suggest a hotel" -- a request, not a lookup --
 *  out of it. Word boundaries on both ends, so "you recommend" does not match. */
export const ASSISTANT_QUESTION_RE =
  /\b(?:you suggested|you recommended|you mentioned|you told me|you provided|you wrote|you created|did you say|can you remind me|remind me what|remind me which|remind me who|remind me how|remind me of)\b/i

/** True when the question is asking about the assistant's own past output. */
export function wantsAssistant(question: string): boolean {
  return ASSISTANT_QUESTION_RE.test(question)
}

type RoleSelect = "off" | "user" | "route"

/** MEMVARA_ROLE_SELECT chooses which turns reach the prompt. "off" is the shipped
 *  behaviour and renders every turn memvara returned. "user" keeps user turns only, which
 *  is the arm the has_answer labels argue for. "route" keeps assistant turns only when the
 *  question asks what the assistant said, and user turns otherwise. Retrieval is untouched
 *  either way: this is a rendering choice, so the ranking still decides what is available. */
function roleSelect(): RoleSelect {
  const raw = process.env.MEMVARA_ROLE_SELECT
  if (raw === undefined || raw === "") return "off"
  if (raw === "off" || raw === "user" || raw === "route") return raw
  throw new Error(`MEMVARA_ROLE_SELECT must be "off", "user" or "route", got "${raw}"`)
}

/** MEMVARA_TOKEN_BUDGET caps the turns block, in o200k_base tokens -- the encoder the
 *  harness reports contextTokens with. Absent or empty means no cap. */
function tokenBudget(): number | null {
  const raw = process.env.MEMVARA_TOKEN_BUDGET
  if (raw === undefined || raw === "") return null
  const n = Number(raw)
  if (!Number.isInteger(n) || n <= 0) {
    throw new Error(`MEMVARA_TOKEN_BUDGET must be a positive integer, got "${raw}"`)
  }
  return n
}

const TURNS_HEADER = "Conversation excerpts (verbatim, with the date they were said):"

/** Fills the turns block greedily in memvara's order and stops at the first line that
 *  would push the whole block -- header included -- past the budget. The first line is
 *  always kept, even alone over budget, because a block of nothing but a header answers
 *  no question at all. Nothing is skipped over and nothing is re-sorted: a budget that
 *  reordered the turns would measure a different ranking from the one being benchmarked. */
function fillToBudget(lines: string[], budget: number): string[] {
  const kept: string[] = []
  for (const line of lines) {
    if (kept.length > 0 && countO200k([TURNS_HEADER, ...kept, line].join("\n")) > budget) break
    kept.push(line)
  }
  return kept
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
 *  omits it renders exactly what it rendered before. */
export function renderMemvaraContext(context: unknown[], question?: string): string {
  // MEMVARA_TURNS_ONLY drops the claims from the prompt while leaving retrieval exactly
  // as it was. Three arms have now shown a model-ingest run scoring below the fast path
  // while retrieving *more* turns and twice the context, which points at the claims
  // sitting alongside them rather than at anything missing. This isolates that: same
  // ranking, same turns, claims removed.
  const turnsOnly = process.env.MEMVARA_TURNS_ONLY === "1"
  const mode = roleSelect()
  const budget = tokenBudget()
  const memories = turnsOnly ? [] : context.filter(isMemory)
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
    const lines = turns.map((t, i) => turnLine(t, i))
    parts.push(
      [TURNS_HEADER, ...(budget === null ? lines : fillToBudget(lines, budget))].join("\n")
    )
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
  return turns.filter((t) => t.role === keep)
}

export function buildMemvaraAnswerPrompt(
  question: string,
  context: unknown[],
  questionDate?: string
): string {
  return `You are a question-answering system with access to a memory of past conversations with the user. Answer the question from the retrieved context below.

Question: ${question}
Question date: ${questionDate || "not specified"}

Retrieved context:
${renderMemvaraContext(context, question)}

How to read the context:
- A memory is a fact the memory system extracted. "valid from" is when the fact became true in the world; "recorded" is when the system learned it. A memory marked "ended" was true for the period shown and has since been replaced; prefer the "live" memory for what is true now, and use "ended" memories for what was true earlier.
- A conversation excerpt is what was actually said, with the date it was said. Use excerpts for details and wording that a memory summarises.
- Resolve relative expressions such as "today", "yesterday", "last week" or "in two months" against the date of the excerpt or memory they appear in, never against the current date. Use the question date only to understand what the question is asking about.

Instructions:
- Think through the problem step by step first.
- Identify which memories and excerpts are relevant, and whether any memory has been updated by a later one.
- If the context contains enough information, give a clear, concise answer.
- If it does not, answer "I don't know" and say what is missing. Do not guess.
- Base the answer only on the context above.

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

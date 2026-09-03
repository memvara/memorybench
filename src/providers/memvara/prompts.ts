import type { ProviderPrompts } from "../../types/prompts"
import { logger } from "../../utils/logger"
import { countO200k } from "../../utils/tokens"
import { roleSelect, tokenBudget, truncationKnobs, turnsOnly } from "./env"
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

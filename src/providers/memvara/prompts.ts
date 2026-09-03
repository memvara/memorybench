import type { ProviderPrompts } from "../../types/prompts"

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
 *  against 67 for a user turn -- and they cannot simply be dropped, because the answer
 *  text sits in an assistant turn 69% of the time against 49% for user turns. So the
 *  saving has to come from inside the turn rather than from choosing different ones.
 *
 *  Measured over the 108 questions of a 199-question run whose answer text was retrieved
 *  in full: cutting every turn to 800 characters keeps 90.7% of them and saves 54% of the
 *  tokens, while keeping the top five whole and cutting the rest to 400 keeps 96.3% and
 *  saves 51%. Rank-aware wins at every token budget, which is what one would hope -- the
 *  turns most likely to hold the answer are the ones ranked highest.
 */
const HEAD_WHOLE = Number(process.env.MEMVARA_HEAD_WHOLE ?? 0)
const TAIL_CHARS = Number(process.env.MEMVARA_TAIL_CHARS ?? 0)

function turnLine(t: MemvaraContextTurn, rank: number): string {
  const content =
    TAIL_CHARS > 0 && rank >= HEAD_WHOLE && t.content.length > TAIL_CHARS
      ? `${t.content.slice(0, TAIL_CHARS)}…`
      : t.content
  return `- [${formatWhen(t.ts)}] ${t.role}: ${content}`
}

/** Memories first, then the raw turns, each in the order memvara ranked them. Nothing
 *  is dropped, merged or re-sorted here: this is a rendering of the ranking, and the
 *  ranking is what the benchmark measures. */
export function renderMemvaraContext(context: unknown[]): string {
  // MEMVARA_TURNS_ONLY drops the claims from the prompt while leaving retrieval exactly
  // as it was. Three arms have now shown a model-ingest run scoring below the fast path
  // while retrieving *more* turns and twice the context, which points at the claims
  // sitting alongside them rather than at anything missing. This isolates that: same
  // ranking, same turns, claims removed.
  const turnsOnly = process.env.MEMVARA_TURNS_ONLY === "1"
  const memories = turnsOnly ? [] : context.filter(isMemory)
  const turns = context.filter(isTurn)
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
    parts.push(
      "Conversation excerpts (verbatim, with the date they were said):\n" +
        turns.map((t, i) => turnLine(t, i)).join("\n")
    )
  }
  return parts.join("\n\n")
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
${renderMemvaraContext(context)}

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

// src/providers/memvara/env.ts

/** Every knob the memvara arms are driven by, resolved from the environment at call time
 *  rather than at import, so a run can set them per arm and one test cannot leak into the
 *  next.
 *
 *  One rule covers all eight: **unset, empty, or only whitespace means the knob is off**.
 *  `MEMVARA_TOKEN_BUDGET=` and `MEMVARA_TOKEN_BUDGET=" "` are both a run script clearing a
 *  knob, not a request, and the four numeric knobs used to disagree about which of those
 *  two was a clear and which was an error.
 *
 *  Anything else must be a value the knob accepts, and a value it does not accept throws
 *  and names the variable rather than coercing. `MEMVARA_TAIL_CHARS=80O` is the typo this
 *  guards: `Number("80O")` is NaN, every comparison against NaN is false, and the arm would
 *  then run the control under the arm's name with nothing anywhere saying so.
 *
 *  Zero is where the knobs differ, because it means different things. `MEMVARA_HEAD_WHOLE`
 *  and `MEMVARA_TAIL_CHARS` are lengths, so zero is off and is accepted. `MEMVARA_SEARCH_K`
 *  and `MEMVARA_TOKEN_BUDGET` are sizes of something that has to exist -- a search that
 *  asks for nothing and a budget that fits nothing are not arms anybody means to run -- so
 *  zero throws. */

/** `k: 30` is what the shipped providers ask their services for. MEMVARA_SEARCH_K overrides
 *  it so an arm can measure how much of the score comes from depth of retrieval rather than
 *  from the prompt. */
export const DEFAULT_SEARCH_K = 30

/** Parses one integer knob. Returns null when the variable is off, so each caller states
 *  its own default rather than sharing one. `min` is 0 for a knob where zero means off and
 *  1 for a knob where zero is a mistake. */
function envInt(name: string, min: 0 | 1): number | null {
  const raw = process.env[name]
  if (raw === undefined || raw.trim() === "") return null
  const n = Number(raw)
  if (!Number.isInteger(n) || n < min) {
    const kind = min === 0 ? "a non-negative integer" : "a positive integer"
    throw new Error(`${name} must be ${kind}, got "${raw}"`)
  }
  return n
}

/** Parses one string knob against a closed set, with the same off rule as `envInt`. */
function envEnum<T extends string>(name: string, allowed: readonly T[], off: T): T {
  const raw = process.env[name]
  if (raw === undefined || raw.trim() === "") return off
  if ((allowed as readonly string[]).includes(raw)) return raw as T
  const list = allowed.map((v) => `"${v}"`).join(", ")
  throw new Error(`${name} must be one of ${list}, got "${raw}"`)
}

/** MEMVARA_TURNS_ONLY drops the claims from the prompt while leaving retrieval exactly as
 *  it was. Three arms have now shown a model-ingest run scoring below the fast path while
 *  retrieving *more* turns and twice the context, which points at the claims sitting
 *  alongside them rather than at anything missing. This isolates that: same ranking, same
 *  turns, claims removed.
 *
 *  Only `"1"` switches it on. It used to accept anything and compare it to `"1"`, so
 *  `MEMVARA_TURNS_ONLY=true` and `MEMVARA_TURNS_ONLY=yes` ran the control while reading, to
 *  whoever wrote the run script, as the arm. */
export function turnsOnly(): boolean {
  const raw = process.env.MEMVARA_TURNS_ONLY
  if (raw === undefined || raw.trim() === "") return false
  if (raw === "1") return true
  throw new Error(`MEMVARA_TURNS_ONLY must be "1" or unset, got "${raw}"`)
}

export type RoleSelect = "off" | "user" | "route"

/** MEMVARA_ROLE_SELECT chooses which turns reach the prompt. "off" is the shipped
 *  behaviour and renders every turn memvara returned. "user" keeps user turns only, which
 *  is the arm the has_answer labels argue for. "route" keeps assistant turns only when the
 *  question asks what the assistant said, and user turns otherwise. Retrieval is untouched
 *  either way: this is a rendering choice, so the ranking still decides what is available. */
export function roleSelect(): RoleSelect {
  return envEnum("MEMVARA_ROLE_SELECT", ["off", "user", "route"] as const, "off")
}

/** How many of the highest-ranked turns are rendered whole, and how far the rest are cut.
 *  Both are read together on every render, so a typo in either one is caught whatever the
 *  other is set to.
 *
 *  Assistant turns are 87% of the context memvara hands over -- a median of 515 tokens
 *  against 67 for a user turn. Whether they can be dropped is not settled, because the two
 *  measurements we have disagree. The answer string appears in an assistant turn 69% of the
 *  time against 49% for user turns; against the dataset's own has_answer labels, though,
 *  842 of the 896 gold turns are user turns, and 51 of the 54 assistant gold turns belong
 *  to single-session-assistant. MEMVARA_ROLE_SELECT is how that gets settled; these two
 *  knobs take the other route and save from inside a turn rather than by choosing
 *  different ones.
 *
 *  Measured over the 108 questions of a 199-question run whose answer text was retrieved
 *  in full: cutting every turn to 800 characters keeps 90.7% of them and saves 54% of the
 *  tokens, while keeping the top five whole and cutting the rest to 400 keeps 96.3% and
 *  saves 51%. Rank-aware wins at every token budget, which is what one would hope -- the
 *  turns most likely to hold the answer are the ones ranked highest. */
export interface TruncationKnobs {
  headWhole: number
  tailChars: number
}

export function truncationKnobs(): TruncationKnobs {
  return {
    headWhole: envInt("MEMVARA_HEAD_WHOLE", 0) ?? 0,
    tailChars: envInt("MEMVARA_TAIL_CHARS", 0) ?? 0,
  }
}

/** MEMVARA_TOKEN_BUDGET caps the turns block, in o200k_base tokens. Absent means no cap.
 *
 *  o200k_base is the encoding `countTokens` uses only for the model ids `getEncoder` maps
 *  to it. For those the budget and the contextTokens in the report are the same count.
 *  Answer with any other id and the report counts in whatever encoder that id resolves to
 *  while the budget still counts in o200k_base, so the two numbers describe the same text
 *  on different scales. */
export function tokenBudget(): number | null {
  return envInt("MEMVARA_TOKEN_BUDGET", 1)
}

export function searchK(): number {
  return envInt("MEMVARA_SEARCH_K", 1) ?? DEFAULT_SEARCH_K
}

export type AnswerPrompt = "v1" | "v2"

/** MEMVARA_ANSWER_PROMPT chooses which answer prompt is built. "v1" is the shipped text,
 *  byte for byte. "v2" keeps every line of it and adds four lines: three reading rules and
 *  one instruction.
 *
 *  The four are aimed at 13 judged questions where every gold excerpt was already in the
 *  prompt and the answer was still wrong -- so nothing about retrieval, the budget or the
 *  rendering would have saved them. They divided into three shapes: a later excerpt
 *  contradicting an earlier one and the earlier value being answered with; a "how many" or
 *  "which" question answered from the first few matching excerpts or abandoned as
 *  undeterminable; and a request for advice answered sensibly but without the user's own
 *  stated preferences. The fourth line closes the gap where the reasoning names the right
 *  values and the answer is "I don't know" anyway.
 *
 *  This is a prompt arm and nothing else: retrieval, selection, truncation and the budget
 *  are untouched, so the two prompts see exactly the same context block. */
export function answerPrompt(): AnswerPrompt {
  return envEnum("MEMVARA_ANSWER_PROMPT", ["v1", "v2"] as const, "v1")
}

/** MEMVARA_CONTEXT_FILE is a path to a JSONL file of context blocks rendered somewhere
 *  else, one object per line: `{"question": "...", "block": "..."}`. When it is set the
 *  matching block is what the answer prompt carries, and this provider's own rendering --
 *  role selection, truncation, the budget -- does not run at all. That is how a context
 *  built by another pipeline is judged on these questions with the same prompt and the same
 *  answering model, so the context block is the only thing that differs between the runs.
 *
 *  A path is the one knob with nothing to check it against, so it is taken as written apart
 *  from the whitespace around it. That is also why a mistyped path throws when the file is
 *  first read rather than at startup like the other seven. */
export function contextFile(): string | null {
  const raw = process.env.MEMVARA_CONTEXT_FILE
  if (raw === undefined || raw.trim() === "") return null
  return raw.trim()
}

/** Every knob as this process resolved it. Logged once at provider init, so an arm's
 *  configuration is in its own log instead of in whoever launched it -- a run whose score
 *  cannot be attributed to a configuration is a run that has to be repeated. Reading them
 *  all here also means a typo in any of the seven knobs that check their value throws at
 *  startup rather than at the first question that happens to touch it. MEMVARA_CONTEXT_FILE
 *  is the eighth and has no value to check, so the path is logged and read later. */
export interface MemvaraSettings {
  turnsOnly: boolean
  roleSelect: RoleSelect
  headWhole: number
  tailChars: number
  tokenBudget: number | null
  searchK: number
  answerPrompt: AnswerPrompt
  contextFile: string | null
}

export function memvaraProviderSettings(): MemvaraSettings {
  const truncation = truncationKnobs()
  return {
    turnsOnly: turnsOnly(),
    roleSelect: roleSelect(),
    headWhole: truncation.headWhole,
    tailChars: truncation.tailChars,
    tokenBudget: tokenBudget(),
    searchK: searchK(),
    answerPrompt: answerPrompt(),
    contextFile: contextFile(),
  }
}

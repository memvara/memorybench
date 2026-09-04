// src/scripts/score-ranked.ts
//
// The offline screen §6 of the model-ranked-recall design calls for before any judged
// spend: score a ranked run's kept set -- the turns the selector marked `selected: true`
// -- against LongMemEval's own has_answer labels, and report gold recall and the non-gold
// keep rate. That is the same pair of numbers `local/compress/extract.py` prints for the
// candidate list it built by hand from a cached model call; this reads them instead from
// the checkpoint a real `MEMVARA_RANKED=1` run against the server actually produced.
//
// Usage: bun run src/scripts/score-ranked.ts <runId> [datasetPath]

import { readFileSync } from "fs"
import { CheckpointManager } from "../orchestrator/checkpoint"
import type { RunCheckpoint } from "../types/checkpoint"
import type { MemvaraContextItem } from "../providers/memvara/prompts"

export const DEFAULT_DATASET = "./data/benchmarks/longmemeval/datasets/longmemeval_s_cleaned.json"

interface DatasetMessage {
  content: string
  has_answer?: boolean
}

interface DatasetItem {
  question_id: string
  haystack_sessions: DatasetMessage[][]
}

/** The has_answer turns' own text, by question id. `loadQuestions` in
 *  `benchmarks/longmemeval/index.ts` deletes this flag when it splits the raw dataset into
 *  the per-question files the benchmark runs from, so it has to be read from the raw file
 *  every time -- a run's own checkpoint never carries it. */
export function goldByQuestion(items: DatasetItem[]): Map<string, Set<string>> {
  const gold = new Map<string, Set<string>>()
  for (const item of items) {
    const contents = new Set<string>()
    for (const session of item.haystack_sessions) {
      for (const message of session) {
        if (message.has_answer) contents.add(message.content)
      }
    }
    gold.set(item.question_id, contents)
  }
  return gold
}

export interface ScoreResult {
  questionsScored: number
  goldKept: number
  goldMissed: number
  nonGoldKept: number
  nonGoldMissed: number
  sawRanking: boolean
}

function isTurn(x: unknown): x is MemvaraContextItem & { kind: "turn" } {
  return typeof x === "object" && x !== null && (x as { kind?: unknown }).kind === "turn"
}

/** Classifies every turn a search phase returned into gold/non-gold and kept/not-kept, the
 *  way `extract.py` classifies its own candidate list. "Kept" is `selected === true`: the
 *  model named the turn. `selected === false` (seen, not kept) and `undefined` (a plain
 *  read, or a turn the selector never saw) both count as not kept -- only the first tells
 *  the caller this checkpoint has ranking to score at all, which is `sawRanking`.
 *
 *  A question with no search results, or with no has_answer turns of its own (an
 *  abstention question, which the routed candidate list may still contain by accident),
 *  does not count toward either total: there is nothing for it to be counted as missing. */
export function score(checkpoint: RunCheckpoint, gold: Map<string, Set<string>>): ScoreResult {
  let questionsScored = 0
  let goldKept = 0
  let goldMissed = 0
  let nonGoldKept = 0
  let nonGoldMissed = 0
  let sawRanking = false

  for (const question of Object.values(checkpoint.questions)) {
    const results = question.phases.search.results
    const goldSet = gold.get(question.questionId)
    if (!results || !goldSet || goldSet.size === 0) continue
    questionsScored++
    for (const item of results) {
      if (!isTurn(item)) continue
      if (item.selected !== undefined) sawRanking = true
      const kept = item.selected === true
      if (goldSet.has(item.content)) {
        if (kept) goldKept++
        else goldMissed++
      } else {
        if (kept) nonGoldKept++
        else nonGoldMissed++
      }
    }
  }

  return { questionsScored, goldKept, goldMissed, nonGoldKept, nonGoldMissed, sawRanking }
}

/** The shape `extract.py` prints: "filter: gold kept G / G+M (recall R), non-gold kept
 *  K / K+U (rate)." */
export function formatScore(r: ScoreResult): string {
  const goldTotal = r.goldKept + r.goldMissed
  const nonGoldTotal = r.nonGoldKept + r.nonGoldMissed
  const recall = goldTotal ? (r.goldKept / goldTotal).toFixed(3) : "n/a"
  const keepRate = nonGoldTotal ? (r.nonGoldKept / nonGoldTotal).toFixed(3) : "n/a"
  return (
    `scored ${r.questionsScored} questions: ` +
    `gold kept ${r.goldKept} / ${goldTotal} (recall ${recall}), ` +
    `non-gold kept ${r.nonGoldKept} / ${nonGoldTotal} (${keepRate})`
  )
}

function main(): void {
  const [runId, datasetPath = DEFAULT_DATASET] = process.argv.slice(2)
  if (!runId) {
    console.log("Usage: bun run src/scripts/score-ranked.ts <runId> [datasetPath]")
    process.exitCode = 1
    return
  }

  const checkpoint = new CheckpointManager().load(runId)
  if (!checkpoint) {
    console.error(`No checkpoint found for run ${runId}`)
    process.exitCode = 1
    return
  }

  const items: DatasetItem[] = JSON.parse(readFileSync(datasetPath, "utf8"))
  const result = score(checkpoint, goldByQuestion(items))

  if (!result.sawRanking) {
    console.error(
      `No result in run ${runId} carries a "selected" field -- this is not a ranked ` +
        "search (MEMVARA_RANKED=1) against a server that returns the selector's outcome, " +
        "so there is nothing to score."
    )
    process.exitCode = 1
    return
  }

  console.log(formatScore(result))
}

if (import.meta.main) main()

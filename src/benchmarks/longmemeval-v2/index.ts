import { existsSync, readFileSync } from "fs"
import { join } from "path"
import type { Benchmark, BenchmarkConfig, QuestionFilter } from "../../types/benchmark"
import type {
  QuestionTypeRegistry,
  UnifiedMessage,
  UnifiedQuestion,
  UnifiedSession,
} from "../../types/unified"
import { logger } from "../../utils/logger"
import type { LongMemEvalV2Question, LongMemEvalV2State, LongMemEvalV2Trajectory } from "./types"

const DEFAULT_DATA_PATH = "./data/benchmarks/longmemeval-v2"
const DEFAULT_TIER = "small"
const TREE_EXCERPT_CHAR_LIMIT = 12000
const COMPACT_UI_CHAR_LIMIT = 12000

type HaystackMap = Record<string, string[]>

function readJsonl<T>(path: string): T[] {
  return readFileSync(path, "utf8")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as T)
}

function requireFile(path: string, description: string): void {
  if (!existsSync(path)) {
    throw new Error(
      `Missing ${description}: ${path}\n` +
        "Download LongMemEval-V2 data first, then set BenchmarkConfig.dataPath or place files under data/benchmarks/longmemeval-v2."
    )
  }
}

function truncate(value: string, limit: number): string {
  if (value.length <= limit) return value
  return `${value.slice(0, limit)}\n...[truncated ${value.length - limit} chars]`
}

function uniquePush(items: string[], seen: Set<string>, value: string): void {
  const normalized = value.replace(/\s+/g, " ").trim()
  if (!normalized || normalized.length < 2 || seen.has(normalized)) return
  seen.add(normalized)
  items.push(normalized)
}

function extractQuotedLabels(line: string): string[] {
  const labels: string[] = []
  const patterns = [
    /\b(?:button|link|menuitem|option|combobox|textbox|searchbox|checkbox|heading|gridcell|cell|rowheader|columnheader|StaticText)\s+'([^']+)'/g,
    /\bvalue='([^']+)'/g,
    /\bplaceholder='([^']+)'/g,
  ]

  for (const pattern of patterns) {
    for (const match of line.matchAll(pattern)) {
      labels.push(match[1])
    }
  }

  return labels
}

function compactAccessibilityTree(tree?: string | null): string {
  if (!tree) return ""

  const roleLinePattern =
    /\b(button|link|menuitem|option|combobox|textbox|searchbox|checkbox|heading|gridcell|cell|rowheader|columnheader|StaticText|listitem)\b/
  const labels: string[] = []
  const roleLines: string[] = []
  const seenLabels = new Set<string>()
  const seenLines = new Set<string>()

  for (const rawLine of tree.split(/\r?\n/)) {
    const line = rawLine.trim()
    if (!line) continue

    for (const label of extractQuotedLabels(line)) {
      uniquePush(labels, seenLabels, label)
    }

    if (roleLinePattern.test(line)) {
      uniquePush(roleLines, seenLines, line)
    }

    const compactLength = labels.join("\n").length + roleLines.join("\n").length
    if (compactLength > COMPACT_UI_CHAR_LIMIT) break
  }

  const sections: string[] = []
  if (labels.length) {
    sections.push(`UI labels and values:\n${labels.map((label) => `- ${label}`).join("\n")}`)
  }
  if (roleLines.length) {
    sections.push(`Relevant accessibility lines:\n${roleLines.join("\n")}`)
  }

  return truncate(sections.join("\n\n"), COMPACT_UI_CHAR_LIMIT)
}

function stateToSession(
  trajectory: LongMemEvalV2Trajectory,
  state: LongMemEvalV2State
): UnifiedSession {
  const compactTree = compactAccessibilityTree(state.accessibility_tree)
  const treeExcerpt = state.accessibility_tree
    ? truncate(state.accessibility_tree, TREE_EXCERPT_CHAR_LIMIT)
    : ""

  const content = [
    "LongMemEval-V2 agent trajectory state",
    `Trajectory ID: ${trajectory.id}`,
    `Domain: ${trajectory.domain}`,
    `Environment: ${trajectory.environment}`,
    `Outcome: ${trajectory.outcome || "unknown"}`,
    `Goal: ${trajectory.goal}`,
    `State index: ${state.state_index}`,
    state.step !== undefined ? `Step: ${state.step}` : "",
    state.url ? `URL: ${state.url}` : "",
    state.action ? `Action: ${state.action}` : "Action: null",
    state.thought ? `Agent thought: ${state.thought}` : "",
    state.screenshot ? `Screenshot path: ${state.screenshot}` : "",
    compactTree ? `\nCompact UI extraction:\n${compactTree}` : "",
    treeExcerpt ? `\nAccessibility tree excerpt:\n${treeExcerpt}` : "",
  ]
    .filter(Boolean)
    .join("\n")

  return {
    sessionId: `lme-v2-${trajectory.id}-state-${state.state_index}`,
    messages: [{ role: "assistant", content }],
    metadata: {
      benchmark: "longmemeval-v2",
      trajectoryId: trajectory.id,
      stateIndex: state.state_index,
      domain: trajectory.domain,
      environment: trajectory.environment,
      outcome: trajectory.outcome,
      url: state.url,
      screenshot: state.screenshot,
    },
  }
}

function trajectoryOverviewToSession(trajectory: LongMemEvalV2Trajectory): UnifiedSession {
  const actionTrace = trajectory.states
    .map((state) => {
      const action = state.action || "null"
      const thought = state.thought ? ` | thought: ${state.thought}` : ""
      return `- state ${state.state_index}: action=${action}${thought}`
    })
    .join("\n")

  const messages: UnifiedMessage[] = [
    {
      role: "user",
      content: [
        "LongMemEval-V2 trajectory overview",
        `Trajectory ID: ${trajectory.id}`,
        `Domain: ${trajectory.domain}`,
        `Environment: ${trajectory.environment}`,
        `Outcome: ${trajectory.outcome || "unknown"}`,
        `Start URL: ${trajectory.start_url || "unknown"}`,
        `Goal: ${trajectory.goal}`,
      ].join("\n"),
    },
    {
      role: "assistant",
      content: `Action/thought trace:\n${truncate(actionTrace, TREE_EXCERPT_CHAR_LIMIT)}`,
    },
  ]

  return {
    sessionId: `lme-v2-${trajectory.id}-overview`,
    messages,
    metadata: {
      benchmark: "longmemeval-v2",
      trajectoryId: trajectory.id,
      domain: trajectory.domain,
      environment: trajectory.environment,
      outcome: trajectory.outcome,
      sessionType: "trajectory-overview",
    },
  }
}

export class LongMemEvalV2Benchmark implements Benchmark {
  name = "longmemeval-v2"
  private questions: UnifiedQuestion[] = []
  private sessionsMap: Map<string, UnifiedSession[]> = new Map()
  private questionTypes: QuestionTypeRegistry = {}

  async load(config?: BenchmarkConfig): Promise<void> {
    const dataPath = config?.dataPath || DEFAULT_DATA_PATH
    const tier = process.env.LONGMEMEVAL_V2_TIER || process.env.LME_V2_TIER || DEFAULT_TIER
    const fullPath = join(process.cwd(), dataPath)
    const questionsPath = join(fullPath, "questions.jsonl")
    const trajectoriesPath = join(fullPath, "trajectories.jsonl")
    const haystackPath = join(fullPath, "haystacks", `lme_v2_${tier}.json`)

    requireFile(questionsPath, "LongMemEval-V2 questions.jsonl")
    requireFile(trajectoriesPath, "LongMemEval-V2 trajectories.jsonl")
    requireFile(haystackPath, `LongMemEval-V2 ${tier} haystack`)

    const rawQuestions = readJsonl<LongMemEvalV2Question>(questionsPath)
    const haystack = JSON.parse(readFileSync(haystackPath, "utf8")) as HaystackMap
    const haystackQuestionIds = new Set(Object.keys(haystack))
    const selectedTrajectoryIds = new Set(Object.values(haystack).flat())

    const trajectorySessions = this.loadTrajectorySessions(trajectoriesPath, selectedTrajectoryIds)

    for (const item of rawQuestions) {
      if (!haystackQuestionIds.has(item.id)) continue

      const sessions = haystack[item.id].flatMap((trajectoryId) => {
        const trajectorySessionList = trajectorySessions.get(trajectoryId)
        if (!trajectorySessionList) {
          logger.warn(`Missing LongMemEval-V2 trajectory ${trajectoryId} for question ${item.id}`)
          return []
        }
        return trajectorySessionList
      })

      this.questions.push({
        questionId: item.id,
        question: item.question,
        questionType: item.question_type,
        groundTruth: item.answer,
        haystackSessionIds: sessions.map((session) => session.sessionId),
        metadata: {
          domain: item.domain,
          environment: item.environment,
          image: item.image,
          evalFunction: item.eval_function,
          haystackTier: tier,
          trajectoryCount: haystack[item.id].length,
        },
      })

      this.sessionsMap.set(item.id, sessions)
      this.questionTypes[item.question_type] ||= {
        id: item.question_type,
        alias: item.question_type,
        description: `${item.question_type} questions`,
      }
    }

    logger.info(`Loaded ${this.questions.length} LongMemEval-V2 ${tier} questions from ${dataPath}`)
  }

  private loadTrajectorySessions(
    trajectoriesPath: string,
    selectedTrajectoryIds: Set<string>
  ): Map<string, UnifiedSession[]> {
    const sessions = new Map<string, UnifiedSession[]>()

    for (const trajectory of readJsonl<LongMemEvalV2Trajectory>(trajectoriesPath)) {
      if (!selectedTrajectoryIds.has(trajectory.id)) continue

      const trajectorySessions = [
        trajectoryOverviewToSession(trajectory),
        ...trajectory.states.map((state) => stateToSession(trajectory, state)),
      ]
      sessions.set(trajectory.id, trajectorySessions)
    }

    logger.info(`Prepared ${sessions.size} LongMemEval-V2 trajectories for ingestion`)
    return sessions
  }

  getQuestions(filter?: QuestionFilter): UnifiedQuestion[] {
    let result = [...this.questions]

    if (filter?.questionTypes?.length) {
      result = result.filter((question) => filter.questionTypes!.includes(question.questionType))
    }

    if (filter?.offset) {
      result = result.slice(filter.offset)
    }

    if (filter?.limit) {
      result = result.slice(0, filter.limit)
    }

    return result
  }

  getHaystackSessions(questionId: string): UnifiedSession[] {
    return this.sessionsMap.get(questionId) || []
  }

  getGroundTruth(questionId: string): string {
    const question = this.questions.find((item) => item.questionId === questionId)
    return question?.groundTruth || ""
  }

  getQuestionTypes(): QuestionTypeRegistry {
    return this.questionTypes
  }
}

export default LongMemEvalV2Benchmark

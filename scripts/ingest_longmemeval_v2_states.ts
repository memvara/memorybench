import { readFileSync } from "fs"
import { Supermemory } from "supermemory"

type JsonObject = Record<string, any>

const DEFAULT_DATA_ROOT = "../memory-bench/data/longmemeval-v2"
const DEFAULT_CONTAINER = "lme-v2-enterprise-small-states-v1"

function parseArg(args: string[], name: string, fallback: string): string {
  const index = args.indexOf(name)
  return index >= 0 && args[index + 1] ? args[index + 1] : fallback
}

function readJsonl(path: string): JsonObject[] {
  return readFileSync(path, "utf8")
    .split(/\r?\n/)
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line))
}

function text(value: unknown): string {
  if (value === undefined || value === null) return ""
  return typeof value === "string" ? value : JSON.stringify(value)
}

function buildStateDocument(
  trajectory: JsonObject,
  state: JsonObject,
  position: number,
  totalStates: number,
  containerTag: string
) {
  const previous = trajectory.states[position - 1]
  const next = trajectory.states[position + 1]
  const stateIndex = String(state.state_index)
  const previousId = previous ? `${trajectory.id}:${previous.state_index}` : "none"
  const nextId = next ? `${trajectory.id}:${next.state_index}` : "none"

  const content = [
    "LongMemEval-V2 trajectory state",
    `Trajectory ID: ${trajectory.id}`,
    `Domain: ${trajectory.domain}`,
    `Environment: ${trajectory.environment}`,
    `Outcome: ${trajectory.outcome || "unknown"}`,
    `Goal: ${text(trajectory.goal)}`,
    `State ID: ${trajectory.id}:${stateIndex}`,
    `State index: ${stateIndex}`,
    `State position: ${position + 1} of ${totalStates}`,
    `Previous state ID: ${previousId}`,
    `Next state ID: ${nextId}`,
    `Step: ${state.step ?? "unknown"}`,
    `URL: ${text(state.url)}`,
    `Action: ${text(state.action) || "null"}`,
    `Agent thought: ${text(state.thought)}`,
    `Screenshot path: ${text(state.screenshot)}`,
    "",
    "Accessibility tree / visible UI text:",
    text(state.accessibility_tree),
  ].join("\n")

  return {
    content,
    containerTag,
    customId: `lme-v2-state-${trajectory.id}-${stateIndex}`,
    metadata: {
      benchmark: "longmemeval-v2",
      tier: "small",
      containerTag,
      memoryKind: "state_observation",
      trajectoryId: trajectory.id,
      stateId: `${trajectory.id}:${stateIndex}`,
      stateIndex,
      previousStateId: previousId,
      nextStateId: nextId,
      domain: String(trajectory.domain || ""),
      environment: String(trajectory.environment || ""),
      outcome: String(trajectory.outcome || "unknown"),
      url: String(state.url || ""),
      screenshot: String(state.screenshot || ""),
    },
  }
}

async function main(): Promise<void> {
  const args = process.argv.slice(2)
  const dataRoot = parseArg(args, "--data-root", DEFAULT_DATA_ROOT)
  const containerTag = parseArg(args, "--container-tag", DEFAULT_CONTAINER)
  const concurrency = Number.parseInt(parseArg(args, "--concurrency", "20"), 10)
  const apiKey = process.env.SUPERMEMORY_API_KEY || process.env.SUPERMEMORY_CODEX_API_KEY
  if (!apiKey) throw new Error("Set SUPERMEMORY_API_KEY or SUPERMEMORY_CODEX_API_KEY")

  const questions = readJsonl(`${dataRoot}/questions.jsonl`)
  const enterpriseQuestion = questions.find(
    (question) => question.domain === "enterprise" && question.environment === "workarena"
  )
  if (!enterpriseQuestion) throw new Error("No enterprise/workarena question found")

  const haystacks = JSON.parse(readFileSync(`${dataRoot}/haystacks/lme_v2_small.json`, "utf8")) as Record<string, string[]>
  const trajectoryIds = haystacks[enterpriseQuestion.id]
  if (!trajectoryIds?.length) throw new Error(`No haystack found for ${enterpriseQuestion.id}`)

  const trajectories = new Map(
    readJsonl(`${dataRoot}/trajectories.jsonl`).map((trajectory) => [trajectory.id, trajectory])
  )
  const documents = trajectoryIds.flatMap((trajectoryId) => {
    const trajectory = trajectories.get(trajectoryId)
    if (!trajectory) return []
    return trajectory.states.map((state: JsonObject, position: number) =>
      buildStateDocument(trajectory, state, position, trajectory.states.length, containerTag)
    )
  })

  const client = new Supermemory({ apiKey })
  const ids: string[] = []
  let completed = 0
  console.error(`Prepared ${documents.length} state documents from ${trajectoryIds.length} trajectories`)
  console.error(`Container: ${containerTag}`)

  for (let start = 0; start < documents.length; start += Math.max(1, concurrency)) {
    const batch = documents.slice(start, start + Math.max(1, concurrency))
    const results = await Promise.all(
      batch.map((document) => client.add(document as any))
    )
    ids.push(...results.map((result) => result.id))
    completed += batch.length
    console.error(`Submitted ${completed}/${documents.length}`)
  }

  const pending = new Set(ids)
  let backoffMs = 2000
  while (pending.size > 0) {
    const statuses = await Promise.all(
      [...pending].map(async (id) => {
        const [document, memory] = await Promise.all([
          client.documents.get(id),
          client.memories.get(id),
        ])
        return { id, documentStatus: document.status, memoryStatus: memory.status }
      })
    )

    const failed = statuses.filter(
      (status) => status.documentStatus === "failed" || status.memoryStatus === "failed"
    )
    if (failed.length) {
      throw new Error(`Indexing failed for ${failed.length} state documents`)
    }

    for (const status of statuses) {
      if (status.documentStatus === "done" && status.memoryStatus === "done") {
        pending.delete(status.id)
      }
    }

    console.error(`Indexed ${ids.length - pending.size}/${ids.length}`)
    if (pending.size > 0) {
      await new Promise((resolve) => setTimeout(resolve, backoffMs))
      backoffMs = Math.min(10000, Math.round(backoffMs * 1.25))
    }
  }

  console.log(JSON.stringify({
    status: "complete",
    containerTag,
    trajectoryCount: trajectoryIds.length,
    stateDocumentCount: documents.length,
  }, null, 2))
}

await main()

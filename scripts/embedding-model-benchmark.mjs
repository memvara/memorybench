import { createHash } from "node:crypto"
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs"
import { basename, join } from "node:path"

const ROOT = process.cwd()
const DATA_DIR = join(ROOT, "data", "embedding-model-benchmark")
const CACHE_DIR = join(DATA_DIR, "cache")
const RESULTS_DIR = join(DATA_DIR, "results")
const LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
const CONVOMEM_BASE =
  "https://huggingface.co/datasets/Salesforce/ConvoMem/resolve/main/core_benchmark/pre_mixed_testcases"
const GENERAL_DATASETS = ["nfcorpus", "scifact", "arguana"]

const CATEGORY_NAMES = {
  1: "multi-hop",
  2: "temporal",
  3: "world-knowledge",
  4: "single-hop",
  5: "adversarial",
}

const CONVOMEM_PATHS = {
  user_evidence: "1_evidence",
  assistant_facts_evidence: "1_evidence",
  changing_evidence: "2_evidence",
  abstention_evidence: "1_evidence",
  preference_evidence: "1_evidence",
  implicit_connection_evidence: "1_evidence",
}

const CONFIGS = [
  {
    id: "gemini-retrieval-1536",
    provider: "gemini",
    model: "gemini-embedding-001",
    dimensions: 1536,
    queryTask: "RETRIEVAL_QUERY",
    documentTask: "RETRIEVAL_DOCUMENT",
    role: "production",
  },
  {
    id: "gemini-qa-1536",
    provider: "gemini",
    model: "gemini-embedding-001",
    dimensions: 1536,
    queryTask: "QUESTION_ANSWERING",
    documentTask: "RETRIEVAL_DOCUMENT",
    role: "task-type",
  },
  {
    id: "gemini-fact-verification-1536",
    provider: "gemini",
    model: "gemini-embedding-001",
    dimensions: 1536,
    queryTask: "FACT_VERIFICATION",
    documentTask: "RETRIEVAL_DOCUMENT",
    role: "task-type",
  },
  {
    id: "gemini-similarity-1536",
    provider: "gemini",
    model: "gemini-embedding-001",
    dimensions: 1536,
    queryTask: "SEMANTIC_SIMILARITY",
    documentTask: "SEMANTIC_SIMILARITY",
    role: "task-type",
  },
  {
    id: "gemini-document-document-1536",
    provider: "gemini",
    model: "gemini-embedding-001",
    dimensions: 1536,
    queryTask: "RETRIEVAL_DOCUMENT",
    documentTask: "RETRIEVAL_DOCUMENT",
    role: "negative-control",
  },
  {
    id: "gemini-classification-1536",
    provider: "gemini",
    model: "gemini-embedding-001",
    dimensions: 1536,
    queryTask: "CLASSIFICATION",
    documentTask: "RETRIEVAL_DOCUMENT",
    role: "negative-control",
  },
  {
    id: "gemini-clustering-1536",
    provider: "gemini",
    model: "gemini-embedding-001",
    dimensions: 1536,
    queryTask: "CLUSTERING",
    documentTask: "RETRIEVAL_DOCUMENT",
    role: "negative-control",
  },
  {
    id: "gemini-code-retrieval-1536",
    provider: "gemini",
    model: "gemini-embedding-001",
    dimensions: 1536,
    queryTask: "CODE_RETRIEVAL_QUERY",
    documentTask: "RETRIEVAL_DOCUMENT",
    role: "negative-control",
  },
  {
    id: "gemini-retrieval-768",
    provider: "gemini",
    model: "gemini-embedding-001",
    dimensions: 768,
    queryTask: "RETRIEVAL_QUERY",
    documentTask: "RETRIEVAL_DOCUMENT",
    role: "dimension",
  },
  {
    id: "gemini-retrieval-3072",
    provider: "gemini",
    model: "gemini-embedding-001",
    dimensions: 3072,
    queryTask: "RETRIEVAL_QUERY",
    documentTask: "RETRIEVAL_DOCUMENT",
    role: "dimension",
  },
  {
    id: "openai-small-1536",
    provider: "openai",
    model: "text-embedding-3-small",
    dimensions: 1536,
    role: "comparison",
  },
  {
    id: "openai-large-1536",
    provider: "openai",
    model: "text-embedding-3-large",
    dimensions: 1536,
    role: "comparison",
  },
  {
    id: "openai-large-3072",
    provider: "openai",
    model: "text-embedding-3-large",
    dimensions: 3072,
    role: "comparison",
  },
]

function parseArgs() {
  const args = process.argv.slice(2)
  const value = (flag, fallback) => {
    const index = args.indexOf(flag)
    return index === -1 ? fallback : args[index + 1]
  }
  return {
    suite: value("--suite", "all"),
    convomemSample: Number(value("--convomem-sample", "50")),
    configs: value("--configs", "all"),
    output: value("--output", new Date().toISOString().replace(/[:.]/g, "-")),
  }
}

function hash(value) {
  return createHash("sha256").update(value).digest("hex")
}

function embeddingSetHash(config, taskType, texts) {
  const digest = createHash("sha256")
  digest.update(`${config.provider}\0${config.model}\0${config.dimensions}\0${taskType ?? "default"}`)
  for (const text of texts) digest.update("\0").update(text)
  return digest.digest("hex")
}

function seededRank(id, seed = "supermemory-embedding-benchmark-v1") {
  return hash(`${seed}:${id}`)
}

async function download(url, path) {
  if (existsSync(path)) return
  const response = await fetch(url)
  if (!response.ok) throw new Error(`Download failed (${response.status}): ${url}`)
  writeFileSync(path, Buffer.from(await response.arrayBuffer()))
}

function messageText(message, context = {}) {
  const speaker = message.speaker ?? message.role ?? "unknown"
  const date = context.date ? `Date: ${context.date}\n` : ""
  return `${date}${speaker}: ${message.text ?? message.content}`
}

function loadLocomo(raw) {
  const cases = []
  for (const item of raw) {
    const documents = []
    for (let i = 1; i <= 100; i++) {
      const messages = item.conversation[`session_${i}`]
      if (!Array.isArray(messages)) break
      const date = item.conversation[`session_${i}_date_time`]
      for (let j = 0; j < messages.length; j++) {
        const message = messages[j]
        documents.push({
          id: message.dia_id,
          text: messageText(message, { date }),
          rawText: message.text,
          sessionId: `session_${i}`,
          position: j,
        })
      }
    }
    const byId = new Map(documents.map((document) => [document.id, document]))
    for (let i = 0; i < item.qa.length; i++) {
      const qa = item.qa[i]
      const evidenceIds = (qa.evidence ?? []).filter((id) => byId.has(id))
      if (qa.category === 5 || evidenceIds.length !== (qa.evidence ?? []).length) continue
      cases.push({
        id: `${item.sample_id}-q${i}`,
        suite: "locomo",
        category: CATEGORY_NAMES[qa.category],
        query: qa.question,
        answer: String(qa.answer),
        documents,
        evidenceIds,
      })
    }
  }
  return cases
}

function findEvidenceIds(documents, evidence) {
  const ids = []
  const used = new Set()
  for (const target of evidence ?? []) {
    const exact = documents.find(
      (document) =>
        !used.has(document.id) &&
        document.rawText === target.text &&
        document.speaker.toLowerCase() === String(target.speaker).toLowerCase()
    )
    if (exact) {
      ids.push(exact.id)
      used.add(exact.id)
    }
  }
  return [...new Set(ids)]
}

function loadConvoMem(files, sampleSize) {
  const cases = []
  for (const [category, testCases] of Object.entries(files)) {
    const items = testCases
      .flatMap((testCase) => testCase.evidenceItems ?? [])
      .map((item, index) => ({ item, index, rank: seededRank(`${category}:${index}:${item.question}`) }))
      .sort((a, b) => a.rank.localeCompare(b.rank))
      .slice(0, sampleSize)

    for (const { item, index } of items) {
      const documents = []
      for (let c = 0; c < (item.conversations ?? []).length; c++) {
        const conversation = item.conversations[c]
        for (let m = 0; m < (conversation.messages ?? []).length; m++) {
          const message = conversation.messages[m]
          documents.push({
            id: `${category}-${index}-c${c}-m${m}`,
            text: messageText(message),
            rawText: message.text,
            speaker: message.speaker,
            sessionId: conversation.id ?? `conversation-${c}`,
            position: m,
            order: documents.length,
          })
        }
      }
      const evidenceIds = findEvidenceIds(documents, item.message_evidences)
      if (evidenceIds.length !== (item.message_evidences ?? []).length) continue
      const evidenceDocuments = evidenceIds.map((id) => documents.find((document) => document.id === id))
      const orderedEvidence = evidenceDocuments.every(
        (document, evidenceIndex) => evidenceIndex === 0 || document.order > evidenceDocuments[evidenceIndex - 1].order
      )
      cases.push({
        id: `${category}-${index}`,
        suite: "convomem",
        category,
        query: item.question,
        answer: String(item.answer),
        documents,
        evidenceIds,
        relationEdges:
          category === "changing_evidence" && orderedEvidence
            ? evidenceIds.slice(1).map((id, relationIndex) => ({
                fromId: evidenceIds[relationIndex],
                toId: id,
                relation: "updates",
              }))
            : [],
      })
    }
  }
  return cases
}

function parseJsonLines(path) {
  return readFileSync(path, "utf8")
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line))
}

async function loadGeneralRetrieval() {
  const cases = []
  for (const dataset of GENERAL_DATASETS) {
    const directory = join(DATA_DIR, dataset)
    mkdirSync(directory, { recursive: true })
    const corpusPath = join(directory, "corpus.jsonl")
    const queriesPath = join(directory, "queries.jsonl")
    const qrelsPath = join(directory, "test.jsonl")
    const base = `https://huggingface.co/datasets/mteb/${dataset}/resolve/main`
    await Promise.all([
      download(`${base}/corpus.jsonl`, corpusPath),
      download(`${base}/queries.jsonl`, queriesPath),
      download(`${base}/qrels/test.jsonl`, qrelsPath),
    ])
    const documents = parseJsonLines(corpusPath).map((document, position) => ({
      id: document._id,
      text: [document.title, document.text].filter(Boolean).join("\n"),
      rawText: [document.title, document.text].filter(Boolean).join("\n"),
      sessionId: dataset,
      position,
    }))
    const qrels = new Map()
    for (const qrel of parseJsonLines(qrelsPath)) {
      const queryId = qrel["query-id"]
      if (!qrels.has(queryId)) qrels.set(queryId, new Map())
      qrels.get(queryId).set(qrel["corpus-id"], Number(qrel.score))
    }
    for (const query of parseJsonLines(queriesPath)) {
      const relevance = qrels.get(query._id)
      if (!relevance?.size) continue
      cases.push({
        id: `${dataset}-${query._id}`,
        suite: "general",
        category: dataset,
        query: query.text,
        answer: "",
        documents,
        evidenceIds: [...relevance.keys()],
        evidenceScores: Object.fromEntries(relevance),
      })
    }
  }
  return cases
}

function normalize(vector) {
  let norm = 0
  for (const value of vector) norm += value * value
  norm = Math.sqrt(norm)
  return norm === 0 ? vector : vector.map((value) => value / norm)
}

async function retry(operation, label, attempts = 6) {
  let lastError
  for (let attempt = 0; attempt < attempts; attempt++) {
    try {
      return await operation()
    } catch (error) {
      lastError = error
      if (attempt === attempts - 1) break
      const delay = Math.min(30_000, 1000 * 2 ** attempt)
      console.warn(`${label} failed; retrying in ${delay}ms: ${error.message}`)
      await new Promise((resolve) => setTimeout(resolve, delay))
    }
  }
  throw lastError
}

async function embedGemini(texts, config, taskType) {
  const output = []
  const batchSize = 50
  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize)
    const embeddings = await retry(async () => {
      const response = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/${config.model}:batchEmbedContents?key=${process.env.GEMINI_API_KEY}`,
        {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            requests: batch.map((text) => ({
              model: `models/${config.model}`,
              content: { parts: [{ text }] },
              taskType,
              outputDimensionality: config.dimensions,
            })),
          }),
        }
      )
      if (!response.ok) throw new Error(`${response.status} ${(await response.text()).slice(0, 500)}`)
      return (await response.json()).embeddings
    }, `${config.id}/${taskType}/${i}`)
    output.push(...embeddings.map((embedding) => normalize(embedding.values)))
    console.log(`  ${config.id}/${taskType}: ${Math.min(i + batch.length, texts.length)}/${texts.length}`)
  }
  return output
}

async function embedOpenAI(texts, config) {
  const output = []
  const batchSize = 256
  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize)
    const embeddings = await retry(async () => {
      const response = await fetch("https://api.openai.com/v1/embeddings", {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
        },
        body: JSON.stringify({
          model: config.model,
          input: batch,
          dimensions: config.dimensions,
          encoding_format: "float",
        }),
      })
      if (!response.ok) throw new Error(`${response.status} ${(await response.text()).slice(0, 500)}`)
      return (await response.json()).data.sort((a, b) => a.index - b.index)
    }, `${config.id}/${i}`)
    output.push(...embeddings.map((item) => normalize(item.embedding)))
    console.log(`  ${config.id}: ${Math.min(i + batch.length, texts.length)}/${texts.length}`)
  }
  return output
}

async function embeddingsFor(texts, config, side) {
  const uniqueTexts = [...new Set(texts)]
  const taskType = side === "query" ? config.queryTask : config.documentTask
  const cacheId = `${config.provider}-${config.model}-${config.dimensions}-${taskType ?? "default"}`
  const cachePath = join(
    CACHE_DIR,
    `${cacheId}-${embeddingSetHash(config, taskType, uniqueTexts).slice(0, 16)}.f32`
  )
  let vectors
  if (existsSync(cachePath)) {
    const bytes = readFileSync(cachePath)
    const floats = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)
    if (floats.length !== uniqueTexts.length * config.dimensions) {
      throw new Error(`Invalid embedding cache size: ${cachePath}`)
    }
    vectors = Array.from({ length: uniqueTexts.length }, (_, index) =>
      floats.subarray(index * config.dimensions, (index + 1) * config.dimensions)
    )
    console.log(`  ${config.id}/${side}: loaded ${vectors.length} cached embeddings`)
  } else {
    vectors =
      config.provider === "gemini"
        ? await embedGemini(uniqueTexts, config, taskType)
        : await embedOpenAI(uniqueTexts, config)
    const floats = new Float32Array(vectors.length * config.dimensions)
    for (let i = 0; i < vectors.length; i++) floats.set(vectors[i], i * config.dimensions)
    writeFileSync(cachePath, Buffer.from(floats.buffer))
  }
  return new Map(uniqueTexts.map((text, index) => [text, vectors[index]]))
}

function dot(a, b) {
  let value = 0
  for (let i = 0; i < a.length; i++) value += a[i] * b[i]
  return value
}

function dcg(rankedIds, evidenceScores, k = 10) {
  let value = 0
  for (let i = 0; i < Math.min(k, rankedIds.length); i++) {
    const relevance = evidenceScores.get(rankedIds[i]) ?? 0
    value += relevance / Math.log2(i + 2)
  }
  return value
}

function neighborsFor(documents) {
  const sessions = new Map()
  for (const document of documents) {
    if (!sessions.has(document.sessionId)) sessions.set(document.sessionId, [])
    sessions.get(document.sessionId).push(document)
  }
  const neighbors = new Map(documents.map((document) => [document.id, new Set()]))
  for (const session of sessions.values()) {
    session.sort((a, b) => a.position - b.position)
    for (let i = 0; i < session.length; i++) {
      if (session[i - 1]) neighbors.get(session[i].id).add(session[i - 1].id)
      if (session[i + 1]) neighbors.get(session[i].id).add(session[i + 1].id)
    }
  }
  return neighbors
}

function recall(ids, evidence) {
  let hits = 0
  for (const id of ids) if (evidence.has(id)) hits++
  return hits / evidence.size
}

function evaluateCase(testCase, queryVector, documentVectors) {
  const evidence = new Set(testCase.evidenceIds)
  const evidenceScores = new Map(
    testCase.evidenceIds.map((id) => [id, testCase.evidenceScores?.[id] ?? 1])
  )
  const ranked = testCase.documents
    .map((document) => ({
      id: document.id,
      text: document.rawText,
      score: dot(queryVector, documentVectors.get(document.text)),
    }))
    .sort((a, b) => b.score - a.score)
  const ids = ranked.map((item) => item.id)
  const evidenceRanks = testCase.evidenceIds
    .map((id) => ids.indexOf(id) + 1)
    .filter((rank) => rank > 0)
    .sort((a, b) => a - b)
  const idealRelevance = [...evidenceScores.values()].sort((a, b) => b - a)
  const ideal = idealRelevance
    .slice(0, 10)
    .reduce((sum, relevance, index) => sum + relevance / Math.log2(index + 2), 0)
  const neighbors = neighborsFor(testCase.documents)
  const expanded = new Set(ids.slice(0, 5))
  for (const id of [...expanded]) for (const neighbor of neighbors.get(id) ?? []) expanded.add(neighbor)
  const edges = testCase.relationEdges ?? []
  const staleRelationIds = new Set(edges.map((edge) => edge.fromId))
  const latestRelationIds = new Set(edges.map((edge) => edge.toId))
  const eligibleTopFive = ids.filter((id) => !staleRelationIds.has(id)).slice(0, 5)
  const productionSeeds = new Set(eligibleTopFive.filter((id) => latestRelationIds.has(id)))
  let changed = true
  while (changed) {
    changed = false
    for (const edge of edges) {
      if (productionSeeds.has(edge.toId) && !productionSeeds.has(edge.fromId)) {
        productionSeeds.add(edge.fromId)
        changed = true
      }
    }
  }

  return {
    id: testCase.id,
    suite: testCase.suite,
    category: testCase.category,
    query: testCase.query,
    answer: testCase.answer,
    evidenceCount: evidence.size,
    firstEvidenceRank: evidenceRanks[0] ?? null,
    recallAt1: recall(ids.slice(0, 1), evidence),
    recallAt5: recall(ids.slice(0, 5), evidence),
    recallAt10: recall(ids.slice(0, 10), evidence),
    recallAt20: recall(ids.slice(0, 20), evidence),
    allEvidenceAt10: evidenceRanks.every((rank) => rank <= 10) ? 1 : 0,
    mrr: evidenceRanks.length ? 1 / evidenceRanks[0] : 0,
    ndcgAt10: ideal ? dcg(ids, evidenceScores, 10) / ideal : 0,
    contextExpansionRecallAt5: recall(expanded, evidence),
    latestUpdateSeedAt5: edges.length > 0 ? (eligibleTopFive.some((id) => latestRelationIds.has(id)) ? 1 : 0) : null,
    updatesHistoryRecallAt5: edges.length > 0 ? recall(productionSeeds, evidence) : null,
    topResults: ranked.slice(0, 10),
    missedEvidence: testCase.documents
      .filter((document) => evidence.has(document.id) && !ids.slice(0, 10).includes(document.id))
      .map((document) => ({ id: document.id, text: document.rawText, rank: ids.indexOf(document.id) + 1 })),
  }
}

function mean(rows, key) {
  const values = rows.map((row) => row[key]).filter((value) => typeof value === "number")
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null
}

function confidenceInterval(rows, key) {
  const values = rows.map((row) => row[key]).filter((value) => typeof value === "number")
  if (values.length < 2) return null
  const average = values.reduce((sum, value) => sum + value, 0) / values.length
  const variance = values.reduce((sum, value) => sum + (value - average) ** 2, 0) / (values.length - 1)
  return 1.96 * Math.sqrt(variance / values.length)
}

function aggregate(rows) {
  return {
    questions: rows.length,
    recallAt1: mean(rows, "recallAt1"),
    recallAt5: mean(rows, "recallAt5"),
    recallAt10: mean(rows, "recallAt10"),
    recallAt20: mean(rows, "recallAt20"),
    allEvidenceAt10: mean(rows, "allEvidenceAt10"),
    mrr: mean(rows, "mrr"),
    ndcgAt10: mean(rows, "ndcgAt10"),
    recallAt10Ci95: confidenceInterval(rows, "recallAt10"),
    contextExpansionRecallAt5: mean(rows, "contextExpansionRecallAt5"),
    latestUpdateSeedAt5: mean(rows, "latestUpdateSeedAt5"),
    updatesHistoryRecallAt5: mean(rows, "updatesHistoryRecallAt5"),
  }
}

function summarize(rows) {
  const groups = new Map()
  groups.set("overall", rows)
  for (const row of rows) {
    const key = `${row.suite}/${row.category}`
    if (!groups.has(key)) groups.set(key, [])
    groups.get(key).push(row)
  }
  return Object.fromEntries([...groups].map(([key, value]) => [key, aggregate(value)]))
}

function percent(value) {
  return value === null ? "n/a" : `${(value * 100).toFixed(1)}%`
}

function markdownReport(run) {
  const lines = [
    "# Embedding model benchmark",
    "",
    `Generated: ${run.generatedAt}`,
    "",
    "## Overall retrieval",
    "",
    "| Configuration | R@5 | R@10 | MRR | nDCG@10 | All evidence@10 | Context expansion R@5 | Latest update seed@5 | Update history R@5 |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for (const result of run.results) {
    const metric = result.summary.overall
    lines.push(
      `| ${result.config.id} | ${percent(metric.recallAt5)} | ${percent(metric.recallAt10)} | ${metric.mrr.toFixed(3)} | ${metric.ndcgAt10.toFixed(3)} | ${percent(metric.allEvidenceAt10)} | ${percent(metric.contextExpansionRecallAt5)} | ${percent(metric.latestUpdateSeedAt5)} | ${percent(metric.updatesHistoryRecallAt5)} |`
    )
  }
  lines.push("", "## Category breakdown", "")
  for (const result of run.results) {
    lines.push(`### ${result.config.id}`, "", "| Category | N | R@5 | R@10 | MRR | nDCG@10 |", "|---|---:|---:|---:|---:|---:|")
    for (const [category, metric] of Object.entries(result.summary)) {
      if (category === "overall") continue
      lines.push(
        `| ${category} | ${metric.questions} | ${percent(metric.recallAt5)} | ${percent(metric.recallAt10)} | ${metric.mrr.toFixed(3)} | ${metric.ndcgAt10.toFixed(3)} |`
      )
    }
    lines.push("")
  }
  lines.push(
    "## Methodology notes",
    "",
    "- LoCoMo excludes adversarial questions and requires every positive evidence message ID to resolve; retrieval is restricted to the corresponding conversation.",
    `- ConvoMem draws a deterministic sample of ${run.settings.convomemSample} candidates per category, then keeps only cases with complete exact evidence-message matches.`,
    "- Documents include speaker and available session date because Supermemory stores contextualized content rather than bare sentences.",
    "- Context expansion adds one previous/next message around each top-five result; it is not labeled as Supermemory graph traversal.",
    "- Update-history expansion is reported only for fully matched, chronologically ordered ConvoMem changing-evidence cases. Only the latest node can seed traversal; history is then recovered by following stored old→new edges backward from the child.",
    "- No reranker, BM25, query rewriting, memory extraction, or answer model is used. These numbers isolate embedding retrieval quality.",
    "- Gemini task-type negative controls are ablations, not recommended production configurations.",
    ""
  )
  return lines.join("\n")
}

async function main() {
  const settings = parseArgs()
  mkdirSync(CACHE_DIR, { recursive: true })
  mkdirSync(RESULTS_DIR, { recursive: true })

  let cases = []
  if (["all", "locomo"].includes(settings.suite)) {
    const path = join(DATA_DIR, "locomo10.json")
    await download(LOCOMO_URL, path)
    cases.push(...loadLocomo(JSON.parse(readFileSync(path, "utf8"))))
  }
  if (["all", "convomem"].includes(settings.suite)) {
    const files = {}
    for (const [category, evidenceFolder] of Object.entries(CONVOMEM_PATHS)) {
      const path = join(DATA_DIR, `convomem-${category}.json`)
      await download(`${CONVOMEM_BASE}/${category}/${evidenceFolder}/batched_000.json`, path)
      files[category] = JSON.parse(readFileSync(path, "utf8"))
    }
    cases.push(...loadConvoMem(files, settings.convomemSample))
  }
  if (settings.suite === "general") {
    cases.push(...(await loadGeneralRetrieval()))
  }

  const requested = settings.configs === "all" ? null : new Set(settings.configs.split(","))
  const configs = CONFIGS.filter((config) => !requested || requested.has(config.id))
  if (configs.length === 0) throw new Error("No matching configurations")
  if (configs.some((config) => config.provider === "gemini") && !process.env.GEMINI_API_KEY) {
    throw new Error("GEMINI_API_KEY is required")
  }
  if (configs.some((config) => config.provider === "openai") && !process.env.OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY is required")
  }

  const documentTexts = [...new Set(cases.flatMap((testCase) => testCase.documents.map((document) => document.text)))]
  const queryTexts = [...new Set(cases.map((testCase) => testCase.query))]
  console.log(`Evaluating ${cases.length} cases, ${documentTexts.length} documents, ${queryTexts.length} queries`)

  const results = []
  for (const config of configs) {
    console.log(`\n${config.id}`)
    const [documentVectors, queryVectors] = await Promise.all([
      embeddingsFor(documentTexts, config, "document"),
      embeddingsFor(queryTexts, config, "query"),
    ])
    const rows = cases.map((testCase) =>
      evaluateCase(testCase, queryVectors.get(testCase.query), documentVectors)
    )
    results.push({
      config,
      summary: summarize(rows),
      failures: [...rows]
        .sort((a, b) => a.recallAt10 - b.recallAt10 || a.mrr - b.mrr)
        .slice(0, 25),
      rows,
    })
  }

  const run = {
    generatedAt: new Date().toISOString(),
    settings,
    dataset: {
      cases: cases.length,
      documents: documentTexts.length,
      queries: queryTexts.length,
      sourceFiles: [
        ...(["all", "locomo"].includes(settings.suite) ? ["locomo10.json"] : []),
        ...(["all", "convomem"].includes(settings.suite)
          ? Object.keys(CONVOMEM_PATHS).map((category) => basename(`convomem-${category}.json`))
          : []),
        ...(settings.suite === "general" ? GENERAL_DATASETS : []),
      ],
    },
    results,
  }
  const jsonPath = join(RESULTS_DIR, `${settings.output}.json`)
  const markdownPath = join(RESULTS_DIR, `${settings.output}.md`)
  writeFileSync(jsonPath, JSON.stringify(run, null, 2))
  writeFileSync(markdownPath, markdownReport(run))
  console.log(`\nWrote ${jsonPath}\nWrote ${markdownPath}`)
}

main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})

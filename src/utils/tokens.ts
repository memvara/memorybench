import { Tiktoken } from "js-tiktoken"
import cl100k_base from "js-tiktoken/ranks/cl100k_base"
import o200k_base from "js-tiktoken/ranks/o200k_base"
import { countTokens as countAnthropicTokens } from "@anthropic-ai/tokenizer"
import type { ModelConfig } from "./models"

export function countTokens(text: string, modelConfig: ModelConfig): number {
  const provider = modelConfig.provider

  if (provider === "openai") {
    return countOpenAITokens(text, modelConfig.id)
  } else if (provider === "anthropic") {
    return countAnthropicTokens(text)
  } else if (provider === "google") {
    // Approximation: Google doesn't provide a client-side tokenizer.
    // char/4 tends to undercount for JSON-heavy content (lots of short tokens
    // like {, ", :) but is reasonable for natural language.
    return Math.ceil(text.length / 4)
  }

  return Math.ceil(text.length / 4)
}

// Cached encoder instances (lazy singletons) to avoid re-instantiation per call
let _o200k: Tiktoken | null = null
let _cl100k: Tiktoken | null = null

function getEncoder(modelId: string): Tiktoken {
  // o200k_base: GPT-4o+, GPT-4.1, GPT-5, and reasoning models (o1, o3, o4)
  if (
    modelId.includes("gpt-4o") ||
    modelId.includes("gpt-4.1") ||
    modelId.includes("gpt-5") ||
    modelId.startsWith("o1") ||
    modelId.startsWith("o3") ||
    modelId.startsWith("o4")
  ) {
    return (_o200k ??= new Tiktoken(o200k_base))
  }
  return (_cl100k ??= new Tiktoken(cl100k_base))
}

/** Tokens under o200k_base -- the encoding `countTokens` uses for gpt-4o and gpt-5, and so
 *  the one the harness reports contextTokens with for those models. Reuses this module's
 *  cached encoder, so a caller sizing a prompt block measures it the way the report will.
 *
 *  The `[], []` disables js-tiktoken's special-token check, which otherwise throws on any
 *  text containing a literal such as `<|endoftext|>`. The LongMemEval haystack contains
 *  one, so the default would turn a sized prompt into a failed question. Counting them as
 *  ordinary text matches python `tiktoken.encode(text, disallowed_special=())`, which is
 *  how the offline measurements this budget is calibrated against were taken. */
export function countO200k(text: string): number {
  return getEncoder("gpt-4o").encode(text, [], []).length
}

/** Tokens under the encoding that matches the model id, which is what the report calls
 *  contextTokens.
 *
 *  The `[], []` is the same disabled special-token check as `countO200k`, and for the same
 *  text: without it a prompt containing `<|endoftext|>` throws here and falls back to
 *  chars/4, so the block a budget sized in real tokens would be recorded in an estimate
 *  that has nothing to do with it. The `catch` stays for everything else, where an estimate
 *  beats failing the question. */
function countOpenAITokens(text: string, modelId: string): number {
  try {
    const encoding = getEncoder(modelId)
    const tokens = encoding.encode(text, [], [])
    return tokens.length
  } catch (error) {
    return Math.ceil(text.length / 4)
  }
}

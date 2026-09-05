import { afterEach, describe, expect, test } from "bun:test"
import { getModelConfig, getModelId, openaiModelPrefix } from "./models"

const saved = process.env.OPENAI_MODEL_PREFIX

afterEach(() => {
  if (saved === undefined) delete process.env.OPENAI_MODEL_PREFIX
  else process.env.OPENAI_MODEL_PREFIX = saved
})

describe("OPENAI_MODEL_PREFIX", () => {
  test("unset or empty leaves every id exactly as the registry names it", () => {
    delete process.env.OPENAI_MODEL_PREFIX
    expect(openaiModelPrefix()).toBe("")
    expect(getModelId("gpt-4o")).toBe("gpt-4o")
    expect(getModelId("gpt-5.4")).toBe("gpt-5.4")
    process.env.OPENAI_MODEL_PREFIX = ""
    expect(getModelId("gpt-5.4")).toBe("gpt-5.4")
  })

  test("is prepended to OpenAI ids, listed and inferred alike, and read at call time", () => {
    process.env.OPENAI_MODEL_PREFIX = "openai/"
    expect(getModelId("gpt-4o")).toBe("openai/gpt-4o")
    const inferred = getModelConfig("gpt-5.4")
    expect(inferred.id).toBe("openai/gpt-5.4")
    expect(inferred.provider).toBe("openai")
    expect(inferred.maxTokensParam).toBe("max_completion_tokens")
  })

  test("never touches a model from another provider", () => {
    process.env.OPENAI_MODEL_PREFIX = "openai/"
    expect(getModelConfig("claude-sonnet-4-5").provider).toBe("anthropic")
    expect(getModelId("claude-sonnet-4-5")).not.toContain("openai/")
  })

  test("does not double a prefix the id already carries", () => {
    process.env.OPENAI_MODEL_PREFIX = "openai/"
    expect(getModelId("openai/gpt-5.4")).toBe("openai/gpt-5.4")
  })

  test("does not mutate the shared registry entry", () => {
    process.env.OPENAI_MODEL_PREFIX = "openai/"
    getModelConfig("gpt-4o")
    delete process.env.OPENAI_MODEL_PREFIX
    expect(getModelId("gpt-4o")).toBe("gpt-4o")
  })
})

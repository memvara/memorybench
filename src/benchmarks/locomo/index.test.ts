import { describe, expect, test } from "bun:test"
import { CATEGORY_TO_TYPE } from "./index"

describe("LoCoMo category numbers", () => {
  // snap-research/locomo's task_eval/evaluation.py scores category 1 by splitting the
  // answer into sub-answers (multi-hop), 5 by checking the abstention (adversarial), and
  // 2, 3, 4 as single answers. A sample of the questions under each number says the rest:
  // "When did Jolene ... ?" is 2, "Would Caroline want to ... ?" is 3, "What type of
  // seminars is John conducting?" is 4. The map once read 1 as single-hop and 3 as
  // temporal, and every per-category number in a report was under the wrong name.
  test("follow the dataset's own numbering", () => {
    expect(CATEGORY_TO_TYPE).toEqual({
      1: "multi-hop",
      2: "temporal",
      3: "world-knowledge",
      4: "single-hop",
      5: "adversarial",
    })
  })
})

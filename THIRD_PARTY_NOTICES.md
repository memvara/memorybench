# Third-party notices

## LongMemEval-V2

The LongMemEval-V2 integration is based on the public
[`xiaowu0162/longmemeval-v2`](https://huggingface.co/datasets/xiaowu0162/longmemeval-v2)
dataset and the LongMemEval-V2 benchmark repository.

- Reference revision: `f152293e235517d504809563c833d7190b8c713b`
- Supermemory adapter oracle: `feat/supermemory` commit
  `2fa6616dce77e0385d7e1c44510dfde8aa3c46e3`
- Structured Accessibility Converter oracle:
  `supermemory_adapter/approaches/Approach_1.py`
- Oracle SHA-256:
  `22cff05fafa9f882040afa8296439da0f911f800c107424de105ab3af5e69236`
- License: Apache License 2.0

MemoryBench downloads the dataset's own `LICENSE` file as part of the pinned,
checksum-verified snapshot. The TypeScript converter is a new implementation
whose behavior is tested against the cited reference.

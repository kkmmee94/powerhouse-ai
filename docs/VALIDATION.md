# Release validation

Validation date: 2026-07-27

This report describes the exact source and benchmark artifacts published with Powerhouse AI
0.1.0. The benchmark runner fingerprints the implementation, locked dependencies, benchmark code,
and benchmark inputs. All three end-to-end runs below have the same source fingerprint:

`3ddc52fe2a45a54a5f26e4c689d7b0f949052fbc02c0574f47d52bea2af4ec94`

## Outcome

On the release machine, Fast and Adaptive both improved the same local model on this
project-focused suite. Fast provided most of the measured quality gain at approximately baseline
latency. Adaptive achieved the highest score but used more model calls and averaged 3.29 times the
baseline latency.

All 44 requests completed in every mode. There were no configured-budget violations, and every
explicitly labelled Python response parsed successfully. The stored raw and scored JSONL files
contain every result; low-scoring answers were not selectively replaced.

## Validated runtime

| Item | Measured value |
|---|---:|
| Operating system | Windows 11 |
| GPU | NVIDIA GeForce RTX 5060 Ti |
| Available VRAM | 16,311 MiB |
| Model | Qwen3.5-9B Q6_K GGUF |
| Model file size | 7,958,818,848 bytes (7.412 GiB) |
| Model SHA-256 | `073A9275E65D9C8CD2819CF5F77B99FBAA6E87BA591DA6BBAA86EC073A64BFEF` |
| Runtime | CUDA llama.cpp b10107 |
| Configured context | 8,192 tokens |
| Warm generation speed | 51.20-51.58 tokens/second |
| Profile peak VRAM | 9,313 MiB |
| VRAM acceptance limit | 15,872 MiB |

The setup scripts pin and verify both the llama.cpp archives and the model artifact. Model weights
and the runtime are downloaded from their upstream projects and are not included in this
repository.

## End-to-end comparison

Each mode used the same Qwen3.5-9B Q6_K model and the same 44 requests. `Plain model` is one direct
model call without Powerhouse memory, retrieval, deterministic tools, specialist stages, synthesis,
or repair. The local judge used temperature zero and the same fixed
correctness/completeness/clarity rubric for every response.

| Metric | Plain model | Powerhouse Fast | Powerhouse Adaptive |
|---|---:|---:|---:|
| Quality score | 76.14% | **93.75%** | **96.59%** |
| Relative quality gain | - | **+23.13%** | **+26.86%** |
| Median latency | 9.637 s | **8.512 s** | 17.380 s |
| Mean latency | 14.118 s | **14.046 s** | 46.502 s |
| Mean model calls | 1.00 | 1.11 | 2.09 |
| In-suite route recall | Not applicable | 100% | 100% |
| In-suite exact routes | Not applicable | 44/44 | 44/44 |
| Pipeline completion | 44/44 | 44/44 | 44/44 |
| Labelled Python syntax | 100% | 100% | 100% |
| Unsupported claims counted | 3 | **0** | **0** |
| Peak VRAM | 9,152 MiB | 9,293 MiB | 9,315 MiB |
| VRAM-budget violations | 0 | 0 | 0 |

Fast's median response was 11.7% faster than the plain-model median, while its mean latency was
effectively equal. Adaptive's slowest request took 319.465 seconds, so its 17.380-second median is
more representative of ordinary requests than its 46.502-second mean. This long-tail cost is
important when choosing a mode.

The baseline does not perform routing, so routing metrics are intentionally not attributed to it.

## Memory, retrieval, and exact tools

| Check | Result |
|---|---:|
| Relevant saved decisions found in top 8 | 20/20 |
| Cross-project memory isolation | Pass |
| Calculator regression | `21`, 0 model calls |
| Unit conversion regression | `2,500 g`, 0 model calls |

The memory benchmark uses an isolated temporary database and leaves the user's application database
unchanged.

## Software verification

- 61 unit and integration tests passed with 83.43% branch-aware coverage.
- Ruff reported no lint errors.
- The locked dependency graph resolved without known vulnerabilities in the release audit.
- Source and wheel builds completed, and the wheel passed a clean-install CLI smoke test.
- PowerShell release scripts parsed without syntax errors.
- FastAPI `/health`, `/ready`, and `/status` passed with the model online.
- Both bundled services remained bound to `127.0.0.1`.
- Python execution remained disabled by default.

## Interpretation and limits

- These measurements compare Powerhouse with **the same local model used alone** on one
  project-focused suite. They do not establish superiority over cloud or frontier models.
- The suite intentionally includes memory and private-document questions. The baseline cannot know
  that local information; retrieval is part of the system capability being measured.
- The 44 routing labels are an in-suite regression check, not an independently authored held-out
  routing benchmark.
- The judge is the same local model, not an independent human panel. Temperature zero improves
  repeatability but does not eliminate evaluation bias.
- A post-run provenance correction excluded ignored Python bytecode caches from the deterministic
  source fingerprint. This changed only the `source_sha256` metadata field in stored records;
  questions, answers, routes, timings, warnings, model-call counts, VRAM samples, and scores are the
  original run data.
- Python syntax validation proves that fenced code parses. It does not establish semantic
  correctness or safely execute untrusted code.
- Held-out routing evaluation and execution-based generated-code evaluation remain open acceptance
  criteria.
- End-to-end results are hardware- and model-dependent. Reproduction requires the documented model
  artifact and a compatible local model server.

## Release artifacts

- Plain raw/scored run: `benchmarks/results/v0.1.0/base.jsonl` and
  `benchmarks/results/v0.1.0/base.scored.jsonl`
- Fast raw/scored run: `benchmarks/results/v0.1.0/fast.jsonl` and
  `benchmarks/results/v0.1.0/fast.scored.jsonl`
- Adaptive raw/scored run: `benchmarks/results/v0.1.0/adaptive.jsonl` and
  `benchmarks/results/v0.1.0/adaptive.scored.jsonl`
- Memory benchmark: `benchmarks/results/v0.1.0/memory.json`
- Q6 profile: `benchmarks/results/v0.1.0/q6-profile.json`
- Artifact digests: `benchmarks/results/v0.1.0/SHA256SUMS`

See `docs/REPRODUCIBILITY.md` for the commands and controls used to reproduce the measurements.

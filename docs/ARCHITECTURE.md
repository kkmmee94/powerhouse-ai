# Powerhouse AI architecture

## Runtime objective

Keep one capable model resident on the RTX 5060 Ti, then selectively add only the context,
specialist instructions, evidence, memory, and deterministic tools required by the current goal.
Inactive knowledge remains in SQLite or on disk and consumes no VRAM.

Version 0.1 exposes text through the CLI and HTTP API. The selected model has vision capability, but
image input and image-projector lifecycle are outside the implemented application interface.

## Request flow

```text
request
  -> retrieve relevant project memory
  -> classify one or more domains
  -> build a dependency-ordered plan
  -> retrieve evidence for each specialist stage
  -> fit history, evidence, and tool metadata within the configured context budget
  -> execute safe deterministic tools where possible
  -> run selected specialist stages sequentially
  -> synthesize cross-domain work
  -> verify complex/power-mode answers against tool results and retrieved evidence
  -> save the messages, goal, steps, evidence links, and tool results
```

## Components

| Component | Implementation | Resource policy |
|---|---|---|
| Core model | Qwen3.5-9B Q6_K through CUDA llama.cpp | One model in VRAM |
| Router | deterministic multilabel patterns with optional model arbitration | No extra model |
| Planner | dependency-ordered specialist stages | CPU only |
| Specialists | domain prompts for eight domains | Reuse the same weights |
| Memory | SQLite tables plus FTS5 search | Disk/RAM, zero VRAM |
| Retrieval | project-scoped and domain-scoped FTS5 chunks | Disk/RAM, zero VRAM |
| Context guard | request-scoped fitting with reserved output capacity | Prevents context overflow |
| Calculator | restricted Python AST evaluator | CPU only |
| Unit conversion | deterministic dimension-checked conversion | CPU only |
| Code checking | Python AST validation plus up to two final repair attempts | No code execution |
| VRAM tracker | periodic `nvidia-smi` sampling | Reports budget violations |
| Interfaces | Typer CLI and FastAPI | Localhost only |

## Execution modes

`fast` uses deterministic routing and one specialist generation. `balanced` decomposes
multi-domain work. `adaptive` keeps simple tasks on the fast path and escalates complex tasks.
`power` additionally enables model thinking and a final verification pass.

## Safety boundary

Generated Python is parsed but not executed by default. The optional Python subprocess is suitable
only for trusted code; it is not a hostile-code sandbox. Calculator expressions use a strict
allowlist. Unambiguous calculator and metric-conversion requests return deterministic tool output
without asking the model to restate the result. Both services bind to `127.0.0.1`.

The built-in API has no authentication and cannot be configured to bind beyond loopback through the
Powerhouse CLI. Project data is stored unencrypted in SQLite. Retrieved documents and memory are
marked as untrusted data in model instructions. Only evidence labels cited by the final answer are
returned as user-facing sources; the run record retains all retrieved evidence IDs for audit.

Installer scripts pin upstream revisions and verify SHA-256 digests before executing a runtime or
loading a model file.

## Deliberate current boundary

Version 0.1 uses prompt specialists rather than trained LoRA adapters. This proves routing,
composition, retrieval, memory, tools, GPU fit, and evaluation first. Adapter training should begin
only after benchmark failures identify where a specialist changes measured outcomes.

# Acceptance criteria

Powerhouse AI is successful only when the measured system meets all hard requirements and shows a
useful improvement over the same base model without orchestration.

## Hard requirements

| Metric | Requirement |
|---|---:|
| Peak VRAM during supported workloads | <= 15,872 MiB |
| Local request processing | 100% |
| Project separation tests | 100% pass |
| Calculator safety tests | 100% pass |
| Unsupported Python execution | disabled by default |
| Domain routing recall on an independently authored evaluation set | >= 90% |
| Relevant memory found in top 8 | >= 85% |
| End-to-end pipeline completion | >= 95% |

## Improvement requirements

Measured on the same questions and model sampling configuration:

| Metric | Full system versus base model |
|---|---:|
| Overall rubric score | >= 20% relative improvement |
| Executable code pass rate | >= 20 percentage-point improvement |
| Unsupported factual claims | >= 30% relative reduction |

Example: a base score of 50/100 requires at least 60/100 for a 20% relative improvement.

## Performance policy

The `fast` path should use one model stage. The `adaptive` path may use more stages only when the
router identifies a complex or multi-domain request. The `power` path deliberately spends more time
on specialist stages and verification.

Profile selection order:

1. Reject any profile that exceeds the VRAM hard limit.
2. Reject profiles with unstable tool calling or structured JSON.
3. Among remaining profiles, calculate quality gained per second of latency.
4. Use the highest-quality responsive profile as the default.

No public model benchmark substitutes for this machine-level acceptance suite.

## Version 0.1 release status

Version 0.1 is an experimental preview, not a declaration that every success criterion is complete.
The independent held-out routing set and execution-based code benchmark remain open. Public release
artifacts must identify the exact source version, model hash, runtime version, questions, and raw
scored records used for any numeric claim.

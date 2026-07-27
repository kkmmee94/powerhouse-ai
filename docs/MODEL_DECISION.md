# Core model decision

Decision date: 2026-07-27

## Selected starting core

`Qwen/Qwen3.5-9B`, served as a GGUF quantization through the CUDA build of `llama.cpp`.

Default deployment quantization: `Q6_K`.

## Why

The official Qwen model card reports:

- 9B language-model parameters
- native multimodal input
- 262,144-token native context
- strong instruction, reasoning, coding, agent, long-context, and vision results
- Apache-2.0 license

Official card:

<https://huggingface.co/Qwen/Qwen3.5-9B>

The selected community GGUF repository:

<https://huggingface.co/bartowski/Qwen_Qwen3.5-9B-GGUF>

Release downloads are pinned to repository revision
`182be2fd6c7bc44887d88a91cb03ff009cc9f549` and verified before use:

| Profile | File SHA-256 |
|---|---|
| Q4_K_M | `D784CE9EDA1A5A7B51E8F705A9E6310844BF4F173654D115823C775FDEA56D43` |
| Q6_K | `073A9275E65D9C8CD2819CF5F77B99FBAA6E87BA591DA6BBAA86EC073A64BFEF` |
| Q8_0 | `B58FE056B5435070240DE259F3F981AA38FEE96825BBD78C088D5FD90E46F2B5` |

The default Q6_K file is approximately 7.96 GB. Q8_0 is approximately 9.8 GB. Both leave materially
more room than a 14B Q4 core for context, tools, and later vision or adapter work.

## Rejected default

Qwen3.6-27B is newer and has more raw model capacity, but its Q4_K_M GGUF is approximately 19.1 GB.
That exceeds the entire 16 GB GPU before KV cache, runtime buffers, or vision projection are added.
CPU offload would make it a useful quality-reference experiment, not the fastest primary runtime.

Official model:

<https://huggingface.co/Qwen/Qwen3.6-27B>

GGUF files:

<https://huggingface.co/ggml-org/Qwen3.6-27B-GGUF>

## Benchmark policy

Vendor results choose candidates; measurements on this exact RTX 5060 Ti choose the production
profile. Q4_K_M, Q6_K, and Q8_0 will be compared on:

- task score
- tool-call correctness
- JSON routing reliability
- tokens per second
- time to first token
- peak VRAM at 8K and 16K context

The production profile is the strongest profile that stays below 15,872 MiB and remains responsive.

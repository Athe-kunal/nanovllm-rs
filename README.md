# nanovllm_rs

A Rust reimplementation of a vLLM-style inference engine, serving models over HTTP with continuous batching. Model execution is pure Rust/candle end to end, with no Python or PyO3 bridge involved in serving. Attention runs through `candle-flash-attn` (native CUDA flash-attention, including a paged/windowed variant for the KV cache) by default, falling back automatically to plain `candle_nn` matmul/softmax attention if that crate isn't built in — see [Attention backend](#attention-backend).

## Prerequisites

- NVIDIA GPU + driver
- CUDA toolkit 12.8 (`nvcc`) and a matching host compiler (`g++`)
- Rust (`rustup`/`cargo`)
- [`uv`](https://github.com/astral-sh/uv) — only used to fetch a model from Hugging Face (a throwaway venv with just `huggingface_hub`); the server itself has no Python dependency

## Setup

```
make download-model MODEL=Qwen/Qwen3-0.6B     # pulls the model from Hugging Face into models/
make serve                                    # builds and starts the server
```

`make serve` also downloads the model itself if it isn't already present, so a separate `download-model` step is optional — useful mainly to pre-fetch a model ahead of time.

Defaults to `Qwen/Qwen3-0.6B` on port 8000 with `--tensor-parallel-size 1`. Override with `MODEL`, `PORT`, `TP_SIZE` — any Qwen3 model works (e.g. `Qwen/Qwen3-4B-Instruct-2507`), since the engine reads model dimensions and chat template from the model's own files rather than hardcoding them:

```
make serve MODEL=Qwen/Qwen3-4B-Instruct-2507 TP_SIZE=2
```

## Using the server

```
make generate PROMPT="The capital of France is"
```

Or hit the endpoints directly:

- `POST /generate` — `{"prompt": str, "max_tokens": int, "temperature": float}`
- `POST /chat` — `{"messages": [{"role": str, "content": str}], "max_tokens": int, "temperature": float}` for multi-turn conversations (send the full message history each call)

The server batches concurrent requests together automatically, so multiple clients can hit it at once without waiting in line.

## Attention backend

By default (`make serve`, `make test`), the build enables the `flash-attn` Cargo feature, which
compiles in `candle-flash-attn` for real fused CUDA flash-attention. If it fails to build for
your GPU architecture or CUDA toolchain (or you just don't want the extra kernel compilation),
build with the `cuda` feature alone instead — attention then falls back to plain `candle_nn`
ops (matmul + softmax, causal masking, GQA), correct but slower:

```
make serve FEATURES=cuda
```

This is a compile-time choice, not a runtime one: whichever feature set the binary was built
with determines which attention path it uses.

## Smoke-testing

```
make test
```

Runs a small client (`test_client`) against an already-running server: sends a few prompts over `/generate` and checks the responses come back sane. Point it at a different host/port with `HOST`/`PORT`.

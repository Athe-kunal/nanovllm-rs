# nanovllm_rs

A Rust reimplementation of a vLLM-style inference engine, serving models over HTTP with continuous batching.

## Prerequisites

- NVIDIA GPU + driver
- CUDA toolkit 12.8 (`nvcc`) and a matching host compiler (`g++`)
- Rust (`rustup`/`cargo`)
- [`uv`](https://github.com/astral-sh/uv)

Run all `make` commands as the **same user** throughout (don't mix `root` and a regular user) — the Python/Rust toolchains are installed per-user, and switching users mid-setup causes permission and `PATH` errors.

## Setup

```
make install-python                          # installs the Python deps (torch, kernels, etc.)
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

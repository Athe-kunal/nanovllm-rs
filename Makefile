.PHONY: serve generate download-model

export PATH := $(HOME)/.cargo/bin:/usr/local/cuda/bin:$(PATH)

# WSL exposes two libcuda.so.1: a real Linux driver copy (pulled in by nvidia-cuda-toolkit,
# talks to a nonexistent /dev/nvidia0) and the WSL passthrough shim in /usr/lib/wsl/lib (talks
# to the Windows host driver via /dev/dxg). It must resolve first or CUDA calls report no device.
export LD_LIBRARY_PATH := /usr/lib/wsl/lib:$(LD_LIBRARY_PATH)

MODEL ?= Qwen/Qwen3-0.6B
MODEL_DIR := models/$(MODEL)
TP_SIZE ?= 1
PORT ?= 8000
HOST ?= localhost
PROMPT ?= The capital of France is
MAX_TOKENS ?= 64
TEMPERATURE ?= 1.0
GPU_MEMORY_UTILIZATION ?= 0.9
# Default (16384) sizes the KV-cache profiling forward pass's peak activation memory; on a
# 4GB card that peak alone can exceed the whole memory budget regardless of utilization.
MAX_NUM_BATCHED_TOKENS ?= 2048

# download-model only needs huggingface_hub, not a full Python toolchain — a tiny venv just
# for that, separate from the (now removed) torch/pyo3 bridge the Rust binary used to need.
DOWNLOAD_VENV := $(CURDIR)/.download-venv

download-model:
	@if [ ! -f "$(MODEL_DIR)/config.json" ]; then \
		[ -x "$(DOWNLOAD_VENV)/bin/python" ] || uv venv --managed-python "$(DOWNLOAD_VENV)" >/dev/null; \
		uv pip install --python "$(DOWNLOAD_VENV)/bin/python" huggingface_hub >/dev/null; \
		echo "Downloading $(MODEL) to $(MODEL_DIR)..."; \
		"$(DOWNLOAD_VENV)/bin/python" -c "from huggingface_hub import snapshot_download; snapshot_download('$(MODEL)', local_dir='$(MODEL_DIR)')"; \
	fi

serve: download-model
	cargo run --release --manifest-path nanovllm_rs/Cargo.toml --features cuda --bin serve -- \
		$(MODEL_DIR) --tensor-parallel-size $(TP_SIZE) --port $(PORT) \
		--gpu-memory-utilization $(GPU_MEMORY_UTILIZATION) \
		--max-num-batched-tokens $(MAX_NUM_BATCHED_TOKENS)

generate:
	curl -s -X POST http://$(HOST):$(PORT)/generate \
		-H "Content-Type: application/json" \
		-d '{"prompt": "$(PROMPT)", "max_tokens": $(MAX_TOKENS), "temperature": $(TEMPERATURE)}'

.PHONY: install-python serve generate download-model nccl-lib

export PATH := $(HOME)/.cargo/bin:/usr/local/cuda/bin:$(PATH)
# torch bundles its own NCCL (pip nvidia-nccl-cu12), which can be a different version than
# any system-wide libnccl.so. cudarc's dynamic loader looks for the unversioned "libnccl.so"
# first; torch's pip package only ships "libnccl.so.2" (no such symlink), so that lookup
# falls through to the system copy and cudarc/torch end up with two ABI-mismatched NCCLs in
# the same process. .nccl_lib/libnccl.so (created below) symlinks to torch's copy so both
# resolve to the exact same file.
export LD_LIBRARY_PATH := $(CURDIR)/.nccl_lib:$(LD_LIBRARY_PATH)
TORCH_NCCL_LIB := /usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib/libnccl.so.2

MODEL ?= Qwen/Qwen3-0.6B
MODEL_DIR := models/$(MODEL)
TP_SIZE ?= 1
PORT ?= 8000
HOST ?= localhost
PROMPT ?= The capital of France is
MAX_TOKENS ?= 64
TEMPERATURE ?= 1.0

install-python:
	uv pip install --system -e nanovllm_rs/python --extra-index-url https://download.pytorch.org/whl/cu128
	chmod -R a+rX /usr/local/lib/python3.10/dist-packages 2>/dev/null || \
		sudo chmod -R a+rX /usr/local/lib/python3.10/dist-packages

nccl-lib:
	mkdir -p .nccl_lib
	ln -sf $(TORCH_NCCL_LIB) .nccl_lib/libnccl.so

download-model:
	@if [ ! -f "$(MODEL_DIR)/config.json" ]; then \
		echo "Downloading $(MODEL) to $(MODEL_DIR)..."; \
		python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$(MODEL)', local_dir='$(MODEL_DIR)')"; \
	fi

serve: download-model nccl-lib
	cargo run --manifest-path nanovllm_rs/Cargo.toml --features cuda --bin serve -- \
		$(MODEL_DIR) --tensor-parallel-size $(TP_SIZE) --port $(PORT)

generate:
	curl -s -X POST http://$(HOST):$(PORT)/generate \
		-H "Content-Type: application/json" \
		-d '{"prompt": "$(PROMPT)", "max_tokens": $(MAX_TOKENS), "temperature": $(TEMPERATURE)}'

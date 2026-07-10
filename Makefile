.PHONY: install-python serve generate download-model nccl-lib

export PATH := $(HOME)/.cargo/bin:/usr/local/cuda/bin:$(PATH)
# torch bundles its own NCCL (pip nvidia-nccl-cu12), which can be a different version than
# any system-wide libnccl.so. cudarc's dynamic loader looks for the unversioned "libnccl.so"
# first; torch's pip package only ships "libnccl.so.2" (no such symlink), so that lookup
# falls through to the system copy and cudarc/torch end up with two ABI-mismatched NCCLs in
# the same process. .nccl_lib/libnccl.so (created below) symlinks to torch's copy so both
# resolve to the exact same file.
# flash-attn ships only a cp310 wheel, so the venv is pinned to Python 3.10 regardless of
# whatever system Python is on PATH. uv fetches that interpreter itself if it's missing.
VENV_DIR := $(CURDIR)/.venv
export VIRTUAL_ENV := $(VENV_DIR)
export PYO3_PYTHON := $(VENV_DIR)/bin/python
export PATH := $(VENV_DIR)/bin:$(PATH)
VENV_SITE_PACKAGES := $(VENV_DIR)/lib/python3.10/site-packages
TORCH_NCCL_LIB := $(VENV_SITE_PACKAGES)/nvidia/nccl/lib/libnccl.so.2
# The venv's python is a symlink into uv's standalone interpreter install: its
# libpython3.10.so.1.0 and stdlib live there, not in the venv, and pyo3's embedded
# interpreter needs PYTHONHOME to find them (the venv itself has no copy of the stdlib).
UV_PYTHON_HOME := $(patsubst %/,%,$(dir $(realpath $(VENV_DIR)/bin/python)))/..
export PYTHONHOME := $(UV_PYTHON_HOME)
# uv's editable install of nanovllm_kernels relies on a .pth-triggered import finder, which
# only runs when site-packages is scanned via sys.prefix (i.e. actual venv startup) — a
# PYTHONHOME override skips that, so the package source dir is added directly instead.
export PYTHONPATH := $(VENV_SITE_PACKAGES):$(CURDIR)/nanovllm_rs/python

# WSL exposes two libcuda.so.1: a real Linux driver copy (pulled in by nvidia-cuda-toolkit,
# talks to a nonexistent /dev/nvidia0) and the WSL passthrough shim in /usr/lib/wsl/lib (talks
# to the Windows host driver via /dev/dxg). It must resolve first or CUDA calls report no device.
export LD_LIBRARY_PATH := /usr/lib/wsl/lib:$(CURDIR)/.nccl_lib:$(UV_PYTHON_HOME)/lib:$(LD_LIBRARY_PATH)

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

install-python:
	uv venv --python 3.10 $(VENV_DIR)
	uv pip install --python $(VENV_DIR)/bin/python -e nanovllm_rs/python --extra-index-url https://download.pytorch.org/whl/cu128

nccl-lib:
	mkdir -p .nccl_lib
	ln -sf $(TORCH_NCCL_LIB) .nccl_lib/libnccl.so

download-model:
	@if [ ! -f "$(MODEL_DIR)/config.json" ]; then \
		echo "Downloading $(MODEL) to $(MODEL_DIR)..."; \
		python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$(MODEL)', local_dir='$(MODEL_DIR)')"; \
	fi

serve: download-model nccl-lib
	cargo run --release --manifest-path nanovllm_rs/Cargo.toml --features cuda --bin serve -- \
		$(MODEL_DIR) --tensor-parallel-size $(TP_SIZE) --port $(PORT) \
		--gpu-memory-utilization $(GPU_MEMORY_UTILIZATION) \
		--max-num-batched-tokens $(MAX_NUM_BATCHED_TOKENS)

generate:
	curl -s -X POST http://$(HOST):$(PORT)/generate \
		-H "Content-Type: application/json" \
		-d '{"prompt": "$(PROMPT)", "max_tokens": $(MAX_TOKENS), "temperature": $(TEMPERATURE)}'

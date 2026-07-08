.PHONY: install-python serve generate

MODEL ?= Qwen/Qwen3-0.6B
TP_SIZE ?= 1
PORT ?= 8000
HOST ?= localhost
PROMPT ?= The capital of France is
MAX_TOKENS ?= 64
TEMPERATURE ?= 1.0

install-python:
	uv pip install --system --reinstall torch --index-url https://download.pytorch.org/whl/cu128
	uv pip install --system -U packaging setuptools wheel psutil ninja
	uv pip install --system --no-build-isolation -e nanovllm_rs/python

serve:
	cargo run --manifest-path nanovllm_rs/Cargo.toml --features cuda --bin serve -- \
		$(MODEL) --tensor-parallel-size $(TP_SIZE) --port $(PORT)

generate:
	curl -s -X POST http://$(HOST):$(PORT)/generate \
		-H "Content-Type: application/json" \
		-d '{"prompt": "$(PROMPT)", "max_tokens": $(MAX_TOKENS), "temperature": $(TEMPERATURE)}'

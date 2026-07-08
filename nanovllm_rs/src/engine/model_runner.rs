use candle_core::{DType, Device};
use pyo3::prelude::*;

use crate::config::{Config, EngineConfig};
use crate::engine::sequence::Sequence;
use crate::models::qwen3::Qwen3ForCausalLM;
use crate::utils::pybridge;

pub struct ModelRunner {
    rank: usize,
    device: Device,
    model: Qwen3ForCausalLM,
    // One (k_cache, v_cache) torch.Tensor pair per layer, each shaped
    // (num_blocks, block_size, num_kv_heads, head_dim). GPU-resident; mutated in
    // place by the Python-side store_kvcache/flash_attn_varlen kernels only.
    kv_caches: Vec<(Py<PyAny>, Py<PyAny>)>,
}

impl ModelRunner {
    pub fn new(config: &Config, engine_config: &EngineConfig, rank: usize) -> Self {
        let device = Device::cuda_if_available(rank).expect("failed to create device");

        // Single-process only (LLMEngine::new already asserts tensor_parallel_size == 1),
        // so no NCCL comm to hand to the model yet.
        let mut model = Qwen3ForCausalLM::new(config, None, &device).expect("failed to build model");

        let kv_caches = Python::with_gil(|py| {
            pybridge::allocate_kv_cache(
                py,
                config.num_hidden_layers,
                engine_config.num_kvcache_blocks,
                engine_config.kvcache_block_size,
                config.num_key_value_heads,
                config.head_dim,
                DType::F32,
            )
        })
        .expect("failed to allocate KV cache");

        model.set_kv_caches(kv_caches.clone());

        // Weight loading from the checkpoint isn't wired up yet: `loader::load_model`
        // requires the model to implement `ModelWeights`, but `Qwen3ForCausalLM`'s
        // packed_modules_mapping mixes string ("q"/"k"/"v") and integer (0/1) shard
        // ids, which that trait's usize-only shard-id type can't represent. Until
        // that's resolved, `model` runs with its freshly-constructed (uninitialized)
        // weights.

        Self { rank, device, model, kv_caches }
    }

    /// Should run the model over `seqs` and return (sampled token ids, indices
    /// into `seqs` that need a sampled logit this step) — mirrors what
    /// `Scheduler::postprocess` expects as input.
    pub fn run(&mut self, _seqs: &mut [Sequence]) -> (Vec<u32>, Vec<usize>) {
        unimplemented!("ModelRunner::run: model forward pass is not implemented yet")
    }

    pub fn exit(&mut self) {}
}

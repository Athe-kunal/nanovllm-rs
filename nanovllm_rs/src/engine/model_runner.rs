use candle_core::{DType, Device, Result, Tensor};
use pyo3::prelude::*;
use std::sync::Arc;

use crate::config::{Config, EngineConfig};
use crate::engine::sequence::Sequence;
use crate::layers::nccl::Comm;
use crate::layers::sampler::Sampler;
use crate::models::qwen3::Qwen3ForCausalLM;
use crate::sampling_params::SamplingParams;
use crate::utils::context::{get_context, reset_context, set_context};
use crate::utils::loader;
use crate::utils::pybridge;

fn dtype_itemsize(dtype: &str) -> usize {
    match dtype {
        "float64" | "int64" => 8,
        "float32" | "int32" => 4,
        "float16" | "bfloat16" => 2,
        "uint8" | "int8" => 1,
        other => panic!("unsupported torch_dtype: {other}"),
    }
}

pub struct ModelRunner {
    config: Config,
    engine_config: EngineConfig,
    rank: usize,
    device: Device,
    model: Qwen3ForCausalLM,
    sampler: Sampler,
    num_kvcache_blocks: usize,
}

impl ModelRunner {
    /// KV cache is allocated separately via `probe_num_kvcache_blocks`/`finish_setup`,
    /// since under TP the block count must be reconciled across ranks first.
    pub fn new(config: &Config, engine_config: &EngineConfig, rank: usize, comm: Option<Arc<Comm>>) -> Self {
        Python::with_gil(|py| pybridge::set_cuda_device(py, rank)).expect("failed to set cuda device");
        // cuda_if_available() creates a CudaDevice on cudarc's NULL/legacy default stream
        // (raw pointer 0x0), which torch.cuda.ExternalStream is not well-behaved for — using
        // new_cuda_with_stream() gets a genuine non-default stream to hand to the DLPack bridge.
        let device = Device::new_cuda_with_stream(rank).expect("failed to create device");
        let mut model = Qwen3ForCausalLM::new(config, comm, &device).expect("failed to build model");
        loader::load_model(&mut model, &engine_config.model_path, &device).expect("failed to load model weights");
        model.tie_weights_if_configured();

        let mut runner = Self {
            config: config.clone(),
            engine_config: engine_config.clone(),
            rank,
            device,
            model,
            sampler: Sampler,
            num_kvcache_blocks: 0,
        };

        runner.warmup_model();
        runner
    }

    pub fn finish_setup(&mut self, num_blocks: usize) {
        self.allocate_kv_cache(num_blocks);
        if !self.engine_config.enforce_eager {
            self.capture_cudagraph();
        }
    }

    fn prepare_block_tables(&self, seqs: &[Sequence]) -> Result<Tensor> {
        let max_len = seqs.iter().map(|s| s.block_table.len()).max().unwrap();
        let mut data = Vec::with_capacity(seqs.len() * max_len);
        for seq in seqs {
            for &b in &seq.block_table {
                data.push(b as i64);
            }
            for _ in seq.block_table.len()..max_len {
                data.push(-1i64);
            }
        }
        Tensor::from_vec(data, (seqs.len(), max_len), &self.device)
    }

    fn prepare_model_input(&self, seqs: &[Sequence]) -> Result<(Tensor, Tensor)> {
        let block_size = self.engine_config.kvcache_block_size;

        let mut input_ids: Vec<i64> = Vec::new();
        let mut positions: Vec<i64> = Vec::new();
        let mut cu_seqlens_q: Vec<i64> = vec![0];
        let mut cu_seqlens_k: Vec<i64> = vec![0];
        let mut max_seqlen_q = 0usize;
        let mut max_seqlen_k = 0usize;
        let mut slot_mapping: Vec<i64> = Vec::new();
        let mut context_lens: Vec<i64> = Vec::new();
        let mut seq_need_compute_logits: Vec<i64> = Vec::new();

        for (seq_index, seq) in seqs.iter().enumerate() {
            let num_context_tokens = seq.num_context_tokens();

            if seq.len() == num_context_tokens && !seq.block_table.is_empty() {
                seq_need_compute_logits.push(seq_index as i64);
            }
            context_lens.push(num_context_tokens as i64);
            input_ids.extend(
                seq.slice_tokens(seq.num_cached_tokens, num_context_tokens)
                    .iter()
                    .map(|&t| t as i64),
            );
            positions.extend((seq.num_cached_tokens as i64)..(num_context_tokens as i64));

            let seqlen_q = seq.num_new_tokens;
            let seqlen_k = num_context_tokens;
            cu_seqlens_q.push(cu_seqlens_q.last().unwrap() + seqlen_q as i64);
            cu_seqlens_k.push(cu_seqlens_k.last().unwrap() + seqlen_k as i64);
            max_seqlen_q = max_seqlen_q.max(seqlen_q);
            max_seqlen_k = max_seqlen_k.max(seqlen_k);

            if seq.block_table.is_empty() {
                // warmup
                continue;
            }

            for i in seq.num_cached_blocks()..seq.block_table.len() {
                let start = if i == seq.num_cached_blocks() {
                    seq.block_table[i] * block_size + seq.num_cached_tokens % block_size
                } else {
                    seq.block_table[i] * block_size
                };
                let end = if i == seq.block_table.len() - 1 {
                    if num_context_tokens % block_size != 0 {
                        seq.block_table[i] * block_size + num_context_tokens % block_size
                    } else {
                        (seq.block_table[i] + 1) * block_size
                    }
                } else {
                    (seq.block_table[i] + 1) * block_size
                };
                slot_mapping.extend((start..end).map(|s| s as i64));
            }
        }

        let block_tables = if *cu_seqlens_k.last().unwrap() > *cu_seqlens_q.last().unwrap() {
            Some(self.prepare_block_tables(seqs)?)
        } else {
            None
        };

        let num_tokens = input_ids.len();
        let input_ids_t = Tensor::from_vec(input_ids, num_tokens, &self.device)?;
        let positions_t = Tensor::from_vec(positions, num_tokens, &self.device)?;
        let cu_seqlens_q_len = cu_seqlens_q.len();
        let cu_seqlens_q_t = Tensor::from_vec(cu_seqlens_q, cu_seqlens_q_len, &self.device)?;
        let cu_seqlens_k_len = cu_seqlens_k.len();
        let cu_seqlens_k_t = Tensor::from_vec(cu_seqlens_k, cu_seqlens_k_len, &self.device)?;
        let slot_mapping_len = slot_mapping.len();
        let slot_mapping_t = Tensor::from_vec(slot_mapping, slot_mapping_len, &self.device)?;
        let context_lens_len = context_lens.len();
        let context_lens_t = Tensor::from_vec(context_lens, context_lens_len, &self.device)?;
        let seq_need_compute_logits_len = seq_need_compute_logits.len();
        let seq_need_compute_logits_t =
            Tensor::from_vec(seq_need_compute_logits, seq_need_compute_logits_len, &self.device)?;

        set_context(
            Some(cu_seqlens_q_t),
            Some(cu_seqlens_k_t),
            max_seqlen_q,
            max_seqlen_k,
            Some(slot_mapping_t),
            Some(context_lens_t),
            block_tables,
            Some(seq_need_compute_logits_t),
        );

        Ok((input_ids_t, positions_t))
    }

    fn prepare_sample(&self, seqs: &[Sequence]) -> Result<Tensor> {
        let temperatures: Vec<f32> = seqs.iter().map(|s| s.temperature as f32).collect();
        let len = temperatures.len();
        let temperatures = Tensor::from_vec(temperatures, len, &self.device)?;

        let context = get_context();
        if let Some(idx) = context.seq_need_compute_logits.as_ref() {
            if idx.elem_count() > 0 {
                return temperatures.index_select(idx, 0);
            }
        }
        Ok(temperatures)
    }

    fn run_model(&mut self, input_ids: &Tensor, positions: &Tensor) -> Result<Tensor> {
        if self.engine_config.enforce_eager || input_ids.dim(0)? > 512 {
            let hidden = self.model.forward(input_ids, positions)?;
            self.model.compute_logits(&hidden)
        } else {
            self.capture_cudagraph_replay(input_ids, positions)
        }
    }

    fn capture_cudagraph_replay(&mut self, _input_ids: &Tensor, _positions: &Tensor) -> Result<Tensor> {
        unimplemented!(
            "CUDA graph replay is unreachable while EngineConfig::enforce_eager defaults to \
             true; see that field's doc comment for why capture can't work across the candle/torch split"
        )
    }

    fn capture_cudagraph(&mut self) {
        unimplemented!(
            "CUDA graph capture can't observe candle's kernel launches (see EngineConfig::enforce_eager doc)"
        )
    }

    pub fn run(&mut self, seqs: &mut [Sequence]) -> (Vec<u32>, Vec<usize>) {
        let (input_ids, positions) = self.prepare_model_input(seqs).expect("failed to prepare model input");
        let temperatures = self.prepare_sample(seqs).expect("failed to prepare sample");
        let logits = self.run_model(&input_ids, &positions).expect("model forward failed");
        let token_ids_tensor = self.sampler.forward(logits, temperatures).expect("sampling failed");
        let token_ids: Vec<u32> = token_ids_tensor
            .to_dtype(DType::U32)
            .and_then(|t| t.to_vec1())
            .expect("failed to extract sampled tokens");

        let context = get_context();
        let seq_need_compute_logits: Vec<usize> = context
            .seq_need_compute_logits
            .as_ref()
            .filter(|t| t.elem_count() > 0)
            .map(|t| t.to_dtype(DType::U32).and_then(|t| t.to_vec1::<u32>()))
            .transpose()
            .expect("failed to extract seq_need_compute_logits")
            .unwrap_or_default()
            .into_iter()
            .map(|i| i as usize)
            .collect();

        reset_context();

        (token_ids, seq_need_compute_logits)
    }

    fn warmup_model(&mut self) {
        let _ = Python::with_gil(|py| -> Result<()> {
            pybridge::cuda_empty_cache(py)?;
            pybridge::cuda_reset_peak_memory_stats(py)?;
            Ok(())
        });

        let max_num_batched_tokens = self.engine_config.max_num_batched_tokens;
        let max_model_len = self.config.max_model_len;
        let num_seqs = (max_num_batched_tokens / max_model_len)
            .min(self.engine_config.max_num_seqs)
            .max(1);
        // Never exceeds max_num_batched_tokens: that's the real ceiling on a single
        // step's token count, so a real request can't hit a length the profiling run didn't.
        let seq_len = max_model_len.min(max_num_batched_tokens);

        let mut seqs: Vec<Sequence> = (0..num_seqs)
            .map(|_| {
                Sequence::with_block_size(
                    vec![0u32; seq_len],
                    SamplingParams::default(),
                    self.engine_config.kvcache_block_size,
                )
            })
            .collect();
        for seq in seqs.iter_mut() {
            seq.num_new_tokens = seq_len;
        }

        let _ = self.run(&mut seqs);

        let _ = Python::with_gil(|py| pybridge::cuda_empty_cache(py));
    }

    fn num_kv_heads_per_rank(&self) -> usize {
        self.config.num_key_value_heads / self.engine_config.tensor_parallel_size
    }

    /// Computes the local KV-cache block budget without allocating; under TP the
    /// caller must reconcile (min) across ranks before calling `finish_setup`.
    pub fn probe_num_kvcache_blocks(&self) -> usize {
        let (free, total) =
            Python::with_gil(|py| pybridge::cuda_mem_get_info(py)).expect("mem_get_info failed");
        let used = total - free;
        let (peak, current) = Python::with_gil(|py| pybridge::cuda_memory_stats_peak_current(py))
            .expect("memory_stats failed");

        let num_kv_heads = self.num_kv_heads_per_rank();
        let head_dim = self.config.head_dim;
        let dtype_bytes = dtype_itemsize(&self.config.torch_dtype);
        let block_bytes = 2
            * self.config.num_hidden_layers
            * self.engine_config.kvcache_block_size
            * num_kv_heads
            * head_dim
            * dtype_bytes;

        let budget = (total as f64 * self.engine_config.gpu_memory_utilization) as i64 - used as i64
            - peak as i64
            + current as i64;
        let num_kvcache_blocks = (budget / block_bytes as i64).max(0) as usize;
        assert!(num_kvcache_blocks > 0, "not enough GPU memory left to allocate the KV cache");
        num_kvcache_blocks
    }

    fn allocate_kv_cache(&mut self, num_kvcache_blocks: usize) {
        let num_kv_heads = self.num_kv_heads_per_rank();
        let head_dim = self.config.head_dim;

        let kv_caches = Python::with_gil(|py| {
            pybridge::allocate_kv_cache(
                py,
                self.config.num_hidden_layers,
                num_kvcache_blocks,
                self.engine_config.kvcache_block_size,
                num_kv_heads,
                head_dim,
                self.config.dtype(),
            )
        })
        .expect("failed to allocate kv cache");

        self.model.set_kv_caches(kv_caches);
        self.num_kvcache_blocks = num_kvcache_blocks;
    }

    pub fn num_kvcache_blocks(&self) -> usize {
        self.num_kvcache_blocks
    }

    pub fn exit(&mut self) {
        let _ = Python::with_gil(|py| pybridge::cuda_synchronize(py));
    }
}

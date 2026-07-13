use candle_core::{Result, Tensor};
#[cfg(not(feature = "flash-attn"))]
use candle_core::IndexOp;

use crate::utils::context::get_context;

pub struct Attention {
    scale: f32,
    k_cache: Option<Tensor>,
    v_cache: Option<Tensor>,
}

impl Attention {
    pub fn new(scale: f32) -> Self {
        Self { scale, k_cache: None, v_cache: None }
    }

    pub fn set_kv_cache(&mut self, k_cache: Tensor, v_cache: Tensor) {
        self.k_cache = Some(k_cache);
        self.v_cache = Some(v_cache);
    }

    #[cfg(feature = "flash-attn")]
    pub fn forward(&mut self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let context = get_context();
        let cu_seqlens_q = context.cu_seqlens_q.as_ref().expect("cu_seqlens_q must be set");
        let cu_seqlens_k = context.cu_seqlens_k.as_ref().expect("cu_seqlens_k must be set");

        if let (Some(k_cache), Some(v_cache)) = (&self.k_cache, &self.v_cache) {
            let slot_mapping = context
                .slot_mapping
                .as_ref()
                .expect("slot_mapping must be set when a kv cache is present");
            crate::layers::kv_cache::store_kv_cache(k, v, k_cache, v_cache, slot_mapping)?;
        }

        let q = q.contiguous()?;
        match context.block_tables.as_ref() {
            Some(block_table) => {
                let k_cache = self.k_cache.as_ref().expect("kv cache must be set for prefix caching");
                let v_cache = self.v_cache.as_ref().expect("kv cache must be set for prefix caching");
                let page_block_size = k_cache.dim(1)?;
                candle_flash_attn::flash_attn_varlen_paged_windowed(
                    &q,
                    k_cache,
                    v_cache,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    block_table,
                    None,
                    context.max_seqlen_q,
                    context.max_seqlen_k,
                    self.scale,
                    None,
                    Some(0),
                    page_block_size,
                    None,
                )
            }
            None => {
                let k = k.contiguous()?;
                let v = v.contiguous()?;
                candle_flash_attn::flash_attn_varlen(
                    &q,
                    &k,
                    &v,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    context.max_seqlen_q,
                    context.max_seqlen_k,
                    self.scale,
                    true,
                )
            }
        }
    }

    /// Fallback for when the `flash-attn` feature is off (e.g. `candle-flash-attn` doesn't
    /// build for an older/unsupported GPU arch, or the extra CUDA kernel compile just isn't
    /// wanted): delegates the actual attention math to candle_nn's own unfused varlen
    /// implementation (`candle_nn::attention::varlen::flash_attn_varlen_unfused`), which
    /// already handles causal masking, the cache-offset case, and GQA. The only piece it
    /// doesn't provide is reading from our paged KV cache, so that's gathered into a flat
    /// per-sequence buffer first when a block_table is present.
    #[cfg(not(feature = "flash-attn"))]
    pub fn forward(&mut self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let context = get_context();
        let cu_seqlens_q = context.cu_seqlens_q.as_ref().expect("cu_seqlens_q must be set").to_vec1::<u32>()?;
        let cu_seqlens_k = context.cu_seqlens_k.as_ref().expect("cu_seqlens_k must be set").to_vec1::<u32>()?;

        if let (Some(k_cache), Some(v_cache)) = (&self.k_cache, &self.v_cache) {
            let slot_mapping = context
                .slot_mapping
                .as_ref()
                .expect("slot_mapping must be set when a kv cache is present");
            crate::layers::kv_cache::store_kv_cache(k, v, k_cache, v_cache, slot_mapping)?;
        }

        let num_seqs = cu_seqlens_q.len() - 1;
        let seqlens_q: Vec<u32> = (0..num_seqs).map(|i| cu_seqlens_q[i + 1] - cu_seqlens_q[i]).collect();
        let seqlens_k: Vec<u32> = (0..num_seqs).map(|i| cu_seqlens_k[i + 1] - cu_seqlens_k[i]).collect();
        let seqlens_q_t = Tensor::new(seqlens_q.as_slice(), q.device())?;
        let seqlens_k_t = Tensor::new(seqlens_k.as_slice(), q.device())?;

        let (k, v) = match context.block_tables.as_ref() {
            Some(block_table) => {
                let k_cache = self.k_cache.as_ref().expect("kv cache must be set for prefix caching");
                let v_cache = self.v_cache.as_ref().expect("kv cache must be set for prefix caching");
                let page_block_size = k_cache.dim(1)?;
                let mut k_parts = Vec::with_capacity(num_seqs);
                let mut v_parts = Vec::with_capacity(num_seqs);
                for i in 0..num_seqs {
                    let row = block_table.i(i)?.to_vec1::<u32>()?;
                    let (k_i, v_i) = gather_paged_kv(k_cache, v_cache, &row, seqlens_k[i] as usize, page_block_size)?;
                    k_parts.push(k_i);
                    v_parts.push(v_i);
                }
                (Tensor::cat(&k_parts, 0)?, Tensor::cat(&v_parts, 0)?)
            }
            None => (k.contiguous()?, v.contiguous()?),
        };

        candle_nn::attention::varlen::flash_attn_varlen_unfused(
            q,
            &k,
            &v,
            None,
            &seqlens_q_t,
            &seqlens_k_t,
            context.max_seqlen_q,
            context.max_seqlen_k,
            self.scale,
            true,
            None,
            None,
        )
    }
}

/// Gathers one sequence's key/value rows out of the paged cache, in token order, via its
/// block_table row. `k_cache`/`v_cache` are `(num_blocks, block_size, num_kv_heads, head_dim)`.
#[cfg(not(feature = "flash-attn"))]
fn gather_paged_kv(
    k_cache: &Tensor,
    v_cache: &Tensor,
    block_table_row: &[u32],
    seqlen_k: usize,
    page_block_size: usize,
) -> Result<(Tensor, Tensor)> {
    let mut slots = Vec::with_capacity(seqlen_k);
    'outer: for &block in block_table_row {
        for s in 0..page_block_size as u32 {
            if slots.len() == seqlen_k {
                break 'outer;
            }
            slots.push(block * page_block_size as u32 + s);
        }
    }
    let idx = Tensor::new(slots.as_slice(), k_cache.device())?;
    let (num_kv_heads, head_dim) = (k_cache.dim(2)?, k_cache.dim(3)?);
    let k_flat = k_cache.reshape(((), num_kv_heads, head_dim))?;
    let v_flat = v_cache.reshape(((), num_kv_heads, head_dim))?;
    Ok((k_flat.index_select(&idx, 0)?, v_flat.index_select(&idx, 0)?))
}

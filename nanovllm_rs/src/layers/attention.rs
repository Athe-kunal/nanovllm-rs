use candle_core::{Result, Tensor};

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
}

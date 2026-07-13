use candle_core::{DType, IndexOp, Result, Tensor, D};

use crate::utils::context::get_context;

/// Pure-candle attention, no flash-attn involved at all: an independent ground truth
/// for the debug comparison below, sharing no code path with either flash-attn variant.
fn manual_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    let n_heads = q.dim(1)?;
    let n_kv_heads = k.dim(1)?;
    let group = n_heads / n_kv_heads;
    let head_dim = q.dim(2)?;
    let total_len = k.dim(0)?;

    let q = q.to_dtype(DType::F32)?.transpose(0, 1)?;
    let k = k.to_dtype(DType::F32)?.transpose(0, 1)?;
    let v = v.to_dtype(DType::F32)?.transpose(0, 1)?;

    let k = k
        .unsqueeze(1)?
        .broadcast_as((n_kv_heads, group, total_len, head_dim))?
        .reshape((n_heads, total_len, head_dim))?
        .contiguous()?;
    let v = v
        .unsqueeze(1)?
        .broadcast_as((n_kv_heads, group, total_len, head_dim))?
        .reshape((n_heads, total_len, head_dim))?
        .contiguous()?;

    let scores = (q.matmul(&k.transpose(1, 2)?)? * scale as f64)?;
    let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
    let out = probs.matmul(&v)?;
    out.transpose(0, 1)?.contiguous()
}

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

        if std::env::var("NANOVLLM_DEBUG_QK").is_ok() {
            let qf = q.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let kf = k.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let stats = |v: &[f32]| -> (f32, f32, f32, bool) {
                let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mean = v.iter().sum::<f32>() / v.len() as f32;
                let has_nan = v.iter().any(|x| x.is_nan());
                (min, max, mean, has_nan)
            };
            let (qmin, qmax, qmean, qnan) = stats(&qf);
            let (kmin, kmax, kmean, knan) = stats(&kf);
            eprintln!(
                "[qk-debug] qshape={:?} q(min={qmin:.4},max={qmax:.4},mean={qmean:.4},nan={qnan}) kshape={:?} k(min={kmin:.4},max={kmax:.4},mean={kmean:.4},nan={knan})",
                q.dims(), k.dims()
            );
        }

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
                let paged = candle_flash_attn::flash_attn_varlen_paged_windowed(
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
                )?;

                if std::env::var("NANOVLLM_DEBUG_PAGED").is_ok() && block_table.dim(0)? == 1 {
                    let num_kv_heads = k_cache.dim(2)?;
                    let head_dim = k_cache.dim(3)?;
                    let seqlen_k = context.max_seqlen_k;
                    let blocks: Vec<u32> = block_table.i(0)?.to_dtype(candle_core::DType::U32)?.to_vec1()?;
                    let mut slot_idx: Vec<u32> = Vec::with_capacity(seqlen_k);
                    for &b in &blocks {
                        for s in 0..page_block_size as u32 {
                            if slot_idx.len() == seqlen_k {
                                break;
                            }
                            slot_idx.push(b * page_block_size as u32 + s);
                        }
                    }
                    let idx = Tensor::new(slot_idx.as_slice(), q.device())?;
                    let k_flat = k_cache.reshape(((), num_kv_heads, head_dim))?;
                    let v_flat = v_cache.reshape(((), num_kv_heads, head_dim))?;
                    let k_gathered = k_flat.index_select(&idx, 0)?;
                    let v_gathered = v_flat.index_select(&idx, 0)?;
                    let seqlens_k_ref = Tensor::new(&[0u32, seqlen_k as u32], q.device())?;
                    let reference = candle_flash_attn::flash_attn_varlen(
                        &q, &k_gathered, &v_gathered, cu_seqlens_q, &seqlens_k_ref,
                        context.max_seqlen_q, seqlen_k, self.scale, true,
                    )?;
                    let diff = reference
                        .to_dtype(DType::F32)?
                        .sub(&paged.to_dtype(DType::F32)?)?
                        .abs()?
                        .flatten_all()?
                        .max(0)?
                        .to_vec0::<f32>()?;

                    let manual_diff = if context.max_seqlen_q == 1 {
                        let manual = manual_attention(&q, &k_gathered, &v_gathered, self.scale)?;
                        let paged_f32 = paged.to_dtype(DType::F32)?;
                        let abs_diff = manual.sub(&paged_f32)?.abs()?.flatten_all()?.max(0)?.to_vec0::<f32>()?;
                        let manual_vals = manual.flatten_all()?.to_vec1::<f32>()?;
                        let paged_vals = paged_f32.flatten_all()?.to_vec1::<f32>()?;
                        let mag = |v: &[f32]| -> f32 { v.iter().cloned().fold(0f32, |a, b| a.max(b.abs())) };
                        Some((abs_diff, mag(&manual_vals), mag(&paged_vals)))
                    } else {
                        None
                    };

                    eprintln!(
                        "[paged-debug] seqlen_k={seqlen_k} vs_gathered_flashattn={diff} vs_manual(abs_diff,manual_mag,paged_mag)={manual_diff:?}"
                    );
                }

                Ok(paged)
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

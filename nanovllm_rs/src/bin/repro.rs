use candle_core::{DType, Device, Result, Tensor, D};
use nanovllm_rs::layers::kv_cache::store_kv_cache;

/// Pure-candle attention with no flash-attn involved at all: an independent ground truth
/// that doesn't share any code path with either flash_attn_varlen or the paged variant.
fn manual_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    let n_heads = q.dim(1)?;
    let n_kv_heads = k.dim(1)?;
    let group = n_heads / n_kv_heads;
    let head_dim = q.dim(2)?;
    let total_len = k.dim(0)?;

    let q = q.to_dtype(DType::F32)?.transpose(0, 1)?; // (n_heads, 1, head_dim)
    let k = k.to_dtype(DType::F32)?.transpose(0, 1)?; // (n_kv_heads, total_len, head_dim)
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

    let scores = (q.matmul(&k.transpose(1, 2)?)? * scale as f64)?; // (n_heads, 1, total_len)
    let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
    let out = probs.matmul(&v)?; // (n_heads, 1, head_dim)
    out.transpose(0, 1)?.contiguous() // (1, n_heads, head_dim)
}

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    for layer in 0..28 {
        run_layer(&device, layer)?;
    }
    Ok(())
}

fn run_layer(device: &Device, layer: usize) -> Result<()> {
    let n_heads = 16;
    let n_kv_heads = 8;
    let head_dim = 128;
    let block_size = 256;
    let prior_len = 6; // tokens already in cache before this decode step
    let total_len = prior_len + 1; // + the new token being decoded

    let mk = |seed: f32, n: usize, h: usize| -> Result<Tensor> {
        let data: Vec<f32> = (0..n * h * head_dim).map(|i| ((i as f32 * seed) % 17.0 - 8.0) / 8.0).collect();
        Tensor::from_vec(data, (n, h, head_dim), device)?.to_dtype(DType::BF16)
    };

    let base = 1.0 + layer as f32 * 0.13;
    let k_prior = mk(0.7 * base, prior_len, n_kv_heads)?;
    let v_prior = mk(1.3 * base, prior_len, n_kv_heads)?;
    let k_new = mk(0.9 * base, 1, n_kv_heads)?;
    let v_new = mk(1.7 * base, 1, n_kv_heads)?;
    let q_new = mk(2.1 * base, 1, n_heads)?;

    let k_full = Tensor::cat(&[&k_prior, &k_new], 0)?;
    let v_full = Tensor::cat(&[&v_prior, &v_new], 0)?;

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Reference: non-paged decode-style call (q_len=1, k_len=total_len), proven-correct path.
    let seqlens_q = Tensor::new(&[0u32, 1u32], device)?;
    let seqlens_k = Tensor::new(&[0u32, total_len as u32], device)?;
    let reference = candle_flash_attn::flash_attn_varlen(
        &q_new, &k_full, &v_full, &seqlens_q, &seqlens_k, 1, total_len, scale, true,
    )?
    .to_dtype(DType::F32)?;

    // Paged: TWO SEPARATE store_kv_cache calls (prefill then decode), exactly matching
    // how model_runner.rs/attention.rs really populate the cache across steps.
    let num_blocks = 1;
    let k_cache = Tensor::zeros((num_blocks, block_size, n_kv_heads, head_dim), DType::BF16, device)?;
    let v_cache = Tensor::zeros((num_blocks, block_size, n_kv_heads, head_dim), DType::BF16, device)?;

    let prior_slots: Vec<i64> = (0..prior_len as i64).collect();
    let prior_slot_mapping = Tensor::new(prior_slots.as_slice(), device)?;
    store_kv_cache(&k_prior, &v_prior, &k_cache, &v_cache, &prior_slot_mapping)?;

    let new_slot_mapping = Tensor::new(&[prior_len as i64], device)?;
    store_kv_cache(&k_new, &v_new, &k_cache, &v_cache, &new_slot_mapping)?;

    let block_table = Tensor::new(&[0u32], device)?.reshape((1, 1))?;

    let paged = candle_flash_attn::flash_attn_varlen_paged_windowed(
        &q_new,
        &k_cache,
        &v_cache,
        &seqlens_q,
        &seqlens_k,
        &block_table,
        None,
        1,
        total_len,
        scale,
        None,
        Some(0),
        block_size,
        None,
    )?
    .to_dtype(DType::F32)?;

    let diff = reference.sub(&paged)?.abs()?.flatten_all()?.max(0)?.to_vec0::<f32>()?;

    let manual = manual_attention(&q_new, &k_full, &v_full, scale)?;
    let manual_vs_paged =
        manual.sub(&paged)?.abs()?.flatten_all()?.max(0)?.to_vec0::<f32>()?;
    let manual_vs_reference =
        manual.sub(&reference)?.abs()?.flatten_all()?.max(0)?.to_vec0::<f32>()?;

    let cache_check = k_cache
        .reshape((num_blocks * block_size, n_kv_heads, head_dim))?
        .narrow(0, 0, total_len)?
        .to_dtype(DType::F32)?;
    let k_full_check = k_full.to_dtype(DType::F32)?;
    let cache_diff = cache_check.sub(&k_full_check)?.abs()?.flatten_all()?.max(0)?.to_vec0::<f32>()?;
    println!(
        "layer={layer} paged_vs_ref={diff} manual_vs_paged={manual_vs_paged} manual_vs_ref={manual_vs_reference} cache_write_diff={cache_diff}"
    );
    Ok(())
}

use candle_core::{IndexOp, D, DType, Device, Result, Tensor};

pub struct RotaryEmbedding{
    head_size: usize,
    rotary_dim: usize,
    max_position_embeddings: usize,
    base: f32,
    cos_sin_cache: Tensor,
}

pub fn apply_rotary_emb(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<Tensor> {
    let last_dim = x.dim(D::Minus1)?;
    let half = last_dim / 2;

    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;

    let y1 = (x1.broadcast_mul(cos)? - x2.broadcast_mul(sin)?)?;
    let y2 = (x2.broadcast_mul(cos)? + x1.broadcast_mul(sin)?)?;

    Tensor::cat(&[&y1, &y2], D::Minus1)
}

impl RotaryEmbedding{
    pub fn new(head_size: usize, rotary_dim: usize, max_position_embeddings: usize, base:f32, device: &Device) -> Result<Self>{
        assert_eq!(head_size,rotary_dim);
        // torch: 1.0 / (base ** (arange(0, rotary_dim, 2).float() / rotary_dim))
        let exponent = (Tensor::arange_step(0u32, rotary_dim as u32, 2u32, device)?
            .to_dtype(DType::F32)?
            / rotary_dim as f64)?;

        // base ** exponent == exp(exponent * ln(base))
        let inv_freq = (exponent * (base as f64).ln())?.exp()?.recip()?;
        let t = Tensor::arange(0u32, max_position_embeddings as u32, device)?.to_dtype(DType::F32)?;
        let t = t.unsqueeze(1)?;
        let inv_freq_row = inv_freq.unsqueeze(0)?;
        let freqs = t.broadcast_mul(&inv_freq_row)?;

        let cos = freqs.cos()?;
        let sin = freqs.sin()?;
        // cat along the dim -> [max_pos, rotary_dim];
        let cache = Tensor::cat(&[&cos, &sin], D::Minus1)?;
        // broadcast along the sequence length
        // [max_pos, 1, rotary_dim], so that the first dimension broadcasts in sequence length
        let cache = cache.unsqueeze(1)?;
        Ok(Self {
            head_size,
            rotary_dim,
            max_position_embeddings,
            base,
            cos_sin_cache: cache
        })
    }

    pub fn forward(&self, positions: &Tensor, query: &Tensor, key: &Tensor) -> Result<(Tensor, Tensor)>{
        let cos_sin = self.cos_sin_cache.index_select(positions, 0)?.to_dtype(query.dtype())?;
        let last_dim = cos_sin.dim(D::Minus1)?;
        let half = last_dim / 2;
        let cos = cos_sin.narrow(D::Minus1, 0, half)?;
        let sin = cos_sin.narrow(D::Minus1, half, half)?;

        if std::env::var("NANOVLLM_DEBUG_ROPE").is_ok() {
            let pos: Vec<i64> = positions.to_dtype(DType::I64)?.flatten_all()?.to_vec1()?;
            let cos0: Vec<f32> = cos.i(0)?.i(0)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            let sin0: Vec<f32> = sin.i(0)?.i(0)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            eprintln!(
                "[rope-debug] positions={pos:?} cos[..3]={:?} sin[..3]={:?}",
                &cos0[..3], &sin0[..3]
            );
        }

        let query = apply_rotary_emb(query, &cos, &sin)?;
        let key = apply_rotary_emb(key, &cos, &sin)?;

        Ok((query, key))
    }
}
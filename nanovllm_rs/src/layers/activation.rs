use candle_core::{Tensor, Result, D};

pub struct SiluAndMul;

impl SiluAndMul {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // last dim 6144 (for qwen 0.6B)
        let last_dim = x.dim(D::Minus1)?;
        // half dim: 3072
        let half = last_dim / 2;
        // x1[...,0:3072]
        let x1 = x.narrow(D::Minus1, 0, half)?;
        // x2[...,3072:6144]
        let x2 = x.narrow(D::Minus1, half, half)?;
        // ?: It outputs Result<Tensor>
        x1.silu()?.mul(&x2)
    }
}
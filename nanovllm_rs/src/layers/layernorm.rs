use candle_core::{Tensor, Result, Device, DType, D};

pub struct RMSNorm{
    weight: Tensor,
    eps: f32,
}

impl RMSNorm {
    pub fn new(hidden_size: usize, eps: f32, device: &Device) -> Result<Self>{
        let weight = Tensor::ones(hidden_size, DType::F32, device)?;
        Ok(Self { weight, eps})
    }

    pub fn weight_loader(&mut self, loaded_weight: Tensor) {
        self.weight = loaded_weight;
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor>{
        let orig_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let variance = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let inv_std = (variance + self.eps as f64)?.sqrt()?.recip()?;
        let x_normed = x_f32.broadcast_mul(&inv_std)?.to_dtype(orig_dtype)?;
        x_normed.broadcast_mul(&self.weight)
    }

    pub fn residual_forward(&self, x: Tensor, residual: Tensor) -> Result<(Tensor, Tensor)>{
        let orig_dtype = x.dtype();

        let x_f32 = (x.to_dtype(DType::F32)? + residual.to_dtype(DType::F32)?)?;

        let new_residual = x_f32.to_dtype(orig_dtype)?;

        let variance = x_f32.sqr()?.mean_keepdim(D::Minus1)?;

        let inv_std = (variance + self.eps as f64)?.sqrt()?.recip()?;
        let x_normed = x_f32.broadcast_mul(&inv_std)?;

        let out = x_normed.to_dtype(orig_dtype)?.broadcast_mul(&self.weight)?;

        Ok((out, new_residual))
    }
}
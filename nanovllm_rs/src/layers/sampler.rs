use candle_core::{DType, Result, Tensor, D};
use candle_nn;

pub struct Sampler;

impl Sampler{
    pub fn forward(&self, logits: Tensor, temperatures: Tensor) -> Result<Tensor>{
        let logits = logits.to_dtype(DType::F32)?;
        let temps = temperatures.to_dtype(DType::F32)?.unsqueeze(1)?;
        let logits = logits.broadcast_div(&temps)?;

        let probs = candle_nn::ops::softmax(&logits, D::Minus1)?;

        let noise = Tensor::rand(0f32, 1f32, probs.shape(), probs.device())?
            .log()?
            .neg()?
            .clamp(1e-10, f32::INFINITY)?;

        let sample_tokens = probs.broadcast_div(&noise)?.argmax(D::Minus1)?;

        Ok(sample_tokens)
    }
}
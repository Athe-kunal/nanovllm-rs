use candle_core::{Result, Tensor, DType};
use crate::config::Config;

pub struct Qwen3ForCausalLM {
    config: Config,
}

impl Qwen3ForCausalLM {
    pub fn from_pretrained(model_path: &str) -> Result<Self> {
        let config = Config::from_pretrained(model_path)?;
        Ok(Self { config })
    }
}

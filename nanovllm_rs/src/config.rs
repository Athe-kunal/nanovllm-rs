use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub max_model_len: usize,
    pub eos_token_id: u32,
    pub hidden_act: String,
    #[serde(default = "default_attention_bias")]
    pub attention_bias: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

fn default_attention_bias() -> bool {
    true
}

impl Config {
    /// Equivalent of `transformers.Qwen3Config.from_pretrained(model_path)`:
    /// reads `config.json` from the model directory.
    pub fn from_pretrained<P: AsRef<Path>>(model_path: P) -> candle_core::Result<Self> {
        let config_path = model_path.as_ref().join("config.json");
        let file = std::fs::File::open(&config_path).map_err(candle_core::Error::wrap)?;
        serde_json::from_reader(file).map_err(candle_core::Error::wrap)
    }

    // Qwen3-0.6B, hardcoded — fill in with your actual model's values
    pub fn qwen3_0_6b() -> Self {
        Self {
            hidden_size: 1024,
            intermediate_size: 3072,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            vocab_size: 151936,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 40960,
            max_model_len: 40960,
            eos_token_id: 151645,
            hidden_act: "silu".to_string(),
            attention_bias: false,
            tie_word_embeddings: true,
        }
    }
}
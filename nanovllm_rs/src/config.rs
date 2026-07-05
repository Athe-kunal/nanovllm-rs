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
    pub max_model_len: usize,
    pub eos_token_id: u32,
}

impl Config {
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
            max_model_len: 40960,
            eos_token_id: 151645,
        }
    }
}
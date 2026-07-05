pub struct SamplingParams {
    pub temperature: f64,
    pub max_tokens: i32,
    pub ignore_eos: bool,
}

impl SamplingParams {
    pub fn new(temperature: f64, max_tokens: i32, ignore_eos: bool) -> Self {
        assert!(temperature > 1e-10, "temperature must be > 1e-10, got {}", temperature);
        Self { temperature, max_tokens, ignore_eos }
    }
}
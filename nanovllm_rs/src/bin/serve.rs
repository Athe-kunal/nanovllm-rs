//! Minimal `vllm serve`-style entry point:
//!
//!     serve <model_path> --tensor-parallel-size 2 --port 8000
//!
//! Exposes POST /generate, body {"prompt": str, "max_tokens": int, "temperature": float}.
use std::sync::{Arc, Mutex};

use axum::{extract::State, routing::post, Json, Router};
use serde::{Deserialize, Serialize};

use nanovllm_rs::config::{Config, EngineConfig};
use nanovllm_rs::engine::llm_engine::LLMEngine;
use nanovllm_rs::sampling_params::SamplingParams;

struct Args {
    model_path: String,
    tensor_parallel_size: usize,
    port: u16,
}

fn parse_args() -> Args {
    let mut model_path: Option<String> = None;
    let mut tensor_parallel_size = 1usize;
    let mut port = 8000u16;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--tensor-parallel-size" => {
                let value = args.next().expect("--tensor-parallel-size requires a value");
                tensor_parallel_size = value.parse().expect("--tensor-parallel-size must be a positive integer");
            }
            "--port" => {
                let value = args.next().expect("--port requires a value");
                port = value.parse().expect("--port must be a valid port number");
            }
            "--model" => {
                model_path = Some(args.next().expect("--model requires a value"));
            }
            other => model_path = Some(other.to_string()),
        }
    }

    Args {
        model_path: model_path.expect("usage: serve <model_path> [--tensor-parallel-size N] [--port P]"),
        tensor_parallel_size,
        port,
    }
}

#[derive(Deserialize)]
struct GenerateRequest {
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: i32,
    #[serde(default = "default_temperature")]
    temperature: f64,
}

fn default_max_tokens() -> i32 {
    64
}

fn default_temperature() -> f64 {
    1.0
}

#[derive(Serialize)]
struct GenerateResponse {
    text: String,
    token_ids: Vec<u32>,
}

type SharedEngine = Arc<Mutex<LLMEngine>>;

async fn generate(State(engine): State<SharedEngine>, Json(req): Json<GenerateRequest>) -> Json<GenerateResponse> {
    let output = tokio::task::spawn_blocking(move || {
        let sampling_params = SamplingParams::new(req.temperature, req.max_tokens, false);
        let mut engine = engine.lock().expect("engine mutex poisoned");
        engine.generate(vec![req.prompt], vec![sampling_params], false).remove(0)
    })
    .await
    .expect("generation task panicked");

    Json(GenerateResponse { text: output.text, token_ids: output.token_ids })
}

#[tokio::main]
async fn main() {
    let args = parse_args();
    let config = Config::from_pretrained(&args.model_path).expect("failed to load model config");
    let engine_config = EngineConfig {
        model_path: args.model_path,
        tensor_parallel_size: args.tensor_parallel_size,
        ..EngineConfig::default()
    };

    let engine: SharedEngine = Arc::new(Mutex::new(LLMEngine::new(config, engine_config)));
    let app = Router::new().route("/generate", post(generate)).with_state(engine);

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", args.port)).await.expect("failed to bind port");
    println!("listening on http://0.0.0.0:{}", args.port);
    axum::serve(listener, app).await.expect("server error");
}

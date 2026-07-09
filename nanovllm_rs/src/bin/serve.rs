//! Minimal `vllm serve`-style entry point:
//!
//!     serve <model_path> --tensor-parallel-size 2 --port 8000
//!
//! Exposes:
//!   POST /generate  body {"prompt": str, "max_tokens": int, "temperature": float}
//!   POST /chat      body {"messages": [{"role": str, "content": str}], "max_tokens": int, "temperature": float}
//!
//! The engine is single-threaded and GPU-bound, so it runs on a dedicated
//! background thread that owns it exclusively. Axum handlers submit requests
//! over a channel and await a oneshot reply; the engine thread keeps draining
//! newly-submitted requests into the scheduler between steps, so concurrent
//! HTTP requests are continuously batched together on the GPU rather than
//! being serialized behind a lock.
use std::collections::HashMap;
use std::sync::mpsc as std_mpsc;
use std::sync::Arc;

use axum::{extract::FromRef, extract::State, routing::post, Json, Router};
use minijinja::{context, Environment};
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

use nanovllm_rs::config::{Config, EngineConfig};
use nanovllm_rs::engine::llm_engine::LLMEngine;
use nanovllm_rs::sampling_params::SamplingParams;

struct Args {
    model_path: String,
    tensor_parallel_size: usize,
    port: u16,
    max_num_batched_tokens: Option<usize>,
}

fn parse_args() -> Args {
    let mut model_path: Option<String> = None;
    let mut tensor_parallel_size = 1usize;
    let mut port = 8000u16;
    let mut max_num_batched_tokens = None;

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
            "--max-num-batched-tokens" => {
                let value = args.next().expect("--max-num-batched-tokens requires a value");
                max_num_batched_tokens =
                    Some(value.parse().expect("--max-num-batched-tokens must be a positive integer"));
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
        max_num_batched_tokens,
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

#[derive(Deserialize, Serialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatRequest {
    messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    max_tokens: i32,
    #[serde(default = "default_temperature")]
    temperature: f64,
}

// Loads the model's own chat_template out of tokenizer_config.json (a Jinja2
// template written by the model author) and renders it with minijinja, rather
// than hand-rolling ChatML — this gets tool-calling / <think> handling for
// free and stays correct for whatever template a different model ships.
fn load_chat_env(model_path: &str) -> Environment<'static> {
    let tokenizer_config_path = std::path::Path::new(model_path).join("tokenizer_config.json");
    let file = std::fs::File::open(&tokenizer_config_path).expect("failed to open tokenizer_config.json");
    let tokenizer_config: serde_json::Value =
        serde_json::from_reader(file).expect("failed to parse tokenizer_config.json");
    let chat_template = tokenizer_config
        .get("chat_template")
        .and_then(|v| v.as_str())
        .expect("tokenizer_config.json has no chat_template")
        .to_string();

    let mut env = Environment::new();
    minijinja_contrib::add_to_environment(&mut env);
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_template_owned("chat", chat_template).expect("invalid chat_template");
    env
}

fn render_chat_prompt(env: &Environment<'static>, messages: &[ChatMessage]) -> String {
    let template = env.get_template("chat").expect("chat template not registered");
    template
        .render(context! { messages => messages, add_generation_prompt => true })
        .expect("chat template render failed")
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

struct EngineRequest {
    prompt: String,
    sampling_params: SamplingParams,
    respond_to: oneshot::Sender<GenerateResponse>,
}

type EngineHandle = std_mpsc::Sender<EngineRequest>;
type ChatEnv = Arc<Environment<'static>>;

#[derive(Clone)]
struct AppState {
    engine: EngineHandle,
    chat_env: ChatEnv,
}

impl FromRef<AppState> for EngineHandle {
    fn from_ref(state: &AppState) -> Self {
        state.engine.clone()
    }
}

impl FromRef<AppState> for ChatEnv {
    fn from_ref(state: &AppState) -> Self {
        state.chat_env.clone()
    }
}

// Owns the engine exclusively on one OS thread. Drains every request that's
// immediately available before each `step()`, so requests submitted while a
// previous one is still generating join the same batch instead of queueing
// behind it.
fn spawn_engine_thread(mut engine: LLMEngine) -> EngineHandle {
    let (tx, rx) = std_mpsc::channel::<EngineRequest>();

    std::thread::spawn(move || {
        let mut pending: HashMap<usize, oneshot::Sender<GenerateResponse>> = HashMap::new();

        loop {
            if pending.is_empty() {
                match rx.recv() {
                    Ok(req) => enqueue(&mut engine, &mut pending, req),
                    Err(_) => break,
                }
            }
            while let Ok(req) = rx.try_recv() {
                enqueue(&mut engine, &mut pending, req);
            }

            let (outputs, _num_tokens) = engine.step();
            for (seq_id, token_ids) in outputs {
                if let Some(sender) = pending.remove(&seq_id) {
                    let text = engine.decode(&token_ids);
                    let _ = sender.send(GenerateResponse { text, token_ids });
                }
            }
        }
    });

    tx
}

fn enqueue(engine: &mut LLMEngine, pending: &mut HashMap<usize, oneshot::Sender<GenerateResponse>>, req: EngineRequest) {
    let seq_id = engine.add_request_text(&req.prompt, req.sampling_params);
    pending.insert(seq_id, req.respond_to);
}

async fn submit(engine: &EngineHandle, prompt: String, sampling_params: SamplingParams) -> GenerateResponse {
    let (tx, rx) = oneshot::channel();
    engine
        .send(EngineRequest { prompt, sampling_params, respond_to: tx })
        .expect("engine thread terminated unexpectedly");
    rx.await.expect("engine thread dropped the response channel")
}

async fn generate(State(engine): State<EngineHandle>, Json(req): Json<GenerateRequest>) -> Json<GenerateResponse> {
    let sampling_params = SamplingParams::new(req.temperature, req.max_tokens, false);
    Json(submit(&engine, req.prompt, sampling_params).await)
}

async fn chat(
    State(engine): State<EngineHandle>,
    State(chat_env): State<ChatEnv>,
    Json(req): Json<ChatRequest>,
) -> Json<GenerateResponse> {
    let prompt = render_chat_prompt(&chat_env, &req.messages);
    let sampling_params = SamplingParams::new(req.temperature, req.max_tokens, false);
    Json(submit(&engine, prompt, sampling_params).await)
}

#[tokio::main]
async fn main() {
    let args = parse_args();
    let config = Config::from_pretrained(&args.model_path).expect("failed to load model config");
    let mut engine_config = EngineConfig {
        model_path: args.model_path,
        tensor_parallel_size: args.tensor_parallel_size,
        ..EngineConfig::default()
    };
    if let Some(max_num_batched_tokens) = args.max_num_batched_tokens {
        engine_config.max_num_batched_tokens = max_num_batched_tokens;
    }

    let chat_env = Arc::new(load_chat_env(&engine_config.model_path));

    let engine = LLMEngine::new(config, engine_config);
    let engine_handle = spawn_engine_thread(engine);

    let app = Router::new()
        .route("/generate", post(generate))
        .route("/chat", post(chat))
        .with_state(AppState { engine: engine_handle, chat_env });

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", args.port)).await.expect("failed to bind port");
    println!("listening on http://0.0.0.0:{}", args.port);
    axum::serve(listener, app).await.expect("server error");
}

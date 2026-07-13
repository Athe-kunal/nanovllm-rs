use serde::Deserialize;
use std::time::Duration;

const PROMPTS: &[&str] = &[
    "What is the capital of France?",
    "What is the capital of Japan?",
    "What is 2 + 2 =",
];

#[derive(Deserialize)]
struct GenerateResponse {
    text: String,
}

#[tokio::main]
async fn main() {
    let mut host = "localhost".to_string();
    let mut port = 8000u16;
    let mut max_tokens = 256u32;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--host" => host = args.next().expect("--host requires a value"),
            "--port" => port = args.next().expect("--port requires a value").parse().expect("--port must be a number"),
            "--max-tokens" => {
                max_tokens = args.next().expect("--max-tokens requires a value").parse().expect("--max-tokens must be a number")
            }
            other => panic!("unknown argument: {other}"),
        }
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .expect("failed to build http client");
    let url = format!("http://{host}:{port}/generate");

    let requests = PROMPTS.iter().map(|&prompt| {
        let client = client.clone();
        let url = url.clone();
        async move {
            let response = client
                .post(&url)
                .json(&serde_json::json!({ "prompt": prompt, "max_tokens": max_tokens }))
                .send()
                .await
                .unwrap_or_else(|e| panic!("request for {prompt:?} failed: {e}"));
            let body: GenerateResponse = response
                .json()
                .await
                .unwrap_or_else(|e| panic!("failed to parse response for {prompt:?}: {e}"));
            body.text
        }
    });

    let results = futures_join_all(requests).await;

    for (prompt, text) in PROMPTS.iter().zip(results) {
        println!("{prompt} -> {text}");
    }
}

/// Minimal stand-in for `futures::future::join_all` so this binary doesn't need
/// the `futures` crate just for one call: spawn each future so they run
/// concurrently, then await them in order.
async fn futures_join_all<F>(futures: impl Iterator<Item = F>) -> Vec<F::Output>
where
    F: std::future::Future + Send + 'static,
    F::Output: Send + 'static,
{
    let handles: Vec<_> = futures.map(tokio::spawn).collect();
    let mut results = Vec::with_capacity(handles.len());
    for handle in handles {
        results.push(handle.await.expect("task panicked"));
    }
    results
}

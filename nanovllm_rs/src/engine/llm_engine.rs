use std::collections::HashMap;
use std::time::Instant;

use indicatif::{ProgressBar, ProgressStyle};
use tokenizers::Tokenizer;

use crate::config::{Config, EngineConfig};
use crate::engine::model_runner::ModelRunner;
use crate::engine::scheduler::Scheduler;
use crate::engine::sequence::Sequence;
use crate::engine::worker::{self, WorkerHandle};
use crate::sampling_params::SamplingParams;

pub struct GeneratedOutput {
    pub text: String,
    pub token_ids: Vec<u32>,
}

pub struct LLMEngine {
    model_runner: ModelRunner,
    workers: Vec<WorkerHandle>,
    tokenizer: Tokenizer,
    scheduler: Scheduler,
    block_size: usize,
}

impl LLMEngine {
    pub fn new(config: Config, mut engine_config: EngineConfig) -> Self {
        let block_size = engine_config.kvcache_block_size;

        let (model_runner, workers) = if engine_config.tensor_parallel_size > 1 {
            worker::init_tensor_parallel(&config, &engine_config)
        } else {
            let mut runner = ModelRunner::new(&config, &engine_config, 0, None);
            let num_blocks = runner.probe_num_kvcache_blocks();
            runner.finish_setup(num_blocks);
            (runner, Vec::new())
        };
        engine_config.num_kvcache_blocks = model_runner.num_kvcache_blocks();

        let tokenizer_path = std::path::Path::new(&engine_config.model_path).join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).expect("failed to load tokenizer");

        let scheduler = Scheduler::new(&config, &engine_config);

        Self { model_runner, workers, tokenizer, scheduler, block_size }
    }

    pub fn add_request_text(&mut self, prompt: &str, sampling_params: SamplingParams) -> usize {
        let encoding = self.tokenizer.encode(prompt, true).expect("tokenizer encode failed");
        self.add_request(encoding.get_ids().to_vec(), sampling_params)
    }

    pub fn add_request(&mut self, prompt: Vec<u32>, sampling_params: SamplingParams) -> usize {
        let seq = Sequence::with_block_size(prompt, sampling_params, self.block_size);
        let seq_id = seq.seq_id;
        self.scheduler.add(seq);
        seq_id
    }

    pub fn decode(&self, token_ids: &[u32]) -> String {
        self.tokenizer.decode(token_ids, true).expect("tokenizer decode failed")
    }

    pub fn step(&mut self) -> (Vec<(usize, Vec<u32>)>, usize) {
        let mut seqs = self.scheduler.schedule();
        for worker in &self.workers {
            worker.run(seqs.clone());
        }
        let (token_ids, seq_need_compute_logits) = self.model_runner.run(&mut seqs);
        self.scheduler.postprocess(&mut seqs, &token_ids, &seq_need_compute_logits);

        let outputs: Vec<(usize, Vec<u32>)> = seqs
            .iter()
            .filter(|seq| seq.is_finished())
            .map(|seq| (seq.seq_id, seq.completion_token_ids().to_vec()))
            .collect();

        let num_total_tokens: usize = seqs.iter().filter(|seq| seq.is_finished()).map(|seq| seq.len()).sum();

        (outputs, num_total_tokens)
    }

    pub fn is_finished(&self) -> bool {
        self.scheduler.is_finished()
    }

    // Python accepts a single `SamplingParams` broadcast to every prompt, or a
    // list matching `prompts` in length — Rust has no such union, so the caller
    // is expected to already supply one `SamplingParams` per prompt.
    pub fn generate(
        &mut self,
        prompts: Vec<String>,
        sampling_params: Vec<SamplingParams>,
        use_tqdm: bool,
    ) -> Vec<GeneratedOutput> {
        assert_eq!(prompts.len(), sampling_params.len());

        let pbar = if use_tqdm {
            let pb = ProgressBar::new(prompts.len() as u64);
            pb.set_style(
                ProgressStyle::with_template("Generating {bar} {pos}/{len} {msg}")
                    .unwrap(),
            );
            Some(pb)
        } else {
            None
        };

        for (prompt, sp) in prompts.into_iter().zip(sampling_params.into_iter()) {
            self.add_request_text(&prompt, sp);
        }

        let mut outputs: HashMap<usize, Vec<u32>> = HashMap::new();
        let mut num_total_tokens = 0usize;
        let start = Instant::now();

        while !self.is_finished() {
            let (output, num_step_tokens) = self.step();
            num_total_tokens += num_step_tokens;

            if let Some(pb) = &pbar {
                let elapsed = start.elapsed().as_secs_f64();
                let total_throughput = num_total_tokens as f64 / elapsed;
                pb.set_message(format!("{}tok/s", total_throughput as u64));
            }

            for (seq_id, token_ids) in output {
                outputs.insert(seq_id, token_ids);
                if let Some(pb) = &pbar {
                    pb.inc(1);
                }
            }
        }

        let mut seq_ids: Vec<usize> = outputs.keys().copied().collect();
        seq_ids.sort();

        let results = seq_ids
            .into_iter()
            .map(|seq_id| {
                let token_ids = outputs.remove(&seq_id).unwrap();
                let text = self
                    .tokenizer
                    .decode(&token_ids, true)
                    .expect("tokenizer decode failed");
                GeneratedOutput { text, token_ids }
            })
            .collect();

        if let Some(pb) = pbar {
            pb.finish();
        }

        results
    }
}

impl Drop for LLMEngine {
    // Equivalent of Python's `atexit.register(self.exit)` — Rust ties cleanup to
    // value lifetime via `Drop` rather than a separate exit-hook registration.
    fn drop(&mut self) {
        self.model_runner.exit();
    }
}

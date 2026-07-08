use std::collections::VecDeque;

use crate::config::{Config, EngineConfig};
use crate::engine::block_manager::BlockManager;
use crate::engine::sequence::{Sequence, SequenceStatus};

pub struct Scheduler {
    pub enable_chunked: bool,
    pub max_model_len: usize,
    pub max_num_seqs: usize,
    pub max_num_batched_tokens: usize,
    pub eos: u32,
    pub block_manager: BlockManager,
    pub waiting: VecDeque<Sequence>,
    pub running: VecDeque<Sequence>,
}

impl Scheduler {
    pub fn new(config: &Config, engine_config: &EngineConfig) -> Self {
        Self {
            enable_chunked: engine_config.chunked_prefill,
            max_model_len: config.max_model_len,
            max_num_seqs: engine_config.max_num_seqs,
            max_num_batched_tokens: engine_config.max_num_batched_tokens,
            eos: config.eos_token_id,
            block_manager: BlockManager::new(
                engine_config.num_kvcache_blocks,
                engine_config.kvcache_block_size,
            ),
            waiting: VecDeque::new(),
            running: VecDeque::new(),
        }
    }

    pub fn is_finished(&self) -> bool {
        self.waiting.is_empty() && self.running.is_empty()
    }

    pub fn add(&mut self, seq: Sequence) {
        assert!(
            seq.len() <= self.max_model_len - 1,
            "Sequence length exceeds max_model_len"
        );
        self.waiting.push_back(seq);
    }

    pub fn preempt(&mut self, mut seq: Sequence) {
        seq.status = SequenceStatus::Waiting;
        self.block_manager.deallocate(&mut seq);
        self.waiting.push_front(seq);
    }

    pub fn postprocess(
        &mut self,
        seqs: &mut [Sequence],
        token_ids: &[u32],
        seq_need_compute_logits: &[usize],
    ) {
        assert_eq!(token_ids.len(), seq_need_compute_logits.len());

        for (&seq_index, &token_id) in seq_need_compute_logits.iter().zip(token_ids.iter()) {
            let seq = &mut seqs[seq_index];
            seq.append_token(token_id);

            let finished = (!seq.ignore_eos && token_id == self.eos)
                || seq.num_completion_tokens() == seq.max_tokens as usize
                || seq.len() >= self.max_model_len;

            if finished {
                if seq.len() >= self.max_model_len {
                    println!(
                        "Sequence {} reached max_model_len {}.",
                        seq.seq_id, self.max_model_len
                    );
                }
                seq.status = SequenceStatus::Finished;
                let seq_id = seq.seq_id;
                self.block_manager.deallocate(seq);

                if let Some(pos) = self.running.iter().position(|s| s.seq_id == seq_id) {
                    self.running.remove(pos);
                }
            }
        }

        for seq in seqs.iter() {
            if seq.status == SequenceStatus::Finished {
                continue;
            }
            if let Some(running_seq) = self.running.iter_mut().find(|s| s.seq_id == seq.seq_id) {
                *running_seq = seq.clone();
                running_seq.num_cached_tokens += running_seq.num_new_tokens;
                running_seq.num_new_tokens = 0;
            }
        }
    }

    pub fn schedule(&mut self) -> Vec<Sequence> {
        let mut scheduled_running_seqs: Vec<Sequence> = Vec::new();
        let mut scheduled_new_reqs: Vec<Sequence> = Vec::new();
        let mut any_preempted = false;
        let mut token_budget = self.max_num_batched_tokens;

        let mut req_index = 0;
        while req_index < self.running.len() && token_budget > 0 {
            let mut seq: Sequence = self.running[req_index].clone();

            let mut num_new_tokens = seq.len() - seq.num_cached_tokens;
            if self.enable_chunked {
                num_new_tokens = num_new_tokens.min(token_budget);
            }
            num_new_tokens = num_new_tokens.min(self.max_model_len - 1 - seq.num_cached_tokens);
            assert!(num_new_tokens > 0);

            loop {
                if self.block_manager.can_append(&seq, num_new_tokens) {
                    seq.num_new_tokens = num_new_tokens;
                    self.block_manager.may_append(&mut seq);
                    break;
                }

                let preempted_seq = self
                    .running
                    .pop_back()
                    .expect("running queue unexpectedly empty during preemption");
                self.preempt(preempted_seq);
                any_preempted = true;

                if self.running.len() == req_index {
                    break;
                }
            }

            if self.running.len() == req_index {
                break;
            }

            token_budget -= seq.num_new_tokens;
            self.running[req_index] = seq.clone();
            scheduled_running_seqs.push(seq);
            req_index += 1;
        }

        // schedule from the waiting queue
        if !any_preempted {
            while !self.waiting.is_empty()
                && token_budget > 0
                && self.running.len() < self.max_num_seqs
            {
                let mut seq = self.waiting.front().unwrap().clone();
                assert!(seq.block_table.is_empty());

                let (num_new_computed_tokens_in_used, num_new_computed_tokens_in_free, mut num_new_tokens) =
                    self.block_manager.get_token_layout(&seq);

                if self.enable_chunked {
                    num_new_tokens = num_new_tokens.min(token_budget);
                }
                assert!(num_new_tokens > 0);

                if num_new_tokens > token_budget
                    || !self
                        .block_manager
                        .can_allocate(num_new_computed_tokens_in_free + num_new_tokens)
                {
                    break;
                }

                seq.num_new_tokens = num_new_tokens;
                self.block_manager.allocate(&mut seq);
                assert_eq!(
                    seq.num_cached_tokens,
                    num_new_computed_tokens_in_free + num_new_computed_tokens_in_used
                );
                token_budget -= num_new_tokens;
                seq.status = SequenceStatus::Running;

                self.waiting.pop_front();
                self.running.push_back(seq.clone());
                scheduled_new_reqs.push(seq);
            }
        }

        let mut scheduled_seqs = scheduled_running_seqs;
        scheduled_seqs.extend(scheduled_new_reqs);
        assert!(!scheduled_seqs.is_empty());
        scheduled_seqs
    }
}

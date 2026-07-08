use crate::sampling_params::SamplingParams;
use std::sync::atomic::{AtomicUsize, Ordering};

pub const BLOCK_SIZE: usize = 256;

// Equivalent of Python's `counter = count()`: a process-wide source of unique
// sequence ids, advanced once per `Sequence::new` call (mirrors `next(Sequence.counter)`).
static COUNTER: AtomicUsize = AtomicUsize::new(0);

fn next_seq_id() -> usize {
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStatus {
    Waiting,
    Running,
    Finished,
}

#[derive(Clone)]
pub struct Sequence{
    pub block_size: usize,
    pub seq_id: usize,
    pub status: SequenceStatus,
    pub token_ids: Vec<u32>,
    pub last_token: u32,
    pub num_tokens: usize,
    pub num_prompt_tokens: usize,
    pub num_cached_tokens: usize,
    pub num_new_tokens: usize,
    pub block_table: Vec<usize>,
    pub temperature: f64,
    pub max_tokens: i32,
    pub ignore_eos: bool,
}

impl Sequence {
    pub fn new(token_ids: Vec<u32>, sampling_params: SamplingParams) -> Self {
        Self::with_block_size(token_ids, sampling_params, BLOCK_SIZE)
    }

    pub fn with_block_size(token_ids: Vec<u32>, sampling_params: SamplingParams, block_size: usize) -> Self {
        let last_token = *token_ids.last().expect("token_ids must be non-empty");
        let num_tokens = token_ids.len();
        let num_prompt_tokens = token_ids.len();

        Self {
            block_size,
            seq_id: next_seq_id(),
            status: SequenceStatus::Waiting,
            token_ids,
            last_token,
            num_tokens,
            num_prompt_tokens,
            num_cached_tokens: 0,
            num_new_tokens: 0,
            block_table: Vec::new(),
            temperature: sampling_params.temperature,
            max_tokens: sampling_params.max_tokens,
            ignore_eos: sampling_params.ignore_eos,
        }
    }

    // __len__
    pub fn len(&self) -> usize {
        self.num_tokens
    }

    pub fn is_empty(&self) -> bool {
        self.num_tokens == 0
    }

    pub fn is_finished(&self) -> bool {
        self.status == SequenceStatus::Finished
    }

    pub fn num_completion_tokens(&self) -> usize {
        self.num_tokens - self.num_prompt_tokens
    }

    pub fn num_context_tokens(&self) -> usize {
        self.num_cached_tokens + self.num_new_tokens
    }

    pub fn prompt_token_ids(&self) -> &[u32] {
        &self.token_ids[..self.num_prompt_tokens]
    }

    pub fn completion_token_ids(&self) -> &[u32] {
        &self.token_ids[self.num_prompt_tokens..]
    }

    // __getitem__ with a slice, e.g. Python's `seq[start:end]`
    pub fn slice_tokens(&self, start: usize, end: usize) -> &[u32] {
        &self.token_ids[start..end]
    }

    pub fn num_cached_blocks(&self) -> usize {
        self.num_cached_tokens / self.block_size
    }

    // Number of blocks actually allocated in `block_table` right now, with a
    // consistency check against how many blocks *should* be needed to hold
    // `num_cached_tokens + num_new_tokens` (the tokens currently in context).
    //
    // Example with block_size = 4: 6 cached tokens + 2 new tokens = 8 in-context
    // tokens -> (6 + 2 + 4 - 1) / 4 = 11 / 4 = 2 blocks expected, so this asserts
    // `block_table.len() == 2`. If block_table had drifted (e.g. a block was
    // allocated/freed without updating the token counts), this would fire.
    pub fn num_current_blocks(&self) -> usize {
        assert_eq!(
            (self.num_cached_tokens + self.num_new_tokens + self.block_size - 1) / self.block_size,
            self.block_table.len()
        );
        self.block_table.len()
    }

    // Number of blocks needed to hold the *whole* sequence (prompt + completions
    // so far), regardless of how many are actually allocated yet.
    //
    // Example with block_size = 4: 10 total tokens -> (10 + 4 - 1) / 4 = 13 / 4 = 3
    // blocks (4 + 4 + 2, with the last block only half full).
    pub fn num_blocks(&self) -> usize {
        (self.num_tokens + self.block_size - 1) / self.block_size
    }

    pub fn last_block_num_tokens(&self) -> usize {
        self.num_tokens - (self.num_blocks() - 1) * self.block_size
    }

    pub fn block(&self, i: usize) -> &[u32] {
        assert!(i < self.num_blocks());
        let start = i * self.block_size;
        let end = ((i + 1) * self.block_size).min(self.token_ids.len());
        &self.token_ids[start..end]
    }

    pub fn append_token(&mut self, token_id: u32) {
        self.token_ids.push(token_id);
        self.last_token = token_id;
        self.num_tokens += 1;
        assert_eq!(self.num_tokens, self.token_ids.len());
    }

    // worker process needs to reconstruct generation state — deliberately
    // excludes block_size, seq_id, status, max_tokens, ignore_eos.
    pub fn get_state(&self) -> SequenceState {
        SequenceState {
            token_ids: self.token_ids.clone(),
            last_token: self.last_token,
            num_tokens: self.num_tokens,
            num_prompt_tokens: self.num_prompt_tokens,
            num_cached_tokens: self.num_cached_tokens,
            num_new_tokens: self.num_new_tokens,
            block_table: self.block_table.clone(),
            temperature: self.temperature,
        }
    }

    // __setstate__
    pub fn set_state(&mut self, state: SequenceState) {
        self.token_ids = state.token_ids;
        self.last_token = state.last_token;
        self.num_tokens = state.num_tokens;
        self.num_prompt_tokens = state.num_prompt_tokens;
        self.num_cached_tokens = state.num_cached_tokens;
        self.num_new_tokens = state.num_new_tokens;
        self.block_table = state.block_table;
        self.temperature = state.temperature;
    }
}

pub struct SequenceState {
    pub token_ids: Vec<u32>,
    pub last_token: u32,
    pub num_tokens: usize,
    pub num_prompt_tokens: usize,
    pub num_cached_tokens: usize,
    pub num_new_tokens: usize,
    pub block_table: Vec<usize>,
    pub temperature: f64,
}

// __getitem__
impl std::ops::Index<usize> for Sequence {
    type Output = u32;

    fn index(&self, key: usize) -> &u32 {
        &self.token_ids[key]
    }
}

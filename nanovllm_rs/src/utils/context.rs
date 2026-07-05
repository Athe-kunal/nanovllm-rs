use candle_core::Tensor;
use std::sync::{Mutex, OnceLock};

#[derive(Clone, Default)]
pub struct Context{
    pub cu_seqlens_q: Option<Tensor>,
    pub cu_seqlens_k: Option<Tensor>,
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
    pub slot_mapping: Option<Tensor>,
    pub context_lens: Option<Tensor>,
    pub block_tables: Option<Tensor>,
    pub seq_need_compute_logits: Option<Tensor>
}

fn context_lock() -> &'static Mutex<Context> {
    static CONTEXT: OnceLock<Mutex<Context>> = OnceLock::new();
    CONTEXT.get_or_init(|| Mutex::new(Context::default()))
}

pub fn get_context() -> Context {
    context_lock().lock().unwrap().clone()
}

#[allow(clippy::too_many_arguments)]
pub fn set_context(
    cu_seqlens_q: Option<Tensor>,
    cu_seqlens_k: Option<Tensor>,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    slot_mapping: Option<Tensor>,
    context_lens: Option<Tensor>,
    block_tables: Option<Tensor>,
    seq_need_compute_logits: Option<Tensor>,
) {
    let mut ctx = context_lock().lock().unwrap();
    *ctx = Context {
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        context_lens,
        block_tables,
        seq_need_compute_logits,
    };
}

pub fn reset_context() {
    let mut ctx = context_lock().lock().unwrap();
    *ctx = Context::default();
}

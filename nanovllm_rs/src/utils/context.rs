use candle_core::Tensor;
use std::cell::RefCell;

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

thread_local! {
    // Under tensor parallelism each rank runs its own OS thread with its own CUDA device;
    // a process-global context would let rank threads race and hand each other tensors
    // from the wrong GPU (see the "device mismatch" bug this replaced).
    static CONTEXT: RefCell<Context> = RefCell::new(Context::default());
}

pub fn get_context() -> Context {
    CONTEXT.with(|ctx| ctx.borrow().clone())
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
    CONTEXT.with(|ctx| {
        *ctx.borrow_mut() = Context {
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            context_lens,
            block_tables,
            seq_need_compute_logits,
        };
    });
}

pub fn reset_context() {
    CONTEXT.with(|ctx| *ctx.borrow_mut() = Context::default());
}

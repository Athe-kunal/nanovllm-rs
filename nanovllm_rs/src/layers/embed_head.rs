use candle_core::{Device, Result, Tensor, DType};
use candle_nn::{Embedding, Module};
use cudarc::nccl::safe::Comm;
use std::rc::Rc;
use crate::{layers::dist_util, utils::context::get_context};

pub struct VocabParallelEmbedding {
    num_embeddings: usize,
    embedding_dim: usize,
    tp_rank: usize,
    tp_size: usize,
    num_embeddings_per_partition: usize,
    vocab_start_idx: usize,
    vocab_end_idx: usize,
    weight: Tensor,
    comm: Option<Rc<Comm>>,
}

impl VocabParallelEmbedding{
    pub fn new(num_embeddings: usize, embedding_dim: usize, tp_rank: usize, tp_size: usize, comm: Option<Rc<Comm>>, device: &Device) -> Result<Self>{
        assert_eq!(num_embeddings % tp_size,0, "num_embeddings should be divisible by tp_size");
        assert!(tp_size == 1 || comm.is_some(), "comm is required when tp_size > 1");

        let num_embeddings_per_partition = num_embeddings / tp_size;
        let weight = Tensor::empty((num_embeddings_per_partition,embedding_dim), DType::F32, device)?;
        let vocab_start_idx = num_embeddings_per_partition * tp_rank;
        let vocab_end_idx = vocab_start_idx + num_embeddings_per_partition;

        Ok(Self {
            num_embeddings,
            embedding_dim,
            tp_rank,
            tp_size,
            num_embeddings_per_partition,
            vocab_start_idx,
            vocab_end_idx,
            weight,
            comm,
        })
    }

    pub fn weight_loader(&mut self, loaded_weight: Tensor) -> Result<()> {
        self.weight = loaded_weight.narrow(0, self.vocab_start_idx, self.num_embeddings_per_partition)?;
        Ok(())
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor>{
        if self.tp_size == 1 {
            return Embedding::new(self.weight.clone(),self.embedding_dim).forward(x);
        }

        let start = Tensor::new(self.vocab_start_idx as i64, x.device())?;
        let mask = x.ge(self.vocab_start_idx as i64)?.mul(&x.lt(self.vocab_end_idx as i64)?)?;
        let x_local = x.broadcast_sub(&start)?.mul(&mask.to_dtype(x.dtype())?)?;

        let y = Embedding::new(self.weight.clone(), self.embedding_dim).forward(&x_local)?;
        let y = y.broadcast_mul(&mask.to_dtype(y.dtype())?.unsqueeze(candle_core::D::Minus1)?)?;
        let y = dist_util::all_reduce_sum(&y, self.comm.as_ref().unwrap())?;
        Ok(y)
    }
}

pub struct ParallelLMHead {
    base: VocabParallelEmbedding,
    bias: Option<Tensor>,
}

impl ParallelLMHead {
    pub fn new(num_embeddings: usize, embedding_dim: usize, bias: bool, tp_rank: usize, tp_size: usize, comm: Option<Rc<Comm>>, device: &Device) -> Result<Self> {
        assert!(!bias, "bias is not supported");

        let base = VocabParallelEmbedding::new(num_embeddings, embedding_dim, tp_rank, tp_size, comm, device)?;

        Ok(Self {
            base,
            bias: None,
        })
    }

    /// Equivalent of `self.lm_head.weight.data = self.model.embed_tokens.weight.data`.
    pub fn tie_weights(&mut self, embed_tokens: &VocabParallelEmbedding) {
        self.base.weight = embed_tokens.weight.clone();
    }
    pub fn forward(&self, x: &Tensor, cu_seqlens_q: &Tensor, seq_need_compute_logits: &[u32]) -> Result<Tensor> {
        let len = cu_seqlens_q.dim(0)?;
        let sliced: Vec<u32> = cu_seqlens_q.narrow(0, 1, len-1)?.to_vec1()?;
        let mut last_indices: Vec<u32> = sliced.iter().map(|v| v - 1).collect();
        
        if !seq_need_compute_logits.is_empty() {
            last_indices = seq_need_compute_logits.iter().map(|&i| last_indices[i as usize]).collect();

        }
        
        let idx = Tensor::new(last_indices.as_slice(), x.device())?;
        let x = x.index_select(&idx, 0)?.contiguous()?;
        let mut logits = x.matmul(&self.base.weight.t()?)?; // F.linear(x, weight) == x @ weight.T

        if self.base.tp_size > 1 {
            logits = dist_util::gather_concat(&logits, self.base.comm.as_ref().unwrap())?; // gather to rank 0, cat along vocab dim
        }
        Ok(logits)
    }
}



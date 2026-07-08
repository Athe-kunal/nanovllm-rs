use candle_core::{Device, Result, Tensor, DType};
use crate::layers::nccl::Comm;
use candle_nn::{Linear, Module};
use crate::layers::dist_util;
use std::sync::Arc;

fn divide(numerator: usize, denominator: usize) -> usize {
    assert_eq!(numerator % denominator, 0, "numerator must be divisible by denominator");
    numerator / denominator
}

pub struct LinearBase{
    input_size: usize,
    output_size: usize,
    tp_dim: Option<usize>,
    weight: Tensor,
    bias: Option<Tensor>,
    tp_rank: usize,
    tp_size: usize,
    comm: Option<Arc<Comm>>,
}

impl LinearBase{
    pub fn new(input_size: usize, output_size: usize, bias: bool, tp_dim: Option<usize>, comm: Option<Arc<Comm>>, device: &Device) -> Result<Self>{
        let (tp_rank, tp_size) = match &comm {
            Some(comm) => (comm.rank(), comm.world_size()),
            None => (0, 1),
        };

        let weight = Tensor::zeros((output_size, input_size), DType::F32, device)?;
        let bias = if bias {
            Some(Tensor::zeros(output_size, DType::F32, device)?)
        } else {
            None
        };

        Ok(Self {
            input_size,
            output_size,
            tp_dim,
            weight,
            bias,
            tp_rank,
            tp_size,
            comm,
        })
    }

    pub fn weight_loader(&mut self, loaded_weight: Tensor) -> Result<()> {
        self.weight = loaded_weight;
        Ok(())
    }

    pub fn forward(&self, _x: &Tensor) -> Result<Tensor> {
        candle_core::bail!("LinearBase::forward is not implemented; use a concrete linear layer (e.g. RowParallelLinear)")
    }
}

pub struct ReplicatedLinear{
    base: LinearBase,
}

impl ReplicatedLinear{
    pub fn new(input_size: usize, output_size: usize, bias: bool, comm: Option<Arc<Comm>>, device: &Device) -> Result<Self> {
        let base = LinearBase::new(input_size, output_size, bias, None, comm, device)?;
        Ok(Self { base })
    }

    pub fn weight_loader(&mut self, loaded_weight: Tensor) -> Result<()> {
        self.base.weight_loader(loaded_weight)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Linear::new(self.base.weight.clone(), self.base.bias.clone()).forward(x)
    }
}

pub struct ColumnParallelLinear{
    base: LinearBase
}

impl ColumnParallelLinear{
    pub fn new(input_size: usize, output_size: usize, bias: bool, comm: Option<Arc<Comm>>, device: &Device) -> Result<Self> {
        let (tp_rank, tp_size) = match &comm {
            Some(comm) => (comm.rank(), comm.world_size()),
            None => (0, 1),
        };
        // Shards output_size, tp_dim=0 (weight shape is (output_size, input_size)): each rank
        // gets a slice of *output/hidden features*, computed for every token — the sequence/
        // batch dims are never split, only the model's hidden dimension is.
        let output_size = divide(output_size, tp_size);
        let base = LinearBase::new(input_size, output_size, bias, Some(0), comm, device)?;
        Ok(Self { base })
    }

    pub fn weight_loader(&mut self, loaded_weight: Tensor) -> Result<()> {
        // narrow along tp_dim=0 (rows of the weight = output features).
        let tp_dim = self.base.tp_dim.expect("ColumnParallelLinear requires tp_dim");
        let shard_size = self.base.weight.dim(tp_dim)?;
        let start_idx = self.base.tp_rank * shard_size;
        self.base.weight = loaded_weight.narrow(tp_dim, start_idx, shard_size)?;
        Ok(())
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Linear::new(self.base.weight.clone(), self.base.bias.clone()).forward(x)
    }
}

pub struct MergedColumnParallelLinear{
    base: ColumnParallelLinear,
    output_sizes: Vec<usize>,
}

impl MergedColumnParallelLinear {
    pub fn new(input_size: usize, output_sizes: Vec<usize>, bias: bool, comm: Option<Arc<Comm>>, device: &Device) -> Result<Self>{
        let output_size = output_sizes.iter().sum();
        let base = ColumnParallelLinear::new(input_size, output_size, bias, comm, device)?;
        Ok(Self { base, output_sizes })
    }

    pub fn weight_loader(&mut self, loaded_weight: Tensor, loaded_shard_id: usize) -> Result<()> {
        let tp_dim = self.base.base.tp_dim.expect("MergedColumnParallelLinear requires tp_dim");
        let tp_rank = self.base.base.tp_rank;
        let tp_size = self.base.base.tp_size;

        // Where this sub-weight (e.g. gate_proj=0, up_proj=1) sits within *this rank's*
        // local merged weight: sum of the full sizes of all earlier sub-weights, converted
        // to the sharded layout by dividing by tp_size (each sub-weight is sharded the same way).
        let shard_offset = self.output_sizes[..loaded_shard_id].iter().sum::<usize>() / tp_size;
        // This rank's slice of the current sub-weight.
        let shard_size = self.output_sizes[loaded_shard_id] / tp_size;

        // The checkpoint gives the full (unsharded) sub-weight; keep only this rank's chunk.
        let loaded_weight = loaded_weight.narrow(tp_dim, tp_rank * shard_size, shard_size)?;

        // candle tensors are immutable, so "write into the middle" means: slice out the
        // untouched region before the target, the untouched region after it, and reassemble
        // before ++ new_shard ++ after around the new value.
        let weight = &self.base.base.weight;
        let total = weight.dim(tp_dim)?;
        let before = weight.narrow(tp_dim, 0, shard_offset)?;
        let after = weight.narrow(tp_dim, shard_offset + shard_size, total - shard_offset - shard_size)?;
        self.base.base.weight = Tensor::cat(&[&before, &loaded_weight, &after], tp_dim)?;
        Ok(())
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.base.forward(x)
    }
}

pub struct RowParallelLinear{
    base: LinearBase,
}

impl RowParallelLinear{
    pub fn new(input_size: usize, output_size: usize, bias: bool, comm: Option<Arc<Comm>>, device: &Device) -> Result<Self> {
        let tp_size = match &comm {
            Some(comm) => comm.world_size(),
            None => 1,
        };
        // Unlike ColumnParallelLinear (shards output_size, tp_dim=0), RowParallelLinear shards
        // input_size and passes tp_dim=1 — each rank only holds a slice of the *input* columns
        // of the weight matrix, since each rank only ever sees a slice of the input activations
        // (the output of the previous ColumnParallelLinear layer). Like ColumnParallelLinear,
        // this is a hidden-dimension shard, not a sequence/batch shard — every rank still sees
        // every token, just a fraction of that token's feature vector.
        let base = LinearBase::new(divide(input_size, tp_size), output_size, bias, Some(1), comm, device)?;
        Ok(Self { base })
    }

    pub fn weight_loader(&mut self, loaded_weight: Tensor) -> Result<()> {
        // Same shard-and-replace logic as ColumnParallelLinear::weight_loader, just along
        // tp_dim=1 (columns of the weight = input features) instead of tp_dim=0 (rows).
        let tp_dim = self.base.tp_dim.expect("RowParallelLinear requires tp_dim");
        let shard_size = self.base.weight.dim(tp_dim)?;
        let start_idx = self.base.tp_rank * shard_size;
        self.base.weight = loaded_weight.narrow(tp_dim, start_idx, shard_size)?;
        Ok(())
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Every rank computes x_shard @ weight_shard.T, which is only a partial sum toward
        // the true output (each rank only saw a slice of the input dimension). Adding the
        // bias on every rank would double/triple/... count it once summed across ranks, so
        // only rank 0 adds it — the all_reduce below then sums everyone's partial output,
        // and rank 0's contribution already includes the bias exactly once.
        let bias = if self.base.tp_rank == 0 {
            self.base.bias.clone()
        } else {
            None
        };
        let y = Linear::new(self.base.weight.clone(), bias).forward(x)?;

        if self.base.tp_size > 1 {
            // Sum the partial outputs across all ranks to get the true full-input-dim result.
            let comm = self
                .base
                .comm
                .as_ref()
                .expect("RowParallelLinear requires comm when tp_size > 1");
            dist_util::all_reduce_sum(&y, comm)
        } else {
            Ok(y)
        }
    }
}

pub struct QKVParallelLinear{
    base: ColumnParallelLinear,
    head_size: usize,
    // Per-rank (already divided by tp_size) head counts, needed to split the fused
    // QKV output back into its Q/K/V pieces in forward: q_size = num_heads * head_size,
    // kv_size = num_kv_heads * head_size.
    num_heads: usize,
    num_kv_heads: usize,
}

impl QKVParallelLinear{
    pub fn new(
        hidden_size: usize,
        head_size: usize,
        total_num_heads: usize,
        total_num_kv_heads: Option<usize>,
        bias: bool,
        comm: Option<Arc<Comm>>,
        device: &Device,
    ) -> Result<Self> {
        let tp_size = match &comm {
            Some(comm) => comm.world_size(),
            None => 1,
        };
        let total_num_kv_heads = total_num_kv_heads.unwrap_or(total_num_heads);
        let num_heads = divide(total_num_heads, tp_size);
        let num_kv_heads = divide(total_num_kv_heads, tp_size);
        // Fused Q+K+V output width: Q gets total_num_heads worth of features, K and V
        // each get total_num_kv_heads worth (see the earlier explanation of the `2 *`).
        let output_size = (total_num_heads + 2 * total_num_kv_heads) * head_size;
        let base = ColumnParallelLinear::new(hidden_size, output_size, bias, comm, device)?;
        Ok(Self { base, head_size, num_heads, num_kv_heads })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.base.forward(x)
    }

    pub fn weight_loader(&mut self, loaded_weight: Tensor, loaded_shard_id: &str) -> Result<()> {
        // Q/K/V each land at a fixed offset within the fused local weight, in Q-then-K-then-V
        // order — mirrors how MergedColumnParallelLinear lays out its sub-weights back to back.
        let (shard_offset, shard_size) = match loaded_shard_id {
            "q" => (0, self.num_heads * self.head_size),
            "k" => (self.num_heads * self.head_size, self.num_kv_heads * self.head_size),
            "v" => (
                self.num_heads * self.head_size + self.num_kv_heads * self.head_size,
                self.num_kv_heads * self.head_size,
            ),
            other => candle_core::bail!("loaded_shard_id must be one of q, k, v, got {other}"),
        };

        let tp_dim = self.base.base.tp_dim.expect("QKVParallelLinear requires tp_dim");
        let tp_rank = self.base.base.tp_rank;

        // The checkpoint gives the full (unsharded) q/k/v weight; keep only this rank's chunk.
        let loaded_weight = loaded_weight.narrow(tp_dim, tp_rank * shard_size, shard_size)?;

        // Splice it into [shard_offset, shard_offset + shard_size) of the local fused weight,
        // same slice-before/slice-after/cat trick used in MergedColumnParallelLinear, since
        // candle tensors can't be mutated in place.
        let weight = &self.base.base.weight;
        let total = weight.dim(tp_dim)?;
        let before = weight.narrow(tp_dim, 0, shard_offset)?;
        let after = weight.narrow(tp_dim, shard_offset + shard_size, total - shard_offset - shard_size)?;
        self.base.base.weight = Tensor::cat(&[&before, &loaded_weight, &after], tp_dim)?;
        Ok(())
    }
}
use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp1, DType, Layout,Result,Shape, Tensor};
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::WrapErr;
use cudarc::nccl::safe::{Comm, ReduceOp};
#[cfg(feature = "cuda")]
use half::bf16;
use std::rc::Rc;

pub struct AllReduce{
    pub comm: Rc<Comm>
}

unsafe impl Sync for AllReduce {}
unsafe impl Send for AllReduce {}

impl CustomOp1 for AllReduce {
    fn name(&self) -> &'static str {
        "allreduce"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("AllReduce is never used on cpu")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, s: &candle_core::CudaStorage, l: &Layout) -> Result<(candle_core::CudaStorage, Shape)> {
        let elem_count = l.shape().elem_count();
        let dev = s.device().clone();
        let dst = match s.dtype() {
            DType::BF16 => {
                let s = s.as_cuda_slice::<bf16>()?;
                let s = match l.contiguous_offsets() {
                    Some((0, l)) if l == s.len() => s,
                    Some(_) | None => candle_core::bail!("input has to be contiguous"),
                };
                let mut dst = unsafe { dev.alloc::<bf16>(elem_count) }.w()?;
                self.comm.all_reduce(s, &mut dst, &ReduceOp::Sum).map_err(candle_core::Error::debug)?;
                candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
            }
            dtype => candle_core::bail!("unsupported dtype {dtype:?}"),
        };
        Ok((dst, l.shape().clone()))
    }
}

pub fn all_reduce_sum(x: &Tensor, comm: &Rc<Comm>) -> Result<Tensor> {
    x.apply_op1_no_bwd(&AllReduce { comm: comm.clone() })
}

pub struct AllGather{
    pub comm: Rc<Comm>
}

unsafe impl Sync for AllGather {}
unsafe impl Send for AllGather {}

impl CustomOp1 for AllGather {
    fn name(&self) -> &'static str {
        "allgather"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("AllGather is never used on cpu")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, s: &candle_core::CudaStorage, l: &Layout) -> Result<(candle_core::CudaStorage, Shape)> {
        let world_size = self.comm.world_size();
        let elem_count = l.shape().elem_count();
        let dev = s.device().clone();
        let dst = match s.dtype() {
            DType::BF16 => {
                let s = s.as_cuda_slice::<bf16>()?;
                let s = match l.contiguous_offsets() {
                    Some((0, l)) if l == s.len() => s,
                    Some(_) | None => candle_core::bail!("input has to be contiguous"),
                };
                let mut dst = unsafe { dev.alloc::<bf16>(elem_count * world_size) }.w()?;
                self.comm.all_gather(s, &mut dst).map_err(candle_core::Error::debug)?;
                candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
            }
            dtype => candle_core::bail!("unsupported dtype {dtype:?}"),
        };
        let mut out_dims = l.shape().dims().to_vec();
        out_dims[0] *= world_size;
        Ok((dst, out_dims.into()))
    }
}

pub fn all_gather(x: &Tensor, comm: &Rc<Comm>) -> Result<Tensor> {
    x.apply_op1_no_bwd(&AllGather { comm: comm.clone() })
}

/// Gathers `x` (sharded along the last dim across ranks) and concatenates the shards
/// back into the full last dim, on every rank (NCCL has no root-only gather primitive).
pub fn gather_concat(x: &Tensor, comm: &Rc<Comm>) -> Result<Tensor> {
    let world_size = comm.world_size();
    let dims = x.dims().to_vec();

    let mut gathered_dims = vec![world_size];
    gathered_dims.extend_from_slice(&dims);
    let gathered = all_gather(x, comm)?.reshape(gathered_dims)?;

    let gathered = gathered.transpose(0, 1)?.contiguous()?;

    let mut out_dims = dims;
    let last = out_dims.len() - 1;
    out_dims[last] *= world_size;
    gathered.reshape(out_dims)
}
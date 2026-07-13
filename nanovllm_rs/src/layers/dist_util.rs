use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp1, DType, Layout, Result, Shape, Tensor};
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
use crate::layers::nccl::Comm;
#[cfg(feature = "cuda")]
use cudarc::nccl::sys::ncclDataType_t;
#[cfg(feature = "cuda")]
use half::bf16;
use std::sync::Arc;

#[cfg(feature = "cuda")]
fn nccl_dtype(dtype: DType) -> Result<ncclDataType_t> {
    match dtype {
        DType::BF16 => Ok(ncclDataType_t::ncclBfloat16),
        DType::F32 => Ok(ncclDataType_t::ncclFloat32),
        dtype => candle_core::bail!("unsupported dtype {dtype:?}"),
    }
}

#[cfg(feature = "cuda")]
fn cuda_stream(dev: &candle_core::CudaDevice) -> cudarc::nccl::sys::cudaStream_t {
    // CUstream (candle's vendored cudarc) and cudaStream_t (this crate's cudarc) are
    // both opaque `*mut CUstream_st` under the hood — same CUDA ABI, different Rust
    // crate instances, so a raw-pointer cast bridges them.
    dev.cuda_stream().cu_stream() as *mut std::ffi::c_void as cudarc::nccl::sys::cudaStream_t
}

pub struct AllReduce {
    pub comm: Arc<Comm>,
}

unsafe impl Sync for AllReduce {}
unsafe impl Send for AllReduce {}

impl CustomOp1 for AllReduce {
    fn name(&self) -> &'static str {
        "allreduce"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("AllReduce is never used on cpu")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, s: &candle_core::CudaStorage, l: &Layout) -> Result<(candle_core::CudaStorage, Shape)> {
        let elem_count = l.shape().elem_count();
        let dev = s.device().clone();
        let candle_stream = dev.cuda_stream();
        // Unlike every other candle CUDA op, this raw NCCL call bypasses candle's own launch
        // path (which always rebinds the context before a kernel), so without this the thread's
        // current CUDA context can be left pointing at a different rank's device.
        candle_stream.context().bind_to_thread().map_err(candle_core::Error::wrap)?;
        let dtype = nccl_dtype(s.dtype())?;
        let stream = cuda_stream(&dev);

        macro_rules! reduce {
            ($t:ty) => {{
                let src = s.as_cuda_slice::<$t>()?;
                let src = match l.contiguous_offsets() {
                    Some((0, len)) if len == elem_count => src,
                    Some(_) | None => candle_core::bail!("input has to be contiguous"),
                };
                let mut dst = unsafe { dev.alloc::<$t>(elem_count) }?;
                let (send_ptr, send_guard) = src.device_ptr(&candle_stream);
                let (recv_ptr, recv_guard) = dst.device_ptr_mut(&candle_stream);
                unsafe { self.comm.all_reduce_sum_raw(send_ptr, recv_ptr, elem_count, dtype, stream) }?;
                drop(send_guard);
                drop(recv_guard);
                candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
            }};
        }
        let dst = match s.dtype() {
            DType::BF16 => reduce!(bf16),
            DType::F32 => reduce!(f32),
            dtype => candle_core::bail!("unsupported dtype {dtype:?}"),
        };
        Ok((dst, l.shape().clone()))
    }
}

pub fn all_reduce_sum(x: &Tensor, comm: &Arc<Comm>) -> Result<Tensor> {
    x.apply_op1_no_bwd(&AllReduce { comm: comm.clone() })
}

pub struct AllGather {
    pub comm: Arc<Comm>,
}

unsafe impl Sync for AllGather {}
unsafe impl Send for AllGather {}

impl CustomOp1 for AllGather {
    fn name(&self) -> &'static str {
        "allgather"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("AllGather is never used on cpu")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, s: &candle_core::CudaStorage, l: &Layout) -> Result<(candle_core::CudaStorage, Shape)> {
        let world_size = self.comm.world_size();
        let elem_count = l.shape().elem_count();
        let dev = s.device().clone();
        let candle_stream = dev.cuda_stream();
        candle_stream.context().bind_to_thread().map_err(candle_core::Error::wrap)?;
        let dtype = nccl_dtype(s.dtype())?;
        let stream = cuda_stream(&dev);

        macro_rules! gather {
            ($t:ty) => {{
                let src = s.as_cuda_slice::<$t>()?;
                let src = match l.contiguous_offsets() {
                    Some((0, len)) if len == elem_count => src,
                    Some(_) | None => candle_core::bail!("input has to be contiguous"),
                };
                let mut dst = unsafe { dev.alloc::<$t>(elem_count * world_size) }?;
                let (send_ptr, send_guard) = src.device_ptr(&candle_stream);
                let (recv_ptr, recv_guard) = dst.device_ptr_mut(&candle_stream);
                unsafe { self.comm.all_gather_raw(send_ptr, recv_ptr, elem_count, dtype, stream) }?;
                drop(send_guard);
                drop(recv_guard);
                candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
            }};
        }
        let dst = match s.dtype() {
            DType::BF16 => gather!(bf16),
            DType::F32 => gather!(f32),
            dtype => candle_core::bail!("unsupported dtype {dtype:?}"),
        };
        let mut out_dims = l.shape().dims().to_vec();
        out_dims[0] *= world_size;
        Ok((dst, out_dims.into()))
    }
}

pub fn all_gather(x: &Tensor, comm: &Arc<Comm>) -> Result<Tensor> {
    x.apply_op1_no_bwd(&AllGather { comm: comm.clone() })
}

/// Gathers `x` (sharded along the last dim across ranks) and concatenates the shards
/// back into the full last dim, on every rank (NCCL has no root-only gather primitive).
pub fn gather_concat(x: &Tensor, comm: &Arc<Comm>) -> Result<Tensor> {
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

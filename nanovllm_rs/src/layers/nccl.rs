//! Minimal NCCL bindings, used instead of `cudarc::nccl::safe::Comm`.
//!
//! candle-core vendors its own cudarc 0.13 internally; this crate depends on cudarc
//! 0.19 directly for NCCL. A candle `CudaSlice` doesn't implement the 0.19
//! `DevicePtr` trait, so `cudarc::nccl::safe::Comm` can't be called on candle
//! tensors. NCCL's C API only takes raw device pointers and a raw CUDA stream
//! handle (ABI-stable across cudarc versions), so this drives NCCL through
//! `cudarc::nccl::{result, sys}` directly instead.
use cudarc::nccl::{result, sys};

#[derive(Debug)]
pub struct Comm {
    comm: sys::ncclComm_t,
    rank: usize,
    world_size: usize,
}

// `ncclComm_t` is an opaque handle into the NCCL library, safe to move across
// threads (each rank's Comm is only ever used from the one thread that owns it).
unsafe impl Send for Comm {}
unsafe impl Sync for Comm {}

impl Comm {
    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Call once on rank 0 and share the result with every other rank.
    pub fn new_id() -> candle_core::Result<sys::ncclUniqueId> {
        result::get_uniqueid().map_err(|e| candle_core::Error::Msg(format!("nccl get_uniqueid failed: {e:?}")))
    }

    /// Every rank in `0..world_size` must call this with the same `id`; blocks until all ranks join.
    pub fn init_rank(rank: usize, world_size: usize, id: sys::ncclUniqueId) -> candle_core::Result<Self> {
        let mut comm = std::mem::MaybeUninit::uninit();
        let comm = unsafe {
            result::comm_init_rank(comm.as_mut_ptr(), world_size as i32, id, rank as i32)
                .map_err(|e| candle_core::Error::Msg(format!("nccl comm_init_rank failed: {e:?}")))?;
            comm.assume_init()
        };
        Ok(Self { comm, rank, world_size })
    }

    /// # Safety
    /// `sendbuff`/`recvbuff` must be valid device pointers holding `count` elements
    /// of `dtype`, on the device `stream` belongs to.
    pub unsafe fn all_reduce_sum_raw(
        &self,
        sendbuff: u64,
        recvbuff: u64,
        count: usize,
        dtype: sys::ncclDataType_t,
        stream: sys::cudaStream_t,
    ) -> candle_core::Result<()> {
        result::all_reduce(
            sendbuff as *const std::ffi::c_void,
            recvbuff as *mut std::ffi::c_void,
            count,
            dtype,
            sys::ncclRedOp_t::ncclSum,
            self.comm,
            stream,
        )
        .map(|_| ())
        .map_err(|e| candle_core::Error::Msg(format!("nccl all_reduce failed: {e:?}")))
    }

    /// # Safety
    /// `sendbuff` must hold `sendcount` elements of `dtype`; `recvbuff` must hold
    /// `sendcount * world_size`. Both on the device `stream` belongs to.
    pub unsafe fn all_gather_raw(
        &self,
        sendbuff: u64,
        recvbuff: u64,
        sendcount: usize,
        dtype: sys::ncclDataType_t,
        stream: sys::cudaStream_t,
    ) -> candle_core::Result<()> {
        result::all_gather(
            sendbuff as *const std::ffi::c_void,
            recvbuff as *mut std::ffi::c_void,
            sendcount,
            dtype,
            self.comm,
            stream,
        )
        .map(|_| ())
        .map_err(|e| candle_core::Error::Msg(format!("nccl all_gather failed: {e:?}")))
    }
}

impl Drop for Comm {
    fn drop(&mut self) {
        unsafe {
            let _ = result::comm_destroy(self.comm);
        }
    }
}

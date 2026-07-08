//! Zero-copy export of a candle CUDA `Tensor` to a DLPack `PyCapsule` that torch consumes
//! via `torch.utils.dlpack.from_dlpack`. This lets the attention bridge hand q/k/v (and a
//! pre-allocated output buffer) straight to flash-attn with no device<->host round trip and
//! no dtype detour — the torch tensor points at the exact same GPU memory candle owns.
//!
//! Ownership: the capsule keeps a clone of the candle `Tensor` alive (candle tensors are
//! Arc-backed, so this just bumps a refcount on the shared GPU buffer). The DLPack `deleter`
//! drops that clone. GPU memory is therefore valid for as long as *either* Rust or torch
//! holds the tensor, and is freed exactly once when both are done. The capsule follows the
//! standard "dltensor" -> "used_dltensor" rename protocol so we only free on the
//! not-consumed path (torch calls the deleter itself once it takes ownership).

use std::ffi::{c_char, c_void, CStr, CString};
use std::os::raw::c_int;
use std::sync::OnceLock;

use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{DType, Device, Result, Storage, Tensor};
use pyo3::ffi;
use pyo3::prelude::*;

// DLPack device type codes (from dlpack.h).
const K_DL_CUDA: c_int = 2;
// DLPack dtype codes.
const K_DL_INT: u8 = 0;
const K_DL_UINT: u8 = 1;
const K_DL_FLOAT: u8 = 2;
const K_DL_BFLOAT: u8 = 4;

#[repr(C)]
#[derive(Clone, Copy)]
struct DLDevice {
    device_type: c_int,
    device_id: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DLDataType {
    code: u8,
    bits: u8,
    lanes: u16,
}

#[repr(C)]
struct DLTensor {
    data: *mut c_void,
    device: DLDevice,
    ndim: c_int,
    dtype: DLDataType,
    shape: *mut i64,
    strides: *mut i64,
    byte_offset: u64,
}

#[repr(C)]
struct DLManagedTensor {
    dl_tensor: DLTensor,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

/// Owns the metadata the DLTensor's `shape`/`strides` pointers reference, plus a clone of the
/// source candle tensor so its GPU buffer outlives every consumer of the capsule.
struct ManagerCtx {
    _tensor: Tensor,
    shape: Vec<i64>,
    strides: Vec<i64>,
}

unsafe extern "C" fn managed_deleter(mt: *mut DLManagedTensor) {
    if mt.is_null() {
        return;
    }
    let mt = Box::from_raw(mt);
    if !mt.manager_ctx.is_null() {
        drop(Box::from_raw(mt.manager_ctx as *mut ManagerCtx));
    }
}

/// Runs only when the capsule is garbage-collected *without* having been consumed by torch.
/// A consumer renames the capsule to "used_dltensor" and takes responsibility for the
/// deleter, so if we still see "dltensor" here we must free it ourselves.
unsafe extern "C" fn capsule_destructor(capsule: *mut ffi::PyObject) {
    let name = ffi::PyCapsule_GetName(capsule);
    if name.is_null() {
        return;
    }
    if CStr::from_ptr(name).to_bytes() != b"dltensor" {
        return; // consumed; torch owns the deleter now
    }
    let mt = ffi::PyCapsule_GetPointer(capsule, name) as *mut DLManagedTensor;
    if mt.is_null() {
        return;
    }
    if let Some(deleter) = (*mt).deleter {
        deleter(mt);
    }
}

fn dltensor_name() -> *const c_char {
    static NAME: OnceLock<CString> = OnceLock::new();
    NAME.get_or_init(|| CString::new("dltensor").unwrap()).as_ptr()
}

fn dl_dtype(dtype: DType) -> Result<DLDataType> {
    let (code, bits) = match dtype {
        DType::BF16 => (K_DL_BFLOAT, 16),
        DType::F16 => (K_DL_FLOAT, 16),
        DType::F32 => (K_DL_FLOAT, 32),
        DType::F64 => (K_DL_FLOAT, 64),
        DType::U8 => (K_DL_UINT, 8),
        DType::U32 => (K_DL_UINT, 32),
        DType::I64 => (K_DL_INT, 64),
    };
    Ok(DLDataType { code, bits, lanes: 1 })
}

/// Reads the CUDA device pointer (to the tensor's first element) and device ordinal out of a
/// contiguous candle tensor. A nonzero `start_offset` is folded into the pointer, so views
/// into a shared buffer export correctly; non-contiguous (strided) tensors are rejected since
/// callers make their inputs contiguous before export (matching the host path's flatten).
fn device_ptr_and_id(t: &Tensor) -> Result<(u64, usize)> {
    let (storage, layout) = t.storage_and_layout();
    if !layout.is_contiguous() {
        candle_core::bail!("dlpack export requires a contiguous tensor");
    }
    let byte_offset = (layout.start_offset() * t.dtype().size_in_bytes()) as u64;
    match &*storage {
        Storage::Cuda(cs) => {
            let id = cs.device.cuda_device().ordinal();
            let base = match t.dtype() {
                DType::BF16 => *cs.as_cuda_slice::<half::bf16>()?.device_ptr(),
                DType::F16 => *cs.as_cuda_slice::<half::f16>()?.device_ptr(),
                DType::F32 => *cs.as_cuda_slice::<f32>()?.device_ptr(),
                DType::F64 => *cs.as_cuda_slice::<f64>()?.device_ptr(),
                DType::U8 => *cs.as_cuda_slice::<u8>()?.device_ptr(),
                DType::U32 => *cs.as_cuda_slice::<u32>()?.device_ptr(),
                DType::I64 => *cs.as_cuda_slice::<i64>()?.device_ptr(),
            };
            Ok((base + byte_offset, id))
        }
        _ => candle_core::bail!("dlpack export requires a CUDA tensor"),
    }
}

/// Builds a DLPack `PyCapsule` (name "dltensor") aliasing `t`'s GPU buffer, ready to pass to
/// `torch.utils.dlpack.from_dlpack`. Zero-copy: no data is moved.
pub fn to_dlpack(py: Python<'_>, t: &Tensor) -> Result<Py<PyAny>> {
    let (device_ptr, device_id) = device_ptr_and_id(t)?;
    let dtype = dl_dtype(t.dtype())?;

    let shape: Vec<i64> = t.dims().iter().map(|&d| d as i64).collect();
    let ndim = shape.len();
    // Compact row-major strides, in elements (DLPack convention).
    let mut strides = vec![1i64; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    let mut ctx = Box::new(ManagerCtx { _tensor: t.clone(), shape, strides });
    let shape_ptr = ctx.shape.as_mut_ptr();
    let strides_ptr = ctx.strides.as_mut_ptr();
    let ctx_raw = Box::into_raw(ctx);

    let mt = Box::new(DLManagedTensor {
        dl_tensor: DLTensor {
            data: device_ptr as *mut c_void,
            device: DLDevice { device_type: K_DL_CUDA, device_id: device_id as c_int },
            ndim: ndim as c_int,
            dtype,
            shape: shape_ptr,
            strides: strides_ptr,
            byte_offset: 0,
        },
        manager_ctx: ctx_raw as *mut c_void,
        deleter: Some(managed_deleter),
    });
    let mt_raw = Box::into_raw(mt);

    unsafe {
        let capsule = ffi::PyCapsule_New(
            mt_raw as *mut c_void,
            dltensor_name(),
            Some(capsule_destructor),
        );
        if capsule.is_null() {
            // Capsule creation failed: reclaim the managed tensor (and its ctx) ourselves.
            managed_deleter(mt_raw);
            return Err(candle_core::Error::Msg("PyCapsule_New failed".into()));
        }
        Ok(Bound::from_owned_ptr(py, capsule).unbind())
    }
}

/// Raw handle of the CUDA stream candle enqueues work on for `device`, as an integer for
/// `torch.cuda.ExternalStream`. Running the torch kernels on this same stream is what makes
/// the zero-copy handoff safe: candle's producing ops, flash-attn, and the output `copy_`
/// all serialize on one stream, so there are no cross-stream read/write hazards and no host
/// sync is needed to order them.
pub fn stream_ptr(device: &Device) -> Result<usize> {
    match device {
        Device::Cuda(cd) => Ok(*cd.cuda_device().cu_stream() as usize),
        _ => candle_core::bail!("stream_ptr requires a CUDA device"),
    }
}

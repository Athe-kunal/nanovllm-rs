use candle_core::{DType, Device, Result, Tensor};
use numpy::{PyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::sync::OnceLock;

fn torch_dtype_str(dtype: DType) -> &'static str {
    match dtype {
        DType::U8 => "uint8",
        DType::U32 => "int32",
        DType::I64 => "int64",
        DType::BF16 => "bfloat16",
        DType::F16 => "float16",
        DType::F32 => "float32",
        DType::F64 => "float64",
    }
}

pub fn kernels_module(py: Python<'_>) -> Result<Bound<'_, PyModule>> {
    static MODULE: OnceLock<Py<PyModule>> = OnceLock::new();
    if let Some(m) = MODULE.get() {
        return Ok(m.bind(py).clone());
    }
    let m = PyModule::import(py, "nanovllm_kernels").map_err(candle_core::Error::wrap)?;
    let _ = MODULE.set(m.clone().unbind());
    Ok(m)
}

/// Resolves `nanovllm_kernels.<name>` once and reuses the handle instead of a fresh
/// `getattr` on every call.
fn cached_fn(py: Python<'_>, cell: &'static OnceLock<Py<PyAny>>, name: &str) -> Result<Py<PyAny>> {
    if let Some(f) = cell.get() {
        return Ok(f.clone_ref(py));
    }
    let f = kernels_module(py)?.getattr(name).map_err(candle_core::Error::wrap)?.unbind();
    let _ = cell.set(f.clone_ref(py));
    Ok(f)
}

pub fn to_cuda_tensor_fn(py: Python<'_>) -> Result<Py<PyAny>> {
    static CELL: OnceLock<Py<PyAny>> = OnceLock::new();
    cached_fn(py, &CELL, "to_cuda_tensor")
}

pub fn to_cuda_int32_tensor_fn(py: Python<'_>) -> Result<Py<PyAny>> {
    static CELL: OnceLock<Py<PyAny>> = OnceLock::new();
    cached_fn(py, &CELL, "to_cuda_int32_tensor")
}

pub fn to_host_array_fn(py: Python<'_>) -> Result<Py<PyAny>> {
    static CELL: OnceLock<Py<PyAny>> = OnceLock::new();
    cached_fn(py, &CELL, "to_host_array")
}

pub fn store_kvcache_fn(py: Python<'_>) -> Result<Py<PyAny>> {
    static CELL: OnceLock<Py<PyAny>> = OnceLock::new();
    cached_fn(py, &CELL, "store_kvcache")
}

pub fn flash_attn_varlen_fn(py: Python<'_>) -> Result<Py<PyAny>> {
    static CELL: OnceLock<Py<PyAny>> = OnceLock::new();
    cached_fn(py, &CELL, "flash_attn_varlen")
}

#[cfg(feature = "cuda")]
pub fn flash_attn_varlen_dlpack_fn(py: Python<'_>) -> Result<Py<PyAny>> {
    static CELL: OnceLock<Py<PyAny>> = OnceLock::new();
    cached_fn(py, &CELL, "flash_attn_varlen_dlpack")
}

// The candle<->torch tensor bridge is deliberately split into a candle half (host copy,
// no GIL) and a Python half (GIL held). Under tensor parallelism a candle device->host or
// host->device copy syncs the candle CUDA stream, which may be blocked on a pending
// cross-rank NCCL collective (see layers/dist_util.rs). If the GIL is held across that
// sync, the peer rank cannot acquire the GIL to launch its half of the collective, the
// collective never completes, the stream never drains, and both ranks deadlock. Keeping the
// candle copies out of `Python::with_gil` is what makes TP > 1 able to make progress.

/// Copies a candle `Tensor` to host f32 data (with its dims and original dtype). Pure candle
/// device->host work — call this WITHOUT the GIL held (see the note above).
pub fn tensor_to_host(t: &Tensor) -> Result<(Vec<f32>, Vec<i64>, DType)> {
    let dtype = t.dtype();
    let dims: Vec<i64> = t.dims().iter().map(|&d| d as i64).collect();
    let data: Vec<f32> = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    Ok((data, dims, dtype))
}

/// Builds a CUDA `torch.Tensor` from host f32 data, preserving dtype and shape (GIL held).
pub fn host_to_torch(py: Python<'_>, data: Vec<f32>, dims: Vec<i64>, dtype: DType) -> Result<Py<PyAny>> {
    let arr = PyArray1::from_vec(py, data);
    to_cuda_tensor_fn(py)?
        .call1(py, (arr, dims, torch_dtype_str(dtype)))
        .map_err(candle_core::Error::wrap)
}

/// Copies a candle index `Tensor` (slot_mapping, cu_seqlens, block_tables, ...) to host i64
/// data. Pure candle device->host work — call this WITHOUT the GIL held.
pub fn index_tensor_to_host(t: &Tensor) -> Result<(Vec<i64>, Vec<i64>)> {
    let dims: Vec<i64> = t.dims().iter().map(|&d| d as i64).collect();
    let data: Vec<i64> = t.to_dtype(DType::I64)?.flatten_all()?.to_vec1()?;
    Ok((data, dims))
}

/// Builds a CUDA `torch.int32` tensor from host i64 index data (GIL held).
pub fn host_to_torch_int32(py: Python<'_>, data: Vec<i64>, dims: Vec<i64>) -> Result<Py<PyAny>> {
    let arr = PyArray1::from_vec(py, data);
    to_cuda_int32_tensor_fn(py)?.call1(py, (arr, dims)).map_err(candle_core::Error::wrap)
}

/// Allocates the paged KV cache directly on the CUDA device (one `(k_cache, v_cache)`
/// torch.Tensor pair per layer, each shaped `(num_blocks, block_size, num_kv_heads,
/// head_dim)`). Done in Python rather than candle + `tensor_to_torch` because the
/// cache is large, GPU-resident for the whole run, and only ever touched in place by
/// the Python-side `store_kvcache`/`flash_attn_varlen` kernels — routing it through
/// candle first would mean a pointless host round-trip for data that never needs to
/// exist in Rust-side memory at all.
#[allow(clippy::too_many_arguments)]
pub fn allocate_kv_cache(
    py: Python<'_>,
    num_layers: usize,
    num_blocks: usize,
    block_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
    dtype: DType,
) -> Result<Vec<(Py<PyAny>, Py<PyAny>)>> {
    let kernels = kernels_module(py)?;
    let result = kernels
        .getattr("allocate_kv_cache")
        .and_then(|f| {
            f.call1((
                num_layers,
                num_blocks,
                block_size,
                num_kv_heads,
                head_dim,
                torch_dtype_str(dtype),
            ))
        })
        .map_err(candle_core::Error::wrap)?;

    result
        .extract::<Vec<(Py<PyAny>, Py<PyAny>)>>()
        .map_err(candle_core::Error::wrap)
}

/// `nanovllm_kernels`'s CUDA calls (mem_get_info, allocate_kv_cache, .cuda(), ...) all
/// operate on torch's implicit current device, which is separate, per-OS-thread state
/// that candle's own `Device::cuda_if_available` doesn't set. Must be called once on
/// every tensor-parallel rank's thread before any other pybridge call on that thread.
pub fn set_cuda_device(py: Python<'_>, device_index: usize) -> Result<()> {
    PyModule::import(py, "torch")
        .and_then(|torch| torch.getattr("cuda"))
        .and_then(|cuda| cuda.call_method1("set_device", (device_index,)))
        .map(|_| ())
        .map_err(candle_core::Error::wrap)
}

pub fn cuda_mem_get_info(py: Python<'_>) -> Result<(u64, u64)> {
    kernels_module(py)?
        .getattr("mem_get_info")
        .and_then(|f| f.call0())
        .and_then(|r| r.extract::<(u64, u64)>())
        .map_err(candle_core::Error::wrap)
}

pub fn cuda_memory_stats_peak_current(py: Python<'_>) -> Result<(u64, u64)> {
    kernels_module(py)?
        .getattr("memory_stats_peak_current")
        .and_then(|f| f.call0())
        .and_then(|r| r.extract::<(u64, u64)>())
        .map_err(candle_core::Error::wrap)
}

pub fn cuda_empty_cache(py: Python<'_>) -> Result<()> {
    kernels_module(py)?
        .getattr("empty_cache")
        .and_then(|f| f.call0())
        .map(|_| ())
        .map_err(candle_core::Error::wrap)
}

pub fn cuda_reset_peak_memory_stats(py: Python<'_>) -> Result<()> {
    kernels_module(py)?
        .getattr("reset_peak_memory_stats")
        .and_then(|f| f.call0())
        .map(|_| ())
        .map_err(candle_core::Error::wrap)
}

pub fn cuda_synchronize(py: Python<'_>) -> Result<()> {
    kernels_module(py)?
        .getattr("synchronize")
        .and_then(|f| f.call0())
        .map(|_| ())
        .map_err(candle_core::Error::wrap)
}

/// Pulls a `torch.Tensor` to host f32 data plus its shape (GIL held). The device->host copy
/// here runs on torch's stream, which never carries NCCL collectives, so it is safe to hold
/// the GIL across it (unlike the candle-side copies above).
pub fn torch_to_host(py: Python<'_>, obj: &Bound<'_, PyAny>) -> Result<(Vec<f32>, Vec<usize>)> {
    let host_array = to_host_array_fn(py)?.call1(py, (obj,)).map_err(candle_core::Error::wrap)?;
    let host_array: PyReadonlyArrayDyn<f32> =
        host_array.bind(py).extract().map_err(candle_core::Error::wrap)?;

    let shape = host_array.shape().to_vec();
    let data = host_array
        .as_array()
        .as_standard_layout()
        .into_owned()
        .into_raw_vec();
    Ok((data, shape))
}

/// Builds a candle `Tensor` on `device` with dtype `dtype` from host f32 data. The
/// host->device copy is pure candle work — call this WITHOUT the GIL held.
pub fn host_to_tensor(data: Vec<f32>, shape: Vec<usize>, device: &Device, dtype: DType) -> Result<Tensor> {
    Tensor::from_vec(data, shape, device)?.to_dtype(dtype)
}

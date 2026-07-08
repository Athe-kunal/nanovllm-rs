use candle_core::{DType, Device, Result, Tensor};
use numpy::{PyArray1, PyReadonlyArrayDyn};
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

/// Converts a candle `Tensor` into a CUDA `torch.Tensor`, preserving its dtype and shape.
pub fn tensor_to_torch(py: Python<'_>, t: &Tensor) -> Result<Py<PyAny>> {
    let dtype = t.dtype();
    let dims: Vec<i64> = t.dims().iter().map(|&d| d as i64).collect();
    let data: Vec<f32> = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;

    let kernels = kernels_module(py)?;
    let arr = PyArray1::from_vec(py, data);
    kernels
        .getattr("to_cuda_tensor")
        .and_then(|f| f.call1((arr, dims, torch_dtype_str(dtype))))
        .map(Bound::unbind)
        .map_err(candle_core::Error::wrap)
}

/// Converts a candle index `Tensor` (slot_mapping, cu_seqlens, block_tables, ...) into a
/// CUDA `torch.int32` tensor.
pub fn index_tensor_to_torch(py: Python<'_>, t: &Tensor) -> Result<Py<PyAny>> {
    let dims: Vec<i64> = t.dims().iter().map(|&d| d as i64).collect();
    let data: Vec<i64> = t.to_dtype(DType::I64)?.flatten_all()?.to_vec1()?;

    let kernels = kernels_module(py)?;
    let arr = PyArray1::from_vec(py, data);
    kernels
        .getattr("to_cuda_int32_tensor")
        .and_then(|f| f.call1((arr, dims)))
        .map(Bound::unbind)
        .map_err(candle_core::Error::wrap)
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

/// Converts a `torch.Tensor` back into a candle `Tensor` on `device` with dtype `dtype`.
pub fn torch_to_tensor(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let kernels = kernels_module(py)?;
    let host_array = kernels
        .getattr("to_host_array")
        .and_then(|f| f.call1((obj,)))
        .map_err(candle_core::Error::wrap)?;
    let host_array: PyReadonlyArrayDyn<f32> =
        host_array.extract().map_err(candle_core::Error::wrap)?;

    let shape = host_array.shape().to_vec();
    let data = host_array
        .as_array()
        .as_standard_layout()
        .into_owned()
        .into_raw_vec();

    Tensor::from_vec(data, shape, device)?.to_dtype(dtype)
}

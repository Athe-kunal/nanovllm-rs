use candle_core::{Result, Tensor};
use pyo3::prelude::*;

use crate::utils::context::get_context;
use crate::utils::pybridge;

pub struct Attention {
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    num_kv_heads: usize,
    k_cache: Option<Py<PyAny>>,
    v_cache: Option<Py<PyAny>>,
}

impl Attention {
    pub fn new(num_heads: usize, head_dim: usize, scale: f32, num_kv_heads: usize) -> Self {
        Self {
            num_heads,
            head_dim,
            scale,
            num_kv_heads,
            k_cache: None,
            v_cache: None,
        }
    }

    pub fn set_kv_cache(&mut self, k_cache: Py<PyAny>, v_cache: Py<PyAny>) {
        self.k_cache = Some(k_cache);
        self.v_cache = Some(v_cache);
    }

    pub fn forward(&mut self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let context = get_context();

        Python::with_gil(|py| -> Result<Tensor> {
            let kernels = pybridge::kernels_module(py)?;

            let py_q = pybridge::tensor_to_torch(py, q)?;
            let py_k = pybridge::tensor_to_torch(py, k)?;
            let py_v = pybridge::tensor_to_torch(py, v)?;

            if let (Some(k_cache), Some(v_cache)) = (&self.k_cache, &self.v_cache) {
                let slot_mapping = context
                    .slot_mapping
                    .as_ref()
                    .expect("slot_mapping must be set when a kv cache is present");
                let py_slot_mapping = pybridge::index_tensor_to_torch(py, slot_mapping)?;
                kernels
                    .getattr("store_kvcache")
                    .and_then(|f| f.call1((&py_k, &py_v, k_cache, v_cache, py_slot_mapping)))
                    .map_err(candle_core::Error::wrap)?;
            }

            let (attn_k, attn_v): (&Py<PyAny>, &Py<PyAny>) = if context.block_tables.is_some() {
                (
                    self.k_cache.as_ref().expect("kv cache must be set for prefix caching"),
                    self.v_cache.as_ref().expect("kv cache must be set for prefix caching"),
                )
            } else {
                (&py_k, &py_v)
            };

            let cu_seqlens_q = context
                .cu_seqlens_q
                .as_ref()
                .expect("cu_seqlens_q must be set");
            let cu_seqlens_k = context
                .cu_seqlens_k
                .as_ref()
                .expect("cu_seqlens_k must be set");
            let py_cu_seqlens_q = pybridge::index_tensor_to_torch(py, cu_seqlens_q)?;
            let py_cu_seqlens_k = pybridge::index_tensor_to_torch(py, cu_seqlens_k)?;
            let py_block_table = context
                .block_tables
                .as_ref()
                .map(|t| pybridge::index_tensor_to_torch(py, t))
                .transpose()?;

            let py_out = kernels
                .getattr("flash_attn_varlen")
                .and_then(|f| {
                    f.call1((
                        &py_q,
                        attn_k,
                        attn_v,
                        py_cu_seqlens_q,
                        py_cu_seqlens_k,
                        context.max_seqlen_q,
                        context.max_seqlen_k,
                        self.scale,
                        true,
                        py_block_table,
                    ))
                })
                .map_err(candle_core::Error::wrap)?;

            pybridge::torch_to_tensor(py, &py_out, q.device(), q.dtype())
        })
    }
}

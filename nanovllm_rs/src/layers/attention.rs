use candle_core::{Result, Tensor};
use pyo3::prelude::*;

use crate::utils::context::get_context;
use crate::utils::pybridge;

/// Selects the zero-copy DLPack bridge. On by default at `tp_size == 1`: correct and
/// ~1.4-1.5x faster end-to-end than the host round-trip (bit-identical greedy output to it),
/// so there's no reason to pay the host path's cost there. `NANOVLLM_DLPACK=0`/`false` is
/// kept as an explicit kill-switch back to the host path, for debugging or A/B comparison.
///
/// Hard-gated off (regardless of the env var) whenever `tp_size > 1`: DLPack currently has a
/// non-deterministic cross-rank race under TP>1 (output occasionally diverges mid-decode) that
/// a per-call device sync does NOT fix — the host path's implicit per-layer syncing was
/// masking a deeper ordering assumption in the TP collectives. That's still unresolved, so a
/// TP>1 deployment must never be able to reach this path no matter how the env var is set.
#[cfg(feature = "cuda")]
fn dlpack_enabled(tp_size: usize) -> bool {
    if tp_size > 1 {
        return false;
    }
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("NANOVLLM_DLPACK")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(true)
    })
}

pub struct Attention {
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    num_kv_heads: usize,
    tp_size: usize,
    k_cache: Option<Py<PyAny>>,
    v_cache: Option<Py<PyAny>>,
}

impl Attention {
    pub fn new(num_heads: usize, head_dim: usize, scale: f32, num_kv_heads: usize, tp_size: usize) -> Self {
        Self {
            num_heads,
            head_dim,
            scale,
            num_kv_heads,
            tp_size,
            k_cache: None,
            v_cache: None,
        }
    }

    pub fn set_kv_cache(&mut self, k_cache: Py<PyAny>, v_cache: Py<PyAny>) {
        self.k_cache = Some(k_cache);
        self.v_cache = Some(v_cache);
    }

    pub fn forward(&mut self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if dlpack_enabled(self.tp_size) {
            return self.forward_dlpack(q, k, v);
        }
        self.forward_host(q, k, v)
    }

    /// Original bridge: marshals q/k/v through host memory (f32) to torch and back. Correct
    /// but does 4 device<->host round trips per layer; see forward_dlpack for the zero-copy
    /// path and pybridge.rs for the GIL/NCCL rationale behind the phase split.
    fn forward_host(&mut self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let context = get_context();

        // Phase 1 (no GIL): copy every candle input down to host. These device->host copies
        // sync the candle CUDA stream, which under tensor parallelism may still be draining a
        // cross-rank NCCL collective from the previous layer. Holding the GIL across that sync
        // deadlocks the peer rank (it needs the GIL to launch its half of the collective), so
        // this must happen before we acquire the GIL. See pybridge.rs for the full rationale.
        let (q_data, q_dims, q_dtype) = pybridge::tensor_to_host(q)?;
        let (k_data, k_dims, k_dtype) = pybridge::tensor_to_host(k)?;
        let (v_data, v_dims, v_dtype) = pybridge::tensor_to_host(v)?;

        // slot_mapping is only consumed when a kv cache is present (store_kvcache); during
        // warmup the cache is unset and slot_mapping is empty, so mirror the original laziness
        // and skip its (empty, driver-rejected) host copy in that case.
        let has_kv_cache = self.k_cache.is_some() && self.v_cache.is_some();
        let slot_mapping_host = if has_kv_cache {
            Some(pybridge::index_tensor_to_host(
                context
                    .slot_mapping
                    .as_ref()
                    .expect("slot_mapping must be set when a kv cache is present"),
            )?)
        } else {
            None
        };

        let cu_seqlens_q_host = pybridge::index_tensor_to_host(
            context.cu_seqlens_q.as_ref().expect("cu_seqlens_q must be set"),
        )?;
        let cu_seqlens_k_host = pybridge::index_tensor_to_host(
            context.cu_seqlens_k.as_ref().expect("cu_seqlens_k must be set"),
        )?;
        let block_table_host = context
            .block_tables
            .as_ref()
            .map(pybridge::index_tensor_to_host)
            .transpose()?;
        let has_block_tables = block_table_host.is_some();

        // Phase 2 (GIL): rebuild the torch tensors on-device, run the kvcache + attention
        // kernels, and pull the result back to host. Every copy here is on torch's stream,
        // which carries no NCCL work, so syncing under the GIL cannot deadlock.
        let (out_data, out_shape) = Python::with_gil(|py| -> Result<(Vec<f32>, Vec<usize>)> {
            let py_q = pybridge::host_to_torch(py, q_data, q_dims, q_dtype)?;
            let py_k = pybridge::host_to_torch(py, k_data, k_dims, k_dtype)?;
            let py_v = pybridge::host_to_torch(py, v_data, v_dims, v_dtype)?;

            if let (Some(k_cache), Some(v_cache), Some((sm_data, sm_dims))) =
                (&self.k_cache, &self.v_cache, slot_mapping_host)
            {
                let py_slot_mapping = pybridge::host_to_torch_int32(py, sm_data, sm_dims)?;
                pybridge::store_kvcache_fn(py)?
                    .call1(py, (&py_k, &py_v, k_cache, v_cache, py_slot_mapping))
                    .map_err(candle_core::Error::wrap)?;
            }

            let (attn_k, attn_v): (&Py<PyAny>, &Py<PyAny>) = if has_block_tables {
                (
                    self.k_cache.as_ref().expect("kv cache must be set for prefix caching"),
                    self.v_cache.as_ref().expect("kv cache must be set for prefix caching"),
                )
            } else {
                (&py_k, &py_v)
            };

            let (cq_data, cq_dims) = cu_seqlens_q_host;
            let (ck_data, ck_dims) = cu_seqlens_k_host;
            let py_cu_seqlens_q = pybridge::host_to_torch_int32(py, cq_data, cq_dims)?;
            let py_cu_seqlens_k = pybridge::host_to_torch_int32(py, ck_data, ck_dims)?;
            let py_block_table = block_table_host
                .map(|(bt_data, bt_dims)| pybridge::host_to_torch_int32(py, bt_data, bt_dims))
                .transpose()?;

            let py_out = pybridge::flash_attn_varlen_fn(py)?
                .call1(
                    py,
                    (
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
                    ),
                )
                .map_err(candle_core::Error::wrap)?;

            pybridge::torch_to_host(py, py_out.bind(py))
        })?;

        // Phase 3 (no GIL): rebuild the candle output tensor (host->device on candle's stream).
        pybridge::host_to_tensor(out_data, out_shape, q.device(), q.dtype())
    }

    /// Zero-copy bridge: q/k/v, the index tensors, and a pre-allocated `out` buffer are all
    /// candle-owned GPU memory handed to torch via DLPack — no host round trip, no f32 detour.
    /// The torch kernels run on candle's stream and write the result straight into `out`, so
    /// no host sync is needed to order or fetch the result. Same GIL discipline as the host
    /// path: `to_dlpack` only reads pointers/metadata (no candle-stream sync), and every
    /// device op stays on the one stream, so the GIL is never held across an NCCL-dependent
    /// sync. Enabled by NANOVLLM_DLPACK=1.
    #[cfg(feature = "cuda")]
    fn forward_dlpack(&mut self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        use crate::utils::dlpack;
        use pyo3::types::PyDict;

        // DEBUG: with NANOVLLM_DLPACK_DEBUG=1, compute the host-path attention for the SAME
        // inputs and diff it against the DLPack result. store_kvcache is idempotent (same K/V
        // to the same slots), so running host first then DLPack is a valid comparison. A large
        // diff localizes the TP>1 corruption to attention itself; ~0 means it's downstream.
        let debug_ref = if std::env::var("NANOVLLM_DLPACK_DEBUG").is_ok() {
            Some(self.forward_host(q, k, v)?)
        } else {
            None
        };

        let context = get_context();
        let device = q.device();
        let stream = dlpack::stream_ptr(device)?;
        // DLPack export needs contiguous buffers (the host path got this for free via
        // flatten_all). q/k here come off rotary/norm and may be non-contiguous; contiguous()
        // is a no-op clone when already packed, else a cheap on-device copy — never a host trip.
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        // flash-attn output matches q's [num_tokens, num_heads, head_dim]; candle owns it and
        // torch writes into it in place via `out.copy_`.
        let out = Tensor::zeros(q.shape(), q.dtype(), device)?;

        let has_kv_cache = self.k_cache.is_some() && self.v_cache.is_some();

        Python::with_gil(|py| -> Result<()> {
            let kwargs = PyDict::new(py);

            let set = |kwargs: &Bound<'_, PyDict>, key: &str, val: Py<PyAny>| -> Result<()> {
                kwargs.set_item(key, val).map_err(candle_core::Error::wrap)
            };
            set(&kwargs, "q", dlpack::to_dlpack(py, &q)?)?;
            set(&kwargs, "k", dlpack::to_dlpack(py, &k)?)?;
            set(&kwargs, "v", dlpack::to_dlpack(py, &v)?)?;
            set(&kwargs, "out", dlpack::to_dlpack(py, &out)?)?;
            set(
                &kwargs,
                "cu_seqlens_q",
                dlpack::to_dlpack(py, context.cu_seqlens_q.as_ref().expect("cu_seqlens_q must be set"))?,
            )?;
            set(
                &kwargs,
                "cu_seqlens_k",
                dlpack::to_dlpack(py, context.cu_seqlens_k.as_ref().expect("cu_seqlens_k must be set"))?,
            )?;

            if has_kv_cache {
                let slot_mapping = context
                    .slot_mapping
                    .as_ref()
                    .expect("slot_mapping must be set when a kv cache is present");
                set(&kwargs, "slot_mapping", dlpack::to_dlpack(py, slot_mapping)?)?;
                kwargs.set_item("k_cache", self.k_cache.as_ref()).map_err(candle_core::Error::wrap)?;
                kwargs.set_item("v_cache", self.v_cache.as_ref()).map_err(candle_core::Error::wrap)?;
            }
            if let Some(block_tables) = context.block_tables.as_ref() {
                set(&kwargs, "block_table", dlpack::to_dlpack(py, block_tables)?)?;
            }

            kwargs.set_item("max_seqlen_q", context.max_seqlen_q).map_err(candle_core::Error::wrap)?;
            kwargs.set_item("max_seqlen_k", context.max_seqlen_k).map_err(candle_core::Error::wrap)?;
            kwargs.set_item("softmax_scale", self.scale).map_err(candle_core::Error::wrap)?;
            kwargs.set_item("stream_ptr", stream).map_err(candle_core::Error::wrap)?;

            pybridge::flash_attn_varlen_dlpack_fn(py)?
                .call(py, (), Some(&kwargs))
                .map_err(candle_core::Error::wrap)?;
            Ok(())
        })?;

        if let Some(host_out) = debug_ref {
            let a = out.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let b = host_out.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let max_diff = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
            let block_tables = context.block_tables.is_some();
            eprintln!(
                "[dlpack-debug] dev={:?} tokens={} block_tables={} attn_max_diff={:.5}",
                device, a.len(), block_tables, max_diff
            );
        }

        Ok(out)
    }
}

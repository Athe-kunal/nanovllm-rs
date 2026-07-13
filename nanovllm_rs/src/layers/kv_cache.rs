#![cfg(feature = "cuda")]
use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CudaStorage, DType, InplaceOp3, Layout, Result, Tensor};
use cudarc::driver::{CudaFunction, CudaModule, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions, Ptx};
use std::cell::RefCell;
use std::sync::{Arc, OnceLock};

const KERNEL_SRC: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" __global__ void store_kvcache_bf16(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    const long long* __restrict__ slot_mapping,
    long long d
) {
    long long slot = slot_mapping[blockIdx.x];
    if (slot < 0) return;
    for (long long i = threadIdx.x; i < d; i += blockDim.x) {
        dst[slot * d + i] = src[blockIdx.x * d + i];
    }
}

extern "C" __global__ void store_kvcache_f16(
    const __half* __restrict__ src,
    __half* __restrict__ dst,
    const long long* __restrict__ slot_mapping,
    long long d
) {
    long long slot = slot_mapping[blockIdx.x];
    if (slot < 0) return;
    for (long long i = threadIdx.x; i < d; i += blockDim.x) {
        dst[slot * d + i] = src[blockIdx.x * d + i];
    }
}
"#;

fn ptx() -> &'static Ptx {
    static PTX: OnceLock<Ptx> = OnceLock::new();
    PTX.get_or_init(|| {
        // NVRTC (unlike nvcc) doesn't know its own toolkit's install location, so
        // cuda_fp16.h/cuda_bf16.h need an explicit include path.
        let cuda_root =
            std::env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".to_string());
        let opts = CompileOptions {
            include_paths: vec![
                format!("{cuda_root}/include"),
                format!("{cuda_root}/targets/x86_64-linux/include"),
            ],
            ..Default::default()
        };
        compile_ptx_with_opts(KERNEL_SRC, opts).expect("failed to compile store_kvcache kernel")
    })
}

thread_local! {
    static FUNCS: RefCell<Option<(CudaFunction, CudaFunction)>> = const { RefCell::new(None) };
}

fn with_functions<R>(
    dev: &candle_core::CudaDevice,
    f: impl FnOnce(&CudaFunction, &CudaFunction) -> R,
) -> Result<R> {
    FUNCS.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            let module: Arc<CudaModule> = dev
                .cuda_stream()
                .context()
                .load_module(ptx().clone())
                .map_err(candle_core::Error::wrap)?;
            let bf16_fn = module.load_function("store_kvcache_bf16").map_err(candle_core::Error::wrap)?;
            let f16_fn = module.load_function("store_kvcache_f16").map_err(candle_core::Error::wrap)?;
            *slot = Some((bf16_fn, f16_fn));
        }
        let (bf16_fn, f16_fn) = slot.as_ref().unwrap();
        Ok(f(bf16_fn, f16_fn))
    })
}

struct StoreKvCache;

impl InplaceOp3 for StoreKvCache {
    fn name(&self) -> &'static str {
        "store-kvcache"
    }

    fn cpu_fwd(
        &self,
        _: &mut CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<()> {
        candle_core::bail!("store_kvcache is cuda-only")
    }

    fn cuda_fwd(
        &self,
        cache: &mut CudaStorage,
        cache_l: &Layout,
        src: &CudaStorage,
        src_l: &Layout,
        slot_mapping: &CudaStorage,
        slot_l: &Layout,
    ) -> Result<()> {
        if !cache_l.is_contiguous() || !src_l.is_contiguous() || !slot_l.is_contiguous() {
            candle_core::bail!("store_kvcache requires contiguous tensors");
        }
        let num_tokens = src_l.dims()[0];
        let d: usize = src_l.dims()[1..].iter().product();
        // cache shape is (num_blocks, block_size, num_kv_heads, head_dim); the per-slot width
        // that must match src's per-token width (num_kv_heads * head_dim) excludes both of the
        // first two dims, not just num_blocks.
        if cache_l.dims()[2..].iter().product::<usize>() != d {
            candle_core::bail!(
                "store_kvcache: cache per-slot width {:?} does not match src width {d}",
                &cache_l.dims()[2..]
            );
        }
        if num_tokens == 0 {
            return Ok(());
        }

        let dev = cache.device().clone();
        let stream = dev.cuda_stream();
        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (d.min(1024) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let d_i64 = d as i64;

        with_functions(&dev, |bf16_fn, f16_fn| -> Result<()> {
            let slots = slot_mapping.as_cuda_slice::<i64>()?;
            let (slots_ptr, _slots_guard) = slots.device_ptr(&stream);

            macro_rules! launch {
                ($ty:ty, $func:expr) => {{
                    let src_slice = src.as_cuda_slice::<$ty>()?;
                    let cache_slice = cache.as_cuda_slice_mut::<$ty>()?;
                    let (src_ptr, _src_guard) = src_slice.device_ptr(&stream);
                    let (cache_ptr, _cache_guard) = cache_slice.device_ptr_mut(&stream);
                    let mut builder = stream.launch_builder($func);
                    builder.arg(&src_ptr).arg(&cache_ptr).arg(&slots_ptr).arg(&d_i64);
                    unsafe { builder.launch(cfg) }.map_err(candle_core::Error::wrap)?;
                }};
            }

            match cache.dtype() {
                DType::BF16 => launch!(half::bf16, bf16_fn),
                DType::F16 => launch!(half::f16, f16_fn),
                dtype => candle_core::bail!("store_kvcache: unsupported dtype {dtype:?}"),
            }
            Ok(())
        })?
    }
}

/// Writes `key`/`value` (shape `(num_tokens, num_kv_heads, head_dim)`) into `k_cache`/`v_cache`
/// (shape `(num_blocks, block_size, num_kv_heads, head_dim)`, flattened as `(num_blocks *
/// block_size, num_kv_heads * head_dim)`) at the row given by each token's `slot_mapping` entry.
/// Rows with a `-1` slot are skipped (matches the old Triton kernel's warmup/padding behavior).
pub fn store_kv_cache(
    key: &Tensor,
    value: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    // `.contiguous()` is a no-op (returns a shared-storage clone) whenever `is_contiguous()`
    // is true — and `Shape::is_contiguous` skips the stride check entirely for any dimension
    // of size 1. During decode, `num_tokens == 1`, so key/value are (1, kv_size) narrow-views
    // into the middle of the fused qkv_proj output buffer: a real offset slice that gets
    // misclassified as contiguous. `store_kvcache`'s custom kernel launch captures a raw
    // pointer into that aliased buffer and runs asynchronously, bypassing candle's normal
    // op-dependency tracking — once the last Rust-side reference to the shared qkv buffer
    // drops, the allocator can reuse that memory for a later tensor before the kernel has
    // actually read it, corrupting the cache write. `force_contiguous()` always copies,
    // closing that gap.
    let key = key.force_contiguous()?;
    let value = value.force_contiguous()?;
    k_cache.inplace_op3(&key, slot_mapping, &StoreKvCache)?;
    v_cache.inplace_op3(&value, slot_mapping, &StoreKvCache)?;
    Ok(())
}

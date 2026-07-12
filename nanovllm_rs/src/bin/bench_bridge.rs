//! Microbenchmark: candle<->torch tensor bridge, host round-trip vs DLPack zero-copy.
//!
//! Both paths take a candle CUDA tensor, hand it to torch, and get an (identical) candle
//! tensor back — exactly what layers/attention.rs does around flash-attn, minus the attention
//! compute. This isolates the *bridge* cost so we can measure what the host round-trip
//! actually costs and what DLPack recovers.
//!
//! Host path (current impl): candle --dtoh--> host f32 --htod--> torch --dtoh--> host f32
//!   --htod--> candle. 2 device<->host round trips per tensor, in f32, each forcing a sync.
//! DLPack path: wrap candle buffers as torch tensors (zero copy), copy_ on candle's stream.
//!   No host transfer, no dtype detour, no forced host sync.

use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use nanovllm_rs::utils::{dlpack, pybridge};
use pyo3::prelude::*;

fn host_roundtrip(dev: &Device, t: &Tensor) -> Tensor {
    let (data, dims, dt) = pybridge::tensor_to_host(t).unwrap();
    Python::with_gil(|py| {
        let py_t = pybridge::host_to_torch(py, data, dims, dt).unwrap();
        let (out_data, out_shape) = pybridge::torch_to_host(py, py_t.bind(py)).unwrap();
        pybridge::host_to_tensor(out_data, out_shape, dev, dt).unwrap()
    })
}

fn dlpack_roundtrip(py: Python<'_>, kernels: &Bound<'_, PyModule>, t: &Tensor, out: &Tensor, stream: usize) {
    let in_caps = dlpack::to_dlpack(py, t).unwrap();
    let out_caps = dlpack::to_dlpack(py, out).unwrap();
    kernels
        .getattr("dlpack_identity")
        .unwrap()
        .call1((in_caps, out_caps, stream))
        .unwrap();
}

fn main() {
    let iters: usize = std::env::var("ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(500);

    let dev = Device::new_cuda_with_stream(0).expect("cuda device");
    Python::with_gil(|py| pybridge::set_cuda_device(py, 0)).expect("set torch device");
    let stream = dlpack::stream_ptr(&dev).expect("stream ptr");

    // (num_tokens, hidden) shapes spanning decode (1 row) to a big prefill (2048 rows).
    // hidden = num_heads(16) * head_dim(128) = 2048, i.e. Qwen3-0.6B's attention q width.
    let shapes: &[(usize, usize)] = &[(1, 2048), (128, 2048), (512, 2048), (2048, 2048)];

    println!(
        "{:>14} | {:>10} | {:>13} | {:>13} | {:>8}",
        "shape", "elems", "host us/iter", "dlpack us/iter", "speedup"
    );
    println!("{}", "-".repeat(72));

    for &(rows, cols) in shapes {
        let t = Tensor::randn(0f32, 1f32, (rows, cols), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let out = Tensor::zeros((rows, cols), DType::BF16, &dev).unwrap();

        // Correctness: DLPack identity must reproduce the input exactly.
        Python::with_gil(|py| {
            let kernels = pybridge::kernels_module(py).unwrap();
            dlpack_roundtrip(py, &kernels, &t, &out, stream);
        });
        dev.synchronize().unwrap();
        let a = t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = out.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let max_diff = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        assert!(max_diff == 0.0, "dlpack identity mismatch: max_diff={max_diff}");

        // Warmup both paths (module import, allocator, kernel setup).
        for _ in 0..20 {
            let _ = host_roundtrip(&dev, &t);
            Python::with_gil(|py| {
                let kernels = pybridge::kernels_module(py).unwrap();
                dlpack_roundtrip(py, &kernels, &t, &out, stream);
            });
        }
        dev.synchronize().unwrap();

        // Host path.
        let start = Instant::now();
        for _ in 0..iters {
            let _ = host_roundtrip(&dev, &t);
        }
        dev.synchronize().unwrap();
        let host_us = start.elapsed().as_secs_f64() * 1e6 / iters as f64;

        // DLPack path.
        let start = Instant::now();
        Python::with_gil(|py| {
            let kernels = pybridge::kernels_module(py).unwrap();
            for _ in 0..iters {
                dlpack_roundtrip(py, &kernels, &t, &out, stream);
            }
        });
        dev.synchronize().unwrap();
        let dlpack_us = start.elapsed().as_secs_f64() * 1e6 / iters as f64;

        println!(
            "{:>14} | {:>10} | {:>13.2} | {:>13.2} | {:>7.1}x",
            format!("{rows}x{cols}"),
            rows * cols,
            host_us,
            dlpack_us,
            host_us / dlpack_us
        );
    }
}

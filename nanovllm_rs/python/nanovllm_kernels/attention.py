import torch
import triton
import triton.language as tl
from torch.utils.dlpack import from_dlpack

from flash_attn import flash_attn_varlen_func

_external_streams: dict = {}  # reuse one Stream object per (stream_ptr, device)


def _external_stream(stream_ptr: int, device) -> torch.cuda.Stream:
    key = (stream_ptr, device)
    s = _external_streams.get(key)
    if s is None:
        s = torch.cuda.ExternalStream(stream_ptr, device=device)
        _external_streams[key] = s
    return s


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


def store_kvcache_external_stream(key, value, k_cache, v_cache, slot_mapping):
    """Same effect as `store_kvcache`, but pure torch indexing instead of the Triton kernel.
    The Triton kernel does not respect the `torch.cuda.stream(ExternalStream(...))` context
    used in `flash_attn_varlen_dlpack` below — it silently writes wrong data (confirmed by
    comparing decode output against the host path), so it must not be used under an
    ExternalStream. Regular torch ops don't have this problem."""
    n, d = key.shape[0], key.shape[1] * key.shape[2]
    valid = slot_mapping >= 0
    if not torch.any(valid):
        return
    idx = slot_mapping[valid].long()
    k_cache.view(-1, d)[idx] = key.reshape(n, d)[valid]
    v_cache.view(-1, d)[idx] = value.reshape(n, d)[valid]


def flash_attn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    block_table: torch.Tensor = None,
) -> torch.Tensor:
    return flash_attn_varlen_func(
        q, k, v,
        max_seqlen_q=max_seqlen_q, cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k, cu_seqlens_k=cu_seqlens_k,
        softmax_scale=softmax_scale, causal=causal, block_table=block_table,
    )


def flash_attn_varlen_dlpack(
    q, k, v, out,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q: int, max_seqlen_k: int, softmax_scale: float,
    slot_mapping=None, block_table=None, k_cache=None, v_cache=None,
    stream_ptr: int = 0,
):
    """Zero-copy attention: q/k/v/out and the index tensors are candle-owned GPU buffers
    wrapped via DLPack (no device<->host copy), written back into candle's pre-allocated
    `out` buffer. Everything runs on candle's own stream via ExternalStream, so no sync is
    needed to order or fetch the result.

    Two bugs had to be fixed to get here: the Triton store_kvcache kernel silently ignores
    the `torch.cuda.stream(ExternalStream(...))` context (see store_kvcache_external_stream);
    and candle's device, when created via `Device::cuda_if_available`, runs on cudarc's
    NULL/legacy-default stream (raw pointer 0x0) rather than a real dedicated stream — wrapping
    that as an ExternalStream is not a well-behaved case for torch (intermittent corruption,
    inconsistent across process launches, that no amount of added synchronization fixed). The
    Rust side must construct its device with `Device::new_cuda_with_stream` for this bridge to
    be reliable."""
    q = from_dlpack(q)
    # Bind the ExternalStream to *this rank's* device explicitly. Under tensor parallelism
    # rank r runs on GPU r with its own candle stream; defaulting the stream's device (and
    # thus leaving torch's kernels on the wrong device's default stream) races against candle's
    # producing ops and NCCL collectives, silently corrupting rank>0 output.
    device = q.device
    stream = _external_stream(stream_ptr, device)
    with torch.cuda.device(device), torch.cuda.stream(stream):
        k = from_dlpack(k)
        v = from_dlpack(v)
        out = from_dlpack(out)
        # flash-attn wants int32 cu_seqlens / block_table; candle produces them as int64.
        cu_seqlens_q = from_dlpack(cu_seqlens_q).to(torch.int32)
        cu_seqlens_k = from_dlpack(cu_seqlens_k).to(torch.int32)

        if slot_mapping is not None and k_cache is not None and v_cache is not None:
            store_kvcache_external_stream(k, v, k_cache, v_cache, from_dlpack(slot_mapping).to(torch.int32))

        if block_table is not None:
            block_table = from_dlpack(block_table).to(torch.int32)
            attn_k, attn_v = k_cache, v_cache
        else:
            attn_k, attn_v = k, v

        result = flash_attn_varlen(
            q, attn_k, attn_v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, softmax_scale, True, block_table,
        )
        out.copy_(result)

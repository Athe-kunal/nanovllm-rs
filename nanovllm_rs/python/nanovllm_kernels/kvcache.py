import torch


def allocate_kv_cache(
    num_layers: int,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: str,
) -> list:
    torch_dtype = getattr(torch, dtype)
    shape = (num_blocks, block_size, num_kv_heads, head_dim)
    return [
        (
            torch.zeros(shape, dtype=torch_dtype, device="cuda"),
            torch.zeros(shape, dtype=torch_dtype, device="cuda"),
        )
        for _ in range(num_layers)
    ]

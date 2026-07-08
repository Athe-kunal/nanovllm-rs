import numpy as np
import torch
from torch.utils.dlpack import from_dlpack


def dlpack_identity(in_capsule, out_capsule, stream_ptr: int):
    """Zero-copy bridge microbenchmark counterpart to the host round-trip path: wrap two
    candle-owned buffers via DLPack and copy input->output entirely on the GPU, on candle's
    stream. No device<->host transfer, no dtype detour."""
    stream = torch.cuda.ExternalStream(stream_ptr)
    with torch.cuda.stream(stream):
        src = from_dlpack(in_capsule)
        dst = from_dlpack(out_capsule)
        dst.copy_(src)


def to_cuda_tensor(flat: np.ndarray, shape: list, dtype: str) -> torch.Tensor:
    t = torch.from_numpy(flat).reshape(shape)
    return t.cuda().to(getattr(torch, dtype))


def to_cuda_int32_tensor(flat: np.ndarray, shape: list) -> torch.Tensor:
    t = torch.from_numpy(flat).reshape(shape)
    return t.cuda().to(torch.int32)


def to_host_array(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).contiguous().cpu().numpy()

import numpy as np
import torch


def to_cuda_tensor(flat: np.ndarray, shape: list, dtype: str) -> torch.Tensor:
    t = torch.from_numpy(flat).reshape(shape)
    return t.cuda().to(getattr(torch, dtype))


def to_cuda_int32_tensor(flat: np.ndarray, shape: list) -> torch.Tensor:
    t = torch.from_numpy(flat).reshape(shape)
    return t.cuda().to(torch.int32)


def to_host_array(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).contiguous().cpu().numpy()

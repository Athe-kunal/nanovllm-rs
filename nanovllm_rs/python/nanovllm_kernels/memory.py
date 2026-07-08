import torch


def mem_get_info() -> tuple:
    free, total = torch.cuda.mem_get_info()
    return free, total


def memory_stats_peak_current() -> tuple:
    stats = torch.cuda.memory_stats()
    return stats["allocated_bytes.all.peak"], stats["allocated_bytes.all.current"]


def empty_cache():
    torch.cuda.empty_cache()


def reset_peak_memory_stats():
    torch.cuda.reset_peak_memory_stats()


def synchronize():
    torch.cuda.synchronize()

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistributedConfig:
    is_distributed: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device


def setup_distributed() -> DistributedConfig:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if is_distributed and not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    return DistributedConfig(
        is_distributed=is_distributed,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
    )


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


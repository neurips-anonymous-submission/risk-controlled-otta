"""
Dynamic memory bank for OTTA.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch


@dataclass
class MemoryEntry:
    image: torch.Tensor
    pseudo_heatmap: torch.Tensor


class DynamicMemoryBank:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: deque[MemoryEntry] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def initialize(self, samples: Iterable[dict]) -> None:
        for sample in samples:
            self.push(sample["image"], sample["pseudo_heatmap"])

    def push(self, image: torch.Tensor, pseudo_heatmap: torch.Tensor) -> None:
        self.buffer.append(
            MemoryEntry(
                image=image.detach().cpu().clone(),
                pseudo_heatmap=pseudo_heatmap.detach().cpu().clone(),
            )
        )

    def get(self, index: int) -> MemoryEntry:
        return self.buffer[index]

    def update_heatmap(self, index: int, new_heatmap: torch.Tensor) -> None:
        self.buffer[index].pseudo_heatmap = new_heatmap.detach().cpu().clone()


def heatmap_confidence_score(heatmaps: torch.Tensor) -> torch.Tensor:
    batch_size, num_keypoints = heatmaps.shape[:2]
    peaks = heatmaps.reshape(batch_size, num_keypoints, -1).amax(dim=-1)
    return peaks.sum(dim=-1)


def u_shaped_sampling_probs(memory_size: int) -> np.ndarray:
    center = (memory_size - 1) / 2.0
    weights = np.array([(index - center) ** 2 + 1.0 for index in range(memory_size)], dtype=np.float64)
    return weights / weights.sum()


def sample_from_memory_bank(
    memory_bank: DynamicMemoryBank,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    if len(memory_bank) == 0:
        raise ValueError("Memory bank is empty.")

    indices = np.random.choice(
        np.arange(len(memory_bank)),
        size=batch_size,
        replace=True,
        p=u_shaped_sampling_probs(len(memory_bank)),
    )
    images = torch.stack([memory_bank.get(int(index)).image for index in indices], dim=0).to(device)
    pseudo_heatmaps = torch.stack(
        [memory_bank.get(int(index)).pseudo_heatmap for index in indices],
        dim=0,
    ).to(device)
    return images, pseudo_heatmaps, indices


@torch.no_grad()
def update_memory_pseudolabels(
    memory_bank: DynamicMemoryBank,
    indices: Sequence[int],
    pred_heatmap: torch.Tensor,
    old_pseudo_heatmap: torch.Tensor,
) -> None:
    pred_scores = heatmap_confidence_score(pred_heatmap)
    old_scores = heatmap_confidence_score(old_pseudo_heatmap)
    for slot, memory_index in enumerate(indices):
        if pred_scores[slot] > old_scores[slot]:
            memory_bank.update_heatmap(int(memory_index), pred_heatmap[slot])


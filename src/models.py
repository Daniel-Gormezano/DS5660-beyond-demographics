"""PyTorch models: a multi-task MLP (shared trunk + three output heads)
and a single-task MLP with the same architecture for the separate-network baseline."""
from __future__ import annotations

import torch
from torch import nn

TASKS = ("depressive", "anxiety", "uncertainty")
N_CLASSES = 4


def trunk(in_dim: int, hidden: tuple[int, ...], dropout: float) -> tuple[nn.Sequential, int]:
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    return nn.Sequential(*layers), d


def head(in_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, N_CLASSES))


class MultiTaskMLP(nn.Module):
    """Shared fully connected layers -> one prediction head per outcome."""

    def __init__(self, in_dim: int, hidden=(128, 64), dropout=0.3, tasks=TASKS):
        super().__init__()
        self.tasks = tuple(tasks)
        self.shared, d = trunk(in_dim, hidden, dropout)
        self.heads = nn.ModuleDict({t: head(d, dropout) for t in self.tasks})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.shared(x)
        return {t: self.heads[t](z) for t in self.tasks}


class SingleTaskMLP(MultiTaskMLP):
    """Same architecture with a single head (separate-network baseline)."""

    def __init__(self, in_dim: int, task: str, hidden=(128, 64), dropout=0.3):
        super().__init__(in_dim, hidden, dropout, tasks=(task,))

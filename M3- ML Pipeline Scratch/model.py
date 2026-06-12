"""Compact MLP for per-slot GSM channel occupancy detection.

Takes 650 per-slot spectral features (26 per slot × 25 slots) and
outputs 25 occupancy logits, one per GSM channel.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from feature_extraction import FEATURE_DIM, N_SLOTS, FEATURES_PER_SLOT

LOGGER = logging.getLogger(__name__)


class GSMOccupancyMLP(nn.Module):

    def __init__(
        self,
        input_dim: int  = FEATURE_DIM,
        n_slots  : int  = N_SLOTS,
        dropout  : float = 0.3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_slots   = n_slots

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, n_slots),   # one logit per slot
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
      
        return self.net(features)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_model_memory_bytes(model: nn.Module,
                                bytes_per_parameter: int = 4) -> int:
    return count_parameters(model) * bytes_per_parameter


def estimate_macs(model: GSMOccupancyMLP) -> int:
    """Estimate MACs for one inference pass through Linear layers."""
    macs = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            macs += module.in_features * module.out_features
    return macs


def log_model_size(model: GSMOccupancyMLP) -> None:
    params  = count_parameters(model)
    memory  = estimate_model_memory_bytes(model)
    macs    = estimate_macs(model)
    LOGGER.info("Model parameters: %d", params)
    LOGGER.info("Estimated float32 parameter memory: %.2f KiB", memory / 1024.0)
    LOGGER.info("Estimated MACs per inference: %d", macs)

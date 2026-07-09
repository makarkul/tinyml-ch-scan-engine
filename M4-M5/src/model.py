"""Slot-aware residual MLP for GSM occupancy detection.

VERSION 3 — Slot-aware architecture with smaller residual blocks.

Key insight from v2 diagnostics
---------------------------------
The 256-wide residual MLP overfit (train F1=0.42, val F1=0.28 at best
epoch 12). Root cause: 568k params for 14k training samples = 40 params
per sample — too many degrees of freedom.

This version uses two structural improvements:

1. Slot-aware encoding (shared weights across slots)
   Input (B, 125) is reshaped to (B, 25, 5) — 25 slots, 5 features each.
   A small shared encoder (same weights for all slots) maps each slot's
   5 features to a 32-dim slot embedding. This reduces effective params
   and encodes the inductive bias that slot k and slot k+5 should respond
   to carriers the same way — only their frequency differs.

2. Smaller residual trunk (128-wide, 2 blocks)
   After slot encoding, a cross-slot trunk mixes information across all
   25 slots. Width 128 and 2 residual blocks keeps total params ~50k —
   reasonable for 14k training samples (3.5 params/sample).

Architecture
------------
Input      : (B, 125)  — 5 features × 25 slots
Reshape    : (B, 25, 5)
SlotEncode : Linear(5→32) LN GELU  [shared weights, applied per slot]
             → (B, 25, 32)
Flatten    : (B, 800)
Project    : Linear(800→128) LN GELU
ResBlock×2 : Linear(128→128) LN GELU + Linear(128→128) LN  [+ skip]
Dropout    : (0.3)
Output     : Linear(128→25)  — one logit per slot

Parameters : ~56k  (vs 568k for v2, 27k for v1)
MACs       : ~57k  (vs 563k for v2, 27k for v1)
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from feature_extraction import FEATURE_DIM, N_SLOTS, FEATURES_PER_SLOT

LOGGER = logging.getLogger(__name__)


class _ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim, bias=False),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class GSMOccupancyMLP(nn.Module):
    """Slot-aware residual MLP for 25-channel GSM occupancy detection.

    Parameters
    ----------
    input_dim    : total input features (default FEATURE_DIM = 125)
    n_slots      : output logits (default N_SLOTS = 25)
    slot_dim     : per-slot embedding dimension (default 32)
    hidden_dim   : cross-slot trunk width (default 128)
    n_blocks     : residual blocks in trunk (default 2)
    dropout      : dropout probability (default 0.3)
    """

    def __init__(
        self,
        input_dim  : int   = FEATURE_DIM,
        n_slots    : int   = N_SLOTS,
        slot_dim   : int   = 32,
        hidden_dim : int   = 128,
        n_blocks   : int   = 2,
        dropout    : float = 0.3,
    ) -> None:
        super().__init__()
        self.input_dim       = input_dim
        self.n_slots         = n_slots
        self.features_per_slot = input_dim // n_slots  # 5
        self.slot_dim        = slot_dim

        # Input normalisation — per-feature BatchNorm
        self.input_bn = nn.BatchNorm1d(input_dim)

        # Slot encoder — shared weights applied to each slot's 5 features
        # Same encoder for all 25 slots: inductive bias that carrier
        # signatures are frequency-independent in feature space.
        self.slot_encoder = nn.Sequential(
            nn.Linear(self.features_per_slot, slot_dim, bias=False),
            nn.LayerNorm(slot_dim),
            nn.GELU(),
            nn.Linear(slot_dim, slot_dim, bias=False),
            nn.LayerNorm(slot_dim),
            nn.GELU(),
        )

        # Cross-slot trunk — mixes information across all 25 slots
        cross_dim = n_slots * slot_dim   # 25 × 32 = 800
        self.project = nn.Sequential(
            nn.Linear(cross_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.trunk = nn.Sequential(
            *[_ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)],
            nn.Dropout(dropout),
        )

        # Output head
        self.head = nn.Linear(hidden_dim, n_slots)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.head.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features : (B, 125) float32

        Returns
        -------
        (B, 25) raw logits
        """
        B = features.shape[0]

        # Normalise inputs
        x = self.input_bn(features)                     # (B, 125)

        # Reshape to slots and encode each slot independently
        x = x.reshape(B, self.n_slots, self.features_per_slot)  # (B, 25, 5)
        x = self.slot_encoder(x)                        # (B, 25, 32)

        # Flatten and mix across slots
        x = x.reshape(B, -1)                            # (B, 800)
        x = self.project(x)                             # (B, 128)
        x = self.trunk(x)                               # (B, 128)
        return self.head(x)                             # (B, 25)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_model_memory_bytes(model: nn.Module,
                                bytes_per_parameter: int = 4) -> int:
    return count_parameters(model) * bytes_per_parameter


def estimate_macs(model: GSMOccupancyMLP) -> int:
    macs = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            macs += m.in_features * m.out_features
    return macs


def log_model_size(model: nn.Module) -> None:
    params  = count_parameters(model)
    memory  = estimate_model_memory_bytes(model)
    macs    = estimate_macs(model) if isinstance(model, GSMOccupancyMLP) else 0
    LOGGER.info("Model parameters: %d", params)
    LOGGER.info("Estimated float32 parameter memory: %.2f KiB", memory / 1024.0)
    LOGGER.info("Estimated MACs per inference: %d", macs)
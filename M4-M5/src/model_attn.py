from __future__ import annotations

import logging
import math

import torch
from torch import nn

from feature_extraction import FEATURE_DIM, N_SLOTS, FEATURES_PER_SLOT

LOGGER = logging.getLogger(__name__)


class _SlotSelfAttentionBlock(nn.Module):
    """Pre-LN transformer block: self-attention + feed-forward, both residual.

    Operates on (B, n_slots, dim) -- n_slots is the sequence length, so
    every slot can attend to every other slot in a single layer (global
    receptive field), unlike a width-3 conv which only sees neighbours.
    """

    def __init__(self, dim: int, n_heads: int, ff_mult: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.qkv   = nn.Linear(dim, dim * 3, bias=False)
        self.proj  = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_slots, dim)"""
        B, S, D = x.shape

        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, S, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # (3, B, n_heads, S, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, n_heads, S, S)
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v                             # (B, n_heads, S, head_dim)
        out = out.transpose(1, 2).reshape(B, S, D)  # (B, S, D)
        out = self.proj(out)
        x = x + out

        x = x + self.ff(self.norm2(x))
        return x


class GSMOccupancySlotAttention(nn.Module):
    """Slot-attention model for 25-channel GSM occupancy detection.

    Parameters
    ----------
    input_dim  : total input features (default FEATURE_DIM)
    n_slots    : output logits (default N_SLOTS = 25)
    slot_dim   : per-slot token dimension (default 32)
    n_heads    : attention heads per block (default 4)
    n_blocks   : self-attention blocks (default 2)
    dropout    : dropout probability (default 0.1)

    FIX (positional embedding)
    ---------------------------
    Self-attention is permutation-invariant by construction: softmax(QK^T)
    is computed purely from content, with no notion that "slot 5 sits next
    to slot 6." But the interference model this project trains against
    (see gsm_dataset_gen_new.py) is explicitly distance-based -- adjacent
    leakage at +/-1 to +/-3 slots, wideband blockers spanning many
    contiguous slots. Without a positional signal, attention can only
    infer these relationships indirectly through the engineered F4/F5
    ratio features, not through its own attention weights.

    A learned per-slot positional embedding is added to each slot's
    encoding before the attention blocks, so attention can learn "prefer
    nearby slots" directly instead of relying entirely on content
    similarity. This mirrors standard transformer positional embeddings
    (Vaswani et al.) -- here learned rather than sinusoidal, since 25 is
    a small fixed sequence length and a learned table is cheap (25*slot_dim
    params) and adds no inference-time cost beyond one elementwise add.
    """

    def __init__(
        self,
        input_dim : int   = FEATURE_DIM,
        n_slots   : int   = N_SLOTS,
        slot_dim  : int   = 32,
        n_heads   : int   = 4,
        n_blocks  : int   = 2,
        dropout   : float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim         = input_dim
        self.n_slots           = n_slots
        self.features_per_slot = input_dim // n_slots
        self.slot_dim          = slot_dim

        self.input_bn = nn.BatchNorm1d(input_dim)

        # Shared per-slot encoder — identical pattern to model.py's slot_encoder.
        self.slot_encoder = nn.Sequential(
            nn.Linear(self.features_per_slot, slot_dim, bias=False),
            nn.LayerNorm(slot_dim),
            nn.GELU(),
        )

        # Learned positional embedding — one vector per slot index, added
        # after slot_encoder so attention gets an explicit "which slot am I"
        # signal alongside the slot's own feature content.
        self.pos_embed = nn.Parameter(torch.zeros(1, n_slots, slot_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            _SlotSelfAttentionBlock(slot_dim, n_heads, ff_mult=2, dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.final_norm = nn.LayerNorm(slot_dim)
        self.dropout = nn.Dropout(dropout)

        # Per-slot classifier — independent weights per slot (flatten +
        # dense), NOT a shared Linear(slot_dim, 1) applied per-token.
        # BUGFIX (kept from earlier fix): a fully shared Linear(slot_dim, 1)
        # receives, for each weight, the average gradient across all 25
        # slot positions -- dominated by the ~90% empty class, collapsing
        # the classifier to all-negative. Flattening to (B, slot_dim*n_slots)
        # and using one Linear(slot_dim*n_slots, n_slots) gives every output
        # slot its own independent weight row, so slot k's gradient is no
        # longer averaged with the other 24 slots.
        self.head = nn.Linear(slot_dim * n_slots, n_slots)

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
        features : (B, input_dim) float32

        Returns
        -------
        (B, n_slots) raw logits
        """
        B = features.shape[0]

        x = self.input_bn(features)                              # (B, input_dim)
        x = x.reshape(B, self.n_slots, self.features_per_slot)    # (B, 25, features_per_slot)
        x = self.slot_encoder(x)                                  # (B, 25, slot_dim)
        x = x + self.pos_embed                                    # inject slot identity/position

        for block in self.blocks:
            x = block(x)                                          # (B, 25, slot_dim)

        x = self.final_norm(x)
        x = self.dropout(x)
        x = x.reshape(x.size(0), -1)           # (B, slot_dim*25)
        logits = self.head(x)                  # (B, 25) — independent weights per slot
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_macs(model: GSMOccupancySlotAttention) -> int:
    """Estimate multiply-accumulate operations for one inference pass."""
    S = model.n_slots
    D = model.slot_dim
    macs = 0
    # Slot encoder: applied to each of S slots
    macs += S * model.features_per_slot * D
    for block in model.blocks:
        # qkv projection + output projection
        macs += S * D * (D * 3) + S * D * D
        # attention matmuls: QK^T and attn@V, both (S,D)x(D,S)-ish
        macs += 2 * S * S * D
        # feed-forward
        ff_dim = block.ff[0].out_features
        macs += S * D * ff_dim + S * ff_dim * D
    # classifier head: flatten(S*D) -> S, independent weights per slot
    macs += (S * D) * S
    return macs


def log_model_size(model: nn.Module) -> None:
    params = count_parameters(model)
    memory = params * 4
    macs   = estimate_macs(model) if isinstance(model, GSMOccupancySlotAttention) else 0
    LOGGER.info("Model parameters: %d", params)
    LOGGER.info("Estimated float32 parameter memory: %.2f KiB", memory / 1024.0)
    LOGGER.info("Estimated MACs per inference: %d", macs)
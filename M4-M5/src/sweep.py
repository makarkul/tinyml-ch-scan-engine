"""DeepSweep-style parallel "spectrum sweep" model for GSM occupancy detection.

Adapted from:
  C. P. Robinson, D. Uvaydov, S. D'Oro, T. Melodia,
  "DeepSweep: Parallel and Scalable Spectrum Sensing via Convolutional
  Neural Networks," IEEE ICMLCN 2024.

Core idea borrowed from the paper
----------------------------------
DeepSweep splits a wideband spectrum into G equal-sized chunks and feeds
all G chunks through a single SHALLOW, SHARED CNN as a batch -- i.e. each
chunk is processed as an *independent example*, not as one of G outputs
computed jointly from one shared layer. The G individual outputs are then
reassembled into one spectrum-wide result.

Why this maps onto our problem
--------------------------------
Our 25 GSM slots are already DeepSweep's "G chunks" (G = N_SLOTS = 25).
Folding the slot dimension into the BATCH dimension before the shared
layers means each slot becomes its own independent training example
with its own gradient -- 7000 training samples becomes 7000*25 = 175,000
single-slot examples, and no shared weight ever sees an average gradient
across slots (the collapse failure mode documented in model_tcn.py /
model_attn.py's earlier slot_bias workaround never applies here).

FIX (context window)
----------------------
First version of this model fed each chunk ONLY its own slot's 6
features, with zero access to neighbouring slots. Training logs showed
a flat, tiny train/val gap for 30+ epochs (VaF1 stuck ~0.51-0.53) -- the
signature of a model that has CONVERGED to the best fit it can express,
not one that's data-starved. Root cause: it structurally cannot
represent "this slot's energy is high because of adjacent-channel
leakage from slot k+2" -- exactly the pattern gsm_dataset_gen_new.py
injects at +/-1 to +/-3 slots.

Fix: widen each chunk to include its immediate left/right neighbours
(+/-1 slot) before folding into the batch dimension. This keeps the
independent-example training benefit (still no joint (B,25,...) tensor
anywhere) while finally giving the classifier the minimal local context
it needs for the dataset's dominant interference pattern. Edge slots
(0 and 24) are zero-padded on the missing side.

What changed from the paper (and why), same as before
---------------------------------------------------------
1. Softmax -> per-example sigmoid: our problem is multi-label (multiple
   slots can be occupied simultaneously), softmax over chunks assumes
   exactly one positive chunk, which doesn't hold here.
2. Chunk content: pre-computed per-slot features (F1..F6), not raw FFT
   bins -- Conv1D+MaxPool1D+Dense+Dropout pipeline from Fig. 3 kept
   structurally intact, applied to the (now widened) feature sequence.

Architecture (per chunk, batch-folded, WITH context)
-------------------------------------------------------
Input        : (B, 25, 6)  -- 6 features per slot
Context      : concat [slot k-1, slot k, slot k+1] -> (B, 25, 18)
               (zero-padded at edges)
Fold         : (B*25, 1, 18)
Conv1D       : Conv1d(1 -> fe_channels, kernel=3, padding=1) + ReLU
MaxPool1D    : kernel=2, stride=2               -> length 9
Flatten      : (B*25, fe_channels*9)
Dense        : Linear(fe_channels*9 -> hidden_dim) + ReLU
Dropout      : (dropout)
Output       : Linear(hidden_dim -> 1)          -- one sigmoid logit
Unfold       : reshape back to (B, 25)

Parameters   : a few thousand, still tiny next to MLP/TCN/ATTN (see
               count_parameters() / log_model_size())
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import nn

from feature_extraction import FEATURE_DIM, N_SLOTS, FEATURES_PER_SLOT

LOGGER = logging.getLogger(__name__)

CONTEXT_RADIUS = 1   # +/-1 neighbour slot included per chunk


class GSMOccupancySweep(nn.Module):
    """DeepSweep-style parallel chunk classifier for 25-channel GSM occupancy,
    with a +/-1 slot context window.

    Unlike GSMOccupancyMLP / GSMOccupancySlotAttention / GSMOccupancyTCN,
    the shared Conv1D/Dense layers never see a joint (B, 25, ...) tensor --
    slots are folded into the batch dimension, so each is classified as an
    independent example. Each chunk now carries its own features PLUS its
    immediate left/right neighbour's features (zero-padded at the band
    edges), giving the classifier the minimal context needed for
    adjacent-channel leakage without reintroducing a joint multi-slot
    tensor anywhere in the shared layers.

    Parameters
    ----------
    input_dim   : total input features (default FEATURE_DIM, e.g. 150 = 25*6)
    n_slots     : number of GSM channels / chunks (default N_SLOTS = 25)
    fe_channels : Conv1D output channels (default 16)
    hidden_dim  : Dense layer width (default 32)
    dropout     : dropout probability before the output layer (default 0.3)
    context_radius : neighbours included on each side (default 1 -> 3x chunk width)
    """

    def __init__(
        self,
        input_dim     : int   = FEATURE_DIM,
        n_slots       : int   = N_SLOTS,
        fe_channels   : int   = 16,
        hidden_dim    : int   = 32,
        dropout       : float = 0.3,
        context_radius: int   = CONTEXT_RADIUS,
    ) -> None:
        super().__init__()
        self.input_dim         = input_dim
        self.n_slots           = n_slots
        self.features_per_slot = input_dim // n_slots      # e.g. 6, own-slot width
        self.context_radius    = context_radius
        self.chunk_len         = self.features_per_slot * (2 * context_radius + 1)  # e.g. 18
        self.fe_channels       = fe_channels
        self.hidden_dim        = hidden_dim

        # Input normalisation over the flat feature vector, same pattern
        # as model.py / model_attn.py -- learns per-feature-index mean/std.
        self.input_bn = nn.BatchNorm1d(input_dim)

        # ── Shared shallow CNN (Fig. 3 of the DeepSweep paper) ──────────
        # Applied identically to every (B*n_slots, 1, chunk_len) chunk --
        # weight sharing here is safe because each chunk (now including
        # its neighbours) is still one independent batch example.
        self.conv = nn.Conv1d(
            in_channels=1, out_channels=fe_channels,
            kernel_size=3, padding=1, bias=False,
        )
        self.bn_conv = nn.BatchNorm1d(fe_channels)
        self.act_conv = nn.ReLU(inplace=True)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        pooled_len = self.chunk_len // 2
        if pooled_len < 1:
            raise ValueError(
                f"chunk_len={self.chunk_len} too small for "
                "MaxPool1d(kernel_size=2, stride=2) -- need >= 2."
            )
        self.pooled_len = pooled_len

        self.dense = nn.Linear(fe_channels * pooled_len, hidden_dim)
        self.act_dense = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        # Single logit per chunk -- sigmoid applied outside (BCEWithLogits/
        # Focal loss expects raw logits), NOT softmax -- multi-label problem.
        self.output = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.output.bias)

    def _build_context_chunks(self, x: torch.Tensor) -> torch.Tensor:
        """
        Widen each slot's own features with its +/-context_radius
        neighbours' features, concatenated along the feature axis.

        Parameters
        ----------
        x : (B, n_slots, features_per_slot)

        Returns
        -------
        (B, n_slots, chunk_len) -- chunk_len = features_per_slot * (2r+1)
        Edge slots are zero-padded on the missing side(s), so every slot
        still gets a fixed-size chunk.
        """
        r = self.context_radius
        # Pad along the slot axis (dim=1) with r zero-slots on each side.
        # F.pad on a 3D tensor pads the LAST dim by default, so permute to
        # put the slot axis last, pad, then permute back.
        x_t = x.transpose(1, 2)                          # (B, features_per_slot, n_slots)
        x_t = F.pad(x_t, (r, r))                          # (B, features_per_slot, n_slots+2r)
        x_padded = x_t.transpose(1, 2)                    # (B, n_slots+2r, features_per_slot)

        windows = []
        for offset in range(-r, r + 1):
            start = r + offset
            end   = start + self.n_slots
            windows.append(x_padded[:, start:end, :])     # (B, n_slots, features_per_slot)

        return torch.cat(windows, dim=-1)                 # (B, n_slots, chunk_len)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features : (B, input_dim) float32, e.g. (B, 150)

        Returns
        -------
        (B, n_slots) raw logits -- one per slot, sigmoid applied by the
        loss function (BCEWithLogitsLoss / FocalLossWithLogits).
        """
        B = features.shape[0]

        x = self.input_bn(features)                                   # (B, input_dim)
        x = x.reshape(B, self.n_slots, self.features_per_slot)         # (B, 25, 6)

        # ── Context fix: widen chunk with +/-1 neighbour before folding ──
        x = self._build_context_chunks(x)                              # (B, 25, chunk_len)

        # ── DeepSweep's core step: fold slots into the batch dim ───────
        # Every one of the B*25 rows is still an independent example --
        # it just now carries its own + neighbours' features, so no
        # shared weight's gradient is ever averaged across slots.
        x = x.reshape(B * self.n_slots, 1, self.chunk_len)             # (B*25, 1, chunk_len)

        x = self.conv(x)                                               # (B*25, fe_channels, chunk_len)
        x = self.bn_conv(x)
        x = self.act_conv(x)
        x = self.pool(x)                                               # (B*25, fe_channels, pooled_len)

        x = x.reshape(B * self.n_slots, -1)                            # (B*25, fe_channels*pooled_len)
        x = self.dense(x)                                              # (B*25, hidden_dim)
        x = self.act_dense(x)
        x = self.dropout(x)
        logits = self.output(x)                                        # (B*25, 1)

        # ── Unfold back to per-sample, per-slot logits ─────────────────
        logits = logits.reshape(B, self.n_slots)                       # (B, 25)
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_model_memory_bytes(model: nn.Module,
                                bytes_per_parameter: int = 4) -> int:
    return count_parameters(model) * bytes_per_parameter


def estimate_macs(model: GSMOccupancySweep) -> int:
    """Estimate multiply-accumulate operations for one inference pass
    (per slot chunk; batching over n_slots parallelises this, doesn't
    reduce per-chunk cost)."""
    L = model.chunk_len
    C = model.fe_channels
    P = model.pooled_len
    macs = 0
    # Conv1d: kernel=3, in_channels=1, out_channels=C, applied at L positions
    macs += 3 * 1 * C * L
    # Dense: (C*P) -> hidden_dim
    macs += (C * P) * model.hidden_dim
    # Output: hidden_dim -> 1
    macs += model.hidden_dim * 1
    # Total per sample = macs * n_slots (all slots processed, just batched)
    return macs * model.n_slots


def log_model_size(model: nn.Module) -> None:
    params  = count_parameters(model)
    memory  = estimate_model_memory_bytes(model)
    macs    = estimate_macs(model) if isinstance(model, GSMOccupancySweep) else 0
    LOGGER.info("GSMOccupancySweep parameters: %d", params)
    LOGGER.info("Estimated float32 parameter memory: %.2f KiB", memory / 1024.0)
    LOGGER.info("Estimated MACs per inference (all %d slots): %d",
                model.n_slots if isinstance(model, GSMOccupancySweep) else 0, macs)
"""End-to-end 1D TCN for GSM per-slot occupancy detection.

Updated with ideas borrowed from:
  D. Uvaydov, S. D'Oro, F. Restuccia, T. Melodia,
  "DeepSense: Fast Wideband Spectrum Sensing Through Real-Time
  In-the-Loop Deep Learning," IEEE INFOCOM 2021.

What was borrowed from DeepSense, and why
--------------------------------------------
DeepSense's CNN (train_test_CNN.py) stacks FOUR Conv1D layers with
GROWING kernel size (3, 3, 5, 5) before its classifier, vs. our previous
seq_processor's two kernel=3 layers (receptive field only +/-2 slots).
The dataset's own adjacent-channel leakage model (gsm_dataset_gen_new.py)
injects interference at +/-1 to +/-3 slots -- wider than what +/-2 could
see. Two ideas taken from DeepSense's stack:

  1. Growing kernel sizes across layers (here: 3 -> 5 -> 3) to widen the
     receptive field without adding many more layers.
  2. LeakyReLU(alpha=0.1) instead of GELU -- avoids dead units when
     gradients are sparse and imbalanced (~90% empty slots), matching
     DeepSense's own choice for the same reason.

What was NOT borrowed, and why
---------------------------------
DeepSense pools between conv layers (MaxPooling1D stride 2) because its
task is a single whole-capture classification (which of 4 signal types
is present) -- pooling collapsing sequence length doesn't cost it
anything. Our task needs one output PER SLOT, so sequence length must
stay exactly n_slots (25) all the way to the classifier -- pooling here
would destroy the slot <-> output correspondence. All new conv layers
below use same-padding (kernel//2) and stride 1, never touching length.

Classifier head fix (independent per-slot weights)
-----------------------------------------------------
Previous version used a fully shared pointwise Conv1d(fe_channels, 1,
k=1) classifier plus a per-slot bias (self.slot_bias) as a cheap
symmetry-breaking patch. That shared layer's gradient for any weight is
still the AVERAGE across all 25 slot positions per sample, dominated by
the ~90% empty class -- the same collapse mechanism documented (and
fixed) in model_attn.py and model.py. Replaced here with the same fix
used there: flatten the (fe_channels, n_slots) feature map and use one
Linear(fe_channels*n_slots, n_slots), giving every output slot its own
independent weight row. slot_bias is removed -- the Linear layer's own
bias already does that job per-output.
"""

from __future__ import annotations

import logging
import math

import torch
from torch import nn

from feature_extraction import N_SLOTS

LOGGER = logging.getLogger(__name__)

# Input dimensions — must match dataset.py _load_raw_iq output
N_IQ_CHANNELS = 2      # I and Q
N_DEC         = 3151   # samples after polyphase decimation to 5 MS/s
SCAN_RATE     = 5_000_000

# GSM slot centre frequencies in Hz — must match feature_extraction.SLOT_FREQ_HZ
# Slot k: centre = -2400 kHz + k × 200 kHz
_SLOT_FREQ_HZ = [-2_400_000 + k * 200_000 for k in range(N_SLOTS)]


class _SlotDemodFrontend(nn.Module):

    def __init__(self, n_dec: int = N_DEC, n_slots: int = N_SLOTS) -> None:
        super().__init__()
        self.n_dec   = n_dec
        self.n_slots = n_slots

        t = torch.arange(n_dec, dtype=torch.float64)
        freqs = torch.tensor(_SLOT_FREQ_HZ, dtype=torch.float64)  # (n_slots,)
        angle = 2 * math.pi * freqs[:, None] * t[None, :] / SCAN_RATE  # (n_slots, n_dec)
        cos_init = (torch.cos(angle) / n_dec).to(torch.float32)
        sin_init = (torch.sin(angle) / n_dec).to(torch.float32)

        # Trainable — initialised to the correct channelizer, not random.
        self.mix_cos = nn.Parameter(cos_init)   # (n_slots, n_dec)
        self.mix_sin = nn.Parameter(sin_init)   # (n_slots, n_dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 2, n_dec) float32 — I in channel 0, Q in channel 1

        Returns
        -------
        (B, 2, n_slots) float32 — (real, imag) matched-filter correlation
        per slot, ready to be projected per-slot into an embedding.
        """
        I = x[:, 0, :]   # (B, n_dec)
        Q = x[:, 1, :]   # (B, n_dec)

        corr_re = I @ self.mix_cos.T + Q @ self.mix_sin.T   # (B, n_slots)
        corr_im = Q @ self.mix_cos.T - I @ self.mix_sin.T   # (B, n_slots)

        return torch.stack([corr_re, corr_im], dim=1)       # (B, 2, n_slots)


class GSMOccupancyTCN(nn.Module):
    """End-to-end 1D TCN for 25-channel GSM occupancy detection.

    Parameters
    ----------
    n_slots     : number of GSM channels (output width). Default 25.
    fe_channels : number of filters in the front-end. Default 32.
    seq_channels: hidden width of the sequence processor. Default 64.
    dropout     : dropout probability in the sequence processor. Default 0.3.
    freeze_frontend : freeze the demod mixing weights at the physically
        correct matched-filter values instead of letting them train.
    """

    def __init__(
        self,
        n_slots     : int   = N_SLOTS,
        fe_channels : int   = 32,
        seq_channels: int   = 64,
        dropout     : float = 0.3,
        freeze_frontend : bool  = False,
    ) -> None:
        super().__init__()
        self.n_slots      = n_slots
        self.fe_channels  = fe_channels
        self.seq_channels = seq_channels

        # ── Stage 1: Spectral front-end ───────────────────────────────
        # Per-slot matched-filter demodulation, followed by a shared
        # per-slot projection from (real, imag) up to fe_channels.
        self.demod = _SlotDemodFrontend(n_dec=N_DEC, n_slots=n_slots)
        if freeze_frontend:
            self.demod.mix_cos.requires_grad_(False)
            self.demod.mix_sin.requires_grad_(False)
            LOGGER.info("TCN front-end frozen at physically-correct matched-filter weights")

        self.slot_project = nn.Sequential(
            nn.Linear(2, fe_channels, bias=False),
            nn.LayerNorm(fe_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.frontend_bn = nn.BatchNorm1d(fe_channels)

        # ── Stage 2: Slot-sequence processor ─────────────────────────
        # DeepSense-inspired: growing kernel sizes (3 -> 5 -> 3) widen
        # the receptive field to +/-4 slots (1+2+1), covering the
        # dataset's +/-3 slot adjacent-leakage model with margin, vs.
        # the previous two-k=3-layer stack's +/-2 slot reach.
        # All layers use same-padding (kernel//2), stride 1 -- sequence
        # length stays exactly n_slots throughout, unlike DeepSense's
        # own pooled stack (see module docstring for why pooling is
        # not used here: we need one output per slot, not one global
        # classification).
        self.seq_processor = nn.Sequential(
            nn.Conv1d(fe_channels,  seq_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(seq_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=dropout),

            nn.Conv1d(seq_channels, seq_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(seq_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=dropout),

            nn.Conv1d(seq_channels, fe_channels,  kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(fe_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # ── Stage 3: Per-slot classifier ─────────────────────────────
        # Shared feature refinement (pointwise, safe to share since it
        # doesn't produce the final per-slot decision), followed by an
        # INDEPENDENT-WEIGHTS head.
        #
        # BUGFIX (matches model.py / model_attn.py / sweep.py convention):
        # a fully shared Conv1d(fe_channels, 1, k=1) classifier's gradient
        # for any weight is the average over all 25 slot positions per
        # sample -- dominated by the ~90% empty class, collapsing the
        # classifier to all-negative. A per-slot bias alone (previous
        # version) only shifts a shared decision boundary, it cannot let
        # the shared weight matrix specialise per slot. Flattening to
        # (B, fe_channels*n_slots) and using one
        # Linear(fe_channels*n_slots, n_slots) gives every output slot
        # its own independent weight row, so slot k's gradient is no
        # longer averaged with the other 24 slots.
        self.classifier_shared = nn.Sequential(
            nn.Conv1d(fe_channels, fe_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(fe_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.classifier_head = nn.Linear(fe_channels * n_slots, n_slots)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming-uniform initialisation for conv/linear layers; constant for norm layers.

        Note: self.demod.mix_cos / mix_sin are intentionally excluded —
        they get their physically-correct sinusoid initialisation inside
        _SlotDemodFrontend.__init__ and must not be overwritten here.
        """
        for name, m in self.named_modules():
            if m is self.demod:
                continue
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="leaky_relu", a=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        nn.init.zeros_(self.classifier_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1: per-slot demodulation + shared projection
        corr = self.demod(x)                       # (B, 2, 25)
        corr = corr.transpose(1, 2)                 # (B, 25, 2)
        feat = self.slot_project(corr)              # (B, 25, fe_channels)
        feat = feat.transpose(1, 2)                 # (B, fe_channels, 25)
        feat = self.frontend_bn(feat)               # (B, fe_channels, 25)

        # Stage 2: slot-sequence processor, residual so it can be a
        # near no-op early in training rather than needing to relearn
        # identity from scratch.
        feat = feat + self.seq_processor(feat)      # (B, fe_channels, 25)

        # Stage 3: shared refinement, then independent-weights head
        feat = self.classifier_shared(feat)         # (B, fe_channels, 25)
        feat = feat.reshape(feat.size(0), -1)       # (B, fe_channels*25)
        out  = self.classifier_head(feat)           # (B, 25) — independent weights per slot
        return out

    # ── Introspection helpers ─────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_macs(self) -> int:
        """Estimate multiply-accumulate operations for one inference pass."""
        macs = 0
        # Stage 1: demod correlation (2 mixes × n_slots × n_dec) + projection
        macs += 2 * self.n_slots * N_DEC
        macs += 2 * self.fe_channels * self.n_slots
        # Stage 2: three conv layers, kernel sizes 3, 5, 3, same length (n_slots)
        macs += self.fe_channels  * 3 * self.seq_channels * self.n_slots
        macs += self.seq_channels * 5 * self.seq_channels * self.n_slots
        macs += self.seq_channels * 3 * self.fe_channels  * self.n_slots
        # Stage 3: shared pointwise refinement + independent-weights head
        macs += self.fe_channels * self.fe_channels * self.n_slots
        macs += (self.fe_channels * self.n_slots) * self.n_slots
        return macs

    def log_model_size(self) -> None:
        params  = self.count_parameters()
        mem_kib = params * 4 / 1024
        macs    = self.estimate_macs()
        LOGGER.info("GSMOccupancyTCN parameters : %d", params)
        LOGGER.info("Estimated float32 memory   : %.2f KiB", mem_kib)
        LOGGER.info("Estimated MACs per inference: %d", macs)
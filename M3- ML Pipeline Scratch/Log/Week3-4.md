# Week 3-4, 2026 (Jun 2–13)
## What we tried
- Fixed three DSP bugs flagged in M2 review: Welch windowing mismatch in
  FFT energy detector, sign convention in PFB channeliser, and N_RESAMPLE
  value in the dataset generator.
- Built the full ML pipeline from scratch — phases 1 through 6. Scalar MLP
  baseline, multi-head auxiliary losses, spectrogram CNN, fading 1D CNN,
  full fused model, ablation study across six configurations.
- Scaled training from 5k to 10k samples and measured improvement.
- Ran int8 post-training quantisation on the full model.
- Tested the trained model on a real Vodafone GSM capture (9.18M samples,
  ~4 MS/s, float32) and compared it against the FFT channeliser on the
  same data.
- Diagnosed distribution shift from the real data test and updated the
  synthetic generator with four fixes.

## What worked
- **Spectrogram CNN was the dominant contributor.** Config C (Branch 1 only,
  22k params) got F1=0.712, nearly matching the full 137k-param model at
  F1=0.724. Adding scalar features and fading vectors gave only marginal
  gains at this dataset size.
- **P_fa and power class accuracy targets met.** P_fa=0.010 (target ≤0.05),
  power class accuracy=0.808 (target ≥0.75).
- **10k samples improved F1 from 0.724 to 0.733.** P_d at SNR 12–20 dB
  crossed 0.90 for the first time.
- **Int8 quantisation was essentially free.** 2.9× size reduction (560 KB →
  196 KB), 1.5× speedup, only −0.002 F1.
- **Real carrier correctly detected.** Model gave slot 12 probability 0.9961
  on the Vodafone capture. The real carrier was at 1268× the noise floor.
- **Leakage rejection: 99.9% reduction.** FFT channeliser flagged 39,615
  leakage false alarms across 3643 windows. The model produced only 55.
  This is the clearest result in the project — the model learned spectral
  shape features that distinguish real carriers from leakage, which the
  energy detector cannot do.

## What failed or surprised
- **Multi-task learning (Phase 3) did not help occupancy F1.** Adding power
  class, GMSK confidence, and impairment flag heads hurt occupancy from
  F1=0.583 to 0.531 before tuning. The trunk lacked capacity to serve all
  four tasks on 5k samples. Impairment F1 remained near zero until
  pos_weight was fixed. Documented as a legitimate finding — auxiliary
  heads on scalar features do not improve the primary task when the trunk
  is the bottleneck.
- **Distribution shift caused two false positives at slots 2 and 22.**
  Structured wideband interference at bins 19–21 (not present in synthetic
  training data) triggered the model. The flat AWGN blocker in the
  generator did not capture this pattern.
- **Model detects carrier in only 0.9% of windows despite it being present
  in all of them.** Root cause: frequency hopping (carrier moves between
  slots 12, 13, 15) and carrier not always centred within the slot. The
  model was trained on fixed-slot, centred carriers and generalised poorly
  to hopping signals.
- **Global scale fix did not improve detection rate.** Per-window scaling was
  inconsistent but correcting it to a global scale did not help because the
  deeper problem was frequency hopping and intra-slot carrier offset — not
  amplitude scaling.
- **All 3643 windows showed peak ratio above 100×.** Expected ~13% from TDMA
  timing, but this capture uses the BCCH beacon channel which transmits
  continuously with no silence. The TDMA hypothesis was wrong for this
  specific capture type.

## Decisions
- SNR range extended from −10 to +25 dB to −10 to +40 dB in generator.
- Adjacent carrier power range extended from 5–20 dB to 5–35 dB.
- Flat AWGN wideband blocker replaced with structured interference: flat
  AWGN, narrowband cluster (LTE/UMTS simulation), and shaped Gaussian
  noise chosen randomly per sample.
- Frequency hopping added to generator: 25% of samples now offset the
  carrier by ±80 kHz within its slot.
- Adjacent carrier distance extended from ±1 to ±3 slots to simulate
  leakage spreading further from very strong carriers.
- Config C (22k params, F1=0.712) identified as the deployment-optimal
  architecture — closest to the 64 KB embedded budget before pruning.

## New idea
The model currently processes each 630 µs window independently with no
memory of previous windows. In deployment the chip sees a continuous
stream. A lightweight running average of per-slot probabilities across
consecutive windows — say an exponential moving average with α=0.3 —
would suppress single-window false positives and reinforce persistent
detections. A carrier at slot 12 active across multiple windows would
accumulate probability toward 1.0. Interference causing a one-window
false positive at slot 2 would not survive the averaging. This requires
no retraining, no parameter changes, and negligible compute. It directly
addresses the 0.9% detection rate problem caused by the carrier being
missed in most windows — the few windows where it is correctly detected
at 0.99 would raise the averaged probability well above threshold.

## Next steps
- Generate 50k samples with the updated generator and retrain. Expected
  to close F1 gap toward 0.85 target and improve detection on frequency-
  hopping signals.
- Implement the exponential moving average post-processing on the real
  capture and measure whether detection rate improves without retraining.
- Prune Config C from 86 KB toward the 64 KB embedded deployment budget.
- Get more real captures with different carriers and channel conditions to
  build a proper real-data validation set with verified ground truth.
- Discuss with mentor: priority between 50k scale-up, raw IQ architecture
  exploration, and two-stage detector implementation.

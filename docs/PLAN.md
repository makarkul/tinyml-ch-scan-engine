# Plan: TinyML-Assisted GSM Channel Scan Engine — 12-Week R&D Execution Plan

## Context

Sampige Semiconductor's R&D brief asks a focused, bounded question: can a
compact ML model, fed low-bit complex IF samples (reference: 5-bit I/Q at
104 MS/s, ~5 MHz scan span), perform GSM channel-presence detection,
coarse power-class estimation, and interference-condition flagging with
detection quality and compute cost that are competitive with classical
FFT / channelizer / energy-detector baselines?

The repository is currently empty (fresh `git init`, branch
`claude/plan-research-experiment-bSqK2`, no commits). This plan
establishes the project structure, dataset / baseline / model design,
benchmark methodology, and milestone sequencing required to deliver the
eight deliverables (D1-D8) in the brief over 12 weeks. The intended
outcome is a reproducible R&D asset — synthetic data pipeline, DSP
baseline, TinyML model(s), benchmark report, and Cortex-M-class
feasibility note — that lets leadership decide whether to continue,
narrow, or deprioritize the ML path.

**Confirmed scope choices**
- Full 12-week execution plan, milestones M1-M6.
- PyTorch + ONNX; complexity (params, MACs, peak activation memory)
  estimated analytically. TFLite-Micro is a stretch only.
- Cortex-M class MCU is the framing target for feasibility (e.g.
  Cortex-M4F @ 100-200 MHz, 256-512 KB SRAM, 1-2 MB flash). The brief
  also asks the mentor to confirm "processor vs DFE-adjacent block vs
  mixed" in Week 1 — flag this for the kickoff.

## Goals and Success Criteria

Primary deliverable bar (the brief's minimum acceptance):
1. End-to-end runnable pipeline (not notebooks-only) from config →
   dataset → train → evaluate → report.
2. At least one credible classical baseline implemented and tuned
   before any ML-superiority claim is made.
3. At least one compact ML model trained and evaluated on the same
   corpus as the baseline.
4. Comparative numbers across **four** operating categories: nominal,
   low-SNR, multi-carrier / adjacent-channel, interference / blocker.
5. Reasoned recommendation: continue / modify / narrow / deprioritize.

Non-goals (out of scope, per brief §6.2):
- Full GSM modem, synchronization chain, traffic receiver.
- Production FPGA/ASIC RTL.
- Generic protocol-agnostic spectrum analyzer.
- Large deep models incompatible with TinyML constraints.

## Repository Architecture

```
tinyml-ch-scan-engine/
├── README.md                       # quickstart, repro instructions
├── pyproject.toml                  # deps (numpy, scipy, torch, onnx,
│                                   #  matplotlib, hydra-core, pytest)
├── configs/                        # Hydra/YAML configs
│   ├── signal/                     #   carrier, SNR, offset, blocker sweeps
│   ├── dataset/                    #   split sizes, label schema, seeds
│   ├── baseline/                   #   FFT / PFB / energy-detector params
│   ├── model/                      #   mlp_v1, cnn1d_v1, cnn1d_quant
│   └── experiment/                 #   compositions for M2..M6 runs
├── src/scan_engine/
│   ├── signals/                    # GSM-like generator, GMSK burst,
│   │   ├── gsm_carrier.py          #   carrier synthesis @ 200 kHz
│   │   ├── impairments.py          #   AWGN, CFO, timing jitter, fading
│   │   ├── quantize.py             #   5-bit (configurable) I/Q quant
│   │   └── scene.py                #   multi-carrier scene composition
│   ├── data/
│   │   ├── generator.py            # parameterized dataset builder
│   │   ├── labels.py               # occupancy / power-class / IF flags
│   │   └── splits.py               # scenario-stratified train/val/test
│   ├── baselines/
│   │   ├── fft_energy.py           # STFT + per-200kHz-bin energy detect
│   │   ├── pfb_channelizer.py      # polyphase filter bank baseline
│   │   └── thresholding.py         # CFAR / Otsu / fixed thresholds
│   ├── models/
│   │   ├── mlp.py                  # MLP over spectral features
│   │   ├── cnn1d.py                # 1D CNN over I/Q windows
│   │   └── heads.py                # multi-task heads (occ / power / IF)
│   ├── train/
│   │   ├── loop.py                 # PyTorch train/val loop
│   │   └── losses.py               # BCE + CE + ordinal for power class
│   ├── eval/
│   │   ├── metrics.py              # Pd, FAR, P/R/F1, ROC, MAE-power
│   │   ├── benchmark.py            # baseline-vs-model harness
│   │   └── plots.py                # ROC, Pd-vs-SNR, confusion matrices
│   ├── complexity/
│   │   ├── counter.py              # param + MAC counter (hooks on
│   │   │                           #  Conv1d/Linear; feature-gen FLOPs)
│   │   └── footprint.py            # peak activation RAM, flash @ int8
│   └── cli/
│       ├── gen_dataset.py
│       ├── run_baseline.py
│       ├── train_model.py
│       ├── eval_all.py             # produces D5 benchmark tables/plots
│       └── demo.py                 # D7 input → occupancy/class output
├── tests/                          # pytest: signal invariants,
│                                   #  label correctness, baseline sanity
├── notebooks/                      # exploration only; not deliverables
├── reports/
│   ├── M1_experiment_plan.md       # D1
│   ├── M5_feasibility_note.md      # D6
│   └── M6_final_report.md          # D8
└── results/                        # versioned by config hash (gitignored
                                    #  raw, kept summary CSVs/PNGs)
```

## Dataset Generation Design (R2 / D2)

Target signal model (per brief §5):
- Complex baseband / IF I/Q, generator runs at 104 MS/s reference
  (configurable). Final low-bit I/Q is 5-bit signed I and 5-bit signed
  Q (configurable; sweep {4, 5, 6, 8} bits for robustness studies).
- GSM-like carrier: GMSK, 270.833 kHz symbol rate, BT≈0.3, 200 kHz
  channel spacing. Burst structure is optional; default to continuous
  modulation since the scan task is occupancy, not demodulation.
- Scan span: 5 MHz nominal (25 candidate 200 kHz channels).

Parameter axes (finalized in M1, prototype values here):
| Axis | Range / Set |
|---|---|
| Carrier count in span | {0, 1, 2, 4, 8} |
| Per-carrier SNR (dB) | {-10, -5, 0, 5, 10, 20} |
| Carrier frequency offset | uniform ±50 kHz of channel center |
| Timing / phase | uniform random |
| Adjacent-channel power | {off, equal, +6 dB, +12 dB} relative |
| Blocker / interferer | {none, CW tone, narrowband noise, chirp} |
| Quantization bits I/Q | {5} default; {4,6,8} for ablation |
| Sample window | 1024-8192 samples (≈10-80 µs) per decision |

Labels (multi-task):
- `occupancy[25]` — bool per 200 kHz channel.
- `power_class[25]` — ordinal {absent, low, mid, high} (3-4 bins).
- `if_flags` — {none, CW_blocker, broadband_blocker, adjacent_leakage}.

Splits:
- **Scenario-stratified holdouts** (mitigates RISK-2): test set contains
  blocker types and SNR points not seen in training, plus a held-out
  random seed range. Document split rule in D1.
- Train / val / test ≈ 70 / 15 / 15 by scene count, sized so total
  training corpus is ~100k-500k windows depending on model size.

Sanity tests (pytest): zero-carrier scene has near-zero in-band energy;
single-carrier scene shows expected 200 kHz PSD bump; quantization
preserves SNR to within analytic bound (6.02·N + 1.76 dB).

## Classical DSP Baseline Design (R3 / D3)

Two baselines, both required so the comparison is credible (RISK-3):

1. **FFT-energy + CFAR detector** (`baselines/fft_energy.py`):
   STFT with Hann window, length matched to channel spacing
   (FFT size N where Fs/N ≤ 200 kHz / 4, so ~512-2048 at 104 MS/s
   after decimation, or use Welch averaging). Per-200-kHz band energy
   → cell-averaging CFAR threshold for occupancy; band-power quantiles
   for power class.

2. **Polyphase channelizer + energy detector** (`baselines/pfb_channelizer.py`):
   25-channel PFB centered on the scan span, per-channel power
   integrated over the window, fixed or adaptive threshold.

Both baselines get thresholds tuned on the **validation set** (not test)
to a target FAR (e.g. 1%). Document the threshold-selection procedure
and report the operating point used for headline comparisons.

## TinyML Model Design (R4 / D4)

Compact candidates (param budget targets: ≤ 50k params, ≤ 200k MACs
per inference for an MCU-friendly cadence):

- **MLP-Spec** (`models/mlp.py`): input = log-magnitude spectrum
  (256-bin), 2 hidden layers (64, 32), multi-task heads. Cheap features,
  cheap model — strong baseline-of-baselines for ML side.
- **CNN1D-IQ** (`models/cnn1d.py`): input = quantized I/Q window
  (2 × 1024), 3-4 Conv1d blocks with stride for downsampling,
  global pooling, multi-task heads. Learns directly from low-bit
  samples — tests whether ML extracts info that FFT/PFB miss.
- (Stretch) **CNN1D-Channelizer**: input = PFB bin magnitudes over
  time, smallest model — addresses brief §16 stretch goal on
  channelizer-bin features.

Training (`train/loop.py`):
- AdamW, cosine LR, mixed multi-task loss (BCE for occupancy, CE for
  power class, BCE for IF flags) with task weights tuned on val.
- Early stopping on val macro-F1.
- Seeds logged; configs hashed into the run directory name.

## Benchmark Methodology (R5 / D5)

Identical test corpus is used for both paths. Required artifacts:
- Per-task confusion matrices (occupancy, power class, IF flags).
- ROC curves and Pd-vs-SNR curves for occupancy.
- MAE / accuracy for power class.
- Scenario-wise breakdown across the **four mandated categories**
  (nominal, low-SNR, multi-carrier/adjacent, interference). One
  consolidated table per category (D5).
- Failure-mode analysis: confusion examples + spectrograms for the
  worst false-alarm and missed-detect cases (brief §10.1 requires both
  positive and negative findings).

Single command `python -m scan_engine.cli.eval_all` regenerates every
table/plot from saved model + dataset configs.

## Complexity & Feasibility (R6 / D6) — Cortex-M Framing

`complexity/counter.py` registers forward hooks on `nn.Linear` and
`nn.Conv1d` to count MACs analytically; param count from
`model.parameters()`. Feature-generation cost (FFT N log N, PFB
M·taps, quantization) is counted separately and **must be included** —
the ML model is not "free" if its input is an expensive transform.

`complexity/footprint.py` estimates:
- Flash: params × 1 byte (assumed int8 post-training quantization).
- Peak SRAM: max(layer_in + layer_out) activation tensor × 1 byte,
  plus input buffer.
- Inference cycles: MACs × cycles-per-MAC for a Cortex-M4F (CMSIS-NN
  reference ~1-2 cycles/MAC for int8 Conv1d). Convert to µs at
  100-200 MHz; compare to the window duration to assess real-time
  feasibility.

Report this from M3 onward, not as a final-week scramble (RISK-5).

## 12-Week Schedule (M1-M6)

**Weeks 1-2 — M1: Requirements & Experiment Design (D1)**
- Mentor kickoff: confirm scan use case, signal ranges, and whether
  feasibility framing is processor / DFE-block / mixed.
- Literature scan: GSM PSD, low-bit IF receivers, RF spectrum-sensing
  ML papers (RadioML, ORACLE, DeepSig — note synthetic-vs-real domain
  shift caveat).
- Freeze: parameter grid, label schema, splits, metrics, baselines.
- Scaffolding: repo skeleton above + pytest harness + first
  `gen_dataset.py` smoke run.
- **Exit:** `reports/M1_experiment_plan.md` approved; tiny dataset
  (1k samples) flows through baseline and an untrained model.

**Weeks 3-4 — M2: Dataset Pipeline + Baseline (D2, D3)**
- Full `signals/` module with impairments + 5-bit quantization.
- Generate dataset v1 (~100k windows) covering nominal + SNR sweep
  + single blocker type.
- Implement FFT-energy + CFAR baseline; tune threshold on val.
- (If time) start PFB baseline.
- First Pd-vs-SNR plot for the baseline; sanity-check it matches
  energy-detector theory at high SNR.
- **Exit:** dataset v1 reproducible from config; baseline numbers
  on core test set committed to `results/`.

**Weeks 5-6 — M3: TinyML v1 (D4 partial)**
- Train MLP-Spec + CNN1D-IQ on dataset v1.
- Multi-task losses, weight tuning, early stopping.
- First side-by-side: model vs baseline, confusion matrices on core
  test set.
- Start complexity tracking now (RISK-5): params + MACs printed every
  run, logged into `results/`.
- **Exit:** first comparative confusion matrices + Pd-vs-SNR; pick
  one model family to take forward.

**Weeks 7-8 — M4: Robustness & Impairments**
- Dataset v2: add multi-carrier, adjacent-channel, CW + broadband
  blocker, larger CFO range, multipath (optional Rayleigh tap).
- Re-run baseline + model; expand to all four mandated scenario
  categories.
- Error analysis: which cases break each path?
- Bring PFB baseline online if not done; add second model variant
  (CNN1D-Channelizer if stretch path is taken).
- **Exit:** robustness benchmark package; clear table of where ML
  beats / matches / loses to DSP.

**Weeks 9-10 — M5: Complexity & Feasibility (D6)**
- Finalize `complexity/counter.py` and `footprint.py`.
- Int8 quantization-aware fine-tune in PyTorch; export ONNX; report
  size and accuracy delta.
- Cortex-M4F latency estimate; compare to window duration.
- If footprint is too high, prune / shrink and re-benchmark.
- **Exit:** `reports/M5_feasibility_note.md`; tuned final-candidate
  model.

**Weeks 11-12 — M6: Demo, Report, Handover (D5, D7, D8)**
- Lock final results; regenerate every figure from a single
  `eval_all` command.
- Write `reports/M6_final_report.md`: problem, methodology, results
  per scenario, complexity, limitations, recommendation.
- Build `cli/demo.py` runnable example.
- Final presentation deck.
- README + reproducibility notes + environment lock file.
- **Exit:** all of D1-D8 in repo; final recommendation answers the
  brief's decision question.

## Risks & Mitigations (mirroring brief §14, with concrete hooks)

| Risk | Mitigation in this plan |
|---|---|
| RISK-1 unrealistic dataset | M1 mentor sign-off on parameter grid; pytest sanity checks on PSD shape; ablation across quant bits. |
| RISK-2 dataset leakage | Scenario-stratified holdouts (held-out SNR points + blocker types + seed range); split rule documented in D1. |
| RISK-3 weak baseline | Two baselines (FFT-CFAR + PFB); thresholds tuned on val before any ML claim. |
| RISK-4 scope creep | Out-of-scope items pinned in README §Scope; no synchronization / demodulation code path. |
| RISK-5 late complexity | Param/MAC count printed every training run from M3; M5 is refinement not first-look. |

## Verification (How We Know It Works)

End-to-end smoke (must work by end of M2, must remain green throughout):
```
pytest tests/                                              # invariants
python -m scan_engine.cli.gen_dataset --config=experiment/smoke
python -m scan_engine.cli.run_baseline --config=experiment/smoke
python -m scan_engine.cli.train_model  --config=experiment/smoke
python -m scan_engine.cli.eval_all     --config=experiment/smoke
python -m scan_engine.cli.demo         --input=sample.npy
```

Per-milestone gates:
- **M2 sanity:** zero-carrier scene → baseline FAR ≈ configured target;
  20 dB single-carrier scene → Pd ≈ 1.0.
- **M3 gate:** ML model ≥ baseline on at least the nominal scenario,
  or a written explanation why not.
- **M4 gate:** results table covers all four mandated categories.
- **M5 gate:** model footprint fits a stated Cortex-M4F budget
  (e.g. ≤ 256 KB flash, ≤ 64 KB peak SRAM) or the gap is quantified.
- **M6 gate:** `eval_all` regenerates the entire D5 report from
  configs; final recommendation is stated.

## Critical Files (to be created)

- `configs/experiment/smoke.yaml` — drives the end-to-end smoke run.
- `src/scan_engine/signals/gsm_carrier.py` — GMSK generator; this is
  the dataset's correctness anchor.
- `src/scan_engine/data/splits.py` — implements the scenario-stratified
  holdout rule (mitigates RISK-2; reviewed by mentor in M1).
- `src/scan_engine/baselines/fft_energy.py` — the baseline ML must
  beat; tuning procedure is a deliverable, not an implementation
  detail.
- `src/scan_engine/complexity/counter.py` — must produce numbers from
  M3 onward, not M5.
- `src/scan_engine/cli/eval_all.py` — the one-command regenerator
  of D5; if this works, the project is reproducible.
- `reports/M1_experiment_plan.md`, `reports/M5_feasibility_note.md`,
  `reports/M6_final_report.md` — D1, D6, D8.

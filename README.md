# TinyML-Assisted Multi-Channel GSM Spectrum Scan Engine

R&D prototype investigating whether a compact ML model, fed low-bit
complex IF samples (reference: 5-bit I/Q at 104 MS/s, ~5 MHz scan span),
can perform GSM channel-presence detection, coarse power-class
estimation, and interference-condition flagging with quality and compute
cost competitive with classical FFT / channelizer / energy-detector
baselines.

See [`docs/PLAN.md`](docs/PLAN.md) for the full 12-week execution plan,
repo architecture, dataset / baseline / model designs, benchmark
methodology, and per-milestone verification gates.

## Status

Milestone M1 — scaffolding. The end-to-end smoke pipeline is wired but
modules are stubs.

## Quickstart

```bash
pip install -e ".[dev]"
pytest tests/
python -m scan_engine.cli.gen_dataset --config configs/experiment/smoke.yaml
python -m scan_engine.cli.run_baseline --config configs/experiment/smoke.yaml
python -m scan_engine.cli.train_model  --config configs/experiment/smoke.yaml
python -m scan_engine.cli.eval_all     --config configs/experiment/smoke.yaml
```

## Layout

```
configs/        Hydra/YAML configs (signal, dataset, baseline, model, experiment)
src/scan_engine/
  signals/      GSM-like generator, impairments, low-bit quantization
  data/         Dataset builder, labels, scenario-stratified splits
  baselines/    FFT-energy + CFAR, polyphase channelizer
  models/       MLP-Spec, CNN1D-IQ (compact, TinyML-targeted)
  train/        PyTorch training loop, multi-task losses
  eval/         Metrics, benchmark harness, plots
  complexity/   Param/MAC counters, Cortex-M footprint estimate
  cli/          gen_dataset, run_baseline, train_model, eval_all, demo
tests/          Sanity tests for signal invariants and label correctness
docs/PLAN.md    Full project plan
reports/        D1, D6, D8 markdown deliverables
results/        Per-run artifacts (raw outputs gitignored; summaries kept)
```

## Scope

In: synthetic dataset pipeline, classical DSP baseline(s), compact
TinyML model(s), benchmark across nominal / low-SNR / multi-carrier /
interference scenarios, Cortex-M-class complexity & feasibility note.

Out: full GSM modem or sync chain, production FPGA/ASIC RTL,
protocol-agnostic spectrum analyzer, large deep models incompatible
with TinyML constraints, deployment-readiness claims without baselined
validation.

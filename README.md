<h1 align="center">gradsnitch</h1>

<p align="center"><em>it tells on your training run</em></p>

<p align="center">
  <a href="https://github.com/devangpratap/gradsnitch/actions/workflows/tests.yml"><img src="https://github.com/devangpratap/gradsnitch/actions/workflows/tests.yml/badge.svg" alt="tests"></a>
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="python 3.9+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT">
  <img src="https://img.shields.io/badge/rules-GS001%E2%80%93GS010-E23D28" alt="10 rules">
</p>

---

A linter/diagnoser for ML training runs. Hook your loop with one line and get
**plain-language verdicts on *why* it broke** — not just charts.

Tracking (W&B / TensorBoard) shows you the loss curve. Snitch *reads* it and
tells you what went wrong and what to try. Every finding cites concrete evidence
and stays **silent unless the signature is unambiguous** — a wrong diagnosis is
worse than none.

```
[GS001] [ERROR] Loss became NaN/Inf
  evidence: train_loss non-finite first at step 7 (lr=0.55)
  likely:   LR too high, fp16/bf16 overflow, or a bad batch.
  try:      Lower LR, add grad clipping, check inputs, or use bf16.
```

> ⚠️ Early project, not production-hardened. Thresholds are tuned on the bundled
> real-run rigs; weird curves (RL/GAN/restarts) may still surprise it. Found a
> false positive? [That is the most useful issue you can file.](https://github.com/devangpratap/gradsnitch/issues/new?template=false_positive.md)

## Install

```bash
pip install gradsnitch            # core (numpy + pandas)
pip install "gradsnitch[torch]"   # + torch, for watch()/framework adapters
```

## Use

Raw PyTorch loop — one line:

```python
import gradsnitch

mon = gradsnitch.watch(model, optimizer, check_every=50)  # prints errors live
for step in range(steps):
    loss = loss_fn(model(x), y)
    loss.backward()
    mon.log(step, loss.item(), val_loss=val)   # grabs grad_norm + lr for you
    optimizer.step(); optimizer.zero_grad()

mon.report()                                    # verdicts at the end
```

Framework callbacks (verified against real transformers / lightning / keras):

```python
from gradsnitch import integrations
Trainer(..., callbacks=[integrations.hf()])         # HuggingFace
Trainer(callbacks=[integrations.lightning()])       # PyTorch Lightning
model.fit(..., callbacks=[integrations.keras()])    # Keras
```

Already have a run? Lint any export — column names are auto-normalized
(`train/loss`, `learning_rate`, `global_step`, … all map):

```python
gradsnitch.lint_csv("wandb_export.csv")
```

Options: `mute={"GS003"}` suppresses a rule by id; `on_alert=integrations.wandb_alert`
pushes verdicts into W&B (Slack/email) on a live run.

## What it catches today

| ID | Rule | Catches |
|----|------|---------|
| GS001 | Loss NaN/Inf | overflow / bad batch / LR too high |
| GS002 | Gradient norm inf | exploding grads (before the loss shows it) |
| GS003 | Gradient-norm spike | unstable update, precursor to a loss spike |
| GS004 | Train/val overfitting | val rises (≥3%) while train keeps falling |
| GS005 | Loss plateau | no early progress (slope-significance t-test) |
| GS006 | Loss divergence | sustained rise above the run's best |
| GS007 | Vanishing gradients | grad_norm collapses while loss stays stuck |
| GS008 | Update/weight ratio off | LR too high/low (Karpathy's ~1e-3), via `watch()` |
| GS009 | Loss oscillation | growing-amplitude swings (GAN/RL constant osc stays silent) |
| GS010 | LR schedule collapsed early | scheduler length mismatch — LR hits ~0 mid-run and the rest trains at zero |

Rule IDs are stable — suppressions and config pin to them, so they are never renumbered.

## How it's built

- **One engine, thin adapters.** Pure detector functions over a metrics
  DataFrame; `Monitor` is the only sink; each framework adapter is a ~10-line
  extractor over a shared `_feed` + `normalize()` alias table. Torch is optional
  (duck-typed); each framework imported lazily.
- **The harness is the moat.** `tests/test_real_runs.py` trains tiny *real* torch
  models that break through real mechanisms (LR 1e4→NaN, 8-pt set→overfit,
  frozen→plateau, corrupted batch→spike, deep-sigmoid→vanishing, tiny/big LR→update
  ratio off, half-length LR schedule→dead second half) **plus negative rigs** that must stay silent (converged val, terminal
  spike, short noisy learner, momentum→decaying oscillation, a correct full-length decay). 25 rigs; correctness
  here is emergent across adapters, so this is where the value lives.

## What could be added (roadmap, not done)

- **More detectors:** dead-ReLU / saturation from activation hooks (the metrics-only
  cousin — grad_norm collapse — already ships as GS007).
- **v2 flagship — Cockpit's Alpha (α):** a principled "LR too high/low" verdict
  from the loss curvature along the update direction. Needs per-sample grads, so
  it's intrusive (breaks log-only/torch-optional) — a separate opt-in mode.
- **Gradient-noise / batch-size test** (Cockpit/McCandlish).
- **Rule-catalog docs page** + richer `mute`/config (Cleanlab-style).
- **Run history** — compare a run to your last N (Aim-style store).

Adding a rule or an adapter: [CONTRIBUTING.md](CONTRIBUTING.md). Design notes: [PLAN.md](PLAN.md).

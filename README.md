<h1 align="center">gradsnitch</h1>

<p align="center"><em>it tells on your training run</em></p>

<p align="center">
  <a href="https://pypi.org/project/gradsnitch/"><img src="https://img.shields.io/pypi/v/gradsnitch?color=E23D28" alt="PyPI"></a>
  <a href="https://github.com/devangpratap/gradsnitch/actions/workflows/tests.yml"><img src="https://github.com/devangpratap/gradsnitch/actions/workflows/tests.yml/badge.svg" alt="tests"></a>
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="python 3.9+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT">
  <img src="https://img.shields.io/badge/rules-GS001%E2%80%93GS010-E23D28" alt="10 rules">
</p>

---

Your loss went to NaN. Your tracker drew you a chart of it going to NaN.

gradsnitch reads the curve and tells you **why**, in one line, while the run is
still going:

```
$ python train.py

[snitch] [GS001] [ERROR] Loss became NaN/Inf
  evidence: train_loss non-finite first at step 7 (lr=0.55)
  likely:   LR too high, fp16/bf16 overflow, or a bad batch.
  try:      Lower LR, add grad clipping, check inputs for NaNs, or use bf16 instead of fp16.
```

That's a real 400-step run, called at step 7 — before the other 393 finished
burning. Every finding cites its evidence and stays **silent unless the
signature is unambiguous**, because a wrong diagnosis is worse than none.

## Install

```bash
pip install gradsnitch            # core (numpy + pandas)
pip install "gradsnitch[torch]"   # + torch, for watch()/framework adapters
```

## Use

**Already have a run?** Lint any export — no code changes, column names are
auto-normalized (`train/loss`, `learning_rate`, `global_step`, … all map):

```python
import gradsnitch

for finding in gradsnitch.lint_csv("wandb_export.csv"):
    print(finding)
```

**Raw PyTorch loop** — one line:

```python
mon = gradsnitch.watch(model, optimizer, check_every=50)  # prints errors live
for step in range(steps):
    loss = loss_fn(model(x), y)
    loss.backward()
    mon.log(step, loss.item(), val_loss=val)   # grabs grad_norm + lr for you
    optimizer.step(); optimizer.zero_grad()

mon.report()                                    # verdicts at the end
```

**Framework callbacks** — verified against real transformers / lightning / keras:

```python
from gradsnitch import integrations
Trainer(..., callbacks=[integrations.hf()])         # HuggingFace
Trainer(callbacks=[integrations.lightning()])       # PyTorch Lightning
model.fit(..., callbacks=[integrations.keras()])    # Keras
```

`mute={"GS003"}` suppresses a rule by id. `on_alert=integrations.wandb_alert`
pushes verdicts into W&B (Slack/email) on a live run.

## What it catches

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
| GS010 | LR schedule collapsed early | scheduler length mismatch — LR hits ~0 mid-run, rest trains at zero |

Rule IDs are stable — suppressions pin to them, so they are never renumbered.

## How it's built

**The harness is the moat.** `tests/test_real_runs.py` trains tiny *real* torch
models that break through real mechanisms — LR 1e4→NaN, 8-point set→overfit,
frozen→plateau, corrupted batch→spike, deep-sigmoid→vanishing, half-length LR
schedule→dead second half — plus negative rigs that must stay silent: converged
val, terminal spike, short noisy learner, momentum→decaying oscillation, a
correct full-length decay. 25 rigs. Correctness is emergent across adapters, so
this is where the value lives.

One engine, thin adapters: pure detector functions over a metrics DataFrame,
`Monitor` as the only sink, each framework adapter a ~10-line extractor over a
shared `_feed` + `normalize()` alias table. Torch is optional (duck-typed).
Design notes in [PLAN.md](PLAN.md).

## Caveats

Early project, not production-hardened. Thresholds are tuned on the bundled
real-run rigs, so weird curves (RL/GAN/restarts) may still surprise it.

Found a false positive? [That is the most useful issue you can file.](https://github.com/devangpratap/gradsnitch/issues/new?template=false_positive.md)
A missed diagnosis is [the second most useful](https://github.com/devangpratap/gradsnitch/issues/new?template=missed_diagnosis.md).

Adding a rule or an adapter: [CONTRIBUTING.md](CONTRIBUTING.md). Next up —
dead-ReLU/saturation hooks, Cockpit's α curvature verdict, gradient-noise
batch-size test, run-to-run history.

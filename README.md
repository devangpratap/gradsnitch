# Snitch — it tells on your training run

`gradsnitch` is a linter/diagnoser for ML training runs. Hook your loop with one
line and get **plain-language verdicts on *why* it broke** — not just charts.

Tracking (W&B / TensorBoard) shows you the loss curve. Snitch *reads* it and
tells you what went wrong and what to try. Every finding cites concrete evidence
and stays **silent unless the signature is unambiguous** — a wrong diagnosis is
worse than none.

> ⚠️ Early/personal project, not production-hardened. Thresholds are tuned on the
> bundled real-run rigs; weird curves (RL/GAN/restarts) may still surprise it.

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

Rule IDs are stable (suppress/config pin to them). Example verdict:

```
[GS001] [ERROR] Loss became NaN/Inf
  evidence: train_loss non-finite first at step 7 (lr=0.55)
  likely:   LR too high, fp16/bf16 overflow, or a bad batch.
  try:      Lower LR, add grad clipping, check inputs, or use bf16.
```

## How it's built

- **One engine, thin adapters.** Pure detector functions over a metrics
  DataFrame; `Monitor` is the only sink; each framework adapter is a ~10-line
  extractor over a shared `_feed` + `normalize()` alias table. Torch is optional
  (duck-typed); each framework imported lazily.
- **The harness is the moat.** `tests/test_real_runs.py` trains tiny *real* torch
  models that break through real mechanisms (LR 1e4→NaN, 8-pt set→overfit,
  frozen→plateau, corrupted batch→spike) **plus negative rigs** that must stay
  silent (converged val, terminal spike, short noisy learner). 19 rigs; correctness
  here is emergent across adapters, so this is where the value lives.

## What could be added (roadmap, not done)

- **More detectors:** loss-oscillation (RL/GAN — ships last, needs a strict gate +
  GAN negative rig), update-to-weight ratio (Karpathy's 1e-3 rule, via `watch()`),
  dead-ReLU / saturation (needs activation hooks).
- **v2 flagship — Cockpit's Alpha (α):** a principled "LR too high/low" verdict
  from the loss curvature along the update direction. Needs per-sample grads, so
  it's intrusive (breaks log-only/torch-optional) — a separate opt-in mode.
- **Gradient-noise / batch-size test** (Cockpit/McCandlish).
- **Rule-catalog docs page** + richer `mute`/config (Cleanlab-style).
- **Run history** — compare a run to your last N (Aim-style store).
- **Packaging** — pip-installable, `tests/` for pytest.

Design notes in [PLAN.md](PLAN.md).

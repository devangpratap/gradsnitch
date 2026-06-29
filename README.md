# Snitch — it tells on your training run

`gradsnitch` is a linter/diagnoser for ML training runs. Hook your loop with one
line and get **plain-language verdicts on *why* it broke** — not just charts.

Tracking (W&B / TensorBoard) shows you the loss curve. Snitch *reads* it and
tells you what went wrong and what to try.

## Use

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

Already have a run? Lint the export instead:

```python
gradsnitch.lint_csv("wandb_export.csv")   # columns: step, train_loss, grad_norm, lr, val_loss
```

## What it catches today

Loss NaN/Inf · gradient-norm spikes · loss divergence · plateau / not-learning ·
train/val overfitting.

Every finding cites concrete evidence and stays **silent unless the signature is
unambiguous** — a wrong diagnosis is worse than none.

## Example

```
[ERROR] Loss became NaN/Inf
  evidence: train_loss non-finite first at step 7 (lr=0.55)
  likely:   LR too high, fp16/bf16 overflow, or a bad batch.
  try:      Lower LR, add grad clipping, check inputs, or use bf16.
```

Roadmap in [PLAN.md](PLAN.md).

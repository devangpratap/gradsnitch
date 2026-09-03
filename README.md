<h1 align="center">gradsnitch</h1>

<p align="center"><em>it tells on your training run</em></p>

<p align="center">
  <a href="https://pypi.org/project/gradsnitch/"><img src="https://img.shields.io/pypi/v/gradsnitch?color=E23D28" alt="PyPI"></a>
  <a href="https://github.com/devangpratap/gradsnitch/actions/workflows/tests.yml"><img src="https://github.com/devangpratap/gradsnitch/actions/workflows/tests.yml/badge.svg" alt="tests"></a>
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT">
</p>

---

Your loss went to NaN. Your tracker drew you a chart of it going to NaN.
gradsnitch reads the curve and tells you **why**, while the run is still going:

```
[snitch] [GS001] [ERROR] Loss became NaN/Inf
  evidence: train_loss non-finite first at step 7 (lr=0.55)
  likely:   LR too high, fp16/bf16 overflow, or a bad batch.
  try:      Lower LR, add grad clipping, check inputs for NaNs, or use bf16.
```

Real 400-step run, called at step 7. Silent unless the signature is unambiguous —
a wrong diagnosis is worse than none.

```bash
pip install gradsnitch
```

```python
import gradsnitch

# lint a finished run, no code changes (column names auto-normalize)
for finding in gradsnitch.lint_csv("wandb_export.csv"):
    print(finding)

# or watch a live loop — one line
mon = gradsnitch.watch(model, optimizer)
mon.log(step, loss.item())     # after loss.backward()
mon.report()
```

HuggingFace, Lightning and Keras get a callback: `integrations.hf()`,
`.lightning()`, `.keras()`. `pip install "gradsnitch[torch]"` for those.

## Rules

| | | |
|----|------|---------|
| GS001 | Loss NaN/Inf | overflow / bad batch / LR too high |
| GS002 | Gradient norm inf | exploding grads, before the loss shows it |
| GS003 | Gradient-norm spike | unstable update, precursor to a loss spike |
| GS004 | Train/val overfitting | val rises while train keeps falling |
| GS005 | Loss plateau | no early progress (slope t-test) |
| GS006 | Loss divergence | sustained rise above the run's best |
| GS007 | Vanishing gradients | grad_norm collapses, loss stuck |
| GS008 | Update/weight ratio off | LR too high/low (Karpathy's ~1e-3) |
| GS009 | Loss oscillation | growing swings (constant GAN/RL osc stays silent) |
| GS010 | LR schedule collapsed | LR hits ~0 mid-run, rest trains at zero |

`mute={"GS003"}` to suppress one. IDs are stable, never renumbered.

## Caveats

Early project. Thresholds are tuned on 25 real-run test rigs, so weird curves
(RL/GAN/restarts) may surprise it. [False positives](https://github.com/devangpratap/gradsnitch/issues/new?template=false_positive.md)
are the most useful issue you can file.

[CONTRIBUTING.md](CONTRIBUTING.md) · [PLAN.md](PLAN.md)

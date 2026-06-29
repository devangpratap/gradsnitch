"""Real-run validation harness — the test that actually has signal.

The self-check in gradsnitch.py runs on synthetic CSVs *shaped to trigger* each
detector: it proves the wiring, not the diagnosis. That's nearly circular.

This harness trains real (tiny) torch models that break through real mechanisms
— LR too high actually overflows to NaN; a near-zero LR actually plateaus; a
big model on 8 points actually memorizes — and captures grad_norm/lr/loss
through the real watch() path. Then it asserts each detector's verdict against
the *known* cause, and asserts a sane run stays silent.

So it does three jobs at once:
  1. tests detectors against real dynamics (not authored shapes),
  2. tunes the thresholds (a rig that won't trip means the threshold is wrong),
  3. is the proof the tool works on real runs.

Functions are named test_* so pytest discovers them for free at packaging time;
runnable now as `python3 tests/test_real_runs.py` with no pytest installed.
Tiny CPU runs (seconds) — deterministic seeds, no GPU.
"""

from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import gradsnitch


def _data(n: int, seed: int = 0):
    """Real regression data: y = Wx + noise. Learnable, so a sane run converges."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 16, generator=g)
    w = torch.randn(16, 1, generator=g)
    y = x @ w + 0.1 * torch.randn(n, 1, generator=g)
    return x, y


def _mlp(hidden: int = 64):
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Linear(16, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1)
    )


def _train(
    *,
    lr,  # float, or callable(step)->lr for a bad schedule
    n_train: int = 512,
    hidden: int = 64,
    steps: int = 400,
    with_val: bool = False,
    spike_at: int | None = None,  # step to inject one corrupted (large-input) batch
    full_batch: bool = False,  # no minibatch noise — for a true flat-line plateau
) -> gradsnitch.Monitor:
    """One real training loop. The failure (if any) emerges from real dynamics."""
    x, y = _data(n_train, seed=0)
    xv, yv = _data(128, seed=1) if with_val else (None, None)
    model = _mlp(hidden)
    opt = torch.optim.SGD(model.parameters(), lr=lr(0) if callable(lr) else lr)
    lossf = torch.nn.MSELoss()
    mon = gradsnitch.watch(model, opt)

    bs = n_train if full_batch else min(64, n_train)
    for step in range(steps):
        if callable(lr):
            for pg in opt.param_groups:
                pg["lr"] = lr(step)
        idx = torch.randint(0, n_train, (bs,))
        xb = x[idx] * (
            100.0 if step == spike_at else 1.0
        )  # corrupted batch = real grad spike
        opt.zero_grad()
        loss = lossf(model(xb), y[idx])
        loss.backward()
        val = None
        if with_val:
            with torch.no_grad():
                val = lossf(model(xv), yv).item()
        mon.log(step, loss.item(), val_loss=val)
        opt.step()
    return mon


def _has(findings, substr: str) -> bool:
    return any(substr.lower() in f.name.lower() for f in findings)


# --- the rigs: each breaks for one real reason, asserts the matching verdict ---


def test_clean_run_is_silent():
    """Sane LR + enough data → converges. The false-positive guard that matters."""
    findings = _train(lr=0.05).report()
    assert findings == [], f"false positive on a healthy real run: {findings}"


def test_nan_from_exploding_lr():
    """LR ~1e4 → updates really overflow to NaN/Inf within a few steps."""
    findings = _train(lr=1e4, steps=60).report()
    assert _has(findings, "nan"), f"missed real NaN blowup: {findings}"


def test_divergence_from_lr_bump():
    """Bad schedule: small LR, then bumped high at the midpoint → loss really climbs."""
    half = 150
    sched = lambda s: (
        0.05 if s < half else 0.2
    )  # 0.2 climbs but stays finite; >=0.3 overflows to NaN
    findings = _train(lr=sched, steps=300).report()
    assert _has(findings, "diverg"), f"missed real divergence: {findings}"
    assert not _has(findings, "nan"), (
        f"diverge rig overflowed — make LR bump gentler: {findings}"
    )


def test_plateau_from_tiny_lr():
    """Near-zero LR → model really can't move; loss is flat. Noisy minibatches on
    purpose: detect_loss_plateau uses a slope-significance (t) test, so zero-mean
    minibatch noise averages out of the trend and a frozen run still reads as flat."""
    findings = _train(lr=1e-7, steps=300).report()
    assert _has(findings, "plateau"), f"missed real plateau: {findings}"


def test_overfit_train_down_val_up():
    """Long training → val improves then U-turns up (~+32%) while train keeps falling.
    The real overfit signature — not just val failing to learn (which isn't overfitting)."""
    findings = _train(lr=0.02, n_train=1024, steps=300, with_val=True).report()
    assert _has(findings, "overfit"), f"missed real overfitting: {findings}"


def test_spike_from_corrupted_batch():
    """One batch with 100x inputs at step 200 → real gradient spike through real backprop."""
    findings = _train(lr=0.02, steps=400, spike_at=200).report()
    assert _has(findings, "spike"), f"missed real grad spike: {findings}"


# --- negative rigs: tricky-but-healthy runs that must stay silent (the FP guards) ---


def test_converged_noisy_val_not_overfit():
    """Val improves then settles on a flat noisy floor (~+2% above its min) → argmin
    lands interior by noise. Must NOT read as overfitting — val never meaningfully rose."""
    findings = _train(
        lr=0.02, n_train=64, hidden=256, steps=2000, with_val=True
    ).report()
    assert not _has(findings, "overfit"), (
        f"false overfitting on converged val: {findings}"
    )


def test_terminal_spike_not_divergence():
    """A corrupted batch near the end spikes the loss briefly, then it's over.
    Grad spike is fair game; sustained divergence is NOT — it didn't sustain."""
    findings = _train(lr=0.02, steps=400, spike_at=380).report()
    assert not _has(findings, "diverg"), (
        f"false divergence on terminal spike: {findings}"
    )


def test_short_noisy_learner_not_plateau():
    """60 steps, noisy minibatches, but genuinely descending → real (slow) progress.
    Must NOT read as a plateau just because the t-test is underpowered on short runs."""
    findings = _train(lr=0.05, steps=60).report()
    assert not _has(findings, "plateau"), (
        f"false plateau on short noisy learner: {findings}"
    )


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok   {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} real-run rigs passed")
    sys.exit(1 if failed else 0)

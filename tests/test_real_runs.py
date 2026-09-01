"""Real-run validation harness — the test that actually has signal.

The self-check in gradsnitch/__init__.py runs on synthetic CSVs *shaped to trigger* each
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
import numpy as np
import pandas as pd
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

    def sched(s):  # 0.2 climbs but stays finite; >=0.3 overflows to NaN
        return 0.05 if s < half else 0.2

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


def test_vanishing_grad_from_deep_saturated_net():
    """Deep sigmoid stack: grads start healthy, then the sigmoids drift into
    saturation during training, so grad_norm really collapses (~0.4 -> 2e-4) while
    the loss can't move. Progressive vanishing — the real GS007 signature."""
    torch.manual_seed(0)
    layers = []
    for _ in range(6):  # deep enough that saturation kills the gradient through it
        layers += [torch.nn.Linear(16, 16), torch.nn.Sigmoid()]
    model = torch.nn.Sequential(*layers, torch.nn.Linear(16, 1))
    x, y = _data(512, seed=0)
    opt = torch.optim.SGD(model.parameters(), lr=0.2)
    lossf = torch.nn.MSELoss()
    mon = gradsnitch.watch(model, opt)
    for step in range(400):
        opt.zero_grad()
        loss = lossf(model(x), y)
        loss.backward()
        mon.log(step, loss.item())
        opt.step()
    findings = mon.report()
    assert _has(findings, "vanish"), f"missed real vanishing gradients: {findings}"


def test_low_lr_shows_in_update_ratio():
    """LR 1e-5: real ||ΔW||/||W|| lands ~1e-5, far below the healthy ~1e-3. The
    update-ratio verdict names the cause plateau (GS005) only hints at."""
    findings = _train(lr=1e-5, steps=300).report()
    assert _has(findings, "lr too low"), f"missed real low-LR ratio: {findings}"


def test_high_lr_shows_in_update_ratio():
    """LR 0.15: updates run ~1e-2 of the weights (too big) but the run stays finite,
    so NaN/spike detectors don't fire — the update ratio is what catches it."""
    findings = _train(lr=0.15, steps=300).report()
    assert _has(findings, "lr too high"), f"missed real high-LR ratio: {findings}"


# --- negative rigs: tricky-but-healthy runs that must stay silent (the FP guards) ---


def test_decaying_oscillation_not_flagged():
    """SGD momentum 0.99 oscillates hard early (200+ smoothed reversals) but the
    amplitude DECAYS as it settles into the minimum. That is healthy convergence, not
    instability — GS009's growing-amplitude gate must keep it silent."""
    torch.manual_seed(0)
    model = _mlp(64)
    x, y = _data(512, seed=0)
    opt = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.99)
    lossf = torch.nn.MSELoss()
    mon = gradsnitch.watch(model, opt)
    for step in range(400):
        opt.zero_grad()
        loss = lossf(model(x), y)
        loss.backward()
        mon.log(step, loss.item())
        opt.step()
    findings = mon.report()
    assert not _has(findings, "oscillation"), (
        f"false oscillation on a decaying/settling run: {findings}"
    )


def test_smooth_then_noisy_not_flagged_as_oscillation():
    """A CSV/W&B import whose early phase is a near-perfect decay: detrending it leaves
    ~1e-15 of float noise, so any ordinary late minibatch jitter used to read as
    astronomical amplitude growth and produce a bogus "LR too high" verdict."""
    n = 200
    loss = np.concatenate(
        [
            np.linspace(5, 1, n // 2),
            1 + np.random.RandomState(0).normal(0, 0.01, n // 2),
        ]
    )
    findings = gradsnitch.lint(pd.DataFrame({"step": range(n), "train_loss": loss}))
    assert not _has(findings, "oscillation"), (
        f"float noise in a flat segment read as a growing swing: {findings}"
    )


def test_early_lr_collapse_from_short_schedule():
    """The classic scheduler-length bug: num_training_steps set to half the run, so
    the LR linearly decays to 0 at step 200 and the last 200 steps do nothing."""
    total = 200  # WRONG: the real run is 400 steps
    mon = _train(lr=lambda s: 3e-2 * max(0.0, 1 - s / total), steps=400)
    f = mon.report()
    assert _has(f, "schedule"), "missed a schedule that died halfway"


def test_full_length_decay_not_flagged():
    """Same schedule, correct length: decays to 0 exactly at the end. By design."""
    mon = _train(lr=lambda s: 3e-2 * max(0.0, 1 - s / 400), steps=400)
    assert not _has(mon.report(), "schedule"), "false positive on a correct schedule"


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


# --- regression rigs for messy real exports (the lint_csv / ordering bugs) ---


def test_unsorted_export_is_sorted_first():
    """Real exports can arrive step-shuffled (async/resume/multi-worker). Detectors
    assume ascending order, so lint() must sort — else evidence and even which
    findings fire go wrong."""
    import gradsnitch as gs

    df = gs._synthetic("nan")  # NaN actually starts at step 250
    shuf = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    assert [f.name for f in gs.lint(df)] == [f.name for f in gs.lint(shuf)], (
        "shuffling the rows changed the diagnosis"
    )
    nan = next(f for f in gs.lint(shuf) if "NaN" in f.name)
    assert "step 250" in nan.evidence, f"wrong step after shuffle: {nan.evidence}"


def test_malformed_csv_raises_not_silent():
    """Wrong columns must error loudly — a diagnoser answering 'all clear' on bad
    input is the worst possible failure."""
    import tempfile
    import os
    import gradsnitch as gs

    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w") as f:
        f.write("step,accuracy\n0,0.5\n1,0.6\n")  # no loss column under any alias
    try:
        raised = False
        try:
            gs.lint_csv(path)
        except ValueError:
            raised = True
        assert raised, "lint_csv silently returned on a CSV missing train_loss"
    finally:
        os.unlink(path)


def test_aliased_columns_are_normalized():
    """Real exports name things differently (loss, learning_rate, eval_loss,
    global_step). normalize() must map them so detectors work with no user renaming."""
    import gradsnitch as gs

    df = gs._synthetic("nan").rename(
        columns={
            "train_loss": "loss",
            "lr": "learning_rate",
            "val_loss": "eval_loss",
            "step": "global_step",
            "grad_norm": "train/grad_norm",
        }
    )
    assert _has(gs.lint(df), "nan"), "aliased export columns were not normalized"
    assert gs.normalize({"loss": 1.0, "learning_rate": 3e-4})["train_loss"] == 1.0, (
        "dict normalize broken (framework adapters depend on it)"
    )
    # W&B slash-style keys must map too
    wb = gs._synthetic("nan").rename(
        columns={
            "train_loss": "train/loss",
            "lr": "train/learning_rate",
            "step": "trainer/global_step",
        }
    )
    assert _has(gs.lint(wb), "nan"), "W&B slash-style keys not normalized"


def test_mute_suppresses_by_rule_id():
    """mute={GS00x} drops that rule's findings — the config use the stable IDs exist for."""
    import gradsnitch as gs

    df = gs._synthetic("nan")
    assert _has(gs.lint(df), "nan"), "baseline should flag the NaN"
    assert gs.lint(df, mute={"GS001"}) == [], "mute did not suppress GS001"


def test_on_alert_hook_fires_for_live_errors():
    """watch(on_alert=cb) calls cb(finding) for each new live error — the plumbing the
    wandb_alert integration rides on (verified without wandb)."""
    import gradsnitch as gs

    seen = []
    mon = gs.watch(check_every=5, on_alert=seen.append)
    for r in gs._synthetic("nan").to_dict("records"):
        mon.log(r["step"], r["train_loss"], grad_norm=r["grad_norm"], lr=r["lr"])
    assert any(f.code == "GS001" for f in seen), "on_alert never fired on a live error"


def test_inf_grad_norm_is_flagged():
    """A real HF run at high LR reports grad_norm=inf while loss is still finite (grads
    overflow first). That unambiguous blow-up must not slip through silently."""
    import gradsnitch as gs
    import numpy as np

    df = gs._synthetic("clean").copy()
    df.loc[150:, "grad_norm"] = np.inf
    assert _has(gs.lint(df), "inf"), "inf grad_norm went unflagged"


def test_huge_flat_loss_not_misdiagnosed_as_plateau():
    """Catastrophic LR can pin loss flat at ~1e26 with exploding grads. Must NOT read as
    'plateau / LR too low' — the opposite advice (the worst kind of wrong diagnosis)."""
    import gradsnitch as gs
    import pandas as pd
    import numpy as np

    n = 400
    df = pd.DataFrame(
        {
            "step": np.arange(n),
            "train_loss": np.full(n, 1e26),
            "grad_norm": np.full(n, np.inf),
            "lr": 5e7,
        }
    )
    findings = gs.lint(df)
    assert not _has(findings, "plateau"), f"blow-up misdiagnosed as plateau: {findings}"
    assert _has(findings, "inf"), "should flag the inf grad instead"


def test_partial_grad_norm_is_not_a_blowup():
    """Lightning emits grad_norm optionally (degrade / grad-accum warmup), so a healthy
    run has some rows with grad_norm missing (NaN). Absent != exploded — must stay
    silent (no false GS002, and plateau not suppressed run-wide)."""
    import gradsnitch as gs
    import numpy as np

    df = gs._synthetic("clean").copy()
    df.loc[[5, 17, 33], "grad_norm"] = (
        np.nan
    )  # framework just didn't log it those steps
    assert gs.lint(df) == [], f"absent grad_norm misread as a failure: {gs.lint(df)}"


def test_loss_only_csv_diagnoses_not_crash():
    """A bare `loss` column (no step/grad) containing a NaN must be diagnosed, not
    crash on a missing 'step' column."""
    import tempfile
    import os
    import gradsnitch as gs

    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w") as f:
        f.write("loss\n1.0\n0.5\n0.4\nnan\n")
    try:
        assert _has(gs.lint_csv(path), "nan"), "loss-only NaN not diagnosed"
    finally:
        os.unlink(path)


def test_shared_feed_handles_framework_streams():
    """The shared _feed (used by hf/lightning/keras adapters) turns framework-style
    metric dicts into Monitor rows — step supplied by the adapter, val carried
    forward — and surfaces a real failure. Verified without any framework installed."""
    from gradsnitch.integrations import _feed
    import gradsnitch as gs

    mon = gs.watch()
    pending = [None]
    for step in range(300):  # steps jump by 5 (logging_steps) — adapter supplies step
        loss = float("nan") if step >= 200 else 1.0 / (step + 1)
        # mix of alias spellings the three frameworks use
        _feed(
            mon,
            step * 5,
            {"loss": loss, "grad_norm": 1.0, "learning_rate": 3e-4},
            pending,
        )
    assert _has(gs.lint(mon.df()), "nan"), "adapter feed didn't surface the NaN"
    _feed(
        mon, 9999, {"eval_loss": 0.5}, pending
    )  # eval-only: no train row, val stashed
    assert pending[0] == 0.5, "eval-only call should carry val forward, not log a row"
    # Lightning-style (val_loss key) and Keras-style (val_loss at epoch) both normalize
    _feed(mon, 10000, {"train_loss": 0.4, "val_loss": 0.6}, pending)
    assert mon.df().iloc[-1]["val_loss"] == 0.6, "val not attached to the train row"


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

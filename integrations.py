"""Framework adapters: thin shims that feed metrics into a gradsnitch Monitor.

Each lazily imports its own framework, so none is a hard dependency and the base
module stays torch-optional. The reusable unit is a tiny per-framework feed (step
derivation + type coercion + name mapping via the shared normalize()); Monitor
stays the single sink for dedup / live alerts / report().
"""

from gradsnitch import watch, normalize


def _hf_feed(mon, global_step, logs, pending):
    """Core HF on_log logic — testable without transformers installed.

    HF fires on_log separately for train (loss/grad_norm/learning_rate) and eval
    (eval_loss); the step lives on state.global_step, NOT in logs. So we read step
    from the caller and carry the latest val forward, attaching it to the next
    train row so val aligns with a real step (and no eval-only NaN row is created).
    """
    m = normalize(dict(logs))
    if "val_loss" in m:
        pending[0] = m["val_loss"]
    if "train_loss" in m:
        extra = {k: m[k] for k in ("grad_norm", "lr") if k in m}
        mon.log(int(global_step), m["train_loss"], val_loss=pending[0], **extra)
        pending[0] = None


def hf(check_every: int = 0):
    """HuggingFace Trainer callback: `Trainer(..., callbacks=[integrations.hf()])`.

    Prints error verdicts live every `check_every` logs; full report at train end.
    """
    from transformers import TrainerCallback  # lazy: optional dependency

    mon = watch(check_every=check_every)  # model=None; metrics come from logs
    pending = [None]

    class SnitchCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs:
                _hf_feed(mon, state.global_step, logs, pending)

        def on_train_end(self, args, state, control, **kw):
            mon.report()

    cb = SnitchCallback()
    cb.monitor = mon  # expose for inspection / a manual .report()
    return cb

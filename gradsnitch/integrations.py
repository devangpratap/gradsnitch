"""Framework adapters: thin shims that feed metrics into a gradsnitch Monitor.

Each lazily imports its own framework, so none is a hard dependency and the base
module stays torch-optional. The reusable unit is the shared `_feed` (name
mapping via normalize() + carry-val-forward + one Monitor row); every adapter
just does its framework-specific *extraction* (where step / metrics / grad_norm
live) and calls `_feed`. Monitor stays the single sink for dedup / live / report.

Every factory (hf/lightning/keras) forwards **watch_kw to watch(), so you can pass
`mute={"GS003"}` or `on_alert=integrations.wandb_alert` to any of them.
"""

from gradsnitch import watch, normalize


def _feed(mon, step, metrics, pending):
    """Shared feed. `metrics` is a plain name->number dict; adapters supply it.

    Frameworks log train (loss/grad_norm/lr) and eval (val_loss) at different
    times, so carry the latest val forward onto the next train row — keeps val
    aligned to a real step and never creates an eval-only NaN row.
    """
    m = normalize(dict(metrics))
    if "val_loss" in m:
        pending[0] = m["val_loss"]
    if "train_loss" in m:
        extra = {k: m[k] for k in ("grad_norm", "lr") if k in m}
        mon.log(int(step), m["train_loss"], val_loss=pending[0], **extra)
        pending[0] = None


def wandb_alert(finding):
    """on_alert hook: push a Snitch verdict into W&B (Slack/email) when a run is live.

    Use: `integrations.hf(on_alert=integrations.wandb_alert)`. wandb is imported
    lazily, so this costs nothing unless you opt in.
    """
    import wandb  # lazy: optional dependency

    if getattr(wandb, "run", None) is not None:
        level = (
            wandb.AlertLevel.ERROR
            if finding.severity == "error"
            else wandb.AlertLevel.WARN
        )
        wandb.alert(
            title=f"[{finding.code}] {finding.name}", text=str(finding), level=level
        )


def hf(check_every: int = 0, **watch_kw):
    """HuggingFace Trainer callback: `Trainer(..., callbacks=[integrations.hf()])`.

    HF's on_log gives loss/grad_norm/learning_rate free; step is on state.global_step.
    """
    from transformers import TrainerCallback  # lazy: optional dependency

    mon = watch(check_every=check_every, **watch_kw)
    pending = [None]

    class SnitchCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs:
                _feed(mon, state.global_step, logs, pending)

        def on_train_end(self, args, state, control, **kw):
            mon.report()

    cb = SnitchCallback()
    cb.monitor = mon
    return cb


def lightning(check_every: int = 0, **watch_kw):
    """PyTorch Lightning callback: `Trainer(callbacks=[integrations.lightning()])`.

    Lightning metrics live in `trainer.callback_metrics` as tensors (coerce to
    float). grad_norm is NOT free — compute it in on_before_optimizer_step via the
    utility and stash it for the next batch end; if unavailable, degrade to
    loss-only (detectors self-silence).
    """
    try:
        from lightning.pytorch import Callback
        from lightning.pytorch.utilities import grad_norm as _ln_grad_norm
    except ImportError:  # older package name
        from pytorch_lightning import Callback
        from pytorch_lightning.utilities import grad_norm as _ln_grad_norm

    mon = watch(check_every=check_every, **watch_kw)
    pending = [None]
    stash = {"grad_norm": None}

    class SnitchCallback(Callback):
        def on_before_optimizer_step(self, trainer, pl_module, optimizer, *a):
            try:
                norms = _ln_grad_norm(pl_module, norm_type=2)
                stash["grad_norm"] = float(norms.get("grad_2.0_norm_total"))
            except Exception:
                stash["grad_norm"] = None  # degrade to loss-only

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            metrics = {}
            for k, v in trainer.callback_metrics.items():
                try:
                    metrics[k] = float(v)
                except (TypeError, ValueError):
                    pass  # skip non-scalar logged metrics rather than crash the callback
            if stash["grad_norm"] is not None:
                metrics["grad_norm"] = stash["grad_norm"]
            _feed(mon, trainer.global_step, metrics, pending)

        def on_train_end(self, trainer, pl_module):
            mon.report()

    cb = SnitchCallback()
    cb.monitor = mon
    return cb


def _keras_lr(model):
    try:
        lr = model.optimizer.learning_rate
        return float(lr)
    except Exception:
        return None  # LR schedules / backends that don't float cleanly -> skip


def keras(check_every: int = 0, **watch_kw):
    """Keras callback: `model.fit(..., callbacks=[integrations.keras()])`.

    Keras gives loss (and val_loss at epoch end) but no grad_norm; lr comes off
    the optimizer. `batch` resets every epoch, so the canonical step is
    epoch*steps_per_epoch + batch (monotonic).
    """
    import keras  # lazy: optional dependency

    mon = watch(check_every=check_every, **watch_kw)
    pending = [None]
    st = {"epoch": 0, "spe": 0}

    class SnitchCallback(keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            st["epoch"] = epoch

        def on_train_batch_end(self, batch, logs=None):
            step = st["epoch"] * st["spe"] + batch
            st["spe"] = max(st["spe"], batch + 1)
            metrics = dict(logs or {})
            lr = _keras_lr(self.model)
            if lr is not None:
                metrics["lr"] = lr
            _feed(mon, step, metrics, pending)

        def on_epoch_end(self, epoch, logs=None):
            # val-ONLY on purpose: _feed makes no row without train_loss, so this never
            # collides with next epoch's batch-0 step (which numerically equals this step).
            # Do NOT add train_loss here or you create a duplicate-step row.
            val = {k: v for k, v in (logs or {}).items() if k.startswith("val")}
            if val:  # surface val so it's carried onto the next train row
                _feed(mon, (epoch + 1) * st["spe"], val, pending)

        def on_train_end(self, logs=None):
            mon.report()

    cb = SnitchCallback()
    cb.monitor = mon
    return cb

"""snitch — it tells on your training run.

Feed it a log of per-step metrics and it tells you what broke and what to try.
Tracking (W&B/TensorBoard) shows you the curve; this reads it.

ponytail: v1 only uses metrics already in a standard export
(step, train_loss, grad_norm, lr, val_loss). Detectors needing extra
instrumentation (dead units, spectral/rank stats) come later, one rule at a time.

Trust rule: a wrong diagnosis is worse than none. Every Finding cites concrete
evidence and hedges ("likely"); detectors return None unless the signature is
unambiguous. The self-check asserts no findings on a clean run.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Finding:
    name: str
    severity: str  # "error" | "warning"
    evidence: str  # concrete: which steps / values
    likely_cause: str
    suggestion: str
    code: str = ""  # stable rule id (GS001…), stamped by lint() from the registry

    def __str__(self) -> str:
        return (
            f"[{self.code}] [{self.severity.upper()}] {self.name}\n"
            f"  evidence: {self.evidence}\n"
            f"  likely:   {self.likely_cause}\n"
            f"  try:      {self.suggestion}"
        )


# --- detectors: each takes the run df, returns a Finding or None ---


def detect_nonfinite_loss(df: pd.DataFrame) -> Finding | None:
    bad = df[~np.isfinite(df["train_loss"])]
    if bad.empty:
        return None
    step = int(bad.iloc[0]["step"])
    lr = df.loc[df["step"] <= step, "lr"].iloc[-1] if "lr" in df else None
    return Finding(
        name="Loss became NaN/Inf",
        severity="error",
        evidence=f"train_loss non-finite first at step {step}"
        + (f" (lr={lr:g})" if lr is not None else ""),
        likely_cause="LR too high, fp16/bf16 overflow, or a bad batch.",
        suggestion="Lower LR, add grad clipping, check inputs for NaNs, "
        "or use bf16 instead of fp16.",
    )


def detect_nonfinite_grad(df: pd.DataFrame) -> Finding | None:
    if "grad_norm" not in df:
        return None
    g = df["grad_norm"]
    bad = g[
        np.isinf(g)
    ]  # NaN = framework didn't log it here (absent), not a blow-up; only inf is
    if bad.empty:
        return None
    step = int(
        df.loc[bad.index[0], "step"]
    )  # grads often overflow before the loss does
    return Finding(
        name="Gradient norm exploded (inf)",
        severity="error",
        evidence=f"grad_norm is inf first at step {step}",
        likely_cause="Exploding gradients — overflow before the loss shows it.",
        suggestion="Add/tighten grad clipping (max_norm=1.0), lower LR, "
        "check inputs; use bf16 over fp16.",
    )


def detect_grad_spike(
    df: pd.DataFrame, k: float = 8.0, win: int = 50
) -> Finding | None:
    if "grad_norm" not in df:
        return None
    g = df["grad_norm"].astype(float)
    med = g.rolling(win, min_periods=10).median()
    spikes = df[(g > k * med) & med.notna()]
    if spikes.empty:
        return None
    row = spikes.iloc[0]
    step = int(row["step"])
    ratio = row["grad_norm"] / med.loc[row.name]
    return Finding(
        name="Gradient-norm spike",
        severity="warning",
        evidence=f"grad_norm={row['grad_norm']:g} at step {step}, "
        f"~{ratio:.0f}x the local median.",
        likely_cause="Unstable update — often precedes a loss spike "
        "(LR too high for this phase, or an outlier batch).",
        suggestion="Add/tighten grad clipping (e.g. max_norm=1.0); "
        "inspect the batch near this step.",
    )


def detect_overfitting(
    df: pd.DataFrame, patience: int = 100, rel: float = 0.03
) -> Finding | None:
    if "val_loss" not in df:
        return None
    v = df.dropna(subset=["val_loss"]).reset_index(drop=True)
    if len(v) < 4:
        return None
    best_i = int(v["val_loss"].idxmin())
    if best_i == len(v) - 1:
        return None  # val still improving — not overfitting
    rose_for = v["step"].iloc[-1] - v["step"].iloc[best_i]
    if rose_for < patience:
        return None
    best = float(v["val_loss"].iloc[best_i])
    rose = (float(v["val_loss"].iloc[-1]) - best) / abs(best) if best else 0.0
    train_falling = v["train_loss"].iloc[-1] < v["train_loss"].iloc[best_i]
    if rose < rel or not train_falling:  # noise floor, or train not still descending
        return None
    return Finding(
        name="Train/val divergence (overfitting)",
        severity="warning",
        evidence=f"val_loss bottomed at step {int(v['step'].iloc[best_i])} "
        f"({v['val_loss'].iloc[best_i]:.3f}) then rose while train_loss kept falling.",
        likely_cause="Model is memorizing past the generalization point.",
        suggestion="Early-stop near the val minimum; add regularization/"
        "augmentation; or reduce epochs.",
    )


def detect_loss_plateau(
    df: pd.DataFrame, frac: float = 0.2, t_thresh: float = 4.0
) -> Finding | None:
    if "grad_norm" in df and bool(np.isinf(df["grad_norm"]).any()):
        return None  # exploding (inf) grads -> blow-up, not a low-LR plateau (GS002 owns it).
        # NaN (grad simply not logged this step) must NOT suppress plateau run-wide.
    s = df["train_loss"].astype(float)
    s = s[np.isfinite(s)]
    n = len(s)
    if n < 30:
        return None
    w = max(30, int(n * frac))
    ys = s.to_numpy()[:w]
    scale = float(np.mean(ys[: max(5, w // 4)]))
    if scale == 0:
        return None
    # Significance test on the early-window trend: fit a line, then ask whether the
    # slope is distinguishable from zero (|t| = |slope / SE|). A fixed % threshold
    # can't tell "no trend" from "noise" — minibatch noise inflates any % gate — but
    # the t-stat normalizes by the noise, so a frozen run reads as flat regardless.
    xs = np.arange(w)
    slope, intercept = np.polyfit(xs, ys, 1)
    resid = ys - (slope * xs + intercept)
    sxx = float(np.sum((xs - xs.mean()) ** 2))
    se = float(np.sqrt(np.sum(resid**2) / (w - 2)) / np.sqrt(sxx)) if sxx else 0.0
    t = (slope / se) if se else (0.0 if slope == 0 else float("inf"))
    if abs(t) >= t_thresh:  # clearly trending (learning or diverging) -> not a plateau
        return None
    rel_drop = (-slope * (w - 1)) / abs(scale)
    lr = float(df["lr"].iloc[0]) if "lr" in df else None
    return Finding(
        name="Loss plateau (no early progress)",
        severity="warning",
        evidence=f"train_loss trend over the first {w} steps is flat "
        f"(slope {slope:.1e}/step, t={t:.1f}; ~{rel_drop * 100:+.1f}% modeled change)"
        + (f", lr={lr:g}" if lr is not None else ""),
        likely_cause="LR too low, dead/poor init, frozen params, or a data/labeling issue.",
        suggestion="Raise LR (or add warmup), confirm the model is actually getting grads, "
        "and sanity-check a few batches.",
    )


def detect_loss_divergence(
    df: pd.DataFrame, win: int = 20, rise: float = 0.3
) -> Finding | None:
    s = df["train_loss"].astype(float)
    if not np.isfinite(s).all():
        return None  # NaN is detect_nonfinite_loss's job
    sm = s.rolling(win, min_periods=5).mean().dropna()
    if len(sm) < 2 * win:
        return None
    lo_i = sm.idxmin()  # best-so-far: lowest smoothed loss (TFCheck cur/lowest_ever)
    lo = float(sm.loc[lo_i])
    after = sm.loc[lo_i:]
    end = float(after.iloc[-1])
    # sustained rise, not a terminal spike: elevated for at least `win` smoothed steps
    if (
        lo <= 0
        or (end - lo) / abs(lo) < rise
        or (after > lo * (1 + rise / 2)).sum() < win
    ):
        return None
    step = int(df["step"].loc[lo_i])
    return Finding(
        name="Loss divergence (sustained rise)",
        severity="error",
        evidence=f"smoothed train_loss rose {((end - lo) / abs(lo)) * 100:.0f}% after "
        f"step {step} ({lo:.3f} -> {end:.3f}) — a trend, not a single spike.",
        likely_cause="LR too high / schedule wrong, or training going unstable.",
        suggestion="Lower LR or revisit the schedule; add grad clipping; "
        "resume from the last good checkpoint.",
    )


def detect_vanishing_grad(
    df: pd.DataFrame, collapse: float = 0.05, stuck: float = 0.2
) -> Finding | None:
    if "grad_norm" not in df:
        return None
    g = df["grad_norm"].astype(float)
    g = g[np.isfinite(g)]  # inf is GS002's job; NaN = not logged this step
    if len(g) < 40:
        return None
    w = max(10, len(g) // 5)  # first/last 20%
    g_early = float(np.median(g.iloc[:w]))
    g_late = float(np.median(g.iloc[-w:]))
    if g_early <= 0 or g_late >= collapse * g_early:
        return None  # grads didn't collapse
    # Shrinking grads are healthy IF the loss actually fell — that's convergence,
    # not vanishing. Only flag when the loss is still stuck near its starting value.
    s = df["train_loss"].astype(float)
    s = s[np.isfinite(s)]
    l_early = float(np.median(s.iloc[:w]))
    l_late = float(np.median(s.iloc[-w:]))
    if l_early == 0 or (l_early - l_late) / abs(l_early) >= stuck:
        return None  # loss made real progress -> grads shrinking is convergence
    step = int(df["step"].iloc[-w])
    return Finding(
        name="Vanishing gradients",
        severity="warning",
        evidence=f"grad_norm collapsed to {g_late:.1e} (~{g_late / g_early * 100:.0f}% "
        f"of its early {g_early:.1e}) by step {step} while train_loss stayed stuck "
        f"({l_early:.3f} -> {l_late:.3f}).",
        likely_cause="Vanishing gradients — dead/saturated units, too-deep stack "
        "without residuals/norm, or LR far too low.",
        suggestion="Check init/normalization and activations (dead ReLU?), add "
        "residual connections or normalization, or raise LR.",
    )


def detect_update_ratio(
    df: pd.DataFrame, low: float = -4.0, high: float = -2.0, skip: float = 0.1
) -> Finding | None:
    # Karpathy's rule of thumb: healthy log10(||ΔW||/||W||) sits near -3. Only fires
    # on the update_ratio signal watch(model, optimizer) captures — absent for
    # csv/log-only users, so it stays silent rather than guessing. Band [-4,-2] is
    # deliberately wide: a wrong LR verdict is worse than none.
    if "update_ratio" not in df:
        return None
    r = df["update_ratio"].astype(float)
    r = r[np.isfinite(r) & (r > 0)]
    if len(r) < 30:
        return None
    r = r.iloc[int(len(r) * skip) :]  # drop warmup: early ratios swing wildly
    med = float(np.median(np.log10(r)))
    if low <= med <= high:
        return None  # healthy
    too_high = med > high
    return Finding(
        name="LR too high (update/weight ratio)"
        if too_high
        else "LR too low (update/weight ratio)",
        severity="warning",
        evidence=f"median update/weight ratio ~1e{med:.1f} "
        f"({'above' if too_high else 'below'} the healthy ~1e-3; "
        f"band is 1e{high:.0f}..1e{low:.0f}).",
        likely_cause="Each step moves the weights "
        + ("too much — LR too high." if too_high else "too little — LR too low."),
        suggestion="Lower the LR (or add clipping)."
        if too_high
        else "Raise the LR (or add warmup); check params are actually training.",
    )


def detect_loss_oscillation(
    df: pd.DataFrame,
    win: int = 11,
    min_reversals: int = 6,
    grow: float = 1.6,
    skip: float = 0.1,
) -> Finding | None:
    # Only flags oscillation with GROWING amplitude — unambiguously unstable. A
    # constant-amplitude oscillation (normal for GANs/RL) or a decaying one (a run
    # settling) stays silent, and minibatch noise is smoothed out first. This gate is
    # what keeps a wrong "LR too high" off healthy adversarial runs.
    s = df["train_loss"].astype(float)
    s = s[np.isfinite(s)]
    n = len(s)
    if n < 60:
        return None
    s = s.iloc[int(n * skip) :]  # drop warmup
    sm = s.rolling(win, min_periods=win).mean().dropna().to_numpy()
    if len(sm) < 4 * win:
        return None
    sign = np.sign(np.diff(sm))
    sign = sign[sign != 0]
    reversals = int(np.sum(sign[1:] != sign[:-1]))
    if reversals < min_reversals:
        return None  # converging/noisy runs rarely reverse once smoothed

    def _amp(seg):  # swing size = spread AROUND the local trend, not the trend itself
        xs = np.arange(
            len(seg)
        )  # so a productive late descent isn't mistaken for a swing
        slope, intercept = np.polyfit(xs, seg, 1)
        return float(np.ptp(seg - (slope * xs + intercept)))

    third = len(sm) // 3
    amp_early = _amp(sm[:third])
    amp_late = _amp(sm[-third:])
    floor = 1e-3 * abs(float(np.median(sm)))  # below this a "swing" is float noise
    if amp_early <= floor or amp_late < grow * amp_early:
        return None  # flat/constant/decaying amplitude = healthy osc or settling
    step = int(df["step"].iloc[-1])
    return Finding(
        name="Loss oscillation (growing amplitude)",
        severity="warning",
        evidence=f"smoothed train_loss swings grew {amp_early:.3f} -> {amp_late:.3f} "
        f"(~{amp_late / amp_early:.1f}x) over {reversals} direction reversals by step {step}.",
        likely_cause="Unstable updates bouncing across the minimum with growing swings "
        "— usually LR too high.",
        suggestion="Lower the LR or add warmup/decay; add grad clipping. "
        "Constant-amplitude oscillation (GANs/RL) is normal and not flagged.",
    )


def detect_lr_collapse(
    df: pd.DataFrame, floor: float = 0.01, before: float = 0.8, improving: float = 0.05
) -> Finding | None:
    # Misconfigured schedule: total_steps/num_training_steps set shorter than the run,
    # so the LR decays to ~0 partway through and the rest of the run is wasted. A
    # cosine/linear schedule that lands at 0 *at the end* is by design, so this only
    # fires when the LR died well before the end AND the loss stopped moving after it
    # while it was still falling before — otherwise stay silent.
    if "lr" not in df:
        return None
    lr = df["lr"].astype(float).to_numpy()
    s = df["train_loss"].astype(float).to_numpy()
    n = len(lr)
    if n < 60 or not np.isfinite(s).all():
        return None
    finite = lr[np.isfinite(lr)]
    if len(finite) == 0 or (peak := float(finite.max())) <= 0:
        return None
    dead = lr <= floor * peak
    if not dead[-1]:
        return None
    i = int(n - np.argmin(dead[::-1]))  # first index of the trailing all-dead run
    if i >= before * n or i < 20 or n - i < 30:
        return None

    def _drop(seg):  # relative improvement across a segment, median-based
        w = max(5, len(seg) // 5)
        a, b = float(np.median(seg[:w])), float(np.median(seg[-w:]))
        return (a - b) / abs(a) if a else 0.0

    pre, post = _drop(s[:i]), _drop(s[i:])
    # Post-collapse "progress" is mostly minibatch noise (with lr=0 the weights are
    # frozen, yet the median still wobbles a few %), so judge it relative to the
    # pre-collapse drop rather than against a fixed floor.
    if pre < improving or post >= max(improving, pre / 8):
        return None  # wasn't learning before, or still learning after -> not the LR
    step = int(df["step"].iloc[i])
    return Finding(
        name="LR schedule collapsed early",
        severity="warning",
        evidence=f"lr fell to {lr[i]:.2g} (<{floor:.0%} of its {peak:.2g} peak) at step "
        f"{step}, {i / n:.0%} into the run, and stayed there; train_loss improved "
        f"{pre * 100:.0f}% before that and {post * 100:.0f}% after.",
        likely_cause="Scheduler length mismatch — total/num_training_steps set shorter "
        "than the actual run, so the remaining steps train at ~zero LR.",
        suggestion="Set the scheduler's total steps to the real number of optimizer "
        "steps (mind grad accumulation/epochs), or give the schedule a min_lr floor.",
    )


# (stable rule id, detector). IDs are permanent — config/suppressions pin to them,
# so never renumber. Add a new rule with the next free GS number.
DETECTORS = [
    ("GS001", detect_nonfinite_loss),
    ("GS002", detect_nonfinite_grad),
    ("GS003", detect_grad_spike),
    ("GS004", detect_overfitting),
    ("GS005", detect_loss_plateau),
    ("GS006", detect_loss_divergence),
    ("GS007", detect_vanishing_grad),
    ("GS008", detect_update_ratio),
    ("GS009", detect_loss_oscillation),
    ("GS010", detect_lr_collapse),
]


# Map foreign metric names (W&B/TB exports, framework log dicts) -> canonical.
# Shared by lint() and every framework adapter — same operation, one table.
_ALIASES = {
    "step": ["global_step", "trainer/global_step", "_step", "iteration", "train/step"],
    "train_loss": [
        "loss",
        "train/loss",
        "loss_step",
        "training_loss",
        "train_loss_step",
    ],
    "val_loss": [
        "eval_loss",
        "val/loss",
        "eval/loss",
        "validation_loss",
        "valid_loss",
        "val_loss_epoch",
        "val_loss_step",
    ],
    "lr": ["learning_rate", "train/learning_rate", "train/lr"],
    "grad_norm": ["train/grad_norm", "gradient_norm", "grad_norm_step", "total_norm"],
}


def normalize(d):
    """Rename known aliases to canonical names. Works on a dict or a DataFrame
    (both expose `in` over keys/columns and `.rename`-equivalent)."""
    keys = d.keys() if isinstance(d, dict) else d.columns
    ren = {}
    for canon, aliases in _ALIASES.items():
        if canon in keys:
            continue
        for a in aliases:
            if a in keys:
                ren[a] = canon
                break
    if not ren:
        return d
    if isinstance(d, dict):
        return {ren.get(k, k): v for k, v in d.items()}
    return d.rename(columns=ren)


def lint(df: pd.DataFrame, mute=()) -> list[Finding]:
    df = normalize(df)
    if df.empty or "train_loss" not in df:
        return []  # nothing to diagnose beats a KeyError
    if (
        "step" not in df
    ):  # loss-only exports: synthesize a step so detectors don't KeyError
        df = df.assign(step=range(len(df)))
    df = df.sort_values("step").reset_index(
        drop=True
    )  # detectors assume ascending order
    out = []
    for code, d in DETECTORS:
        if code in mute:  # suppress by stable rule id, e.g. mute={"GS003"}
            continue
        f = d(df)
        if f is not None:
            f.code = code  # stamp the stable id from the registry
            out.append(f)
    return out


def lint_csv(path: str, mute=()) -> list[Finding]:
    df = normalize(pd.read_csv(path))
    if "train_loss" not in df:  # don't silently say "all clear" on the wrong columns
        raise ValueError(f"{path}: need a loss column; got {list(df.columns)}")
    return lint(df, mute=mute)


# --- seamless capture: one-line in-loop monitor (no copy-paste, no heavy deps) ---


class Monitor:
    """Capture standard signals live and run detectors. Framework-free.

    Usage in a raw PyTorch loop, right after loss.backward() (grads still live):

        mon = watch(model, optimizer)
        ...
        loss.backward()
        mon.log(step, loss.item(), val_loss=val)   # grabs grad_norm + lr for you
        optimizer.step(); optimizer.zero_grad()
        ...
        mon.report()

    No torch import here — `model`/`optimizer` are duck-typed, so the module
    loads (and tests run) without torch installed.
    """

    def __init__(
        self, model=None, optimizer=None, check_every: int = 0, mute=(), on_alert=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.check_every = check_every  # >0: print errors live every N steps
        self.mute = set(mute)  # suppress these rule ids (e.g. {"GS003"})
        self.on_alert = on_alert  # optional callback(Finding) for each new live error
        self.rows: list[dict] = []
        self._announced: set = set()  # dedupe live alerts: each finding fires once
        self._prev_params = None  # last step's params, for the update/weight ratio

    def _grad_norm(self):
        if self.model is None:
            return None
        total, seen = 0.0, False
        for p in self.model.parameters():
            g = getattr(p, "grad", None)
            if g is not None:
                seen = True
                total += float(g.detach().norm()) ** 2
        return total**0.5 if seen else None

    def _lr(self):
        groups = getattr(self.optimizer, "param_groups", None)
        return float(groups[0]["lr"]) if groups else None

    def _update_ratio(self):
        # ||ΔW|| / ||W|| across one optimizer step (Karpathy's ~1e-3 rule). Uses the
        # actual param delta, not lr*grad — Adam's update isn't lr*grad. First call
        # has no baseline -> None. ponytail: keeps one model-sized snapshot; that's
        # small next to the optimizer state (Adam already holds ~2x the params), and
        # only when a model is passed. Torch stays duck-typed (no import here).
        if self.model is None:
            return None
        cur = [p.detach() for p in self.model.parameters()]
        prev = self._prev_params
        self._prev_params = [c.clone() for c in cur]
        if prev is None or len(prev) != len(cur):
            return None
        num = sum(float((c - p).norm()) ** 2 for c, p in zip(cur, prev))
        den = sum(float(p.norm()) ** 2 for p in prev)
        return (num**0.5) / (den**0.5) if den > 0 else None

    def log(self, step: int, loss: float, val_loss: float | None = None, **extra):
        row = {"step": int(step), "train_loss": float(loss)}
        gn = self._grad_norm()
        if gn is not None:
            row["grad_norm"] = gn
        lr = self._lr()
        if lr is not None:
            row["lr"] = lr
        ur = self._update_ratio()
        if ur is not None:
            row["update_ratio"] = ur
        if val_loss is not None:
            row["val_loss"] = float(val_loss)
        row.update(extra)  # caller-supplied signals (grad_norm/lr/per-layer/...) win
        self.rows.append(row)
        # trigger on number of feeds, not step value: callback feeds arrive at
        # step 500,1000,... (logging_steps), where step % check_every fires erratically
        if self.check_every and len(self.rows) % self.check_every == 0:
            for f in lint(self.df(), mute=self.mute):
                if f.severity == "error" and f.code not in self._announced:
                    self._announced.add(f.code)
                    print(f"\n[snitch] {f}\n")
                    if self.on_alert:  # e.g. push to wandb.alert / Slack
                        self.on_alert(f)
        return self

    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def report(self) -> list[Finding]:
        findings = lint(self.df(), mute=self.mute)
        for f in findings:
            print(f, "\n")
        return findings


def watch(
    model=None, optimizer=None, check_every: int = 0, mute=(), on_alert=None
) -> Monitor:
    return Monitor(model, optimizer, check_every, mute=mute, on_alert=on_alert)


# --- self-check / demo: synthetic broken runs ---


def _synthetic(kind: str, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    step = np.arange(n)
    loss = 3.0 * np.exp(-step / 120) + 0.3 + rng.normal(0, 0.02, n)
    grad = np.abs(rng.normal(1.0, 0.2, n))
    lr = np.full(n, 3e-4)
    val = loss + 0.05
    ratio = np.full(n, 1e-3)  # healthy update/weight ratio (Karpathy's ~1e-3)
    if kind == "hotlr":  # updates far too big relative to weights
        ratio = np.full(n, 5e-2) * np.abs(rng.normal(1.0, 0.05, n))
    elif kind == "coldlr":  # updates far too small
        ratio = np.full(n, 1e-5) * np.abs(rng.normal(1.0, 0.05, n))
    if kind == "nan":
        loss[250:] = np.nan
    elif kind == "spike":
        grad[200] = 30.0
        loss[201:210] += 1.5
    elif kind == "overfit":
        val[200:] = val[200] + (step[200:] - 200) * 0.002
    elif kind == "plateau":
        loss = 2.8 + rng.normal(0, 0.02, n)
        val = loss + 0.05
    elif kind == "diverge":
        base = 3.0 * np.exp(-step / 120) + 0.3
        base[200:] = base[200] + (step[200:] - 200) * 0.01
        loss = base + rng.normal(0, 0.02, n)
        val = loss + 0.05
    elif kind == "vanish":  # grads decay to ~0, loss stays stuck (no progress)
        loss = 2.8 + rng.normal(0, 0.02, n)
        val = loss + 0.05
        grad = np.abs(1.0 * np.exp(-step / 40) + rng.normal(0, 0.005, n))
    elif kind == "osc":  # growing-amplitude oscillation, flat mean -> unstable
        loss = 1.0 + (0.05 + step / n * 0.9) * np.sin(step / 3) + rng.normal(0, 0.02, n)
        val = loss + 0.05
    elif kind == "lrdead":  # schedule hits 0 halfway; loss frozen from there on
        half = n // 2
        lr = np.where(step < half, 3e-4 * (1 - step / half), 0.0)
        loss = np.where(step < half, 3.0 * np.exp(-step / 60) + 0.3, 0.3) + rng.normal(
            0, 0.01, n
        )
        val = loss + 0.05
    elif kind == "cosine":  # full-length cosine to 0 with real progress -> silent
        lr = 3e-4 * 0.5 * (1 + np.cos(np.pi * step / n))
        val = loss + 0.05
    elif kind == "healthy_osc":  # constant-amplitude (GAN/RL) -> must stay silent
        loss = 1.0 + 0.2 * np.sin(step / 3) + rng.normal(0, 0.02, n)
        val = loss + 0.05
    return pd.DataFrame(
        {
            "step": step,
            "train_loss": loss,
            "grad_norm": grad,
            "lr": lr,
            "val_loss": val,
            "update_ratio": ratio,
        }
    )


def _selfcheck() -> None:
    assert any("NaN" in f.name for f in lint(_synthetic("nan"))), "missed NaN"
    assert any("spike" in f.name.lower() for f in lint(_synthetic("spike"))), (
        "missed spike"
    )
    assert any("overfit" in f.name.lower() for f in lint(_synthetic("overfit"))), (
        "missed overfit"
    )
    assert any("plateau" in f.name.lower() for f in lint(_synthetic("plateau"))), (
        "missed plateau"
    )
    assert any("diverg" in f.name.lower() for f in lint(_synthetic("diverge"))), (
        "missed divergence"
    )
    assert any("vanish" in f.name.lower() for f in lint(_synthetic("vanish"))), (
        "missed vanishing grads"
    )
    # convergence must NOT read as vanishing: clean run's grads are steady, loss falls
    assert not any("vanish" in f.name.lower() for f in lint(_synthetic("clean"))), (
        "false vanishing on a converging run"
    )
    assert any("too high" in f.name.lower() for f in lint(_synthetic("hotlr"))), (
        "missed high-LR update ratio"
    )
    assert any("too low" in f.name.lower() for f in lint(_synthetic("coldlr"))), (
        "missed low-LR update ratio"
    )
    assert any("oscillation" in f.name.lower() for f in lint(_synthetic("osc"))), (
        "missed growing-amplitude oscillation"
    )
    # constant-amplitude oscillation (GAN/RL) is healthy — must NOT be flagged
    assert not any(
        "oscillation" in f.name.lower() for f in lint(_synthetic("healthy_osc"))
    ), "false oscillation on a constant-amplitude (GAN-like) run"
    assert any("schedule" in f.name.lower() for f in lint(_synthetic("lrdead"))), (
        "missed early LR-schedule collapse"
    )
    # a cosine schedule that lands at 0 at the *end* of the run is by design
    assert not any("schedule" in f.name.lower() for f in lint(_synthetic("cosine"))), (
        "false LR-collapse on a normal full-length cosine schedule"
    )
    clean = lint(_synthetic("clean"))
    assert clean == [], f"false positive on clean run: {clean}"

    # watch() plumbing: replay a broken run through log(), report() must catch it
    mon = watch()  # no model -> caller supplies signals
    for r in _synthetic("spike").to_dict("records"):
        mon.log(
            r["step"],
            r["train_loss"],
            val_loss=r["val_loss"],
            grad_norm=r["grad_norm"],
            lr=r["lr"],
        )
    assert any("spike" in f.name.lower() for f in lint(mon.df())), (
        "watch() lost signals"
    )
    print("selfcheck ok")


# selfcheck/demo entrypoint lives in __main__.py so `python -m gradsnitch` runs it.

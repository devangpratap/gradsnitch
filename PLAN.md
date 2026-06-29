# gradsnitch — plan

## Thesis
Tracking ≠ diagnosis. W&B/TensorBoard plot the curve; nobody ships the tool that
*reads* it and says what broke. Snitch gives verdicts. (Closest prior art:
Cockpit, NeurIPS 2021 — it hands you instruments, not verdicts, and is stale.)

## Done
- **Core:** pure detector functions → `Finding(severity, where, what, likely_cause, fix)`.
  5 detectors: NaN/Inf loss, grad-norm spike, loss divergence, plateau/not-learning, overfitting.
- **Seamless capture:** `watch()`/`Monitor` — one-line in-loop, framework-free, deduped live alerts.
- **Validated on a real torch run** (caught a live NaN divergence from one `mon.log()`).
- **Self-check:** a synthetic broken run per detector; assert each fires + zero false positives on a clean run.

## Tiering (advanced ≠ heavy)
- **Tier 0** (core, scipy-only): upgrade detector internals from naive thresholds →
  robust stats / changepoint (CUSUM, Bayesian online changepoint), trend tests
  (Mann-Kendall / Theil-Sen), curve-fit-to-expected-decay, cheap Gradient Noise
  Scale (McCandlish — from micro-batch grad-norm variance, no per-example grads).
- **Tier 1** (`watch()` unlocks cheaply): per-layer grad norms, micro-batch grad variance.
  No SVD, no per-example grads.
- **Tier 2** (opt-in `advanced=True`): spectral alignment (early-warning hero,
  arXiv 2510.04202), attention-entropy collapse (2303.06296), dead-unit activation
  capture. Label verdicts "experimental."

## Next
1. README + screenshot → publish. (the broadcast)
2. Run on real training logs; let real data pick which detector to sharpen first.
3. Robustify detector #1 (grad spike → changepoint/MAD) as the first Tier-0 proof.
4. Framework adapters: HF Trainer / Lightning callbacks (thin wrappers over `watch()`).
5. pip packaging.

## Anti-bloat rule
Every new feature is **a detector or an adapter**. If it's neither, it doesn't go
in. No tracker rebuild, no UI, no LLM-as-diagnoser (an LLM may *phrase* findings
later — it never *decides* them).

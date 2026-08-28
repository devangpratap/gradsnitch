# Contributing

The whole project is one rule: **every change is a detector or an adapter.**
Anything that is neither — a UI, a tracker, a config framework, an LLM that
decides verdicts — is out of scope. An LLM may one day *phrase* a finding; it
will never *decide* one.

The other rule: **a wrong diagnosis is worse than none.** A detector that fires
on a healthy run is a bug, not a tradeoff. Every rule returns `None` unless the
signature is unambiguous.

## Adding a detector

1. Write a pure function `detect_x(df) -> Finding | None` in `gradsnitch/__init__.py`.
   It gets a DataFrame with canonical columns (`step`, `train_loss`, `grad_norm`,
   `lr`, `val_loss`) and returns a `Finding` with **concrete evidence** — which
   steps, which values, not "the loss looked bad".
2. Register it in `DETECTORS` with the next free `GS0NN`. **Rule ids are permanent** —
   people pin `mute={"GS003"}` to them, so never renumber.
3. Add **two** rigs to `tests/test_real_runs.py`:
   - a positive rig that trains a tiny real torch model that breaks through the
     real mechanism (LR `1e4` → NaN, 8-point set → overfit, deep sigmoid → vanishing),
     and asserts your verdict fires;
   - a negative rig — the nearest *healthy* curve that could be mistaken for it —
     and asserts it stays **silent**.

   The negative rig is the part that matters. Synthetic arrays that hit your
   thresholds by construction prove nothing; the harness only has value because
   the runs are real.
4. Document it in the README rule table.

## Adding a framework adapter

Adapters are ~10 lines: pull the framework's metric dict, hand it to the shared
`_feed` + `normalize()` alias table, done. Import the framework **lazily** — it
must never become a hard dependency. Add its log key spellings to `_ALIASES`
rather than special-casing in the adapter.

## Reporting a false positive

This is the most useful issue you can file. Use the false-positive template and
attach the run's metrics CSV if you can — real curves that fool a rule are how
thresholds get sharpened, and they become the next negative rig.

## Running the tests

```bash
pip install -e ".[dev]"
pytest -q          # 23 rigs, ~6s
python -m gradsnitch   # selfcheck + a demo verdict
```

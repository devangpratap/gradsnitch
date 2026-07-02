"""`python -m gradsnitch` — run the selfcheck, then demo a known-broken run."""

from gradsnitch import _selfcheck, lint, _synthetic

_selfcheck()
print("\n--- demo: a known-broken run (grad spike) ---")
for f in lint(_synthetic("spike")):
    print(f, "\n")

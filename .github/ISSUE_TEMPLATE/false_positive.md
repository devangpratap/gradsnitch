---
name: False positive
about: A rule fired on a run that was actually fine
labels: false-positive
---

**Which rule fired** (e.g. GS003):

**Why the run was actually healthy:**

**The verdict it printed:**
```
paste the [GSxxx] block
```

**Metrics** — attach the CSV export if you can (step / train_loss / grad_norm /
lr / val_loss). A real curve that fools a rule becomes a negative test rig.

**Setup:** framework (raw torch / HF / Lightning / Keras), gradsnitch version.

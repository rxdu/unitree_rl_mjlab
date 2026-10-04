# Lessons

Operational lessons for this fork. Check before changing dependencies or verifying a training run.

### Smoke tests must run the default configuration
- **Pattern:** A training smoke test was run with `--agent.logger=tensorboard` to avoid the wandb login. It passed, but the default wandb logger still crashed in `rsl_rl` (`wandb.Settings(start_method=...)` rejected by wandb 0.30.0), because the test never exercised that path.
- **Correction:** Smoke-test the exact command users run and only override what cannot run unattended, preferably via the environment rather than a config switch (e.g. `WANDB_MODE=offline` keeps the wandb code path but uploads nothing). Shorten runs with `--agent.max-iterations=3`.
- **Context:** `scripts/train.py`, rsl-rl-lib 5.0.1, wandb.

### Upstream pins only top-level packages; transitive dependencies drift
- **Pattern:** Upstream `setup.py` pins `mjlab==1.2.0` and `mujoco-warp==3.5.0`, but those declare only lower bounds (`mujoco>=3.5.0`, `warp-lang>=1.12.0`, `wandb>=0.22.3`), and mjlab imports `scipy` without declaring it. A fresh install pulled the newest releases, which broke at import or startup one at a time (`mjENBL_MULTICCD`, `scipy`, `wp.context`, `wandb.Settings`).
- **Correction:** When one import fails from version drift, audit every lower-bound-only dependency of the pinned packages (`Requires-Dist` in each `*.dist-info/METADATA`) instead of fixing errors one by one. `mujoco-warp X.Y` matches `mujoco X.Y`. Pin the working set in `setup.py` and verify with `pip install -e . --dry-run` plus a smoke test.
- **Context:** Python packaging, mjlab 1.2.0 / mujoco-warp 3.5.0 stack.

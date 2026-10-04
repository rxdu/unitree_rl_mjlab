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

### A recomputed mjlab observation is not a fresh one
- **Pattern:** The conformance generator set a new command, then called `observation_manager.compute_group("actor")` to observe it. The command term in the result still held the previous, randomly sampled command: with `update_history=False`, mjlab serves each term's value from its observation buffers, i.e. as of the last step or reset. Only the generator's independent reference check (the contract's definitions in numpy) caught it — row 0, observation index 8.
- **Correction:** Do not recompute observations outside `step()`/`reset()`. Read the observation the environment produced, together with the state at that same instant, and write any input you want observed (a command) *before* the step whose end observes it.
- **Context:** mjlab 1.2.0 `ObservationManager.compute` / `compute_group`; `scripts/export_conformance.py`.

### One SceneEntityCfg must not be shared between event terms
- **Pattern:** A single `SceneEntityCfg("robot", joint_names=(...))` instance was passed to both the joint-damping and the joint-friction events. Training failed at startup with "Inconsistent joint names and indices": mjlab writes the resolved joint ids back into the config, so the second term saw a regex plus already-resolved ids.
- **Correction:** Build a fresh `SceneEntityCfg` per term (a small factory function), never one shared instance. Start a short CPU run after any config change; it fails in seconds where reading the code did not.
- **Context:** mjlab 1.2.0 event manager; `src/tasks/velocity/config/go2/env_cfgs.py`.

### Python's csv module writes CRLF unless told otherwise
- **Pattern:** `conformance.csv` was written with `csv.DictWriter` defaults, which end lines with `\r\n`. The C++ reader split on `\n`, leaving `\r` on the last header name, and failed with "no column qtarget_RL_calf_joint" — far from the cause.
- **Correction:** Pass `lineterminator="\n"` when a file is consumed outside Python, state the line ending in the format's spec, and make readers strip a trailing `\r` anyway.
- **Context:** Python `csv`; any cross-language text fixture.

### A capacity sized from one sample of a random quantity
- **Pattern:** `play.py Unitree-Go2-Rough` failed with "nconmax overflow (nconmax must be >= 38)". Raising `nconmax` to 48 made the next run start, and that was committed — but the play terrain is regenerated with a random seed on every run, and the following run needed 62.
- **Correction:** Before sizing a buffer from an observed value, find out whether that value is random; if it is, either derive the size (here `nconmax=None`, which mujoco_warp sizes from the actual initial state) or test several draws. One passing run of a randomized setup is one sample.
- **Context:** mjlab 1.2.0 / mujoco_warp 3.5.0 `put_data`; Go2 rough play config.

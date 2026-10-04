"""Record conformance vectors for an exported policy artifact.

Produces `conformance.csv` next to an artifact written by training
(`<run>/policy_artifact/`), per xmAppLeggedController
`docs/design/control/policy-contract.md` §8 (ADR-0011 D5): rollouts of the
training environment with observation noise off, recording the inputs as the
runtime receives them, the observation mjlab built, and the action the policy
returned.

Before writing, every row is checked twice, so a vector file that disagrees
with the contract it accompanies is never produced:
  1. the contract's own §4 definitions, evaluated here in numpy from the
     recorded inputs, must reproduce mjlab's observation -- this checks the
     SPECIFICATION against the training code, before any runtime exists;
  2. the artifact's policy.bin, evaluated as §7 prescribes, must reproduce the
     policy's action -- this checks the artifact against the checkpoint.

Usage:
  python scripts/export_conformance.py Unitree-Go2-Flat \
    --checkpoint-file logs/rsl_rl/go2_velocity/<run>/model_<n>.pt
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np
import torch
import tyro
import yaml

import mjlab.tasks  # noqa: F401  (registers mjlab tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.tasks  # noqa: E402, F401  (registers this repo's tasks)
from src.tasks.velocity import mdp  # noqa: E402
from src.tasks.velocity.rl.policy_contract import (  # noqa: E402
  POLICY_YAML,
  evaluate_bin,
  read_bin,
)
from src.tasks.velocity.rl.runner import ARTIFACT_DIR  # noqa: E402

HAL_ORDER = [
  f"{leg}_{joint}_joint"
  for leg in ("FR", "FL", "RR", "RL")
  for joint in ("hip", "thigh", "calf")
]

# Command segments: walk, stand, turn, below the stand threshold, walk back.
# Each runs for `steps_per_segment` inferences; a manual reset precedes the
# third so the file holds more than one episode start.
SEGMENTS = [
  (0.6, 0.0, 0.0),
  (0.0, 0.0, 0.0),
  (0.3, 0.2, 0.4),
  (0.05, 0.0, 0.05),
  (-0.4, 0.0, -0.4),
]
RESET_BEFORE_SEGMENT = 2

class _ParamsLoader(yaml.SafeLoader):
  """SafeLoader plus the one tag mjlab's dump_yaml emits (tuples)."""


_ParamsLoader.add_constructor(
  "tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node))
)

def _deep_merge(base: dict, override: dict) -> dict:
  out = dict(base)
  for key, value in override.items():
    if isinstance(value, dict) and isinstance(out.get(key), dict):
      out[key] = _deep_merge(out[key], value)
    else:
      out[key] = value
  return out


OBS_TOL = (1e-5, 1e-5)  # (atol, rtol), policy-contract.md §8
ACT_TOL = (1e-4, 1e-4)


@dataclass
class Args:
  task: tyro.conf.Positional[str]
  checkpoint_file: str
  artifact_dir: str | None = None
  """Defaults to `<checkpoint dir>/policy_artifact`."""
  steps_per_segment: int = 20
  device: str = "cpu"


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
  w, x, y, z = q / np.linalg.norm(q)
  return np.array(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
  )


def reference_observation(
  contract: dict, row: dict, k: int, last_action: np.ndarray
) -> np.ndarray:
  """policy-contract.md §4, implemented from the spec text, not from mjlab."""
  names = contract["joints"]["names"]
  default = np.array(contract["joints"]["default_position_rad"])
  q = np.array([row[f"q_{n}"] for n in names])
  qd = np.array([row[f"qd_{n}"] for n in names])
  ranges = contract["command"]["ranges"]
  cmd = np.array(
    [
      np.clip(row["cmd_vx"], *ranges["lin_vel_x"]),
      np.clip(row["cmd_vy"], *ranges["lin_vel_y"]),
      np.clip(row["cmd_wz"], *ranges["ang_vel_z"]),
    ]
  )
  quat = np.array([row["quat_w"], row["quat_x"], row["quat_y"], row["quat_z"]])
  parts = []
  for term in contract["observation"]["terms"]:
    n = term["name"]
    if n == "base_ang_vel":
      v = np.array([row["gyro_x"], row["gyro_y"], row["gyro_z"]])
    elif n == "projected_gravity":
      v = _quat_to_rot(quat).T @ np.array([0.0, 0.0, -1.0])
    elif n == "command":
      v = cmd
    elif n == "gait_phase":
      dt = contract["timing"]["policy_period_s"]
      phi = ((k * dt) % term["period_s"]) / term["period_s"]
      v = np.array([np.sin(2 * np.pi * phi), np.cos(2 * np.pi * phi)])
      if np.linalg.norm(cmd) < term["stand_threshold"]:
        v = np.zeros(2)
    elif n == "joint_pos_rel":
      v = q - default
    elif n == "joint_vel_rel":
      v = qd
    elif n == "last_action":
      v = last_action
    else:
      raise ValueError(f"unknown term {n}")
    parts.append(term["scale"] * v)
  return np.concatenate(parts)


def _check(name: str, got: np.ndarray, ref: np.ndarray, tol: tuple, where: str) -> None:
  atol, rtol = tol
  bad = np.abs(got - ref) > atol + rtol * np.abs(ref)
  if np.any(bad):
    i = int(np.argmax(np.abs(got - ref)))
    raise SystemExit(
      f"[conformance] {name} mismatch at {where}, index {i}: "
      f"got {got[i]:.8g}, expected {ref[i]:.8g}"
    )


def main(args: Args) -> None:
  artifact_dir = args.artifact_dir or os.path.join(
    os.path.dirname(os.path.abspath(args.checkpoint_file)), ARTIFACT_DIR
  )
  with open(os.path.join(artifact_dir, POLICY_YAML)) as f:
    contract = yaml.safe_load(f)
  net = contract["network"]
  tensors = read_bin(os.path.join(artifact_dir, net["weights_file"]), net["tensors"])

  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = 1
  twist = env_cfg.commands["twist"]
  twist.resampling_time_range = (1e9, 1e9)  # segments set the command
  twist.heading_command = False
  twist.ranges.heading = None  # mjlab refuses a heading range without heading control
  twist.rel_standing_envs = 0.0
  twist.rel_heading_envs = 0.0
  if env_cfg.observations["actor"].enable_corruption:
    raise SystemExit("observation noise must be off (play config) for conformance")

  # The network shape is whatever the run trained, which may differ from the
  # task's defaults: read the run's own params/agent.yaml when it exists.
  agent_yaml = os.path.join(
    os.path.dirname(os.path.abspath(args.checkpoint_file)), "params", "agent.yaml"
  )
  # The dump is taken after rsl_rl has popped the `class_name` keys, so the
  # task defaults supply those and the run's values override everything else.
  agent_cfg = asdict(load_rl_cfg(args.task))
  if os.path.exists(agent_yaml):
    with open(agent_yaml) as f:
      agent_cfg = _deep_merge(agent_cfg, yaml.load(f, Loader=_ParamsLoader))  # noqa: S506
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.get("clip_actions"))
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, agent_cfg, device=args.device)
  runner.load(
    args.checkpoint_file, load_cfg={"actor": True}, strict=True, map_location=args.device
  )
  policy = runner.alg.get_policy().as_onnx(verbose=False).to(args.device).eval()

  robot = env.scene["robot"]
  twist_term = env.command_manager.get_term("twist")
  names = list(robot.joint_names)
  if names != contract["joints"]["names"]:
    raise SystemExit(f"env joint order {names} differs from the artifact's")
  hal_idx = [names.index(n) for n in HAL_ORDER]
  default = np.array(contract["joints"]["default_position_rad"])
  scale = np.array(contract["action"]["scale"])
  clip = contract["action"]["clip"]

  header = (
    ["episode_start", "k"]
    + ["quat_w", "quat_x", "quat_y", "quat_z", "gyro_x", "gyro_y", "gyro_z"]
    + ["cmd_vx", "cmd_vy", "cmd_wz"]
    + [f"q_{n}" for n in HAL_ORDER]
    + [f"qd_{n}" for n in HAL_ORDER]
    + [f"obs_{i}" for i in range(contract["observation"]["dim"])]
    + [f"act_{i}" for i in range(12)]
    + [f"qtarget_{n}" for n in HAL_ORDER]
  )
  rows = []
  # The contract's episode semantics (§6), tracked independently of mjlab and
  # then compared with it: k counts inferences since the episode start, and
  # the previous action is zero at k = 0.
  k_ref, prev_a = 0, np.zeros(12)
  # Each row pairs the observation the environment computed at the end of the
  # last step (or reset) with the state and command at that same instant.
  # Recomputing instead is wrong: with update_history=False mjlab serves term
  # values from its observation buffers, i.e. as of the last update. So the
  # next row's command is written BEFORE the step whose end observes it; a
  # reset row observes the command the reset sampled.
  plan = [c for c in SEGMENTS for _ in range(args.steps_per_segment)]
  reset_rows = {RESET_BEFORE_SEGMENT * args.steps_per_segment}
  wrapped.reset()
  for i in range(len(plan)):
    if i in reset_rows:
      wrapped.reset()
    obs = env.observation_manager.compute()["actor"][0:1]
    with torch.no_grad():
      action = policy(obs)
    if clip is not None:
      action = torch.clamp(action, -clip, clip)

    k = int(env.episode_length_buf[0])
    q = robot.data.joint_pos[0].cpu().numpy()
    qd = robot.data.joint_vel[0].cpu().numpy()
    a = action[0].cpu().numpy().astype(np.float64)
    row = {
      "episode_start": int(k == 0),
      "k": k,
      **dict(zip(["quat_w", "quat_x", "quat_y", "quat_z"],
                 robot.data.root_link_quat_w[0].cpu().numpy().tolist(), strict=True)),
      **dict(zip(["gyro_x", "gyro_y", "gyro_z"],
                 mdp.builtin_sensor(env, "robot/imu_ang_vel")[0].cpu().numpy().tolist(),
                 strict=True)),
      **dict(zip(["cmd_vx", "cmd_vy", "cmd_wz"],
                 twist_term.vel_command_b[0].cpu().numpy().tolist(), strict=True)),
      **{f"q_{n}": float(q[i]) for n, i in zip(HAL_ORDER, hal_idx, strict=True)},
      **{f"qd_{n}": float(qd[i]) for n, i in zip(HAL_ORDER, hal_idx, strict=True)},
    }
    obs_np = obs[0].cpu().numpy().astype(np.float64)
    where = f"row {len(rows)} (k={k})"
    if k == 0:
      k_ref, prev_a = 0, np.zeros(12)
    if k != k_ref:
      raise SystemExit(f"[conformance] episode step {k} != contract k {k_ref} at {where}")
    _check("observation", obs_np, reference_observation(contract, row, k_ref, prev_a),
           OBS_TOL, where)
    _check("action", evaluate_bin(net, tensors, obs_np[None, :])[0].astype(np.float64),
           a, ACT_TOL, where)

    target = default + scale * a
    row.update({f"obs_{i}": float(v) for i, v in enumerate(obs_np)})
    row.update({f"act_{i}": float(v) for i, v in enumerate(a)})
    row.update({f"qtarget_{n}": float(target[i]) for n, i in zip(HAL_ORDER, hal_idx, strict=True)})
    rows.append(row)
    k_ref, prev_a = k_ref + 1, a
    if i + 1 < len(plan):
      twist_term.vel_command_b[:] = torch.tensor(plan[i + 1], device=env.device)
    wrapped.step(action)

  out = os.path.join(artifact_dir, "conformance.csv")
  with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=header)
    w.writeheader()
    for r in rows:
      # 9 significant digits: exact for float32 values, 1e-9 relative otherwise.
      w.writerow({h: (format(v, ".9g") if isinstance(v, float) else v) for h, v in r.items()})
  starts = sum(r["episode_start"] for r in rows)
  standing = sum(
    1 for r in rows if np.linalg.norm([r["cmd_vx"], r["cmd_vy"], r["cmd_wz"]]) < 0.1
  )
  print(f"[conformance] wrote {len(rows)} rows ({starts} episode starts, "
        f"{standing} below the stand threshold) -> {out}")
  env.close()


if __name__ == "__main__":
  main(tyro.cli(Args))

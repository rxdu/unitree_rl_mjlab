"""Export a trained velocity policy as a policy artifact (contract v1).

The format is specified by the runtime that consumes the artifact, not
here; this module is the producer half. It reads the training environment
and the trained policy and writes `policy.yaml` + `policy.bin`, and it REFUSES
anything contract v1 cannot express -- an artifact that a runtime would
misread is worse than no artifact, because the failure is silent: the right
network runs and the robot falls.

Everything written is read from the live environment and policy rather than
restated, so the contract cannot drift from what was trained.
"""

from __future__ import annotations

import datetime
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.rl.exporter_utils import get_base_metadata

from src.tasks.velocity import mdp

CONTRACT_VERSION = 1
POLICY_YAML = "policy.yaml"
POLICY_BIN = "policy.bin"

# The runtime's joint vocabulary.
_HAL_JOINT_RE = re.compile(r"^(FR|FL|RR|RL)_(hip|thigh|calf)_joint$")
_NUM_JOINTS = 12


class ContractUnsupportedError(ValueError):
  """The environment or policy uses something contract v1 cannot express."""


@dataclass(frozen=True)
class Provenance:
  task: str
  run: str
  checkpoint: str
  seed: int


# --------------------------------------------------------------------------
# Observation terms
# --------------------------------------------------------------------------


def _scalar_scale(name: str, scale: Any) -> float:
  if scale is None:
    return 1.0
  if isinstance(scale, (int, float)):
    return float(scale)
  raise ContractUnsupportedError(
    f"observation term '{name}': only a scalar scale is expressible in "
    f"contract v1, got {type(scale).__name__}"
  )


def _observation_terms(env: ManagerBasedRlEnv, command_name: str) -> list[dict]:
  """Map the actor group's terms onto the contract's closed vocabulary."""
  om = env.observation_manager
  group_cfg = om.cfg["actor"]
  if group_cfg.history_length not in (None, 0, 1):
    raise ContractUnsupportedError(
      f"actor group history_length={group_cfg.history_length}: observation "
      "history is not expressible in contract v1"
    )

  names = om.active_terms["actor"]
  dims = [int(np.prod(d)) for d in om.group_obs_term_dim["actor"]]
  cfgs = [om.get_term_cfg("actor", n) for n in names]
  terms: list[dict] = []
  for name, dim, cfg in zip(names, dims, cfgs, strict=True):
    if cfg.clip is not None:
      raise ContractUnsupportedError(f"term '{name}': clipping is not in v1")
    if cfg.history_length not in (0, 1):
      raise ContractUnsupportedError(f"term '{name}': history is not in v1")
    if cfg.delay_max_lag > 0:
      raise ContractUnsupportedError(
        f"term '{name}': observation delay is not expressible in contract v1"
      )
    entry: dict[str, Any] = {"dim": dim, "scale": _scalar_scale(name, cfg.scale)}
    func, params = cfg.func, cfg.params or {}

    if func is mdp.builtin_sensor and params.get("sensor_name") == "robot/imu_ang_vel":
      entry["name"] = "base_ang_vel"
    elif func is mdp.projected_gravity:
      entry["name"] = "projected_gravity"
    elif func is mdp.generated_commands:
      if params.get("command_name") != command_name:
        raise ContractUnsupportedError(
          f"term '{name}' observes command '{params.get('command_name')}', "
          f"not the velocity command '{command_name}'"
        )
      entry["name"] = "command"
    elif func is mdp.phase:
      if params.get("command_name") != command_name:
        raise ContractUnsupportedError(
          f"phase term masks on command '{params.get('command_name')}', not "
          f"'{command_name}'"
        )
      entry["name"] = "gait_phase"
      entry["period_s"] = float(params["period"])
      entry["stand_threshold"] = float(mdp.PHASE_STAND_THRESHOLD)
    elif func is mdp.joint_pos_rel:
      entry["name"] = "joint_pos_rel"
    elif func is mdp.joint_vel_rel:
      robot = env.scene["robot"]
      if torch.count_nonzero(robot.data.default_joint_vel) != 0:
        raise ContractUnsupportedError(
          "joint_vel_rel with a non-zero default joint velocity is not in v1"
        )
      entry["name"] = "joint_vel_rel"
    elif func is mdp.last_action:
      if params.get("action_name") not in (None, "joint_pos"):
        raise ContractUnsupportedError(f"term '{name}': unexpected action_name")
      entry["name"] = "last_action"
    else:
      raise ContractUnsupportedError(
        f"observation term '{name}' ({getattr(func, '__name__', func)}) is not "
        "in contract v1's observation vocabulary"
      )
    # Keep the canonical key order: name first, then dim and the rest.
    terms.append({"name": entry.pop("name"), **entry})

  expected = {
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "command": 3,
    "gait_phase": 2,
    "joint_pos_rel": _NUM_JOINTS,
    "joint_vel_rel": _NUM_JOINTS,
    "last_action": _NUM_JOINTS,
  }
  for t in terms:
    if t["dim"] != expected[t["name"]]:
      raise ContractUnsupportedError(
        f"term '{t['name']}' has dim {t['dim']}, contract v1 fixes it at "
        f"{expected[t['name']]}"
      )
  return terms


# --------------------------------------------------------------------------
# Joints, action, command
# --------------------------------------------------------------------------


def _joints_and_action(env: ManagerBasedRlEnv, clip_actions: float | None) -> tuple[dict, dict]:
  robot = env.scene["robot"]
  joint_names = list(robot.joint_names)
  if len(joint_names) != _NUM_JOINTS or not all(
    _HAL_JOINT_RE.match(n) for n in joint_names
  ):
    raise ContractUnsupportedError(
      f"joints {joint_names} are not the 12 HAL joints "
      "({FR,FL,RR,RL}_{hip,thigh,calf}_joint)"
    )
  if len(set(joint_names)) != _NUM_JOINTS:
    raise ContractUnsupportedError(f"duplicate joint names: {joint_names}")

  terms = env.action_manager.active_terms
  if list(terms) != ["joint_pos"]:
    raise ContractUnsupportedError(f"expected one action term 'joint_pos', got {terms}")
  action = env.action_manager.get_term("joint_pos")
  if not isinstance(action, JointPositionAction):
    raise ContractUnsupportedError("the 'joint_pos' action is not JointPositionAction")
  # One joint order serves observations and actions in v1. Measured equal for
  # the Go2 on 2026-10-04; checked here rather than assumed.
  if list(action.target_names) != joint_names:
    raise ContractUnsupportedError(
      f"action order {list(action.target_names)} differs from joint order "
      f"{joint_names}; contract v1 has a single policy joint order"
    )

  default_pos = robot.data.default_joint_pos[0].detach().cpu().numpy()
  scale = np.broadcast_to(
    np.asarray(
      action._scale if isinstance(action._scale, float)  # noqa: SLF001
      else action._scale[0].detach().cpu().numpy(),  # noqa: SLF001
      dtype=np.float64,
    ),
    (_NUM_JOINTS,),
  )
  offset = np.broadcast_to(
    np.asarray(
      action._offset if isinstance(action._offset, float)  # noqa: SLF001
      else action._offset[0].detach().cpu().numpy(),  # noqa: SLF001
      dtype=np.float64,
    ),
    (_NUM_JOINTS,),
  )
  # The contract's action is default + scale * a, so the offset must BE the
  # default pose (use_default_offset=True).
  if not np.allclose(offset, default_pos, atol=1e-6):
    raise ContractUnsupportedError(
      "action offset differs from the default joint pose; contract v1 expresses "
      "only joint_position_offset about the default pose"
    )
  if clip_actions is not None and not clip_actions > 0:
    raise ContractUnsupportedError(f"clip_actions={clip_actions} must be > 0")

  meta = get_base_metadata(env, run_path="")
  if list(meta["joint_names"]) != joint_names:
    raise ContractUnsupportedError("ONNX metadata joint order differs from joint order")

  joints = {
    "names": joint_names,
    "default_position_rad": [float(v) for v in default_pos],
    "stiffness_nm_per_rad": [float(v) for v in meta["joint_stiffness"]],
    "damping_nms_per_rad": [float(v) for v in meta["joint_damping"]],
  }
  action_entry = {
    "type": "joint_position_offset",
    "scale": [float(v) for v in scale],
    "clip": None if clip_actions is None else float(clip_actions),
  }
  return joints, action_entry


def _command_ranges(env: ManagerBasedRlEnv, command_name: str) -> dict:
  term = env.command_manager.get_term(command_name)
  ranges = getattr(term.cfg, "ranges", None)
  if ranges is None or not hasattr(term, "vel_command_b"):
    raise ContractUnsupportedError(f"command '{command_name}' is not a velocity command")
  out = {}
  for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
    lo, hi = (float(v) for v in getattr(ranges, key))
    if not lo <= hi:
      raise ContractUnsupportedError(f"command range {key}=({lo}, {hi})")
    out[key] = [lo, hi]
  return out


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------


def _network_tensors(policy: nn.Module, obs_dim: int) -> tuple[list[tuple[str, np.ndarray]], dict]:
  """Flatten the policy into contract tensors: normalizer, then layers."""
  if list(getattr(policy, "obs_groups", ["actor"])) != ["actor"]:
    raise ContractUnsupportedError(
      f"policy reads observation groups {policy.obs_groups}, not just 'actor'"
    )
  det = policy.distribution.as_deterministic_output_module()
  if type(det).__name__ != "_IdentityDeterministicOutput":
    raise ContractUnsupportedError(
      f"deterministic output {type(det).__name__} is not the identity; v1 "
      "assumes the action is the MLP output"
    )

  tensors: list[tuple[str, np.ndarray]] = []
  normalizer = None
  norm = policy.obs_normalizer
  if isinstance(norm, nn.Identity):
    pass
  elif hasattr(norm, "_mean") and hasattr(norm, "_std") and hasattr(norm, "eps"):
    mean = norm._mean.squeeze(0)  # noqa: SLF001
    # Exactly what EmpiricalNormalization.forward divides by, in float32.
    denominator = norm._std.squeeze(0) + norm.eps  # noqa: SLF001
    tensors += [("obs_mean", mean), ("obs_denominator", denominator)]
    normalizer = {"mean": "obs_mean", "denominator": "obs_denominator"}
  else:
    raise ContractUnsupportedError(f"unknown normalizer {type(norm).__name__}")

  modules = list(policy.mlp)
  layers = []
  for i, m in enumerate(modules):
    if isinstance(m, nn.Linear):
      idx = len(layers)
      tensors += [(f"layer{idx}.weight", m.weight), (f"layer{idx}.bias", m.bias)]
      layers.append({"weight": f"layer{idx}.weight", "bias": f"layer{idx}.bias"})
      is_last = i == len(modules) - 1
      if not is_last and not isinstance(modules[i + 1], nn.ELU):
        raise ContractUnsupportedError(
          f"layer {idx} is followed by {type(modules[i + 1]).__name__}; v1 "
          "admits only ELU between linear layers"
        )
    elif isinstance(m, nn.ELU):
      if m.alpha != 1.0:
        raise ContractUnsupportedError(f"ELU alpha={m.alpha}; v1 fixes alpha = 1")
    else:
      raise ContractUnsupportedError(f"MLP module {type(m).__name__} is not in v1")
  if not isinstance(modules[-1], nn.Linear):
    raise ContractUnsupportedError("the MLP must end in a linear layer")

  arrays = [(n, t.detach().to("cpu", torch.float32).numpy().copy()) for n, t in tensors]
  first_in = arrays[2 if normalizer else 0][1].shape[1]
  last_out = arrays[-1][1].shape[0]
  if first_in != obs_dim or last_out != _NUM_JOINTS:
    raise ContractUnsupportedError(
      f"network maps {first_in} -> {last_out}, expected {obs_dim} -> {_NUM_JOINTS}"
    )
  for name, a in arrays:
    if not np.all(np.isfinite(a)):
      raise ContractUnsupportedError(f"tensor '{name}' has non-finite values")
  if normalizer and not np.all(arrays[1][1] > 0):
    raise ContractUnsupportedError("normalizer denominator must be > 0 everywhere")
  network = {"type": "mlp", "activation": "elu", "normalizer": normalizer, "layers": layers}
  return arrays, network


def evaluate_bin(network: dict, tensors: dict[str, np.ndarray], obs: np.ndarray) -> np.ndarray:
  """Reference evaluation of contract tensors, as the runtime must do it.

  float32 throughout. Used for the export-time self-check and by tests.
  """
  x = obs.astype(np.float32)
  if network["normalizer"]:
    x = (x - tensors[network["normalizer"]["mean"]]) / tensors[
      network["normalizer"]["denominator"]
    ]
  layers = network["layers"]
  for i, layer in enumerate(layers):
    x = x @ tensors[layer["weight"]].T + tensors[layer["bias"]]
    if i < len(layers) - 1:
      x = np.where(x > 0, x, np.expm1(np.minimum(x, 0))).astype(np.float32)
  return x


def read_bin(path: str, table: list[dict]) -> dict[str, np.ndarray]:
  raw = np.fromfile(path, dtype="<f4")
  return {
    t["name"]: raw[t["offset"] : t["offset"] + int(np.prod(t["shape"]))].reshape(t["shape"])
    for t in table
  }


# --------------------------------------------------------------------------
# Provenance and export
# --------------------------------------------------------------------------


def _git(*args: str) -> str | None:
  repo = os.path.dirname(os.path.abspath(__file__))
  try:
    return subprocess.run(
      ["git", "-C", repo, *args], capture_output=True, text=True, check=True
    ).stdout.strip()
  except (OSError, subprocess.CalledProcessError):
    return None


def robot_name_for_task(task: str) -> str:
  """`Unitree-Go2-Flat` -> `go2`, the robot name the runtime expects."""
  m = re.match(r"^Unitree-([A-Za-z0-9]+)-", task)
  if not m:
    raise ContractUnsupportedError(f"cannot derive a robot name from task '{task}'")
  return m.group(1).lower()


def export_policy_contract(
  env: ManagerBasedRlEnv,
  policy: nn.Module,
  out_dir: str,
  provenance: Provenance,
  clip_actions: float | None,
  name: str | None = None,
  command_name: str = "twist",
) -> str:
  """Write policy.yaml + policy.bin into `out_dir`. Returns the YAML path.

  Raises ContractUnsupportedError if anything is outside contract v1, and
  RuntimeError if the written weights do not reproduce the policy.
  """
  terms = _observation_terms(env, command_name)
  obs_dim = sum(t["dim"] for t in terms)
  joints, action = _joints_and_action(env, clip_actions)
  arrays, network = _network_tensors(policy, obs_dim)

  table, offset = [], 0
  for tname, a in arrays:
    table.append({"name": tname, "shape": list(a.shape), "offset": offset})
    offset += a.size
  os.makedirs(out_dir, exist_ok=True)
  bin_path = os.path.join(out_dir, POLICY_BIN)
  with open(bin_path, "wb") as f:
    for _, a in arrays:
      f.write(np.ascontiguousarray(a, dtype="<f4").tobytes())

  # Self-check: the bytes on disk must reproduce the trained policy. Inputs are
  # random but scaled like real observations (joint velocities reach tens of
  # rad/s), so the normalizer and every layer are exercised.
  probe = torch.randn(64, obs_dim, generator=torch.Generator().manual_seed(0)) * 3.0
  export_module = policy.as_onnx(verbose=False).to("cpu").eval()
  with torch.no_grad():
    expected = export_module(probe).numpy()
  got = evaluate_bin(network, read_bin(bin_path, table), probe.numpy())
  if not np.allclose(got, expected, atol=1e-5, rtol=1e-5):
    err = float(np.max(np.abs(got - expected)))
    raise RuntimeError(f"policy.bin does not reproduce the policy (max |err| {err:.3e})")

  commit = _git("rev-parse", "HEAD")
  dirty = _git("status", "--porcelain")
  contract = {
    "contract_version": CONTRACT_VERSION,
    "name": name or re.sub(r"[^a-z0-9_]+", "_", provenance.task.lower()).strip("_"),
    "robot": robot_name_for_task(provenance.task),
    "provenance": {
      "producer": "unitree_rl_mjlab",
      "git_commit": commit or "unknown",
      "git_dirty": bool(dirty) if dirty is not None else True,
      "task": provenance.task,
      "run": provenance.run,
      "checkpoint": provenance.checkpoint,
      "seed": int(provenance.seed),
      "exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds"
      ),
    },
    "timing": {"policy_period_s": float(env.step_dt)},
    "joints": joints,
    "action": action,
    "command": {"ranges": _command_ranges(env, command_name)},
    "observation": {"dim": obs_dim, "terms": terms},
    "network": {
      **network,
      "weights_file": POLICY_BIN,
      "weights_size_bytes": os.path.getsize(bin_path),
      "tensors": table,
    },
  }
  yaml_path = os.path.join(out_dir, POLICY_YAML)
  with open(yaml_path, "w") as f:
    f.write(
      "# Policy artifact, contract v1\n"
    )
    yaml.safe_dump(contract, f, sort_keys=False, default_flow_style=None, width=100)
  return yaml_path

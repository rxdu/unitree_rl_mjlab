"""Unit tests for the policy-contract exporter's pure parts.

Run: python -m unittest discover -s tests

The end-to-end check -- a trained policy, its artifact, and conformance
vectors that must reproduce it -- is scripts/export_conformance.py, which
self-checks every row before writing.
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import torch
from rsl_rl.modules.normalization import EmpiricalNormalization
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tasks.velocity.rl.policy_contract import (  # noqa: E402
  ContractUnsupportedError,
  _network_tensors,
  evaluate_bin,
  read_bin,
  robot_name_for_task,
)


class _IdentityDeterministicOutput(nn.Module):
  """Stands in for rsl_rl's Gaussian deterministic output (matched by name)."""

  def forward(self, x):
    return x


class _MeanSliceDeterministicOutput(_IdentityDeterministicOutput):
  pass


class _Distribution:
  def __init__(self, det_cls):
    self._det_cls = det_cls

  def as_deterministic_output_module(self):
    return self._det_cls()


class _FakePolicy(nn.Module):
  """The attributes the exporter reads from rsl_rl's MLPModel, nothing more."""

  def __init__(self, mlp, obs_dim, det_cls=_IdentityDeterministicOutput):
    super().__init__()
    self.obs_groups = ["actor"]
    self.distribution = _Distribution(det_cls)
    self.obs_normalizer = EmpiricalNormalization(obs_dim)
    self.mlp = mlp
    with torch.no_grad():
      self.obs_normalizer._mean.uniform_(-1, 1)
      self.obs_normalizer._std.uniform_(0.1, 3.0)

  def reference(self, x):
    return self.mlp(self.obs_normalizer(x))


def _mlp(obs_dim, act=nn.ELU):
  torch.manual_seed(0)
  return nn.Sequential(nn.Linear(obs_dim, 16), act(), nn.Linear(16, 8), act(), nn.Linear(8, 12))


class NetworkExportTest(unittest.TestCase):
  OBS_DIM = 47

  def test_bin_round_trip_reproduces_the_policy(self):
    policy = _FakePolicy(_mlp(self.OBS_DIM), self.OBS_DIM)
    arrays, network = _network_tensors(policy, self.OBS_DIM)
    self.assertEqual(
      [n for n, _ in arrays],
      ["obs_mean", "obs_denominator", "layer0.weight", "layer0.bias",
       "layer1.weight", "layer1.bias", "layer2.weight", "layer2.bias"],
    )
    table, offset = [], 0
    with tempfile.TemporaryDirectory() as d:
      path = os.path.join(d, "policy.bin")
      with open(path, "wb") as f:
        for name, a in arrays:
          table.append({"name": name, "shape": list(a.shape), "offset": offset})
          offset += a.size
          f.write(a.astype("<f4").tobytes())
      self.assertEqual(os.path.getsize(path), 4 * offset)
      tensors = read_bin(path, table)
    x = torch.randn(32, self.OBS_DIM) * 3.0
    with torch.no_grad():
      expected = policy.reference(x).numpy()
    got = evaluate_bin(network, tensors, x.numpy())
    np.testing.assert_allclose(got, expected, atol=1e-5, rtol=1e-5)

  def test_denominator_is_std_plus_eps(self):
    policy = _FakePolicy(_mlp(self.OBS_DIM), self.OBS_DIM)
    arrays = dict(_network_tensors(policy, self.OBS_DIM)[0])
    expected = (policy.obs_normalizer._std.squeeze(0) + policy.obs_normalizer.eps).numpy()
    np.testing.assert_array_equal(arrays["obs_denominator"], expected)

  def test_refuses_an_activation_outside_v1(self):
    policy = _FakePolicy(_mlp(self.OBS_DIM, act=nn.Tanh), self.OBS_DIM)
    with self.assertRaises(ContractUnsupportedError):
      _network_tensors(policy, self.OBS_DIM)

  def test_refuses_a_non_identity_deterministic_output(self):
    policy = _FakePolicy(_mlp(self.OBS_DIM), self.OBS_DIM, _MeanSliceDeterministicOutput)
    with self.assertRaises(ContractUnsupportedError):
      _network_tensors(policy, self.OBS_DIM)

  def test_refuses_a_width_mismatch(self):
    policy = _FakePolicy(_mlp(self.OBS_DIM), self.OBS_DIM)
    with self.assertRaises(ContractUnsupportedError):
      _network_tensors(policy, self.OBS_DIM + 1)


class RobotNameTest(unittest.TestCase):
  def test_derives_the_runtime_robot_name(self):
    self.assertEqual(robot_name_for_task("Unitree-Go2-Flat"), "go2")

  def test_refuses_an_unrecognised_task(self):
    with self.assertRaises(ContractUnsupportedError):
      robot_name_for_task("Mjlab-Velocity-Flat-Unitree-Go1")


if __name__ == "__main__":
  unittest.main()

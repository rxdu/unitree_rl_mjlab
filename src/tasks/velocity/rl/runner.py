import os
import shutil

import wandb
from rsl_rl.env import VecEnv

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner

from .policy_contract import (
  ContractUnsupportedError,
  Provenance,
  export_policy_contract,
)

# Next to the checkpoints: the policy artifact
# (policy.yaml + policy.bin + a copy of policy.onnx), refreshed on every save.
ARTIFACT_DIR = "policy_artifact"


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    task_id: str | None = None,
  ) -> None:
    super().__init__(env, train_cfg, log_dir, device)
    self.task_id = task_id
    self._contract_unsupported: str | None = None

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    run_name: str = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)
    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
    self._export_contract(path, onnx_path)

  def _export_contract(self, checkpoint_path: str, onnx_path: str) -> None:
    """Write the policy artifact; never let it end a training run.

    An unsupported task (e.g. a rough-terrain policy observing a height scan)
    is reported once and skipped. A self-check failure is reported on every
    save, because it means the artifact would not reproduce the policy.
    """
    if self.task_id is None or self._contract_unsupported is not None:
      return
    run_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    out_dir = os.path.join(run_dir, ARTIFACT_DIR)
    provenance = Provenance(
      task=self.task_id,
      run=os.path.basename(run_dir),
      checkpoint=os.path.basename(checkpoint_path),
      seed=int(self.cfg.get("seed", -1)),
    )
    try:
      export_policy_contract(
        self.env.unwrapped,
        self.alg.get_policy(),
        out_dir,
        provenance,
        clip_actions=self.cfg.get("clip_actions"),
      )
      shutil.copyfile(onnx_path, os.path.join(out_dir, "policy.onnx"))
    except ContractUnsupportedError as e:
      self._contract_unsupported = str(e)
      print(f"[INFO] policy artifact not exported for {self.task_id}: {e}")
    except RuntimeError as e:
      print(f"[ERROR] policy artifact export failed for {self.task_id}: {e}")

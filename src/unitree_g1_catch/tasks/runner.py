"""RSL-RL runner for transferring the tracking actor into stage two."""

from pathlib import Path

from mjlab.rl import MjlabOnPolicyRunner


class ThrowOnPolicyRunner(MjlabOnPolicyRunner):
  """Warm-start only the deployable actor; learn a new critic and optimizer."""

  def __init__(
    self,
    env,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    pretrained = train_cfg.get("pretrained_actor_checkpoint")
    if pretrained and train_cfg.get("resume"):
      raise ValueError(
        "Use either pretrained_actor_checkpoint for a new stage-two run or "
        "resume for a stage-two checkpoint, not both."
      )
    super().__init__(env, train_cfg, log_dir, device)
    if pretrained:
      checkpoint = Path(pretrained).expanduser().resolve()
      if not checkpoint.is_file():
        raise FileNotFoundError(f"Pretrained actor checkpoint not found: {checkpoint}")
      self.load(
        str(checkpoint),
        load_cfg={
          "actor": True,
          "critic": False,
          "optimizer": False,
          "iteration": False,
          "rnd": False,
        },
        strict=True,
        map_location=device,
      )
      self.current_learning_iteration = 0
      self.env.unwrapped.common_step_counter = 0
      print(f"[INFO] Warm-started stage-two actor from: {checkpoint}")

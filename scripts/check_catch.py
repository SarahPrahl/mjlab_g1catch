"""CPU regression check: .venv/bin/python scripts/check_catch.py."""

import re
from types import SimpleNamespace

import torch

from unitree_g1_catch.assets.g1_inspire import get_spec
from unitree_g1_catch.mdp.commands import BallThrowCommand
from unitree_g1_catch.mdp.rewards import finger_contact_reward, palm_contact_reward
from unitree_g1_catch.tasks.env_cfg import unitree_g1_inspire_throw_env_cfg


class GraspFixture(BallThrowCommand):
  robot_anchor_pos_w = torch.zeros(3, 3)
  robot_anchor_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1)
  ball_pos_w = torch.tensor([[0.4, 0.0, 0.1], [0.4, 0.0, 0.371], [0.4, 0.0, 0.1]])
  ball_palm_distance = torch.zeros(3)
  ball_hand_relative_speed = torch.zeros(3)
  catch_pose_score = torch.ones(3)
  synthetic_weight = torch.ones(3)

  def _contact_state(self):
    return torch.ones(3, dtype=torch.bool), torch.tensor([True, True, False])


def main():
  cfg = unitree_g1_inspire_throw_env_cfg()
  command = object.__new__(GraspFixture)
  command.cfg = cfg.commands["throw"]
  assert command.cfg.rehearsal_fraction == 0.6
  for name in (
    "launched",
    "left_palm_contact",
    "right_palm_contact",
    "stable_grasp",
    "caught",
    "finger_grip_latch",
  ):
    setattr(command, name, torch.zeros(3, dtype=torch.bool))
  command.launched[:] = True
  command.hold_steps = torch.zeros(3)
  command.required_hold_steps = torch.full((3,), 2.0)
  command.contact_loss_steps = torch.zeros(3, dtype=torch.long)
  command.finger_release_steps = torch.zeros(3, dtype=torch.long)
  command._contact_grace_steps = 4
  ids = torch.arange(3)
  command._update_grasp(ids)
  assert command.stable_grasp.tolist() == [True, False, False]
  command._update_grasp(ids)
  assert command.caught.tolist() == [True, False, False]
  # Exiting the box resets an unfinished hold, despite contact grace.
  command.caught[:] = False
  command.ball_pos_w = command.ball_pos_w.clone()
  command.ball_pos_w[0, 2] = 0.371
  command._update_grasp(ids)
  assert command.hold_steps.tolist() == [0.0, 0.0, 0.0]
  command.ball_pos_w[0, 2] = 0.1
  command._update_grasp(ids)
  sensors = {
    name: SimpleNamespace(data=SimpleNamespace(found=torch.ones(3, 5)))
    for name in ("left", "right")
  }
  env = SimpleNamespace(
    scene=sensors, command_manager=SimpleNamespace(get_term=lambda _: command)
  )
  score = finger_contact_reward(env, "throw", "left", "right", 2)
  assert score.tolist() == [1.0, 0.0, 0.0]
  sensors["right"].data.found.zero_()
  assert finger_contact_reward(env, "throw", "left", "right", 2)[0] == 0.5
  sensors["left"].data.found.zero_()
  assert finger_contact_reward(env, "throw", "left", "right", 2).sum() == 0
  assert palm_contact_reward(env, "throw")[1] == 0
  # Each sensor must resolve to five distal bodies with active colliders.
  spec = get_spec()
  for sensor in cfg.scene.sensors:
    if sensor.name in ("left_finger_ball_contact", "right_finger_ball_contact"):
      bodies = [b for b in spec.bodies if re.fullmatch(sensor.primary.pattern, b.name)]
      assert len(bodies) == 5
      assert all(
        any(
          g.parent.name == b.name and (g.contype or g.conaffinity) for g in spec.geoms
        )
        for b in bodies
      )
  print("PASS: box gate, hold reset, bilateral palm/finger reward, sensors, rehearsal")


if __name__ == "__main__":
  main()

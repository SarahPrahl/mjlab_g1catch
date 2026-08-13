"""Named boundary between the mjlab policy and Unitree hardware interfaces."""

from __future__ import annotations

import torch

# This is the combined MJCF's natural actuated-joint order. mjlab uses it for the
# actor's joint observations, previous action, and raw policy output.
POLICY_ACTION_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "l_ih_thumb_proximal_yaw",
  "l_ih_thumb_proximal_pitch",
  "l_ih_index_proximal",
  "l_ih_middle_proximal",
  "l_ih_ring_proximal",
  "l_ih_pinky_proximal",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
  "r_ih_thumb_proximal_yaw",
  "r_ih_thumb_proximal_pitch",
  "r_ih_index_proximal",
  "r_ih_middle_proximal",
  "r_ih_ring_proximal",
  "r_ih_pinky_proximal",
)

# Unitree G1 low-level motor IDs 0..28, as used by the official deployment config.
G1_MOTOR_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)

# Unitree's RH56DFX DDS order: right pinky/ring/middle/index/bend/rotation,
# followed by the same six motors for the left hand.
DFX_INSPIRE_JOINT_NAMES = (
  "r_ih_pinky_proximal",
  "r_ih_ring_proximal",
  "r_ih_middle_proximal",
  "r_ih_index_proximal",
  "r_ih_thumb_proximal_pitch",
  "r_ih_thumb_proximal_yaw",
  "l_ih_pinky_proximal",
  "l_ih_ring_proximal",
  "l_ih_middle_proximal",
  "l_ih_index_proximal",
  "l_ih_thumb_proximal_pitch",
  "l_ih_thumb_proximal_yaw",
)


def _indices(target_names: tuple[str, ...]) -> tuple[int, ...]:
  source_index = {name: index for index, name in enumerate(POLICY_ACTION_JOINT_NAMES)}
  return tuple(source_index[name] for name in target_names)


G1_MOTOR_POLICY_INDICES = _indices(G1_MOTOR_JOINT_NAMES)
DFX_INSPIRE_POLICY_INDICES = _indices(DFX_INSPIRE_JOINT_NAMES)


def split_policy_actions(actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Reorder raw policy actions into G1 motor-ID and DFX hand-DDS order.

  These are still normalized policy actions. The deployment controller must apply
  the same per-joint scale, default-position offset, clipping, and encoder-bias
  convention as mjlab before sending position targets to either device.
  """
  if actions.shape[-1] != len(POLICY_ACTION_JOINT_NAMES):
    raise ValueError(
      f"Expected {len(POLICY_ACTION_JOINT_NAMES)} actions, got {actions.shape[-1]}."
    )
  body = actions[..., list(G1_MOTOR_POLICY_INDICES)]
  hands = actions[..., list(DFX_INSPIRE_POLICY_INDICES)]
  return body, hands

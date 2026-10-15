from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.entity import EntityCfg

from unitree_g1_catch.assets.g1_inspire import (
  G1_INSPIRE_ACTION_SCALE,
  get_g1_inspire_robot_cfg,
)


class Humanoid:
  """A simple humanoid model for a general humanoid robot.
  Purpose: to provide a common interface for different humanoid robots, such as the Unitree G1 and G1 Inspire.
  """

  def __init__(self):
    """Initialize the humanoid."""
    pass

  def get_cfg(self) -> EntityCfg:
    """Return a fresh EntityCfg for this humanoid."""
    raise NotImplementedError("Subclasses must implement get_cfg()")

  # ========================== GETTING SPECIFIC JOINT GROUPS ==========================
  @property
  def joint_names(self) -> tuple[str, ...]:
    """Return the all actuated joint names for this humanoid."""
    raise NotImplementedError("Subclasses must implement joint_names()")

  @property
  def actuated_joints(self) -> tuple[str, ...]:
    """Return the regex patterns of all actuated joints for this humanoid."""
    raise NotImplementedError("Subclasses must implement actuated_joints()")

  @property
  def arm_joints(self) -> tuple[str, ...]:
    """Return the regex patterns of all arm joints for this humanoid."""
    raise NotImplementedError("Subclasses must implement arm_joints()")

  @property
  def arm_ready_joints(self) -> tuple[str, ...]:
    """Return the regex patterns of all arm ready joints (excluding elbow joints) for this humanoid."""
    raise NotImplementedError("Subclasses must implement arm_ready_joints()")

  @property
  def elbow_joints(self) -> tuple[str, ...]:
    """Return the regex patterns of all elbow joints for this humanoid."""
    raise NotImplementedError("Subclasses must implement elbow_joints()")

  @property
  def finger_joints(self) -> tuple[str, ...]:
    """Return the regex patterns of all finger joints for this humanoid."""
    raise NotImplementedError("Subclasses must implement finger_joints()")

  @property
  def lower_body_joints(self) -> tuple[str, ...]:
    """Return the regex patterns of all lower body joints for this humanoid."""
    raise NotImplementedError("Subclasses must implement lower_body_joints()")

  # ========================== GETTING THE ACTION SCALE FOR ALL JOINTS ==========================
  @property
  def action_scale(self) -> dict[str, float]:
    """Return the action scale for this humanoid."""
    raise NotImplementedError("Subclasses must implement action_scale()")

  # ========================== GETTING SPECIFIC GEOM COLLISION GROUPS ==========================
  @property
  def foot_collisions(self) -> tuple[str, ...]:
    """Return the regex patterns offoot collision geometries for this humanoid."""
    raise NotImplementedError("Subclasses must implement foot_collisions()")

  @property
  def palm_collisions(self) -> tuple[str, ...]:
    """Return the regex patterns of all palm collision geometries for this humanoid."""
    raise NotImplementedError("Subclasses must implement palm_collisions()")

  @property
  def palm_collisions_by_hand(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return left and right palm collision patterns."""
    raise NotImplementedError("Subclasses must implement palm_collisions_by_hand()")

  # ========================== GETTING SPECIFIC BODY GROUPS ==========================
  @property
  def motion_tracking_bodies(self) -> tuple[str, ...]:
    """Return the  body names used for motion tracking for this humanoid."""
    raise NotImplementedError("Subclasses must implement motion_tracking_bodies()")

  @property
  def anchor_body(self) -> str:
    """Return the name of the anchor body (typically torso) for this humanoid."""
    raise NotImplementedError("Subclasses must implement anchor_body()")

  @property
  def root_body(self) -> str:
    """Return the root body used for self-contact filtering."""
    raise NotImplementedError("Subclasses must implement root_body()")

  @property
  def ee_bodies(self) -> tuple[str, ...]:
    """Return the end-effector body names for this humanoid."""
    raise NotImplementedError("Subclasses must implement ee_bodies()")

  @property
  def finger_bodies(self) -> tuple[str, ...]:
    """Return the regex patterns of finger body names for this humanoid."""
    raise NotImplementedError("Subclasses must implement finger_bodies()")

  @property
  def finger_bodies_by_hand(self) -> tuple[str, ...]:
    """Return left and right finger-body patterns."""
    raise NotImplementedError("Subclasses must implement finger_bodies_by_hand()")

  @property
  def palm_bodies(self) -> tuple[str, ...]:
    """Return the palm body names for this humanoid."""
    raise NotImplementedError("Subclasses must implement palm_bodies()")

  @property
  def hand_bodies(self) -> tuple[str, ...]:
    """Return the regex patterns of hand body names for this humanoid."""
    raise NotImplementedError("Subclasses must implement hand_bodies()")

  # ========================== GETTING SPECIFIC SITE GROUPS ==========================
  @property
  def palm_sites(self) -> tuple[str, ...]:
    """Return the palm site names for this humanoid."""
    raise NotImplementedError("Subclasses must implement palm_sites()")

  @property
  def finger_grip_targets(self) -> dict[str, float]:
    """Return finger-joint target values used by grasp posture rewards."""
    return {}


class UnitreeG1(Humanoid):
  """Class to store the configuration of the Unitree G1 humanoid robot."""

  def get_cfg(self) -> EntityCfg:
    """Return a fresh EntityCfg for this humanoid."""
    return get_g1_robot_cfg()

  # ========================== GETTING SPECIFIC JOINT GROUPS ==========================
  @property
  def joint_names(self) -> tuple[str, ...]:
    return (
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

  @property
  def actuated_joints(self) -> tuple[str, ...]:
    return (".*_joint",)

  @property
  def arm_joints(self) -> tuple[str, ...]:
    return (
      ".*_shoulder_.*_joint",
      ".*_elbow_joint",
      ".*_wrist_.*_joint",
    )

  @property
  def arm_ready_joints(self) -> tuple[str, ...]:
    return (
      ".*_shoulder_.*_joint",
      ".*_wrist_.*_joint",
    )

  @property
  def elbow_joints(self) -> tuple[str, ...]:
    return (".*_elbow_joint",)

  @property
  def finger_joints(self) -> tuple[str, ...]:
    return ()

  @property
  def lower_body_joints(self) -> tuple[str, ...]:
    return (
      "waist_.*_joint",
      ".*_hip_.*_joint",
      ".*_knee_joint",
      ".*_ankle_.*_joint",
    )

  # ========================== GETTING THE ACTION SCALE FOR ALL JOINTS ==========================
  @property
  def action_scale(self) -> dict[str, float]:
    return G1_ACTION_SCALE

  # ========================== GETTING SPECIFIC GEOM COLLISION GROUPS ==========================
  @property
  def foot_collisions(self) -> tuple[str, ...]:
    return ("(left|right)_foot[1-7]_collision",)

  @property
  def palm_collisions(self) -> tuple[str, ...]:
    return ("(left|right)_hand_collision",)

  @property
  def palm_collisions_by_hand(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (("left_hand_collision",), ("right_hand_collision",))

  # ========================== GETTING SPECIFIC BODY GROUPS ==========================
  @property
  def motion_tracking_bodies(self) -> tuple[str, ...]:
    return (
      "pelvis",
      "left_hip_roll_link",
      "left_knee_link",
      "left_ankle_roll_link",
      "right_hip_roll_link",
      "right_knee_link",
      "right_ankle_roll_link",
      "torso_link",
      "left_shoulder_roll_link",
      "left_elbow_link",
      "left_wrist_yaw_link",
      "right_shoulder_roll_link",
      "right_elbow_link",
      "right_wrist_yaw_link",
    )

  @property
  def anchor_body(self) -> str:
    return "torso_link"

  @property
  def root_body(self) -> str:
    return "pelvis"

  @property
  def ee_bodies(self) -> tuple[str, ...]:
    return (
      "left_ankle_roll_link",
      "right_ankle_roll_link",
      "left_wrist_yaw_link",
      "right_wrist_yaw_link",
    )

  @property
  def finger_bodies(self) -> tuple[str, ...]:
    return ()

  @property
  def finger_bodies_by_hand(self) -> tuple[str, ...]:
    return ()

  @property
  def palm_bodies(self) -> tuple[str, ...]:
    return ("left_wrist_yaw_link", "right_wrist_yaw_link")

  @property
  def hand_bodies(self) -> tuple[str, ...]:
    return self.palm_bodies

  # ========================== GETTING SPECIFIC SITE GROUPS ==========================
  @property
  def palm_sites(self) -> tuple[str, ...]:
    return ("left_palm", "right_palm")


class UnitreeG1Inspire(UnitreeG1):
  """Class to store the configuration of the Unitree G1 humanoid robot with Inspire Hands.
  Inherits from UnitreeG1 and overrides methods to include Inspire hand-specific configurations.
  """

  def get_cfg(self) -> EntityCfg:
    """Return a fresh EntityCfg for this humanoid."""
    return get_g1_inspire_robot_cfg()

  # ========================== GETTING SPECIFIC JOINT GROUPS ==========================
  @property
  def joint_names(self) -> tuple[str, ...]:
    return super().joint_names + (
      "l_ih_thumb_proximal_yaw",
      "l_ih_thumb_proximal_pitch",
      "l_ih_thumb_intermediate",
      "l_ih_thumb_distal",
      "l_ih_index_proximal",
      "l_ih_index_intermediate",
      "l_ih_middle_proximal",
      "l_ih_middle_intermediate",
      "l_ih_ring_proximal",
      "l_ih_ring_intermediate",
      "l_ih_pinky_proximal",
      "l_ih_pinky_intermediate",
      "r_ih_thumb_proximal_yaw",
      "r_ih_thumb_proximal_pitch",
      "r_ih_thumb_intermediate",
      "r_ih_thumb_distal",
      "r_ih_index_proximal",
      "r_ih_index_intermediate",
      "r_ih_middle_proximal",
      "r_ih_middle_intermediate",
      "r_ih_ring_proximal",
      "r_ih_ring_intermediate",
      "r_ih_pinky_proximal",
      "r_ih_pinky_intermediate",
    )

  @property
  def actuated_joints(self) -> tuple[str, ...]:
    return super().actuated_joints + (
      "[lr]_ih_thumb_proximal_(yaw|pitch)",
      "[lr]_ih_(index|middle|ring|pinky)_proximal",
    )

  @property
  def finger_joints(self) -> tuple[str, ...]:
    return (
      "[lr]_ih_thumb_proximal_(yaw|pitch)",
      "[lr]_ih_(index|middle|ring|pinky)_proximal",
    )

  # ========================== GETTING THE ACTION SCALE FOR ALL JOINTS ==========================
  @property
  def action_scale(self) -> dict[str, float]:
    return G1_INSPIRE_ACTION_SCALE

  # ========================== GETTING SPECIFIC GEOM COLLISION GROUPS ==========================
  @property
  def foot_collisions(self) -> tuple[str, ...]:
    return ("(left|right)_ankle_roll_link_collision_.*",)

  @property
  def palm_collisions(self) -> tuple[str, ...]:
    return ("(l|r)_ih_hand_base_collision_[5-6]",)

  @property
  def palm_collisions_by_hand(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (
      (r"l_ih_hand_base_collision_[5-6]$",),
      (r"r_ih_hand_base_collision_[5-6]$",),
    )

  # ========================== GETTING SPECIFIC BODY GROUPS ==========================
  @property
  def motion_tracking_bodies(self) -> tuple[str, ...]:
    return super().motion_tracking_bodies + (
      "l_ih_hand_base",
      "l_ih_thumb_distal",
      "l_ih_index_intermediate",
      "l_ih_middle_intermediate",
      "r_ih_hand_base",
      "r_ih_thumb_distal",
      "r_ih_index_intermediate",
      "r_ih_middle_intermediate",
    )

  @property
  def finger_bodies(self) -> tuple[str, ...]:
    return ("[lr]_ih_(thumb_distal|(index|middle|ring|pinky)_intermediate)",)

  @property
  def finger_bodies_by_hand(self) -> tuple[str, ...]:
    return (
      r"l_ih_(thumb_distal|(index|middle|ring|pinky)_intermediate)$",
      r"r_ih_(thumb_distal|(index|middle|ring|pinky)_intermediate)$",
    )

  @property
  def palm_bodies(self) -> tuple[str, ...]:
    return ("l_ih_hand_base", "r_ih_hand_base")

  @property
  def hand_bodies(self) -> tuple[str, ...]:
    return ("[lr]_ih_(thumb_distal|(index|middle|ring|pinky)_intermediate|hand_base)",)

  # ========================== GETTING SPECIFIC SITE GROUPS ==========================
  @property
  def palm_sites(self) -> tuple[str, ...]:
    return ("left_palm_site", "right_palm_site")

  @property
  def finger_grip_targets(self) -> dict[str, float]:
    return {
      ".*_ih_thumb_proximal_yaw": 0.3,
      ".*_ih_thumb_proximal_pitch": 0.0,
      ".*_ih_index_proximal": 0.4,
      ".*_ih_middle_proximal": 0.4,
      ".*_ih_ring_proximal": 0.4,
      ".*_ih_pinky_proximal": 0.45,
    }


# if __name__ == "__main__":
#     g1_inspire = UnitreeG1Inspire()
#     print("G1 Inspire actuated joints:", g1_inspire.finger_bodies)

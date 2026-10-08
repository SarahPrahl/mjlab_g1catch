"""Unitree G1 with Inspire-hand constants and model loading."""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  G1_ACTION_SCALE,
  G1_ACTUATOR_4010,
  G1_ACTUATOR_5020,
  G1_ACTUATOR_7520_14,
  G1_ACTUATOR_7520_22,
  G1_ACTUATOR_ANKLE,
  G1_ACTUATOR_WAIST,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
G1_INSPIRE_XML = (
  REPOSITORY_ROOT / "third_party" / "unitree_g1_inspire" / "g1_29dof_inspire_hand.xml"
)

FINGER_ACTUATOR_EXPR = (
  "[lr]_ih_thumb_proximal_(yaw|pitch)",
  "[lr]_ih_(index|middle|ring|pinky)_proximal",
)


def get_spec() -> mujoco.MjSpec:
  """Load and sanitize the vendored G1 + Inspire MJCF."""
  if not G1_INSPIRE_XML.exists():
    raise FileNotFoundError(
      f"Missing Inspire-hand submodule asset: {G1_INSPIRE_XML}. "
      "Run `git submodule update --init --recursive`."
    )

  spec = mujoco.MjSpec.from_file(str(G1_INSPIRE_XML))

  # The source MJCF uses torque motors. mjlab supplies position actuators below,
  # matching the stock G1 tracking task and giving the policy joint targets.
  for actuator in list(spec.actuators):
    spec.delete(actuator)

  # Hand links contain both a visible mesh and a transparent collision mesh.
  # Disable contact on the duplicate visible mesh, retaining the collision mesh
  # and palm primitives. Active hand colliders live in geometry group 3 and are
  # orange so Viser can reveal the exact physics geometry without showing it by
  # default (geometry groups 3-5 start hidden).
  geom_counts: dict[str, int] = {}
  for geom in spec.geoms:
    body_name = geom.parent.name
    is_hand = body_name.startswith(("l_ih_", "r_ih_"))
    is_visible_contact_mesh = (
      is_hand
      and geom.type == mujoco.mjtGeom.mjGEOM_MESH
      and (geom.contype != 0 or geom.conaffinity != 0)
      and geom.rgba[3] > 0.1
    )
    if is_visible_contact_mesh:
      geom.contype = 0
      geom.conaffinity = 0

    if is_hand and (geom.contype != 0 or geom.conaffinity != 0):
      geom.group = 3
      geom.rgba = (1.0, 0.25, 0.0, 0.35)

    geom_counts[body_name] = geom_counts.get(body_name, 0) + 1
    kind = "collision" if geom.contype != 0 or geom.conaffinity != 0 else "visual"
    geom.name = f"{body_name}_{kind}_{geom_counts[body_name]}"

  # Sites at the center of each palm are used only for reward shaping.
  spec.body("l_ih_hand_base").add_site(
    name="left_palm_site", pos=(0.0, -0.075, 0.0), size=(0.008,)
  )
  spec.body("r_ih_hand_base").add_site(
    name="right_palm_site", pos=(0.0, -0.075, 0.0), size=(0.008,)
  )
  return spec


FINGER_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=FINGER_ACTUATOR_EXPR,
  stiffness=8.0,
  damping=0.2,
  effort_limit=1.0,
  armature=0.005,
  frictionloss=0.05,
)

G1_INSPIRE_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14,
    G1_ACTUATOR_7520_22,
    G1_ACTUATOR_4010,
    G1_ACTUATOR_WAIST,
    G1_ACTUATOR_ANKLE,
    FINGER_ACTUATOR,
  ),
  soft_joint_pos_limit_factor=0.9,
)

G1_INSPIRE_HOME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.76),
  joint_pos={
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
    "l_ih_index_intermediate": -0.04545,
    "l_ih_middle_intermediate": -0.04545,
    "l_ih_ring_intermediate": -0.04545,
    "l_ih_pinky_intermediate": -0.04545,
    "r_ih_index_intermediate": -0.04545,
    "r_ih_middle_intermediate": -0.04545,
    "r_ih_ring_intermediate": -0.04545,
    "r_ih_pinky_intermediate": -0.04545,
  },
  joint_vel={".*": 0.0},
)


def get_g1_inspire_robot_cfg() -> EntityCfg:
  """Return a fresh G1 + Inspire-hand robot configuration."""
  return EntityCfg(
    init_state=G1_INSPIRE_HOME,
    spec_fn=get_spec,
    articulation=G1_INSPIRE_ARTICULATION,
  )


G1_INSPIRE_ACTION_SCALE = dict(G1_ACTION_SCALE)
for expr in FINGER_ACTUATOR_EXPR:
  G1_INSPIRE_ACTION_SCALE[expr] = 0.5

if __name__ == "__main__":
  # View the G1 Inspire MJCF in a standalone viewer for debugging.
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_g1_inspire_robot_cfg())

  viewer.launch(robot.spec.compile())

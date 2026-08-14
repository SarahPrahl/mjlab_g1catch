"""Ball-conditioned Unitree G1 + Inspire-hand tracking environment."""

import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking import mdp as tracking_mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.terrains import (
  BoxFlatTerrainCfg,
  HfWaveTerrainCfg,
  TerrainEntityCfg,
  TerrainGeneratorCfg,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from unitree_g1_catch import mdp as catch_mdp
from unitree_g1_catch.assets import (
  G1_INSPIRE_ACTION_SCALE,
  get_g1_inspire_robot_cfg,
)
from unitree_g1_catch.mdp import CatchMotionCommandCfg

BALL_DIAMETER_M = 0.15
BALL_RADIUS_M = BALL_DIAMETER_M / 2.0
BALL_MASS_KG = 0.112

ACTUATED_JOINTS = (
  ".*_joint",
  "[lr]_ih_thumb_proximal_(yaw|pitch)",
  "[lr]_ih_(index|middle|ring|pinky)_proximal",
)


def _training_terrain_cfg() -> TerrainEntityCfg:
  """Keep most environments flat and add millimetre-scale floor variation."""
  return TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
      curriculum=True,
      size=(4.0, 4.0),
      num_rows=4,
      color_scheme="height",
      sub_terrains={
        "flat": BoxFlatTerrainCfg(proportion=0.75),
        "slight_undulations": HfWaveTerrainCfg(
          proportion=0.25,
          amplitude_range=(0.001, 0.003),
          num_waves=2,
          horizontal_scale=0.05,
          vertical_scale=0.0005,
          border_width=0.25,
        ),
      },
    ),
  )


def get_ball_spec() -> mujoco.MjSpec:
  """Create the 15 cm, 112 g dynamic ball specified for this task."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="ball", pos=(100.0, 0.0, 100.0))
  body.add_freejoint(name="ball_freejoint")
  body.add_geom(
    name="ball_collision",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=(BALL_RADIUS_M, 0.0, 0.0),
    mass=BALL_MASS_KG,
    rgba=(0.95, 0.45, 0.08, 1.0),
    friction=(0.8, 0.01, 0.001),
    condim=4,
    priority=1,
  )
  return spec


def _catch_motion_cfg(source: MotionCommandCfg) -> CatchMotionCommandCfg:
  return CatchMotionCommandCfg(
    entity_name=source.entity_name,
    ball_entity_name="ball",
    resampling_time_range=source.resampling_time_range,
    debug_vis=source.debug_vis,
    pose_range=source.pose_range,
    velocity_range=source.velocity_range,
    joint_position_range=source.joint_position_range,
    motion_file=source.motion_file,
    anchor_body_name="torso_link",
    body_names=(
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
      "l_ih_hand_base",
      "l_ih_thumb_distal",
      "l_ih_index_intermediate",
      "l_ih_middle_intermediate",
      "right_shoulder_roll_link",
      "right_elbow_link",
      "right_wrist_yaw_link",
      "r_ih_hand_base",
      "r_ih_thumb_distal",
      "r_ih_index_intermediate",
      "r_ih_middle_intermediate",
    ),
    adaptive_kernel_size=source.adaptive_kernel_size,
    adaptive_lambda=source.adaptive_lambda,
    adaptive_uniform_ratio=source.adaptive_uniform_ratio,
    adaptive_alpha=source.adaptive_alpha,
    sampling_mode=source.sampling_mode,
    viz=source.viz,
  )


def unitree_g1_inspire_catch_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the reactive G1 + Inspire ball-catching task."""
  cfg = make_tracking_env_cfg()
  if not play:
    cfg.scene.terrain = _training_terrain_cfg()
  cfg.scene.entities = {
    "robot": get_g1_inspire_robot_cfg(),
    "ball": EntityCfg(spec_fn=get_ball_spec),
  }

  self_collision = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  hand_ball_sensors = tuple(
    ContactSensorCfg(
      name=f"{side}_ball_contact",
      primary=ContactMatch(
        mode="subtree", pattern=f"{prefix}_ih_hand_base", entity="robot"
      ),
      secondary=ContactMatch(mode="body", pattern="ball", entity="ball"),
      fields=("found", "force"),
      reduce="maxforce",
      num_slots=1,
      history_length=4,
    )
    for side, prefix in (("left", "l"), ("right", "r"))
  )
  cfg.scene.sensors = (self_collision, *hand_ball_sensors)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_INSPIRE_ACTION_SCALE

  source_motion = cfg.commands["motion"]
  assert isinstance(source_motion, MotionCommandCfg)
  motion_cmd = _catch_motion_cfg(source_motion)
  cfg.commands["motion"] = motion_cmd

  actor_joint_cfg = SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS)
  actor_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=tracking_mdp.base_lin_vel,
      noise=Unoise(n_min=-0.15, n_max=0.15),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=tracking_mdp.base_ang_vel,
      noise=Unoise(n_min=-0.1, n_max=0.1),
    ),
    "projected_gravity": ObservationTermCfg(
      func=tracking_mdp.projected_gravity,
      noise=Unoise(n_min=-0.03, n_max=0.03),
    ),
    "anchor_height": ObservationTermCfg(
      func=catch_mdp.robot_anchor_height,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_pos": ObservationTermCfg(
      func=tracking_mdp.joint_pos_rel,
      params={"asset_cfg": actor_joint_cfg, "biased": True},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=tracking_mdp.joint_vel_rel,
      params={"asset_cfg": actor_joint_cfg},
      noise=Unoise(n_min=-0.25, n_max=0.25),
    ),
    "last_action": ObservationTermCfg(func=tracking_mdp.last_action),
    "ball_position": ObservationTermCfg(
      func=catch_mdp.ball_position_b,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "ball_velocity": ObservationTermCfg(
      func=catch_mdp.ball_velocity_b,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.1, n_max=0.1),
    ),
    "ball_visible": ObservationTermCfg(
      func=catch_mdp.ball_visible, params={"command_name": "motion"}
    ),
  }
  cfg.observations["actor"] = ObservationGroupCfg(
    terms=actor_terms,
    concatenate_terms=True,
    enable_corruption=True,
  )
  cfg.observations["critic"].terms["ball_state"] = ObservationTermCfg(
    func=catch_mdp.ball_position_b, params={"command_name": "motion"}
  )
  cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
    func=tracking_mdp.base_lin_vel
  )
  cfg.observations["critic"].terms["base_ang_vel"] = ObservationTermCfg(
    func=tracking_mdp.base_ang_vel
  )
  cfg.observations["critic"].terms["ball_velocity"] = ObservationTermCfg(
    func=catch_mdp.ball_velocity_b, params={"command_name": "motion"}
  )
  cfg.observations["critic"].terms["reference_ball_state"] = ObservationTermCfg(
    func=catch_mdp.reference_ball_state_b, params={"command_name": "motion"}
  )

  cfg.rewards["ball_position_tracking"] = RewardTermCfg(
    func=catch_mdp.ball_position_tracking_exp,
    weight=1.0,
    params={"command_name": "motion", "std": 0.25},
  )
  cfg.rewards["ball_velocity_tracking"] = RewardTermCfg(
    func=catch_mdp.ball_velocity_tracking_exp,
    weight=0.25,
    params={"command_name": "motion", "std": 1.5},
  )
  cfg.rewards["palm_ball_proximity"] = RewardTermCfg(
    func=catch_mdp.palm_ball_proximity_exp,
    weight=1.0,
    params={
      "command_name": "motion",
      "std": 0.3,
      "asset_cfg": SceneEntityCfg(
        "robot", site_names=("left_palm_site", "right_palm_site")
      ),
    },
  )
  cfg.rewards["ball_contact"] = RewardTermCfg(
    func=catch_mdp.ball_contact_reward,
    weight=2.0,
    params={
      "command_name": "motion",
      "left_sensor_name": "left_ball_contact",
      "right_sensor_name": "right_ball_contact",
    },
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_ankle_roll_link_collision_.*$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )

  cfg.viewer.body_name = "torso_link"
  cfg.viewer.distance = 3.2
  cfg.sim.nconmax = 256
  cfg.sim.njmax = 2048

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.joint_position_range = (0.0, 0.0)
    motion_cmd.sampling_mode = "start"

  return cfg

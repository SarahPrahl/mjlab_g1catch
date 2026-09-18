"""Ball-conditioned Unitree G1 + Inspire-hand tracking environment."""

from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
  CurriculumTermCfg,
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  TerminationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking import mdp as tracking_mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.terrains import (
  BoxFlatTerrainCfg,
  SubTerrainCfg,
  TerrainEntityCfg,
  TerrainGeneratorCfg,
)
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from unitree_g1_catch import mdp as catch_mdp
from unitree_g1_catch.assets import (
  G1_INSPIRE_ACTION_SCALE,
  get_g1_inspire_robot_cfg,
)
from unitree_g1_catch.mdp import (
  BallThrowCommandCfg,
  CatchMotionCommandCfg,
  RehearsalThrowCommandCfg,
)

BALL_DIAMETER_M = 0.15
BALL_RADIUS_M = BALL_DIAMETER_M / 2.0
BALL_MASS_KG = 0.112
BALL_CONTACT_SOLREF = (0.02, 1.2)
BALL_CONTACT_SOLIMP = (0.3, 0.9, 0.01, 0.5, 2.0)

ACTUATED_JOINTS = (
  ".*_joint",
  "[lr]_ih_thumb_proximal_(yaw|pitch)",
  "[lr]_ih_(index|middle|ring|pinky)_proximal",
)

ARM_JOINTS = (
  ".*_shoulder_.*_joint",
  ".*_elbow_joint",
  ".*_wrist_.*_joint",
)

ARM_READY_JOINTS = (
  ".*_shoulder_.*_joint",
  ".*_wrist_.*_joint",
)

ELBOW_JOINTS = (".*_elbow_joint",)

FINGER_JOINTS = (
  "[lr]_ih_thumb_proximal_(yaw|pitch)",
  "[lr]_ih_(index|middle|ring|pinky)_proximal",
)

LOWER_BODY_JOINTS = (
  "waist_.*_joint",
  ".*_hip_.*_joint",
  ".*_knee_joint",
  ".*_ankle_.*_joint",
)


@dataclass(kw_only=True)
class _BoxUnevenTerrainCfg(SubTerrainCfg):
  """Millimetre-scale uneven floor made from GPU-friendly box tiles."""

  amplitude_range: tuple[float, float] = (0.001, 0.003)
  tiles_per_side: int = 4

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    amplitude = self.amplitude_range[0] + difficulty * (
      self.amplitude_range[1] - self.amplitude_range[0]
    )
    heights = rng.uniform(
      -amplitude, amplitude, (self.tiles_per_side, self.tiles_per_side)
    )
    tile_size = np.asarray(self.size) / self.tiles_per_side
    depth = 0.05
    body = spec.body("terrain")
    geometries: list[TerrainGeometry] = []
    for row in range(self.tiles_per_side):
      for col in range(self.tiles_per_side):
        height = heights[row, col]
        geom = body.add_geom(
          type=mujoco.mjtGeom.mjGEOM_BOX,
          size=(tile_size[0] / 2, tile_size[1] / 2, depth / 2),
          pos=(
            (row + 0.5) * tile_size[0],
            (col + 0.5) * tile_size[1],
            height - depth / 2,
          ),
        )
        geometries.append(TerrainGeometry(geom=geom, color=(0.5, 0.5, 0.5, 1.0)))

    center = self.tiles_per_side // 2
    center_height = np.max(
      heights[center - 1 : center + 1, center - 1 : center + 1], axis=(0, 1)
    )
    origin = np.array([self.size[0] / 2, self.size[1] / 2, center_height])
    return TerrainOutput(origin=origin, geometries=geometries)


def _training_terrain_cfg() -> TerrainEntityCfg:
  """Create flat and millimetre-scale uneven patches for the curriculum."""
  return TerrainEntityCfg(
    terrain_type="generator",
    max_init_terrain_level=0,
    terrain_generator=TerrainGeneratorCfg(
      curriculum=True,
      size=(4.0, 4.0),
      num_rows=4,
      color_scheme="height",
      sub_terrains={
        "flat": BoxFlatTerrainCfg(proportion=0.5),
        "slight_undulations": _BoxUnevenTerrainCfg(
          proportion=0.5,
          amplitude_range=(0.001, 0.003),
          tiles_per_side=4,
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
    solref=BALL_CONTACT_SOLREF,
    solimp=BALL_CONTACT_SOLIMP,
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


def _actor_observation_terms(
  command_name: str, occlude_in_catch_zone: bool = False
) -> dict[str, ObservationTermCfg]:
  """Return the deployment observation interface shared by both training stages."""
  actor_joint_cfg = SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS)
  return {
    "base_ang_vel": ObservationTermCfg(
      func=tracking_mdp.base_ang_vel,
      noise=Unoise(n_min=-0.1, n_max=0.1),
    ),
    "projected_gravity": ObservationTermCfg(
      func=tracking_mdp.projected_gravity,
      noise=Unoise(n_min=-0.03, n_max=0.03),
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
      params={
        "command_name": command_name,
        "occlude_in_catch_zone": occlude_in_catch_zone,
      },
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "ball_velocity": ObservationTermCfg(
      func=catch_mdp.ball_velocity_b,
      params={
        "command_name": command_name,
        "occlude_in_catch_zone": occlude_in_catch_zone,
      },
      noise=Unoise(n_min=-0.1, n_max=0.1),
    ),
    "ball_visible": ObservationTermCfg(
      func=catch_mdp.ball_visible,
      params={
        "command_name": command_name,
        "occlude_in_catch_zone": occlude_in_catch_zone,
      },
    ),
  }


def unitree_g1_inspire_catch_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the reactive G1 + Inspire ball-catching task."""
  cfg = make_tracking_env_cfg()
  if not play:
    cfg.scene.terrain = _training_terrain_cfg()
    cfg.curriculum["terrain_mix"] = CurriculumTermCfg(
      func=catch_mdp.terrain_mix,
      params={"flat_steps": 60_000, "curriculum_steps": 240_000},
    )
  cfg.scene.entities = {
    "robot": get_g1_inspire_robot_cfg(),
    "ball": EntityCfg(spec_fn=get_ball_spec),
  }

  self_collision = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(
      mode="body",
      pattern=".*",
      entity="robot",
      exclude=("[lr]_ih_.*",),
    ),
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
  cfg.rewards["joint_limit"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=(".*_joint",)
  )

  source_motion = cfg.commands["motion"]
  assert isinstance(source_motion, MotionCommandCfg)
  motion_cmd = _catch_motion_cfg(source_motion)
  cfg.commands["motion"] = motion_cmd

  cfg.observations["actor"] = ObservationGroupCfg(
    terms=_actor_observation_terms("motion"),
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


def unitree_g1_inspire_throw_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create stage-two catching with procedurally launched physical balls."""
  cfg = unitree_g1_inspire_catch_env_cfg(play=play)
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, CatchMotionCommandCfg)
  rehearsal_reward_names = (
    "motion_global_root_pos",
    "motion_global_root_ori",
    "motion_body_pos",
    "motion_body_ori",
    "motion_body_lin_vel",
    "motion_body_ang_vel",
    "ball_position_tracking",
    "ball_velocity_tracking",
    "palm_ball_proximity",
    "ball_contact",
  )
  rehearsal_rewards = {name: cfg.rewards[name] for name in rehearsal_reward_names}
  rehearsal_terminations = {
    name: cfg.terminations[name]
    for name in ("anchor_pos", "anchor_ori", "ee_body_pos")
  }

  palm_sensors = tuple(
    ContactSensorCfg(
      name=f"{side}_palm_ball_contact",
      primary=ContactMatch(
        mode="geom",
        pattern=rf"{prefix}_ih_hand_base_collision_[5-6]$",
        entity="robot",
      ),
      secondary=ContactMatch(mode="body", pattern="ball", entity="ball"),
      fields=("found", "force", "normal", "pos"),
      reduce="maxforce",
      num_slots=1,
      history_length=4,
    )
    for side, prefix in (("left", "l"), ("right", "r"))
  )
  finger_sensors = tuple(
    ContactSensorCfg(
      name=f"{side}_finger_ball_contact",
      primary=ContactMatch(
        mode="body",
        pattern=rf"{prefix}_ih_(thumb_distal|(index|middle|ring|pinky)_intermediate)$",
        entity="robot",
      ),
      secondary=ContactMatch(mode="body", pattern="ball", entity="ball"),
      fields=("found",),
      reduce="maxforce",
      num_slots=1,
    )
    for side, prefix in (("left", "l"), ("right", "r"))
  )
  body_ball_sensor = ContactSensorCfg(
    name="body_ball_contact",
    primary=ContactMatch(
      mode="body",
      pattern=".*",
      entity="robot",
      exclude=("[lr]_ih_.*",),
    ),
    secondary=ContactMatch(mode="body", pattern="ball", entity="ball"),
    fields=("found",),
    reduce="maxforce",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (
    *cfg.scene.sensors,
    *palm_sensors,
    *finger_sensors,
    body_ball_sensor,
  )

  throw_cmd = (
    BallThrowCommandCfg(
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
    )
    if play
    else RehearsalThrowCommandCfg(
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      motion_cfg=motion_cmd,
    )
  )
  if play:
    # Playback samples the complete trained distribution immediately.
    throw_cmd.initial_azimuth_rad = throw_cmd.final_azimuth_rad
    throw_cmd.initial_target_y_range_m = throw_cmd.final_target_y_range_m
    throw_cmd.initial_target_z_range_m = throw_cmd.final_target_z_range_m
    throw_cmd.initial_distance_max_m = throw_cmd.final_distance_max_m
    throw_cmd.initial_hold_time_s = throw_cmd.final_hold_time_s
  cfg.commands = {"throw": throw_cmd}

  cfg.observations["actor"] = ObservationGroupCfg(
    terms=_actor_observation_terms("throw", occlude_in_catch_zone=True),
    concatenate_terms=True,
    enable_corruption=not play,
  )
  critic_joint_cfg = SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS)
  cfg.observations["critic"] = ObservationGroupCfg(
    terms={
      "base_lin_vel": ObservationTermCfg(func=tracking_mdp.base_lin_vel),
      "base_ang_vel": ObservationTermCfg(func=tracking_mdp.base_ang_vel),
      "projected_gravity": ObservationTermCfg(func=tracking_mdp.projected_gravity),
      "anchor_height": ObservationTermCfg(
        func=catch_mdp.robot_anchor_height,
        params={"command_name": "throw"},
      ),
      "joint_pos": ObservationTermCfg(
        func=tracking_mdp.joint_pos_rel,
        params={"asset_cfg": critic_joint_cfg, "biased": True},
      ),
      "joint_vel": ObservationTermCfg(
        func=tracking_mdp.joint_vel_rel,
        params={"asset_cfg": critic_joint_cfg},
      ),
      "last_action": ObservationTermCfg(func=tracking_mdp.last_action),
      "ball_position": ObservationTermCfg(
        func=catch_mdp.ball_position_b,
        params={"command_name": "throw"},
      ),
      "ball_velocity": ObservationTermCfg(
        func=catch_mdp.ball_velocity_b,
        params={"command_name": "throw"},
      ),
      "ball_visible": ObservationTermCfg(
        func=catch_mdp.ball_visible,
        params={"command_name": "throw"},
      ),
      "throw_state": ObservationTermCfg(
        func=tracking_mdp.generated_commands,
        params={"command_name": "throw"},
      ),
    },
    concatenate_terms=True,
    enable_corruption=False,
  )

  body_joint_cfg = SceneEntityCfg("robot", joint_names=(".*_joint",))
  arm_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINTS)
  arm_ready_cfg = SceneEntityCfg("robot", joint_names=ARM_READY_JOINTS)
  elbow_cfg = SceneEntityCfg("robot", joint_names=ELBOW_JOINTS)
  finger_cfg = SceneEntityCfg("robot", joint_names=FINGER_JOINTS)
  lower_body_cfg = SceneEntityCfg("robot", joint_names=LOWER_BODY_JOINTS)
  miss_params = {
    "command_name": "throw",
    "minimum_ball_height_m": BALL_RADIUS_M + 0.01,
    "maximum_ball_distance_m": 3.5,
  }
  cfg.rewards = {
    "standing_posture": RewardTermCfg(
      func=catch_mdp.standing_posture_exp,
      weight=3.5,
      params={
        "command_name": "throw",
        "target_height_m": 0.76,
        "tilt_std": 0.35,
        "height_std": 0.12,
        "posture_std": 0.7,
        "asset_cfg": lower_body_cfg,
      },
    ),
    "base_motion": RewardTermCfg(
      func=catch_mdp.base_motion_l2,
      weight=-1.0,
      params={
        "command_name": "throw",
        "position_std": 0.2,
        "velocity_std": 0.3,
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "controlled_arm_posture": RewardTermCfg(
      func=catch_mdp.controlled_arm_posture_exp,
      weight=2.0,
      params={
        "command_name": "throw",
        "ready_std": 0.4,
        "elbow_target_rad": 0.25,
        "elbow_std": 0.4,
        "velocity_std": 1.0,
        "ready_cfg": arm_ready_cfg,
        "elbow_cfg": elbow_cfg,
        "arm_cfg": arm_cfg,
      },
    ),
    "arm_motion": RewardTermCfg(
      func=catch_mdp.arm_motion_l2,
      weight=-2.0,
      params={
        "command_name": "throw",
        "velocity_limit_rad_s": 2.0,
        "acceleration_limit_rad_s2": 20.0,
        "asset_cfg": arm_cfg,
      },
    ),
    "finger_posture": RewardTermCfg(
      func=catch_mdp.finger_posture_exp,
      weight=1.0,
      params={
        "command_name": "throw",
        "open_std": 0.15,
        "grip_std": 0.25,
        "velocity_limit_rad_s": 1.0,
        "velocity_std": 0.5,
        "grip_targets": {
          ".*_ih_thumb_proximal_yaw": 0.3,
          ".*_ih_thumb_proximal_pitch": 0.0,
          ".*_ih_index_proximal": 0.4,
          ".*_ih_middle_proximal": 0.4,
          ".*_ih_ring_proximal": 0.4,
          ".*_ih_pinky_proximal": 0.45,
        },
        "asset_cfg": finger_cfg,
      },
    ),
    "palm_contact": RewardTermCfg(
      func=catch_mdp.palm_contact_reward,
      weight=4.0,
      params={"command_name": "throw"},
    ),
    "finger_contact": RewardTermCfg(
      func=catch_mdp.finger_contact_reward,
      weight=4.0,
      params={
        "command_name": "throw",
        "left_sensor_name": "left_finger_ball_contact",
        "right_sensor_name": "right_finger_ball_contact",
        "contacts_per_hand": 2,
      },
    ),
    "interception": RewardTermCfg(
      func=catch_mdp.interception_exp,
      weight=1.5,
      params={
        "command_name": "throw",
        "speed_std": 0.5,
        "distance_std": 0.35,
      },
    ),
    "catch_pose": RewardTermCfg(
      func=catch_mdp.catch_pose_reward,
      weight=2.0,
      params={"command_name": "throw"},
    ),
    "stable_grasp": RewardTermCfg(
      func=catch_mdp.stable_grasp_reward,
      weight=5.0,
      params={"command_name": "throw"},
    ),
    "catch_success": RewardTermCfg(
      func=catch_mdp.catch_success_reward,
      weight=10.0,
      params={"command_name": "throw"},
    ),
    "body_ball_contact": RewardTermCfg(
      func=catch_mdp.body_ball_contact_penalty,
      weight=-3.0,
      params={"command_name": "throw", "sensor_name": "body_ball_contact"},
    ),
    "joint_torques_l2": RewardTermCfg(
      func=tracking_mdp.joint_torques_l2,
      weight=-1.0e-5,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "joint_limit": RewardTermCfg(
      func=tracking_mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": body_joint_cfg},
    ),
    "self_collisions": RewardTermCfg(
      func=tracking_mdp.self_collision_cost,
      weight=-10.0,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
  }
  if not play:
    for name, term in rehearsal_rewards.items():
      params = {key: value for key, value in term.params.items() if key != "command_name"}
      cfg.rewards[f"rehearsal_{name}"] = RewardTermCfg(
        func=catch_mdp.rehearsal_reward,
        weight=term.weight,
        params={
          "command_name": "throw",
          "reward_func": term.func,
          **params,
        },
      )

  cfg.terminations = {
    "time_out": TerminationTermCfg(func=tracking_mdp.time_out, time_out=True),
    "fallen_orientation": TerminationTermCfg(
      func=tracking_mdp.bad_orientation,
      params={"limit_angle": 0.8, "asset_cfg": SceneEntityCfg("robot")},
    ),
    "fallen_height": TerminationTermCfg(
      func=tracking_mdp.root_height_below_minimum,
      params={"minimum_height": 0.45, "asset_cfg": SceneEntityCfg("robot")},
    ),
    "caught": TerminationTermCfg(
      func=catch_mdp.catch_succeeded,
      params={"command_name": "throw"},
    ),
    "missed": TerminationTermCfg(func=catch_mdp.throw_missed, params=miss_params),
  }
  if not play:
    for name, term in rehearsal_terminations.items():
      params = {key: value for key, value in term.params.items() if key != "command_name"}
      cfg.terminations[f"rehearsal_{name}"] = TerminationTermCfg(
        func=catch_mdp.rehearsal_termination,
        params={
          "command_name": "throw",
          "termination_func": term.func,
          **params,
        },
      )
  cfg.events["reset_scene"] = EventTermCfg(
    func=tracking_mdp.reset_scene_to_default,
    mode="reset",
  )
  if "push_robot" in cfg.events:
    cfg.events["push_robot"].params["velocity_range"] = {
      "x": (-0.2, 0.2),
      "y": (-0.2, 0.2),
      "z": (-0.05, 0.05),
      "roll": (-0.1, 0.1),
      "pitch": (-0.1, 0.1),
      "yaw": (-0.2, 0.2),
    }

  cfg.episode_length_s = 6.0
  cfg.is_finite_horizon = True
  return cfg

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
  Humanoid,
  UnitreeG1,
  UnitreeG1Inspire,
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


def _catch_motion_cfg(
  source: MotionCommandCfg, humanoid: Humanoid
) -> CatchMotionCommandCfg:
  """Create a motion command generator configuration for the catching task for the selected humanoid."""
  return CatchMotionCommandCfg(
    entity_name=source.entity_name,
    ball_entity_name="ball",
    resampling_time_range=source.resampling_time_range,
    debug_vis=source.debug_vis,
    pose_range=source.pose_range,
    velocity_range=source.velocity_range,
    joint_position_range=source.joint_position_range,
    motion_file=source.motion_file,
    anchor_body_name=humanoid.anchor_body,
    body_names=humanoid.motion_tracking_bodies,
    adaptive_kernel_size=source.adaptive_kernel_size,
    adaptive_lambda=source.adaptive_lambda,
    adaptive_uniform_ratio=source.adaptive_uniform_ratio,
    adaptive_alpha=source.adaptive_alpha,
    sampling_mode=source.sampling_mode,
    humanoid=humanoid,
    viz=source.viz,
  )


def _actor_observation_terms(
  command_name: str,
  humanoid: Humanoid,
  occlude_in_catch_zone: bool = False,
) -> dict[str, ObservationTermCfg]:
  """Return the deployment observation interface shared by both training stages."""
  actor_joint_cfg = SceneEntityCfg(
    "robot", joint_names=humanoid.actuated_joints
  )
  return {
    # Root link angular velocity in body frame
    "base_ang_vel": ObservationTermCfg(
      func=tracking_mdp.base_ang_vel,
      noise=Unoise(n_min=-0.1, n_max=0.1),
    ),
    # Gravity vector projected into body frame
    "projected_gravity": ObservationTermCfg(
      func=tracking_mdp.projected_gravity,
      noise=Unoise(n_min=-0.03, n_max=0.03),
    ),
    # Joint positions of all actuated joints, relative to the initial pose of the robot, with small noise to improve robustness.
    "joint_pos": ObservationTermCfg(
      func=tracking_mdp.joint_pos_rel,
      params={"asset_cfg": actor_joint_cfg, "biased": True},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    # Joint velocities of all actuated joints, relative to the initial pose of the robot, with small noise to improve robustness.
    "joint_vel": ObservationTermCfg(
      func=tracking_mdp.joint_vel_rel,
      params={"asset_cfg": actor_joint_cfg},
      noise=Unoise(n_min=-0.25, n_max=0.25),
    ),
    # Action taken in the previous step
    "last_action": ObservationTermCfg(func=tracking_mdp.last_action),
    # Ball position relative to the robot anchor, in the anchor frame.
    "ball_position": ObservationTermCfg(
      func=catch_mdp.ball_position_b,
      params={
        "command_name": command_name,
        "occlude_in_catch_zone": occlude_in_catch_zone,
      },
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    # Ball velocity relative to the robot anchor, in the anchor frame.
    "ball_velocity": ObservationTermCfg(
      func=catch_mdp.ball_velocity_b,
      params={
        "command_name": command_name,
        "occlude_in_catch_zone": occlude_in_catch_zone,
      },
      noise=Unoise(n_min=-0.1, n_max=0.1),
    ),
    # Wheather the ball can currently be tracked
    "ball_visible": ObservationTermCfg(
      func=catch_mdp.ball_visible,
      params={
        "command_name": command_name,
        "occlude_in_catch_zone": occlude_in_catch_zone,
      },
    ),
    # Whether the robot is currently in a catching state
    "catch_enabled": ObservationTermCfg(
      func=catch_mdp.catch_enabled,
      params={"command_name": command_name},
    ),
  }


def unitree_g1_catch_env_cfg(
  play: bool = False,
  inspire: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the reactive G1 ball-catching task, optionally with Inspire hands."""
  cfg = make_tracking_env_cfg()
  if not play:
    cfg.scene.terrain = _training_terrain_cfg()
    cfg.curriculum["terrain_mix"] = CurriculumTermCfg(
      func=catch_mdp.terrain_mix,
      params={"flat_steps": 60_000, "curriculum_steps": 240_000},
    )
  humanoid = UnitreeG1Inspire() if inspire else UnitreeG1()
  robot = humanoid.get_cfg()
  cfg.scene.entities = {
    "robot": robot,
    "ball": EntityCfg(spec_fn=get_ball_spec),
  }

  self_collision = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(
      mode="body",
      pattern=".*",
      entity="robot",
      exclude=humanoid.finger_bodies,
    ),
    secondary=ContactMatch(mode="subtree", pattern=humanoid.root_body, entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  hand_ball_sensors = tuple(
    ContactSensorCfg(
      name=f"{side}_ball_contact",
      primary=ContactMatch(
        mode="geom",
        pattern=patterns[0],
        entity="robot",
      ),
      secondary=ContactMatch(mode="body", pattern="ball", entity="ball"),
      fields=("found", "force"),
      reduce="maxforce",
      num_slots=1,
      history_length=4,
    )
    for side, patterns in zip(
      ("left", "right"), humanoid.palm_collisions_by_hand, strict=True
    )
  )
  cfg.scene.sensors = (self_collision, *hand_ball_sensors)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = humanoid.action_scale
  cfg.rewards["joint_limit"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=humanoid.actuated_joints
  )

  source_motion = cfg.commands["motion"]
  assert isinstance(source_motion, MotionCommandCfg)
  motion_cmd = _catch_motion_cfg(source_motion, humanoid)
  cfg.commands["motion"] = motion_cmd

  cfg.observations["actor"] = ObservationGroupCfg(
    terms=_actor_observation_terms("motion", humanoid),
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
  palm_cfg = (
    SceneEntityCfg("robot", site_names=humanoid.palm_sites)
    if humanoid.palm_sites
    else SceneEntityCfg("robot", geom_names=humanoid.palm_collisions)
  )
  cfg.rewards["palm_ball_proximity"] = RewardTermCfg(
    func=catch_mdp.palm_ball_proximity_exp,
    weight=1.0,
    params={
      "command_name": "motion",
      "std": 0.3,
      "asset_cfg": palm_cfg,
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

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = humanoid.foot_collisions
  cfg.events["base_com"].params["asset_cfg"].body_names = (humanoid.anchor_body,)
  cfg.terminations["ee_body_pos"].params["body_names"] = humanoid.ee_bodies

  cfg.viewer.body_name = humanoid.anchor_body
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


def unitree_g1_throw_env_cfg(
  play: bool = False,
  inspire: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create stage-two catching with procedurally launched physical balls."""
  cfg = unitree_g1_catch_env_cfg(play=play, inspire=inspire)
  humanoid = UnitreeG1Inspire() if inspire else UnitreeG1()
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
    name: cfg.terminations[name] for name in ("anchor_pos", "anchor_ori", "ee_body_pos")
  }

  palm_sensors = tuple(
    ContactSensorCfg(
      name=f"{side}_palm_ball_contact",
      primary=ContactMatch(
        mode="geom",
        pattern=patterns[0],
        entity="robot",
      ),
      secondary=ContactMatch(mode="body", pattern="ball", entity="ball"),
      fields=("found", "force", "normal", "pos"),
      reduce="maxforce",
      num_slots=1,
      history_length=4,
    )
    for side, patterns in zip(
      ("left", "right"), humanoid.palm_collisions_by_hand, strict=True
    )
  )
  finger_sensors = ()
  body_ball_sensor = ()
  if humanoid.finger_joints:
    finger_sensors = tuple(
      ContactSensorCfg(
        name=f"{side}_finger_ball_contact",
        primary=ContactMatch(
          mode="body",
          pattern=pattern,
          entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="ball", entity="ball"),
        fields=("found",),
        reduce="maxforce",
        num_slots=1,
      )
      for side, pattern in zip(
        ("left", "right"), humanoid.finger_bodies_by_hand, strict=True
      )
    )
  body_ball_sensor = (
    ContactSensorCfg(
      name="body_ball_contact",
      primary=ContactMatch(
        mode="body",
        pattern=".*",
        entity="robot",
        exclude=humanoid.finger_bodies,
      ),
      secondary=ContactMatch(mode="body", pattern="ball", entity="ball"),
      fields=("found",),
      reduce="maxforce",
      num_slots=1,
      history_length=4,
    ),
  )
  cfg.scene.sensors = (
    *cfg.scene.sensors,
    *palm_sensors,
    *finger_sensors,
    *body_ball_sensor,
  )

  throw_cmd = (
    BallThrowCommandCfg(
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      anchor_body_name=humanoid.anchor_body,
      hand_body_names=humanoid.palm_bodies,
      palm_site_names=humanoid.palm_sites,
    )
    if play
    else RehearsalThrowCommandCfg(
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      motion_cfg=motion_cmd,
      anchor_body_name=humanoid.anchor_body,
      hand_body_names=humanoid.palm_bodies,
      palm_site_names=humanoid.palm_sites,
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
    terms=_actor_observation_terms(
      "throw", humanoid, occlude_in_catch_zone=True
    ),
    concatenate_terms=True,
    enable_corruption=not play,
  )
  critic_joint_cfg = SceneEntityCfg(
    "robot", joint_names=humanoid.actuated_joints
  )
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

  body_joint_cfg = SceneEntityCfg("robot", joint_names=humanoid.actuated_joints)
  arm_cfg = SceneEntityCfg("robot", joint_names=humanoid.arm_joints)
  arm_ready_cfg = SceneEntityCfg("robot", joint_names=humanoid.arm_ready_joints)
  elbow_cfg = SceneEntityCfg("robot", joint_names=humanoid.elbow_joints)
  finger_cfg = SceneEntityCfg(
    "robot",
    joint_names=humanoid.finger_joints,
  )
  lower_body_cfg = SceneEntityCfg("robot", joint_names=humanoid.lower_body_joints)
  miss_params = {
    "command_name": "throw",
    "minimum_ball_height_m": BALL_RADIUS_M + 0.01,
    "maximum_ball_distance_m": 3.5,
  }
  cfg.rewards = {
    "idle_posture": RewardTermCfg(
      func=catch_mdp.idle_posture_exp,
      weight=2.0,
      params={
        "command_name": "throw",
        "asset_cfg": arm_cfg,
        "posture_std": 0.4,
        "velocity_std": 1.0,
      },
    ),
    "ball_released": RewardTermCfg(
      func=catch_mdp.ball_released_reward,
      weight=2.0,
      params={"command_name": "throw"},
    ),
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
    "palm_contact": RewardTermCfg(
      func=catch_mdp.palm_contact_reward,
      weight=4.0,
      params={"command_name": "throw"},
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
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=humanoid.actuated_joints)},
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
  if humanoid.finger_joints:
    cfg.rewards.update(
      {
        "finger_posture": RewardTermCfg(
          func=catch_mdp.finger_posture_exp,
          weight=1.0,
          params={
            "command_name": "throw",
            "open_std": 0.15,
            "grip_std": 0.25,
            "velocity_limit_rad_s": 1.0,
            "velocity_std": 0.5,
            "grip_targets": humanoid.finger_grip_targets,
            "asset_cfg": finger_cfg,
          },
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
      }
    )
  if not play:
    for name, term in rehearsal_rewards.items():
      params = {
        key: value for key, value in term.params.items() if key != "command_name"
      }
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
    "release_finished": TerminationTermCfg(
      func=catch_mdp.release_finished,
      params={"command_name": "throw"},
    ),
    "missed": TerminationTermCfg(func=catch_mdp.throw_missed, params=miss_params),
  }
  if not play:
    cfg.terminations["rehearsal_time_out"] = TerminationTermCfg(
      func=catch_mdp.rehearsal_time_out,
      time_out=True,
      params={"command_name": "throw", "duration_s": 6.0},
    )
    for name, term in rehearsal_terminations.items():
      params = {
        key: value for key, value in term.params.items() if key != "command_name"
      }
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

  cfg.episode_length_s = 14.0
  cfg.is_finite_horizon = True
  return cfg

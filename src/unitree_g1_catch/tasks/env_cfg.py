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
  TerminationTermCfg,
)
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

from unitree_g1_catch.tasks.obs_cfgs import catch_observation_cfg
from unitree_g1_catch.tasks.reward_cfgs import catch_reward_cfg

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

def unitree_g1_catch_env_cfg(
  play: bool = False,
  inspire: bool = False,
  obs_version: int = 0,
  reward_version: int = 0,
) -> ManagerBasedRlEnvCfg:
  """Create the reactive G1 ball-catching task, optionally with Inspire hands."""
  # Get Mjlab configuration for basic imitation learning (tracking)
  cfg = make_tracking_env_cfg()

  # For training grow uneven terrain
  if not play:
    cfg.scene.terrain = _training_terrain_cfg()
    cfg.curriculum["terrain_mix"] = CurriculumTermCfg(
      func=catch_mdp.terrain_mix,
      params={"flat_steps": 60_000, "curriculum_steps": 240_000},
    )

  # Define the robot and set robot and ball into the scene
  humanoid = UnitreeG1Inspire() if inspire else UnitreeG1()
  robot = humanoid.get_cfg()
  cfg.scene.entities = {
    "robot": robot,
    "ball": EntityCfg(spec_fn=get_ball_spec),
  }

  # -------------------------------- Define Sensors -----------------------------------------
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

  # Configure oint actions and limits
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = humanoid.action_scale
  cfg.rewards["joint_limit"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=humanoid.actuated_joints
  )

  # Define the motion command generator
  source_motion = cfg.commands["motion"]
  assert isinstance(source_motion, MotionCommandCfg)
  motion_cmd = _catch_motion_cfg(source_motion, humanoid)
  cfg.commands["motion"] = motion_cmd

  # Define the observation space
  cfg.observations = catch_observation_cfg(
    command_name="motion",
    humanoid=humanoid,
    version=obs_version,
    prev_critic_obs=cfg.observations["critic"].terms,
  )

  # print("Critic observation terms:", cfg.observations["critic"].terms.keys())

  # Define the reward configuration
  cfg.rewards = catch_reward_cfg(
    command_name="motion",
    humanoid=humanoid,
    version=reward_version,
    keep_prev=True,
    play=play,
    prev_rewards=cfg.rewards,
  )

  # Define Events
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = humanoid.foot_collisions
  cfg.events["base_com"].params["asset_cfg"].body_names = (humanoid.anchor_body,)
  cfg.terminations["ee_body_pos"].params["body_names"] = humanoid.ee_bodies

  # Sim and viewer parameters
  cfg.viewer.body_name = humanoid.anchor_body
  cfg.viewer.distance = 3.2
  cfg.sim.nconmax = 256
  cfg.sim.njmax = 2048

  # Configure playing the policy
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
  obs_version: int = 0,
  reward_version: int = 0,
) -> ManagerBasedRlEnvCfg:
  """Create stage-two catching with procedurally launched physical balls."""
  # Load the Stage 1 Configuration
  cfg = unitree_g1_catch_env_cfg(play=play, inspire=inspire, obs_version=obs_version, reward_version=reward_version)

  # Get robot and motion command
  humanoid = UnitreeG1Inspire() if inspire else UnitreeG1()
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, CatchMotionCommandCfg)

  rehearsal_terminations = {
    name: cfg.terminations[name] for name in ("anchor_pos", "anchor_ori", "ee_body_pos")
  }

  # -------------------------------- Define Sensors -----------------------------------------
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

  # -------------------------------- Define Throw Command -----------------------------------------
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

  # Define the observation space
  cfg.observations = catch_observation_cfg(
    command_name="throw",
    humanoid=humanoid,
    version=obs_version,
    occlude_in_catch_zone=True,
  )

  # Define the rewards
  cfg.rewards = catch_reward_cfg(
    command_name="throw",
    humanoid=humanoid,
    version=reward_version,
    keep_prev=False,
    play=play,
    prev_rewards=cfg.rewards,
  )

  # Define the terminations
  miss_params = {
    "command_name": "throw",
    "minimum_ball_height_m": BALL_RADIUS_M + 0.01,
    "maximum_ball_distance_m": 3.5,
  }

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

  # Define the events
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

from mjlab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from mjlab.tasks.tracking import mdp as tracking_mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from unitree_g1_catch import mdp as catch_mdp
from unitree_g1_catch.assets import Humanoid

# Lists of actor observations for differnt observation versions, shared between both phases
ACTOR_OBSERVATION_TERMS = [
  # v0
  [
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "last_action",
    "ball_position",
    "ball_velocity",
    "ball_visible",
    "catch_enabled",
  ],
  # v1
  [
    "base_lin_vel",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "ball_position",
    "ball_velocity",
    "ball_visible",
    "catch_enabled",
  ],
]

# Critic observations for Phase 1
CRITIC_OBSERVATION_TERMS_CATCH = [
  # v0
  [
    "ball_position",
    "base_lin_vel",
    "base_ang_vel",
    "ball_velocity",
    "reference_ball_state",
  ],
  # v1
  [
    "ball_position",
    "base_lin_vel",
    "base_ang_vel",
    "ball_velocity",
    "reference_ball_state",
  ],
]

# Critic observations for Phase 2
CRITIC_OBSERVATION_TERMS_THROW = [
  # v0
  [
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "anchor_height",
    "joint_pos",
    "joint_vel",
    "last_action",
    "ball_position",
    "ball_velocity",
    "ball_visible",
    "throw_state",
  ],
  # v1
  [
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "anchor_height",
    "joint_pos",
    "joint_vel",
    "last_action",
    "ball_position",
    "ball_velocity",
    "ball_visible",
    "throw_state",
  ],
]


def all_observations_dict(
  command_name: str,
  humanoid: Humanoid,
  occlude_in_catch_zone: bool = False,
  noise: bool = True,
) -> dict[str, ObservationTermCfg]:
  """
  Return a dictionary with all Observation Term Configurations possibly used for catching and random throws.
  """
  joint_cfg = SceneEntityCfg("robot", joint_names=humanoid.actuated_joints)

  return {
    # Root link linear velocity in body frame
    "base_lin_vel": ObservationTermCfg(
      func=tracking_mdp.base_lin_vel,
      noise=Unoise(n_min=-0.1, n_max=0.1) if noise else None,
    ),
    # Root link angular velocity in body frame
    "base_ang_vel": ObservationTermCfg(
      func=tracking_mdp.base_ang_vel,
      noise=Unoise(n_min=-0.1, n_max=0.1) if noise else None,
    ),
    # Gravity vector projected into body frame
    "projected_gravity": ObservationTermCfg(
      func=tracking_mdp.projected_gravity,
      noise=Unoise(n_min=-0.03, n_max=0.03) if noise else None,
    ),
    # Joint positions of all actuated joints, relative to the initial pose of the robot, with small noise to improve robustness.
    "joint_pos": ObservationTermCfg(
      func=tracking_mdp.joint_pos_rel,
      params={"asset_cfg": joint_cfg, "biased": True},
      noise=Unoise(n_min=-0.01, n_max=0.01) if noise else None,
    ),
    # Joint velocities of all actuated joints, relative to the initial pose of the robot, with small noise to improve robustness.
    "joint_vel": ObservationTermCfg(
      func=tracking_mdp.joint_vel_rel,
      params={"asset_cfg": joint_cfg},
      noise=Unoise(n_min=-0.25, n_max=0.25) if noise else None,
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
      noise=Unoise(n_min=-0.01, n_max=0.01) if noise else None,
    ),
    # Ball velocity relative to the robot anchor, in the anchor frame.
    "ball_velocity": ObservationTermCfg(
      func=catch_mdp.ball_velocity_b,
      params={
        "command_name": command_name,
        "occlude_in_catch_zone": occlude_in_catch_zone,
      },
      noise=Unoise(n_min=-0.1, n_max=0.1) if noise else None,
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
    # Privileged reference ball state relative to the reference anchor
    "reference_ball_state": ObservationTermCfg(
      func=catch_mdp.reference_ball_state_b, params={"command_name": command_name}
    ),
    # Robot anchor height above the local environment origin.
    "anchor_height": ObservationTermCfg(
      func=catch_mdp.robot_anchor_height,
      params={"command_name": command_name},
    ),
    "throw_state": ObservationTermCfg(
      func=tracking_mdp.generated_commands,
      params={"command_name": command_name},
    ),
  }


def catch_observation_cfg(
  command_name: str,
  humanoid: Humanoid,
  version: int = 0,
  occlude_in_catch_zone: bool = False,
  prev_actor_obs: dict[str, ObservationTermCfg] = {},
  prev_critic_obs: dict[str, ObservationTermCfg] = {},
) -> dict[str, ObservationGroupCfg]:
  """
  Returns the full observation configuration as a dictionary of reward names and RewardTermCfg.
  Params:
      command_name: Name of the current task command.
      humanoid: Instance of the used Humanoid Class.
      version: Integer of the desired version of observation configuration.
      occlude_in_catch_zone: Whether to make the ball unobservable in the catch zone.
      prev_actor_obs: Dictionary of ObservationTermCfg to retain for the actor observations.
      prev_critic_obs: Dictionary of ObservationTermCfg to retain for the ctritic observations.
  """

  actor_terms = {}
  # Keep previous actor observations
  actor_terms.update(prev_actor_obs)
  # Add desired actor observations based on the version
  actor_terms.update(
    _actor_observation_terms(
      command_name=command_name,
      humanoid=humanoid,
      version=version,
      occlude_in_catch_zone=occlude_in_catch_zone,
    )
  )

  critic_terms = {}
  # Keep previous critic observations
  critic_terms.update(prev_critic_obs)
  # Add desired critic observations based on the version and the current task command
  if command_name == "throw":
    critic_terms.update(
      _critic_observation_terms_throw(
        command_name=command_name,
        humanoid=humanoid,
        version=version,
      )
    )
  else:
    critic_terms.update(
      _critic_observation_terms_catch(
        command_name=command_name,
        humanoid=humanoid,
        version=version,
      )
    )

  # Combine actor and critic obervation configurations into one dictionary
  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }
  return observations


def _actor_observation_terms(
  command_name: str,
  humanoid: Humanoid,
  version: int = 0,
  occlude_in_catch_zone: bool = False,
) -> dict[str, ObservationTermCfg]:
  """Return the deployment actor observation interface shared by both training stages."""
  # get all possible observations
  all_obs = all_observations_dict(
    command_name=command_name,
    humanoid=humanoid,
    occlude_in_catch_zone=occlude_in_catch_zone,
    noise=True,
  )

  # Extract observations based on the desired version
  obs_keys = ACTOR_OBSERVATION_TERMS[version]
  return {key: all_obs[key] for key in obs_keys}


def _critic_observation_terms_catch(
  command_name: str,
  humanoid: Humanoid,
  version: int = 0,
) -> dict[str, ObservationTermCfg]:
  """Return the critic observations for Phase 1 (IL)."""

  # get all possible observations
  all_obs = all_observations_dict(
    command_name=command_name,
    humanoid=humanoid,
    occlude_in_catch_zone=False,
    noise=False,
  )

  # Extract desired observations
  obs_keys = CRITIC_OBSERVATION_TERMS_CATCH[version]
  return {key: all_obs[key] for key in obs_keys}


def _critic_observation_terms_throw(
  command_name: str,
  humanoid: Humanoid,
  version: int = 0,
) -> dict[str, ObservationTermCfg]:
  """Return the critic observations for Phase 1 (RL)."""

  # get all possible observations
  all_obs = all_observations_dict(
    command_name=command_name,
    humanoid=humanoid,
    occlude_in_catch_zone=False,
    noise=False,
  )

  # Extract desired observations
  obs_keys = CRITIC_OBSERVATION_TERMS_THROW[version]
  return {key: all_obs[key] for key in obs_keys}

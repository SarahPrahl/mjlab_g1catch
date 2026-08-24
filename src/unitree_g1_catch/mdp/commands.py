"""Reference-motion command with a dynamic tracked ball."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.tasks.tracking.mdp.commands import (
  MotionCommand,
  MotionCommandCfg,
  MotionLoader,
)
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_inv,
  sample_uniform,
)

if TYPE_CHECKING:
  from collections.abc import Callable
  from typing import Any

  import viser
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class CatchMotionLoader(MotionLoader):
  """Load standard mjlab motion arrays plus ball and clip-boundary metadata."""

  def __init__(
    self, motion_file: str, body_indexes: torch.Tensor, device: str = "cpu"
  ) -> None:
    super().__init__(motion_file, body_indexes, device=device)
    with np.load(motion_file, allow_pickle=False) as data:
      missing = [
        key
        for key in ("ball_pos_w", "ball_lin_vel_w", "ball_visible")
        if key not in data
      ]
      if missing:
        raise ValueError(f"Catch motion is missing required arrays: {missing}")

      self.ball_pos_w = torch.tensor(
        data["ball_pos_w"], dtype=torch.float32, device=device
      )
      self.ball_lin_vel_w = torch.tensor(
        data["ball_lin_vel_w"], dtype=torch.float32, device=device
      )
      self.ball_visible = torch.tensor(
        data["ball_visible"], dtype=torch.bool, device=device
      )
      self.clip_starts = (
        torch.tensor(data["clip_starts"], dtype=torch.long, device=device)
        if "clip_starts" in data
        else torch.zeros(1, dtype=torch.long, device=device)
      )
      self.clip_lengths = (
        torch.tensor(data["clip_lengths"], dtype=torch.long, device=device)
        if "clip_lengths" in data
        else torch.tensor([self.time_step_total], dtype=torch.long, device=device)
      )
      self.clip_names = (
        tuple(str(name) for name in data["clip_names"].tolist())
        if "clip_names" in data
        else tuple(f"clip_{index + 1}" for index in range(len(self.clip_starts)))
      )

    if self.ball_pos_w.shape != (self.time_step_total, 3):
      raise ValueError(
        "ball_pos_w must have shape "
        f"({self.time_step_total}, 3), got {tuple(self.ball_pos_w.shape)}."
      )
    if self.ball_lin_vel_w.shape != (self.time_step_total, 3):
      raise ValueError(
        "ball_lin_vel_w must have shape "
        f"({self.time_step_total}, 3), got {tuple(self.ball_lin_vel_w.shape)}."
      )
    if self.ball_visible.shape != (self.time_step_total,):
      raise ValueError("ball_visible must have one value per motion frame.")
    if self.clip_starts.shape != self.clip_lengths.shape:
      raise ValueError("clip_starts and clip_lengths must have matching shapes.")
    if len(self.clip_names) != len(self.clip_starts):
      raise ValueError("clip_names must have one name per clip boundary.")
    if torch.any(self.clip_lengths <= 0):
      raise ValueError("All reference clips must contain at least one frame.")
    if torch.any(self.clip_starts[1:] < self.clip_starts[:-1]):
      raise ValueError("clip_starts must be sorted.")
    final_frame = int((self.clip_starts[-1] + self.clip_lengths[-1]).item())
    if final_frame > self.time_step_total:
      raise ValueError("Reference clip metadata exceeds the motion array length.")


class CatchMotionCommand(MotionCommand):
  """Track a robot motion while initializing and observing a physical ball."""

  cfg: CatchMotionCommandCfg
  motion: CatchMotionLoader

  def __init__(self, cfg: CatchMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    # MotionCommand deliberately knows only mjlab's standard motion schema. Replace
    # its loader with our compatible extension instead of patching mjlab internals.
    self.motion = CatchMotionLoader(
      cfg.motion_file, self.body_indexes, device=self.device
    )
    self.ball: Entity = env.scene[cfg.ball_entity_name]
    self._ball_visible_state = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.metrics["error_ball_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_ball_lin_vel"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def ball_reference_pos_w(self) -> torch.Tensor:
    return self.motion.ball_pos_w[self.time_steps] + self._env.scene.env_origins

  @property
  def ball_reference_lin_vel_w(self) -> torch.Tensor:
    return self.motion.ball_lin_vel_w[self.time_steps]

  @property
  def ball_visible(self) -> torch.Tensor:
    return self.motion.ball_visible[self.time_steps]

  @property
  def ball_pos_w(self) -> torch.Tensor:
    return self.ball.data.root_link_pos_w

  @property
  def ball_lin_vel_w(self) -> torch.Tensor:
    return self.ball.data.root_link_lin_vel_w

  def _update_metrics(self) -> None:
    super()._update_metrics()
    visible = self.ball_visible.float()
    self.metrics["error_ball_pos"] = (
      torch.norm(self.ball_reference_pos_w - self.ball_pos_w, dim=-1) * visible
    )
    self.metrics["error_ball_lin_vel"] = (
      torch.norm(self.ball_reference_lin_vel_w - self.ball_lin_vel_w, dim=-1) * visible
    )

  def _write_ball_reference_state(self, env_ids: torch.Tensor) -> None:
    visible = self.ball_visible[env_ids]
    root_state = self.ball.data.default_root_state[env_ids].clone()
    root_state[:, 0:3] = self.ball_reference_pos_w[env_ids]
    root_state[:, 3] = 1.0
    root_state[:, 4:7] = 0.0
    root_state[:, 7:10] = self.ball_reference_lin_vel_w[env_ids]
    root_state[:, 10:13] = 0.0

    # Keep the physical ball well outside the scene until the recording reports it.
    inactive_pos = self._env.scene.env_origins[env_ids][~visible].clone()
    inactive_pos += torch.tensor((100.0, 0.0, 100.0), device=self.device)
    root_state[~visible, 0:3] = inactive_pos
    root_state[~visible, 7:13] = 0.0
    self.ball.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.ball.reset(env_ids=env_ids)
    self._ball_visible_state[env_ids] = visible

  def _write_reference_state_to_sim(
    self,
    env_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_ori: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    super()._write_reference_state_to_sim(
      env_ids,
      root_pos,
      root_ori,
      root_lin_vel,
      root_ang_vel,
      joint_pos,
      joint_vel,
    )
    self._write_ball_reference_state(env_ids)

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    previous_visible = self._ball_visible_state.clone()
    previous_steps = self.time_steps.clone()
    scoped_ids = (
      torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
    )
    clip_ids = torch.bucketize(
      previous_steps[scoped_ids], self.motion.clip_starts[1:], right=True
    )
    clip_ends = self.motion.clip_starts[clip_ids] + self.motion.clip_lengths[clip_ids]
    reached_clip_end = previous_steps[scoped_ids] + 1 >= clip_ends
    # Training resamples at internal boundaries and lets the base command handle the
    # final dataset wrap. Play mode loops whichever named clip the viewer selected.
    boundary_mask = (
      reached_clip_end
      if self.cfg.sampling_mode == "start"
      else reached_clip_end & (clip_ends < self.motion.time_step_total)
    )
    boundary_ids = scoped_ids[boundary_mask]
    boundary_clip_ids = clip_ids[boundary_mask]

    super()._update_command(env_ids)
    if boundary_ids.numel() > 0:
      if self.cfg.sampling_mode == "start":
        for clip_id in torch.unique(boundary_clip_ids):
          clip_env_ids = boundary_ids[boundary_clip_ids == clip_id]
          frame = int(self.motion.clip_starts[clip_id].item())
          self.reset_to_frame(clip_env_ids, frame)
      else:
        self._resample_command(boundary_ids)
      self._pending_forward = False
      self._env.sim.forward()
      self.update_relative_body_poses()

    current_visible = self.ball_visible
    spawn_ids = torch.where(~previous_visible & current_visible)[0]
    if spawn_ids.numel() > 0:
      self._write_ball_reference_state(spawn_ids)
      self._env.sim.forward()
    self._ball_visible_state[:] = current_visible

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw the reference robot and recorded ball target."""
    super()._debug_vis_impl(visualizer)
    reference_positions = self.ball_reference_pos_w
    visible = self.ball_visible
    for env_idx in visualizer.get_env_indices(self.num_envs):
      if not bool(visible[env_idx].item()):
        continue
      visualizer.add_sphere(
        center=reference_positions[env_idx].detach().cpu().numpy(),
        radius=self.cfg.reference_ball_viz_radius_m,
        color=self.cfg.reference_ball_viz_color,
        label=f"reference_ball_{env_idx}",
      )

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Create the normal motion scrubber plus a named-clip selector."""
    max_frame = int(self.motion.time_step_total) - 1
    clip_options = tuple(
      f"{index + 1}: {clip_name}"
      for index, clip_name in enumerate(self.motion.clip_names)
    )

    with server.gui.add_folder(name.capitalize()):
      clip_dropdown = server.gui.add_dropdown(
        "Clip", options=clip_options, initial_value=clip_options[0]
      )
      scrubber = server.gui.add_slider(
        "Frame", min=0, max=max_frame, step=1, initial_value=0
      )

      @scrubber.on_update
      def _(_) -> None:
        idx = get_env_idx()
        self.time_steps[idx] = int(scrubber.value)
        if on_change is not None:
          on_change()

      @clip_dropdown.on_update
      def _(_) -> None:
        clip_index = clip_options.index(clip_dropdown.value)
        frame = int(self.motion.clip_starts[clip_index].item())
        scrubber.value = frame
        idx = get_env_idx()
        self.time_steps[idx] = frame
        if on_change is not None:
          on_change()

      all_envs_cb = server.gui.add_checkbox("All envs", initial_value=True)
      start_btn = server.gui.add_button("Start Here")

      @start_btn.on_click
      def _(_) -> None:
        if request_action is not None:
          request_action(
            "CUSTOM",
            {"type": "gui_reset", "all_envs": all_envs_cb.value},
          )

    self._scrubber_handles = (scrubber, all_envs_cb, start_btn, clip_dropdown)
    self._set_scrubber_disabled(True)


@dataclass(kw_only=True)
class CatchMotionCommandCfg(MotionCommandCfg):
  ball_entity_name: str = "ball"
  # Slightly larger than the 7.5 cm physical ball radius so the marker remains
  # visible as a shell when the physical and reference balls coincide.
  reference_ball_viz_radius_m: float = 0.0825
  reference_ball_viz_color: tuple[float, float, float, float] = (
    0.1,
    0.55,
    1.0,
    0.4,
  )

  def build(self, env: ManagerBasedRlEnv) -> CatchMotionCommand:
    return CatchMotionCommand(self, env)


class BallThrowCommand(CommandTerm):
  """Launch a physical ball once, then leave its trajectory entirely to MuJoCo."""

  cfg: BallThrowCommandCfg

  def __init__(self, cfg: BallThrowCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.ball: Entity = env.scene[cfg.ball_entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
    self.hand_body_indexes = torch.tensor(
      self.robot.find_bodies(cfg.hand_body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )
    self.palm_site_indexes = torch.tensor(
      self.robot.find_sites(cfg.palm_site_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    shape = (self.num_envs,)
    self.launch_delay = torch.zeros(shape, device=self.device)
    self.flight_time = torch.zeros(shape, device=self.device)
    self.difficulty = torch.zeros(shape, device=self.device)
    self.target_offset_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.spawn_offset_w = torch.zeros_like(self.target_offset_w)
    self._target_pos_w = torch.zeros_like(self.target_offset_w)
    self._spawn_pos_w = torch.zeros_like(self.target_offset_w)
    self._launch_anchor_pos_w = torch.zeros_like(self.target_offset_w)
    self.launched = torch.zeros(shape, dtype=torch.bool, device=self.device)
    self.left_palm_contact = torch.zeros_like(self.launched)
    self.right_palm_contact = torch.zeros_like(self.launched)
    self.stable_grasp = torch.zeros_like(self.launched)
    self.caught = torch.zeros_like(self.launched)
    self.hold_steps = torch.zeros(shape, device=self.device)
    self.required_hold_steps = torch.ones(shape, device=self.device)
    self.contact_loss_steps = torch.zeros(shape, dtype=torch.long, device=self.device)
    self._contact_grace_steps = max(
      0, math.ceil(cfg.palm_contact_grace_time_s / env.step_dt)
    )
    self.finger_grip_latch = torch.zeros(shape, dtype=torch.bool, device=self.device)
    self.finger_release_steps = torch.zeros(shape, dtype=torch.long, device=self.device)
    self._finger_release_window_steps = max(
      1, math.ceil(cfg.finger_open_release_time_s / env.step_dt)
    )

    self.metrics["catch_success"] = torch.zeros(shape, device=self.device)
    self.metrics["palm_contact"] = torch.zeros(shape, device=self.device)
    self.metrics["stable_grasp"] = torch.zeros(shape, device=self.device)
    self.metrics["catch_pose"] = torch.zeros(shape, device=self.device)
    self.metrics["hold_progress"] = torch.zeros(shape, device=self.device)
    self.metrics["required_hold_time"] = torch.zeros(shape, device=self.device)
    self.metrics["throw_difficulty"] = torch.zeros(shape, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return torch.cat(
      (
        self.target_offset_w,
        self.flight_time[:, None],
        self.difficulty[:, None],
        self.launched[:, None].float(),
        self.caught[:, None].float(),
      ),
      dim=-1,
    )

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def ball_pos_w(self) -> torch.Tensor:
    return self.ball.data.root_link_pos_w

  @property
  def ball_lin_vel_w(self) -> torch.Tensor:
    return self.ball.data.root_link_lin_vel_w

  @property
  def ball_visible(self) -> torch.Tensor:
    return self.launched

  @property
  def target_pos_w(self) -> torch.Tensor:
    live_target = self.robot_anchor_pos_w + self.target_offset_w
    return torch.where(self.launched[:, None], self._target_pos_w, live_target)

  @property
  def spawn_pos_w(self) -> torch.Tensor:
    live_spawn = self.robot_anchor_pos_w + self.spawn_offset_w
    return torch.where(self.launched[:, None], self._spawn_pos_w, live_spawn)

  @property
  def time_to_arrival(self) -> torch.Tensor:
    elapsed = self._env.episode_length_buf * self._env.step_dt
    return torch.clamp(self.launch_delay + self.flight_time - elapsed, min=0.0)

  @property
  def palm_midpoint_w(self) -> torch.Tensor:
    return self.robot.data.site_pos_w[:, self.palm_site_indexes].mean(dim=1)

  @property
  def ball_palm_distance(self) -> torch.Tensor:
    return torch.norm(self.ball_pos_w - self.palm_midpoint_w, dim=-1)

  @property
  def ball_hand_relative_speed(self) -> torch.Tensor:
    hand_velocity = self.robot.data.body_link_lin_vel_w[:, self.hand_body_indexes].mean(
      dim=1
    )
    return torch.norm(self.ball_lin_vel_w - hand_velocity, dim=-1)

  @property
  def catch_pose_score(self) -> torch.Tensor:
    """Score a demonstration-like catch in front of the upright torso."""
    ball_offset_b = quat_apply(
      quat_inv(self.robot_anchor_quat_w), self.ball_pos_w - self.robot_anchor_pos_w
    )
    x_excess = torch.relu(self.cfg.catch_x_range_m[0] - ball_offset_b[:, 0])
    x_excess += torch.relu(ball_offset_b[:, 0] - self.cfg.catch_x_range_m[1])
    y_excess = torch.relu(torch.abs(ball_offset_b[:, 1]) - self.cfg.catch_y_abs_max_m)
    z_excess = torch.relu(self.cfg.catch_z_range_m[0] - ball_offset_b[:, 2])
    z_excess += torch.relu(ball_offset_b[:, 2] - self.cfg.catch_z_range_m[1])
    zone_error = x_excess.square() + y_excess.square() + z_excess.square()

    ball_above_palms_w = self.ball_pos_w - self.palm_midpoint_w
    horizontal_error = torch.sum(ball_above_palms_w[:, :2].square(), dim=-1)
    height_error = (
      ball_above_palms_w[:, 2] - self.cfg.support_target_height_m
    ).square()
    torso_up_w = quat_apply(
      self.robot_anchor_quat_w,
      torch.tensor((0.0, 0.0, 1.0), device=self.device).expand(self.num_envs, -1),
    )
    tilt_error = torch.sum(torso_up_w[:, :2].square(), dim=-1)
    return (
      torch.exp(-zone_error / self.cfg.catch_zone_std_m**2)
      * torch.exp(
        -horizontal_error / self.cfg.support_horizontal_max_m**2
        - height_error / self.cfg.support_height_std_m**2
      )
      * torch.exp(-tilt_error / math.sin(self.cfg.maximum_catch_tilt_rad) ** 2)
      * self.ball_visible.float()
    )

  @property
  def pre_catch_pose_weight(self) -> torch.Tensor:
    """Fade the quiet ready pose out as the launched ball reaches the hands."""
    distance_weight = torch.clamp(
      (self.ball_palm_distance - self.cfg.ready_pose_near_distance_m)
      / (self.cfg.ready_pose_far_distance_m - self.cfg.ready_pose_near_distance_m),
      min=0.0,
      max=1.0,
    )
    return torch.where(self.launched, distance_weight, 1.0) * (~self.caught).float()

  @property
  def finger_open_weight(self) -> torch.Tensor:
    """Keep fingers open until the ball is stably contained, then fade out."""
    window = max(1, self._finger_release_window_steps)
    ramp = 1.0 - torch.clamp(self.finger_release_steps.float() / window, max=1.0)
    return (~self.caught).float() * ramp

  @property
  def finger_grip_weight(self) -> torch.Tensor:
    """Fade in a gentle stabilizing grip after the first stable grasp."""
    window = max(1, self._finger_release_window_steps)
    ramp = torch.clamp(self.finger_release_steps.float() / window, max=1.0)
    return (~self.caught).float() * self.finger_grip_latch.float() * ramp

  def _palm_face_contact_details(
    self, sensor_name: str, hand_index: int
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return active/accepted masks and positions for one physical palm face."""
    sensor = self._env.scene[sensor_name]
    assert sensor.data.found is not None
    assert sensor.data.force is not None
    assert sensor.data.normal is not None
    assert sensor.data.pos is not None
    force = torch.norm(sensor.data.force, dim=-1)
    active = (sensor.data.found > 0) & (force > self.cfg.contact_force_threshold_n)

    hand_pos_w = self.robot.data.body_link_pos_w[:, self.hand_body_indexes[hand_index]]
    hand_quat_w = self.robot.data.body_link_quat_w[
      :, self.hand_body_indexes[hand_index]
    ]
    num_contacts = sensor.data.normal.shape[1]
    inverse_hand_quat = quat_inv(hand_quat_w)[:, None, :].expand(-1, num_contacts, -1)
    normal_b = quat_apply(inverse_hand_quat, sensor.data.normal)
    contact_pos_b = quat_apply(
      inverse_hand_quat,
      sensor.data.pos - hand_pos_w[:, None, :],
    )

    # Both Inspire hands use local -X as the inward-facing palm normal. Position
    # bounds isolate the broad palm patch from the sides and edges of its boxes.
    normal_alignment = -normal_b[..., 0]
    inside_patch = (
      (contact_pos_b[..., 0] >= self.cfg.palm_face_x_range_m[0])
      & (contact_pos_b[..., 0] <= self.cfg.palm_face_x_range_m[1])
      & (contact_pos_b[..., 1] >= self.cfg.palm_face_y_range_m[0])
      & (contact_pos_b[..., 1] <= self.cfg.palm_face_y_range_m[1])
      & (torch.abs(contact_pos_b[..., 2]) <= self.cfg.palm_face_z_abs_max_m)
    )
    accepted = (
      active
      & inside_patch
      & (normal_alignment >= self.cfg.minimum_palm_normal_alignment)
    )
    return active, accepted, sensor.data.pos

  def _contact_state(
    self,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    _, left_contacts, _ = self._palm_face_contact_details(
      self.cfg.left_palm_sensor_name, 0
    )
    _, right_contacts, _ = self._palm_face_contact_details(
      self.cfg.right_palm_sensor_name, 1
    )
    left_palm = torch.any(left_contacts, dim=-1)
    right_palm = torch.any(right_contacts, dim=-1)
    return left_palm, right_palm

  def _update_metrics(self) -> None:
    self.metrics["catch_success"][:] = self.caught.float()
    self.metrics["palm_contact"][:] = 0.5 * (
      self.left_palm_contact.float() + self.right_palm_contact.float()
    )
    self.metrics["stable_grasp"][:] = self.stable_grasp.float()
    self.metrics["catch_pose"][:] = self.catch_pose_score
    self.metrics["hold_progress"][:] = torch.clamp(
      self.hold_steps.float() / self.required_hold_steps.float(), max=1.0
    )
    self.metrics["required_hold_time"][:] = (
      self.required_hold_steps.float() * self._env.step_dt
    )
    self.metrics["throw_difficulty"][:] = self.difficulty

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    num = len(env_ids)
    if num == 0:
      return

    progress = min(
      self._env.common_step_counter / max(self.cfg.curriculum_steps, 1), 1.0
    )
    difficulty = torch.full((num,), progress, device=self.device)
    self.difficulty[env_ids] = difficulty
    required_hold_time = self.cfg.initial_hold_time_s + progress * (
      self.cfg.final_hold_time_s - self.cfg.initial_hold_time_s
    )
    self.required_hold_steps[env_ids] = max(
      1, math.ceil(required_hold_time / self._env.step_dt)
    )

    angle_limit = self.cfg.initial_azimuth_rad + progress * (
      self.cfg.final_azimuth_rad - self.cfg.initial_azimuth_rad
    )
    azimuth = sample_uniform(-angle_limit, angle_limit, (num,), self.device)
    target_y_limit = self.cfg.initial_target_y_range_m + progress * (
      self.cfg.final_target_y_range_m - self.cfg.initial_target_y_range_m
    )
    target_z_limit = self.cfg.initial_target_z_range_m + progress * (
      self.cfg.final_target_z_range_m - self.cfg.initial_target_z_range_m
    )
    distance_max = self.cfg.initial_distance_max_m + progress * (
      self.cfg.final_distance_max_m - self.cfg.initial_distance_max_m
    )
    self.target_offset_w[env_ids, 0] = sample_uniform(
      *self.cfg.target_x_range_m, (num,), self.device
    )
    self.target_offset_w[env_ids, 1] = sample_uniform(
      -target_y_limit, target_y_limit, (num,), self.device
    )
    self.target_offset_w[env_ids, 2] = self.cfg.target_z_center_m + sample_uniform(
      -target_z_limit, target_z_limit, (num,), self.device
    )
    distance = sample_uniform(
      self.cfg.distance_min_m, distance_max, (num,), self.device
    )
    self.spawn_offset_w[env_ids] = self.target_offset_w[env_ids]
    self.spawn_offset_w[env_ids, 0] += distance * torch.cos(azimuth)
    self.spawn_offset_w[env_ids, 1] += distance * torch.sin(azimuth)
    self.spawn_offset_w[env_ids, 2] += sample_uniform(
      *self.cfg.spawn_height_offset_range_m, (num,), self.device
    )
    elevation = sample_uniform(
      *self.cfg.launch_elevation_range_rad, (num,), self.device
    )
    vertical_delta = self.target_offset_w[env_ids, 2] - self.spawn_offset_w[env_ids, 2]
    self.flight_time[env_ids] = torch.sqrt(
      torch.clamp(
        2.0 * (distance * torch.tan(elevation) - vertical_delta) / 9.81,
        min=1.0e-4,
      )
    )
    self.launch_delay[env_ids] = sample_uniform(
      *self.cfg.launch_delay_range_s, (num,), self.device
    )

    self.launched[env_ids] = False
    self.stable_grasp[env_ids] = False
    self.caught[env_ids] = False
    self.hold_steps[env_ids] = 0
    self.contact_loss_steps[env_ids] = 0
    self.finger_grip_latch[env_ids] = False
    self.finger_release_steps[env_ids] = 0
    self._target_pos_w[env_ids] = 0.0
    self._spawn_pos_w[env_ids] = 0.0
    self._launch_anchor_pos_w[env_ids] = 0.0
    self.left_palm_contact[env_ids] = False
    self.right_palm_contact[env_ids] = False

    root_state = self.ball.data.default_root_state[env_ids].clone()
    root_state[:, 0:3] = self._env.scene.env_origins[env_ids]
    root_state[:, 0] += self.cfg.inactive_position_w[0]
    root_state[:, 1] += self.cfg.inactive_position_w[1]
    root_state[:, 2] += self.cfg.inactive_position_w[2]
    root_state[:, 3] = 1.0
    root_state[:, 4:13] = 0.0
    self.ball.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.ball.reset(env_ids=env_ids)

  def _launch(self, env_ids: torch.Tensor) -> None:
    anchor_pos = self.robot_anchor_pos_w[env_ids]
    # Throws use the environment's yaw frame. The robot resets facing world +X,
    # so torso motion before launch cannot rotate or tilt the throw distribution.
    target_pos = anchor_pos + self.target_offset_w[env_ids]
    spawn_pos = anchor_pos + self.spawn_offset_w[env_ids]
    flight_time = self.flight_time[env_ids, None]
    gravity = torch.tensor((0.0, 0.0, -9.81), device=self.device)
    launch_velocity = (
      target_pos - spawn_pos - 0.5 * gravity * torch.square(flight_time)
    ) / flight_time

    root_state = self.ball.data.default_root_state[env_ids].clone()
    root_state[:, 0:3] = spawn_pos
    root_state[:, 3] = 1.0
    root_state[:, 4:7] = 0.0
    root_state[:, 7:10] = launch_velocity
    root_state[:, 10:13] = 0.0
    self.ball.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.ball.reset(env_ids=env_ids)
    self._target_pos_w[env_ids] = target_pos
    self._spawn_pos_w[env_ids] = spawn_pos
    self._launch_anchor_pos_w[env_ids] = anchor_pos
    self.launched[env_ids] = True

    # The environment has already forwarded before command updates. Refresh the
    # teleported ball so the launch is visible in this step's observations.
    self._env.sim.forward()

  def _update_grasp(self, env_ids: torch.Tensor) -> None:
    left_palm, right_palm = self._contact_state()
    self.left_palm_contact[env_ids] = left_palm[env_ids]
    self.right_palm_contact[env_ids] = right_palm[env_ids]
    both_palms = left_palm & right_palm
    self.contact_loss_steps[env_ids] = torch.where(
      both_palms[env_ids],
      0,
      self.contact_loss_steps[env_ids] + 1,
    )
    palm_contact_maintained = both_palms | (
      (self.hold_steps > 0) & (self.contact_loss_steps <= self._contact_grace_steps)
    )
    contained = self.ball_palm_distance < self.cfg.containment_radius_m
    slow = self.ball_hand_relative_speed < self.cfg.max_hold_relative_speed_mps
    stable = self.launched & contained & slow & palm_contact_maintained
    self.stable_grasp[env_ids] = stable[env_ids]
    self.finger_grip_latch[env_ids] |= stable[env_ids]
    self.finger_release_steps[env_ids] = torch.where(
      self.finger_grip_latch[env_ids],
      self.finger_release_steps[env_ids] + 1,
      0,
    )
    pose_quality = self.catch_pose_score
    self.hold_steps[env_ids] = torch.where(
      stable[env_ids],
      self.hold_steps[env_ids] + pose_quality[env_ids],
      0.0,
    )
    self.caught[env_ids] |= (
      self.hold_steps[env_ids] >= self.required_hold_steps[env_ids]
    )

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    scoped_ids = (
      torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
    )
    elapsed = self._env.episode_length_buf[scoped_ids] * self._env.step_dt
    ready = (~self.launched[scoped_ids]) & (elapsed >= self.launch_delay[scoped_ids])
    launch_ids = scoped_ids[ready]
    if launch_ids.numel() > 0:
      self._launch(launch_ids)
    self._update_grasp(scoped_ids)
    self._update_metrics()

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    for env_idx in visualizer.get_env_indices(self.num_envs):
      if self.cfg.show_catch_zone_viz:
        catch_center_b = torch.tensor(
          (
            sum(self.cfg.catch_x_range_m) / 2.0,
            0.0,
            sum(self.cfg.catch_z_range_m) / 2.0,
          ),
          device=self.device,
        )
        catch_center_w = self.robot_anchor_pos_w[env_idx] + quat_apply(
          self.robot_anchor_quat_w[env_idx], catch_center_b
        )
        catch_half_extents = np.asarray(
          (
            (self.cfg.catch_x_range_m[1] - self.cfg.catch_x_range_m[0]) / 2.0,
            self.cfg.catch_y_abs_max_m,
            (self.cfg.catch_z_range_m[1] - self.cfg.catch_z_range_m[0]) / 2.0,
          )
        )
        catch_rotation = matrix_from_quat(self.robot_anchor_quat_w[env_idx])
        visualizer.add_box(
          center=catch_center_w.detach().cpu().numpy(),
          size=catch_half_extents,
          mat=catch_rotation.detach().cpu().numpy(),
          color=(0.1, 0.75, 0.9, 0.12),
          label=f"preferred_catch_zone_{env_idx}",
        )
        support_target_w = self.palm_midpoint_w[env_idx] + torch.tensor(
          (0.0, 0.0, self.cfg.support_target_height_m), device=self.device
        )
        visualizer.add_sphere(
          center=support_target_w.detach().cpu().numpy(),
          radius=0.018,
          color=(0.2, 0.9, 0.35, 0.8),
          label=f"palm_support_target_{env_idx}",
        )

      spawn = self.spawn_pos_w[env_idx].detach().cpu().numpy()
      target = self.target_pos_w[env_idx].detach().cpu().numpy()
      visualizer.add_sphere(
        center=target,
        radius=0.025,
        color=(0.1, 0.55, 1.0, 0.65),
        label=f"throw_target_{env_idx}",
      )
      visualizer.add_sphere(
        center=spawn,
        radius=0.025,
        color=(0.85, 0.1, 0.75, 0.65),
        label=f"throw_source_{env_idx}",
      )
      visualizer.add_arrow(
        start=spawn,
        end=target,
        color=(0.85, 0.1, 0.75, 0.8),
        width=0.008,
        label=f"throw_path_{env_idx}",
      )
      if self.launched[env_idx]:
        forward_start = self._launch_anchor_pos_w[env_idx]
        forward_end = forward_start + torch.tensor((0.75, 0.0, 0.0), device=self.device)
        visualizer.add_arrow(
          start=forward_start.detach().cpu().numpy(),
          end=forward_end.detach().cpu().numpy(),
          color=(0.9, 0.15, 0.1, 0.8),
          width=0.008,
          label=f"launch_forward_{env_idx}",
        )

      palm_pos_w = self.robot.data.site_pos_w[env_idx, self.palm_site_indexes]
      palm_quat_w = self.robot.data.body_link_quat_w[env_idx, self.hand_body_indexes]
      inward_normal_w = quat_apply(
        palm_quat_w,
        torch.tensor((-1.0, 0.0, 0.0), device=self.device).expand(2, -1),
      )
      for hand_index, side in enumerate(("left", "right")):
        normal_start = palm_pos_w[hand_index]
        normal_end = normal_start + 0.12 * inward_normal_w[hand_index]
        visualizer.add_arrow(
          start=normal_start.detach().cpu().numpy(),
          end=normal_end.detach().cpu().numpy(),
          color=(0.1, 0.8, 0.9, 0.85),
          width=0.006,
          label=f"{side}_palm_normal_{env_idx}",
        )
        sensor_name = (
          self.cfg.left_palm_sensor_name
          if hand_index == 0
          else self.cfg.right_palm_sensor_name
        )
        active, accepted, positions = self._palm_face_contact_details(
          sensor_name, hand_index
        )
        for contact_index in range(positions.shape[1]):
          if not bool(active[env_idx, contact_index].item()):
            continue
          visualizer.add_sphere(
            center=positions[env_idx, contact_index].detach().cpu().numpy(),
            radius=0.012,
            color=(
              (0.1, 0.9, 0.2, 0.9)
              if bool(accepted[env_idx, contact_index].item())
              else (0.95, 0.1, 0.1, 0.9)
            ),
            label=f"{side}_palm_contact_{env_idx}_{contact_index}",
          )


@dataclass(kw_only=True)
class BallThrowCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  ball_entity_name: str = "ball"
  anchor_body_name: str = "torso_link"
  hand_body_names: tuple[str, str] = ("l_ih_hand_base", "r_ih_hand_base")
  palm_site_names: tuple[str, str] = ("left_palm_site", "right_palm_site")
  left_palm_sensor_name: str = "left_palm_ball_contact"
  right_palm_sensor_name: str = "right_palm_ball_contact"
  launch_delay_range_s: tuple[float, float] = (0.25, 0.5)
  target_x_range_m: tuple[float, float] = (0.35, 0.5)
  target_z_center_m: float = 0.22
  initial_target_y_range_m: float = 0.1
  final_target_y_range_m: float = 0.25
  initial_target_z_range_m: float = 0.05
  final_target_z_range_m: float = 0.15
  initial_azimuth_rad: float = math.radians(15.0)
  final_azimuth_rad: float = math.radians(75.0)
  distance_min_m: float = 1.5
  initial_distance_max_m: float = 2.0
  final_distance_max_m: float = 3.0
  spawn_height_offset_range_m: tuple[float, float] = (-0.1, 0.15)
  launch_elevation_range_rad: tuple[float, float] = (
    math.radians(25.0),
    math.radians(40.0),
  )
  curriculum_steps: int = 120_000
  containment_radius_m: float = 0.18
  max_hold_relative_speed_mps: float = 0.35
  catch_x_range_m: tuple[float, float] = (0.25, 0.6)
  catch_y_abs_max_m: float = 0.35
  catch_z_range_m: tuple[float, float] = (0.0, 0.42)
  catch_zone_std_m: float = 0.1
  support_target_height_m: float = 0.075
  support_height_std_m: float = 0.08
  support_horizontal_max_m: float = 0.14
  maximum_catch_tilt_rad: float = 0.45
  show_catch_zone_viz: bool = True
  initial_hold_time_s: float = 0.4
  final_hold_time_s: float = 2.0
  palm_contact_grace_time_s: float = 0.08
  finger_open_release_time_s: float = 0.5
  contact_force_threshold_n: float = 0.1
  minimum_palm_normal_alignment: float = math.cos(math.radians(30.0))
  palm_face_x_range_m: tuple[float, float] = (-0.03, -0.008)
  palm_face_y_range_m: tuple[float, float] = (-0.124, -0.052)
  palm_face_z_abs_max_m: float = 0.035
  ready_pose_near_distance_m: float = 0.5
  ready_pose_far_distance_m: float = 1.5
  inactive_position_w: tuple[float, float, float] = (100.0, 0.0, 100.0)

  def build(self, env: ManagerBasedRlEnv) -> BallThrowCommand:
    return BallThrowCommand(self, env)

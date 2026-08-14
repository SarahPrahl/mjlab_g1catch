"""Reference-motion command with a dynamic tracked ball."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.tasks.tracking.mdp.commands import (
  MotionCommand,
  MotionCommandCfg,
  MotionLoader,
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

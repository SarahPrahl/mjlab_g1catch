#!/usr/bin/env python3
"""Minimal conversion and validation pipeline for retargeted Unitree G1 motions."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CACHE_ROOT = PROJECT_ROOT / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))

ROBOT_INPUT_KEYS = (
  "fps",
  "root_pos",
  "root_rot_xyzw",
  "dof_pos",
  "joint_names",
)
CATCH_MOTION_KEYS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
  "ball_pos_w",
  "ball_lin_vel_w",
  "ball_visible",
)

FINGER_NAME_MAP = {
  "left_thumb_1_joint": "l_ih_thumb_proximal_yaw",
  "left_thumb_2_joint": "l_ih_thumb_proximal_pitch",
  "left_thumb_3_joint": "l_ih_thumb_intermediate",
  "left_thumb_4_joint": "l_ih_thumb_distal",
  "left_index_1_joint": "l_ih_index_proximal",
  "left_index_2_joint": "l_ih_index_intermediate",
  "left_middle_1_joint": "l_ih_middle_proximal",
  "left_middle_2_joint": "l_ih_middle_intermediate",
  "left_ring_1_joint": "l_ih_ring_proximal",
  "left_ring_2_joint": "l_ih_ring_intermediate",
  "left_little_1_joint": "l_ih_pinky_proximal",
  "left_little_2_joint": "l_ih_pinky_intermediate",
  "right_thumb_1_joint": "r_ih_thumb_proximal_yaw",
  "right_thumb_2_joint": "r_ih_thumb_proximal_pitch",
  "right_thumb_3_joint": "r_ih_thumb_intermediate",
  "right_thumb_4_joint": "r_ih_thumb_distal",
  "right_index_1_joint": "r_ih_index_proximal",
  "right_index_2_joint": "r_ih_index_intermediate",
  "right_middle_1_joint": "r_ih_middle_proximal",
  "right_middle_2_joint": "r_ih_middle_intermediate",
  "right_ring_1_joint": "r_ih_ring_proximal",
  "right_ring_2_joint": "r_ih_ring_intermediate",
  "right_little_1_joint": "r_ih_pinky_proximal",
  "right_little_2_joint": "r_ih_pinky_intermediate",
}


@dataclass
class ResampledMotion:
  input_fps: float
  input_frames: int
  fps: float
  root_pos: np.ndarray
  root_quat_wxyz: np.ndarray
  root_lin_vel: np.ndarray
  root_ang_vel: np.ndarray
  joint_pos: np.ndarray
  joint_vel: np.ndarray
  joint_names: tuple[str, ...]
  finger_joint_pos: np.ndarray | None = None
  finger_joint_names: tuple[str, ...] = ()
  ball_pos: np.ndarray | None = None
  ball_lin_vel: np.ndarray | None = None
  ball_visible: np.ndarray | None = None

  @property
  def frames(self) -> int:
    return int(self.root_pos.shape[0])


def _configure_warp_cache() -> None:
  import warp as wp

  wp.config.kernel_cache_dir = str(CACHE_ROOT / "warp")


def _continuous_quaternions_xyzw(quaternions: np.ndarray) -> np.ndarray:
  quaternions = np.asarray(quaternions, dtype=np.float64).copy()
  norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
  if np.any(norms < 1.0e-8):
    raise ValueError("Root motion contains a zero-length quaternion.")
  quaternions /= norms
  for index in range(1, len(quaternions)):
    if np.dot(quaternions[index - 1], quaternions[index]) < 0.0:
      quaternions[index] *= -1.0
  return quaternions


def _interpolate_columns(
  input_times: np.ndarray, values: np.ndarray, output_times: np.ndarray
) -> np.ndarray:
  flat = values.reshape(values.shape[0], -1)
  interpolated = np.stack(
    [
      np.interp(output_times, input_times, flat[:, index])
      for index in range(flat.shape[1])
    ],
    axis=1,
  )
  return interpolated.reshape((len(output_times), *values.shape[1:]))


def _angular_velocity_w(quaternions_wxyz: np.ndarray, time_step: float) -> np.ndarray:
  from scipy.spatial.transform import Rotation

  if len(quaternions_wxyz) < 2:
    return np.zeros((len(quaternions_wxyz), 3), dtype=np.float64)
  if len(quaternions_wxyz) == 2:
    previous = quaternions_wxyz[:1]
    following = quaternions_wxyz[1:]
    denominator = time_step
  else:
    previous = quaternions_wxyz[:-2]
    following = quaternions_wxyz[2:]
    denominator = 2.0 * time_step

  def conjugate(quaternion: np.ndarray) -> np.ndarray:
    result = quaternion.copy()
    result[:, 1:] *= -1.0
    return result

  def multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left.T
    rw, rx, ry, rz = right.T
    return np.stack(
      (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
      ),
      axis=1,
    )

  relative_wxyz = multiply(following, conjugate(previous))
  relative_xyzw = relative_wxyz[:, (1, 2, 3, 0)]
  angular = Rotation.from_quat(relative_xyzw).as_rotvec() / denominator
  if len(quaternions_wxyz) == 2:
    return np.repeat(angular, 2, axis=0)
  return np.concatenate((angular[:1], angular, angular[-1:]), axis=0)


def load_and_resample_motion(path: Path, output_fps: float) -> ResampledMotion:
  """Validate and resample a retargeted G1 NPZ into mjlab's 50 Hz convention."""
  from scipy.spatial.transform import Rotation, Slerp

  with np.load(path, allow_pickle=False) as data:
    missing = [key for key in ROBOT_INPUT_KEYS if key not in data]
    if missing:
      raise ValueError(f"{path} is missing required keys: {missing}")
    input_fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    root_pos = np.asarray(data["root_pos"], dtype=np.float64)
    root_quat_xyzw = _continuous_quaternions_xyzw(data["root_rot_xyzw"])
    joint_pos = np.asarray(data["dof_pos"], dtype=np.float64)
    joint_names = tuple(str(name) for name in data["joint_names"].tolist())
    finger_joint_pos = (
      np.asarray(data["finger_dof_pos"], dtype=np.float64)
      if "finger_dof_pos" in data
      else None
    )
    finger_joint_names = (
      tuple(str(name) for name in data["finger_joint_names"].tolist())
      if "finger_joint_names" in data
      else ()
    )
    ball_pos = (
      np.asarray(data["ball_pos"], dtype=np.float64)
      if "ball_pos" in data and data["ball_pos"].shape == (root_pos.shape[0], 3)
      else None
    )
    ball_visible = (
      np.asarray(data["ball_visible"], dtype=bool)
      if "ball_visible" in data and data["ball_visible"].shape == (root_pos.shape[0],)
      else None
    )

  frame_count = root_pos.shape[0]
  if frame_count < 2:
    raise ValueError(f"{path} needs at least two frames, got {frame_count}.")
  if root_pos.shape != (frame_count, 3):
    raise ValueError(f"Unexpected root_pos shape in {path}: {root_pos.shape}")
  if root_quat_xyzw.shape != (frame_count, 4):
    raise ValueError(
      f"Unexpected root_rot_xyzw shape in {path}: {root_quat_xyzw.shape}"
    )
  if joint_pos.shape != (frame_count, len(joint_names)):
    raise ValueError(
      f"dof_pos shape {joint_pos.shape} does not match "
      f"{len(joint_names)} joint names in {path}."
    )
  if len(set(joint_names)) != len(joint_names):
    raise ValueError(f"Duplicate joint names in {path}.")
  if finger_joint_pos is not None and finger_joint_pos.shape != (
    frame_count,
    len(finger_joint_names),
  ):
    raise ValueError(
      f"finger_dof_pos shape {finger_joint_pos.shape} does not match "
      f"{len(finger_joint_names)} finger names in {path}."
    )
  if input_fps <= 0.0 or output_fps <= 0.0:
    raise ValueError("Input and output frame rates must be positive.")
  for name, value in (
    ("root_pos", root_pos),
    ("root_rot_xyzw", root_quat_xyzw),
    ("dof_pos", joint_pos),
  ):
    if not np.isfinite(value).all():
      raise ValueError(f"Non-finite values found in {name} for {path}.")

  duration = (frame_count - 1) / input_fps
  output_times = np.arange(0.0, duration, 1.0 / output_fps, dtype=np.float64)
  input_times = np.arange(frame_count, dtype=np.float64) / input_fps
  if len(output_times) < 2:
    raise ValueError(f"Motion {path} is too short after resampling.")

  output_root_pos = _interpolate_columns(input_times, root_pos, output_times)
  output_joint_pos = _interpolate_columns(input_times, joint_pos, output_times)
  output_finger_joint_pos = (
    _interpolate_columns(input_times, finger_joint_pos, output_times)
    if finger_joint_pos is not None
    else None
  )
  output_quat_xyzw = Slerp(input_times, Rotation.from_quat(root_quat_xyzw))(
    output_times
  ).as_quat(canonical=False)
  output_quat_xyzw = _continuous_quaternions_xyzw(output_quat_xyzw)
  output_quat_wxyz = output_quat_xyzw[:, (3, 0, 1, 2)]
  time_step = 1.0 / output_fps

  output_ball_pos = None
  output_ball_velocity = None
  output_ball_visible = None
  if ball_pos is not None and ball_visible is not None and np.any(ball_visible):
    visible_indexes = np.flatnonzero(ball_visible)
    visible_times = input_times[visible_indexes]
    output_ball_pos = _interpolate_columns(
      visible_times, ball_pos[visible_indexes], output_times
    )
    nearest_input = np.clip(
      np.rint(output_times * input_fps).astype(np.int64), 0, frame_count - 1
    )
    output_ball_visible = ball_visible[nearest_input]
    output_ball_velocity = np.zeros_like(output_ball_pos)
    output_visible_indexes = np.flatnonzero(output_ball_visible)
    if len(output_visible_indexes) >= 2:
      output_ball_velocity[output_visible_indexes] = np.gradient(
        output_ball_pos[output_visible_indexes], time_step, axis=0
      )

  return ResampledMotion(
    input_fps=input_fps,
    input_frames=frame_count,
    fps=output_fps,
    root_pos=output_root_pos.astype(np.float32),
    root_quat_wxyz=output_quat_wxyz.astype(np.float32),
    root_lin_vel=np.gradient(output_root_pos, time_step, axis=0).astype(np.float32),
    root_ang_vel=_angular_velocity_w(output_quat_wxyz, time_step).astype(np.float32),
    joint_pos=output_joint_pos.astype(np.float32),
    joint_vel=np.gradient(output_joint_pos, time_step, axis=0).astype(np.float32),
    joint_names=joint_names,
    finger_joint_pos=(
      output_finger_joint_pos.astype(np.float32)
      if output_finger_joint_pos is not None
      else None
    ),
    finger_joint_names=finger_joint_names,
    ball_pos=(
      output_ball_pos.astype(np.float32) if output_ball_pos is not None else None
    ),
    ball_lin_vel=(
      output_ball_velocity.astype(np.float32)
      if output_ball_velocity is not None
      else None
    ),
    ball_visible=output_ball_visible,
  )


def _discover_motions(input_path: Path, pattern: str) -> tuple[list[Path], Path]:
  if input_path.is_file():
    return [input_path], input_path.parent
  if not input_path.is_dir():
    raise FileNotFoundError(f"Input path does not exist: {input_path}")
  paths = sorted(input_path.rglob(pattern))
  if not paths:
    raise FileNotFoundError(f"No files matching {pattern!r} under {input_path}")
  return paths, input_path


def _project_inspire_mimic_joints(
  joint_pos: np.ndarray, joint_names: tuple[str, ...]
) -> np.ndarray:
  """Project recorded finger angles onto the Inspire hand's equality constraints."""
  result = joint_pos.copy()
  index = {name: i for i, name in enumerate(joint_names)}
  for side in ("l", "r"):
    thumb_pitch = result[:, index[f"{side}_ih_thumb_proximal_pitch"]]
    result[:, index[f"{side}_ih_thumb_intermediate"]] = 1.334 * thumb_pitch
    result[:, index[f"{side}_ih_thumb_distal"]] = 0.667 * thumb_pitch
    for finger in ("index", "middle", "ring", "pinky"):
      proximal = result[:, index[f"{side}_ih_{finger}_proximal"]]
      result[:, index[f"{side}_ih_{finger}_intermediate"]] = (
        -0.04545 + 1.06399 * proximal
      )
  return result


def _write_robot_frame(
  robot: Any,
  sim: Any,
  root_pos: np.ndarray,
  root_quat_wxyz: np.ndarray,
  root_lin_vel: np.ndarray,
  root_ang_vel: np.ndarray,
  joint_pos: np.ndarray,
  joint_vel: np.ndarray,
  device: str,
) -> None:
  import torch

  root_state = robot.data.default_root_state.clone()
  root_state[:, 0:3] = torch.from_numpy(root_pos).to(device)
  root_state[:, 3:7] = torch.from_numpy(root_quat_wxyz).to(device)
  root_state[:, 7:10] = torch.from_numpy(root_lin_vel).to(device)
  root_state[:, 10:13] = torch.from_numpy(root_ang_vel).to(device)
  robot.write_root_state_to_sim(root_state)
  robot.write_joint_state_to_sim(
    torch.from_numpy(joint_pos[None]).to(device),
    torch.from_numpy(joint_vel[None]).to(device),
  )
  sim.forward()


def _compute_spherical_foot_ground_offset(
  robot: Any,
  sim: Any,
  motion: ResampledMotion,
  joint_pos: np.ndarray,
  joint_vel: np.ndarray,
  foot_geom_pattern: str,
  target_clearance: float,
  device: str,
) -> tuple[np.ndarray, np.ndarray]:
  """Find a smooth per-frame root-Z correction from the foot collision spheres."""
  import mujoco
  from scipy.signal import savgol_filter

  foot_local_ids, foot_names = robot.find_geoms(foot_geom_pattern)
  if not foot_local_ids:
    raise ValueError(f"No foot collision geoms matched {foot_geom_pattern!r}.")
  global_ids = robot.indexing.geom_ids[foot_local_ids].cpu().numpy()
  geom_types = sim.mj_model.geom_type[global_ids]
  if np.any(
    geom_types != int(mujoco.mjtGeom.mjGEOM_SPHERE)  # ty: ignore[unresolved-attribute]
  ):
    bad = [
      name
      for name, geom_type in zip(foot_names, geom_types, strict=True)
      if geom_type != int(mujoco.mjtGeom.mjGEOM_SPHERE)  # ty: ignore[unresolved-attribute]
    ]
    raise ValueError(f"Grounding currently expects spherical foot geoms, got: {bad}")
  radii = sim.mj_model.geom_size[global_ids, 0]
  frame_minimums: list[float] = []
  for frame in range(motion.frames):
    _write_robot_frame(
      robot,
      sim,
      motion.root_pos[frame],
      motion.root_quat_wxyz[frame],
      motion.root_lin_vel[frame],
      motion.root_ang_vel[frame],
      joint_pos[frame],
      joint_vel[frame],
      device,
    )
    centers_z = robot.data.geom_pos_w[0, foot_local_ids, 2].cpu().numpy()
    frame_minimums.append(float(np.min(centers_z - radii)))
  frame_minimum_array = np.asarray(frame_minimums, dtype=np.float64)
  raw_offsets = np.maximum(target_clearance - frame_minimum_array, 0.0)
  window_length = min(
    9, len(raw_offsets) if len(raw_offsets) % 2 else len(raw_offsets) - 1
  )
  if window_length >= 3:
    offsets = savgol_filter(
      raw_offsets, window_length=window_length, polyorder=2, mode="interp"
    )
  else:
    offsets = raw_offsets
  offsets = np.maximum(offsets, 0.0)
  clearance_shortfall = target_clearance - np.min(frame_minimum_array + offsets)
  offsets += max(float(clearance_shortfall), 0.0)
  return offsets.astype(np.float32), frame_minimum_array


def build_catch_dataset(
  manifest: list[dict[str, Any]], output_dir: Path
) -> Path | None:
  """Merge converted catch clips with boundaries for multi-motion training."""
  records = [record for record in manifest if record["status"] == "converted"]
  if not records:
    return None

  arrays: dict[str, list[np.ndarray]] = {key: [] for key in CATCH_MOTION_KEYS}
  clip_starts: list[int] = []
  clip_lengths: list[int] = []
  clip_names: list[str] = []
  source_files: list[str] = []
  joint_names: np.ndarray | None = None
  fps: float | None = None
  frame_cursor = 0
  for record in records:
    path = Path(record["output"])
    with np.load(path, allow_pickle=False) as data:
      current_fps = float(np.asarray(data["fps"]).reshape(-1)[0])
      current_joint_names = np.asarray(data["joint_names"])
      if fps is None:
        fps = current_fps
        joint_names = current_joint_names
      elif current_fps != fps or not np.array_equal(current_joint_names, joint_names):
        raise ValueError(f"Incompatible catch clip in dataset: {path}")

      length = int(data["joint_pos"].shape[0])
      clip_starts.append(frame_cursor)
      clip_lengths.append(length)
      clip_names.append(path.stem)
      source_files.append(str(np.asarray(data["source_file"]).item()))
      frame_cursor += length
      for key in CATCH_MOTION_KEYS:
        arrays[key].append(np.asarray(data[key]))

  assert fps is not None and joint_names is not None
  destination = output_dir / "g1_inspire_catch_dataset.npz"
  temporary = destination.with_suffix(".tmp.npz")
  concatenated = {key: np.concatenate(values, axis=0) for key, values in arrays.items()}
  np.savez_compressed(
    temporary,
    fps=np.asarray([fps], dtype=np.float64),
    joint_pos=concatenated["joint_pos"],
    joint_vel=concatenated["joint_vel"],
    body_pos_w=concatenated["body_pos_w"],
    body_quat_w=concatenated["body_quat_w"],
    body_lin_vel_w=concatenated["body_lin_vel_w"],
    body_ang_vel_w=concatenated["body_ang_vel_w"],
    ball_pos_w=concatenated["ball_pos_w"],
    ball_lin_vel_w=concatenated["ball_lin_vel_w"],
    ball_visible=concatenated["ball_visible"],
    joint_names=joint_names,
    clip_starts=np.asarray(clip_starts, dtype=np.int64),
    clip_lengths=np.asarray(clip_lengths, dtype=np.int64),
    clip_names=np.asarray(clip_names),
    source_files=np.asarray(source_files),
    ball_diameter_m=np.asarray(0.15, dtype=np.float64),
    ball_mass_kg=np.asarray(0.112, dtype=np.float64),
  )
  temporary.replace(destination)
  print(
    f"Wrote catch dataset: {destination} ({len(records)} clips, {frame_cursor} frames)"
  )
  return destination


def convert_catch_motions(
  input_path: Path,
  output_dir: Path,
  pattern: str,
  output_fps: float,
  ground_clearance: float,
  device: str,
  overwrite: bool,
  limit: int | None,
) -> list[dict[str, Any]]:
  """Convert G1 body, Inspire fingers, and ball tracks for the catch task."""
  _configure_warp_cache()

  from mjlab.entity import Entity
  from mjlab.scene import Scene
  from mjlab.sim.sim import Simulation, SimulationCfg

  from unitree_g1_catch.tasks.env_cfg import unitree_g1_inspire_catch_env_cfg

  paths, source_root = _discover_motions(input_path, pattern)
  if limit is not None:
    paths = paths[:limit]

  sim_cfg = SimulationCfg(nconmax=256, njmax=2048)
  sim_cfg.mujoco.timestep = 1.0 / output_fps
  scene = Scene(unitree_g1_inspire_catch_env_cfg().scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  robot: Entity = scene["robot"]

  output_dir.mkdir(parents=True, exist_ok=True)
  existing_manifest_file = output_dir / "manifest.json"
  existing_by_source: dict[str, dict[str, Any]] = {}
  if existing_manifest_file.exists():
    existing_records = json.loads(existing_manifest_file.read_text(encoding="utf-8"))
    existing_by_source = {
      record["source"]: record
      for record in existing_records
      if record.get("status") == "converted"
    }
  manifest: list[dict[str, Any]] = []
  for motion_index, source_path in enumerate(paths, start=1):
    relative = source_path.relative_to(source_root)
    destination = output_dir / relative.parent / f"{source_path.stem}_catch_mjlab.npz"
    if destination.exists() and not overwrite:
      print(f"[{motion_index}/{len(paths)}] exists, skipping: {destination}")
      source_key = str(source_path.resolve())
      if source_key in existing_by_source:
        manifest.append(existing_by_source[source_key])
      else:
        with np.load(destination, allow_pickle=False) as existing:
          manifest.append(
            {
              "source": source_key,
              "output": str(destination.resolve()),
              "status": "converted",
              "output_frames": int(existing["joint_pos"].shape[0]),
              "output_fps": float(np.asarray(existing["fps"]).reshape(-1)[0]),
            }
          )
      continue

    motion = load_and_resample_motion(source_path, output_fps)
    if (
      motion.finger_joint_pos is None
      or motion.ball_pos is None
      or motion.ball_lin_vel is None
      or motion.ball_visible is None
    ):
      record = {
        "source": str(source_path.resolve()),
        "status": "skipped_missing_fingers_or_ball",
      }
      manifest.append(record)
      print(f"[{motion_index}/{len(paths)}] missing ball/fingers, skipping")
      continue

    source_names = (*motion.joint_names, *motion.finger_joint_names)
    mapped_names = (
      *motion.joint_names,
      *(FINGER_NAME_MAP.get(name, name) for name in motion.finger_joint_names),
    )
    if len(set(mapped_names)) != len(mapped_names):
      raise ValueError(f"Mapped joint names are not unique for {source_path}.")
    source_by_model_name = {
      model_name: source_index for source_index, model_name in enumerate(mapped_names)
    }
    missing = sorted(set(robot.joint_names) - set(source_by_model_name))
    extra = sorted(set(mapped_names) - set(robot.joint_names))
    if missing or extra:
      raise ValueError(
        f"G1 + Inspire joint mismatch for {source_path}: "
        f"missing={missing}, extra={extra}, source={source_names}"
      )

    source_joint_pos = np.concatenate(
      (motion.joint_pos, motion.finger_joint_pos), axis=1
    )
    source_order = [source_by_model_name[name] for name in robot.joint_names]
    model_joint_pos = source_joint_pos[:, source_order]
    model_joint_pos = _project_inspire_mimic_joints(
      model_joint_pos, robot.joint_names
    ).astype(np.float32)
    model_joint_vel = np.gradient(model_joint_pos, 1.0 / output_fps, axis=0).astype(
      np.float32
    )

    scene.reset()
    ground_offsets, foot_heights = _compute_spherical_foot_ground_offset(
      robot,
      sim,
      motion,
      model_joint_pos,
      model_joint_vel,
      r"^(left|right)_ankle_roll_link_collision_.*$",
      ground_clearance,
      device,
    )
    grounded_root_pos = motion.root_pos.copy()
    grounded_root_pos[:, 2] += ground_offsets
    grounded_root_lin_vel = np.gradient(
      grounded_root_pos, 1.0 / output_fps, axis=0
    ).astype(np.float32)
    grounded_ball_pos = motion.ball_pos.copy()
    grounded_ball_pos[:, 2] += ground_offsets
    grounded_ball_lin_vel = motion.ball_lin_vel.copy()
    grounded_ball_lin_vel[:, 2] += np.gradient(ground_offsets, 1.0 / output_fps)

    log: dict[str, list[np.ndarray]] = {
      "joint_pos": [],
      "joint_vel": [],
      "body_pos_w": [],
      "body_quat_w": [],
      "body_lin_vel_w": [],
      "body_ang_vel_w": [],
    }
    for frame in range(motion.frames):
      _write_robot_frame(
        robot,
        sim,
        grounded_root_pos[frame],
        motion.root_quat_wxyz[frame],
        grounded_root_lin_vel[frame],
        motion.root_ang_vel[frame],
        model_joint_pos[frame],
        model_joint_vel[frame],
        device,
      )
      log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
      log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
      log["body_pos_w"].append(robot.data.body_link_pos_w[0].cpu().numpy().copy())
      log["body_quat_w"].append(robot.data.body_link_quat_w[0].cpu().numpy().copy())
      log["body_lin_vel_w"].append(
        robot.data.body_link_lin_vel_w[0].cpu().numpy().copy()
      )
      log["body_ang_vel_w"].append(
        robot.data.body_link_ang_vel_w[0].cpu().numpy().copy()
      )

    output_data = {
      key: np.stack(values).astype(np.float32) for key, values in log.items()
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.npz")
    np.savez_compressed(
      temporary,
      fps=np.asarray([output_fps], dtype=np.float64),
      joint_pos=output_data["joint_pos"],
      joint_vel=output_data["joint_vel"],
      body_pos_w=output_data["body_pos_w"],
      body_quat_w=output_data["body_quat_w"],
      body_lin_vel_w=output_data["body_lin_vel_w"],
      body_ang_vel_w=output_data["body_ang_vel_w"],
      ball_pos_w=grounded_ball_pos.astype(np.float32),
      ball_lin_vel_w=grounded_ball_lin_vel.astype(np.float32),
      ball_visible=motion.ball_visible.astype(bool),
      joint_names=np.asarray(robot.joint_names),
      source_file=np.asarray(str(source_path.resolve())),
      ground_offset_m=ground_offsets,
      ball_diameter_m=np.asarray(0.15, dtype=np.float64),
      ball_mass_kg=np.asarray(0.112, dtype=np.float64),
    )
    temporary.replace(destination)

    record = {
      "source": str(source_path.resolve()),
      "output": str(destination.resolve()),
      "status": "converted",
      "input_frames": motion.input_frames,
      "input_fps": motion.input_fps,
      "output_frames": motion.frames,
      "output_fps": output_fps,
      "joint_count": len(robot.joint_names),
      "actuated_joint_count": len(robot.actuator_names),
      "body_count": len(robot.body_names),
      "ball_visible_frames": int(motion.ball_visible.sum()),
      "minimum_foot_height_before_m": float(foot_heights.min()),
      "foot_height_5th_percentile_before_m": float(np.quantile(foot_heights, 0.05)),
      "ground_offset_min_m": float(ground_offsets.min()),
      "ground_offset_median_m": float(np.median(ground_offsets)),
      "ground_offset_max_m": float(ground_offsets.max()),
      "corrected_foot_height_min_m": float(np.min(foot_heights + ground_offsets)),
      "target_ground_clearance_m": ground_clearance,
      "ball_diameter_m": 0.15,
      "ball_mass_kg": 0.112,
    }
    manifest.append(record)
    print(
      f"[{motion_index}/{len(paths)}] {source_path.name}: "
      f"{motion.frames} frames, ground shift "
      f"{ground_offsets.min():+.4f}..{ground_offsets.max():+.4f} m -> "
      f"{destination}"
    )

  manifest_file = output_dir / "manifest.json"
  manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
  print(f"Wrote catch conversion manifest: {manifest_file}")
  build_catch_dataset(manifest, output_dir)
  return manifest


def _path(value: str) -> Path:
  return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  subparsers = parser.add_subparsers(dest="command", required=True)

  catch_parser = subparsers.add_parser(
    "convert-catch",
    help="convert body, Inspire fingers, and ball tracks for reactive catching",
  )
  catch_parser.add_argument("--input", type=_path, default=PROJECT_ROOT / "data")
  catch_parser.add_argument(
    "--output-dir", type=_path, default=PROJECT_ROOT / "converted_catch"
  )
  catch_parser.add_argument("--pattern", default="*_filtered_g1.npz")
  catch_parser.add_argument("--output-fps", type=float, default=50.0)
  catch_parser.add_argument("--ground-clearance", type=float, default=0.005)
  catch_parser.add_argument("--device", default="cpu")
  catch_parser.add_argument("--overwrite", action="store_true")
  catch_parser.add_argument("--limit", type=int)
  return parser


def main() -> None:
  args = build_parser().parse_args()
  if args.command == "convert-catch":
    convert_catch_motions(
      input_path=args.input,
      output_dir=args.output_dir,
      pattern=args.pattern,
      output_fps=args.output_fps,
      ground_clearance=args.ground_clearance,
      device=args.device,
      overwrite=args.overwrite,
      limit=args.limit,
    )
  else:
    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
  main()

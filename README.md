# Unitree G1 reactive ball catching

This is the standalone version of the working `mjlab` G1 + Inspire-hand
motion-imitation experiment. It extends `mjlab` as an ordinary Python package;
it does not modify or vendor mjlab source files.

The registered task is `Mjlab-Catch-Flat-Unitree-G1-Inspire`. Its actor receives
only live robot and ball state (140 values) and outputs 41 normalized
joint-position actions: 29 G1 body joints and 12 independently driven Inspire
joints. mjlab converts them to position targets with the configured scale and
default offset. The critic and reward functions may use the complete reference
motion during PPO training.

The ball is a dynamic MuJoCo sphere with a diameter of 0.15 m and a mass of
0.112 kg. It collides with the robot and ground; the recorded trajectory is a
training reference, not a visual-only animation.

## Repository boundaries

- `mjlab==1.6.0` is a normal uv dependency and supplies simulation, the stock
  tracking environment, PPO, viewers, and ONNX export.
- `third_party/unitree_g1_inspire` is the pinned simulation model. Its upstream
  repository currently has no open-source license file, so confirm its terms
  before redistribution.
- `third_party/unitree_rl_mjlab` is Unitree's pinned official integration
  reference. It is not imported into training because it pins mjlab 1.2.0. Its
  G1 motor constants match mjlab 1.6.0; its `unitree_sdk2`/C++ deployment path is
  the basis for a later real-robot adapter.
- `third_party/dfx_inspire_service` is Unitree's official 12-motor RH56DFX hand
  controller and defines the DDS command/state order used by our deployment map.

Initialize all submodules after cloning:

```bash
git submodule update --init --recursive
```

## Install

macOS evaluation/conversion:

```bash
uv sync --extra cpu
```

Linux training with CUDA 12.8 wheels:

```bash
uv sync --extra cu128
```

Training requires an NVIDIA GPU. Conversion and playback work on CPU.

## Data conversion

The source recordings are not assumed to use mjlab's schema. Convert them with:

```bash
uv run g1-catch convert-catch \
  --input ../data \
  --output-dir ../converted_catch \
  --overwrite
```

The converter validates the actual arrays, resamples to 50 Hz, maps all recorded
finger joints, enforces the Inspire mechanical couplings, grounds the motion with
the real collision geometry, transforms the ball consistently, derives MuJoCo
body states, and writes clip-boundary metadata. The merged result is
`../converted_catch/g1_inspire_catch_dataset.npz`.

## Train and play

The wrappers register this external task and then call mjlab's unmodified CLIs:

```bash
uv run g1-catch-train Mjlab-Catch-Flat-Unitree-G1-Inspire \
  --env.commands.motion.motion-file \
    ../converted_catch/g1_inspire_catch_dataset.npz \
  --env.scene.num-envs 4096
```

The mjlab default logger is W&B. Add `--agent.logger tensorboard` for an entirely
local run.

```bash
uv run g1-catch-play Mjlab-Catch-Flat-Unitree-G1-Inspire \
  --checkpoint-file /absolute/path/to/model_N.pt \
  --motion-file \
    ../converted_catch/example/example_catch_mjlab.npz \
  --viewer viser --device cpu
```

When the merged dataset is selected, Viser includes a named clip dropdown. PPO
samples all clips; the dropdown only selects what the single interactive rollout
shows.

## Code map

- `assets/g1_inspire.py` loads the pinned combined MJCF, replaces its motors with
  mjlab position actuators, and adds the 12 independent finger actions.
- `mdp/commands.py` extends mjlab's motion command with physical-ball arrays,
  clip boundaries, ball spawning, and the Viser clip selector.
- `mdp/observations.py` exposes live torso-relative ball state to the actor and
  reference ball state to the critic.
- `mdp/rewards.py` defines ball tracking, palm proximity, and hand contact terms.
- `tasks/env_cfg.py` composes those additions with mjlab's tracking environment.
- `tasks/rl_cfg.py` configures the existing RSL-RL PPO runner.
- `motions/pipeline.py` validates and converts the retargeted NPZ recordings.

## Policy interface

The actor's 140 inputs are concatenated in this order:

| Observation | Size |
| --- | ---: |
| Base linear velocity | 3 |
| Base angular velocity | 3 |
| Projected gravity | 3 |
| Torso height | 1 |
| Joint positions | 41 |
| Joint velocities | 41 |
| Previous action | 41 |
| Torso-relative ball position | 3 |
| Torso-relative ball velocity | 3 |
| Ball visibility | 1 |

The actor is an MLP with hidden dimensions `(512, 256, 128)` and ELU
activations. Its 41 normalized outputs become joint-position targets after
mjlab applies per-joint scales and default-position offsets. Reference robot and
ball states are privileged training information used only by the critic and
rewards.

## Real-robot boundary

The policy's deployment output must be split **by joint name**, then reordered for
the two hardware APIs:

```text
41 policy actions in combined-MJCF order
├── named 29 G1 actions -> G1 motor-ID order -> Unitree SDK2
└── named 12 hand actions -> DFX right/left DDS order -> rt/inspire/cmd
```

The hand joints occur between the left and right arm joints in the policy vector,
so a contiguous `[:29]`/`[29:]` split is wrong. The name-based permutations live in
`unitree_g1_catch.deployment`. They operate on raw normalized actions; the real
controller must still reproduce mjlab's per-joint scaling, default offsets,
clipping, and encoder-bias convention.

Real deployment also needs a synchronized 140-value observation adapter in the
same named joint order, using G1 joint/IMU state and an external 3D ball tracker.
The reference-motion-only values used by the critic are not needed on the robot.
Before physical deployment, compare observation order, normalization, action
scales, offsets, and clipping with `third_party/unitree_rl_mjlab/deploy/robots/g1`,
then validate the controller in its integrated `unitree_mujoco` path.

## Generated MuJoCo type stubs

MuJoCo exposes most of its Python API from compiled extension modules without
distributing type stubs. This repository checks in stubs generated from the
locked MuJoCo version so ty can resolve that API.

Regenerate them after updating MuJoCo or the pinned stub generator:

```bash
bash typings/generate_mujoco_stubs.sh
```

The workflow is adapted from mjlab's own stub generator. It uses a pinned
`pybind11-stubgen`, then removes platform-specific rendering bindings and other
machine-dependent output so `typings/mujoco` is reproducible on macOS and Linux.

## Checks

```bash
uv run ruff format --check
uv run ruff check
bash typings/generate_mujoco_stubs.sh
git diff --exit-code -- typings/mujoco
uv run ty check
```

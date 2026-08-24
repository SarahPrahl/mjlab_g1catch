# Unitree G1 catch

This package adds a ball-catching task to
[mjlab](https://mujocolab.github.io/mjlab/main/index.html). It uses the Unitree
G1 model with Inspire hands.

The project has two training stages:

1. Stage 1 tracks retargeted catch recordings.
2. Stage 2 starts from the stage 1 actor and learns from simulated ball throws.

The environment uses a dynamic MuJoCo ball with a diameter of 0.15 m and a mass
of 0.112 kg.

## Requirements

- Install [uv](https://docs.astral.sh/uv/).
- Use Linux x86-64 and an NVIDIA GPU for training.

Initialize the submodules:

```bash
git submodule update --init --recursive
```

Install the CUDA environment on the training computer:

```bash
uv sync --extra cu128
```

For CPU use, instead install this environment:

```bash
uv sync --extra cpu
```

The repository includes the source NPZ files in `data/`. It excludes converted
datasets, checkpoints, and logs.

## Prepare the motion data

Put the retargeted NPZ files in `data/`. The converter selects all NPZ files by
default.

```bash
uv run g1-catch convert-catch
```

The converter validates the input, converts each clip, and writes the merged
dataset to `converted_catch/g1_inspire_catch_dataset.npz`. Use
`uv run g1-catch convert-catch --help` to change the paths or file pattern.

## Train stage 1

Stage 1 uses the recordings as reference motions. Training uses a mix of flat
floors and small floor variations.

```bash
uv run g1-catch-train Mjlab-Catch-Flat-Unitree-G1-Inspire \
  --env.commands.motion.motion-file \
    converted_catch/g1_inspire_catch_dataset.npz \
  --env.scene.num-envs 2048 \
  --agent.logger tensorboard
```

## Train stage 2

Stage 2 does not use a motion file. It creates physical ball throws and trains a
new critic. It loads only the actor and its observation normalizer from stage 1.

```bash
uv run g1-catch-train Mjlab-Catch-Throws-Unitree-G1-Inspire \
  --agent.pretrained-actor-checkpoint \
    path/to/stage-1/model_N.pt \
  --env.scene.num-envs 2048 \
  --agent.logger tensorboard
```

Do not use `--agent.resume` for the first stage 2 run. Use it only to continue a
stage 2 run.

Training writes checkpoints and logs below `logs/rsl_rl/`. Start TensorBoard at
any time:

```bash
uv run tensorboard --logdir logs/rsl_rl --port 6006
```

## Play a checkpoint

Play a stage 1 checkpoint with a converted clip or with the merged dataset:

```bash
uv run g1-catch-play Mjlab-Catch-Flat-Unitree-G1-Inspire \
  --checkpoint-file path/to/model_N.pt \
  --motion-file converted_catch/g1_inspire_catch_dataset.npz \
  --viewer viser
```

Play a stage 2 checkpoint:

```bash
uv run g1-catch-play Mjlab-Catch-Throws-Unitree-G1-Inspire \
  --checkpoint-file path/to/model_N.pt \
  --viewer viser
```

The command prints the Viser URL. Add `--device cpu` for CPU playback.

## Use the remote workstation

Connect to the workstation and go to the repository:

```bash
ssh workstation
cd path/to/clone
```

Replace `workstation` with the SSH target or alias.

Start a play command from the previous section. Keep that terminal open. Start
TensorBoard in a second workstation terminal if you need metrics:

```bash
uv run tensorboard --logdir logs/rsl_rl --port 6006
```

On the local computer, open one tunnel for both services:

```bash
ssh -N \
  -L 8080:127.0.0.1:8080 \
  -L 6006:127.0.0.1:6006 \
  workstation
```

Open <http://localhost:8080> for Viser. Open <http://localhost:6006> for
TensorBoard.

## Policy interface

Both actors use the same 136-value observation:

| Observation                         | Values |
| ----------------------------------- | -----: |
| Base angular velocity               |      3 |
| Projected gravity                   |      3 |
| Joint positions                     |     41 |
| Joint velocities                    |     41 |
| Previous action                     |     41 |
| Ball position relative to the torso |      3 |
| Ball velocity relative to the torso |      3 |
| Ball visibility                     |      1 |

The actor is an MLP with layers of 512, 256, and 128 units. It uses ELU
activations. It outputs 41 normalized joint-position actions for the G1 body and
the Inspire hands.

The critic and the rewards can use privileged simulation or reference data. The
actor does not use this data.

## Project structure

| Path                      | Purpose                                           |
| ------------------------- | ------------------------------------------------- |
| `assets/g1_inspire.py`    | Robot, actuators, and hand collision model        |
| `deployment/interface.py` | Policy-to-hardware joint order                    |
| `mdp/commands.py`         | Recorded motions and simulated throws             |
| `mdp/observations.py`     | Actor and critic input functions                  |
| `mdp/rewards.py`          | Reward functions                                  |
| `mdp/terminations.py`     | Stage 2 success test                              |
| `motions/pipeline.py`     | NPZ validation and conversion                     |
| `tasks/env_cfg.py`        | Environments, ball model, terrain, and task terms |
| `tasks/rl_cfg.py`         | PPO and actor configuration                       |
| `tasks/runner.py`         | Stage 1 to stage 2 actor transfer                 |
| `cli.py`                  | mjlab train and play wrappers                     |
| `third_party/`            | G1 with Inspire hands model submodule             |
| `typings/`                | MuJoCo type stubs used by `ty`              |

The `deployment` package only maps policy actions to hardware joint order. This
repository does not include a complete real-robot controller yet.

For the framework design and command options, read the mjlab
[architecture guide](https://mujocolab.github.io/mjlab/main/source/architecture_overview.html)
and [RSL-RL training guide](https://mujocolab.github.io/mjlab/main/source/training/rsl_rl.html).

## Checks

```bash
uv run ruff format
uv run ruff check
uv run ty check
```

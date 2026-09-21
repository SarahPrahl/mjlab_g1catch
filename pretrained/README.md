# Pretrained models

Run all commands from the repository root. These models match the current
137-input MLP actor and 41 joint actions, including the idle/armed flag.
Older 136-input models are deliberately not included.

| Model | Use it for | Training |
| --- | --- | --- |
| [phase1_tracking.pt](phase1_tracking.pt) | Starting a new Phase-2 experiment from recorded catching motions. Flag is always armed in Phase 1; release is not learned yet. | 20k iterations, 2,560 environments, corrected 149-recording dataset and flat-to-rough terrain curriculum. |
| [phase2_catch_release.pt](phase2_catch_release.pt) | Trying the latest complete catching policy in simulation. Includes idle/armed/release behavior, lowered hard catch box, stronger finger-contact rewards and 60% rehearsal. | Another 20k iterations, initialized from the Phase-1 model above. Latest result, not a claim of best performance. |

## Try the latest policy

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and
[Git LFS](https://git-lfs.com/), then:

```bash
git lfs install
git lfs pull
git submodule update --init --recursive
uv sync --locked --extra cpu --no-dev
uv run --no-sync g1-catch-play Mjlab-Catch-Throws-Unitree-G1-Inspire \
  --checkpoint-file pretrained/phase2_catch_release.pt \
  --viewer viser --device cpu --num-envs 1
```

Open the Viser URL printed in the terminal, normally http://localhost:8080.
No motion dataset is needed for Phase-2 playback. Modes switch automatically.
CPU playback can be slower than real time. Ctrl+C stops it.

## Start a new Phase-2 training experiment

Requires Linux, a CUDA GPU, and a converted catch dataset. Replace the dataset
path below with your own. This starts a new run from the Phase-1 actor and its
observation normalizer; it does not resume its critic, optimizer, or iteration.

```bash
uv sync --locked --extra cu128 --no-dev
uv run --no-sync g1-catch-train Mjlab-Catch-Throws-Unitree-G1-Inspire \
  --agent.pretrained-actor-checkpoint pretrained/phase1_tracking.pt \
  --env.commands.throw.motion-cfg.motion-file /path/to/g1_inspire_catch_dataset.npz \
  --env.scene.num-envs 2048 \
  --agent.max-iterations 20000 \
  --agent.run-name pretrained-phase2 \
  --agent.logger tensorboard
```

Do not add `--agent.resume` to this command. The dataset supplies the reference
rehearsal portion of Phase-2 training. Adapt environment count to available VRAM.

## Provenance and compatibility

Both files are unchanged copies of `model_19999.pt` (zero-based final iteration):

- Phase 1: `2026-09-18_18-02-37_mlp-mode-2560-20k-phase1`.
- Phase 2: `2026-09-19_06-30-13_mlp-mode-2560-20k-phase2`.

The current idle/armed-mode source changes must be shared alongside these files;
they are still uncommitted as of September 21, 2026. Commit `5a83b9b` alone does
not support these models. No commit or push was made when creating this folder.

These are simulation checkpoints, not a real-robot deployment package. Keep the
matching observation/action order and robot model when using them. Training
settings and all historical checkpoints remain in the separate local archive.

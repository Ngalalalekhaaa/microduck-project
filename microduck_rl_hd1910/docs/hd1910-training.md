# HD1910 walking baselines

Independent HOME correction experiment: [HOME calibration training](home-calibration-training.md).
It uses privileged true-pose rewards/critic and a separate sensor-history actor;
its 392D/14-increment interface is not the walking policy interface below.

These tasks use the public LuwuDynamics 1910 BAM M6 parameter file. They keep
this Microduck revision's walk geometry, HOME, rewards and policy contract
(50 Hz, 61 actor observations, 14 position actions). Both start with random
policy weights unless checkpoint loading is explicitly requested.

| Task | Experiment directory |
|---|---|
| `Mjlab-Velocity-Flat-MicroDuck-HD1910` | `microduck_hd1910_velocity` |
| `Mjlab-Velocity-Flat-Backlash-MicroDuck-HD1910` | `microduck_hd1910_velocity_backlash` |

The actuator settings are `kp_fw=5`, supply 7.4–8.0 V, load-drop gain 0–0.2,
voltage floor 7.0 V, and command delay 3–6 physics steps (15–30 ms). Backlash
uses the walk model with assumed ±1° hinges. The encoder-through-backlash
position feedback and per-world Feetech target-history reset are both active.
Neither task uses the XL330 actuator.

The robot's actual backlash, mass/inertia, firmware settings, supply sag and
Radxa/HAT/IMU timing still need measurement. In particular, ±1° play and the
output-side encoder assumption are simulation assumptions. Parameter source,
fixed revision, checksum and license: [1910 provenance](../src/mjlab_microduck/robot/hd1910/README.md).

## Validate before training

From this worktree, reuse the existing environment without installing into it:

```bash
export PYTHONPATH=/home/luckysir/microduck/microduck_rl_hd1910/src
export HD1910_PYTHON=/home/luckysir/microduck/microduck_rl/.venv/bin/python
CUDA_VISIBLE_DEVICES='' "$HD1910_PYTHON" -m pytest -q \
  tests/test_hd1910_cfg.py tests/test_feetech_bam.py
```

Only after the preceding GPU jobs finish, run the integration gate:

```bash
MICRODUCK_TEST_GPU=1 "$HD1910_PYTHON" -m pytest -q -s \
  tests/test_hd1910_rollout.py
```

This checks both tasks with four worlds: actual rollout, finite outputs,
61/14 dimensions and partial resets. It also holds HOME for at least three
seconds at fixed 7.4 V with no domain randomization, pushes or automatic resets.
One world starts nominally; three receive deterministic joint and roll/pitch
perturbations up to 0.01 rad. Each world's maximum trunk tilt must be below
15°, and final trunk height above 0.09 m. Tilt and final height are printed as
`HD1910_HOME_HOLD` JSON. Failure blocks the long run; these thresholds are a
simulation screening check, not evidence of hardware stability.

Then perform a five-iteration, 64-environment training smoke test and standard
ONNX export for each task before the requested long training runs. Use the
task's own config to evaluate physics. The existing `scripts/infer_policy.py`
still defaults to XL330 BAM and is not an HD1910 evaluation path.

## Checks performed during integration

CPU configuration and actuator tests cover parameter checksum, registered
train/play configurations, unchanged walk geometry, 61 observations, 14 active
servos, factory isolation, Feetech state initialization, continuous limiter
history, reset selections, and output-side position/motor-side velocity.
They do not execute Warp physics. The GPU tests above remain a separate gate.

## Robust continuation (2026-10-02)

New, separate tasks keep the 61/14 policy contract and use the same public
HD1910 BAM parameters:

- `Mjlab-Velocity-Flat-MicroDuck-HD1910-Robust`
- `Mjlab-Velocity-Flat-Backlash-MicroDuck-HD1910-Robust`

The source is the supplied `model_5000_flat.pt` / `model_5000_backlash.pt`,
not the earlier locally trained 9750/9729 checkpoints. Architecture, strict
weight loading and ONNX parity were checked. The original environment files
for the supplied checkpoints are unavailable; the continuation saves its own
complete environment and PPO configuration.

Changes:

- Actor positions use 4096-count quantization and velocities use consecutive
  20 ms encoder samples, including output-side backlash.
- Ground friction ends at 0.45–1.25; trunk/head mass and inertia vary ±10%,
  leg mass/inertia ±5%. Trunk COM varies up to ±8/6/4 mm, head ±5/5/3 mm.
- Per-joint P response varies ±12%, BAM friction ±20%, torque ceiling 0.9–1.0;
  voltage 6.8–8.0 V, modeled load sag with a 6.4 V floor, delays 10–40 ms.
- Encoder observation bias reaches ±2.86°, IMU mounting error ±6°,
  residual gyro bias ±0.015 rad/s. Encoder bias currently models observation
  error; it does not also perturb the deployment angle-to-ticks conversion.
- Backlash variant varies the assumed ±1° half-width by 0.75–1.25.
- Velocity perturbations reach ±0.25 m/s every 5–9 seconds.
- Head commands: neck/head pitch ±10°, yaw ±20°, roll ±5°, 80% exact zero;
  body command remains zero. Head rewards follow the commanded offset.
- Separate planar and yaw tracking rewards, L1 speed error cost, deliberate
  low-speed, backward, lateral and turn samples. Smoothness increases gradually;
  measured joint-limit proximity is priced without clipping policy targets.

DR strengthens at 0/250/750/1500 additional iterations relative to restored
environment counter 120024. These ranges are assumptions for robustness,
not a new identification of the actual replica.

From `/home/luckysir/microduck`, after the CPU/GPU configuration checks:

```bash
microduck_rl/.venv/bin/python scripts/train_hd1910_robust.py \
  --phase smoke --output runs/20261002_robust_hd1910_head_motion

microduck_rl/.venv/bin/python scripts/train_hd1910_robust.py \
  --phase train --output runs/20261002_robust_hd1910_head_motion \
  --target-iteration 10000 --num-envs 2048
```

Smoke runs 64 worlds for five PPO iterations per variant, exports with the
official exporter, then compares checkpoint and ONNX on 128 raw observations.
The long-run gate verifies source/config fingerprints. Training alternates
1000-label chunks, starting with backlash, preserving original checkpoints.
RSL-RL repeats the saved iteration label on resume; the supervisor accounts
for that when requesting exact checkpoint labels.

Checkpoints save every 500 iterations. Every 1000, export and evaluate 14
commands for 30 seconds on three heldout seeds, 32 worlds per command/seed,
under both nominal and final-DR stress physics. Source weights are evaluated
under those same conditions. JSON/CSV reports record first-episode survival,
speed MAE/RMSE, target jumps and calibrated encoder-limit proxies. Tracking
excludes the failure step and all samples after a world's first failure.
`candidate_ranking.json` flags stationary policies that fail command tracking;
no result is treated as proof of real-world walking.

The supervisor checks free space every five seconds. It starts stages only
above 3.5 GiB and stops its child below 2.5 GiB; it never deletes original
weights. Stop it with:

```bash
touch runs/20261002_robust_hd1910_head_motion/STOP
```

Status is in `train_status.json`; exports, matched evaluations and ranking are
inside that output directory. The Radxa policy is not replaced by training.

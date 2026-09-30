# HD1910 walking baselines

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

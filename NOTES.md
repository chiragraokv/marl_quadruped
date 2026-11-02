# Acrobatics implementation notes

This file records Phase 0 recon, verification gates, and every implementation deviation from [`agent/acrobat/instructions.md`](agent/acrobat/instructions.md).

## Phase 0 — repository findings

1. **Go2 config and geometry:** `src/assets/robots/unitree_go2/go2_constants.py`; MJCF is `src/assets/robots/unitree_go2/xmls/go2.xml`. Its configured root pose is `(0, 0, 0.32)` and default joint map is `.*thigh_joint=0.9`, `.*calf_joint=-1.8`, `.*R_hip_joint=0.1`, `.*L_hip_joint=-0.1`. The Go2 jump task measures a settled standing root height of `0.224 m` with the tuned torque PD controller. The 12 leg joints in compiled MJCF order are:

   | Joint order | Name | Default (rad) | Limit (rad) |
   |---:|---|---:|---:|
   | 1 | FL hip | -0.1 | [-1.0472, 1.0472] |
   | 2 | FL thigh | 0.9 | [-1.5708, 3.4907] |
   | 3 | FL calf | -1.8 | [-2.7227, -0.83776] |
   | 4 | FR hip | +0.1 | [-1.0472, 1.0472] |
   | 5 | FR thigh | 0.9 | [-1.5708, 3.4907] |
   | 6 | FR calf | -1.8 | [-2.7227, -0.83776] |
   | 7 | RL hip | -0.1 | [-1.0472, 1.0472] |
   | 8 | RL thigh | 0.9 | [-0.5236, 4.5379] |
   | 9 | RL calf | -1.8 | [-2.7227, -0.83776] |
   | 10 | RR hip | +0.1 | [-1.0472, 1.0472] |
   | 11 | RR thigh | 0.9 | [-0.5236, 4.5379] |
   | 12 | RR calf | -1.8 | [-2.7227, -0.83776] |

   The velocity task defines built-in position actuators: hip and thigh stiffness/damping `20/1`, calf `40/2`, effort limits `23.5/23.5/45 Nm` (hip/thigh/calf). The XML has no actuator elements; mjlab creates these from `GO2_ARTICULATION`.

2. **Actuators / runtime gains:** installed mjlab provides built-in position, motor, velocity, muscle, ideal-PD, DC, delayed, learned, and XML actuator classes. Go2 position-actuator gains are baked into the model; the action term does not change them per environment while stepping. Explicit PD torques through motor actuators are used for the acrobatics action term. mjlab has reset-time domain-randomization helpers for model fields, including PD/joint gains.

3. **Registration / CLI:** tasks register through `register_mjlab_task(task_id, env_cfg, play_env_cfg, rl_cfg, runner_cls)` and are discovered by `src/tasks/__init__.py`'s package importer. The train and play scripts parse the task ID first, then tyro maps `--env.*` and `--agent.*` into nested config dataclasses. PPO uses `MjlabOnPolicyRunner` and `RslRlVecEnvWrapper`; actor/critic hidden layers, activation, initial standard deviation, and fixed/adaptive schedules are configurable. Resume uses `agent.resume`, `agent.load_run`, and `agent.load_checkpoint`.

4. **Contacts:** the velocity config matches `FR/FL/RR/RL_foot_collision` against terrain. Sensor force is a per-environment vector layout `[env, match, slot, xyz]`; `found` is `[env, match, slot]`. Non-foot ground contact uses the collision-geom pattern with foot geoms excluded and force history; illegal contact checks force magnitude against a threshold.

5. **Step order:** `ManagerBasedRlEnv.step` computes terminations, stores `termination_manager.terminated` / `time_outs`, computes rewards, then resets done environments. Terminal reward functions can read same-step flags and trackers. The acrobatics tracker update is a no-op termination term that runs before the reward manager; reward code then snapshots terminal metrics before reset clears the tracker.

6. **Versions / runner:** `mjlab 1.2.0`, `rsl-rl-lib 5.0.1`, `mujoco-warp 3.5.0`, `mujoco 3.5.0`. `scripts/train.py` defaults to `MjlabOnPolicyRunner`; resume path selection uses `get_checkpoint_path` under the configured experiment directory. `--help` requires the task ID first (bare `scripts/train.py --help` reports the missing task argument).

7. **Randomization:** installed reset-time helpers include `geom_friction`, `body_mass`, `body_com_offset`, `pseudo_inertia`, joint stiffness/damping, actuator fields, and `pd_gains`; the velocity task uses friction, encoder-bias, and base-CoM startup events. Acrobatics randomization is not enabled in this implementation because Gate 4 did not pass.

8. **Self-collision:** Go2 uses `FULL_COLLISION` with `contype=1`, `conaffinity=0`; this intentionally excludes robot self-collision. Ground contact for non-foot collision geoms remains detectable through the sensor.

9. **Timing:** MuJoCo physics timestep `0.005 s`, decimation `4`, environment timestep `0.02 s` (50 Hz control).

**Gate 0:** passed on the RTX 3070. `scripts/train.py Unitree-Go2-Flat --env.scene.num-envs=64 --agent.max-iterations=2` completed both iterations with exit code 0. It used 64 environments.

## Gates and current status

- **Gate 1 — passed:** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./.venv/bin/python -m pytest -q tests/acrobatics` passed the eight pure-math tests before environment smoke cases were added. These cover reward kernels/branches, Go2 home and mirrored mapping, phase transitions/reset, RunningNorm, and quadratic ARS convergence.
- **Gate 2 — direct checks passed with a documented reward deviation:** one-environment standing rollout (149 control steps) stayed within 1 cm of the measured `0.224 m` standing height without early termination; zero and random actions remained finite; the scripted extension visited TAKEOFF→FLIGHT→LANDING, rose above standing height, and the tracked maximum height matched offline replay within 2 cm. Forced-fall and timeout branches are exercised. A short timeout immediately after reset yields a small positive score because the settling transient raises `h_bar`; the test checks a bounded normalized score, not zero. We did not launch `scripts/play.py`'s graphical viewer; dummy policies were exercised through the manager environment.
- **Gate 3 — not passed:** both motions received 500-iteration ARS runs with 32 directions/64 environments and the corrected crash penalty. Best in-place training eval reached 46.9% flight/no-early-contact success (step 229); final training eval was 18.8%. Best forward training eval return occurred at step 379, but mean forward displacement peaked at `0.224 m` (step 409), below the `0.3 m` requirement. Fresh 64-episode evaluations of the saved best-return checkpoints were run with `scripts/eval_ars.py`: in-place achieved 54.7% jump success (mean height gain `0.140 m`); forward achieved 26.6% distance success (mean `d_bar=0.138 m`). Neither motion reached the 80% gate. See TensorBoard runs `logs/ars/go2_in_place/20261103_022245` and `logs/ars/go2_forward/20261103_032818`.
- **Gate 4 — not passed:** one PPO imitation iteration at 64 environments completed using the generated ARS checkpoint; one PPO refine iteration at 64 environments completed and logged separate dense terms. The 150-iteration imitation threshold, ARS-vs-PPO 256-episode comparison, and requested performance metrics were not run.
- **Gate 5 — not started:** per the plan's gate rule, soft springs and randomized retraining are gated on Gate 4. The factory rejects `soft=True` and `randomize=True` rather than silently ignoring them. The mjlab version does expose per-environment mass, CoM, inertia, and joint-field randomization APIs, so this is a gate-order deferral rather than an API limitation.
- **Gate 6 — not started:** back-flip/pronking are optional and gated on earlier stages.

## Deviations

1. **Standing controller gains and home height:** copied velocity-task gains caused the robot to collapse on initial touchdown in the standing test. Per the plan's fallback, gains were increased to hip `40/2`, thigh `50/3`, calf `80/4` (kp/kd). A 10-second settling rollout measured `h_stand≈0.224 m`; the jump env now resets at that root height and uses `h_max=0.524 m`, `stage2_h_min=0.194 m`. This changes the entity's configured `z=0.32 m` only for the new acrobatics task; existing velocity tasks are untouched.

2. **Gate 2 command form:** no GUI viewer was launched in the headless run. `--agent zero` and `--agent random` behavior was checked with direct manager-environment rollouts at one environment; finite observations/rewards were verified.

3. **Training depth:** ARS and PPO were tested with small smoke runs (4 and 64 envs, respectively) to verify code paths. The long learning gates and 256-episode evaluation are not claimed as passed. This is why Gates 3–5 remain incomplete and the optional soft/DR branch was not begun.

4. **Convenience wrapper:** the optional `train_two_stage.py` wrapper was omitted; the individual ARS and standard `train.py` commands are the source of truth.

5. **Test tooling:** the checkout's Python 3.13 virtualenv had no pytest. Pytest 6.2.5 was installed from the local uv cache; its plugin autoload conflicted with the installed AnyIO plugin, so tests run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. No repository dependency file was changed.

6. **Generated logs:** smoke training wrote ignored checkpoints and TensorBoard/W&B logs under `logs/`; these are run artifacts, not source changes.

7. **Sparse height normalization:** the plan specifies `h_n = h_bar / h_max`, but that makes a standing episode earn height reward because the robot's standing height is nonzero, contrary to the Gate 2 standing-timeout sanity check. The implementation uses `clip((h_bar - h_stand)/(h_max - h_stand), 0, 1)` for terminal jump height and the Stage 2 forward/in-place bonus. This yields zero for an ideally settled stand. A real short timeout measured reward `0.162` due to reset settling overshoot, so the smoke test allows a bounded normalized value. This alters literal paper equations and should be revisited if matching those equations takes priority over the stated sanity check.

8. **Stage 1 CLI env count:** the training script's production default is 512 environments. Per the user instruction, all executed tests used at most 64 environments (ARS smoke: 4; PPO smoke: 64); no test exceeded 500. The usage doc explicitly overrides the default.

9. **Sparse crash reward tuned after learning evidence:** the initial `failure_penalty=0.08` and `crash_scale=0.096` made early crashes score positively after moderate height gains, contradicting the plan's sanity rule that crashes should earn negative return. A 500-iteration in-place run using 64 env was stopped at iteration 181 after latest evaluation return `0.612`, height gain `0.057 m`, early-termination rate `100%`, and success rate `0%`. Defaults are now `failure_penalty=1.2`, `crash_scale=1.0`; at maximum normalized height they make both motions' sparse early-termination reward negative. A kernel regression test checks this. The run was intentionally stopped to avoid spending the remaining iterations optimizing a reward with the wrong sign. Its saved checkpoint is only evidence of the pre-tuning behavior and is not a final policy.

10. **Gate 3 runtime result and resource cap:** each full run used 32 directions (64 env) rather than the paper-inspired 256 (512 env), because the user capped testing at 500 env. In-place best in-run flight success was 46.9%, while forward's best mean displacement was `0.224 m`; both missed target. A fresh 64-episode evaluation of each saved best-return checkpoint confirmed in-place jump success `54.7%`, forward distance success `26.6%` at `d_bar≥0.3 m`. Early illegal contacts remain the common failure mode. The results are noisy and suggest that the small direction batch, reward weighting, and landing robustness need further tuning. Per the gate rule, long Stage 2 learning remains deferred for both branches until Stage 1 passes. The reusable `scripts/eval_ars.py` reports per-episode aggregate metrics and rejects requests over 500 environments.

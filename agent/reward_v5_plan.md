# Reward V5: coding-agent plan

Status: design only. Do not treat this document as an implemented reward or as
evidence that V5 improves gait. Implement and measure it in the MASQ project.

## Objective and evidence

Task: keep the V4 incentive to obey velocity commands while getting the four
leg agents to produce a stable, coordinated gait. Deliver `agent/reward_v5.yaml`,
the corresponding runnable reward/config changes, tests, an experiment, and a
report with pros, cons, and measured limitations, as required by
`agent/instructions.md`.

Read `agent/reward_v4.yaml`, `agent/reward_v4_report.md`, the V4 plots under
`logs/rsl_rl/go2_masq/V4/plots/`, and the active implementation before editing.
The V4 plot shows mean episode length falling from about 12 to 6–7 while mean
reward improves; the long-stance term is flat at zero. These are observations,
not proof of a single cause. A zero-valued penalty does not reward lifting a
foot. The likely loopholes to test are V4's speed-only `alive_moving`, the
touchdown-only air-time reward, lack of a gait/contact schedule reward, and
the absence of a fall cost in active MASQ rewards. The V4 command sampler also
includes pronk (all four legs swing together) as a valid gait. Do not judge a
pronk-commanded flight as a trot failure.

## Proposed V5 reward structure

Use one shared body-level reward, broadcast to the four PPO value heads as
the current algorithm does. A joint contact-pattern term gives each leg an
incentive to coordinate with the other three; do not give each leg a separate
independent touchdown bonus. Weights below are *initial rates*, not tuned
results. `RewardManager` multiplies each raw term by `weight * env.step_dt`;
verify the actual `step_dt` before interpreting magnitudes. Keep the V4
negative tails for planar and yaw tracking.

| Term (YAML id → active config key) | Initial weight | Raw value / rule | Reason |
|---|---:|---|---|
| `track_lin_vel_xy` → `track_linear_velocity` | +1.5 | V4 planar error kernel and negative tail | Maintain forward progress and command accuracy. |
| `track_ang_vel_z` → `track_angular_velocity` | +0.75 | V4 yaw error kernel and negative tail | Preserve turning and heading control. |
| `alive_moving` → `alive_moving` | +0.2 | V4 `2*tanh(speed/0.1)-1`, only for a planar movement command **and** at least one supporting foot | Keep movement pressure without paying for unsupported flight. |
| `gait_contact_match` → same | +1.0 | `active * mean_i[(2*c_i-1)*(2*d_i-1)]` | Bounded, signed whole-body match to the commanded gait; `c_i` is contact (0/1), `d_i` desired stance (0/1). |
| `support_deficit` → same | -0.5 | `active * max(0, 2-n_contact)/2` on trot/pace; use a gait-specific minimum for bound | Favors planted supporting feet without demanding all four stay planted. |
| `all_air_nonpronk` → same | -3.0 | `active * 1[n_contact=0]` for trot/pace/bound | Directly discourages the observed four-leg launch. Exclude commanded pronk. |
| `fall_termination` → same | -50.0 | `mjlab.envs.mdp.is_terminated * clamp((iteration - 200)/600, 0, 1)`, one step on non-timeout termination | Preserves early exploration; from iteration 800, a 20 ms step costs about -1 per fall. |
| `body_orientation`, `lin_vel_z`, `ang_vel_xy`, `base_height`, `undesired_contact` → existing config aliases | V4 weights (-1, -2, -0.05, -10, -1) | Preserve V4 definitions | Keep uprightness, base-height, and illegal-contact constraints. |
| `feet_slip` → `feet_slip` | -0.1 | Existing contact-weighted foot velocity penalty | Discourage dragging planted feet. |
| `action_rate` → `action_rate` | -0.01 | Existing first action difference squared | Keep V4 jitter control. |
| `action_smooth_2` → same | -0.005 | `mjlab.envs.mdp.action_acc_l2` (second action difference squared) | Add jerk control without duplicating `action_rate`. |
| `dof_acc` → same | -2.5e-7 | `mjlab.envs.mdp.joint_acc_l2` | Penalize rapid joint acceleration; validate magnitude from rollouts. |
| `feet_air_time`, `feet_stance_time` → same | 0.0 initially | Retain as logged ablations; no active incentive | V4 air-time can pay on collective landing; the 1.25 s long-stance penalty never activated and lowering its threshold could worsen foot lifting. |

Definitions: `active = (||v_cmd_xy|| > 0.1 m/s) OR (|yaw_cmd| > 0.2 rad/s)`.
For a standing command, gait and support terms are neutral; retain yaw/linear
tracking and stability. Use sensor contact in the explicit foot order
`(FR, FL, RR, RL)`, which is also the order of `GAIT_OFFSETS`. Derive desired
stance with `d_i = ((phase + phase_offsets_i) % 1 < duty_factor)` using the
*existing* `GaitCommand.phase`, which already advances by `dt*cadence`.
Choose and document a duty factor around 0.55–0.60, and optionally smooth
the contact-match transition near phase boundaries to avoid discontinuous
gradients/noisy contacts. Confirm sensor indexing at runtime.

Start the V5 comparison with commanded trot only (`gait_id=1` in the current
offset table). This directly tests inter-leg coordination without pronk
flight being an intended behavior. Preserve the four-gait command interface.
After stable trot passes the gates below, expand to pace and bound, then test
pronk separately with a bounded flight phase and gait-specific support rule.
Do not enable a blanket all-air penalty for pronk. If the project must train
all four gaits concurrently from the start, make every support/flight term
conditional on `gait_id` and report metrics per gait; this is a riskier
ablation, not the first V5 run.

## Before training: cumulative reward audit

Build a small, deterministic scenario table in the report or a test helper.
Calculate *all active weighted terms together*, then multiply by `step_dt`.
At minimum compare (1) matched trot with two supporting feet, (2) stationary
under a 0.5 m/s command, (3) all four feet airborne while apparently tracking,
(4) all four feet planted under a movement command, (5) zero-command standing,
(6) valid gait transition, and (7) a non-timeout fall. Use the actual logged
V4 distributions for action acceleration, joint acceleration, and foot slip
to set final regularizer weights. The proposed nominal ordering should be:
stable commanded trot > standing under motion command > unsupported flight,
and non-timeout fall should be worse than continuing safely. The idealized
tracked-trot positive rate is 1.5 + 0.75 + 0.2 + 1.0 = 3.45 before
stability/regularization; this is an algebraic check, not a measured return.
Report each term's min/max and typical rate so a large penalty cannot silently
dominate tracking. The `Episode_Reward/*` logger divides by maximum episode
length; compare raw per-step sums as well as the existing plots.

## Coding workflow (execute later, in order)

1. **Baseline and source mapping.** Inspect `src/tasks/masq/mdp/rewards.py`,
   `src/tasks/masq/masq_env_cfg.py`, `src/tasks/masq/mdp/commands.py`,
   `src/tasks/masq/mdp/observations.py`, and stock `velocity/mdp/rewards.py`.
   Check reward aliases, contact tensor shapes/order, term time scaling,
   termination flags, and `step_dt`. Record V4 values for mean/median episode
   length, fall/illegal-contact rate, tracking error, and all-air frequency.
2. **Fix signal consistency.** `GaitCommand.phase` already integrates cadence,
   but `temporal_director()` multiplies it by cadence again. Align actor and
   critic phase observations with the same `phase + offset` used by the gait
   reward. Review the curriculum: its current `reward_ema > 0.8` on total
   reward is not a trustworthy gate after reward changes. For the initial
   trot run, hold gait assignment fixed; later use per-gait tracking/contact
   success as the promotion condition. Do not modify the PPO architecture.
3. **Write `agent/reward_v5.yaml` as the authoritative table.** Include each
   active/disabled term, exact formula, weight, gate, unit, and YAML-to-config
   alias. Make `MASQ_REWARD_SCALES` and `cfg.rewards` agree with it. Keep one
   first-difference action term; add the second-difference term once.
4. **Implement new terms.** Add contact-match, support-deficit,
   nonpronk-all-air, and failure cost in `src/tasks/masq/mdp/rewards.py` and
   register them in `masq_env_cfg.py`. Use vectorized `[num_envs, 4]` contact
   tensors and explicit gait masks. Ensure reset, zero commands, phase wrap,
   and sensor false contacts have defined behavior. Log raw diagnostics:
   contact-count histogram, four-feet-airborne fraction by gait, gait match,
   support deficit, speed/command error, and fall reason.
5. **Tests before training.** Unit-test each formula on hand-built batches:
   perfect trot/pace/bound/pronk, phase wrap, zero command, all feet up,
   all feet down, one-foot support, and touchdown/reset. Check finite outputs,
   signs, shapes, and `RewardManager` time scaling. Check the configured
   weights against the YAML, and test that the unchanged V4 tracking terms
   can still become negative. Update existing V2/V3 tests that reference
   removed `survival_bonus`/`stand_still_penalty` or expect disabled rewards
   to be active; keep historical YAML files intact.
6. **Smoke then controlled experiment.** Run formatting/type checks used by
   this repo, `pytest`, and a short simulator run with a small environment
   count. Confirm every term is logged and no NaN/shape errors occur. Compare
   V4 and V5 using the same command distribution, seed set (at least three
   seeds), training steps, terrain, and checkpoint selection. Because the
   first V5 run is trot-only, use a trot-only V4 comparison run as well;
   the existing four-gait V4 plots are diagnostic context, not a matched
   training baseline. Save separate run directories and videos. Keep the
   shared RTX 3070 excluded: set
   `CUDA_VISIBLE_DEVICES=''` and explicitly select CPU (`gpu_ids=None`), or
   choose only physical GTX 1080 indices `0,2` after checking the launcher's
   index mapping. `scripts/train.py` defaults `gpu_ids=[0]`; inspect its
   effective device print before a long run. Never target GPU 1. For every
   smoke or full run set `env.scene.num_envs <= 4096`; do not infer a larger
   value from a task default. Keep W&B disabled (`cfg.logger = "tensorboard"`)
   so the run is local and TensorBoard remains the only scalar source.
7. **Accept or tune from evidence.** Require lower all-air frequency on
   nonpronk commands, no regression in mean/median episode length or fall
   rate, and no material tracking-error regression. Inspect videos for foot
   order and sustained grounded locomotion. If a gate fails, adjust one term
   or command-sampling choice at a time and rerun the scenario audit. Keep
   V4 as an untouched baseline; name each V5 ablation/config explicitly.
8. **Final handoff.** Write `agent/reward_v5_report.md` with exact YAML/code
   changes, scenario-table math, actual test/training commands and results,
   plots for current configured reward components/episode length/falls/contact
   pattern, plus recorded videos for each tested gait. Do not decide whether a
   gait is acceptable from reward plots alone; inspect the videos for foot
   order, support, and sustained locomotion.
   pros and cons, known limitations, and a diff review. Do not claim better
   gait from reward curves alone.

## Expected advantages and tradeoffs to report

- Advantages to test: coordinated foot timing, explicit ground support,
  reduced incentive for synchronized four-leg launch, retained command
  tracking, and smoother actions.
- Risks: excessive gait shaping can force a stiff trot or reduce speed;
  stronger fall penalties can deter exploration; jerk penalties can prevent
  quick recovery; contact sensor errors and phase timing can corrupt the
  shaping signal; pronk and other valid gaits need different flight rules.
- Evidence limit: reward design predicts incentives, not learned behavior.
  Multi-seed training and video/physical metrics decide whether V5 succeeds.

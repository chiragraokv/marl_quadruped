# Task1: Experiment MARL on unitree go2

## Goal
evaluate Multi agent reinforcement learning methods on single quadruped for smooth gait learning.

# source: http://arxiv.org/abs/2408.13759
Reward config has been saved at : /agent/reward.yaml

# MASQ → `unitree_rl_mjlab` Implementation Plan

Target paper: Liu et al., *"MASQ: Multi-Agent Reinforcement Learning for Single Quadruped Robot Locomotion"*, arXiv:2408.13759.
Target repo: `unitreerobotics/unitree_rl_mjlab` (depends on `mjlab==1.2.0`, `rsl-rl-lib==5.0.1`). All paths/APIs below were verified by cloning the repo and the two dependency wheels — not guessed.

New task to build: **`Unitree-Go2-MASQ-Flat`**, sitting next to the existing `Unitree-Go2-Flat` task.

---

## 0. TL;DR of the hard part

This is *not* a reward-tuning task. The paper's core contribution is architectural: each leg is a PPO agent with **its own log-prob/ratio**, parameters are **shared** across the 4 agents, and the critic is **global** with **4 value heads**. Stock `rsl_rl` (vendored by `mjlab`) hard-codes:
- critic output dim `1` (`rsl_rl/algorithms/ppo.py::PPO.construct_algorithm`)
- `RolloutStorage.values / .actions_log_prob / .returns / .advantages` allocated with last-dim `1` (`rsl_rl/storage/rollout_storage.py`)
- one joint action distribution over all 12 action dims (one scalar log-prob), not 4 independent 3-dim ones

None of this needs upstream patches. `rsl_rl.utils.resolve_callable` resolves `class_name` strings as **dotted/colon import paths**, not just a closed registry (see §5.2). So the whole MAPPO extension is new code in `src/tasks/masq/rl/`, referenced from config by string, e.g. `class_name="src.tasks.masq.rl.actor_model:MASQActorModel"`.

---

## 1. Verified repo facts (do not re-derive these — use them)

```
unitree_rl_mjlab/
  src/tasks/velocity/
    velocity_env_cfg.py          # make_velocity_env_cfg() factory — COPY this as the base
    mdp/{observations,rewards,terminations,curriculums,velocity_command}.py
    config/go2/{env_cfgs.py, rl_cfg.py, __init__.py}
    rl/runner.py                 # VelocityOnPolicyRunner(MjlabOnPolicyRunner) — onnx export + wandb save
  src/assets/robots/unitree_go2/go2_constants.py   # get_go2_robot_cfg()
```
- Task registration: `mjlab.tasks.registry.register_mjlab_task(task_id, env_cfg, play_env_cfg, rl_cfg, runner_cls)`.
- `rl_cfg.py` returns `mjlab.rl.RslRlOnPolicyRunnerCfg(actor=RslRlModelCfg(...), critic=RslRlModelCfg(...), algorithm=RslRlPpoAlgorithmCfg(...), ...)`.
- `RslRlModelCfg.class_name` defaults to `"MLPModel"`; `rsl_rl.models.RNNModel` also exists and already supports `rnn_type="gru"` — **use it directly as the base class for the actor**, don't reimplement a GRU.
- `PPO.construct_algorithm` (static method in `rsl_rl/algorithms/ppo.py`):
  ```python
  actor  = actor_class(obs, obs_groups, "actor", env.num_actions, **cfg["actor"])
  critic = critic_class(obs, obs_groups, "critic", 1, **cfg["critic"])   # <-- hard-coded 1
  storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
  ```
- `RolloutStorage.__init__` allocates `self.values/.actions_log_prob/.returns/.advantages = torch.zeros(T, num_envs, 1, ...)` — fixed, and `add_transition` does `self.actions_log_prob[step].copy_(transition.actions_log_prob.view(-1,1))` — a hard crash/corruption risk for any non-`(N,1)` log-prob.
- `Distribution.log_prob/.entropy/.kl_divergence` are defined as "summed over the last dimension" — this is exactly what makes the fold-agents-into-batch trick work (§5.3).
- Existing reusable `mjlab.envs.mdp.rewards` functions: `joint_torques_l2, joint_vel_l2, joint_acc_l2, action_rate_l2, action_acc_l2, joint_pos_limits, posture, flat_orientation_l2, is_terminated`.
- Existing reusable `mjlab.envs.mdp.dr` (domain randomization) functions: `body_mass, body_com_offset, pd_gains (motor strength), joint_default_pos (calibration offset), geom_friction, encoder_bias, effort_limits`. No built-in `restitution` or `gravity` randomizer — flagged as new in §7.
- Existing reusable `mjlab.envs.mdp.events`: `push_by_setting_velocity, apply_external_force_torque, apply_body_impulse` (class, `mode="interval"`) — exactly fits the paper's periodic gravity-impulse disturbance.
- `src/tasks/velocity/mdp/rewards.py` already has `feet_slip`, `feet_gait` (Van-der-Pol-style desired-contact-schedule reward), `feet_clearance`, `soft_landing`, `body_angular_velocity_penalty` — reusable templates for several MASQ terms (§4).
- `src/tasks/velocity/mdp/velocity_command.py::UniformVelocityCommand(CommandTerm)` is the template for any new stateful per-env command term (temporal director, gait id) — see §4.3.

---

## 2. Directory layout to add

```
src/tasks/masq/
  __init__.py                  # (empty, package marker)
  masq_env_cfg.py               # make_masq_env_cfg() — copy of make_velocity_env_cfg(), edited per §3-7
  mdp/
    __init__.py                 # from mjlab.envs.mdp import *; from .{observations,rewards,commands,curriculums,events} import *
    observations.py             # per-leg obs builder, temporal director, agent-id one-hot
    rewards.py                  # the 7 genuinely-new reward terms (§4)
    commands.py                 # GaitCommand(CommandTerm): velocity+height+gait-id+phase offsets
    curriculums.py               # multi-gait curriculum (promote gait mix on reward threshold)
    events.py                   # gravity-impulse + restitution randomization (missing from mjlab.dr)
  config/go2/
    __init__.py                 # register_mjlab_task("Unitree-Go2-MASQ-Flat", ...)
    env_cfgs.py                 # unitree_go2_masq_flat_env_cfg(play=False) — Go2-specific overrides
    rl_cfg.py                   # RslRlOnPolicyRunnerCfg pointing class_name at masq.rl classes
  rl/
    __init__.py
    actor_model.py               # MASQActorModel(RNNModel) — the core novel class (§5.3-5.5)
    critic_model.py              # thin subclass note: usually NOT needed, see §5.6
    rollout_storage.py           # MASQRolloutStorage(RolloutStorage) — N_AGENTS-wide buffers (§5.4)
    algorithm.py                 # MASQPPO(PPO) — overrides construct_algorithm only (§5.7)
    runner.py                    # MASQOnPolicyRunner(MjlabOnPolicyRunner) — mirrors VelocityOnPolicyRunner
```

---

## 3. Observations

Robot: Unitree Go2, legs ordered `FR, FL, RR, RL` (matches `go2/env_cfgs.py::foot_names`), `N_AGENTS = 4`, 3 motors/leg (`hip, thigh, calf`).

### 3.1 Actor group (per agent, 35-dim; concatenated to 140-dim before the fold-into-batch reshape)

| Field | Dim | Source | Notes |
|---|---|---|---|
| `q_t` (motor pos) | 3 | `mdp.joint_pos_rel`, sliced to this leg's 3 joints | |
| `q̇_t` (motor vel) | 3 | `mdp.joint_vel_rel`, sliced to this leg | |
| `a_{t-1}` | 3 | `mdp.last_action`, sliced to this leg | |
| `a_t` | 3 | **ambiguity** — paper lists both `a_{t-1}` and `a_t` as inputs to a model that *produces* `a_t`; read as the *previous* 2-step action history (`a_{t-1}, a_{t-2}`) for a GRU warm-start, not a true lookahead. Implement as `a_{t-2}` sliced to this leg. Flag in code comments as a paper-ambiguity resolution. | |
| `d_t` (temporal director) | 1 | new, `mdp.temporal_director` (§3.3) | per-leg phase signal, **not shared** across legs |
| `g_t` (projected gravity) | 3 | `mdp.projected_gravity` | shared across all 4 agents |
| `v_cmd` | 15 | `mdp.generated_commands("gait")` | shared; see §4.1 for the 15-dim breakdown |
| `v_b` (body speed) | 3 | `mdp.builtin_sensor("robot/imu_lin_vel")` | shared |
| `e_t` (agent id) | 1 | new, one-hot-as-scalar index `i/3` or true one-hot — **use a true 4-way one-hot and drop it from the 35 count, OR treat `e_t` as the scalar leg index normalized to `[0,1]`**; paper's dim-count (1) only fits a scalar encoding, so use `leg_index / 3.0`. | |

Total: 3+3+3+3+1+3+15+3+1 = **35**, ×4 legs = **140**, matches paper exactly.

### 3.2 Critic group (global, 73-dim)

| Field | Dim | Source |
|---|---|---|
| `q_t` (all motors) | 12 | `mdp.joint_pos_rel` |
| `q̇_t` | 12 | `mdp.joint_vel_rel` |
| `a_{t-1}` | 12 | `mdp.last_action` |
| `a_t` (→ `a_{t-2}`, same resolution as §3.1) | 12 | |
| `d_t` (all 4 directors) | 4 | `mdp.temporal_director_all` |
| `g_t` | 3 | `mdp.projected_gravity` |
| `v_cmd` | 15 | `mdp.generated_commands("gait")` |
| `v_b` | 3 | `mdp.builtin_sensor("robot/imu_lin_vel")` |

Total: 12+12+12+12+4+3+15+3 = **73**, matches paper.

### 3.3 Temporal director (new function, `masq/mdp/observations.py`)

```python
def temporal_director(env, leg_index: int, command_name: str = "gait") -> torch.Tensor:
    gait_cmd = env.command_manager.get_term(command_name)  # GaitCommand
    k = gait_cmd.cadence              # (num_envs,) scaling factor of the gait cycle
    phase = gait_cmd.phase            # (num_envs,) running phase t, advanced each env.step
    delta_i = gait_cmd.phase_offsets[:, leg_index]  # (num_envs,) Δ_i, set per sampled gait
    return torch.sin(2 * torch.pi * (k * phase + delta_i)).unsqueeze(-1)
```
`temporal_director_all` is the same, stacked over `leg_index in range(4)` → `(num_envs, 4)`, used by the critic group.

Build 4 separate `ObservationTermCfg` actor terms (`leg0_obs ... leg3_obs`, each assembling the full 35-dim vector for that leg via `params={"leg_index": i}`), concatenated in leg order — **do not** build one 140-dim term and reshape downstream; keeping the manager-level split explicit makes it trivial to verify per-leg slices independently.

---

## 4. Commands: the gait system (`masq/mdp/commands.py`)

One new `CommandTerm`, `GaitCommand`, replacing `UniformVelocityCommandCfg` as the single command (`"gait"` instead of `"twist"`). Internally wraps a `UniformVelocityCommand`-style velocity/height sampler (copy the resample logic from `velocity_command.py`) **plus**:

| Field | Shape | Role |
|---|---|---|
| `vel_cmd` | (N,3) | lin x, lin y, ang z — reuse `UniformVelocityCommandCfg.Ranges` |
| `height_cmd` | (N,1) | commanded base/jump height `h_j`, used by the Jump reward |
| `gait_id` | (N,) int in `{0,1,2,3}` | which of {pace, trot, bound, pronk} this env is currently training |
| `cadence` k | (N,1) | gait frequency scale, resample per episode in a reasonable range, e.g. `(1.0, 2.5)` Hz-equivalent — **not specified by the paper, tune empirically** |
| `phase` | (N,1) | monotonically increasing each `env.step`, `phase += step_dt`, wrapped mod 1 |
| `phase_offsets` | (N,4) | `Δ_i` per leg, looked up from `gait_id` via the table below |

Total command vector exposed to `generated_commands("gait")` = `vel_cmd(3) + height_cmd(1) + gait_id_onehot(4) + cadence(1) + phase(1)` ... **pad/trim to 15 dims to match the paper's `v_cmd ∈ R^15`**; exact composition isn't specified — this is the single biggest "make it 15" fitting exercise. Suggested composition (15 = 3+1+4+1+1+... needs 5 more): add `phase_offsets` for this env's active gait only is already `(4,)`-wide from `phase_offsets`? Recount: `3(vel) + 1(height) + 4(gait one-hot) + 1(cadence) + 1(phase) + 4(phase_offsets) = 14`. Add 1 reserved/zero-padding dim, or a `rel_standing` flag (1 bool), to reach 15. **Flag this composition explicitly as an assumption to validate**, not a verified fact.

### 4.1 Gait phase-offset table (ASSUMPTION — paper gives no numbers; using the standard convention from Margolis & Agrawal, *"Walk These Ways"*, and Rudin et al. legged_gym)

Leg order `(FR, FL, RR, RL)`:

| Gait | Δ_FR | Δ_FL | Δ_RR | Δ_RL | Pattern |
|---|---|---|---|---|---|
| Pace | 0.0 | 0.5 | 0.0 | 0.5 | same-side pairs in phase |
| Trot | 0.0 | 0.5 | 0.5 | 0.0 | diagonal pairs in phase |
| Bound | 0.0 | 0.0 | 0.5 | 0.5 | front/rear pairs in phase |
| Pronk | 0.0 | 0.0 | 0.0 | 0.0 | all four in phase |

---

## 5. Reward terms → implementation mapping

| # | Term | Scale | Status | Implementation |
|---|---|---|---|---|
| 1 | Tracking Linear Velocity | 1.0 | Adapt | copy `velocity.mdp.track_linear_velocity`, point `command_name="gait"`, read `.vel_cmd` |
| 2 | Tracking Angular Velocity | 0.5 | Adapt | same, `track_angular_velocity` |
| 3 | Linear Velocity Z | −2e-2 | New (trivial) | `(base_lin_vel_b[:,2])**2` |
| 4 | Angular Velocity XY | −1e-3 | New (trivial) | `(base_ang_vel_b[:,:2]**2).sum(-1)` — paper's formula literally sums the full vector; **XY in the name suggests roll+pitch only, used here**, flag as assumption |
| 5 | Torques | −1e-5 | **Reuse as-is** | `mjlab.envs.mdp.rewards.joint_torques_l2` |
| 6 | DOF Velocity | −1e-4 | **Reuse as-is** | `mjlab.envs.mdp.rewards.joint_vel_l2` |
| 7 | DOF Acceleration | −2.5e-7 | **Reuse as-is** | `mjlab.envs.mdp.rewards.joint_acc_l2` (same constant already used in stock `Unitree-Go2-Flat`!) |
| 8 | Collision | −5.0 | New | copy the `nonfoot_ground_touch` `ContactSensorCfg` pattern from `go2/env_cfgs.py`, indicator `force_norm > 0.1` |
| 9 | Action Rate | −1e-2 | **Reuse as-is** | `mjlab.envs.mdp.rewards.action_rate_l2` |
| 10 | Jump | 10.0 | New | `-(base_height - height_cmd)**2`, read `height_cmd` from `GaitCommand` |
| 11 | Feet Slip | −4e-2 | **Reuse, retune weight** | `velocity.mdp.feet_slip` (already exists in repo) |
| 12 | Action Smoothness 1 | −0.1 | **Reuse (duplicate entry)** | identical formula to term 9; add a second `RewardTermCfg` entry pointing at `action_rate_l2` with weight `-0.1` — paper lists them as two separate terms with different weights despite being the same formula |
| 13 | Action Smoothness 2 | −0.1 | **Reuse as-is** | `mjlab.envs.mdp.rewards.action_acc_l2` (second-order action difference — matches exactly) |
| 14 | Feet Impact Velocity | 0.0 | Stub, inactive | implement for completeness but `weight=0.0` (paper's own table has scale `-0.0`, i.e. disabled) |
| 15 | Raibert Heuristic | −10.0 | New, non-trivial | desired-vs-actual foot placement error under the Raibert heuristic; use Rudin et al. *legged_gym* / Margolis *Walk These Ways* public implementations as the reference formula (not vendored in this repo) |
| 16 | Tracking Contacts Shaped Velocity | 4.0 | Adapt | `velocity.mdp.feet_gait` already implements a Van-der-Pol desired-contact-schedule reward in this exact style — fork it into `tracking_contacts_shaped_velocity`, replacing the contact-force kernel with the foot-velocity kernel `1 - exp(-||v_f||² / σ_gv)` gated by the desired-contact indicator `c_d` from `GaitCommand`'s phase |

Missing constants (not given by the paper) — use these defaults, mark as tunable:
- `σ_t = σ_yaw = σ_gv = 0.25` (standard value in legged_gym-style reward kernels)
- `c_f, c_s, c_d` — binary foot-contact indicators from the existing `feet_ground_contact` `ContactSensorCfg`; `c_d` additionally gated by whether the gait schedule (`GaitCommand.phase`, `phase_offsets`) says this foot *should* be in stance right now

---

## 6. Domain randomization & disturbances (`masq/mdp/events.py` + reused `mjlab.envs.mdp.events`/`dr`)

| Paper requirement | mjlab primitive | Status |
|---|---|---|
| Body mass | `dr.body_mass` | Reuse |
| Motor strength | `dr.pd_gains` or `dr.effort_limits` | Reuse |
| Joint position calibration | `dr.joint_default_pos` | Reuse |
| Ground friction | `dr.geom_friction` (already wired for Go2 feet) | Reuse |
| Restitution | none in `mjlab.envs.mdp.dr` | **New**: add `restitution` randomizer (MuJoCo geom `solref`/`priority` or `condim`-level coefficient) in `masq/mdp/events.py`, or confirm via `mjlab` docs whether a newer dr submodule covers it before writing one from scratch |
| Orientation (at reset) | `mdp.reset_root_state_uniform` (`yaw` range) — already present | Reuse |
| Gravity magnitude/direction, impulse every 8s for 0.99s | `mjlab.envs.mdp.events.apply_body_impulse` (class, `mode="interval"`, `interval_range_s=(8.0, 8.0)`) — check its params support a duration field; if not, implement as `apply_external_force_torque` toggled on/off across a 0.99s window tracked in a small per-env timer in a new `EventTermCfg` | Adapt |
| Timestep randomization every 6s | mirror the above interval-event pattern against `cfg.sim.mujoco.timestep` jitter, or treat as decimation jitter — **flag as low-priority / optional**, not critical to reproducing the locomotion result | New, optional |
| General randomization refresh every 4s | reuse `mode="interval"` on whichever of the above should refresh fastest | Reuse pattern |

---

## 7. Curriculum: multi-gait (`masq/mdp/curriculums.py`)

`CurriculumTermCfg` running each `env.step` (or on reset):
1. Track an exponential moving average of each env's per-gait reward.
2. For each env currently assigned a given `gait_id`, if its EMA reward crosses a threshold, re-sample `gait_id` uniformly over `{pace, trot, bound, pronk}` (paper: *"the quadruped robot learns these four gaits simultaneously, and the progress is updated based on evaluating whether the gaits' rewards meet specific thresholds"* — read as **all 4 gaits trained concurrently across the env batch**, with per-env gait reassignment gated by a reward threshold rather than a classic single global curriculum stage). Implement as a `GaitCommand`-coupled curriculum that writes into `gait_cmd.gait_id`/`phase_offsets` directly.
3. Reward threshold value: **not specified** — start with "mean per-step tracking reward > 0.8 × max achievable" and tune.

---

## 8. MAPPO core (`src/tasks/masq/rl/`)

### 8.1 Why nothing here is a stock-class drop-in
See §0/§1. Three blockers: critic output dim, storage buffer widths, single joint action distribution.

### 8.2 Extension mechanism
`rsl_rl.utils.resolve_callable(name)` resolves, in order: a direct callable, `"module.path:Attr"`, `"module.path.Attr"`, or a bare name searched inside `rsl_rl`. Point every `class_name` field at a fully-qualified path to the new classes below, e.g.:
```python
actor=RslRlModelCfg(class_name="src.tasks.masq.rl.actor_model:MASQActorModel", ...),
critic=RslRlModelCfg(class_name="rsl_rl.models:MLPModel", distribution_cfg=None, ...),  # see §8.6
algorithm=RslRlPpoAlgorithmCfg(class_name="src.tasks.masq.rl.algorithm:MASQPPO", ...),
```

### 8.3 Actor: `MASQActorModel(RNNModel)` — the fold/unfold trick

Core idea: fold the 4-agent dimension into the batch dimension for every call into the shared MLP+GRU+distribution stack, then unfold it back out **before** exposing anything through the model's public interface, so generic `PPO`/`RolloutStorage` code (which always assumes dim-0 = `num_envs × num_steps`) stays correct.

```python
class MASQActorModel(RNNModel):
  N_AGENTS = 4
  def __init__(self, obs, obs_groups, obs_set, output_dim, *a, rnn_type="gru", **kw):
      # output_dim passed in will be env.num_actions == 12; override internally to
      # build the distribution over 3 dims (one leg), called N_AGENTS times via folding.
      super().__init__(obs, obs_groups, obs_set, output_dim=3, *a, rnn_type=rnn_type, **kw)

  def get_latent(self, obs, masks=None, hidden_state=None):
      latent = MLPModel.get_latent(self, obs)          # (B, 140) normalized, obs groups already concatenated
      latent = latent.view(-1, self.N_AGENTS, 35).reshape(-1, 35)   # (B*4, 35)
      return self.rnn(latent, masks, self._fold_hidden(hidden_state)).squeeze(0)  # (B*4, rnn_hidden_dim)

  def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
      out = super().forward(obs, masks, hidden_state, stochastic_output)  # (B*4, 3)
      B = out.shape[0] // self.N_AGENTS
      return out.view(B, self.N_AGENTS, 3).reshape(B, 12)       # (B, 12) — fed straight to env.step()

  def get_output_log_prob(self, outputs):       # outputs: (B, 12)
      B = outputs.shape[0]
      flat = outputs.view(B, self.N_AGENTS, 3).reshape(B * self.N_AGENTS, 3)
      logp = self.distribution.log_prob(flat)              # (B*4,) — summed over the 3 leg-dims, per rsl_rl contract
      return logp.view(B, self.N_AGENTS)                    # (B, 4) — one log-prob per agent, dim-0 stays B

  @property
  def output_entropy(self):
      ent = self.distribution.entropy                        # (B*4,)
      B = ent.shape[0] // self.N_AGENTS
      return ent.view(B, self.N_AGENTS)

  @property
  def output_distribution_params(self):
      # each param tensor is (B*4, k); reshape every one to (B, 4, k) so KL-divergence
      # and any [:original_batch_size] slicing upstream in PPO.update() treats dim-0 as B.
      return tuple(p.view(p.shape[0] // self.N_AGENTS, self.N_AGENTS, *p.shape[1:])
                   for p in self.distribution.params)

  def get_kl_divergence(self, old_params, new_params):
      # unfold -> compute per (B*4,) row via base distribution -> fold back to (B,4) -> caller .mean()s it anyway
      ...

  def reset(self, dones=None, hidden_state=None):
      dones4 = dones.repeat_interleave(self.N_AGENTS) if dones is not None else None
      super().reset(dones4, hidden_state)

  def get_hidden_state(self):
      # shape (num_layers, B*4, hidden_dim) internally; MjlabOnPolicyRunner/RolloutStorage
      # treat this as an opaque blob keyed by env-time indexing on dim 1 — verify
      # recurrent_mini_batch_generator's `[last_was_done][first_traj:last_traj]` indexing
      # still lines up when B is folded ×4; if it does not, store/restore hidden state
      # reshaped as (num_layers, B, 4*hidden_dim) instead (flatten agents into the feature
      # dim rather than the batch dim) and reshape to (num_layers, B*4, hidden_dim) only
      # immediately before/after the RNN call. TEST THIS EXPLICITLY (§10, Milestone 3).
      return super().get_hidden_state()
```
**This is the one place in the whole plan most likely to hide a subtle bug** (hidden-state bookkeeping interacting with `recurrent_mini_batch_generator`'s done-flag-based trajectory splitting, which operates per-env, not per-env-per-leg). Treat §8.3's last method as the riskiest line of code in the project and write a standalone unit test for it before wiring into full training (§10, Milestone 3).

### 8.4 ONNX/JIT export must bake in the same fold/unfold
`MLPModel.as_onnx`/`as_jit` wrap only `obs_normalizer + mlp + deterministic_output`, assuming a direct `(B, obs_dim) → (B, output_dim)` map. Override `as_onnx`/`as_jit` in `MASQActorModel` to wrap a module that does the same `(B,140)→(B*4,35)→mlp/rnn→(B*4,3)→(B,12)` reshape internally, so the exported `policy.onnx` still takes the natural 140-dim observation and natural 12-dim action — the sim2real C++ pipeline in `deploy/robots/go2` is **not** aware of agents/legs and must not need to change.

### 8.5 `MASQRolloutStorage(RolloutStorage)`
Override `__init__` to allocate `values/actions_log_prob/returns/advantages` as `(T, num_envs, N_AGENTS)` instead of `(..., 1)`, and fix `add_transition`'s `.view(-1,1)` calls to `.view(-1, N_AGENTS)` (or just `.copy_()` without any view, since shapes should already match). `mini_batch_generator`/`recurrent_mini_batch_generator` need no changes — they only flatten/index generically.

### 8.6 Critic: probably a stock-class drop-in
If `MLPModel`/`RNNModel` genuinely treat `output_dim` as "how many numbers to produce" with `distribution_cfg=None` (deterministic path), then **no custom critic class is needed** — just instantiate it with `output_dim=4` via `MASQPPO.construct_algorithm`'s override (§8.7). Confirm this with a shape-only smoke test before writing a custom class; only subclass if something downstream assumes `output_dim==1` for the critic specifically.

### 8.7 `MASQPPO(PPO)`
Only override the static `construct_algorithm`:
```python
@staticmethod
def construct_algorithm(obs, env, cfg, device):
    ...  # same as PPO.construct_algorithm, except:
    critic = critic_class(obs, obs_groups, "critic", N_AGENTS, **cfg["critic"])   # 4, not 1
    storage = MASQRolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs,
                                  [env.num_actions], device, value_dim=N_AGENTS)
    alg = alg_class(actor, critic, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"])
    return alg
```
Everything else in `PPO.update()` — surrogate loss, clipped value loss, KL/adaptive-LR — works unmodified once shapes are `(B,4)`-consistent: `torch.squeeze` is a no-op on a width-4 last dim, and `.mean()` over `(B,4)` is equivalent (up to a constant factor of `1/N_AGENTS`, which just rescales the effective loss coefficients) to the paper's `Σᵢ 𝔼ₜ[...]` in Eq. 7. **Do not** try to literally reproduce the `Σᵢ` via a `.sum(dim=-1)` unless you also retune `value_loss_coef`/`entropy_coef`/`learning_rate` to compensate for the ×4 scale change — simplest is to leave `PPO.update()` untouched and accept the `.mean()` normalization.

### 8.8 Open modeling question: per-agent vs. shared advantage (flag prominently, pick one, document the choice)
The paper's Eq. 4 defines one global `Â_t`, but §IV-B says the critic's 4 outputs are "used to calculate the advantage function," implying 4 separate GAE traces. **Recommended default: Variant A (4 separate GAE traces, one per critic head, reward broadcast identically into all 4 via the `(num_envs,1)` reward tensor broadcasting against `(num_envs,4)` values in `compute_returns` — this already happens automatically, no code change needed beyond §8.5/§8.7).** Variant B (single scalar critic, shared advantage, only the actor is multi-agent) is a valid, simpler fallback if Variant A proves unstable — keep `critic` output dim configurable (`N_AGENTS` vs `1`) to allow an ablation switch.

---

## 9. Task registration & CLI

`src/tasks/masq/config/go2/__init__.py`:
```python
register_mjlab_task(
  task_id="Unitree-Go2-MASQ-Flat",
  env_cfg=unitree_go2_masq_flat_env_cfg(),
  play_env_cfg=unitree_go2_masq_flat_env_cfg(play=True),
  rl_cfg=unitree_go2_masq_ppo_runner_cfg(),
  runner_cls=MASQOnPolicyRunner,
)
```
Train: `python scripts/train.py Unitree-Go2-MASQ-Flat --env.scene.num-envs=4096`
Play: `python scripts/play.py Unitree-Go2-MASQ-Flat --checkpoint_file=logs/rsl_rl/go2_masq/.../model_xx.pt`
Sim2real: unchanged — reuse `deploy/robots/go2` as-is; only `policy.onnx` content changes (§8.4).

---

## 10. Milestones & acceptance checks (in order — do not skip ahead)

| # | Milestone | Acceptance check |
|---|---|---|
| 1 | Observations/commands/rewards wired, **stock single-agent PPO** (unmodified `Unitree-Go2-Flat`-style rl_cfg) | Env steps without NaN for 100 iterations at `num_envs=64`; per-term reward logged and sane-signed |
| 2 | `MASQActorModel`/critic shape smoke test, no training | Unit test (no env): random `(B,140)` actor-obs tensor in → `(B,12)` action out; `get_output_log_prob` returns `(B,4)`; critic `(B,73)`→`(B,4)` |
| 3 | GRU hidden-state fold/unfold | Unit test: run 2 sequential steps through `MASQActorModel` with a `dones` reset mid-sequence on a subset of envs; verify only those envs' per-leg hidden states reset, others persist, by inspecting hidden-state tensor diffs directly (do not rely on downstream training signal to catch this) |
| 4 | `MASQRolloutStorage` + `MASQPPO.construct_algorithm` | `num_envs=16`, `num_steps_per_env=4`, run one full `act → step → process_env_step → compute_returns → update()` cycle; assert no shape errors, `loss_dict["value"]`/`["surrogate"]` finite |
| 5 | Full training, flat terrain, single gait (trot only, curriculum disabled) | Converges to forward-walking trot within the same order-of-magnitude iterations as `Unitree-Go2-Flat` baseline |
| 6 | Multi-gait curriculum enabled | All 4 gaits individually achieve nonzero tracking reward; `play.py` visibly produces distinct gaits when `gait_id` is forced per-env |
| 7 | ONNX export parity | Export `policy.onnx`, feed it the same `(1,140)` dummy input outside training, confirm `(1,12)` output and that it matches `MASQActorModel.forward()` numerically (deterministic path) to float tolerance |
| 8 | Sim2real smoke test | `simulate/build/unitree_mujoco` + `deploy/robots/go2/build/g2_ctrl --network=lo` runs with the exported policy without crashing; robot stands/walks |
| 9 | Paper-claim check | Reward curves: MASQ reaches a given tracking-reward threshold in fewer iterations than the stock single-agent `Unitree-Go2-Flat` PPO baseline, mean of ≥3 seeds (paper reports mean of 5 runs) |
| 10 | Robustness test | Reproduce §V-C2: continuous bounding-gait jumps, apply a push mid-cycle, confirm recovery to steady contact pattern within one gait cycle, by logging foot contact-force traces as the paper does in Fig. 4 |

---

## 11. Open questions / assumptions log (surface these to the user before or during implementation — do not silently guess and bury it)

1. `a_t` in the 35-dim actor obs (§3.1) — resolved as `a_{t-2}` (2-step action history), not a true lookahead.
2. `e_t` agent-id encoding — resolved as scalar `leg_index/3.0`, not a 4-wide one-hot, to keep the dim budget at 35.
3. 15-dim `v_cmd` composition (§4) — only 14 dims accounted for from known fields; 1 dim unexplained.
4. Gait phase offsets `Δ_i` (§4.1) — paper gives none; using standard quadruped-gait literature values.
5. Kernel widths `σ_t, σ_yaw, σ_gv` and contact indicators `c_f, c_s, c_d` — not specified; using `0.25` and contact-sensor-derived indicators by analogy with `legged_gym`/`Walk These Ways`.
6. Cadence `k` range, curriculum reward threshold, GRU hidden size, learning rate, PPO clip/entropy coefficients — paper doesn't give these; start from the stock `unitree_go2_ppo_runner_cfg()` values and retune.
7. Advantage computation: per-agent (4 GAE traces) vs. shared single GAE (§8.8) — recommended default stated, but genuinely ambiguous in the paper's own text (Eq. 4 vs. §IV-B prose).
8. Actuator network (Hwangbo et al. 2019) for sim2real PD-error correction — paper uses one; whether `unitree_rl_mjlab`'s existing `deploy/` pipeline already includes an equivalent is unverified in this plan (not inspected) — check `deploy/robots/go2` before assuming an ideal PD controller is sufficient.
9. Restitution and gravity-impulse-duration randomization (§6) — no built-in `mjlab.envs.mdp.dr` primitive found for restitution; may exist under a different name — search before writing new code.
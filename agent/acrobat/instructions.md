# instructions.md — Two-Stage ES→PPO Acrobatics in `unitree_rl_mjlab`

**Paper:** Vezzi, Ding, Raffin, Kober, Della Santina — *Two-Stage Learning of Highly Dynamic Motions with Rigid and Articulated Soft Quadrupeds* (arXiv 2309.09682v2)
**Target repo:** `unitreerobotics/unitree_rl_mjlab` (built on `mjlab` = Isaac-Lab-style manager API + MuJoCo-Warp, trained with `rsl_rl`)
**Target robot:** Unitree **Go2** (the paper used Go1; Go1 is not in this repo — Go2 is the quadruped to use)
**Audience:** you, the coding agent. I am the planner. Follow this document in order, respect the gates, and document every deviation.

---

## 0. Rules of engagement

1. **Explore before coding.** I have only seen the repo README, not the source tree. Every file path, class name, and mjlab API name below is a *concept* unless you verify it. Never invent an API: `grep`, read the Go2 velocity task, and mirror its patterns.
2. **Do not modify existing tasks, `deploy/`, or `simulate/`.** All work is additive: new task package, new scripts, new tests, one doc file.
3. **Match repo conventions** (config dataclasses, `tyro` CLI, task registration, logging layout under `logs/`).
4. **Gates.** Each phase ends with a verification gate (Section 12). Do not start phase N+1 until gate N passes. If a gate cannot pass, write the failure and your hypothesis to `NOTES.md` and stop that branch rather than papering over it.
5. **Ambiguities are pre-decided** in Section 11. Use those defaults; do not block waiting for answers. Put every tunable in a config dataclass, never as a magic number in a function body.
6. **Everything is batched on GPU.** No per-env Python loops in the step path. All trackers/state machines are `torch` tensors of shape `(num_envs, ...)` and must be reset through the env's reset hook.
7. Commit per phase with a clear message.

---

## 1. What the paper does (the part you must reproduce)

**Idea:** learn highly dynamic motions (jump in place, jump forward, pronk, back-flip) *from scratch* with no reference motion and minimal reward shaping.

| Stage | Algorithm | Policy | Reward | Extras |
|---|---|---|---|---|
| **1** | **ARS** (Augmented Random Search, gradient-free) | **Linear:** `a = W s + b` | **Sparse**, paid only at episode end | Behavior-selector state machine + action low-pass filter |
| **2a** | PPO (warm start) | MLP 2×64 tanh | Imitation: `r_ini = w_a · exp(−w_b · ‖a − a_ars‖²)` | State machine + LPF **removed**; flight flag added to obs |
| **2b** | PPO (refine) | same network | **Dense** task reward (Eq. 8–12) | optional domain randomization |

Optional "soft" variant: parallel springs on the sagittal joints (thigh, calf). Paper reports ~+15% jump height in place and ~+23% jump distance forward vs the rigid robot (do not treat these numbers as pass/fail criteria).

**Scope split**
- **MUST implement:** jump-in-place, jump-forward, rigid Go2, Stage 1 + Stage 2a + 2b, logging, tests, docs.
- **SHOULD implement:** parallel-spring ("soft") option, domain randomization (Table II analogue).
- **MAY implement (only after everything above passes):** back-flip, then pronking.
- **OUT OF SCOPE:** sim2real/ONNX deploy changes, hardware. (Note in docs: `h` and `ḣ` need state estimation or mocap on hardware.)

---

## 2. Phase 0 — Recon (write findings to `NOTES.md`)

Run and record answers to each question. Do not proceed with guesses.

```bash
cd unitree_rl_mjlab
cat doc/setup_en.md                       # install steps
ls scripts src src/assets
grep -rn "Unitree-Go2-Flat" src scripts   # find task registration + Go2 velocity cfg
python scripts/train.py --help
python scripts/play.py --help
pip show mjlab rsl-rl-lib mujoco-warp 2>/dev/null | grep -E "Name|Version"
```

Questions `NOTES.md` must answer:

1. Where is the Go2 entity/robot cfg (MJCF path, `init_state` default joint pos, joint limits, actuator type, kp/kd, effort limits)? Print the 12 joint names **in simulator order** and the default pose.
2. Which actuator types does this mjlab version offer? Can PD gains (`kp`, `kd`) be changed **per env at runtime**? If not, torque (motor) actuators + explicit PD in the action term is the fallback (Section 5.3).
3. How are tasks registered (function/decorator, env-cfg + rl-cfg pairing)? How does the CLI map `--env.*` / `--agent.*` onto dataclass fields?
4. How is a **contact sensor** defined (feet geoms, `found`/force fields, per-env force vectors)? How does the velocity task detect illegal contacts (copy that for knee/thigh/calf/trunk termination)?
5. **Step ordering:** in `env.step`, are rewards computed *before* the reset of terminated envs, and can a reward fn read `termination_manager.terminated` / `time_outs` of the same step? (Required for terminal-only rewards. If not, see Section 7.1 fallback.)
6. Which `rsl_rl` version and which runner/wrapper does `scripts/train.py` use? How do you resume (`--agent.resume`, `load_run`)? Is the actor/critic MLP configurable (hidden dims, activation, init noise std)? Is LR schedule configurable (`fixed` vs `adaptive`)?
7. Does the env cfg expose events for domain randomization (mass, CoM, joint stiffness/damping)? List what exists.
8. Does the Go2 MJCF have **self-collision** enabled between thigh/calf/trunk? (Needed for the "links collide" termination.)
9. Env `dt`, decimation (control rate). Default is likely 200 Hz sim / 50 Hz control; keep it unless tests show it is inadequate.

**Gate 0:** `NOTES.md` complete; `python scripts/train.py Unitree-Go2-Flat --env.scene.num-envs=64 --agent.max-iterations=2` runs end-to-end on this machine.

---

## 3. Proposed layout (adapt names to what Phase 0 finds)

```
src/<tasks_pkg>/acrobatics/
  __init__.py                    # registers tasks (below)
  cfg.py                         # AcrobaticsCfg dataclass: ALL hyperparameters
  env_cfg.py                     # make_env_cfg(motion, stage, soft, randomize) factory
  rl_cfg.py                      # rsl_rl PPO cfgs for Imitate / Refine
  mdp/
    sagittal.py                  # symmetric 6-DoF <-> 12-DoF mapping, home pose, joint-limit scaling
    phase.py                     # PhaseTracker (TAKEOFF/FLIGHT/LANDING) + flight flag
    trackers.py                  # per-episode stats: h_bar, d_bar, theta_bar, peak force, jump list
    actions.py                   # SagittalPDAction (selector, LPF, PD torque)
    observations.py              # h, hdot, pitch, q(6), qdot(6), [flight flag]
    rewards.py                   # kernels, sparse terminal reward, dense Stage-2 rewards, imitation reward
    terminations.py              # fall, illegal contact, tilt
    events.py                    # randomization + spring setup
  ars/
    linear_policy.py             # BatchedLinearPolicy, RunningNorm
    ars.py                       # ARSTrainer (V2-t)
scripts/
  train_ars.py                   # Stage 1 entry point
  play_ars.py                    # visualize the ARS policy (+ metrics)
  train_two_stage.py             # optional convenience wrapper: ARS -> Imitate -> Refine
tests/acrobatics/
  test_kernels.py  test_ars_toy.py  test_phase.py  test_sagittal.py  test_env_smoke.py
doc/acrobatics.md                # short usage doc (<= 1 page)
NOTES.md
```

**Task IDs to register** (names follow the repo's `Unitree-<Robot>-<Task>` style):

```
Unitree-Go2-JumpInPlace-ARS        Unitree-Go2-JumpForward-ARS
Unitree-Go2-JumpInPlace-Imitate    Unitree-Go2-JumpForward-Imitate
Unitree-Go2-JumpInPlace-Refine     Unitree-Go2-JumpForward-Refine
# soft variants: suffix "-Soft"  (e.g. Unitree-Go2-JumpInPlace-Refine-Soft)
# optional later: Backflip-*, Pronk-*
```

All of these come from one factory `make_env_cfg(motion, stage, soft, randomize)` — no copy-pasted configs.

---

## 4. Phase 1 — Shared building blocks

### 4.1 Sagittal-symmetric mapping (`mdp/sagittal.py`)

The paper learns only **6 DoF**: (hip, thigh, calf) for the *front* leg pair and the *rear* leg pair; left/right are mirrored.

- **Policy action** `a ∈ [-1, 1]^6`, order `[front_hip, front_thigh, front_calf, rear_hip, rear_thigh, rear_calf]`.
- **Expand to 12 joints** by copying to left/right. **Hip abduction has mirrored sign conventions** between left and right legs on Go2 (default hip angles are +/− of each other). Determine empirically from the MJCF joint axes and default pose which joints need a sign flip; encode it as a `hip_sign` vector, and write a unit test that "symmetric action ⇒ left/right legs mirror each other physically" (e.g. both feet's lateral offsets are mirror images).
- **Joint-limit scaling** (paper: "maximum actions correspond to the joint limits"):
  `q_ref = q_lo + 0.5·(a+1)·(q_hi − q_lo)` per DoF, with limits read from the MJCF (never hard-code). For the left/right hip, apply `hip_sign`.
- **Home pose:** `q_home` = the robot's default standing pose from the entity cfg. `a_home = inverse_map(q_home)`. Test: `map(a_home) ≈ q_home` to 1e-5.
- **Observation averaging:** `q_i = (q_i^left + q_i^right)/2` with the same `hip_sign` convention; same for `q̇`.

### 4.2 Contact signals

From the foot contact sensor, per env:
- `F_contact` = **total** contact force magnitude summed over the 4 feet (scalar, N).
- `n_contact` = number of feet with force > ε (ε ≈ 1 N).
- `airborne` = `F_contact < F_eps` (default `F_eps = 1.0 N`), with a 1-step debounce option (`airborne_debounce_steps`, default 1).

### 4.3 Episode trackers (`mdp/trackers.py`)

Per env, updated **every control step**, reset on env reset:

| Name | Definition |
|---|---|
| `h_bar` | max base height `h` since reset |
| `d_bar` | max horizontal displacement from reset position, **signed along the body's initial forward axis (x)**; `d_bar = max(0, max_t x_t − x_0)` for forward jump; for in-place use `max_t |x_t − x_0|` |
| `theta_bar` | max |pitch| since reset (see 4.4 for backflip-safe pitch) |
| `peak_force` | max `F_contact` since reset |
| `n_air_steps` | number of airborne steps (for diagnostics) |
| `terminated_early` | bool: terminated by fall/contact/tilt (not time-out) |

Normalization (paper Eq. 4): `h_n = clip(h_bar / h_max, 0, 1)`, `d_n = clip(d_bar / d_max, 0, 1)`; `h_max`, `d_max` are config values ("tuned heuristically" in the paper).

### 4.4 Pitch

For jumps use the sagittal pitch from the base rotation (`atan2` of the body x-axis z-component vs x-component in the world x–z plane), not raw Euler angles. For the (optional) back-flip, **unwrap** it into a continuous angle (accumulate `Δθ` each step using wrapped differences) so it can exceed ±π.

### 4.5 Phase tracker (`mdp/phase.py`) — the "behavior selector"

State per env: `phase ∈ {TAKEOFF=0, FLIGHT=1, LANDING=2}`, reset to `TAKEOFF`.
Transitions (evaluated every control step, in this order):

```python
# state: phase (E,), land_counter (E,)
to_flight  = (phase == TAKEOFF) & airborne                       # guard: F_contact == 0
to_landing = (phase == FLIGHT)  & (hdot < 0) & (F_contact > F_eps)   # guard: hdot<0 and F_contact>0
# repeated motions only (pronking): LANDING -> TAKEOFF once contact has persisted
land_counter = where(phase == LANDING, land_counter + 1, 0)
to_takeoff = repeat & (phase == LANDING) & (land_counter >= landing_settle_steps)
phase = where(to_flight, FLIGHT, phase)
phase = where(to_landing, LANDING, phase)
phase = where(to_takeoff, TAKEOFF, phase)
```

Expose: `phase`, `flight_flag = (phase == FLIGHT)` (float 0/1; this is the paper's Boolean `b`), and `jump_count` (increments on each `FLIGHT→LANDING`).

The tracker is used in **both** stages. In Stage 1 it also *changes actions*; in Stage 2 it is passive (used for the observation flag and the imitation target).

---

## 5. Phase 2 — Environment

### 5.1 Observations (`mdp/observations.py`)

State `s = {h, ḣ, θ, q(6), q̇(6)}` → **15 dims** (+1 flight flag `b` in Stage 2 = **16 dims**).

- `h`: base (trunk) height above flat ground (world z).
- `ḣ`: world-frame vertical base linear velocity.
- `θ`: sagittal pitch (4.4).
- Add white observation noise (paper: "white noise is added to emulate sensor noise"). Defaults (assumed): `σ_h=0.01, σ_ḣ=0.05, σ_θ=0.01, σ_q=0.01, σ_q̇=0.05`. Config-driven; zero in play/eval mode.
- **Stage 1 / ARS obs: no flight flag.** Stage 2: flag appended as the last entry.
- **Critic (Stage 2) privileged obs — recommended:** clean actor obs + `phase` one-hot + `h_bar, d_bar, theta_bar` + `t/T`. Rationale: the terminal bonus depends on episode-level stats the actor cannot see. Keep actor obs paper-faithful (16 dims, no timer).

### 5.2 Terminations (`mdp/terminations.py`)

Early termination (counts as `terminated_early`, which the rewards use):

1. **Fall:** `h < h_fall` (paper 0.1 m; Go2 default).
2. **Illegal contact:** trunk, thigh, or calf geoms touching the ground, or links self-colliding (paper: "knees touch the ground, or links collide"). Reuse the velocity task's illegal-contact mechanism; allow only foot geoms to touch the ground.
3. **Tilt:** `dot(body_z_axis, world_z) < 0.85` (paper). **Disabled for back-flip.**
4. **Time-out:** `episode_length_s` (defaults in Section 11), *not* an early termination.

### 5.3 Action term (`mdp/actions.py`) — `SagittalPDAction`

Pipeline per control step (stage-dependent flags in config):

```
a_policy (E,6)
  └─ [Stage 1 only, use_behavior_selector]  a_sel = where(phase==TAKEOFF, a_policy, a_home)
  └─ [Stage 1 only, use_lpf]                a_f   = α·a_f_prev + (1−α)·a_sel      # reset to a_home on env reset
  └─ clip to [-1, 1]
  └─ expand + joint-limit scaling → q_ref (E,12)
  └─ PD torque:  τ = kp·(q_ref − q) − kd·q̇     (kp,kd per joint class; × landing_gain_scale when phase==LANDING and use_behavior_selector)
  └─ clip to per-joint effort limits → apply
```

- `landing_gain_scale` (Stage 1 only): paper reduces effective joint stiffness in landing to get a compliant touchdown. Config default `0.5` for `kp` and `kd` (assumed; tune in `[0.2, 0.8]`).
- `lpf_alpha` default `0.5` (assumed; paper gives no value).
- **PD implementation choice (decide in Phase 0):**
  (a) if the actuator class supports per-env runtime kp/kd, use it; else
  (b) use torque/motor actuators and compute `τ` explicitly in `apply_actions` (preferred for fidelity: matches `τ = kp(q_ref − q) − kd·q̇` exactly).
  Default `kp, kd` = the Go2 velocity task's values unless the standing test (Gate 2) fails.
- Stage 2 flags: `use_behavior_selector=False`, `use_lpf=False`, constant `kp, kd` (paper: "we remove these additions in the second stage").
- Provide `reset(env_ids)` that resets LPF state to `a_home` and phase tracker.

### 5.4 Reset state

Start at `q_home`, base at standing height, zero velocity, with small noise (`±0.02 rad` joint, `±0.01 m` height, `±0.02` rad pitch) to avoid exact determinism. Config-driven.

### 5.5 Optional "soft" robot (parallel springs) — do this after Gate 4

Paper: parallel springs on the **sagittal thigh and calf joints** only. Torque from the spring on joint *i*: `τ_s = −k_i (q_i − q_i⁰) − c_i q̇_i`.

> The paper's Appendix A formula writes the spring term using the *commanded* angle `q_ref`; that is almost certainly a typo for the measured angle `q` (a spring acts on joint deflection). **Implement the physical version above** and note the deviation in `NOTES.md`.

Implementation (preferred, no custom physics code): set MuJoCo joint `stiffness`, `springref` (= `q⁰`) and `damping` on the 4 thigh+calf joints via the entity cfg / `MjSpec` edit, behind `cfg.soft=True`. Since springs are passive and act in parallel with actuators, no change to the action term is needed. If mjlab's per-env model fields allow it, expose `k, c` for domain randomization (Section 8).

Defaults (assumed; the paper's text does not list them): `k_thigh=k_calf=8.0 N·m/rad`, `c=0.1 N·m·s/rad`, `q⁰` = joint angles at near-full leg extension (so crouching stores energy that is released at take-off). Provide a sweep script option `--soft.k` and report jump height vs `k ∈ {0, 4, 8, 16}`.

---

## 6. Phase 3 — Stage 1: ARS with a linear policy

### 6.1 Policy parameterization

`π(s) = clip(W·ŝ + b, −1, 1)` where `ŝ = (s − μ)/(σ + ε)` (ARS **V2** observation normalization). `W ∈ R^{6×15}`, `b ∈ R^6`; flatten to `θ ∈ R^{96}`. Initialize `θ = 0` (⇒ policy outputs the mid-range action, so the robot is not at home pose; consider `b_init = a_home` so iteration 0 stands still — recommended, document it).

### 6.2 Batched evaluation (the key engineering trick)

With `N` directions, evaluate `2N` perturbed policies **in parallel in one env batch** (`num_envs = 2N · rollouts_per_candidate`). Env `e` uses its own `θ_e`:

```python
# W_all: (E, A, S)   b_all: (E, A)    obs: (E, S)
a = torch.bmm(W_all, ((obs - mu)/(sd + 1e-8)).unsqueeze(-1)).squeeze(-1) + b_all
a = a.clamp(-1, 1)
```

### 6.3 ARS iteration (V2-t)

```python
def ars_iteration(theta, N, top_b, nu, alpha):
    delta  = torch.randn(N, P, device=dev)
    thetas = torch.cat([theta + nu*delta, theta - nu*delta], 0)      # (2N, P)
    R      = evaluate(thetas)                                         # (2N,) one episode each
    r_pos, r_neg = R[:N], R[N:]
    idx    = torch.topk(torch.maximum(r_pos, r_neg), top_b).indices
    sigma_R = torch.cat([r_pos[idx], r_neg[idx]]).std().clamp_min(1e-6)
    step   = ((r_pos[idx] - r_neg[idx]).unsqueeze(-1) * delta[idx]).sum(0)
    return theta + alpha / (top_b * sigma_R) * step
```

Defaults (assumed; the paper does not list ARS hyperparameters — these are sensible for a 96-parameter policy): `N=256` (⇒ 512 envs), `top_b=64`, `nu=0.05`, `alpha=0.02`, `iterations=500`, `rollouts_per_candidate=1`, optional cosine decay of `alpha` and `nu` (off by default).

**`evaluate()` semantics**
- Reset all envs; run until each env hits its **first** `done` (terminated *or* timed out); record that env's terminal reward then; ignore it afterwards (mask) while others finish; stop when all have finished at least once. Cap loop at `max_episode_steps + 1`.
- Update the running obs normalizer **only after** the iteration, using all observations seen by the unmasked steps (Welford/batch merge). Freeze stats during an iteration so all candidates are comparable.
- Save `ars_policy.pt` containing `{W, b, mu, sd, obs_dim, act_dim, cfg}` at `logs/ars/<robot>_<motion>/<datetime>/` — both `best` (by mean of top candidates' evaluated return of the *unperturbed* θ, evaluated every `eval_every=10` iterations over 64 noisy episodes) and `last`.
- Log to TensorBoard (match `train.py`'s logging backend): mean/max/min return, `h_bar`, `d_bar`, `theta_bar`, peak force, early-termination rate, `‖θ‖`.

### 6.4 Stage-1 environment config

- Rewards: none at the per-step level; **one** terminal-only reward term (Section 7.1).
- Action term: `use_behavior_selector=True`, `use_lpf=True`.
- Obs: no flight flag, with noise.
- Randomization: **off** for the first ARS run (paper trains DR variants as a *retraining*, "ARS-r").

### 6.5 Entry point `scripts/train_ars.py`

```bash
python scripts/train_ars.py Unitree-Go2-JumpInPlace-ARS \
  --ars.n-directions=256 --ars.top-b=64 --ars.nu=0.05 --ars.alpha=0.02 \
  --ars.iterations=500 --env.scene.num-envs=512
```

Use the **same env construction + tensor wrapper** that `scripts/train.py` uses, so obs/dones/extras are torch tensors on the same device. No gradients, `torch.inference_mode()`.

`scripts/play_ars.py` loads `ars_policy.pt`, runs the env with the viewer, prints per-episode `h_bar, d_bar, theta_bar, peak_force`.

---

## 7. Rewards

### 7.1 Stage 1 — sparse terminal reward (Eqs. 5–7)

Nothing during the episode; at the step an episode **ends** (early termination *or* time-out) pay `r_end + r_bonus`.
Kernel: `g(x; a, b) = a · exp(−b·|x|)`.

**Jump in place** (Eq. 6):
```
r_end   = [ g_θ(θ̄) + g_d(d_n) + c_h ] · h_n
r_bonus = + b · h_n                      if NOT early termination
        = −( q + m · h_n )               if early termination
```
**Jump forward** (Eq. 7) — distance enters *linearly* instead of through a kernel:
```
r_end   = [ g_θ(θ̄) + c_d · d_n + c_h ] · h_n
r_bonus = + b · (h_n + d_n)              if NOT early termination
        = −( q + m · (h_n + d_n) )       if early termination
```
(`θ̄`, `h_n`, `d_n` come from the trackers in 4.3.)

**Implementation requirement:** the reward term must know "episode ending now" and "was it early termination". Use the termination manager's `terminated` / `time_outs` buffers *as seen by the reward manager in the same step* (verify ordering in Phase 0, question 5). **Fallback if mjlab resets before computing rewards:** compute the terminal reward in a termination-term side-effect that writes into a persistent `env.extras`-style buffer, or snapshot trackers just before reset in a `pre_reset` hook; whichever you choose, unit-test that a scripted early-termination gives the penalty branch and a full-length episode gives the bonus branch.

**Hyperparameters.** The PDF's Table III extracted badly (the jump-in-place row is empty, column labels are garbled). Values I could read for *jump forward*, in header order `h_f, a_θ, b_θ, d_f, c_d, c_h, b, q, m`: `0.3, 0.25, 0.0225, 1.3, 0.5, 25, 0.1, 0.08, 0.096`. Treat these as **unreliable**. Procedure:
1. If you have network access, open the PDF at `https://arxiv.org/pdf/2309.09682v2` (Appendix C, Tables III–IV) and read them visually.
2. Otherwise use the defaults in Section 11 and tune with the sanity rule: **every bracketed term should be O(1)**, so a decent jump earns `r_end ≈ 1–3` and a crash earns `≈ −0.1 … −1`. (Note `b_θ = 0.0225` is only sensible if θ̄ is in **degrees** — keep an explicit `pitch_unit` field in the config and document which unit each kernel uses.)

### 7.2 Stage 2a — imitation reward (warm start)

`r_ini = w_a · exp(−w_b · ‖a − a_ars‖²)`, with `a` the NN action and `a_ars` the Stage-1 *effective* action.

Because the Stage-1 state machine overrides the linear policy during flight/landing, define the target as the Stage-1 pipeline's **pre-filter, pre-gain-scaling output** for the current phase:

```python
a_lin  = clip(W @ ((s_clean - mu)/(sd+eps)) + b, -1, 1)       # s_clean: the 15-d state, clean, no flag
a_ars  = where(phase == TAKEOFF, a_lin, a_home)                # phase from the passive PhaseTracker
```

Implement as a reward term that loads `ars_policy.pt` once (frozen tensors, `requires_grad=False`). Defaults: `w_a = 1.0`, `w_b = 4.0` (assumed — the paper does not give them; pick `w_b` so a typical error of 0.2 in action-norm gives ≈0.85).

Stage-2a runs for `imitation_iterations` (default 100–200 PPO iterations) with **only** `r_ini` active. Initialize the actor with a small action std (`init_noise_std ≈ 0.1–0.3`) so it stays near the target while learning to copy it.

*Note (flag in docs):* the paper's wording ("replicating the observation–action pair provided by the first-stage policy") does not say how flight/landing are imitated; the construction above is my interpretation.

### 7.3 Stage 2b — dense task reward (Eqs. 8–12)

`r = r_h + r_c + r_d + r_s + r_θ + r_bonus`, all as separate named reward terms (so each is logged and weight-tunable). Kernel `g` as above.

| Term | Definition (per control step unless stated) | Notes |
|---|---|---|
| `r_h` | `a_h · h` if `h_min ≤ h ≤ h_max` else `0` | encourages height, no reward when not jumping or when above the cap; uses time-varying `h` |
| `r_c` | `−a_c · f` if `f ≥ f_min` else `0` | `f = F_contact / max(n_contact,1)` (avg per-foot force); soft landing |
| `r_d` (in place) | `g_d(d) = a_d·exp(−b_d·|d|)` | `d` = current horizontal displacement |
| `r_d` (forward) | `k_d · d` | `d` = current forward displacement |
| `r_s` | `g_s(δτ) = a_s·exp(−b_s·‖δτ‖)` | `δτ` = difference of the 6 sagittal torques (front/rear × hip/thigh/calf, take left-leg or L/R mean) between consecutive steps |
| `r_θ` | `g_θ(θ) = a_θ·exp(−b_θ·|θ|)` | current pitch |
| `r_bonus` (in place) | at episode end: `0` if no early term., `−m · h̄` if early term. | implement as written, but see Section 11 note |
| `r_bonus` (forward) | at episode end: `b · (h̄ + d̄)` if no early term., else `0` | |

Stage-2b defaults (read from the extracted Table IV; **verify against the PDF if possible**, and rescale for Go2 — see Section 11):

- **In place:** `a_h=0.01, h_min=0.29, h_max=1.3, a_c=1.5e-4, f_min=800, a_d=6.5e-4, b_d=40, a_s=3e-3, b_s=0.1, a_θ=4.2e-3, b_θ=26, m=0.25`
- **Forward:** `a_h=6.5e-3, h_min=0.29, h_max=1.1, a_c=1.2e-4, f_min=800, k_d=1.52e-2, d_max=1.3, a_s=3e-3, b_s=0.1, a_θ=4.2e-3, b_θ=26, b=0.025`

Go2 adaptation: `h_min` should be ≈ `h_stand − 0.03` where `h_stand` is Go2's measured standing height in sim (the paper's `0.29` belongs to Go1 with `z₀ = 0.32 m`). Go2 is heavier than Go1, so scale `f_min` by mass ratio (or verify that `f_min` ≈ 6–7× body weight is still the intended impact threshold).

### 7.4 PPO configuration (Stage 2a and 2b)

Map the paper's SB3/RL-Zoo settings onto `rsl_rl`:

| Paper | rsl_rl setting |
|---|---|
| MLP 2×64, tanh | actor and critic `hidden_dims=[64, 64]`, `activation="tanh"` |
| lr `2e-4` | `learning_rate=2e-4`, **`schedule="fixed"`** |
| γ `0.999` | `gamma=0.999`, `lam=0.95` |
| clip `0.1` | `clip_param=0.1` |
| batch `4096` | choose `num_mini_batches = num_envs · num_steps_per_env / 4096` (e.g. 4096 envs × 24 steps ⇒ 24 mini-batches) |
| — | `num_steps_per_env` ≥ episode length in steps is desirable because of γ=0.999 and terminal bonuses (try 64–128 for 150-step episodes), `entropy_coef` small (`~1e-3`), `num_learning_epochs=5`, obs normalization **on** |

**Handoff 2a → 2b:** run 2b by *resuming* the 2a checkpoint (actor weights carry over; the critic will re-fit to the new reward, which is fine). Prefer two sequential `train.py` invocations (`...-Imitate` then `...-Refine` with `--agent.resume`) — simplest and debuggable. The convenience script `train_two_stage.py` just calls them in sequence. (Optional alternative: one run with a curriculum that swaps reward weights at iteration N — only if the repo's curriculum manager makes it trivial.)

---

## 8. Phase 5 — Domain randomization (SHOULD)

Paper Table II analogue, applied **at reset** (per env), only in the **retraining** runs (`ARS-r` / `ARS+DRL-r` in the paper) via `randomize=True`:

| Quantity | Range |
|---|---|
| leg link masses | ±20% |
| trunk mass | added payload in `[0, 4] kg` (paper: "[0 4] kg") |
| CoM offset (trunk) | `±[0.2, 0, 0.2]` m **as written in the paper** — this is very large for a 0.3 m robot; clamp to `±[0.02, 0.01, 0.02]` m as the default and keep the paper value behind a flag `paper_com_offset` |
| inertia | rescaled consistently with mass |
| spring stiffness (soft only) | ±30% |
| spring damping (soft only) | ±30% |

Use mjlab's existing randomization events if present (Phase 0 Q7). If a quantity is not supported, skip it, log a warning, and note it in `NOTES.md` — do not hack MuJoCo model arrays unless the per-env model-field mechanism is clearly documented.

Expected qualitative effect from the paper: higher success rate; forward-jump distance may drop slightly. Do not chase those numbers.

---

## 9. Phase 6 (MAY) — Back-flip, then pronking

Only after Gates 1–5 pass. Reuse every component above; change only the following.

**Back-flip**
- Terminations: remove the tilt termination.
- Pitch: use the unwrapped continuous pitch (4.4); `θ_n = clip(θ_max_reached / (2π), 0, 1)`.
- Stage-1 reward (Eq. 16): `r = w_h·h̄ + w_θ·θ_n + w_hθ·(h̄·θ_n) + b_succ` where `b_succ` is a bonus for ending without early termination. Defaults: `w_h=1, w_θ=1, w_hθ=2, b_succ=0.5` (assumed).
- Stage-2 reward: Eq. 8 without `r_d`; `r_θ` becomes **proportional to** pitch (`k_θ·θ`, rewarding rotation); bonus (Eq. 17) `w_bonus · h̄ · θ_n` if no early termination.
- Expect the paper's observation that it is hard without parallel springs: run with `soft=True`.

**Pronking** (repeated jumps; Appendix B.1, Eqs. 13–15)
- Phase tracker `repeat=True`; maintain per-env list `p_sj[0..J_max]` of per-jump scores `p_sj = w_h·h_n + w_d·d_n` (store at each `FLIGHT→LANDING`, `J_max=16`).
- `p_avg` = mean of recorded jumps; `S(p)` = entropy of normalized `p` (`p_i / Σp`) — higher entropy = more uniform jumps.
- Stage 1: `r = w_avg·r_avg + w_max·p_max + w_c·c_j + b`, with `r_avg = p_avg·{g_θ(θ̄) + w_t·t_n + w_s·S(p) + k}`, `c_j` = number of jumps above `p_thresh`, `t_n` = episode_end_time/T_max. Defaults (assumed): all `w=1`, `k=0`, `p_thresh=0.3`, `b=0.5`.
- Stage 2 adds `r_p = g_p(E)` (energy `E = Σ|τ·q̇|·dt`) and `r_j = w_j·p_avg·[g_s(S) + l_j]` with `w_j = 0` when `p_avg < p_thresh`.
- Success check: vertical CoM and pitch converge to a limit cycle (plot `h(t)` over a long episode).

---

## 10. Logging, play, docs

- All new reward terms log individually (per-term mean per iteration).
- Log per-iteration eval metrics from trackers: `h_bar` (and height gain over stand), `d_bar`, `theta_bar`, `peak_force`, early-termination rate, jump-success rate (`airborne ≥ 3 steps AND no early termination`).
- `scripts/play.py <Stage-2 task> --checkpoint_file=...` must work unchanged with the new tasks (it should, if registration follows repo convention). Add `--motion_file`-style extras only if unavoidable.
- `doc/acrobatics.md` (≤ 1 page): the 3 commands (ARS → Imitate → Refine), config knobs, expected metrics, known deviations.

---

## 11. Defaults & decisions (so you never have to guess)

| Item | Decision | Source |
|---|---|---|
| Robot | Go2 | repo supports Go2, not Go1 |
| Control rate | repo default (assume 50 Hz) | assumed |
| Episode length | jumps: 3.0 s; back-flip: 3.0 s; pronk: 6.0 s | assumed (paper omits) |
| `h_fall` | 0.10 m | paper |
| Tilt threshold | `dot(z_body, z_world) < 0.85` | paper |
| Network | 2×64 tanh | paper |
| PPO lr / γ / clip | 2e-4 / 0.999 / 0.1 | paper |
| PPO batch | 4096 | paper |
| ARS (N, b, ν, α) | 256, 64, 0.05, 0.02 | assumed |
| ARS obs-norm | on (V2-t) | Mania et al. standard |
| LPF α | 0.5 | assumed |
| Landing gain scale | 0.5 | assumed |
| `r_ini` weights | `w_a=1, w_b=4` | assumed |
| Imitation length | 150 PPO iterations | assumed |
| Spring (soft) | `k=8 N·m/rad`, `c=0.1`, `q⁰≈extended` | assumed |
| `h_max`, `d_max` (normalization) | jump height gain ≈ 0.3 m ⇒ `h_max = h_stand + 0.3`; `d_max = 1.3 m` (forward) | paper (partly) + assumed |
| Spring torque | physical `−k(q−q⁰) − cq̇` (paper's `q_ref` appears to be a typo) | decision |
| In-place bonus on early termination: `−m·h̄` | implement **as written**; log a warning in docs that it penalizes high-then-crash episodes more | paper as written |
| Table III/IV values | use as starting points only; re-read PDF if possible | extraction unreliable |
| Joint count | learn 6 sagittal DoF, mirror to 12 | paper |

---

## 12. Verification gates

Run each gate's commands; paste outputs into `NOTES.md`.

**Gate 1 — pure-math unit tests (`pytest tests/acrobatics`)**
- Kernel `g` values, clip/normalization, reward Eqs. 6–7 on hand-computed examples (both branches).
- `sagittal` map: `map(a_home) == q_home`; symmetric action gives mirrored legs; round-trip `inverse_map`.
- `PhaseTracker`: scripted sequences `(F, ḣ)` produce `TAKEOFF→FLIGHT→LANDING`; `repeat=True` returns to `TAKEOFF`; reset works.
- **ARS toy test:** maximize `−‖Wx−y‖²` or a 2-D quadratic with the same `ars_iteration`; return must improve monotonically-ish and reach >95% of optimum within 200 iterations. This proves the optimizer independent of the sim.

**Gate 2 — environment sanity**
- `python scripts/play.py Unitree-Go2-JumpInPlace-ARS --agent zero` and `--agent random` run without NaNs (use whatever dummy-agent flags exist in this repo/mjlab version).
- Constant `a = a_home` for 3 s: no termination, `h` within ±1 cm of `h_stand`.
- **Scripted open-loop jump** (hand-made action sequence that extends the legs quickly): phase log shows all three phases in order, `h_bar` exceeds `h_stand`, trackers match a replay computed offline from the logged trajectory.
- Terminal reward: forced fall ⇒ penalty branch; full-length standing episode ⇒ bonus branch with `h_n≈0`.

**Gate 3 — Stage 1 works**
- `train_ars.py` for jump-in-place: mean return rises over iterations; the final policy gets **sustained flight** (`≥ 3` airborne steps) in ≥ 80% of 64 evaluation episodes and `h_bar − h_stand ≥ 0.05 m`. Save a short video/GIF or `play_ars.py` metrics table.
- Repeat for jump-forward: `d_bar ≥ 0.3 m` in ≥ 80% episodes.
- If stuck at zero-reward plateaus: first raise `nu`, then check `h_max`/`d_max` normalization, then check the contact-force airborne detection.

**Gate 4 — Stage 2**
- Imitation: `r_ini` mean > 0.9·`w_a` after the imitation phase; NN rollout reaches ≥ 90% of the ARS policy's `h_bar` (no state machine, no LPF).
- Refine: dense return increases; compare ARS vs ARS+DRL on 256 eval episodes. Report mean±std for `h_bar`, `d_bar`, `peak_force`, success rate. Expected direction (paper): ARS+DRL ≥ ARS in height, and lower landing force for in-place. Report honestly if not met, with a hypothesis.

**Gate 5 — Soft & randomized**
- Rigid vs soft table (`k` sweep) with the same seeds; randomized-retrain success rate vs non-randomized policy evaluated on randomized envs.

**Gate 6 (optional)** — back-flip / pronking criteria from Section 9.

---

## 13. Common pitfalls

- **Terminal reward ordering** (reward computed after reset ⇒ trackers are already zeroed). Test explicitly.
- **Phase/LPF/tracker state not reset** for envs that reset mid-batch ⇒ silent corruption. Reset them from the same `env_ids` hook as everything else.
- **Left/right hip sign** mismatch makes the robot splay or collapse sideways. Verify with the mirror test.
- **Airborne detection** from contact force can chatter at touchdown; use the debounce option before blaming the optimizer.
- **ARS normalizer drift:** stats must be saved and reused by the imitation target and by `play_ars.py`.
- **`γ=0.999` + terminal bonus + short rollouts** ⇒ poor credit assignment. Use longer `num_steps_per_env` and the privileged critic obs.
- **Go1 → Go2 scale mismatch** (mass, standing height, force thresholds). Re-derive thresholds from measured Go2 values.
- **Don't break the existing velocity/tracking tasks**: run `train.py Unitree-Go2-Flat` for 2 iterations at the end as a regression check.

---

## 14. Definition of done

1. `python scripts/train_ars.py Unitree-Go2-JumpInPlace-ARS ...` → `ars_policy.pt`, jump demonstrated (Gate 3).
2. `python scripts/train.py Unitree-Go2-JumpInPlace-Imitate --<ars path> ...` then `...-Refine --agent.resume ...` → checkpoint that plays with `scripts/play.py` (Gate 4).
3. Same for JumpForward.
4. Unit tests green; existing tasks unaffected.
5. `NOTES.md` records Phase-0 findings, every deviation, and the final metrics table; `doc/acrobatics.md` exists.
6. Soft/randomized options present and tested (Gate 5), or an explicit note in `NOTES.md` explaining which mjlab capability blocked them.
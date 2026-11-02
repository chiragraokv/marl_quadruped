# Go2 acrobatics

Three-stage workflow for rigid Go2 jump-in-place and jump-forward tasks. Keep every test or evaluation at 500 environments or fewer. PPO uses `--env.scene.num-envs`; ARS uses `--n-directions` and creates twice that many environments, while `eval_ars.py` uses `--num-episodes`.

```bash
# Stage 1: ARS; 32 directions use 64 parallel environments (default is 512)
python scripts/train_ars.py Unitree-Go2-JumpInPlace-ARS --n-directions 32 --top-b 8
# Stage 2a: warm start from ARS
python scripts/train.py Unitree-Go2-JumpInPlace-Imitate \
  --env.actions.sagittal-pd.acrobatics.ars-policy-path=PATH/ars_policy_best.pt \
  --env.scene.num-envs=64
# Stage 2b: resume the imitation checkpoint with dense rewards
python scripts/train.py Unitree-Go2-JumpInPlace-Refine --agent.resume \
  --agent.load-run=RUN_NAME --agent.load-checkpoint=CHECKPOINT \
  --env.scene.num-envs=64
# Evaluate the selected ARS checkpoint on 64 episodes (repeat for the forward task)
python scripts/eval_ars.py --policy-file PATH/ars_policy_best.pt \
  --task-id Unitree-Go2-JumpInPlace-ARS --num-episodes 64
```

Replace `JumpInPlace` with `JumpForward` for the forward motion. Stage 1 exposes direction count, top directions, perturbation scale, learning rate, iterations, and evaluation cadence as CLI options; 32 directions evaluate 64 policies/environments. Episode length, PD gains, landing scaling, filters, reward weights, and phase thresholds live in `src/tasks/acrobatics/cfg.py`. PPO uses 2x64 tanh networks with fixed learning rate. Imitation needs the ARS checkpoint path; refinement resumes from the Stage 2a run. Stage 1's parser takes the task ID positionally and has no separate env-count flag.

Track `h_bar`, standing-relative height gain, `d_bar`, `theta_bar`, peak contact force, airborne steps, early termination causes, and each named dense reward. Two 500-iteration ARS runs did not meet Gate 3: in-place jump success was 54.7% on a fresh 64-episode evaluation, and forward distance success was 26.6%. Gate 4's long imitation/refinement evaluation and 256-episode comparison remain unverified; only one-iteration PPO smoke runs passed. Soft springs and domain randomization are gated on Gate 4 and are not implemented. In-place terminal height uses gain above standing height so a standing timeout scores zero in the ideal settled state; reset settling can produce a small positive score. Crash penalties were increased after the first ARS run showed positive reward for high-then-crash episodes. Go2 standing height was measured at about 0.224 m; controller gains were raised after the original gains failed the standing check. Hardware use requires state estimation or motion capture for height and vertical velocity; no deployment changes are included.

# Acrobatics implementation report

## Result

The rigid Go2 acrobatics package, ARS training/play scripts, three-stage task registrations for in-place and forward jumps, smoke tests, and usage documentation are implemented. The deliverable has passed code-path and environment smoke verification. The full plan's learning gates are **not complete**: a short ARS smoke run does not demonstrate a learned jump, and one PPO iteration does not establish imitation/refinement performance. Following the plan's gate ordering, soft springs, domain randomization, back-flips, and pronking were not implemented.

## Verification

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./.venv/bin/python -m pytest -q tests/acrobatics` — **15 passed**. Covers pure reward/ARS/mapping/phase tests and simulator standing, finite-action, scripted-jump, terminal-reward, and Stage 2 environment smoke paths.
- Flat-task regression: `scripts/train.py Unitree-Go2-Flat --env.scene.num-envs=64 --agent.max-iterations=2` — completed with exit code 0.
- ARS launch smoke: one iteration, two directions/top one (four environments), wrote best/last policy checkpoints; perturbed return mean `0.98253`, max `1.02109`. This is not the Gate 3 success evaluation.
- PPO launch smoke: one imitation iteration at 64 env using the ARS checkpoint, then one refine iteration at 64 env — both completed and logged rewards/checkpoints. This is not the Gate 4 evaluation.
- In-place ARS 500-iteration run at 64 env — best training eval return `0.119`, maximum in-run jump success `46.9%`; final eval success `18.8%`. Fresh 64-episode evaluation of the saved best-return checkpoint: mean height `0.364 m`, mean displacement `0.121 m`, jump success `54.7%`, early termination `45.3%` (all illegal-contact except none due fall/tilt).
- Forward ARS 500-iteration run at 64 env — best mean displacement `0.224 m`, best generic flight success `51.6%`; final mean displacement `0.205 m`, final flight success `34.4%`. Fresh 64-episode evaluation of the saved best-return checkpoint: mean height `0.332 m`, mean displacement `0.138 m`, distance success at ≥0.3 m `26.6%`, early termination `75.0%` (73.4% illegal contact, 1.6% tilt).
- `scripts/eval_ars.py` independently evaluated each saved policy on exactly 64 environments. It rejects values over 500.
- Stage 2 imitation and refine each constructed and stepped in one-env smoke tests.
- All environment counts used during tests were 64 or fewer. No test exceeded the user's 500-environment limit.

## Gate status

| Gate | Status | Evidence / remaining work |
|---|---|---|
| 0 Recon | Passed | Findings in repository `NOTES.md`; flat task ran 2 iterations at 64 env. |
| 1 Pure math | Passed | Pure tests are included in the 14-pass suite. |
| 2 Environment sanity | Passed with deviation | Stand, zero/random, scripted jump, replay and terminal branches checked. GUI playback not run. A short timeout scores about 0.13–0.16 from reset settling overshoot. |
| 3 ARS learning | Not passed | Both motions completed 500 iterations at 64 env. Fresh 64-episode evals of best-return checkpoints: in-place jump success 54.7%; forward distance success 26.6% at ≥0.3 m. Required threshold is 80%. |
| 4 PPO learning | Not passed | Only one iteration per stage; no 150-iteration imitation or 256-episode comparison. |
| 5 Soft / randomization | Not started | Explicitly gated on Gate 4. |
| 6 Back-flip / pronking | Not started | Optional and gated on prior stages. |

The full implementation objective remains incomplete because Gate 3 did not pass. Per the instructions' gate rule, Stage 2 long-run training, soft/randomized training, and optional motions were not started after these failures. Stage 2 code paths have smoke coverage only.

## Deviations and notes

Every implementation deviation and recon detail is recorded in repository [`NOTES.md`](../../NOTES.md); user-facing workflow and known limits are in [`doc/acrobatics.md`](../../doc/acrobatics.md). Main deviations:

1. Raised per-joint PD gains to hip 40/2, thigh 50/3, calf 80/4 after the copied velocity gains failed the standing test. Measured settled height is about 0.224 m.
2. The plan's literal `h_bar / h_max` gives a standing robot nonzero jump score. The implementation normalizes height gain above `h_stand` instead; reset settling can still produce a small positive terminal value. The live test bounds it to `[0,1]` and tests failure versus timeout branches.
3. No GUI viewer or two-stage convenience wrapper was added; simulator manager paths and separate documented commands cover smoke and training workflows.
4. Pytest was absent from the venv and installed from the local cache; plugin autoload must be disabled because the installed AnyIO plugin conflicts with the old pytest version.

Existing user edits in `scripts/train.py` and `src/tasks/velocity/rl/runner.py` were present and left untouched. Python import also changed a tracked `__pycache__` bytecode file; it is unrelated to the implementation and should be restored by the repository owner if desired.

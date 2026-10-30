# Reward V3 Implementation Report

Date: 2026-10-29

## Implemented changes

- Applied all 15 entries from `agent/reward_v3.yaml`.
- Corrected the YAML indentation of `stand_still_penalty` and made scientific
  notation scales parse as numbers.
- Replaced the stock tracking adapters with exact v3 kernels:
  - planar XY linear-velocity tracking, scale `1.5`;
  - yaw-only angular-velocity tracking, scale `0.75`.
- Added `feet_stance_time`, scale `-0.05`. It sums only contact time beyond
  `stance_limit=1.25 s` for each foot.
- Added `stand_still_penalty`, scale `-1.0`. It activates only when planar
  command magnitude exceeds `0.1 m/s`, using `sigma_still=0.1`.
- Changed `survival_bonus` itself to multiply the saturating alive-time value
  by the exact planar tracking-quality kernel (`std=0.5`, equivalent
  denominator `sigma_lin=0.25`). The separate wrapper applies only the ramp.
- Retained all legacy reward terms omitted by v3 at weight zero.
- Automatic run graphs include both new reward terms through
  `all_reward_terms.png` and `all_scalars.csv`.

## Verification

- `agent/reward_v3.yaml` parsed successfully with 15 terms.
- Seven configuration/unit tests passed, including scale mapping and numerical
  checks for both new reward functions.
- Python compilation passed for reward and environment modules.
- A two-iteration CPU smoke test with four environments passed:
  - all 25 retained/new reward terms initialized;
  - both v3 terms executed and were logged separately;
  - PPO completed 192 environment samples;
  - nine post-training graph/CSV artifacts were generated automatically.

Smoke-test run:

```text
logs/rsl_rl/go2_masq/2026-10-29_21-01-52
```

## Notes

- `feet_stance_time` is expected to remain zero until a foot has stayed in
  contact longer than 1.25 seconds; a two-iteration smoke test is too short to
  exercise that overdue region naturally.
- `stand_still_penalty` approaches one only for nearly frozen joints under a
  live movement command. Random initial policies generally move enough for the
  exponential value to be near zero.
- A longer controlled training run is still required to determine whether v3
  improves visual gait quality and reduces the freeze exploit.

## results:
Resulted in reward hacking
- stay survival penality scaled by veocity tracking didnt help: improvement would be to penalise it when it dosent move
- possible regularization may be making it find a reward plateau and not encourageing exploration
- The kernel is bounded in (0, 1], so it never goes negative. The threshold penalty from the spec was never added, so standing under an active command still earns positive reward:Command 0.5 m/s, robot standing: error = 0.25, reward = exp(-1) ≈ +0.37 per step.That’s the reason the policy has no incentive to walk. The tracked-command case gives about 1.0, but standing still is still well above zero.
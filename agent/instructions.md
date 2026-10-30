# Task5: MAKING IT COLLABORATE

## Goal
Making the MASQ (MARL) agents collaborate with each other for good gait. So far due to the rewad structure which encourages movement and staying alive, disourages feet not moving. Now they have to collaborate for good gait.

## Issues
The progress focused reward system has forced the quadruped to move, but thie issue is it launches all 4 legs at the same time, Hence making it fall. Now this issues is keeping this focus on movement and focusing on collaboration bw legs.

Looking at graphs:
(/marl_quadruped/logs/rsl_rl/go2_masq/V4/plots/survival_and_tracking.png)

(/marl_quadruped/logs/rsl_rl/go2_masq/V4/plots/episode_performance.png)

- as designed tracking is -ve and not only +ve which is desired 
- Episode mean reward has gone up (tho its -ve) while mean episode length goes down. which is not desired.
- One interesting thing is feet_stance_time penalty is 0. Hence this encouraged it to not plant it feet for long hence making it launch itself. This has to be adressed. 
- Major challenge is understanding the what the whole reward structure will cumulatively affect the performance before training.

## Requirements
1. recomend a reward_v5 which solves all the listed issues abouve. 
2. add a Gait reward 
3. add regularization rewards for jerky movement
4. plan a reward structure that is encourages collaboration bw legs for gait
5. Finally give a report on the reward structure listing pros and cons


## Workflow
1. Inspect the relevant code, tests, and project documentation.
2. Identify the implementation approach and any important risks.
3. Implement the smallest complete solution.
4. Add or update tests for the new behavior and relevant edge cases.
5. Run the appropriate tests, linters, and type checks.
6. Review the diff for regressions, unrelated changes, and incomplete requirements.

If a major ambiguity or architectural decision could materially affect the result, ask before proceeding. Otherwise, make reasonable assumptions and state them.

## Acceptance criteria
- running code base, for gathering experiments 

## Final report 
Summarize:
- In agent/ make a md file for summary after you are done
- What changed and why.
- Important implementation decisions.
- Tests and commands actually run, with their results.
- Known limitations or unresolved issues.
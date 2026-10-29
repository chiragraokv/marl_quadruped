# Task2: MAKING IT WORK

## Goal
Getting a visually appealing gait traind usig multi agent single quadruped architecture

## Issue
Previously we tried to train the multi agent single quadruped, but due to reward hacking or some other issues the gait was looking very bad. So i visually evaluated the gait in sim and will give some advice on reward tweaks.

## Requirements
1. Tweak the rewards as per recomendatoion, add a agent/reward_v1.yaml with these info:
    ```
    - id: track_lin_vel
    name: Tracking Linear Velocity
    type: task
    scale: 1.0
    equation: exp(-||v_cmd - v_b||^2 / sigma_t)
    range: (0, 1]
    notes: Exponential kernel; maximal when v_b equals v_cmd.
    ```
2. update the train.sh and play.sh with proper relative paths such that it runs on both gpu device and cpu device
3. Produce a FINAL REPORT at the end of updates
4. add wandb loggging for saving config and log the rewards (both seperate and combined reward)

## proposed changes in reward structure:
From velocity/mdp/rewards add
1. body_orientation_l2
2. self_collision
3. body_angular_penalty 
4. angular_momentum_penalty

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
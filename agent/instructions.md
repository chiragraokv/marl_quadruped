# Task4: MAKING IT WORK

## Goal
Getting a visually appealing gait traind usig multi agent single quadruped architecture

## Issue
Previous rewards lead to reward hacking where agent found a reward plateau which was rewarding substantial for standing still and not trying to fall. survival_bonus reward was 1 towards the end which means it has learned to walk while staying alive but it didnt learn it. 

## Requirements
1. Tweak the rewards as per recomendatoion in agent/reward_v4.yaml
2. the rewards have to be added to masq task and scaled appropriately otherwise its useless
3. The quadruped has to learn to live(not fall) while learning to walk.


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
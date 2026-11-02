"""Compact PPO settings for acrobatics imitation and refinement."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def acrobatics_ppo_runner_cfg(stage: str = "refine") -> RslRlOnPolicyRunnerCfg:
  # Actor/critic input sizes are inferred from the environment's observation groups.
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(64, 64), activation="tanh", obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.2, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(64, 64), activation="tanh", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.1,
      entropy_coef=1e-3, num_learning_epochs=5, num_mini_batches=4,
      learning_rate=2e-4, schedule="fixed", gamma=0.999, lam=0.95,
      desired_kl=0.01, max_grad_norm=1.0,
    ),
    experiment_name=f"go2_acrobatics_{stage}", logger="tensorboard",
    save_interval=100, num_steps_per_env=128, max_iterations=10001,
  )

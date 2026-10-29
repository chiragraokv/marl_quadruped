from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

def unitree_go2_masq_ppo_runner_cfg():
  from src.tasks.velocity.config.go2.rl_cfg import unitree_go2_ppo_runner_cfg
  cfg = unitree_go2_ppo_runner_cfg()
  cfg.actor.class_name = "src.tasks.masq.rl.actor_model:MASQActorModel"
  cfg.critic.class_name = "rsl_rl.models:MLPModel"
  cfg.algorithm.class_name = "src.tasks.masq.rl.algorithm:MASQPPO"
  cfg.experiment_name = "go2_masq"
  # The MASQ runner supplies a compatibility adapter for newer wandb releases.
  # RSL-RL logs Reward/* terms separately and Train/mean_reward as the sum.
  cfg.logger = "wandb"
  cfg.wandb_project = "unitree-go2-masq"
  return cfg

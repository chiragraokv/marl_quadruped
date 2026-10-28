from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

def unitree_go2_masq_ppo_runner_cfg():
  from src.tasks.velocity.config.go2.rl_cfg import unitree_go2_ppo_runner_cfg
  cfg = unitree_go2_ppo_runner_cfg()
  cfg.actor.class_name = "src.tasks.masq.rl.actor_model:MASQActorModel"
  cfg.critic.class_name = "rsl_rl.models:MLPModel"
  cfg.algorithm.class_name = "src.tasks.masq.rl.algorithm:MASQPPO"
  cfg.experiment_name = "go2_masq"
  # rsl_rl 5.0.1's WandB adapter is incompatible with newer wandb releases
  # (it passes the removed Settings(start_method=...) argument).
  cfg.logger = "tensorboard"
  return cfg

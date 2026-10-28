from rsl_rl.algorithms import PPO
from rsl_rl.algorithms.ppo import resolve_callable, resolve_obs_groups
from .rollout_storage import MASQRolloutStorage

class MASQPPO(PPO):
  def update(self):
    return super().update()

  def process_env_step(self, obs, rewards, dones, extras):
    # The environment emits one scalar reward per body; MASQ stores one
    # advantage/value per leg, so broadcast before rsl_rl bootstrapping.
    if rewards.ndim == 1:
      rewards = rewards.unsqueeze(-1).expand(-1, 4)
    return super().process_env_step(obs, rewards, dones, extras)

  @staticmethod
  def construct_algorithm(obs, env, cfg, device):
    alg_class = resolve_callable(cfg["algorithm"].pop("class_name"))
    actor_class = resolve_callable(cfg["actor"].pop("class_name"))
    critic_class = resolve_callable(cfg["critic"].pop("class_name"))
    cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
    # rsl_rl 5.0.1's Logger expects this optional key even when RND is unused.
    cfg["algorithm"].setdefault("rnd_cfg", None)
    actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
    cfg["algorithm"].pop("share_cnn_encoders", None)
    critic = critic_class(obs, cfg["obs_groups"], "critic", 4, **cfg["critic"]).to(device)
    # Our recurrent generator yields unpadded full trajectories; prevent the
    # stock MLP critic from applying its padded-trajectory unflattening helper.
    critic.is_recurrent = True
    storage = MASQRolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device, value_dim=4)
    return alg_class(actor, critic, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"])

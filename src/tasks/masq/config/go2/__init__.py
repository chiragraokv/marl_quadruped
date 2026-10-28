from mjlab.tasks.registry import register_mjlab_task
from src.tasks.masq.rl.runner import MASQOnPolicyRunner
from src.tasks.masq.masq_env_cfg import make_masq_env_cfg
from .rl_cfg import unitree_go2_masq_ppo_runner_cfg

register_mjlab_task("Unitree-Go2-MASQ-Flat", make_masq_env_cfg(),
                    make_masq_env_cfg(play=True),
                    unitree_go2_masq_ppo_runner_cfg(), MASQOnPolicyRunner)

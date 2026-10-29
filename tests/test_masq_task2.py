from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
import yaml

from src.tasks.masq.config.go2.rl_cfg import unitree_go2_masq_ppo_runner_cfg
from src.tasks.masq.masq_env_cfg import make_masq_env_cfg
from src.tasks.masq.mdp import rewards as masq_rewards
from src.tasks.velocity.mdp import rewards as velocity_rewards


class TestMasqTask2(unittest.TestCase):
  def test_reward_v1_documents_required_velocity_term(self):
    reward_file = Path(__file__).parents[1] / "agent" / "reward_v1.yaml"
    data = yaml.safe_load(reward_file.read_text())
    terms = {term["id"]: term for term in data["rewards"]}
    self.assertEqual(terms["track_lin_vel"], {
      "id": "track_lin_vel",
      "name": "Tracking Linear Velocity",
      "type": "task",
      "scale": 1.0,
      "equation": "exp(-||v_cmd - v_b||^2 / sigma_t)",
      "range": "(0, 1]",
      "notes": "Exponential kernel; maximal when v_b equals v_cmd.",
    })

  def test_masq_stability_rewards_and_wandb_are_enabled(self):
    env_cfg = make_masq_env_cfg()
    expected = {
      "body_orientation_l2": -1.0,
      "self_collisions": 0.0,
      "body_ang_vel": 0.0,
      "angular_momentum": 0.0,
      "base_height": -10.0,
      "feet_air_time": 1.0,
    }
    self.assertEqual(
      {name: env_cfg.rewards[name].weight for name in expected}, expected
    )
    self.assertTrue(
      any(sensor.name == "self_collision" for sensor in env_cfg.scene.sensors)
    )

    runner_cfg = unitree_go2_masq_ppo_runner_cfg()
    self.assertEqual(runner_cfg.logger, "wandb")
    self.assertEqual(runner_cfg.wandb_project, "unitree-go2-masq")

  def test_stability_rewards_are_explicitly_reexported_by_masq(self):
    self.assertIs(masq_rewards.body_orientation_l2,
                  velocity_rewards.body_orientation_l2)
    self.assertIs(masq_rewards.self_collision_cost,
                  velocity_rewards.self_collision_cost)
    self.assertIs(masq_rewards.body_angular_velocity_penalty,
                  velocity_rewards.body_angular_velocity_penalty)
    self.assertIs(masq_rewards.angular_momentum_penalty,
                  velocity_rewards.angular_momentum_penalty)

  def test_reward_scale_dictionary_controls_every_environment_weight(self):
    env_cfg = make_masq_env_cfg()
    configured = {name: term.weight for name, term in env_cfg.rewards.items()}
    self.assertEqual(configured, masq_rewards.MASQ_REWARD_SCALES)

  def test_reward_v2_scales_and_omitted_rewards(self):
    # Check coverage without freezing tunable dictionary values. The separate
    # dictionary-to-environment test proves edits propagate to the task.
    expected_terms = {
      "track_linear_velocity", "track_angular_velocity",
      "body_orientation_l2", "lin_vel_z", "ang_vel_xy", "base_height",
      "feet_air_time", "feet_slip", "collision", "torques", "dof_acc",
      "action_rate", "episode_time",
    }
    self.assertTrue(expected_terms <= set(masq_rewards.MASQ_REWARD_SCALES))

  def test_episode_time_penalty_has_configurable_500_iteration_warmup(self):
    env = SimpleNamespace(
      common_step_counter=500 * 24 - 1,
      episode_length_buf=torch.tensor([0, 25, 100]),
      step_dt=0.02,
    )
    before = masq_rewards.episode_time_penalty(env)
    self.assertTrue(torch.equal(before, torch.zeros(3)))

    env.common_step_counter = 500 * 24
    after = masq_rewards.episode_time_penalty(env)
    expected = torch.exp(-env.episode_length_buf.float() * env.step_dt)
    self.assertTrue(torch.allclose(after, expected))


if __name__ == "__main__":
  unittest.main()

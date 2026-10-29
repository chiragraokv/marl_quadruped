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
      "feet_air_time", "feet_stance_time", "stand_still_penalty",
      "feet_slip", "collision", "torques", "dof_acc", "action_rate",
      "survival",
    }
    self.assertTrue(expected_terms <= set(masq_rewards.MASQ_REWARD_SCALES))

  def test_survival_bonus_and_warmup_ramp(self):
    env = SimpleNamespace(
      common_step_counter=0,
      episode_length_buf=torch.tensor([0, 25, 100]),
      step_dt=0.02,
      extras={"log": {}},
      command_manager=SimpleNamespace(
        get_command=lambda _: torch.zeros(3, 3)
      ),
      scene={
        "robot": SimpleNamespace(data=SimpleNamespace(
          root_link_lin_vel_b=torch.zeros(3, 3)
        ))
      },
    )
    raw = masq_rewards.survival_bonus(env)
    expected = 1.0 - torch.exp(-env.episode_length_buf.float() * env.step_dt)
    self.assertTrue(torch.allclose(raw, expected))
    self.assertTrue(torch.equal(masq_rewards.survival_reward(env), torch.zeros(3)))

    env.common_step_counter = 100 * 24
    half = masq_rewards.survival_reward(env)
    self.assertTrue(torch.allclose(half, expected * 0.5))
    self.assertAlmostEqual(env.extras["log"]["Metrics/survival_ramp"].item(), 0.5)

    env.common_step_counter = 200 * 24
    full = masq_rewards.survival_reward(env)
    self.assertTrue(torch.allclose(full, expected))

    env.command_manager = SimpleNamespace(
      get_command=lambda _: torch.ones(3, 3)
    )
    gated = masq_rewards.survival_bonus(env)
    expected_gate = torch.exp(torch.tensor(-8.0))  # XY error 2 / std^2 .25.
    self.assertTrue(torch.allclose(gated, expected * expected_gate))

  def test_reward_v3_new_terms_and_scales(self):
    data = yaml.safe_load(
      (Path(__file__).parents[1] / "agent" / "reward_v3.yaml").read_text()
    )
    yaml_scales = {term["id"]: term["scale"] for term in data["rewards"]}
    aliases = {
      "track_lin_vel_xy": "track_linear_velocity",
      "track_ang_vel_z": "track_angular_velocity",
      "body_orientation": "body_orientation_l2",
      "undesired_contact": "collision",
    }
    for yaml_name, scale in yaml_scales.items():
      config_name = aliases.get(yaml_name, yaml_name)
      self.assertEqual(masq_rewards.MASQ_REWARD_SCALES[config_name], scale)

    sensor = SimpleNamespace(data=SimpleNamespace(
      current_contact_time=torch.tensor([[0.5, 1.25, 1.5, 2.0]])
    ))
    robot = SimpleNamespace(data=SimpleNamespace(
      joint_vel=torch.zeros(2, 12),
      root_link_lin_vel_b=torch.zeros(2, 3),
      root_link_ang_vel_b=torch.zeros(2, 3),
    ))
    commands = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    env = SimpleNamespace(
      scene={"feet_ground_contact": sensor, "robot": robot},
      command_manager=SimpleNamespace(get_command=lambda _: commands),
    )
    self.assertTrue(torch.allclose(
      masq_rewards.feet_stance_time(env), torch.tensor([1.0])
    ))
    self.assertTrue(torch.allclose(
      masq_rewards.stand_still_penalty(env), torch.tensor([1.0, 0.0])
    ))


if __name__ == "__main__":
  unittest.main()

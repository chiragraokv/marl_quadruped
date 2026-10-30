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

  def test_masq_v5_rewards_and_local_logging_are_enabled(self):
    env_cfg = make_masq_env_cfg()
    expected = {
      "body_orientation_l2": -1.0,
      "base_height": -10.0,
      "gait_contact_match": 1.0,
      "support_deficit": -0.5,
      "all_air_nonpronk": -3.0,
      "fall_termination": -200.0,
    }
    self.assertEqual(
      {name: env_cfg.rewards[name].weight for name in expected}, expected
    )
    self.assertTrue(
      any(sensor.name == "self_collision" for sensor in env_cfg.scene.sensors)
    )

    runner_cfg = unitree_go2_masq_ppo_runner_cfg()
    self.assertEqual(runner_cfg.logger, "tensorboard")

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
    self.assertEqual(
      configured,
      {name: masq_rewards.MASQ_REWARD_SCALES[name] for name in configured},
    )

  def test_reward_v5_yaml_scales_and_aliases(self):
    data = yaml.safe_load(
      (Path(__file__).parents[1] / "agent" / "reward_v5.yaml").read_text()
    )
    yaml_scales = {
      term["id"]: term["scale"] for term in data["rewards"] if term["enabled"]
    }
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
      found=torch.tensor([[1, 0, 0, 1]])
    ))
    robot = SimpleNamespace(data=SimpleNamespace(
      root_link_lin_vel_b=torch.tensor([[1., 0., 0.]]),
    ))
    commands = torch.tensor([[1.0, 0.0, 0.0]])
    gait = SimpleNamespace(
      gait_id=torch.tensor([1]), phase=torch.tensor([[0.25]]),
      phase_offsets=torch.tensor([[0., .5, .5, 0.]]),
    )
    env = SimpleNamespace(
      device=torch.device("cpu"), extras={"log": {}},
      scene={"feet_ground_contact": sensor, "robot": robot},
      command_manager=SimpleNamespace(
        get_command=lambda _: commands, get_term=lambda _: gait
      ),
    )
    self.assertTrue(torch.allclose(
      masq_rewards.gait_contact_match(env), torch.tensor([1.0])
    ))
    self.assertTrue(torch.allclose(
      masq_rewards.support_deficit(env), torch.tensor([0.0])
    ))
    self.assertTrue(torch.allclose(
      masq_rewards.all_air_nonpronk(env), torch.tensor([0.0])
    ))

    sensor.data.found.zero_()
    self.assertTrue(torch.allclose(
      masq_rewards.support_deficit(env), torch.tensor([1.0])
    ))
    self.assertTrue(torch.allclose(
      masq_rewards.all_air_nonpronk(env), torch.tensor([1.0])
    ))


if __name__ == "__main__":
  unittest.main()

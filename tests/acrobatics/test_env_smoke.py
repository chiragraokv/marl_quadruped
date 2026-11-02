import torch

import src.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from src.tasks.acrobatics.mdp.sagittal import home_action


def test_go2_acrobatics_env_steps_and_stands():
  cfg = load_env_cfg("Unitree-Go2-JumpInPlace-ARS")
  cfg.scene.num_envs = 1
  cfg.events["reset_base"].params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "yaw": (0.0, 0.0)}
  cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (0.0, 0.0)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0" if torch.cuda.is_available() else "cpu")
  try:
    obs, _ = env.reset()
    assert obs["actor"].shape == (1, 15)
    action = home_action(device=env.device).unsqueeze(0)
    h_stand = cfg.actions["sagittal_pd"].acrobatics.h_stand
    saw_done = False
    for _ in range(149):
      obs, reward, terminated, time_out, _ = env.step(action)
      assert torch.isfinite(obs["actor"]).all()
      assert torch.isfinite(reward).all()
      saw_done |= bool(terminated.any())
    final_h = env.scene["robot"].data.root_link_pos_w[0, 2].item()
    assert not saw_done
    assert abs(final_h - h_stand) <= 0.01
  finally:
    env.close()


def test_zero_and_random_actions_remain_finite():
  for random_policy in (False, True):
    cfg = load_env_cfg("Unitree-Go2-JumpInPlace-ARS")
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0" if torch.cuda.is_available() else "cpu")
    try:
      obs, _ = env.reset()
      for _ in range(25):
        action = torch.rand((1, 6), device=env.device) * 2 - 1 if random_policy else torch.zeros((1, 6), device=env.device)
        obs, reward, terminated, timeout, _ = env.step(action)
        assert torch.isfinite(obs["actor"]).all()
        assert torch.isfinite(reward).all()
    finally:
      env.close()


def test_open_loop_jump_visits_takeoff_flight_landing():
  cfg = load_env_cfg("Unitree-Go2-JumpInPlace-ARS")
  cfg.scene.num_envs = 1
  cfg.events["reset_base"].params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "yaw": (0.0, 0.0)}
  cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (0.0, 0.0)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0" if torch.cuda.is_available() else "cpu")
  try:
    env.reset()
    home = home_action(device=env.device)
    extend = home.clone()
    extend[1], extend[2], extend[4], extend[5] = 0.2, 0.3, -0.25, 0.3
    phases = []
    heights = []
    for step in range(40):
      action = extend if step < 4 else home
      _, reward, terminated, _, _ = env.step(action.unsqueeze(0))
      assert torch.isfinite(reward).all()
      phases.append(int(env.acrobatics_state.phase.phase[0]))
      heights.append(float(env.scene["robot"].data.root_link_pos_w[0, 2]))
      if terminated.any():
        break
    collapsed = list(dict.fromkeys(phases))
    assert collapsed[:3] == [0, 1, 2]
    assert max(heights) > cfg.actions["sagittal_pd"].acrobatics.h_stand
    assert abs(env.acrobatics_state.h_bar[0].item() - max(heights)) < 0.02
  finally:
    env.close()


def test_ppo_stage_reward_configs_step(tmp_path):
  for task, stage in (("Unitree-Go2-JumpInPlace-Imitate", "imitate"),
                      ("Unitree-Go2-JumpForward-Refine", "refine")):
    cfg = load_env_cfg(task)
    cfg.scene.num_envs = 1
    if stage == "imitate":
      from src.tasks.acrobatics.mdp.sagittal import home_action
      policy_path = tmp_path / "ars_policy.pt"
      torch.save({"W": torch.zeros(6, 15), "b": home_action(), "mu": torch.zeros(15),
                  "sd": torch.ones(15), "obs_dim": 15, "act_dim": 6}, policy_path)
      acro = cfg.actions["sagittal_pd"].acrobatics
      acro.ars_policy_path = str(policy_path)
      cfg.rewards["imitation"].params["policy_path"] = str(policy_path)
      cfg.rewards["imitation"].params["acrobatics"] = acro
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0" if torch.cuda.is_available() else "cpu")
    try:
      obs, _ = env.reset()
      assert obs["actor"].shape == (1, 16)
      assert obs["critic"].shape == (1, 23)
      action = torch.zeros((1, 6), device=env.device)
      _, reward, _, _, _ = env.step(action)
      assert torch.isfinite(reward).all()
    finally:
      env.close()


def test_terminal_reward_uses_same_step_termination_and_timeout():
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cfg = load_env_cfg("Unitree-Go2-JumpInPlace-ARS")
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    env.reset()
    root_state = env.scene["robot"].data.default_root_state.clone()
    root_state[:, 2] = 0.09
    env.scene["robot"].write_root_state_to_sim(root_state)
    env.sim.forward()
    _, reward, terminated, timeout, _ = env.step(torch.zeros((1, 6), device=device))
    assert terminated.item()
    assert not timeout.item()
    assert reward.item() < 0.0
  finally:
    env.close()

  cfg = load_env_cfg("Unitree-Go2-JumpInPlace-ARS")
  cfg.scene.num_envs = 1
  cfg.episode_length_s = 0.04
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    env.reset()
    action = home_action(device=env.device).unsqueeze(0)
    _, _, terminated, _, _ = env.step(action)
    _, reward, terminated, timeout, _ = env.step(action)
    assert timeout.item()
    assert not terminated.item()
    # Settling overshoot gives a small positive height score; it remains normalized.
    assert 0.0 <= reward.item() <= 1.0
  finally:
    env.close()

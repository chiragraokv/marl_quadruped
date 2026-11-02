"""Shared, serializable configuration for Go2 acrobatics experiments."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class AcrobaticsCfg:
  motion: Literal["in_place", "forward"] = "in_place"
  stage: Literal["ars", "imitate", "refine"] = "ars"
  soft: bool = False
  randomize: bool = False
  repeat: bool = False
  episode_length_s: float = 3.0
  h_fall: float = 0.10
  tilt_threshold: float = 0.85
  contact_force_epsilon: float = 1.0
  airborne_debounce_steps: int = 1
  landing_settle_steps: int = 2
  lpf_alpha: float = 0.5
  landing_gain_scale: float = 0.5
  kp_hip: float = 40.0
  kd_hip: float = 2.0
  kp_thigh: float = 50.0
  kd_thigh: float = 3.0
  kp_calf: float = 80.0
  kd_calf: float = 4.0
  effort_hip: float = 23.5
  effort_thigh: float = 23.5
  effort_calf: float = 45.0
  sigma_h: float = 0.01
  sigma_hdot: float = 0.05
  sigma_pitch: float = 0.01
  sigma_q: float = 0.01
  sigma_qdot: float = 0.05
  h_stand: float = 0.224
  h_max: float = 0.524
  d_max: float = 1.3
  forward_success_distance: float = 0.3
  w_a: float = 1.0
  w_b: float = 4.0
  imitation_iterations: int = 150
  ars_policy_path: str = ""
  ars_directions: int = 256
  ars_top_b: int = 64
  ars_nu: float = 0.05
  ars_alpha: float = 0.02
  ars_iterations: int = 500
  spring_k: float = 8.0
  spring_c: float = 0.1
  paper_com_offset: bool = False
  sparse_theta_amplitude: float = 1.0
  sparse_theta_decay: float = 1.0
  sparse_height_bias: float = 0.0
  sparse_distance_amplitude: float = 1.0
  sparse_distance_decay: float = 1.0
  sparse_forward_distance_weight: float = 1.0
  sparse_success_bonus: float = 0.1
  sparse_failure_penalty: float = 1.2
  sparse_crash_scale: float = 1.0
  stage2_h_min: float = 0.194
  stage2_h_max: float = 1.3
  stage2_height_weight: float = 0.01
  stage2_contact_min: float = 800.0
  stage2_contact_weight: float = 1.5e-4
  foot_force_count_epsilon: float = 1.0
  illegal_contact_threshold: float = 10.0
  stage2_distance_amplitude: float = 6.5e-4
  stage2_distance_decay: float = 40.0
  stage2_forward_distance_weight: float = 1.52e-2
  stage2_smooth_weight: float = 3e-3
  stage2_smooth_decay: float = 0.1
  stage2_pitch_weight: float = 4.2e-3
  stage2_pitch_decay: float = 26.0
  stage2_failure_height_weight: float = 0.25
  stage2_forward_bonus_weight: float = 0.025
  reset_xy_noise: float = 0.01
  reset_height_noise: float = 0.01
  reset_pitch_noise: float = 0.02
  reset_yaw_noise: float = 0.02
  reset_joint_pos_noise: float = 0.02
  reset_joint_vel_noise: float = 0.02

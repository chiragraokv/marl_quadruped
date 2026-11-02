# THE EFFECTS OF REWARD MISSPECIFICATION: MAPPING AND MITIGATING MISALIGNED MODELS

Capable agents turn towards reward hacking
### Misspecifications
- Missweighing: corrective opjective wrong weight
- ontological: wrong representation of data
- scope: correct objective but restricted domain


### Experiments:

#### Ontological (Different quantity that correlates with intent)
- B1_speed_not_tracking: track_lin_vel_xy: "||v_b_xy||"         
- B2_airtime_raw: feet_air_time: "sum_i(t_air_i)" 

#### Misweighting
- A1_penalties_underweighted:  
    - body_orientation: -0.1
    - lin_vel_z: -0.2
    - ang_vel_xy: -0.005
    - base_height: -1.0
    - feet_slip: -0.01
    - undesired_contact: -0.1
    - action_rate: -0.001
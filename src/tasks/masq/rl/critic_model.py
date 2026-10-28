"""MASQ critic compatibility alias.

The stock deterministic MLP supports a four-dimensional output; MASQPPO
constructs it with output_dim=4, so no custom critic behavior is required.
"""
from rsl_rl.models import MLPModel

MASQCriticModel = MLPModel

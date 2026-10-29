"""Shared-parameter four-agent actor using the fold/unfold convention."""
import copy
import torch
from torch import nn

try:
  from rsl_rl.models import MLPModel, RNNModel
  from rsl_rl.modules.rnn import RNN
except ImportError:  # permits importing task metadata without RL extras
  MLPModel = RNNModel = RNN = object

class MASQActorModel(RNNModel):
  N_AGENTS = 4
  LEG_OBS = 35
  LEG_ACTIONS = 3

  def __init__(self, obs, obs_groups, obs_set, output_dim, *args, **kwargs):
    super().__init__(obs, obs_groups, obs_set, self.LEG_ACTIONS, *args, **kwargs)
    # RNNModel constructs its recurrent input width from the full 140D group;
    # MASQ folds first, so replace it with a 35D recurrent input.
    self.rnn = RNN(self.LEG_OBS, self.rnn.rnn.hidden_size, self.rnn.rnn.num_layers,
                   kwargs.get("rnn_type", "gru"))
    # rsl_rl's recurrent minibatch generator indexes hidden states per
    # environment; folded leg states require a dedicated generator. Keep the
    # rollout path feed-forward until that generator is implemented.
    self.is_recurrent = True

  def forward(self, obs, *args, **kwargs):
    # RSL-RL passes a TensorDict; get_latent performs the fold before the
    # shared MLP/GRU/distribution stack sees the observations.
    sequence = len(obs.batch_size) == 2
    if sequence:
      t, b = obs.batch_size
    else:
      t, b = 1, obs.batch_size[0]
    out = super().forward(obs, *args, **kwargs)
    return out.reshape(t, b, 12) if sequence else out.reshape(b, 12)

  def get_latent(self, obs, masks=None, hidden_state=None):
    latent = MLPModel.get_latent(self, obs)
    if latent.shape[-1] != self.N_AGENTS * self.LEG_OBS:
      raise ValueError(f"MASQ actor expects 140 observations, got {latent.shape[-1]}")
    sequence = latent.ndim == 3
    if sequence:
      t, b, _ = latent.shape
      folded = latent.reshape(t, b, self.N_AGENTS, self.LEG_OBS).permute(0, 1, 2, 3).reshape(t, b * self.N_AGENTS, self.LEG_OBS)
      if masks is not None:
        masks = masks.repeat_interleave(self.N_AGENTS, dim=1)
      if hidden_state is not None:
        hidden_state = hidden_state.clone()
    else:
      b = latent.shape[0]
      folded = latent.reshape(b * self.N_AGENTS, self.LEG_OBS)
    if sequence and hidden_state is not None:
      # Our storage provides unpadded full trajectories; bypass rsl_rl's
      # padding helper, which assumes one hidden state per environment rather
      # than four folded leg states.
      out, _ = self.rnn.rnn(folded, hidden_state)
    else:
      out = self.rnn(folded, masks, hidden_state)
    if not sequence:
      out = out.squeeze(0)
    # Truncate BPTT between simulator steps; PPO stores log-probs, not the
    # live autograd graph from the rollout.
    if masks is None and self.rnn.hidden_state is not None:
      self.rnn.hidden_state = self.rnn.hidden_state.detach()
    return out

  def get_output_log_prob(self, outputs):
    if outputs.ndim == 3:
      t, b, _ = outputs.shape
      logp = self.distribution.log_prob(outputs.reshape(t, b * 4, 3))
      return logp.reshape(t, b, 4)
    b = outputs.shape[0]
    logp = self.distribution.log_prob(outputs.reshape(b * 4, 3))
    return logp.reshape(b, 4)

  def reset(self, dones=None, hidden_state=None):
    # RNN state is folded as [env0-leg0..env0-leg3, env1-leg0..].
    if dones is not None:
      dones = dones.repeat_interleave(self.N_AGENTS)
    return super().reset(dones, hidden_state)

  @property
  def output_entropy(self):
    return self.distribution.entropy.reshape(-1, 4)

  @property
  def output_distribution_params(self):
    params = []
    for p in self.distribution.params:
      # Distribution parameters may retain the recurrent [T,B*4,...] shape.
      if p.ndim >= 3:
        p = p.reshape(-1, *p.shape[2:])
      params.append(p.reshape(p.shape[0] // 4, 4, *p.shape[1:]))
    return tuple(params)

  def get_kl_divergence(self, old_params, new_params):
    # Distribution.kl_divergence expects the same batch shape for old and new
    # parameters.  Rollout parameters can be [T, B, 4, D], while parameters
    # produced during the update may already be [B, 4, D].  Flatten time,
    # environments, and legs into one distribution batch in both cases.
    def flatten_params(params):
      return tuple(p.reshape(-1, p.shape[-1]) for p in params)

    flattened_old = flatten_params(old_params)
    flattened_new = flatten_params(new_params)

    # The recurrent MASQ update can evaluate the current policy on one
    # simulator-step batch while the rollout stores all T steps.  Match the
    # current distribution to the stored rollout batch before asking PyTorch
    # for KL; otherwise Normal distributions fail on [T*B*4, D] vs [B*4, D].
    aligned_new = []
    for old, new in zip(flattened_old, flattened_new):
      if old.shape[0] != new.shape[0]:
        if old.shape[0] % new.shape[0] != 0:
          raise ValueError(
            f"MASQ KL batch mismatch: old={old.shape}, new={new.shape}"
          )
        new = new.repeat(old.shape[0] // new.shape[0], 1)
      aligned_new.append(new)

    return self.distribution.kl_divergence(flattened_old, tuple(aligned_new))

  def as_onnx(self, verbose: bool):
    return _MASQOnnxActor(self, verbose)


class _MASQOnnxActor(nn.Module):
  """Stateless deterministic export: 140D observation -> 12D action."""
  is_recurrent = False

  def __init__(self, model, verbose=False):
    super().__init__()
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.mlp = copy.deepcopy(model.mlp)
    self.rnn = copy.deepcopy(model.rnn.rnn)
    self.deterministic_output = model.distribution.as_deterministic_output_module()
    self.input_size = model.obs_dim

  def forward(self, x):
    x = self.obs_normalizer(x)
    batch = x.shape[0]
    x = x.reshape(batch * 4, 35)
    h0 = torch.zeros(self.rnn.num_layers, batch * 4, self.rnn.hidden_size,
                     device=x.device, dtype=x.dtype)
    x, _ = self.rnn(x.unsqueeze(0), h0)
    x = self.deterministic_output(self.mlp(x.squeeze(0)))
    return x.reshape(batch, 12)

  def get_dummy_inputs(self):
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self):
    return ["obs"]

  @property
  def output_names(self):
    return ["actions"]

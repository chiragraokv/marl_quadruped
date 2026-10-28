import torch
from rsl_rl.storage import RolloutStorage

class MASQRolloutStorage(RolloutStorage):
  def __init__(self, *args, value_dim=4, **kwargs):
    super().__init__(*args, **kwargs)
    self.rewards = torch.zeros((*self.rewards.shape[:-1], value_dim), device=self.rewards.device)
    self.values = torch.zeros((*self.values.shape[:-1], value_dim), device=self.values.device)
    self.actions_log_prob = torch.zeros((*self.actions_log_prob.shape[:-1], value_dim), device=self.actions_log_prob.device)
    self.returns = torch.zeros((*self.returns.shape[:-1], value_dim), device=self.returns.device)
    self.advantages = torch.zeros((*self.advantages.shape[:-1], value_dim), device=self.advantages.device)
    self._masq_actor_hidden = None

  def add_transitions(self, transition):
    self.add_transition(transition)

  def add_transition(self, transition):
    if self.step >= self.num_transitions_per_env:
      raise OverflowError("MASQ rollout buffer overflow")
    self.observations[self.step].copy_(transition.observations)
    self.actions[self.step].copy_(transition.actions)
    reward = transition.rewards
    self.rewards[self.step].copy_(reward if reward.ndim == 2 else reward.unsqueeze(-1).expand(-1, self.rewards.shape[-1]))
    self.dones[self.step].copy_(transition.dones.view(-1, 1))
    self.values[self.step].copy_(transition.values)
    self.actions_log_prob[self.step].copy_(transition.actions_log_prob)
    if self.distribution_params is None:
      self.distribution_params = tuple(torch.zeros(self.num_transitions_per_env, *p.shape, device=self.device)
                                       for p in transition.distribution_params)
    for i, p in enumerate(transition.distribution_params):
      self.distribution_params[i][self.step].copy_(p)
    actor_hidden = transition.hidden_states[0]
    if actor_hidden is not None:
      if isinstance(actor_hidden, tuple):
        actor_hidden = actor_hidden[0]
      if self._masq_actor_hidden is None:
        self._masq_actor_hidden = torch.zeros(self.num_transitions_per_env, *actor_hidden.shape, device=self.device)
      self._masq_actor_hidden[self.step].copy_(actor_hidden)
    self.step += 1

  def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
    """Yield full time-major batches with hidden state folded per leg."""
    if self._masq_actor_hidden is None:
      raise RuntimeError("MASQ actor hidden states were not recorded")
    batch_size = self.num_envs // num_mini_batches
    masks_all = (~self.dones.squeeze(-1).bool())
    for _ in range(num_epochs):
      for i in range(num_mini_batches):
        start, stop = i * batch_size, (i + 1) * batch_size
        hidden = self._masq_actor_hidden[0, :, start * 4:stop * 4, :].contiguous()
        masks = masks_all[:, start:stop]
        yield RolloutStorage.Batch(
          observations=self.observations[:, start:stop],
          actions=self.actions[:, start:stop],
          values=self.values[:, start:stop],
          advantages=self.advantages[:, start:stop],
          returns=self.returns[:, start:stop],
          old_actions_log_prob=self.actions_log_prob[:, start:stop],
          old_distribution_params=tuple(p[:, start:stop] for p in self.distribution_params),
          hidden_states=(hidden, None), masks=masks,
        )

from enum import Enum

import dynet as dy
import numpy as np

from xnmt import events, losses, param_initializers
from xnmt.modelparts import transforms
from xnmt.persistence import Ref, bare, Serializable, serializable_init
from xnmt.reports import Reportable

class PolicyGradient(Serializable, Reportable):
  """
  (Sequence) policy gradient class. It holds a policy network that will perform a linear regression to the output_dim decision labels.
  This class works by calling the sample_action() that will sample some actions from the current policy, given the input state.

  Every time sample_action is called, the actions and the softmax are being recorded and then the network can be trained by passing the reward
  to the calc_loss function.
  
  Depending on the passed flags, currently it supports the calculation of additional losses of:
  - baseline: Linear regression that predicts future reward from the current input state
  - conf_penalty: The confidence penalty loss. Good when policy network are too confident.
  
  Args:
    sample: The number of samples being drawn from the policy network.
    use_baseline: Whether to turn on baseline reward discounting or not.
    weight: The weight of the reinforce_loss.
    output_dim: The size of the predicitions.
  """
  yaml_tag = '!PolicyGradient'
  @serializable_init
  @events.register_xnmt_handler
  def __init__(self,
               policy_network=None,
               baseline=None,
               z_normalization=True,
               conf_penalty=None,
               input_dim=Ref("exp_global.default_layer_dim"),
               output_dim=2,
               param_init=Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init=Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer))):
    self.input_dim = input_dim
    self.policy_network = self.add_serializable_component("policy_network",
                                                           policy_network,
                                                          lambda: transforms.Linear(input_dim=self.input_dim,
                                                                                    output_dim=output_dim,
                                                                                    param_init=param_init,
                                                                                    bias_init=bias_init))
    self.baseline = self.add_serializable_component("baseline",
                                                    baseline,
                                                    lambda: transforms.Linear(input_dim=self.input_dim,
                                                                              output_dim=1,
                                                                              param_init=param_init,
                                                                              bias_init=bias_init))

    self.confidence_penalty = self.add_serializable_component("conf_penalty",
                                                              conf_penalty,
                                                              lambda: conf_penalty) if conf_penalty is not None else None
    self.z_normalization = z_normalization

  def sample_action(self, state, argmax=False, sample_pp=None, predefined_actions=None, valid_pos=None):
    """
    state: Input state.
    argmax: Whether to perform argmax or sampling.
    sample_pp: Stands for sample post_processing.
               Every time the sample are being drawn, this method will be invoked with sample_pp(sample).
    predefined_actions: Whether to forcefully the network to assign the action value to some predefined actions.
                        This predefined actions can be from the gold distribution or some probability priors.
                        It should be calculated from the outside.
    """
    policy = dy.log_softmax(self.policy_network.transform(state))

    # Select actions
    if predefined_actions is not None:
      actions = predefined_actions
    else:
      if argmax:
        actions = policy.tensor_value().argmax().as_numpy()[0]
      else:
        actions = policy.tensor_value().categorical_sample_log_prob().as_numpy()[0]
      if len(actions.shape) == 0:
        actions = [actions]
    # Post Processing
    if sample_pp is not None:
      actions = sample_pp(actions)
    
    # Return
    try:
      return actions
    finally:
      self.policy_lls.append(policy)
      self.actions.append(actions)
      self.states.append(state)
      self.valid_pos.append(valid_pos)

  def calc_loss(self, policy_reward, results={}):
    """
    Calc policy networks loss.
    """
    assert len(policy_reward) == len(self.states), "There should be a reward for every action taken"
    loss = losses.FactoredLossExpr()
    
    # Calculate the baseline loss of the reinforce loss for each timestep:
    # b = W_b * s + b_b
    # R = r - b
    # Also calculate the baseline loss
    # b = r_p (predicted)
    # loss_b = squared_distance(r_p - r_r)
    rewards = []
    baseline_loss = []
    for i, state in enumerate(self.states):
      r_p = self.baseline.transform(dy.nobackprop(state))
      rewards.append(policy_reward[i] - r_p)
      if self.valid_pos[i] is not None:
        r_p = dy.pick_batch_elems(r_p, self.valid_pos[i])
        r_r = dy.pick_batch_elems(policy_reward[i], self.valid_pos[i])
      else:
        r_r = policy_reward[i]
      baseline_loss.append(dy.squared_distance(r_p, r_r))
    loss.add_loss("rl_baseline", dy.esum(baseline_loss))
    
    # Z Normalization
    # R = R - mean(R) / std(R)
    rewards = dy.concatenate(rewards, d=0)
    r_dim = rewards.dim()
    if self.z_normalization:
      rewards_shape = dy.reshape(rewards, (r_dim[0][0], r_dim[1]))
      rewards_mean = dy.mean_elems(rewards_shape)
      rewards_std = dy.std_elems(rewards) + 1e-20
      rewards = (rewards - rewards_mean.value()) / rewards_std.value()
    rewards = dy.nobackprop(rewards)

    # Calculate Confidence Penalty
    if self.confidence_penalty:
      loss.add_loss("rl_confpen", self.confidence_penalty.calc_loss(self.policy_lls))
    
    # Calculate Reinforce Loss
    # L = - sum([R-b] * pi_ll)
    reinf_loss = []
    for i, (policy, action) in enumerate(zip(self.policy_lls, self.actions)):
      reward = dy.pick(rewards, i)
      ll = dy.pick_batch(policy, action)
      if self.valid_pos[i] is not None:
        ll = dy.pick_batch_elems(ll, self.valid_pos[i])
        reward = dy.pick_batch_elems(reward, self.valid_pos[i])
      reinf_loss.append(dy.sum_batches(dy.cmult(ll, reward)))
    loss.add_loss("rl_reinf", -dy.esum(reinf_loss))

    # Pack up + return
    try:
      return loss
    finally:
      if results is not None:
        results.update({
          "pg_loss": loss,
          "pg_policy_reward": policy_reward,
          "pg_rewards": rewards,
          "pg_policy_ll": self.policy_lls,
          "pg_actions": self.actions,
          "pg_valid_pos": self.valid_pos
        })

  def shared_params(self):
    return [{".input_dim", ".policy_network.input_dim"},
            {".input_dim", ".baseline.input_dim"}]

  @events.handle_xnmt_event
  def on_start_sent(self, src_sent):
    self.policy_lls = []
    self.actions = []
    self.states = []
    self.valid_pos = []

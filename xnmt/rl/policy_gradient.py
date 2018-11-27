from enum import Enum

import dynet as dy
import numpy as np

from xnmt import events, losses, param_initializers
from xnmt.modelparts import transforms
from xnmt.persistence import Ref, bare, Serializable, serializable_init

class PolicyGradient(Serializable):
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
  def __init__(self, policy_network=None,
                     baseline=None,
                     z_normalization=True,
                     conf_penalty=None,
                     weight=1.0,
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
    self.baseline = self.add_serializable_component("baseline", baseline,
                                                    lambda: transforms.Linear(input_dim=self.input_dim, output_dim=1,
                                                                              param_init=param_init,
                                                                              bias_init=bias_init))

    self.confidence_penalty = self.add_serializable_component("conf_penalty", conf_penalty, lambda: conf_penalty) if conf_penalty is not None else None
    self.weight = weight
    self.z_normalization = z_normalization

  """
  state: Input state.
  argmax: Whether to perform argmax or sampling.
  sample_pp: Stands for sample post_processing. Every time the sample are being drawn, this method will be invoked with sample_pp(sample).
  predefined_actions: Whether to forcefully the network to assign the action value to some predefined actions. This predefined actions can be 
                      from the gold distribution or some probability priors. It should be calculated from the outside.
  """
  def sample_action(self, state, argmax=False, sample_pp=None, predefined_actions=None):
    policy = dy.log_softmax(self.policy_network.transform(state))
    if predefined_actions is not None:
      # Use defined action value
      self.sampling_action = self.SamplingAction.PREDEFINED
      actions = predefined_actions
    else:
      # sample from policy
      sample = self.sample_from_policy(policy, argmax=argmax)
      if sample_pp is not None:
        sample = sample_pp(sample)
      actions = sample
    try:
      return actions
    finally:
      self.policy_lls.append(policy)
      self.actions.append(actions)
      self.states.append(state)

  """
  Calc policy networks loss.
  """
  def calc_loss(self, policy_reward):
    loss = losses.FactoredLossExpr()
    ## Calculate baseline
    pred_reward, baseline_loss = self.calc_baseline_loss(policy_reward)
    rewards = [policy_reward - pw_i for pw_i in pred_reward]
    loss.add_loss("rl_baseline", baseline_loss)
    ## Z-Normalization
    rewards = dy.concatenate(rewards, d=0)
    if self.z_normalization:
      rewards_value = rewards.value()
      rewards_mean = np.mean(rewards_value)
      rewards_std = np.std(rewards_value) + 1e-10
      rewards = (rewards - rewards_mean) / rewards_std
    ## Calculate Confidence Penalty
    if self.confidence_penalty:
      cp_loss = self.confidence_penalty.calc_loss(self.policy_lls)
      loss.add_loss("rl_confpen", cp_loss)
    ## Calculate Reinforce Loss
    reinf_loss = []
    # Loop through all action in one sequence
    for i, (policy, action) in enumerate(zip(self.policy_lls, self.actions)):
      # Main Reinforce calculation
      reward = dy.pick(rewards, i)
      ll = dy.pick_batch(policy, action)
      if self.valid_pos is not None:
        ll = dy.pick_batch_elems(ll, self.valid_pos[i])
        reward = dy.pick_batch_elems(reward, self.valid_pos[i])
      reinf_loss.append(dy.sum_batches(ll * reward))
    loss.add_loss("rl_reinf", -self.weight * dy.esum(reinf_loss))
    ## the composed losses
    return loss

  def shared_params(self):
    return [{".input_dim", ".policy_network.input_dim"},
            {".input_dim", ".baseline.input_dim"}]

  @events.handle_xnmt_event
  def on_start_sent(self, src_sent):
    self.policy_lls = []
    self.actions = []
    self.states = []
    self.baseline_input = None
    self.valid_pos = src_sent.mask.get_valid_position() if src_sent.mask else None 
    self.sampling_action = self.SamplingAction.NONE

  def sample_from_policy(self, policy, argmax=False):
    batch_size = policy.dim()[1]
    if argmax:
      action = policy.tensor_value().argmax().as_numpy()[0]
    else:
      action = policy.tensor_value().categorical_sample_log_prob().as_numpy()[0]
    if batch_size == 1:
      action = [action]
    try:
      return action
    finally:
      self.sampling_action = self.SamplingAction.POLICY_CLP if not argmax else self.SamplingAction.POLICY_AMAX

  def calc_baseline_loss(self, reward):
    pred_rewards = []
    cur_losses = []
    for i, state in enumerate(self.states):
      pred_reward = self.baseline.transform(dy.nobackprop(state))
      pred_rewards.append(dy.nobackprop(pred_reward))
      if self.valid_pos is not None:
        pred_reward = dy.pick_batch_elems(pred_reward, self.valid_pos[i])
        act_reward = dy.pick_batch_elems(reward, self.valid_pos[i])
      else:
        act_reward = reward
      cur_losses.append(dy.sum_batches(dy.squared_distance(pred_reward, dy.nobackprop(act_reward))))
    return pred_rewards, dy.esum(cur_losses)
  
  class SamplingAction(Enum):
    POLICY_CLP = 0
    POLICY_AMAX = 1
    PREDEFINED = 2
    NONE = 3


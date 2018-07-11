
import numpy as np
import dynet as dy

from xnmt.events import handle_xnmt_event, register_xnmt_handler
from xnmt.transform import Linear
from xnmt.persistence import Ref, bare, Path, Serializable, serializable_init
from xnmt.rl.eps_greedy import EpsilonGreedy
from xnmt.constants import EPSILON
from xnmt.loss import FactoredLossExpr
from xnmt.enum import Enum

class PolicyGradient(Serializable):
  yaml_tag = '!PolicyGradient'
  @serializable_init
  @register_xnmt_handler
  def __init__(self, policy_network=None,
                     baseline=None,
                     z_normalization=True,
                     conf_penalty=None,
                     sample=1,
                     weight=1.0):
    self.policy_network = self.add_serializable_component("policy_network", policy_network, lambda: policy_network)
    self.confidence_penalty = self.add_serializable_component("conf_penalty", conf_penalty, lambda: conf_penalty) if conf_penalty is not None else None
    self.baseline = self.add_serializable_component("baseline", baseline, lambda: baseline) if baseline is not None else None
    self.z_normalization = z_normalization
    self.sample = sample
    self.weight = weight
  
  @handle_xnmt_event
  def on_start_sent(self, src_sent):
    self.policy_lls = []
    self.actions = []
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

  def sample_action(self, state, argmax=False, sample_pp=None, predefined_actions=None):
    policy = dy.log_softmax(self.policy_network(state))
    actions = []
    if predefined_actions is not None:
      # Use defined action value
      self.sampling_action = self.SamplingAction.PREDEFINED
      actions = predefined_actions
    else:
      # sample from policy
      for k in range(self.sample):
        sample = self.sample_from_policy(policy, argmax=argmax)
        if sample_pp is not None:
          sample = sample_pp(sample)
        actions.append(sample)
        # only one sample during argmax
        if argmax:
          break
    try:
      return actions
    finally:
      self.policy_lls.append(policy)
      self.actions.append(actions)

  def set_baseline_input(self, baseline_input):
    self.baseline_input = baseline_input

  def calc_baseline_loss(self, rewards):
    avg_rewards = dy.average(rewards)
    pred_rewards = []
    loss = []
    for i, enc in enumerate(self.baseline_input):
      pred_reward = self.baseline(dy.nobackprop(enc))
      pred_rewards.append(pred_reward)
      if self.valid_pos is not None:
        pred_reward = dy.pick_batch_elems(pred_reward, self.valid_pos[i])
        avg_reward = dy.pick_batch_elems(avg_rewards, self.valid_pos[i])
      else:
        avg_reward = avg_rewards
      loss.append(dy.sum_batches(dy.squared_distance(pred_reward, avg_reward)))
    return pred_rewards, dy.esum(loss)

  def calc_loss(self, rewards):
    loss = FactoredLossExpr()
    ## Z-Normalization
    if self.z_normalization:
      reward_batches = dy.concatenate_to_batch(rewards)
      mean_batches = dy.mean_batches(reward_batches)
      std_batches = dy.std_batches(reward_batches)
      rewards = [dy.cdiv(reward-mean_batches, std_batches) for reward in rewards]
    ## Calculate baseline   
    if self.baseline is not None:
      pred_reward, baseline_loss = self.calc_baseline_loss(rewards)
      loss.add_loss("rl_baseline", baseline_loss)
    ## Calculate Confidence Penalty
    if self.confidence_penalty:
      loss.add_loss("rl_confpen", self.confidence_penalty.calc_loss(self.policy_lls))
    ## Calculate Reinforce Loss
    reinf_loss = []
    # Loop through all action in one sequence
    for i, (policy, action_sample) in enumerate(zip(self.policy_lls, self.actions)):
      # Discount the reward if we use baseline
      if self.baseline is not None:
        rewards = [reward-pred_reward[i] for reward in rewards]
      # Main Reinforce calculation
      for action, reward in zip(action_sample, rewards):
        ll = dy.pick_batch(policy, action)
        if self.valid_pos is not None:
          ll = dy.pick_batch_elems(ll, self.valid_pos[i])
          reward = dy.pick_batch_elems(reward, self.valid_pos[i])
        reinf_loss.append(dy.sum_batches(ll*reward))
    loss.add_loss("rl_reinf", self.weight * -dy.esum(reinf_loss))
    ## the composed losses
    return loss
  
  class SamplingAction(Enum):
    POLICY_CLP = 0
    POLICY_AMAX = 1
    PREDEFINED = 2
    NONE = 3

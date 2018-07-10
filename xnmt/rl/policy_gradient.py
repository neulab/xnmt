
import numpy as np
import dynet as dy

from xnmt.events import handle_xnmt_event, register_xnmt_handler
from xnmt.transform import Linear
from xnmt.persistence import Ref, bare, Path, Serializable, serializable_init

class PolicyGradient(Serializable):
  yaml_tag = '!PolicyGradient'
  @serializable_init
  @register_xnmt_handler
  def __init__(self, policy_network=bare(Linear),
                     baseline=None,
                     z_normalization=True,
                     sample=1):
    self.policy_network = self.add_serializable_component("policy_network", policy_network, lambda: policy_network)
    self.z_normalization = z_normalization
    self.sample = sample

    if baseline is not None:
      self.baseline = self.add_serializable_component("baseline", baseline, lambda: baseline)
    else:
      self.baseline = None
  
  @handle_xnmt_event
  def on_start_sent(self, src_sent):
    self.policy_lls = []
    self.actions = []
    self.mask = src_sent.mask.np_arr.transpose() if src_sent.mask else None

  def sample_action(self, state, argmax=False):
    policy = dy.log_softmax(self.policy_network(state))
    actions = []
    for k in range(self.sample):
      if argmax:
        actions.append(policy.tensor_value().argmax().as_numpy()[0])
        break # only sample one time during argmax
      else:
        actions.append(policy.tensor_value().categorical_sample_log_prob().as_numpy()[0])
    
    self.policy_lls.append(policy)
    self.actions.append(actions)

    return actions


  def calculate_loss(self, reward_func):
    pass

  def get_num_sample(self):
    return self.sample


def sample_from_softmax(policy):
  return 


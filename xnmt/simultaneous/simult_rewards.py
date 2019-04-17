
import numpy as np
import dynet as dy

import xnmt.thirdparty.dl4mt_simul_trans.reward as simult_reward

class SimultaneousReward(object):
  def __init__(self, src_batch, trg_batch, actions, outputs, trg_vocab):
    self.src_batch = src_batch
    self.trg_batch = trg_batch
    self.trg_vocab = trg_vocab
    self.actions = actions
    self.outputs = outputs
    
  def calculate(self):
    ret = []
    bleus = []
    delays = []
    instant_rewards = []
    for inp, ref, action, output in zip(self.src_batch, self.trg_batch, self.actions, self.outputs):
      reward, bleu, delay, instant_reward = simult_reward.return_reward(output, ref, action, inp.len_unpadded()+1)
      ret.append(reward)
      bleus.append(bleu)
      delays.append(delay)
      instant_rewards.append(instant_reward)
    return [dy.scalarInput(x) for x in np.hstack(ret)], bleus, delays, instant_rewards

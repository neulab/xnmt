
import sys
import logging
import numpy as np

import xnmt.reports as reports

from xnmt import utils
from xnmt.persistence import serializable_init, Serializable, Ref, Path


class SimultLogger(Serializable, reports.Reporter):
  yaml_tag = "!SimultLogger"

  @serializable_init
  def __init__(self,
               report_path:str = None,
               src_vocab=Ref(Path("model.src_reader.vocab")),
               trg_vocab=Ref(Path("model.trg_reader.vocab"))):
    self.src_vocab = src_vocab
    self.trg_vocab = trg_vocab
    self.logger = logging.getLogger("simult")
    
    if report_path is not None:
      utils.make_parent_dir(report_path)
      stream = open(report_path, "w")
    else:
      stream = sys.stderr
    
    self.logger.addHandler(logging.StreamHandler(stream))
    self.logger.setLevel("INFO")

  def create_sent_report(self,
                         sim_outputs,
                         sim_actions,
                         sim_inputs,
                         pg_loss,
                         pg_policy_reward,
                         pg_rewards,
                         pg_policy_ll,
                         pg_actions,
                         sim_bleu,
                         sim_delay,
                         sim_instant_reward,
                         **kwargs):
    length = [len(x) for x in sim_actions]
   
    def repack(arr):
      ret = []
      start = 0
      for i in range(len(length)):
        if i == len(length) - 1:
          ret.append(arr[start:])
        else:
          ret.append(arr[start:(start+length[i])])
          start += length[i]
          ret.append(arr[start:])
      return ret
    
    pg_rewards = repack(pg_rewards)
    pg_policy_ll = repack(pg_policy_ll)
  
  
    for i, (input, actions, outputs) in enumerate(zip(sim_inputs, sim_actions, sim_outputs)):
      src = [str(i) + ":" + self.src_vocab[x] for i, x in enumerate(input)]
      out = [str(i) + ":" + self.trg_vocab[x] for i, x in enumerate(outputs)]
      self.logger.info("SRC: " + " ".join(src))
      self.logger.info("OUT: " + " ".join(out))
      box = Box()
      now_read = 0
      now_write = 0
      for action, reward, ll in zip(actions, pg_rewards[i], pg_policy_ll[i]):
        ll = np.exp(ll.npvalue()[action]) * 100
        reward = reward.value()
        
        if action == 0:
          box.read("f{:d}".format(now_read), "R/{:.0f}/{:.4f}".format(ll, reward))
          now_read += 1
        else:
          box.write("e{:d}".format(now_write), "W/{:.0f}/{:.4f}".format(ll, reward))
          now_write += 1
        if box.is_full():
          box.print(self.logger)
          box = Box()
      
      if not box.is_full():
        box.print(self.logger)
      self.logger.info("BLEU: {}".format(sim_bleu[i]))
      self.logger.info("Delay: {}".format(sim_delay[i]))
      self.logger.info("Instant Reward: {}".format(sim_instant_reward[i]))
      self.logger.info("________")
      
      
class Box:
  def __init__(self, row=1000, col=30):
    self.row = row
    self.col = col
    self.total_read = 0
    self.total_write = 0
    self.buffer = [[" " for _ in range(col+1)] for _ in range(row+1)]
    self.srcs = [""]
    self.outs = []
    self.ptr = [1,1]
    
  def read(self, src, msg="R"):
    self.total_read += 1
    self.srcs.append(str(src))
    self.buffer[self.ptr[0]][self.ptr[1]] = msg
    self.ptr[1] += 1
    
  def write(self, out, msg="W"):
    self.total_write += 1
    self.outs.append(str(out))
    self.buffer[self.ptr[0]][self.ptr[1]] = msg
    self.ptr[0] += 1
    
  def is_full(self):
    return self.total_read >= self.col or self.total_write >= self.row
  
  def print(self, logger):
    self.srcs += [""]
    self.outs += [""]
    self.buffer = self.buffer[:self.total_write+2]
    max_col = [0 for _ in range(self.total_read+2)]
    self.buffer[0][:] = [x for x in self.srcs]
    for i in range(len(self.buffer)):
      self.buffer[i] = self.buffer[i][:self.total_read+2]
      if i > 0:
        self.buffer[i][0] = self.outs[i-1]
      max_col = [max(a, len(b)+2) for a, b in zip(max_col, self.buffer[i])]
    str_format = "".join("{:%ds}" % (mc) for mc in max_col)
    for buffer in self.buffer:
      logger.info(str_format.format(*buffer))
    logger.info("_")
    
  
    
    
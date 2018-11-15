
import logging
import numpy as np
import math

from xnmt import utils
from xnmt.events import register_xnmt_handler, handle_xnmt_event
from xnmt.persistence import serializable_init, Serializable, Ref, Path
from xnmt.specialized_encoders.segmenting_encoder.segmenting_encoder import SegmentingSeqTransducer

class SegmentPLLogger(Serializable):
  yaml_tag = "!SegmentPLLogger"

  @serializable_init
  @register_xnmt_handler
  def __init__(self, report_path:str, src_vocab=Ref(Path("model.src_reader.vocab"))):
    self.src_vocab = src_vocab
    self.logger = logging.getLogger("segmenting_reporter")
    utils.make_parent_dir(report_path)
    self.logger.addHandler(logging.StreamHandler(open(report_path, "w")))
    self.logger.setLevel("INFO")

  @handle_xnmt_event
  def on_start_sent(self, src_sent):
    self.src_sent = src_sent
    self.idx = np.random.randint(low=0, high=src_sent.batch_size())

  def report_process(self, encoder: SegmentingSeqTransducer):
    src = self.src_sent[self.idx]
    src_len = src.len_unpadded()
    src_word = [self.src_vocab[c] for c in src]
    
    actions = encoder.segment_actions
    table = []
    format = []
    table.append(["SRC"] + src_word)
    format.append("{:>5}")
    sample_action = actions[self.idx]
    sample_dense = [1 if j in sample_action else 0 for j in range(src.sent_len())]
    format.append("{:>5}")
    table.append(["ACT"] + sample_dense)
    if encoder.policy_learning is not None:
      if self.src_sent.batch_size() == 1:
        policy_lls = [encoder.policy_learning.policy_lls[j].npvalue().transpose() for j in range(src_len)]
      else:
        policy_lls = [encoder.policy_learning.policy_lls[j].npvalue().transpose()[self.idx] for j in range(src_len)]
      table.append(["LLS"] + ["{:.4f}".format(math.exp(policy_lls[j][sample_dense[j]])) for j in range(src_len)])
      self.pad_last(table)
      if encoder.policy_learning.valid_pos is not None:
        valid_pos = [1 if self.idx in x else 0 for x in encoder.policy_learning.valid_pos]
      else:
        valid_pos = [1 for _ in range(src.sent_len())]
      table.append(["MSK"] + valid_pos)
      format.append("{:>8}")
      format.append("{:>5}")
    arr = np.array(table)
    format = format[::-1]
    self.logger.info("SRC: %s", self.apply_segmentation(src_word, sample_action))
    arr = np.flip(arr, 0).transpose()

    row_format = "".join(format)
    for i, item in enumerate(arr):
      self.logger.info(row_format.format(*item))

  def pad_last(self, table):
    table[-1].extend(["-" for _ in range(len(table[-2])-len(table[-1]))])

  def apply_segmentation(self, src, sample_action):
    begin = 0
    word = []
    for end in sample_action:
      word.append("".join(src[begin:end+1]))
      begin = end+1
    return str(word)


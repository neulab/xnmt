

import logging
import numpy as np
import math

from xnmt import utils
from xnmt.events import register_xnmt_handler, handle_xnmt_event
from xnmt.persistence import serializable_init, Serializable, Ref, Path
from xnmt.transducers.char_compose.segmenting_encoder import SegmentingSeqTransducer

class SimultReporter(Serializable):
  yaml_tag = "!SimultReporter"

  @serializable_init
  @register_xnmt_handler
  def __init__(self, report_path:str, src_vocab=Ref(Path("model.src_reader.vocab"))):
    self.src_vocab = src_vocab
    self.logger = logging.getLogger("simult")
    utils.make_parent_dir(report_path)
    self.logger.addHandler(logging.StreamHandler(open(report_path, "w")))
    self.logger.setLevel("INFO")

  @handle_xnmt_event
  def on_start_sent(self, src_sent):
    self.src_sent = src_sent

  def report_process(self):
    pass



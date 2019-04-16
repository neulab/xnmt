
import sys
import logging

import xnmt.reports as reports

from xnmt import utils
from xnmt.events import register_xnmt_handler, handle_xnmt_event
from xnmt.persistence import serializable_init, Serializable, Ref, Path


class SimultLogger(Serializable, reports.Reporter):
  yaml_tag = "!SimultLogger"

  @serializable_init
  def __init__(self, report_path:str = None, src_vocab=Ref(Path("model.src_reader.vocab"))):
    self.src_vocab = src_vocab
    self.logger = logging.getLogger("simult")
    
    if report_path is not None:
      utils.make_parent_dir(report_path)
      stream = open(report_path, "w")
    else:
      stream = sys.stderr
    
    self.logger.addHandler(stream)
    self.logger.setLevel("INFO")

  def create_sent_report(self, **kwargs):
    pass


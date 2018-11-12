import numbers
import warnings
import numpy as np
from typing import Optional

from xnmt import logger
from xnmt.sent import ArraySentence
from xnmt.input_readers.input_reader import InputReader
from xnmt.persistence import Serializable, serializable_init

with warnings.catch_warnings():
  warnings.simplefilter("ignore", lineno=36)
  import h5py


class H5Reader(InputReader, Serializable):
  """
  Handles the case where sents are sequences of continuous-space vectors.

  The input is a ".h5" file, which can be created for example using xnmt.preproc.MelFiltExtractor

  The data items are assumed to be labeled with integers 0, 1, .. (converted to strings).

  Each data item will be a 2D matrix representing a sequence of vectors. They can
  be in either order, depending on the value of the "transpose" variable:
  * sents[sent_id][feat_ind,timestep] if transpose=False
  * sents[sent_id][timestep,feat_ind] if transpose=True

  Args:
    transpose: whether inputs are transposed or not.
    feat_from: use feature dimensions in a range, starting at this index (inclusive)
    feat_to: use feature dimensions in a range, ending at this index (exclusive)
    feat_skip: stride over features
    timestep_skip: stride over timesteps
    timestep_truncate: cut off timesteps if sequence is longer than specified value
  """
  yaml_tag = u"!H5Reader"

  @serializable_init
  def __init__(self,
               transpose: bool = False,
               feat_from: Optional[numbers.Integral] = None,
               feat_to: Optional[numbers.Integral] = None,
               feat_skip: Optional[numbers.Integral] = None,
               timestep_skip: Optional[numbers.Integral] = None,
               timestep_truncate: Optional[numbers.Integral] = None):
    self.transpose = transpose
    self.feat_from = feat_from
    self.feat_to = feat_to
    self.feat_skip = feat_skip
    self.timestep_skip = timestep_skip
    self.timestep_truncate = timestep_truncate

  def read_sents(self, filename, filter_ids=None):
    with h5py.File(filename, "r") as hf:
      h5_keys = sorted(hf.keys(), key=lambda x: int(x))
      if filter_ids is not None:
        filter_ids = sorted(filter_ids)
        h5_keys = [h5_keys[i] for i in filter_ids]
        h5_keys.sort(key=lambda x: int(x))
      for sent_no, key in enumerate(h5_keys):
        inp = hf[key][:]
        if self.transpose:
          inp = inp.transpose()

        sub_inp = inp[self.feat_from: self.feat_to: self.feat_skip, :self.timestep_truncate:self.timestep_skip]
        if sub_inp.size < inp.size:
          inp = np.empty_like(sub_inp)
          np.copyto(inp, sub_inp)
        else:
          inp = sub_inp

        if sent_no % 1000 == 999:
          logger.info(f"Read {sent_no+1} lines ({float(sent_no+1)/len(h5_keys)*100:.2f}%) of {filename} at {key}")
        yield ArraySentence(idx=filter_ids[sent_no] if filter_ids else sent_no, nparr=inp)

  def count_sents(self, filename):
    with h5py.File(filename, "r") as hf:
      l = len(hf.keys())
    return l

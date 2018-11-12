import numbers
import numpy as np

from typing import Optional

from xnmt import logger
from xnmt.sent import ArraySentence
from xnmt.input_readers.input_reader import InputReader
from xnmt.persistence import serializable_init, Serializable


class NpzReader(InputReader, Serializable):
  """
  Handles the case where sents are sequences of continuous-space vectors.

  The input is a ".npz" file, which consists of multiply ".npy" files, each
  corresponding to a single sequence of continuous features. This can be
  created in two ways:
  * Use the builtin function numpy.savez_compressed()
  * Create a bunch of .npy files, and run "zip" on them to zip them into an archive.

  The file names should be named XXX_0, XXX_1, etc., where the final number after the underbar
  indicates the order of the sequence in the corpus. This is done automatically by
  numpy.savez_compressed(), in which case the names will be arr_0, arr_1, etc.

  Each numpy file will be a 2D matrix representing a sequence of vectors. They can
  be in either order, depending on the value of the "transpose" variable.
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
  yaml_tag = u"!NpzReader"

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
    npz_file = np.load(filename, mmap_mode=None if filter_ids is None else "r")
    npz_keys = sorted(npz_file.files, key=lambda x: int(x.split('_')[-1]))
    if filter_ids is not None:
      filter_ids = sorted(filter_ids)
      npz_keys = [npz_keys[i] for i in filter_ids]
      npz_keys.sort(key=lambda x: int(x.split('_')[-1]))
    for sent_no, key in enumerate(npz_keys):
      inp = npz_file[key]
      if self.transpose:
        inp = inp.transpose()

      sub_inp = inp[self.feat_from: self.feat_to: self.feat_skip, :self.timestep_truncate:self.timestep_skip]
      if sub_inp.size < inp.size:
        inp = np.empty_like(sub_inp)
        np.copyto(inp, sub_inp)
      else:
        inp = sub_inp

      if sent_no % 1000 == 999:
        logger.info(f"Read {sent_no+1} lines ({float(sent_no+1)/len(npz_keys)*100:.2f}%) of {filename} at {key}")
      yield ArraySentence(idx=filter_ids[sent_no] if filter_ids else sent_no, nparr=inp)
    npz_file.close()

  def count_sents(self, filename):
    npz_file = np.load(filename, mmap_mode="r")  # for counting sentences, only read the index
    try:
      return len(npz_file.files)
    finally:
      npz_file.close()

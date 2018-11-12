import numpy as np
import numbers

from itertools import zip_longest
from typing import Optional

from xnmt import batchers, logger
from xnmt.input_readers.input_reader import InputReader


def read_parallel_corpus(src_reader: InputReader,
                         trg_reader: InputReader,
                         src_file: str,
                         trg_file: str,
                         batcher: batchers.Batcher=None,
                         sample_sents: Optional[numbers.Integral] = None,
                         max_num_sents: Optional[numbers.Integral] = None,
                         max_src_len: Optional[numbers.Integral] = None,
                         max_trg_len: Optional[numbers.Integral] = None) -> tuple:
  """
  A utility function to read a parallel corpus.

  Args:
    src_reader: src reader
    trg_reader: trg reader
    src_file: path to src
    trg_file: path to trg
    batcher: the batcher
    sample_sents: if not None, denote the number of sents that should be randomly chosen from all available sents.
    max_num_sents: if not None, read only the first this many sents
    max_src_len: skip pair if src side is too long
    max_trg_len: skip pair if trg side is too long

  Returns:
    A tuple of (src_data, trg_data, src_batches, trg_batches) where ``*_batches = *_data`` if ``batcher=None``
  """
  src_data = []
  trg_data = []
  if sample_sents:
    logger.info(f"Starting to read {sample_sents} parallel sentences of {src_file} and {trg_file}")
    src_len = src_reader.count_sents(src_file)
    trg_len = trg_reader.count_sents(trg_file)
    if src_len != trg_len: raise RuntimeError(f"training src sentences don't match trg sentences: {src_len} != {trg_len}!")
    if max_num_sents and max_num_sents < src_len: src_len = trg_len = max_num_sents
    filter_ids = np.random.choice(src_len, sample_sents, replace=False)
  else:
    logger.info(f"Starting to read {src_file} and {trg_file}")
    filter_ids = None
    src_len, trg_len = 0, 0
  src_train_iterator = src_reader.read_sents(src_file, filter_ids)
  trg_train_iterator = trg_reader.read_sents(trg_file, filter_ids)
  for src_sent, trg_sent in zip_longest(src_train_iterator, trg_train_iterator):
    if src_sent is None or trg_sent is None:
      raise RuntimeError(f"training src sentences don't match trg sentences: {src_len or src_reader.count_sents(src_file)} != {trg_len or trg_reader.count_sents(trg_file)}!")
    if max_num_sents and (max_num_sents <= len(src_data)):
      break
    src_len_ok = max_src_len is None or src_sent.sent_len() <= max_src_len
    trg_len_ok = max_trg_len is None or trg_sent.sent_len() <= max_trg_len
    if src_len_ok and trg_len_ok:
      src_data.append(src_sent)
      trg_data.append(trg_sent)

  logger.info(f"Done reading {src_file} and {trg_file}. Packing into batches.")

  # Pack batches
  if batcher is not None:
    src_batches, trg_batches = batcher.pack(src_data, trg_data)
  else:
    src_batches, trg_batches = src_data, trg_data

  logger.info(f"Done packing batches.")

  return src_data, trg_data, src_batches, trg_batches

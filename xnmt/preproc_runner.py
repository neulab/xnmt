import os.path
from typing import List, Optional

from xnmt import logger
from xnmt.preproc import Normalizer, SentenceFilterer, VocabFilterer
from xnmt.persistence import serializable_init, Serializable
from xnmt.util import make_parent_dir

class PreprocTask(object):
  def run_preproc_task(self, overwrite=False):
    ...

class PreprocRunner(Serializable):
  """
  Preprocess and filter the input files, and create the vocabulary.

  Args:
    tasks: A list of preprocessing steps, usually parametrized by in_files (the input files), out_files (the output files), and spec for that particular preprocessing type
           The types of arguments that preproc_spec expects:
           * Option("in_files", help_str="list of paths to the input files"),
           * Option("out_files", help_str="list of paths for the output files"),
           * Option("spec", help_str="The specifications describing which type of processing to use. For normalize and vocab, should consist of the 'lang' and 'spec', where 'lang' can either be 'all' to apply the same type of processing to all languages, or a zero-indexed integer indicating which language to process."),
    overwrite: Whether to overwrite files if they already exist.
  """
  yaml_tag = "!PreprocRunner"

  @serializable_init
  def __init__(self, tasks:Optional[List[PreprocTask]]=None, overwrite:bool=False):
    if tasks is None: tasks = []
    logger.info("> Preprocessing")

    for task in tasks:
      # Sanity check
      if len(task.in_files) != len(task.out_files):
        raise RuntimeError("Length of in_files and out_files in preprocessor must be identical")
      task.run_preproc_task(overwrite = overwrite)

class PreprocExtract(PreprocTask, Serializable):
  yaml_tag = "!PreprocExtract"
  @serializable_init
  def __init__(self, in_files, out_files, specs):
    self.in_files = in_files
    self.out_files = out_files
    self.specs = specs
  def run_preproc_task(self, overwrite=False):
    extractor = self.specs
    for in_file, out_file in zip(self.in_files, self.out_files):
      if overwrite or not os.path.isfile(out_file):
        make_parent_dir(out_file)
        extractor.extract_to(in_file, out_file)

class PreprocTokenize(PreprocTask, Serializable):
  yaml_tag = "!PreprocTokenize"
  @serializable_init
  def __init__(self, in_files, out_files, specs):
    self.in_files = in_files
    self.out_files = out_files
    self.specs = specs
  def run_preproc_task(self, overwrite=False):
    tokenizers = {my_opts["filenum"]: [tok
          for tok in my_opts["tokenizers"]]
          for my_opts in self.specs}
    for file_num, (in_file, out_file) in enumerate(zip(self.in_files, self.out_files)):
      if overwrite or not os.path.isfile(out_file):
        make_parent_dir(out_file)
        my_tokenizers = tokenizers.get(file_num, tokenizers["all"])
        with open(out_file, "w", encoding='utf-8') as out_stream, \
             open(in_file, "r", encoding='utf-8') as in_stream:
          for tokenizer in my_tokenizers:
            in_stream = tokenizer.tokenize_stream(in_stream)
          for line in in_stream:
            out_stream.write(f"{line}\n")

class PreprocNormalize(PreprocTask, Serializable):
  yaml_tag = "!PreprocNormalize"
  @serializable_init
  def __init__(self, in_files, out_files, specs):
    self.in_files = in_files
    self.out_files = out_files
    self.specs = specs
  def run_preproc_task(self, overwrite=False):
    normalizers = {my_opts["filenum"]: [norm
          for norm in my_opts["normalizers"]]
          for my_opts in self.specs}
    for i, (in_file, out_file) in enumerate(zip(self.in_files, self.out_files)):
      if overwrite or not os.path.isfile(out_file):
        make_parent_dir(out_file)
        my_normalizers = normalizers.get(i, normalizers["all"])
        with open(out_file, "w", encoding='utf-8') as out_stream, \
             open(in_file, "r", encoding='utf-8') as in_stream:
          for line in in_stream:
            line = line.strip()
            for normalizer in my_normalizers:
              line = normalizer.normalize(line)
            out_stream.write(line + "\n")

class PreprocFilter(PreprocTask, Serializable):
  yaml_tag = "!PreprocFilter"
  @serializable_init
  def __init__(self, in_files, out_files, specs):
    self.in_files = in_files
    self.out_files = out_files
    self.specs = specs
  def run_preproc_task(self, overwrite=False):
    # TODO: This will only work with plain-text sentences at the moment. It would be nice if it plays well with the readers
    #       in input.py
    filters = SentenceFilterer.from_spec(self.specs)
    out_streams = [open(x, 'w', encoding='utf-8') if overwrite or not os.path.isfile(x) else None for x in self.out_files]
    if any(x is not None for x in out_streams):
      in_streams = [open(x, 'r', encoding='utf-8') for x in self.in_files]
      for in_lines in zip(*in_streams):
        in_lists = [line.strip().split() for line in in_lines]
        if all([my_filter.keep(in_lists) for my_filter in filters]):
          for in_line, out_stream in zip(in_lines, out_streams):
            out_stream.write(in_line)
      for x in in_streams:
        x.close()
    for x in out_streams:
      if x is not None:
        x.close()


class PreprocVocab(PreprocTask, Serializable):
  yaml_tag = "!PreprocVocab"
  @serializable_init
  def __init__(self, in_files, out_files, specs):
    self.in_files = in_files
    self.out_files = out_files
    self.specs = specs
  def run_preproc_task(self, overwrite=False):
    filters = {my_opts["filenum"]: [norm
          for norm in my_opts["filters"]]
          for my_opts in self.specs}
    for i, (in_file, out_file) in enumerate(zip(self.in_files, self.out_files)):
      if overwrite or not os.path.isfile(out_file):
        make_parent_dir(out_file)
        with open(out_file, "w", encoding='utf-8') as out_stream, \
             open(in_file, "r", encoding='utf-8') as in_stream:
          vocab = {}
          for line in in_stream:
            for word in line.strip().split():
              vocab[word] = vocab.get(word, 0) + 1
          for my_filter in filters.get(i, filters["all"]):
            vocab = my_filter.filter(vocab)
          for word in vocab.keys():
            out_stream.write((word + u"\n"))

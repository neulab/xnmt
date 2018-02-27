import logging
logger = logging.getLogger('xnmt')
import os.path
import io

from xnmt.preproc import Normalizer, SentenceFilterer, VocabFilterer
from xnmt.serialize.serializable import Serializable
##### Main function

def make_parent_dir(filename):
  if not os.path.exists(os.path.dirname(filename)):
    try:
      os.makedirs(os.path.dirname(filename))
    except OSError as exc: # Guard against race condition
      if exc.errno != os.errno.EEXIST:
        raise

class PreprocRunner(Serializable):
  yaml_tag = "!PreprocRunner"
  def __init__(self, tasks=[], overwrite=False):
    """Preprocess and filter the input files, and create the vocabulary
    :param tasks (list): A specification for a preprocessing step, including in_files (the input files), out_files (the output files), type (normalize/filter/vocab), and spec for that particular preprocessing type
                         Expected is a list of PreprocTask objects (e.g. !PreprocTokenize, !PreprocNormalize, !PreprocFilter, or !PreprocVocab), and for each the following arguments:
                               Option("in_files", help_str="list of paths to the input files"),
                               Option("out_files", help_str="list of paths for the output files"),
                               Option("spec", help_str="The specifications describing which type of processing to use. For normalize and vocab, should consist of the 'lang' and 'spec', where 'lang' can either be 'all' to apply the same type of processing to all languages, or a zero-indexed integer indicating which language to process."),
    :param overwrite (bool): Whether to overwrite files if they already exist.
    """
    logger.info("> Preprocessing")
    
    for task in tasks:

      # Sanity check
      if len(task.in_files) != len(task.out_files):
        raise RuntimeError("Length of in_files and out_files in preprocessor must be identical")

      task.run_preproc_task(overwrite = overwrite)

class PreprocTask(object):
  def run_preproc_task(self, overwrite=False):
    ...

class PreprocExtract(PreprocTask, Serializable):
  yaml_tag = "!PreprocExtract"
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
        with io.open(out_file, "w", encoding='utf-8') as out_stream, \
             io.open(in_file, "r", encoding='utf-8') as in_stream:
          for tokenizer in my_tokenizers:
            in_stream = tokenizer.tokenize_stream(in_stream)
          for line in in_stream:
            out_stream.write(f"{line}\n")

class PreprocNormalize(PreprocTask, Serializable):
  yaml_tag = "!PreprocNormalize"
  def __init__(self, in_files, out_files, specs):
    self.in_files = in_files
    self.out_files = out_files
    self.specs = specs
  def run_preproc_task(self, overwrite=False):
    normalizers = {my_opts["filenum"]: Normalizer.from_spec(my_opts["spec"]) for my_opts in self.specs}
    for i, (in_file, out_file) in enumerate(zip(self.in_files, self.out_files)):
      if overwrite or not os.path.isfile(out_file):
        make_parent_dir(out_file)
        my_normalizers = normalizers.get(i, normalizers["all"])
        with io.open(out_file, "w", encoding='utf-8') as out_stream, \
             io.open(in_file, "r", encoding='utf-8') as in_stream:
          for line in in_stream:
            line = line.strip()
            for normalizer in my_normalizers:
              line = normalizer.normalize(line)
            out_stream.write(line + "\n")

class PreprocFilter(PreprocTask, Serializable):
  yaml_tag = "!PreprocFilter"
  def __init__(self, in_files, out_files, specs):
    self.in_files = in_files
    self.out_files = out_files
    self.specs = specs
  def run_preproc_task(self, overwrite=False):
    # TODO: This will only work with plain-text sentences at the moment. It would be nice if it plays well with the readers
    #       in input.py
    filters = SentenceFilterer.from_spec(self.specs)
    out_streams = [io.open(x, 'w', encoding='utf-8') if overwrite or not os.path.isfile(x) else None for x in self.out_files]
    if any(x is not None for x in out_streams):
      in_streams = [io.open(x, 'r', encoding='utf-8') for x in self.in_files]
      for in_lines in zip(*in_streams):
        in_lists = [line.strip().split() for line in in_lines]
        if all([my_filter.keep(in_lists) for my_filter in filters]):
          for in_line, out_stream in zip(in_lines, out_streams):
            out_stream.write(in_line)
      for x in in_streams:
        x.close()
    for x in out_streams:
      if x != None:
        x.close()

class PreprocVocab(PreprocTask, Serializable):
  yaml_tag = "!PreprocVocab"
  def __init__(self, in_files, out_files, specs):
    self.in_files = in_files
    self.out_files = out_files
    self.specs = specs
  def run_preproc_task(self, overwrite=False):
    filters = {my_opts["filenum"]: VocabFilterer.from_spec(my_opts["spec"]) for my_opts in self.specs}
    for i, (in_file, out_file) in enumerate(zip(self.in_files, self.out_files)):
      if overwrite or not os.path.isfile(out_file):
        make_parent_dir(out_file)
        with io.open(out_file, "w", encoding='utf-8') as out_stream, \
             io.open(in_file, "r", encoding='utf-8') as in_stream:
          vocab = {}
          for line in in_stream:
            for word in line.strip().split():
              vocab[word] = vocab.get(word, 0) + 1
          for my_filter in filters.get(i, filters["all"]):
            vocab = my_filter.filter(vocab)
          for word in vocab.keys():
            out_stream.write((word + u"\n"))

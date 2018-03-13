import logging
logger = logging.getLogger('xnmt')
import os.path

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
  """
  Preprocess and filter the input files, and create the vocabulary.

  Args:
    preproc_specs (list): A specification for a preprocessing step, including in_files (the input files), out_files (the output files), type (normalize/filter/vocab), and spec for that particular preprocessing type
                          The types of arguments that preproc_spec expects:
                          * Option("in_files", help_str="list of paths to the input files"),
                          * Option("out_files", help_str="list of paths for the output files"),
                          * Option("type", help_str="type of preprocessing (normalize,filter,vocab)"),
                          * Option("spec", help_str="The specifications describing which type of processing to use. For normalize and vocab, should consist of the 'lang' and 'spec', where 'lang' can either be 'all' to apply the same type of processing to all languages, or a zero-indexed integer indicating which language to process."),
    overwrite (bool): Whether to overwrite files if they already exist.
  """
  yaml_tag = "!PreprocRunner"
  def __init__(self, preproc_specs, overwrite=False):
    logger.info("> Preprocessing")

    args = dict(preproc_specs=preproc_specs, overwrite=overwrite)

    if args["preproc_specs"] == None:
      return

    for arg in args["preproc_specs"]:

      # Sanity check
      if len(arg["in_files"]) != len(arg["out_files"]):
        raise RuntimeError("Length of in_files and out_files in preprocessor must be identical")

      # Perform tokenization
      if arg["type"] == 'tokenize':
        tokenizers = {my_opts["filenum"]: [tok
              for tok in my_opts["tokenizers"]]
              for my_opts in arg["specs"]}
        for file_num, (in_file, out_file) in enumerate(zip(arg["in_files"], arg["out_files"])):
          if args["overwrite"] or not os.path.isfile(out_file):
            make_parent_dir(out_file)
            my_tokenizers = tokenizers.get(file_num, tokenizers["all"])
            with open(out_file, "w", encoding='utf-8') as out_stream, \
                 open(in_file, "r", encoding='utf-8') as in_stream:
              for tokenizer in my_tokenizers:
                in_stream = tokenizer.tokenize_stream(in_stream)
              for line in in_stream:
                out_stream.write("{}\n".format(line))

      # Perform normalization
      elif arg["type"] == 'normalize':
        normalizers = {my_opts["filenum"]: Normalizer.from_spec(my_opts["spec"]) for my_opts in arg["specs"]}
        for i, (in_file, out_file) in enumerate(zip(arg["in_files"], arg["out_files"])):
          if args["overwrite"] or not os.path.isfile(out_file):
            make_parent_dir(out_file)
            my_normalizers = normalizers.get(i, normalizers["all"])
            with open(out_file, "w", encoding='utf-8') as out_stream, \
                 open(in_file, "r", encoding='utf-8') as in_stream:
              for line in in_stream:
                line = line.strip()
                for normalizer in my_normalizers:
                  line = normalizer.normalize(line)
                out_stream.write(line + "\n")

      # Perform filtering
      # TODO: This will only work with plain-text sentences at the moment. It would be nice if it plays well with the readers
      #       in input.py
      elif arg["type"] == 'filter':
        filters = SentenceFilterer.from_spec(arg["specs"])
        out_streams = [open(x, 'w', encoding='utf-8') if args["overwrite"] or not os.path.isfile(x) else None for x in arg["out_files"]]
        if any(x is not None for x in out_streams):
          in_streams = [open(x, 'r', encoding='utf-8') for x in arg["in_files"]]
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

      # Vocabulary selection
      # TODO: This will only work with plain-text sentences at the moment. It would be nice if it plays well with the readers
      #       in input.py
      elif arg["type"] == 'vocab':
        filters = {my_opts["filenum"]: VocabFilterer.from_spec(my_opts["spec"]) for my_opts in arg["specs"]}
        for i, (in_file, out_file) in enumerate(zip(arg["in_files"], arg["out_files"])):
          if args["overwrite"] or not os.path.isfile(out_file):
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
                out_stream.write((word + "\n"))

      else:
        raise RuntimeError("Unknown preprocessing type {}".format(arg['type']))


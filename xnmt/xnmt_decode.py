# coding: utf-8

import io
from output import *
from serializer import *
import sys
from retriever import *
from translator import *
from reports import *
from search_strategy import *
from options import OptionParser, Option

import dynet as dy

'''
This will be the main class to perform decoding.
'''

options = [
  Option("dynet-mem", int, required=False),
  Option("dynet-gpu-ids", int, required=False),
  Option("model_file", force_flag=True, required=True, help_str="pretrained (saved) model path"),
  Option("src_file", help_str="path of input src file to be translated"),
  Option("trg_file", help_str="path of file where expected trg translatons will be written"),
  Option("max_src_len", int, required=False, help_str="Remove sentences from data to decode that are longer than this on the source side"),
  Option("input_format", default_value="text", help_str="format of input data: text/contvec"),
  Option("post_process", default_value="none", help_str="post-processing of translation outputs: none/join-char/join-bpe"),
  Option("candidate_id_file", required=False, default_value=None, help_str="if we are doing something like retrieval where we select from fixed candidates, sometimes we want to limit our candidates to a certain subset of the full set. this setting allows us to do this."),
  Option("report_path", str, required=False, help_str="a path to which decoding reports will be written"),
  Option("beam", int, default_value=1),
  Option("max_len", int, default_value=100),
  Option("len_norm_type", str, default_value="NoNormalization"),
  Option("len_norm_params", dict, default_value={}),
]

NO_DECODING_ATTEMPTED = u"@@NO_DECODING_ATTEMPTED@@"

def xnmt_decode(args, model_elements=None):
  """
  :param model_elements: If None, the model will be loaded from args.model_file. If set, should
  equal (corpus_parser, generator).
  """
  if model_elements is None:
    raise RuntimeError("xnmt_decode with model_element=None needs to be updated to run with the new YamlSerializer")
#    TODO: These lines need to be updated / removed!
#    model = dy.Model()
#    model_serializer = JSONSerializer()
#    serialize_container = model_serializer.load_from_file(args.model_file, model)
#
#    src_vocab = Vocab(serialize_container.src_vocab)
#    trg_vocab = Vocab(serialize_container.trg_vocab)
#
#    generator = DefaultTranslator(serialize_container.src_embedder, serialize_container.encoder,
#                                  serialize_container.attender, serialize_container.trg_embedder,
#                                  serialize_container.decoder)
#
  else:
    corpus_parser, generator = model_elements

  # Corpus
  src_corpus = corpus_parser.src_reader.read_sents(args.src_file)
  # Vocab
  src_vocab = corpus_parser.src_reader.vocab if hasattr(corpus_parser.src_reader, "vocab") else None
  trg_vocab = corpus_parser.trg_reader.vocab if hasattr(corpus_parser.trg_reader, "vocab") else None
  # Perform initialization
  generator.set_train(False)
  generator.initialize(args)
  generator.set_vocabs(src_vocab, trg_vocab)
  # TODO: Structure it better. not only Translator can have post processes
  if issubclass(generator.__class__, Translator):
    generator.set_post_processor(output_processor_for_spec(args.post_process))
  # Perform generation of output
  with io.open(args.trg_file, 'wt', encoding='utf-8') as fp:  # Saving the translated output to a trg file
    for i, src in enumerate(src_corpus):
      # Do the decoding
      if args.max_src_len is not None and len(src) > args.max_src_len:
        outputs = NO_DECODING_ATTEMPTED
      else:
        dy.renew_cg()
        outputs = generator.generate_output(src, i)
      # Printing to trg file
      fp.write(u"{}\n".format(outputs))
      # Generating html report
      if hasattr(generator, "generate_html_report") and args.report_path is not None:
        generator.generate_html_report()

def output_processor_for_spec(spec):
  if spec == "none":
    return PlainTextOutputProcessor()
  elif spec == "join-char":
    return JoinedCharTextOutputProcessor()
  elif spec == "join-bpe":
    return JoinedBPETextOutputProcessor()
  else:
    raise RuntimeError("Unknown postprocessing argument {}".format(spec))

if __name__ == "__main__":
  # Parse arguments
  parser = OptionParser()
  parser.add_task("decode", options)
  args = parser.args_from_command_line("decode", sys.argv[1:])
  # Load model
  xnmt_decode(args)


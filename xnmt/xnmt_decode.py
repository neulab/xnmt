# coding: utf-8

import io
import sys

import dynet as dy

from xnmt.output import *
from xnmt.serializer import *
from xnmt.retriever import *
from xnmt.translator import *
from xnmt.search_strategy import *
from xnmt.options import OptionParser, Option
from xnmt import training_strategy
from xnmt.training_strategy import TrainingStrategy

'''
This will be the main class to perform decoding.
'''

NO_DECODING_ATTEMPTED = u"@@NO_DECODING_ATTEMPTED@@"

def xnmt_decode(src_file, trg_file, model_elements=None, model_file=None, ref_file=None, max_src_len=None,
                input_format="text", post_process="none", candidate_id_file=None, report_path=None, report_type="html",
                beam=1, max_len=100, len_norm_type=None, mode="onebest", training_strategy=TrainingStrategy()):
  """
  :param src_file: path of input src file to be translated
  :param trg_file: path of file where trg translatons will be written
  :param model_elements: If None, the model will be loaded from model_file. If set, should equal (corpus_parser, generator).
  :param model_file: pretrained (saved) model path (required onless model_elements is given)
  :param ref_file: path of file with reference translations, e.g. for forced decoding
  :param max_src_len (int): Remove sentences from data to decode that are longer than this on the source side
  :param input_format: format of input data: text/contvec
  :param post_process: post-processing of translation outputs: none/join-char/join-bpe
  :param candidate_id_file: if we are doing something like retrieval where we select from fixed candidates, sometimes we want to limit our candidates to a certain subset of the full set. this setting allows us to do this.
  :param report_path: a path to which decoding reports will be written
  :param report_type: report to generate file/html. Can be multiple, separate with comma.
  :param beam (int):
  :param max_len (int):
  :param len_norm_type:
  :param mode: type of decoding to perform. onebest: generate one best. forced: perform forced decoding. forceddebug: perform forced decoding, calculate training loss, and make suer the scores are identical for debugging purposes.
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
  
  args = dict(model_file=model_file, src_file=src_file, trg_file=trg_file, ref_file=ref_file, max_src_len=max_src_len,
                input_format=input_format, post_process=post_process, candidate_id_file=candidate_id_file, report_path=report_path, report_type=report_type,
                beam=beam, max_len=max_len, len_norm_type=len_norm_type, mode=mode)

  is_reporting = issubclass(generator.__class__, Reportable) and args["report_path"] is not None
  # Corpus
  src_corpus = list(corpus_parser.src_reader.read_sents(args["src_file"]))
  # Get reference if it exists and is necessary
  if args["mode"] == "forced" or args["mode"] == "forceddebug":
    if args["ref_file"] == None:
      raise RuntimeError("When performing {} decoding, must specify reference file".format(args["mode"]))
    ref_corpus = list(corpus_parser.trg_reader.read_sents(args["ref_file"]))
  else:
    ref_corpus = None
  # Vocab
  src_vocab = corpus_parser.src_reader.vocab if hasattr(corpus_parser.src_reader, "vocab") else None
  trg_vocab = corpus_parser.trg_reader.vocab if hasattr(corpus_parser.trg_reader, "vocab") else None
  # Perform initialization
  generator.set_train(False)
  generator.initialize_generator(**args)

  # TODO: Structure it better. not only Translator can have post processes
  if issubclass(generator.__class__, Translator):
    generator.set_post_processor(output_processor_for_spec(args["post_process"]))
    generator.set_trg_vocab(trg_vocab)
    generator.set_reporting_src_vocab(src_vocab)

  if is_reporting:
    generator.set_report_resource("src_vocab", src_vocab)
    generator.set_report_resource("trg_vocab", trg_vocab)

  # If we're debugging, calculate the loss for each target sentence
  ref_scores = None
  if args["mode"] == 'forceddebug':
    batcher = xnmt.batcher.InOrderBatcher(32) # Arbitrary
    batched_src, batched_ref = batcher.pack(src_corpus, ref_corpus)
    ref_scores = []
    for src, ref in zip(batched_src, batched_ref):
      dy.renew_cg()
      loss_expr = generator.calc_loss(src, ref, training_strategy)
      ref_scores.extend(loss_expr.value())
    ref_scores = [-x for x in ref_scores]

  # Perform generation of output
  with io.open(args["trg_file"], 'wt', encoding='utf-8') as fp:  # Saving the translated output to a trg file
    for i, src in enumerate(src_corpus):
      # Do the decoding
      if args["max_src_len"] is not None and len(src) > args["max_src_len"]:
        output_txt = NO_DECODING_ATTEMPTED
      else:
        dy.renew_cg()
        ref_ids = ref_corpus[i] if ref_corpus != None else None
        output = generator.generate_output(src, i, forced_trg_ids=ref_ids)
        # If debugging forced decoding, make sure it matches
        if ref_scores != None and (abs(output[0].score-ref_scores[i]) / abs(ref_scores[i])) > 1e-5:
          print('Forced decoding score {} and loss {} do not match at sentence {}'.format(output[0].score, ref_scores[i], i))
        output_txt = output[0].plaintext
      # Printing to trg file
      fp.write(u"{}\n".format(output_txt))

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
  args = parser.args_from_command_line("decode", sys.argv[1:])
  # Load model
  xnmt_decode(args)


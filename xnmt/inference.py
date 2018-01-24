# coding: utf-8

import io

from simple_settings import settings

import dynet as dy

from xnmt.loss_calculator import LossCalculator
import xnmt.output
from xnmt.reports import Reportable
from xnmt.serialize.serializable import Serializable
from xnmt.serialize.tree_tools import Ref, Path

'''
This will be the main class to perform decoding.
'''

NO_DECODING_ATTEMPTED = u"@@NO_DECODING_ATTEMPTED@@"

class SimpleInference(Serializable):
  yaml_tag = u'!SimpleInference'
  def __init__(self, model_file=None, src_file=None, trg_file=None, ref_file=None, max_src_len=None,
                  input_format="text", post_process="none", report_path=None, report_type="html",
                  beam=1, max_len=100, len_norm_type=None, mode="onebest", batcher=Ref(Path("train.batcher"), required=False)):
    """
    :param model_file: pretrained (saved) model path (required onless model_elements is given)
    :param src_file: path of input src file to be translated
    :param trg_file: path of file where trg translatons will be written
    :param ref_file: path of file with reference translations, e.g. for forced decoding
    :param max_src_len (int): Remove sentences from data to decode that are longer than this on the source side
    :param input_format: format of input data: text/contvec
    :param post_process: post-processing of translation outputs: none/join-char/join-bpe/join-piece
    :param report_path: a path to which decoding reports will be written
    :param report_type: report to generate file/html. Can be multiple, separate with comma.
    :param beam (int):
    :param max_len (int):
    :param len_norm_type:
    :param mode: type of decoding to perform. onebest: generate one best. forced: perform forced decoding. forceddebug: perform forced decoding, calculate training loss, and make suer the scores are identical for debugging purposes.
    """
    self.model_file = model_file
    self.src_file = src_file
    self.trg_file = trg_file
    self.ref_file = ref_file
    self.max_src_len = max_src_len
    self.input_format = input_format
    self.post_process = post_process
    self.report_path = report_path
    self.report_type = report_type
    self.beam = beam
    self.max_len = max_len
    self.len_norm_type = len_norm_type
    self.mode = mode
    self.batcher = batcher
    

  def __call__(self, generator, src_file=None, trg_file=None, candidate_id_file=None):
    """
    :param src_file: path of input src file to be translated
    :param trg_file: path of file where trg translatons will be written
    :param candidate_id_file: if we are doing something like retrieval where we select from fixed candidates, sometimes we want to limit our candidates to a certain subset of the full set. this setting allows us to do this.
    :param model_elements: If None, the model will be loaded from model_file. If set, should equal (corpus_parser, generator).
    """
    args = dict(model_file=self.model_file, src_file=src_file or self.src_file, trg_file=trg_file or self.trg_file, ref_file=self.ref_file, max_src_len=self.max_src_len,
                  input_format=self.input_format, post_process=self.post_process, candidate_id_file=candidate_id_file, report_path=self.report_path, report_type=self.report_type,
                  beam=self.beam, max_len=self.max_len, len_norm_type=self.len_norm_type, mode=self.mode)
  
    is_reporting = issubclass(generator.__class__, Reportable) and args["report_path"] is not None
    # Corpus
    src_corpus = list(generator.src_reader.read_sents(args["src_file"]))
    # Get reference if it exists and is necessary
    if args["mode"] == "forced" or args["mode"] == "forceddebug":
      if args["ref_file"] == None:
        raise RuntimeError("When performing {} decoding, must specify reference file".format(args["mode"]))
      ref_corpus = list(generator.trg_reader.read_sents(args["ref_file"]))
    else:
      ref_corpus = None
    # Vocab
    src_vocab = generator.src_reader.vocab if hasattr(generator.src_reader, "vocab") else None
    trg_vocab = generator.trg_reader.vocab if hasattr(generator.trg_reader, "vocab") else None
    # Perform initialization
    generator.set_train(False)
    generator.initialize_generator(**args)
  
    if hasattr(generator, "set_post_processor"):
      generator.set_post_processor(self.get_output_processor())
    if hasattr(generator, "set_trg_vocab"):
      generator.set_trg_vocab(trg_vocab)
    if hasattr(generator, "set_reporting_src_vocab"):
      generator.set_reporting_src_vocab(src_vocab)
      
    if is_reporting:
      generator.set_report_resource("src_vocab", src_vocab)
      generator.set_report_resource("trg_vocab", trg_vocab)
  
    # If we're debugging, calculate the loss for each target sentence
    ref_scores = None
    if args["mode"] == 'forceddebug':
      some_batcher = xnmt.batcher.InOrderBatcher(32) # Arbitrary
      batched_src, batched_ref = some_batcher.pack(src_corpus, ref_corpus)
      ref_scores = []
      for src, ref in zip(batched_src, batched_ref):
        dy.renew_cg(immediate_compute=settings.IMMEDIATE_COMPUTE, check_validity=settings.CHECK_VALIDITY)
        loss_expr = generator.calc_loss(src, ref, loss_calculator=LossCalculator())
        ref_scores.extend(loss_expr.value())
      ref_scores = [-x for x in ref_scores]
  
    # Perform generation of output
    with io.open(args["trg_file"], 'wt', encoding='utf-8') as fp:  # Saving the translated output to a trg file
      src_ret=[]
      for i, src in enumerate(src_corpus):
        # This is necessary when the batcher does some sort of pre-processing, e.g.
        # when the batcher pads to a particular number of dimensions
        if self.batcher:
          self.batcher.add_single_batch(src_curr=[src], trg_curr=None, src_ret=src_ret, trg_ret=None)
          src = src_ret.pop()[0]

        # Do the decoding
        if args["max_src_len"] is not None and len(src) > args["max_src_len"]:
          output_txt = NO_DECODING_ATTEMPTED
        else:
          dy.renew_cg(immediate_compute=settings.IMMEDIATE_COMPUTE, check_validity=settings.CHECK_VALIDITY)
          ref_ids = ref_corpus[i] if ref_corpus != None else None
          output = generator.generate_output(src, i, forced_trg_ids=ref_ids)
          # If debugging forced decoding, make sure it matches
          if ref_scores != None and (abs(output[0].score-ref_scores[i]) / abs(ref_scores[i])) > 1e-5:
            print('Forced decoding score {} and loss {} do not match at sentence {}'.format(output[0].score, ref_scores[i], i))
          output_txt = output[0].plaintext
        # Printing to trg file
        fp.write(u"{}\n".format(output_txt))
  
  def get_output_processor(self):
    spec = self.post_process
    if spec == "none":
      return xnmt.output.PlainTextOutputProcessor()
    elif spec == "join-char":
      return xnmt.output.JoinedCharTextOutputProcessor()
    elif spec == "join-bpe":
      return xnmt.output.JoinedBPETextOutputProcessor()
    elif spec == "join-piece":
      return xnmt.output.JoinedPieceTextOutputProcessor()
    else:
      raise RuntimeError("Unknown postprocessing argument {}".format(spec))

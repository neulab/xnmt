# coding: utf-8

import logging
logger = logging.getLogger('xnmt')
from collections.abc import Iterable

from simple_settings import settings

import dynet as dy

from xnmt.loss_calculator import LossCalculator
import xnmt.output
from xnmt.reports import Reportable
from xnmt.serialize.serializable import Serializable, serializable_init, Ref, Path

NO_DECODING_ATTEMPTED = "@@NO_DECODING_ATTEMPTED@@"

class SimpleInference(Serializable):
  """
  Main class to perform decoding.
  
  Args:
    model_file (str): pretrained (saved) model path (required onless model_elements is given)
    src_file (str): path of input src file to be translated
    trg_file (str): path of file where trg translatons will be written
    ref_file (str): path of file with reference translations, e.g. for forced decoding
    max_src_len (int): Remove sentences from data to decode that are longer than this on the source side
    post_process (str): post-processing of translation outputs: ``none/join-char/join-bpe/join-piece``
    report_path (str): a path to which decoding reports will be written
    report_type (str): report to generate ``file/html``. Can be multiple, separate with comma.
    beam (int):
    max_len (int):
    len_norm_type (LengthNormalization):
    mode (str): type of decoding to perform. ``onebest``: generate one best. ``forced``: perform forced decoding. ``forceddebug``: perform forced decoding, calculate training loss, and make suer the scores are identical for debugging purposes.
    batcher (Batcher):
  """
  
  yaml_tag = '!SimpleInference'

  @serializable_init
  def __init__(self, model_file=None, src_file=None, trg_file=None, ref_file=None, max_src_len=None,
                  post_process="none", report_path=None, report_type="html",
                  beam=1, max_len=100, len_norm_type=None, mode="onebest", batcher=Ref(Path("train.batcher"), required=False)):
    self.model_file = model_file
    self.src_file = src_file
    self.trg_file = trg_file
    self.ref_file = ref_file
    self.max_src_len = max_src_len
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
    Args:
      generator (GeneratorModel): the model to be used
      src_file (str): path of input src file to be translated
      trg_file (str): path of file where trg translatons will be written
      candidate_id_file (str): if we are doing something like retrieval where we select from fixed candidates, sometimes we want to limit our candidates to a certain subset of the full set. this setting allows us to do this.
    """
    args = dict(model_file=self.model_file, src_file=src_file or self.src_file, trg_file=trg_file or self.trg_file, ref_file=self.ref_file, max_src_len=self.max_src_len,
                  post_process=self.post_process, candidate_id_file=candidate_id_file, report_path=self.report_path, report_type=self.report_type,
                  beam=self.beam, max_len=self.max_len, len_norm_type=self.len_norm_type, mode=self.mode)

    is_reporting = issubclass(generator.__class__, Reportable) and args["report_path"] is not None
    # Corpus
    src_corpus = list(generator.src_reader.read_sents(args["src_file"]))
    # Get reference if it exists and is necessary
    if args["mode"] == "forced" or args["mode"] == "forceddebug":
      if args["ref_file"] == None:
        raise RuntimeError("When performing {} decoding, must specify reference file".format(args["mode"]))
      ref_corpus = list(generator.trg_reader.read_sents(args["ref_file"]))
      if self.max_len and any(len(s)>self.max_len for s in ref_corpus):
        logger.warning("Forced decoding with some targets being longer than max_len. Increase max_len to avoid unexpected behavior.")
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
      some_batcher = self.batcher or xnmt.batcher.InOrderBatcher(32) # Arbitrary
      if not isinstance(some_batcher, xnmt.batcher.InOrderBatcher):
        raise ValueError(f"forceddebug requires InOrderBatcher, got: {some_batcher}")
      batched_src, batched_ref = some_batcher.pack(src_corpus, ref_corpus)
      ref_scores = []
      for src, ref in zip(batched_src, batched_ref):
        dy.renew_cg(immediate_compute=settings.IMMEDIATE_COMPUTE, check_validity=settings.CHECK_VALIDITY)
        loss_expr = generator.calc_loss(src, ref, loss_calculator=LossCalculator())
        if isinstance(loss_expr.value(), Iterable):
          ref_scores.extend(loss_expr.value())
        else:
          ref_scores.append(loss_expr.value())
      ref_scores = [-x for x in ref_scores]

    # Perform generation of output
    with open(args["trg_file"], 'wt', encoding='utf-8') as fp:  # Saving the translated output to a trg file
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
            logger.error(f'Forced decoding score {output[0].score} and loss {ref_scores[i]} do not match at sentence {i}')
          output_txt = output[0].plaintext
        # Printing to trg file
        fp.write(f"{output_txt}\n")
  
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

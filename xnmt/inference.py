from collections.abc import Iterable
from typing import Optional

from xnmt.settings import settings

import dynet as dy

from xnmt.batcher import Batcher
from xnmt.model_base import GeneratorModel
from xnmt import logger
from xnmt.loss_calculator import MLELoss
import xnmt.output
from xnmt.reports import Reportable
from xnmt.persistence import serializable_init, Serializable, Ref, bare
from xnmt.search_strategy import SearchStrategy, BeamSearch
from xnmt.util import make_parent_dir

NO_DECODING_ATTEMPTED = "@@NO_DECODING_ATTEMPTED@@"

class SimpleInference(Serializable):
  """
  Main class to perform decoding.
  
  Args:
    src_file: path of input src file to be translated
    trg_file: path of file where trg translatons will be written
    ref_file: path of file with reference translations, e.g. for forced decoding
    max_src_len: Remove sentences from data to decode that are longer than this on the source side
    post_process: post-processing of translation outputs: ``none/join-char/join-bpe/join-piece``
    report_path: a path to which decoding reports will be written
    report_type: report to generate ``file/html``. Can be multiple, separate with comma.
    search_strategy: a search strategy used during decoding.
    mode: type of decoding to perform.

            * ``onebest``: generate one best.
            * ``forced``: perform forced decoding.
            * ``forceddebug``: perform forced decoding, calculate training loss, and make suer the scores are identical
              for debugging purposes.
    batcher: inference batcher, needed e.g. in connection with ``pad_src_token_to_multiple``
  """
  
  yaml_tag = '!SimpleInference'

  @serializable_init
  def __init__(self, src_file: Optional[str] = None, trg_file: Optional[str] = None, ref_file: Optional[str] = None,
               max_src_len: Optional[int] = None, post_process: str = "none", report_path: Optional[str] = None,
               report_type: str = "html", search_strategy: SearchStrategy = bare(BeamSearch), mode: str = "onebest",
               batcher: Optional[Batcher] = Ref("train.batcher", default=None)):
    self.src_file = src_file
    self.trg_file = trg_file
    self.ref_file = ref_file
    self.max_src_len = max_src_len
    self.post_process = post_process
    self.report_path = report_path
    self.report_type = report_type
    self.mode = mode
    self.batcher = batcher
    self.search_strategy = search_strategy

  def __call__(self, generator: GeneratorModel, src_file: str = None, trg_file: str = None,
               candidate_id_file: str = None):
    """
    Perform inference.

    Args:
      generator: the model to be used
      src_file: path of input src file to be translated
      trg_file: path of file where trg translatons will be written
      candidate_id_file: if we are doing something like retrieval where we select from fixed candidates, sometimes we
                         want to limit our candidates to a certain subset of the full set. this setting allows us to do
                         this.
    """
    # TODO: should be broken into smaller methods

    src_file = src_file or self.src_file
    trg_file = trg_file or self.trg_file

    is_reporting = issubclass(generator.__class__, Reportable) and self.report_path is not None
    # Corpus
    src_corpus = list(generator.src_reader.read_sents(src_file))
    # Get reference if it exists and is necessary
    if self.mode == "forced" or self.mode == "forceddebug" or self.mode == "score":
      if self.ref_file is None:
        raise RuntimeError("When performing {} decoding, must specify reference file".format(self.mode))
      score_src_corpus = []
      ref_corpus = []
      with open(self.ref_file, "r", encoding="utf-8") as fp:
        for line in fp:
          if self.mode == "score":
            nbest = line.split("|||")
            assert len(nbest) > 1, "When performing scoring, ref_file must have nbest format 'index ||| hypothesis'"
            src_index = int(nbest[0].strip())
            assert src_index < len(src_corpus),\
              f"The src_file has only {len(src_corpus)} instances, nbest file has invalid src_index {src_index}"
            score_src_corpus.append(src_corpus[src_index])
            trg_input = generator.trg_reader.read_sent(nbest[1].strip())
          else:
            trg_input = generator.trg_reader.read_sent(line)
          ref_corpus.append(trg_input)
      if self.mode == "score":
        src_corpus = score_src_corpus
    else:
      ref_corpus = None
    # Vocab
    src_vocab = generator.src_reader.vocab if hasattr(generator.src_reader, "vocab") else None
    trg_vocab = generator.trg_reader.vocab if hasattr(generator.trg_reader, "vocab") else None
    # Perform initialization
    generator.set_train(False)
    generator.initialize_generator(src_file=src_file, trg_file=trg_file, ref_file=self.ref_file,
                                   max_src_len=self.max_src_len, post_process=self.post_process,
                                   candidate_id_file=candidate_id_file, report_path=self.report_path,
                                   report_type=self.report_type, mode=self.mode)

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
    if self.mode == 'forceddebug' or self.mode == 'score':
      some_batcher = xnmt.batcher.InOrderBatcher(32) # Arbitrary
      if not isinstance(some_batcher, xnmt.batcher.InOrderBatcher):
        raise ValueError(f"forceddebug requires InOrderBatcher, got: {some_batcher}")
      batched_src, batched_ref = some_batcher.pack(src_corpus, ref_corpus)
      ref_scores = []
      for src, ref in zip(batched_src, batched_ref):
        dy.renew_cg(immediate_compute=settings.IMMEDIATE_COMPUTE, check_validity=settings.CHECK_VALIDITY)
        loss_expr = generator.calc_loss(src, ref, loss_calculator=MLELoss())
        if isinstance(loss_expr.value(), Iterable):
          ref_scores.extend(loss_expr.value())
        else:
          ref_scores.append(loss_expr.value())
      ref_scores = [-x for x in ref_scores]

    # Make the parent directory if necessary
    make_parent_dir(trg_file)

    # Perform generation of output
    if self.mode != 'score':
      with open(trg_file, 'wt', encoding='utf-8') as fp:  # Saving the translated output to a trg file
        src_ret=[]
        for i, src in enumerate(src_corpus):
          # This is necessary when the batcher does some sort of pre-processing, e.g.
          # when the batcher pads to a particular number of dimensions
          if self.batcher:
            self.batcher.add_single_batch(src_curr=[src], trg_curr=None, src_ret=src_ret, trg_ret=None)
            src = src_ret.pop()[0]
          # Do the decoding
          if self.max_src_len is not None and len(src) > self.max_src_len:
            output_txt = NO_DECODING_ATTEMPTED
          else:
            dy.renew_cg(immediate_compute=settings.IMMEDIATE_COMPUTE, check_validity=settings.CHECK_VALIDITY)
            ref_ids = ref_corpus[i] if ref_corpus is not None else None
            output = generator.generate_output(src, i, forced_trg_ids=ref_ids, search_strategy=self.search_strategy)
            # If debugging forced decoding, make sure it matches
            if ref_scores is not None and (abs(output[0].score - ref_scores[i]) / abs(ref_scores[i])) > 1e-5:
              logger.error(f'Forced decoding score {output[0].score} and loss {ref_scores[i]} do not match at sentence {i}')
            output_txt = output[0].plaintext
          # Printing to trg file
          fp.write(f"{output_txt}\n")
    else:
      with open(trg_file, 'wt', encoding='utf-8') as fp:
        with open(self.ref_file, "r", encoding="utf-8") as nbest_fp:
          for nbest, score in zip(nbest_fp, ref_scores):
            fp.write("{} ||| score={}\n".format(nbest.strip(), score))
  
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

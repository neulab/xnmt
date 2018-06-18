from collections.abc import Iterable
from typing import Optional

from xnmt.settings import settings

import dynet as dy

from xnmt.batcher import Batcher
from xnmt.model_base import GeneratorModel
from xnmt.loss_calculator import MLELoss
import xnmt.output
from xnmt.reports import Reportable
from xnmt.persistence import serializable_init, Serializable, Ref, bare
from xnmt.search_strategy import SearchStrategy, BeamSearch
from xnmt.util import make_parent_dir

NO_DECODING_ATTEMPTED = "@@NO_DECODING_ATTEMPTED@@"

class Inference(object):
  """
  A template class for classes that perform inference.
  """

  def __call__(self, generator: GeneratorModel, src_file: str = None, trg_file: str = None,
               candidate_id_file: str = None) -> None:
    """
    Perform inference by reading inputs from ``src_file`` and writing out the hypotheses to ``trg_file``.

    Args:
      generator: the model to be used
      src_file: path of input src file to be translated
      trg_file: path of file where trg translatons will be written
      candidate_id_file: if we are doing something like retrieval where we select from fixed candidates, sometimes we
                         want to limit our candidates to a certain subset of the full set. this setting allows us to do
                         this.
    """
    raise NotImplementedError("to be implemented by subclasses")

class ClassifierInference(Inference, Serializable):
  yaml_tag = "!ClassifierInference"

  @serializable_init
  def __init__(self, batcher=Ref("train.batcher", default=None), post_process: str = "none",):
    self.batcher = batcher
    self.post_processor = xnmt.output.OutputProcessor.get_output_processor(post_process)

  def __call__(self, generator, src_file=None, trg_file=None, candidate_id_file=None):
    src_corpus = list(generator.src_reader.read_sents(src_file))
    with open(trg_file, 'wt', encoding='utf-8') as fp:  # Saving the translated output to a trg file
      for i, src in enumerate(src_corpus):
        # This is necessary when the batcher does some sort of pre-processing, e.g.
        # when the batcher pads to a particular number of dimensions
        if self.batcher:
          src = self.batcher.create_single_batch(src_sents=[src])[0]

        dy.renew_cg(immediate_compute=settings.IMMEDIATE_COMPUTE, check_validity=settings.CHECK_VALIDITY)
        output = generator.generate(src, i)
        self.post_processor.process_outputs(output)
        output_txt = output[0].plaintext
        fp.write(f"{output_txt}\n")

class AutoRegressiveInference(Inference, Serializable):
  """
  Main class to perform decoding.

  Args:
    src_file: path of input src file to be translated
    trg_file: path of file where trg translatons will be written
    ref_file: path of file with reference translations, e.g. for forced decoding
    max_src_len: Remove sentences from data to decode that are longer than this on the source side
    max_num_sents:
    post_process: post-processing of translation outputs: ``none/join-char/join-bpe/join-piece``
    report_path: a path to which decoding reports will be written
    report_type: report to generate ``file/html``. Can be multiple, separate with comma.
    search_strategy: a search strategy used during decoding.
    mode: type of decoding to perform.

            * ``onebest``: generate one best.
            * ``forced``: perform forced decoding.
            * ``forceddebug``: perform forced decoding, calculate training loss, and make sure the scores are identical
              for debugging purposes.
            * ``score``: output scores, useful for rescoring
    batcher: inference batcher, needed e.g. in connection with ``pad_src_token_to_multiple``
  """

  yaml_tag = '!AutoRegressiveInference'

  @serializable_init
  def __init__(self, src_file: Optional[str] = None, trg_file: Optional[str] = None, ref_file: Optional[str] = None,
               max_src_len: Optional[int] = None, max_num_sents: Optional[int] = None, post_process: str = "none",
               report_path: Optional[str] = None, report_type: str = "html",
               search_strategy: SearchStrategy = bare(BeamSearch), mode: str = "onebest",
               batcher: Optional[Batcher] = Ref("train.batcher", default=None)):
    self.src_file = src_file
    self.trg_file = trg_file
    self.ref_file = ref_file
    self.max_src_len = max_src_len
    self.max_num_sents = max_num_sents
    self.post_processor = xnmt.output.OutputProcessor.get_output_processor(post_process)
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
    src_file = src_file or self.src_file
    trg_file = trg_file or self.trg_file

    is_reporting = issubclass(generator.__class__, Reportable) and self.report_path is not None

    ref_corpus, src_corpus = self._read_corpus(generator, src_file)

    src_vocab = generator.src_reader.vocab if hasattr(generator.src_reader, "vocab") else None
    trg_vocab = generator.trg_reader.vocab if hasattr(generator.trg_reader, "vocab") else None

    self._init_generator(candidate_id_file, generator, is_reporting, src_file, src_vocab, trg_file, trg_vocab)

    ref_scores = None
    if self.mode == 'forceddebug' or self.mode == 'score':
      ref_scores = self._compute_losses(generator, ref_corpus, src_corpus)

    make_parent_dir(trg_file)

    if self.mode == 'score':
      self._rescore_output(ref_scores, trg_file)
    else:
      self._generate_output(generator, ref_corpus, ref_scores, src_corpus, trg_file)

  def _rescore_output(self, ref_scores, trg_file):
    with open(trg_file, 'wt', encoding='utf-8') as fp:
      with open(self.ref_file, "r", encoding="utf-8") as nbest_fp:
        for nbest, score in zip(nbest_fp, ref_scores):
          fp.write("{} ||| score={}\n".format(nbest.strip(), score))

  def _generate_output(self, generator, ref_corpus, ref_scores, src_corpus, trg_file):
    with open(trg_file, 'wt', encoding='utf-8') as fp:  # Saving the translated output to a trg file
      for i, src in enumerate(src_corpus):
        # This is necessary when the batcher does some sort of pre-processing, e.g.
        # when the batcher pads to a particular number of dimensions
        if self.batcher:
          src = self.batcher.create_single_batch(src_sents=[src])[0]
        # Do the decoding
        if self.max_src_len is not None and len(src) > self.max_src_len:
          output_txt = NO_DECODING_ATTEMPTED
        else:
          dy.renew_cg(immediate_compute=settings.IMMEDIATE_COMPUTE, check_validity=settings.CHECK_VALIDITY)
          ref_ids = ref_corpus[i] if ref_corpus is not None else None
          output = generator.generate(src, i, forced_trg_ids=ref_ids, search_strategy=self.search_strategy)
          self.post_processor.process_outputs(output)
          # If debugging forced decoding, make sure it matches
          if ref_scores is not None and (abs(output[0].score - ref_scores[i]) / abs(ref_scores[i])) > 1e-5:
            raise ValueError(
              f'Forced decoding score {output[0].score} and loss {ref_scores[i]} do not match at sentence {i}')
          output_txt = output[0].plaintext
        # Printing to trg file
        fp.write(f"{output_txt}\n")

  def _compute_losses(self, generator, ref_corpus, src_corpus):
    some_batcher = xnmt.batcher.InOrderBatcher(32)  # Arbitrary
    if not isinstance(some_batcher, xnmt.batcher.InOrderBatcher):
      raise ValueError(f"modes 'forceddebug' and 'score' require InOrderBatcher, got: {some_batcher}")
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
    return ref_scores

  def _init_generator(self, candidate_id_file, generator, is_reporting, src_file, src_vocab, trg_file, trg_vocab):
    generator.set_train(False)
    generator.initialize_generator(src_file=src_file, trg_file=trg_file, ref_file=self.ref_file,
                                   max_src_len=self.max_src_len, candidate_id_file=candidate_id_file,
                                   report_path=self.report_path, report_type=self.report_type, mode=self.mode)
    if hasattr(generator, "set_trg_vocab"):
      generator.set_trg_vocab(trg_vocab)
    if hasattr(generator, "set_reporting_src_vocab"):
      generator.set_reporting_src_vocab(src_vocab)
    if is_reporting:
      generator.set_report_resource("src_vocab", src_vocab)
      generator.set_report_resource("trg_vocab", trg_vocab)

  def _read_corpus(self, generator, src_file):
    src_corpus = list(generator.src_reader.read_sents(src_file))
    # Get reference if it exists and is necessary
    if self.mode == "forced" or self.mode == "forceddebug" or self.mode == "score":
      if self.ref_file is None:
        raise RuntimeError(f"When performing '{self.mode}' decoding, must specify reference file")
      score_src_corpus = []
      ref_corpus = []
      with open(self.ref_file, "r", encoding="utf-8") as fp:
        for line in fp:
          if self.mode == "score":
            nbest = line.split("|||")
            assert len(nbest) > 1, "When performing scoring, ref_file must have nbest format 'index ||| hypothesis'"
            src_index = int(nbest[0].strip())
            assert src_index < len(src_corpus), \
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
    return ref_corpus, src_corpus


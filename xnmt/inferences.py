import collections.abc
from itertools import islice
from typing import List, Optional, Tuple, Sequence, Union
from typing.io import TextIO
import numbers
import shutil
import contextlib

import xnmt
import xnmt.tensor_tools as tt
from xnmt import batchers, event_trigger
from xnmt import events, logger, losses, loss_calculators, output, reports, search_strategies, sent, utils
from xnmt.models import base as models
from xnmt.persistence import serializable_init, Serializable, bare
from xnmt.settings import settings

if xnmt.backend_torch:
  import torch

NO_DECODING_ATTEMPTED = "@@NO_DECODING_ATTEMPTED@@"

class Inference(object):
  """
  A template class for classes that perform inference.

  Args:
    src_file: path of input src file to be translated
    trg_file: path of file where trg translatons will be written
    ref_file: path of file with reference translations, e.g. for forced decoding
    max_src_len: Remove sentences from data to decode that are longer than this on the source side
    max_num_sents: Stop decoding after the first n sentences.
    mode: type of decoding to perform.

            * ``onebest``: generate one best.
            * ``score``: output scores, useful for rescoring
            * ``forced``: perform forced decoding.
            * ``forceddebug``: perform forced decoding, calculate training loss, and make sure the scores are identical
              for debugging purposes.
    batcher: inference batcher, needed e.g. in connection with ``pad_src_token_to_multiple``
    reporter: a reporter to create reports for each decoded sentence
  """
  # TODO: Support k-best inference?
  @events.register_xnmt_handler
  def __init__(self,
               src_file: Optional[str] = None,
               trg_file: Optional[str] = None,
               ref_file: Optional[str] = None,
               max_src_len: Optional[int] = None,
               max_num_sents: Optional[int] = None,
               mode: str = "onebest",
               batcher: batchers.InOrderBatcher = bare(batchers.InOrderBatcher, batch_size=1),
               reporter: Union[None, reports.Reporter, Sequence[reports.Reporter]] = None) -> None:
    self.src_file = src_file
    self.trg_file = trg_file
    self.ref_file = ref_file
    self.max_src_len = max_src_len
    self.max_num_sents = max_num_sents
    self.mode = mode
    self.batcher = batcher
    self.reporter = reporter

  def generate_one(self, generator: 'models.GeneratorModel', src: batchers.Batch) \
          -> List[sent.ReadableSentence]:
    raise NotImplementedError("must be implemented by subclasses")

  def compute_losses_one(self, generator: 'models.GeneratorModel', src: sent.Sentence,
                         ref: sent.Sentence) -> losses.FactoredLossExpr:
    raise NotImplementedError("must be implemented by subclasses")

  def perform_inference(self, generator: 'models.GeneratorModel', src_file: str = None, trg_file: str = None, ref_file: str = None) \
          -> None:
    """
    Perform inference.

    Args:
      generator: the model to be used
      src_file: path of input src file to be translated
      trg_file: path of file where trg translatons will be written
    """
    src_file = src_file or self.src_file
    trg_file = trg_file or self.trg_file
    ref_file = ref_file or self.ref_file

    if trg_file is not None:
      utils.make_parent_dir(trg_file)

    if ref_file is not None:
      logger.info(f'Performing inference on {src_file} and {ref_file}')
    else:
      logger.info(f'Performing inference on {src_file}')

    event_trigger.set_train(False)

    ref_scores = None

    if self.mode in ['score', 'forceddebug']:
      ref_corpus, src_corpus = self._read_corpus(generator, src_file, mode=self.mode, ref_file=self.ref_file)
      ref_scores = self._compute_losses(generator, ref_corpus, src_corpus, self.max_num_sents)

    if self.mode == 'score':
      self._write_rescored_output(ref_scores, self.ref_file, trg_file)
    elif self.mode == 'forced' or self.mode == 'forceddebug':
      self._forced_decode(generator=generator, src_file=src_file,
                          ref_file=ref_file, batcher=self.batcher,
                          max_src_len=self.max_src_len,
                          assert_scores=ref_scores)
      if trg_file is not None:
        shutil.copyfile(ref_file, trg_file)
    else:
      self._generate_output(generator=generator,
                            src_file=src_file, trg_file=trg_file, batcher=self.batcher,
                            max_src_len=self.max_src_len)

  def _generate_one_batch(self, generator: 'models.GeneratorModel',
                                batcher: Optional[batchers.Batcher] = None,
                                src_batch: batchers.Batch = None,
                                max_src_len: Optional[int] = None,
                                fp: TextIO = None):
    """
    Generate outputs for a single batch and write them to the output file.
    """
    batch_size = len(src_batch)
    src_batches = batcher.pack(src_batch, None)
    src_batch = src_batches[0]
    src_len = src_batch.sent_len()
    if max_src_len is not None and src_len > max_src_len:
      output_txt = "\n".join([NO_DECODING_ATTEMPTED] * batch_size)
      fp.write(f"{output_txt}\n")
    else:
      with utils.ReportOnException({"src": src_batch, "graph": utils.print_cg_conditional}):
        tt.reset_graph()
        with torch.no_grad() if xnmt.backend_torch else utils.dummy_context_mgr():
          outputs = self.generate_one(generator, src_batch)
        if self.reporter: self._create_sent_report()
        for i in range(len(outputs)):
          output_txt = outputs[i].sent_str(custom_output_procs=self.post_processor)
          fp.write(f"{output_txt}\n")

  def _generate_output(self, generator: 'models.GeneratorModel', src_file: str,
                       trg_file: str, batcher: Optional[batchers.Batcher] = None, max_src_len: Optional[int] = None) -> None:
    """
    Generate outputs and write them to file.

    Args:
      generator: generator model to use
      src_file: a file of src-side inputs to generate outputs for
      trg_file: file to write outputs to
      batcher: necessary with some cases of input pre-processing such as padding or truncation
      max_src_len: if given, skip inputs that are too long
    """
    src_in = generator.src_reader.read_sents(src_file)

    # Reporting is commenced if there is some defined reporters
    is_reporting = self.reporter is not None
    event_trigger.set_reporting(is_reporting)

    # Saving the translated output to a trg file
    with open(trg_file, 'wt', encoding='utf-8') as fp:
      src_batch = []
      for curr_sent_i, src_line in islice(enumerate(src_in), self.max_num_sents):
        src_batch.append(src_line)
        if len(src_batch) == batcher.batch_size:
          self._generate_one_batch(generator, batcher, src_batch, max_src_len, fp)
          src_batch = []
        if settings.PRETEND: break
      if len(src_batch) != 0:
        self._generate_one_batch(generator, batcher, src_batch, max_src_len, fp)

    # Finishing up
    try:
      if is_reporting:
        self._conclude_report()
    finally:
      # Reporting is done in _generate_output only
      event_trigger.set_reporting(False)

  def _forced_decode_one_batch(self, generator: 'models.GeneratorModel',
                               batcher: Optional[batchers.Batcher] = None,
                               src_batch: batchers.Batch = None,
                               ref_batch: batchers.Batch = None,
                               assert_scores: batchers.Batch = None,
                               max_src_len: Optional[int] = None):
    """
    Performs forced decoding for a single batch.
    """
    batch_size = len(src_batch)
    src_batches, ref_batches = batcher.pack(src_batch, ref_batch)
    src_batch = src_batches[0]
    src_len = src_batch.sent_len()

    if max_src_len is None or src_len <= max_src_len is not None and src_len > max_src_len:
      with utils.ReportOnException({"src": src_batch, "graph": utils.print_cg_conditional}):
        tt.reset_graph()
        outputs = self.generate_one(generator, src_batch)
        if self.reporter: self._create_sent_report()
        for i in range(len(outputs)):
          if assert_scores is not None:
            # If debugging forced decoding, make sure it matches
            assert batch_size == len(outputs), "debug forced decoding not supported with nbest inference"
            if (abs(outputs[i].score - assert_scores[i]) / abs(assert_scores[i])) > 1e-5:
              raise ValueError(
                f'Forced decoding score {outputs[i].score} and loss {assert_scores[i]} do not match at '
                f'sentence {i}')

  def _forced_decode(self, generator: 'models.GeneratorModel', src_file: str,
                      ref_file: str, batcher: Optional[batchers.Batcher] = None, max_src_len: Optional[int] = None,
                      assert_scores: Optional[Sequence[numbers.Real]] = None) -> None:
    """
    Perform forced decoding.

    Args:
      generator: generator model to use
      src_file: a file of src-side inputs to generate outputs for
      ref_file: path of file with reference translations
      batcher: necessary with some cases of input pre-processing such as padding or truncation
      max_src_len: if given, skip inputs that are too long
      assert_scores: if given, raise exception if the scores for generated outputs don't match the given scores
    """
    src_in = generator.src_reader.read_sents(src_file)

    # If we have a "assert scores" list return it, otherwise return "None" infinitely
    assert_in = assert_scores if assert_scores else iter(lambda: None, 1)

    # Reporting is commenced if there is some defined reporters
    is_reporting = self.reporter is not None
    event_trigger.set_reporting(is_reporting)

    # Saving the translated output to a trg file
    src_batch, ref_batch, assert_batch = [], [], []
    for curr_sent_i, (src_line, assert_line) in islice(enumerate(zip(src_in, assert_in)), self.max_num_sents):
      src_batch.append(src_line)
      assert_batch.append(assert_line)
      if len(src_batch) == batcher.batch_size:
        self._forced_decode_one_batch(generator, batcher, src_batch, assert_batch, max_src_len)
        src_batch, ref_batch, assert_batch = [], [], []
    if len(src_batch) != 0:
      self._forced_decode_one_batch(generator, batcher, src_batch, assert_batch, max_src_len)

    # Finishing up
    try:
      if is_reporting:
        self._conclude_report()
    finally:
      # Reporting is done in _generate_output only
      event_trigger.set_reporting(False)

  def _create_sent_report(self):
    assert self.reporter is not None
    if not isinstance(self.reporter, collections.abc.Iterable):
      self.reporter = [self.reporter]
    report_context = event_trigger.get_report_input(context=reports.ReportInfo())
    for report_input in report_context.sent_info:
      for reporter in self.reporter:
        reporter.create_sent_report(**report_input, **report_context.glob_info)

  def _conclude_report(self):
    assert self.reporter is not None
    if not isinstance(self.reporter, collections.abc.Iterable):
      self.reporter = [self.reporter]
    for reporter in self.reporter:
      reporter.conclude_report()

  def _compute_losses(self, generator, ref_corpus, src_corpus, max_num_sents) -> List[numbers.Real]:
    batched_src, batched_ref = self.batcher.pack(src_corpus, ref_corpus)
    ref_scores = []
    for sent_count, (src, ref) in enumerate(zip(batched_src, batched_ref)):
      if max_num_sents and sent_count >= max_num_sents: break
      tt.reset_graph()
      loss = self.compute_losses_one(generator, src, ref)
      if isinstance(loss.value(), collections.abc.Iterable):
        ref_scores.extend(loss.value())
      else:
        ref_scores.append(loss.value())
    ref_scores = [-x for x in ref_scores]
    return ref_scores


  @staticmethod
  def _write_rescored_output(ref_scores: Sequence[numbers.Real], ref_file: str, trg_file: str) -> None:
    """
    Write scored sequences and scores to file when mode=='score'.

    Args:
      ref_scores: list of score values
      ref_file: filename where sequences to be scored a read from
      trg_file: filename to write to
    """
    with open(trg_file, 'wt', encoding='utf-8') as fp:
      with open(ref_file, "r", encoding="utf-8") as nbest_fp:
        for nbest, score in zip(nbest_fp, ref_scores):
          fp.write("{} ||| score={}\n".format(nbest.strip(), score))

  @staticmethod
  def _read_corpus(generator: 'models.GeneratorModel', src_file: str, mode: str, ref_file: str) -> Tuple[List, List]:
    src_corpus = list(generator.src_reader.read_sents(src_file))
    # Get reference if it exists and is necessary
    if mode == "forced" or mode == "forceddebug" or mode == "score":
      if ref_file is None:
        raise RuntimeError(f"When performing '{mode}' decoding, must specify reference file")
      score_src_corpus = []
      ref_corpus = []
      with open(ref_file, "r", encoding="utf-8") as fp:
        for idx, line in enumerate(fp):
          if mode == "score":
            nbest = line.split("|||")
            assert len(nbest) > 1, "When performing scoring, ref_file must have nbest format 'index ||| hypothesis'"
            src_index = int(nbest[0].strip())
            assert src_index < len(src_corpus), \
              f"The src_file has only {len(src_corpus)} instances, nbest file has invalid src_index {src_index}"
            score_src_corpus.append(src_corpus[src_index])
            trg_input = generator.trg_reader.read_sent(idx=idx, line=nbest[1].strip())
          else:
            trg_input = generator.trg_reader.read_sent(idx=idx, line=line)
          ref_corpus.append(trg_input)
      if mode == "score":
        src_corpus = score_src_corpus
    else:
      ref_corpus = None
    return ref_corpus, src_corpus


class IndependentOutputInference(Inference, Serializable):
  """
  Inference when outputs are produced independently, including for classifiers that produce only a single output.

  Assumes that generator.generate() takes arguments src, idx

  Args:
    src_file: path of input src file to be translated
    trg_file: path of file where trg translatons will be written
    ref_file: path of file with reference translations, e.g. for forced decoding
    max_src_len: Remove sentences from data to decode that are longer than this on the source side
    max_num_sents: Stop decoding after the first n sentences.
    post_process: post-processing of translation outputs (available string shortcuts:  ``none``, ``join-char``,
                  ``join-bpe``, ``join-piece``)
    mode: type of decoding to perform.

          * ``onebest``: generate one best.
          * ``score``: output scores, useful for rescoring

    batcher: inference batcher, needed e.g. in connection with ``pad_src_token_to_multiple``
    reporter: a reporter to create reports for each decoded sentence
  """
  yaml_tag = "!IndependentOutputInference"

  @serializable_init
  def __init__(self, src_file: Optional[str] = None, trg_file: Optional[str] = None, ref_file: Optional[str] = None,
               max_src_len: Optional[int] = None, max_num_sents: Optional[int] = None,
               post_process: Union[None, str, output.OutputProcessor, Sequence[output.OutputProcessor]] = None,
               mode: str = "onebest",
               batcher: batchers.InOrderBatcher = bare(batchers.InOrderBatcher, batch_size=1),
               reporter: Union[None, reports.Reporter, Sequence[reports.Reporter]] = None):
    super().__init__(src_file=src_file, trg_file=trg_file, ref_file=ref_file, max_src_len=max_src_len,
                     max_num_sents=max_num_sents, mode=mode, batcher=batcher, reporter=reporter)
    self.post_processor = output.OutputProcessor.get_output_processor(post_process) or None

  def generate_one(self,
                   generator: 'models.GeneratorModel',
                   src: batchers.Batch) -> List[sent.Sentence]:
    outputs = generator.generate(src)
    return outputs

  def compute_losses_one(self,
                         generator: 'models.GeneratorModel',
                         src: sent.Sentence,
                         ref: sent.Sentence) -> losses.FactoredLossExpr:
    loss_expr = loss_calculators.MLELoss().calc_loss(generator, src, ref)
    return loss_expr

class AutoRegressiveInference(Inference, Serializable):
  """
  Performs inference for auto-regressive models that expand based on their own previous outputs.

  Assumes that generator.generate() takes arguments src, idx, search_strategy, forced_trg_ids

  Args:
    src_file: path of input src file to be translated
    trg_file: path of file where trg translatons will be written
    ref_file: path of file with reference translations, e.g. for forced decoding
    max_src_len: Remove sentences from data to decode that are longer than this on the source side
    max_num_sents: Stop decoding after the first n sentences.
    post_process: post-processing of translation outputs
                  (available string shortcuts:  ``none``,``join-char``,``join-bpe``,``join-piece``)
    search_strategy: a search strategy used during decoding.
    mode: type of decoding to perform.

            * ``onebest``: generate one best.
            * ``score``: output scores, useful for rescoring
    batcher: inference batcher, needed e.g. in connection with ``pad_src_token_to_multiple``
    reporter: a reporter to create reports for each decoded sentence
  """

  yaml_tag = '!AutoRegressiveInference'

  @serializable_init
  def __init__(self,
               src_file: Optional[str] = None,
               trg_file: Optional[str] = None,
               ref_file: Optional[str] = None,
               max_src_len: Optional[int] = None,
               max_num_sents: Optional[int] = None,
               post_process: Union[str, output.OutputProcessor, Sequence[output.OutputProcessor]] = [],
               search_strategy: search_strategies.SearchStrategy = bare(search_strategies.BeamSearch),
               mode: str = "onebest",
               batcher: batchers.InOrderBatcher = bare(batchers.InOrderBatcher, batch_size=1),
               reporter: Union[None, reports.Reporter, Sequence[reports.Reporter]] = None) -> None:
    super().__init__(src_file=src_file, trg_file=trg_file, ref_file=ref_file, max_src_len=max_src_len,
                     max_num_sents=max_num_sents, mode=mode, batcher=batcher, reporter=reporter)

    self.post_processor = output.OutputProcessor.get_output_processor(post_process) or None
    self.search_strategy = search_strategy

  def generate_one(self,
                   generator: 'models.GeneratorModel',
                   src: batchers.Batch) -> List[sent.Sentence]:
    outputs = generator.generate(src, search_strategy=self.search_strategy)
    return outputs

  def compute_losses_one(self,
                         generator: 'models.GeneratorModel',
                         src: sent.Sentence,
                         ref: sent.Sentence) -> losses.FactoredLossExpr:
    loss_expr = loss_calculators.MLELoss().calc_loss(generator, src, ref)
    return loss_expr

class CascadeInference(Inference, Serializable):
  """Inference class that performs inference as a series of independent inference steps.

  Steps are performed using a list of inference sub-objects and a list of models.
  Intermediate outputs are written out to disk and then read by the next time step.

  The generator passed to ``perform_inference`` must be a :class:`xnmt.models.CascadeGenerator`.

  Args:
    steps: list of inference objects
  """

  yaml_tag = "!CascadeInference"
  @serializable_init
  def __init__(self, steps: Sequence[Inference]) -> None:
    self.steps = steps

  def perform_inference(self, generator: 'models.CascadeGenerator', src_file: str = None, trg_file: str = None, ref_file: str = None) \
          -> None:
    assert isinstance(generator, models.CascadeGenerator)
    assert len(generator.generators) == len(self.steps)
    src_files = [src_file] + [f"{trg_file}.tmp.{i}" for i in range(len(self.steps)-1)]
    trg_files = src_files[1:] + [trg_file]
    for step_i, step in enumerate(self.steps):
      step.perform_inference(generator=generator.generators[step_i],
                             src_file=src_files[step_i],
                             trg_file=trg_files[step_i])

  def compute_losses_one(self, *args, **kwargs):
    raise ValueError("cannot call CascadedInference.compute_losses_one() directly, use the sub-inference objects")
  def generate_one(self, *args, **kwargs):
    raise ValueError("cannot call CascadedInference.generate_one() directly, use the sub-inference objects")

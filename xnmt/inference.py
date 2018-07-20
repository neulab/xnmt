import collections.abc
from typing import List, Optional, Tuple, Sequence, Union

from xnmt.settings import settings

import dynet as dy

import xnmt.batcher
from xnmt import events, loss, loss_calculator, model_base, output, reports, search_strategy, util
from xnmt import logger, loss, loss_calculator, model_base, output, reports, search_strategy, util
from xnmt.persistence import serializable_init, Serializable, bare

NO_DECODING_ATTEMPTED = "@@NO_DECODING_ATTEMPTED@@"

class Inference(reports.Reportable):
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
            * ``forced``: perform forced decoding.
            * ``forceddebug``: perform forced decoding, calculate training loss, and make sure the scores are identical
              for debugging purposes.
            * ``score``: output scores, useful for rescoring
    batcher: inference batcher, needed e.g. in connection with ``pad_src_token_to_multiple``
    reporter: a reporter to create reports for each decoded sentence
  """
  @events.register_xnmt_handler
  def __init__(self, src_file: Optional[str] = None, trg_file: Optional[str] = None, ref_file: Optional[str] = None,
               max_src_len: Optional[int] = None, max_num_sents: Optional[int] = None,
               mode: str = "onebest",
               batcher: xnmt.batcher.InOrderBatcher = bare(xnmt.batcher.InOrderBatcher, batch_size=1),
               reporter: Union[None, reports.Reporter, Sequence[reports.Reporter]] = None):
    self.src_file = src_file
    self.trg_file = trg_file
    self.ref_file = ref_file
    self.max_src_len = max_src_len
    self.max_num_sents = max_num_sents
    self.mode = mode
    self.batcher = batcher
    self.reporter = reporter

  def generate_one(self, generator: 'model_base.GeneratorModel', src: xnmt.batcher.Batch, src_i: int, forced_ref_ids) \
          -> List[output.Output]:
    raise NotImplementedError("must be implemented by subclasses")

  def compute_losses_one(self, generator: 'model_base.GeneratorModel', src: xnmt.input.Input,
                         ref: xnmt.input.Input) -> loss.FactoredLossExpr:
    raise NotImplementedError("must be implemented by subclasses")


  def perform_inference(self, generator: 'model_base.GeneratorModel', src_file: str = None, trg_file: str = None,
                        ref_file_to_report=None):
    """
    Perform inference.

    Args:
      generator: the model to be used
      src_file: path of input src file to be translated
      trg_file: path of file where trg translatons will be written
    """
    src_file = src_file or self.src_file
    trg_file = trg_file or self.trg_file
    util.make_parent_dir(trg_file)

    logger.info(f'Performing inference on {src_file}')

    ref_corpus, src_corpus = self._read_corpus(generator, src_file, mode=self.mode, ref_file=self.ref_file)

    generator.set_train(False)

    ref_scores = None
    if self.mode == 'score':
      ref_scores = self._compute_losses(generator, ref_corpus, src_corpus, self.max_num_sents)
      self._write_rescored_output(ref_scores, self.ref_file, trg_file)

    if self.mode == 'forceddebug':
      ref_scores = self._compute_losses(generator, ref_corpus, src_corpus, self.max_num_sents)

    if self.mode != 'score':
      self._generate_output(generator=generator, forced_ref_corpus=ref_corpus, assert_scores=ref_scores,
                            src_corpus=src_corpus, trg_file=trg_file, batcher=self.batcher,
                            max_src_len=self.max_src_len, ref_file_to_report=ref_file_to_report)
    self.end_inference()


  def _generate_output(self, generator: 'model_base.GeneratorModel', src_corpus: Sequence[xnmt.input.Input],
                       trg_file: str, batcher: Optional[xnmt.batcher.Batcher] = None, max_src_len: Optional[int] = None,
                       forced_ref_corpus: Optional[Sequence[xnmt.input.Input]] = None,
                       assert_scores: Optional[Sequence[float]] = None,
                       ref_file_to_report: Union[None,str,Sequence[str]] = None) -> None:
    """
    Generate outputs and write them to file.

    Args:
      generator: generator model to use
      src_corpus: src-side inputs to generate outputs for
      trg_file: file to write outputs to
      batcher: necessary with some cases of input pre-processing such as padding or truncation
      max_src_len: if given, skip inputs that are too long
      forced_ref_corpus: if given, perform forced decoding with the given trg-side inputs
      assert_scores: if given, raise exception if the scores for generated outputs don't match the given scores
      ref_file_to_report: if given, report reference file line by line so that the reference can be included in a report
    """
    with open(trg_file, 'wt', encoding='utf-8') as fp:  # Saving the translated output to a trg file
      if forced_ref_corpus:
        src_batches, ref_batches = batcher.pack(src_corpus, forced_ref_corpus)
      else:
        src_batches = batcher.pack(src_corpus, None)
      cur_sent_i = 0
      ref_batch = None
      if ref_file_to_report:
        if isinstance(ref_file_to_report, list): ref_file_to_report = ref_file_to_report[0]
        ref_file = open(ref_file_to_report)
      for batch_i, src_batch in enumerate(src_batches):
        batch_size = src_batch.batch_size()
        if ref_file_to_report:
          for _ in range(batch_size):
            ref_sent = ref_file.readline().strip()
          self.add_sent_for_report({"reference": ref_sent, "output_proc": self.post_processor})
        src_len = src_batch.sent_len()
        if max_src_len is not None and src_len > max_src_len:
          output_txt = "\n".join([NO_DECODING_ATTEMPTED] * batch_size)
          fp.write(f"{output_txt}\n")
        else:
          if forced_ref_corpus: ref_batch = ref_batches[batch_i]
          dy.renew_cg(immediate_compute=settings.IMMEDIATE_COMPUTE, check_validity=settings.CHECK_VALIDITY)
          outputs = self.generate_one(generator, src_batch, range(cur_sent_i,cur_sent_i+batch_size), ref_batch)
          if self.reporter: self._create_report()
          for i in range(len(outputs)):
            if assert_scores is not None:
              # If debugging forced decoding, make sure it matches
              assert batch_size == len(outputs), "debug forced decoding not supported with nbest inference"
              if (abs(outputs[i].score - assert_scores[cur_sent_i + i]) / abs(assert_scores[cur_sent_i + i])) > 1e-5:
                raise ValueError(
                  f'Forced decoding score {outputs[0].score} and loss {assert_scores[cur_sent_i + i]} do not match at '
                  f'sentence {cur_sent_i + i}')
            output_txt = outputs[i].apply_post_processor(self.post_processor)
            fp.write(f"{output_txt}\n")
        cur_sent_i += batch_size
        if self.max_num_sents and cur_sent_i >= self.max_num_sents: break
      if ref_file_to_report:
        ref_file.close()

  @events.register_xnmt_event
  def end_inference(self):
    pass

  def _create_report(self):
    assert self.reporter is not None
    if not isinstance(self.reporter, collections.abc.Iterable):
      self.reporter = [self.reporter]
    report_inputs = self.reporter[0].get_report_input(context={})
    for report_input in report_inputs:
      for reporter in self.reporter:
        reporter.create_report(**report_input)

  def _compute_losses(self, generator, ref_corpus, src_corpus, max_num_sents) -> List[float]:
    batched_src, batched_ref = self.batcher.pack(src_corpus, ref_corpus)
    ref_scores = []
    for sent_count, (src, ref) in enumerate(zip(batched_src, batched_ref)):
      if max_num_sents and sent_count >= max_num_sents: break
      dy.renew_cg(immediate_compute=settings.IMMEDIATE_COMPUTE, check_validity=settings.CHECK_VALIDITY)
      loss_expr = self.compute_losses_one(generator, src, ref)
      if isinstance(loss_expr.value(), collections.abc.Iterable):
        ref_scores.extend(loss_expr.value())
      else:
        ref_scores.append(loss_expr.value())
    ref_scores = [-x for x in ref_scores]
    return ref_scores


  @staticmethod
  def _write_rescored_output(ref_scores: Sequence[float], ref_file: str, trg_file: str) -> None:
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
  def _read_corpus(generator: 'model_base.GeneratorModel', src_file: str, mode: str, ref_file: str) -> Tuple[List, List]:
    src_corpus = list(generator.src_reader.read_sents(src_file))
    # Get reference if it exists and is necessary
    if mode == "forced" or mode == "forceddebug" or mode == "score":
      if ref_file is None:
        raise RuntimeError(f"When performing '{mode}' decoding, must specify reference file")
      score_src_corpus = []
      ref_corpus = []
      with open(ref_file, "r", encoding="utf-8") as fp:
        for line in fp:
          if mode == "score":
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
          * ``forced``: perform forced decoding.
          * ``forceddebug``: perform forced decoding, calculate training loss, and make sure the scores are identical
            for debugging purposes.
          * ``score``: output scores, useful for rescoring

    batcher: inference batcher, needed e.g. in connection with ``pad_src_token_to_multiple``
  """
  yaml_tag = "!IndependentOutputInference"

  @serializable_init
  def __init__(self, src_file: Optional[str] = None, trg_file: Optional[str] = None, ref_file: Optional[str] = None,
               max_src_len: Optional[int] = None, max_num_sents: Optional[int] = None,
               post_process: Union[str, output.OutputProcessor] = bare(output.PlainTextOutputProcessor),
               mode: str = "onebest",
               batcher: xnmt.batcher.InOrderBatcher = bare(xnmt.batcher.InOrderBatcher, batch_size=1),
               reporter: Union[None, reports.Reporter, Sequence[reports.Reporter]] = None):
    super().__init__(src_file=src_file, trg_file=trg_file, ref_file=ref_file, max_src_len=max_src_len,
                     max_num_sents=max_num_sents, mode=mode, batcher=batcher, reporter=reporter)
    self.post_processor = output.OutputProcessor.get_output_processor(post_process)

  def generate_one(self, generator: 'model_base.GeneratorModel', src: xnmt.batcher.Batch, src_i: int, forced_ref_ids)\
          -> List[output.Output]:
    outputs = generator.generate(src, src_i, forced_trg_ids=forced_ref_ids)
    return outputs

  def compute_losses_one(self, generator: 'model_base.GeneratorModel', src: xnmt.input.Input,
                         ref: xnmt.input.Input) -> loss.FactoredLossExpr:
    loss_expr = generator.calc_loss(src, ref, loss_calculator=loss_calculator.AutoRegressiveMLELoss())
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
            * ``forced``: perform forced decoding.
            * ``forceddebug``: perform forced decoding, calculate training loss, and make sure the scores are identical
              for debugging purposes.
            * ``score``: output scores, useful for rescoring
    batcher: inference batcher, needed e.g. in connection with ``pad_src_token_to_multiple``
  """

  yaml_tag = '!AutoRegressiveInference'

  @serializable_init
  def __init__(self, src_file: Optional[str] = None, trg_file: Optional[str] = None, ref_file: Optional[str] = None,
               max_src_len: Optional[int] = None, max_num_sents: Optional[int] = None,
               post_process: Union[str, output.OutputProcessor] = bare(output.PlainTextOutputProcessor),
               search_strategy: search_strategy.SearchStrategy = bare(search_strategy.BeamSearch),
               mode: str = "onebest",
               batcher: xnmt.batcher.InOrderBatcher = bare(xnmt.batcher.InOrderBatcher, batch_size=1),
               reporter: Union[None, reports.Reporter, Sequence[reports.Reporter]] = None):
    super().__init__(src_file=src_file, trg_file=trg_file, ref_file=ref_file, max_src_len=max_src_len,
                     max_num_sents=max_num_sents, mode=mode, batcher=batcher, reporter=reporter)

    self.post_processor = output.OutputProcessor.get_output_processor(post_process)
    self.search_strategy = search_strategy

  def generate_one(self, generator: 'model_base.GeneratorModel', src: xnmt.batcher.Batch, src_i: int, forced_ref_ids)\
          -> List[output.Output]:
    outputs = generator.generate(src, src_i, forced_trg_ids=forced_ref_ids, search_strategy=self.search_strategy)
    return outputs

  def compute_losses_one(self, generator: 'model_base.GeneratorModel', src: xnmt.input.Input,
                         ref: xnmt.input.Input) -> loss.FactoredLossExpr:
    loss_expr = generator.calc_loss(src, ref, loss_calculator=loss_calculator.AutoRegressiveMLELoss())
    return loss_expr

class CascadeInference(Inference, Serializable):
  """Inference class that performs inference as a series of independent inference steps.

  Steps are performed using a list of inference sub-objects and a list of models.
  Intermediate outputs are written out to disk and then read by the next time step.

  The generator passed to ``perform_inference`` must be a :class:`xnmt.model_base.CascadeGenerator`.

  Args:
    steps: list of inference objects
  """

  yaml_tag = "!CascadeInference"
  @serializable_init
  def __init__(self, steps: Sequence[Inference]) -> None:
    self.steps = steps

  def perform_inference(self, generator: 'model_base.CascadeGenerator', src_file: str = None, trg_file: str = None,
                        ref_file_to_report = None):
    assert isinstance(generator, model_base.CascadeGenerator)
    assert len(generator.generators) == len(self.steps)
    src_files = [src_file] + [f"{trg_file}.tmp.{i}" for i in range(len(self.steps)-1)]
    trg_files = src_files[1:] + [trg_file]
    for step_i, step in enumerate(self.steps):
      step.perform_inference(generator=generator.generators[step_i],
                             src_file=src_files[step_i],
                             trg_file=trg_files[step_i],
                             ref_file_to_report=ref_file_to_report)

  def compute_losses_one(self, *args, **kwargs):
    raise ValueError("cannot call CascadedInference.compute_losses_one() directly, use the sub-inference objects")
  def generate_one(self, *args, **kwargs):
    raise ValueError("cannot call CascadedInference.generate_one() directly, use the sub-inference objects")

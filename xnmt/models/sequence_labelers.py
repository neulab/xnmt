from typing import Optional, Set, Sequence, Union
import numbers

import dynet as dy
import numpy as np

from xnmt import batchers, event_trigger, events, inferences, input_readers, reports, sent, vocabs
from xnmt.modelparts import attenders, embedders, scorers, transforms
from xnmt.models import base as models
from xnmt.transducers import recurrent, base as transducers
from xnmt.persistence import serializable_init, Serializable, bare

class SeqLabeler(models.ConditionedModel, models.GeneratorModel, Serializable, reports.Reportable):
  """
  A simple sequence labeler based on an encoder and an output softmax layer.

  Args:
    src_reader: A reader for the source side.
    trg_reader: A reader for the target side.
    src_embedder: A word embedder for the input language
    encoder: An encoder to generate encoded inputs
    transform: A transform to be applied before making predictions
    scorer: The class to actually make predictions
    inference: The inference method used for this model
    auto_cut_pad: If ``True``, cut or pad target sequences so the match the length of the encoded inputs.
                  If ``False``, an error is thrown if there is a length mismatch.
  """

  yaml_tag = '!SeqLabeler'

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               src_reader: input_readers.InputReader,
               trg_reader: input_readers.InputReader,
               src_embedder: embedders.Embedder = bare(embedders.SimpleWordEmbedder),
               encoder: transducers.SeqTransducer = bare(recurrent.BiLSTMSeqTransducer),
               transform: transforms.Transform = bare(transforms.NonLinear),
               scorer: scorers.Scorer = bare(scorers.Softmax),
               inference: inferences.Inference = bare(inferences.IndependentOutputInference),
               auto_cut_pad: bool = False) -> None:
    super().__init__(src_reader=src_reader, trg_reader=trg_reader)
    self.src_embedder = src_embedder
    self.encoder = encoder
    self.attender = attenders
    self.transform = transform
    self.scorer = scorer
    self.inference = inference
    self.auto_cut_pad = auto_cut_pad

  def shared_params(self) -> Sequence[Set[str]]:
    return [{".src_embedder.emb_dim", ".encoder.input_dim"},]

  def _encode_src(self, src: Union[sent.Sentence, batchers.Batch]) -> tuple:
    event_trigger.start_sent(src)
    embeddings = self.src_embedder.embed_sent(src)
    encodings = self.encoder.transduce(embeddings)
    encodings_tensor = encodings.as_tensor()
    ((hidden_dim, seq_len), batch_size) = encodings.dim()
    encoding_reshaped = dy.reshape(encodings_tensor, (hidden_dim,), batch_size=batch_size * seq_len)
    outputs = self.transform.transform(encoding_reshaped)
    return batch_size, encodings, outputs, seq_len

  def calc_nll(self, src: Union[batchers.Batch, sent.Sentence], trg: Union[batchers.Batch, sent.Sentence]) \
          -> dy.Expression:
    assert batchers.is_batched(src) and batchers.is_batched(trg)
    batch_size, encodings, outputs, seq_len = self._encode_src(src)

    if trg.sent_len() != seq_len:
      if self.auto_cut_pad:
        trg = self._cut_or_pad_targets(seq_len, trg)
      else:
        raise ValueError(f"src/trg length do not match: {seq_len} != {len(trg[0])}")

    ref_action = np.asarray([trg_sent.words for trg_sent in trg]).reshape((seq_len * batch_size,))
    loss_expr_perstep = self.scorer.calc_loss(outputs, batchers.mark_as_batch(ref_action))
    # loss_expr_perstep = dy.pickneglogsoftmax_batch(outputs, ref_action)
    loss_expr_perstep = dy.reshape(loss_expr_perstep, (seq_len,), batch_size=batch_size)
    if trg.mask:
      loss_expr_perstep = dy.cmult(loss_expr_perstep, dy.inputTensor(1.0-trg.mask.np_arr.T, batched=True))
    loss_expr = dy.sum_elems(loss_expr_perstep)

    return loss_expr

  def _cut_or_pad_targets(self, seq_len: numbers.Integral, trg: batchers.Batch) -> batchers.Batch:
    old_mask = trg.mask
    if len(trg[0]) > seq_len:
      trunc_len = len(trg[0]) - seq_len
      trg = batchers.mark_as_batch([trg_sent.get_truncated_sent(trunc_len=trunc_len) for trg_sent in trg])
      if old_mask:
        trg.mask = batchers.Mask(np_arr=old_mask.np_arr[:, :-trunc_len])
    else:
      pad_len = seq_len - len(trg[0])
      trg = batchers.mark_as_batch([trg_sent.create_padded_sent(pad_len=pad_len) for trg_sent in trg])
      if old_mask:
        trg.mask = np.pad(old_mask.np_arr, pad_width=((0, 0), (0, pad_len)), mode="constant", constant_values=1)
    return trg

  def generate(self,
               src: batchers.Batch,
               forced_trg_ids: Sequence[numbers.Integral]=None,
               normalize_scores: bool = False) -> Sequence[sent.ReadableSentence]:
    if not batchers.is_batched(src):
      src = batchers.mark_as_batch([src])
      if forced_trg_ids:
        forced_trg_ids = batchers.mark_as_batch([forced_trg_ids])
    assert src.batch_size() == 1, "batch size > 1 not properly tested"

    batch_size, encodings, outputs, seq_len = self._encode_src(src)
    score_expr = self.scorer.calc_log_softmax(outputs) if normalize_scores else self.scorer.calc_scores(outputs)
    scores = score_expr.npvalue() # vocab_size x seq_len

    if forced_trg_ids:
      output_actions = forced_trg_ids
    else:
      output_actions = [np.argmax(scores[:, j]) for j in range(seq_len)]
    score = np.sum([scores[output_actions[j], j] for j in range(seq_len)])

    outputs = [sent.SimpleSentence(words=output_actions, idx=src[0].idx,
                                   vocab=self.trg_vocab if hasattr(self, "trg_vocab") else None,
                                   output_procs=self.trg_reader.output_procs,
                                   score=score)]

    return outputs

  def set_trg_vocab(self, trg_vocab: Optional[vocabs.Vocab] = None) -> None:
    """
    Set target vocab for generating outputs. If not specified, word IDs are generated instead.

    Args:
      trg_vocab: target vocab, or None to generate word IDs
    """
    self.trg_vocab = trg_vocab

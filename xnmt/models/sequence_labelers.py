from typing import Optional, Set, Sequence, Union
import numbers

import numpy as np

import xnmt.tensor_tools as tt
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
    encoding_reshaped = tt.merge_time_batch_dims(encodings_tensor)
    outputs = self.transform.transform(encoding_reshaped)
    return tt.batch_size(encodings_tensor), encodings, outputs, tt.sent_len(encodings_tensor)

  def calc_nll(self, src: Union[batchers.Batch, sent.Sentence], trg: Union[batchers.Batch, sent.Sentence]) \
          -> tt.Tensor:
    assert batchers.is_batched(src) and batchers.is_batched(trg)
    batch_size, encodings, outputs, seq_len = self._encode_src(src)

    if trg.sent_len() != seq_len:
      if self.auto_cut_pad:
        trg = self._cut_or_pad_targets(seq_len, trg)
      else:
        raise ValueError(f"src/trg length do not match: {seq_len} != {trg.sent_len()}")

    ref_action = np.asarray([trg_sent.words for trg_sent in trg]).reshape((seq_len * batch_size,))
    loss_expr_perstep = self.scorer.calc_loss(outputs, batchers.mark_as_batch(ref_action))
    loss_expr_perstep = tt.unmerge_time_batch_dims(loss_expr_perstep, batch_size)
    loss_expr = tt.aggregate_masked_loss(loss_expr_perstep, trg.mask)

    return loss_expr

  def _cut_or_pad_targets(self, seq_len: numbers.Integral, trg: batchers.Batch) -> batchers.Batch:
    old_mask = trg.mask
    if trg.sent_len() > seq_len:
      trunc_len = trg.sent_len() - seq_len
      trg = batchers.mark_as_batch([trg_sent.create_truncated_sent(trunc_len=trunc_len) for trg_sent in trg])
      if old_mask:
        trg.mask = batchers.Mask(np_arr=old_mask.np_arr[:, :-trunc_len])
    else:
      pad_len = seq_len - trg.sent_len()
      trg = batchers.mark_as_batch([trg_sent.create_padded_sent(pad_len=pad_len) for trg_sent in trg])
      if old_mask:
        trg.mask = np.pad(old_mask.np_arr, pad_width=((0, 0), (0, pad_len)), mode="constant", constant_values=1)
    return trg

  def generate(self,
               src: batchers.Batch,
               normalize_scores: bool = False) -> Sequence[sent.ReadableSentence]:
    if not batchers.is_batched(src):
      src = batchers.mark_as_batch([src])
    assert src.batch_size() == 1, "batch size > 1 not properly tested"

    batch_size, encodings, outputs, seq_len = self._encode_src(src)

    best_words, best_scores = self.scorer.best_k(outputs, k=1, normalize_scores=normalize_scores)
    best_words = best_words[0, :]
    score = np.sum(best_scores, axis=1)

    outputs = [sent.SimpleSentence(words=best_words, idx=src[0].idx,
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

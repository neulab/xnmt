from typing import Sequence, Set, Union

import dynet as dy
import numpy as np

from xnmt import batchers, event_trigger, events, input_readers, sent
from xnmt.modelparts import embedders, scorers, transforms
from xnmt.models import base as models
from xnmt.transducers import base as transducers
from xnmt.transducers import recurrent
from xnmt.persistence import serializable_init, Serializable, bare

class LanguageModel(models.ConditionedModel, Serializable):
  """
  A simple unidirectional language model predicting the next token.

  Args:
    src_reader: A reader for the source side.
    src_embedder: A word embedder for the input language
    rnn: An RNN, usually unidirectional LSTM with one or more layers
    transform: A transform to be applied before making predictions
    scorer: The class to actually make predictions
  """

  yaml_tag = '!LanguageModel'

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               src_reader:input_readers.InputReader,
               src_embedder: embedders.Embedder=bare(embedders.SimpleWordEmbedder),
               rnn:transducers.SeqTransducer=bare(recurrent.UniLSTMSeqTransducer),
               transform: transforms.Transform=bare(transforms.NonLinear),
               scorer: scorers.Scorer=bare(scorers.Softmax)) -> None:
    super().__init__(src_reader=src_reader, trg_reader=src_reader)
    self.src_embedder = src_embedder
    self.rnn = rnn
    self.transform = transform
    self.scorer = scorer

  def shared_params(self) -> Sequence[Set[str]]:
    return [{".src_embedder.emb_dim", ".encoder.input_dim"},]

  def calc_nll(self, src: Union[batchers.Batch, sent.Sentence], trg: Union[batchers.Batch, sent.Sentence]) \
          -> dy.Expression:
    if not batchers.is_batched(src):
      src = batchers.ListBatch([src])

    src_inputs = batchers.ListBatch([s[:-1] for s in src], mask=batchers.Mask(src.mask.np_arr[:, :-1]) if src.mask else None)
    src_targets = batchers.ListBatch([s[1:] for s in src], mask=batchers.Mask(src.mask.np_arr[:, 1:]) if src.mask else None)

    event_trigger.start_sent(src)
    embeddings = self.src_embedder.embed_sent(src_inputs)
    encodings = self.rnn.transduce(embeddings)
    encodings_tensor = encodings.as_tensor()
    ((hidden_dim, seq_len), batch_size) = encodings.dim()
    encoding_reshaped = dy.reshape(encodings_tensor, (hidden_dim,), batch_size=batch_size * seq_len)
    outputs = self.transform.transform(encoding_reshaped)

    ref_action = np.asarray([sent.words for sent in src_targets]).reshape((seq_len * batch_size,))
    loss_expr_perstep = self.scorer.calc_loss(outputs, batchers.mark_as_batch(ref_action))
    loss_expr_perstep = dy.reshape(loss_expr_perstep, (seq_len,), batch_size=batch_size)
    if src_targets.mask:
      loss_expr_perstep = dy.cmult(loss_expr_perstep, dy.inputTensor(1.0-src_targets.mask.np_arr.T, batched=True))
    loss = dy.sum_elems(loss_expr_perstep)

    return loss

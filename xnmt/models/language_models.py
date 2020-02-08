from typing import Sequence, Set, Union

import numpy as np

import xnmt
import xnmt.tensor_tools as tt
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
          -> tt.Tensor:
    if not batchers.is_batched(src):
      src = batchers.ListBatch([src])

    src_inputs = batchers.ListBatch([s[:-1] for s in src], mask=batchers.Mask(src.mask.np_arr[:, :-1]) if src.mask else None)
    src_targets = batchers.ListBatch([s[1:] for s in src], mask=batchers.Mask(src.mask.np_arr[:, 1:]) if src.mask else None)

    event_trigger.start_sent(src)
    embeddings = self.src_embedder.embed_sent(src_inputs)
    encodings = self.rnn.transduce(embeddings)
    encodings_tensor = encodings.as_tensor()

    encoding_reshaped = tt.merge_time_batch_dims(encodings_tensor)
    seq_len = tt.sent_len(encodings_tensor)
    batch_size = tt.batch_size(encodings_tensor)

    outputs = self.transform.transform(encoding_reshaped)

    ref_action = np.asarray([sent.words for sent in src_targets]).reshape((seq_len * batch_size,))
    loss_expr_perstep = self.scorer.calc_loss(outputs, batchers.mark_as_batch(ref_action))

    loss_expr_perstep = tt.unmerge_time_batch_dims(loss_expr_perstep, batch_size)

    loss = tt.aggregate_masked_loss(loss_expr_perstep, src_targets.mask)

    return loss


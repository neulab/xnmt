import numbers
from typing import Optional, Sequence, Union

import dynet as dy
import numpy as np

from xnmt import batchers, event_trigger, inferences, input_readers, sent
from xnmt.modelparts import transforms
from xnmt.modelparts import scorers
from xnmt.modelparts import embedders
from xnmt.transducers import recurrent
from xnmt.transducers import base as transducers
from xnmt.models import base as models
from xnmt.persistence import serializable_init, Serializable, bare

class SequenceClassifier(models.ConditionedModel, models.GeneratorModel, Serializable):
  """
  A sequence classifier.

  Runs embeddings through an encoder, feeds the average over all encoder outputs to a transform and scoring layer.

  Args:
    src_reader: A reader for the source side.
    trg_reader: A reader for the target side.
    src_embedder: A word embedder for the input language
    encoder: An encoder to generate encoded inputs
    inference: how to perform inference
    transform: A transform performed before the scoring function
    scorer: A scoring function over the multiple choices
  """

  yaml_tag = '!SequenceClassifier'

  @serializable_init
  def __init__(self,
               src_reader: input_readers.InputReader,
               trg_reader: input_readers.InputReader,
               src_embedder: embedders.Embedder = bare(embedders.SimpleWordEmbedder),
               encoder: transducers.SeqTransducer = bare(recurrent.BiLSTMSeqTransducer),
               inference=bare(inferences.IndependentOutputInference),
               transform: transforms.Transform = bare(transforms.NonLinear),
               scorer: scorers.Scorer = bare(scorers.Softmax)) -> None:
    super().__init__(src_reader=src_reader, trg_reader=trg_reader)
    self.src_embedder = src_embedder
    self.encoder = encoder
    self.transform = transform
    self.scorer = scorer
    self.inference = inference

  def shared_params(self):
    return [{".src_embedder.emb_dim", ".encoder.input_dim"},
            {".encoder.hidden_dim", ".transform.input_dim"},
            {".transform.output_dim", ".scorer.input_dim"}]

  def _encode_src(self, src):
    event_trigger.start_sent(src)
    embeddings = self.src_embedder.embed_sent(src)
    self.encoder.transduce(embeddings)
    h = self.encoder.get_final_states()[-1].main_expr()
    return self.transform.transform(h)

  def calc_nll(self, src: Union[batchers.Batch, sent.Sentence], trg: Union[batchers.Batch, sent.Sentence]) \
          -> dy.Expression:
    h = self._encode_src(src)
    ids = trg.value if not batchers.is_batched(trg) else batchers.ListBatch([trg_i.value for trg_i in trg])
    loss_expr = self.scorer.calc_loss(h, ids)
    return loss_expr

  def generate(self,
               src: Union[batchers.Batch, sent.Sentence],
               normalize_scores: bool = False):
    if not batchers.is_batched(src):
      src = batchers.mark_as_batch([src])
    h = self._encode_src(src)
    best_words, best_scores = self.scorer.best_k(h, k=1, normalize_scores=normalize_scores)
    assert best_words.shape == (1, src.batch_size())
    assert best_scores.shape == (1, src.batch_size())

    outputs = []
    for batch_i in range(src.batch_size()):
      if src.batch_size() > 1:
        word = best_words[0, batch_i]
        score = best_scores[0, batch_i]
      else:
        word = best_words[0]
        score = best_scores[0]
      outputs.append(sent.ScalarSentence(value=word, score=score))
    return outputs

  def get_nobp_state(self, state):
    output_state = state.as_vector()
    return output_state

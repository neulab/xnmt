import dynet as dy
import numpy as np

from xnmt import batcher, embedder, input_reader, loss, lstm, model_base, output, scorer, transducer, transform
from xnmt.persistence import serializable_init, Serializable, bare
import xnmt.inference

class SequenceClassifier(model_base.ConditionedModel, model_base.GeneratorModel, Serializable, model_base.EventTrigger):
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
               src_reader: input_reader.InputReader,
               trg_reader: input_reader.InputReader,
               src_embedder: embedder.Embedder = bare(embedder.SimpleWordEmbedder),
               encoder: transducer.SeqTransducer = bare(lstm.BiLSTMSeqTransducer),
               inference=bare(xnmt.inference.IndependentOutputInference),
               transform: transform.Transform = bare(transform.NonLinear),
               scorer: scorer.Scorer = bare(scorer.Softmax)):
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
    self.start_sent(src)
    embeddings = self.src_embedder.embed_sent(src)
    self.encoder.transduce(embeddings)
    h = self.encoder.get_final_states()[-1].main_expr()
    return self.transform(h)

  def calc_loss(self, src, trg, loss_calculator):
    h = self._encode_src(src)
    ids = trg.value if not batcher.is_batched(trg) else batcher.ListBatch([trg_i.value for trg_i in trg])
    loss_expr = self.scorer.calc_loss(h, ids)
    classifier_loss = loss.FactoredLossExpr({"mle" : loss_expr})
    return classifier_loss

  def generate(self, src, idx, forced_trg_ids=None, normalize_scores=False):
    if not batcher.is_batched(src):
      src = batcher.mark_as_batch([src])
      if forced_trg_ids:
        forced_trg_ids = batcher.mark_as_batch([forced_trg_ids])
    h = self._encode_src(src)
    scores = self.scorer.calc_log_probs(h) if normalize_scores else self.scorer.calc_scores(h)
    np_scores = scores.npvalue()
    if forced_trg_ids:
      output_action = forced_trg_ids
    else:
      output_action = np.argmax(np_scores, axis=0)
    outputs = []
    for batch_i in range(src.batch_size()):
      score = np_scores[:, batch_i][output_action[batch_i]]
      outputs.append(output.ScalarOutput(actions=[output_action],
                                         vocab=None,
                                         score=score))
    return outputs

  def get_primary_loss(self):
    return "mle"

  def get_nobp_state(self, state):
    output_state = state.rnn_state.output()
    return output_state

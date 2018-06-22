import dynet as dy
import numpy as np

from xnmt import batcher, embedder, input_reader, loss, lstm, mlp, model_base, output, transducer
from xnmt.persistence import serializable_init, Serializable, bare
import xnmt.inference

class SequenceClassifier(model_base.GeneratorModel, Serializable, model_base.EventTrigger):
  """
  A sequence classifier.

  Runs embeddings through an encoder, feeds the average over all encoder outputs to a MLP softmax output layer.

  Args:
    src_reader: A reader for the source side.
    trg_reader: A reader for the target side.
    src_embedder: A word embedder for the input language
    encoder: An encoder to generate encoded inputs
    inference: how to perform inference
    mlp: final prediction MLP layer
  """

  yaml_tag = '!SequenceClassifier'

  @serializable_init
  def __init__(self,
               src_reader: input_reader.InputReader,
               trg_reader: input_reader.InputReader,
               src_embedder: embedder.Embedder = bare(embedder.SimpleWordEmbedder),
               encoder: transducer.Transducer = bare(lstm.BiLSTMSeqTransducer),
               inference=bare(xnmt.inference.IndependentOutputInference),
               mlp: mlp.MLP = bare(mlp.OutputMLP)):
    super().__init__(src_reader=src_reader, trg_reader=trg_reader)
    self.src_embedder = src_embedder
    self.encoder = encoder
    self.mlp = mlp
    self.inference = inference

  def shared_params(self):
    return [{".src_embedder.emb_dim", ".encoder.input_dim"},
            {".encoder.hidden_dim", ".mlp.input_dim"}]

  def _encode_src(self, src):
    self.start_sent(src)
    embeddings = self.src_embedder.embed_sent(src)
    self.encoder.transduce(embeddings)
    scores = self.mlp(self.encoder.get_final_states()[-1].main_expr())
    return scores

  def calc_loss(self, src, trg, loss_calculator):
    scores = self._encode_src(src)
    if not batcher.is_batched(trg):
      loss_expr = dy.pickneglogsoftmax(scores, trg.value)
    else:
      loss_expr = dy.pickneglogsoftmax_batch(scores, [trg_i.value for trg_i in trg])
    classifier_loss = loss.FactoredLossExpr({"mle" : loss_expr})
    return classifier_loss

  def generate(self, src, idx, forced_trg_ids=None):
    if not batcher.is_batched(src):
      src = batcher.mark_as_batch([src])
    outputs = []
    for sents in src:
      scores = self._encode_src(sents)
      logsoftmax = dy.log_softmax(scores).npvalue()
      if forced_trg_ids:
        output_action = forced_trg_ids[0]
      else:
        output_action = np.argmax(logsoftmax)
      score = logsoftmax[output_action]
      # Append output to the outputs
      outputs.append(output.ScalarOutput(actions=[output_action],
                                         vocab=None,
                                         score=score))
    return outputs

  def get_primary_loss(self):
    return "mle"

  def get_nobp_state(self, state):
    output_state = state.rnn_state.output()
    return output_state

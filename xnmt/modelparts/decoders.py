from typing import Any
import numbers

import dynet as dy

from xnmt import batchers, param_collections
from xnmt.modelparts import bridges, transforms, scorers
from xnmt.transducers import recurrent
from xnmt.persistence import serializable_init, Serializable, bare, Ref

class Decoder(object):
  """
  A template class to convert a prefix of previously generated words and
  a context vector into a probability distribution over possible next words.
  """

  def calc_loss(self, x, ref_action):
    raise NotImplementedError('must be implemented by subclasses')
  def calc_score(self, calc_scores_logsoftmax):
    raise NotImplementedError('must be implemented by subclasses')
  def calc_prob(self, calc_scores_logsoftmax):
    raise NotImplementedError('must be implemented by subclasses')
  def calc_log_prob(self, calc_scores_logsoftmax):
    raise NotImplementedError('must be implemented by subclasses')
  def add_input(self, mlp_dec_state, trg_embedding):
    raise NotImplementedError('must be implemented by subclasses')
  def initial_state(self, enc_final_states, ss_expr):
    raise NotImplementedError('must be implemented by subclasses')

class AutoRegressiveDecoderState(object):
  """A state holding all the information needed for AutoRegressiveDecoder
  
  Args:
    rnn_state: a DyNet RNN state
    context: a DyNet expression
  """
  def __init__(self, rnn_state=None, context=None):
    self.rnn_state = rnn_state
    self.context = context

class AutoRegressiveDecoder(Decoder, Serializable):
  """
  Standard autoregressive-decoder.

  Args:
    input_dim: input dimension
    trg_embed_dim: dimension of target embeddings
    input_feeding: whether to activate input feeding
    bridge: how to initialize decoder state
    rnn: recurrent decoder
    transform: a layer of transformation between rnn and output scorer
    scorer: the method of scoring the output (usually softmax)
    truncate_dec_batches: whether the decoder drops batch elements as soon as these are masked at some time step.
  """

  yaml_tag = '!AutoRegressiveDecoder'

  @serializable_init
  def __init__(self,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               trg_embed_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               input_feeding: bool = True,
               bridge: bridges.Bridge = bare(bridges.CopyBridge),
               rnn: recurrent.UniLSTMSeqTransducer = bare(recurrent.UniLSTMSeqTransducer),
               transform: transforms.Transform = bare(transforms.AuxNonLinear),
               scorer: scorers.Scorer = bare(scorers.Softmax),
               truncate_dec_batches: bool = Ref("exp_global.truncate_dec_batches", default=False)) -> None:
    self.param_col = param_collections.ParamManager.my_params(self)
    self.input_dim = input_dim
    self.truncate_dec_batches = truncate_dec_batches
    self.bridge = bridge
    self.rnn = rnn
    self.transform = transform
    self.scorer = scorer
    # Input feeding
    self.input_feeding = input_feeding
    rnn_input_dim = trg_embed_dim
    if input_feeding:
      rnn_input_dim += input_dim
    assert rnn_input_dim == rnn.input_dim, "Wrong input dimension in RNN layer: {} != {}".format(rnn_input_dim, rnn.input_dim)

  def shared_params(self):
    return [{".trg_embed_dim", ".rnn.input_dim"},
            {".input_dim", ".rnn.decoder_input_dim"},
            {".input_dim", ".transform.input_dim"},
            {".input_feeding", ".rnn.decoder_input_feeding"},
            {".rnn.layers", ".bridge.dec_layers"},
            {".rnn.hidden_dim", ".bridge.dec_dim"},
            {".rnn.hidden_dim", ".transform.aux_input_dim"},
            {".transform.output_dim", ".scorer.input_dim"}]

  def initial_state(self, enc_final_states: Any, ss_expr: dy.Expression) -> AutoRegressiveDecoderState:
    """Get the initial state of the decoder given the encoder final states.

    Args:
      enc_final_states: The encoder final states. Usually but not necessarily an :class:`xnmt.expression_sequence.ExpressionSequence`
      ss_expr: first input
    Returns:
      initial decoder state
    """
    rnn_state = self.rnn.initial_state()
    rnn_s = self.bridge.decoder_init(enc_final_states)
    rnn_state = rnn_state.set_s(rnn_s)
    zeros = dy.zeros(self.input_dim) if self.input_feeding else None
    rnn_state = rnn_state.add_input(dy.concatenate([ss_expr, zeros]) if self.input_feeding else ss_expr)
    return AutoRegressiveDecoderState(rnn_state=rnn_state, context=zeros)

  def add_input(self, mlp_dec_state: AutoRegressiveDecoderState, trg_embedding: dy.Expression) -> AutoRegressiveDecoderState:
    """Add an input and update the state.

    Args:
      mlp_dec_state: An object containing the current state.
      trg_embedding: The embedding of the word to input.
    Returns:
      The updated decoder state.
    """
    inp = trg_embedding
    if self.input_feeding:
      inp = dy.concatenate([inp, mlp_dec_state.context])
    rnn_state = mlp_dec_state.rnn_state
    if self.truncate_dec_batches: rnn_state, inp = batchers.truncate_batches(rnn_state, inp)
    return AutoRegressiveDecoderState(rnn_state=rnn_state.add_input(inp),
                                      context=mlp_dec_state.context)

  def _calc_transform(self, mlp_dec_state: AutoRegressiveDecoderState) -> dy.Expression:
    h = dy.concatenate([mlp_dec_state.rnn_state.output(), mlp_dec_state.context])
    return self.transform.transform(h)

  def calc_scores(self, mlp_dec_state: AutoRegressiveDecoderState) -> dy.Expression:
    """Get scores given a current state.

    Args:
      mlp_dec_state: Decoder state with last RNN output and optional context vector.
    Returns:
      Scores over the vocabulary given this state.
    """
    return self.scorer.calc_scores(self._calc_transform(mlp_dec_state))

  def calc_log_probs(self, mlp_dec_state):
    return self.scorer.calc_log_probs(self._calc_transform(mlp_dec_state))

  def calc_loss(self, mlp_dec_state, ref_action):
    return self.scorer.calc_loss(self._calc_transform(mlp_dec_state), ref_action)


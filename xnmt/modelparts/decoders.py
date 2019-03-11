from typing import Any
import numbers

import xnmt
import xnmt.tensor_tools as tt
from xnmt import param_collections
from xnmt.modelparts import bridges, transforms, scorers, embedders
from xnmt.transducers import recurrent
from xnmt.persistence import serializable_init, Serializable, bare, Ref

if xnmt.backend_dynet:
  import dynet as dy

class Decoder(object):
  """
  A template class to convert a prefix of previously generated words and
  a context vector into a probability distribution over possible next words.
  """
  def initial_state(self, enc_final_states, ss_expr):
    raise NotImplementedError('must be implemented by subclasses')
  def add_input(self, dec_state, trg_embedding):
    raise NotImplementedError('must be implemented by subclasses')
  def calc_loss(self, dec_state, ref_action):
    raise NotImplementedError('must be implemented by subclasses')
  def calc_score(self, dec_state, action, normalize=False):
    raise NotImplementedError('must be implemented by subclasses')
  def best_k(self, dec_state, k, normalize_scores=False):
    raise NotImplementedError('must be implemented by subclasses')
  def sample(self, dec_state, n, temperature=1.0):
    raise NotImplementedError('must be implemented by subclasses')

class DecoderState(object):
  """A state that holds whatever information is required for the decoder.
     Child classes must implement the as_vector() method, which will be
     used by e.g. the attention mechanism"""

  def as_vector(self):
    raise NotImplementedError('must be implemented by subclass')

class AutoRegressiveDecoderState(DecoderState):
  """A state holding all the information needed for AutoRegressiveDecoder
  
  Args:
    rnn_state: a DyNet RNN state
    context: a DyNet expression
  """
  def __init__(self, rnn_state=None, context=None):
    self._rnn_state = rnn_state
    self._context = context

  def as_vector(self):
    return self.rnn_state.output()
  
  @property
  def rnn_state(self):
    return self._rnn_state
  
  @property
  def context(self):
    return self._context
  
  @context.setter
  def context(self, value):
    self._context = value

# @xnmt.require_dynet
class AutoRegressiveDecoder(Decoder, Serializable):
  """
  Standard autoregressive-decoder.

  Args:
    input_dim: input dimension
    embedder: embedder for target words
    input_feeding: whether to activate input feeding
    bridge: how to initialize decoder state
    rnn: recurrent decoder
    transform: a layer of transformation between rnn and output scorer
    scorer: the method of scoring the output (usually softmax)
  """

  yaml_tag = '!AutoRegressiveDecoder'

  @serializable_init
  def __init__(self,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               embedder: embedders.Embedder = bare(embedders.SimpleWordEmbedder),
               input_feeding: bool = True,
               bridge: bridges.Bridge = bare(bridges.CopyBridge),
               rnn: recurrent.UniLSTMSeqTransducer = bare(recurrent.UniLSTMSeqTransducer),
               transform: transforms.Transform = bare(transforms.AuxNonLinear),
               scorer: scorers.Scorer = bare(scorers.Softmax)) -> None:
    self.param_col = param_collections.ParamManager.my_params(self)
    self.input_dim = input_dim
    self.embedder = embedder
    self.bridge = bridge
    self.rnn = rnn
    self.transform = transform
    self.scorer = scorer
    # Input feeding
    self.input_feeding = input_feeding
    rnn_input_dim = embedder.emb_dim
    if input_feeding:
      rnn_input_dim += input_dim
    assert rnn_input_dim == rnn.total_input_dim, "Wrong input dimension in RNN layer: {} != {}".format(rnn_input_dim, rnn.total_input_dim)

  def shared_params(self):
    return [{".embedder.emb_dim", ".rnn.input_dim"},
            {".input_dim", ".rnn.decoder_input_dim"},
            {".input_dim", ".transform.input_dim"},
            {".input_feeding", ".rnn.decoder_input_feeding"},
            {".rnn.layers", ".bridge.dec_layers"},
            {".rnn.hidden_dim", ".bridge.dec_dim"},
            {".rnn.hidden_dim", ".transform.aux_input_dim"},
            {".transform.output_dim", ".scorer.input_dim"}]

  def initial_state(self, enc_final_states: Any, ss: Any) -> AutoRegressiveDecoderState:
    """Get the initial state of the decoder given the encoder final states.

    Args:
      enc_final_states: The encoder final states. Usually but not necessarily an :class:`xnmt.expression_sequence.ExpressionSequence`
      ss: first input
    Returns:
      initial decoder state
    """
    rnn_state = self.rnn.initial_state()
    rnn_s = self.bridge.decoder_init(enc_final_states)
    rnn_state = rnn_state.set_s(rnn_s)
    ss_expr = self.embedder.embed(ss)
    zeros = tt.zeroes(hidden_dim=self.input_dim,batch_size=tt.batch_size(ss_expr)) if self.input_feeding else None
    rnn_state = rnn_state.add_input(tt.concatenate([ss_expr, zeros]) if self.input_feeding else ss_expr)
    return AutoRegressiveDecoderState(rnn_state=rnn_state, context=zeros)

  def add_input(self, dec_state: AutoRegressiveDecoderState, trg_word: Any) -> AutoRegressiveDecoderState:
    """
    Add an input and return a *new* update the state.

    Args:
      dec_state: An object containing the current state.
      trg_word: The word to input.
    Returns:
      The updated decoder state.
    """
    trg_embedding = self.embedder.embed(trg_word)
    inp = trg_embedding
    if self.input_feeding:
      inp = tt.concatenate([inp, dec_state.context])
    rnn_state = dec_state.rnn_state
    return AutoRegressiveDecoderState(rnn_state=rnn_state.add_input(inp),
                                      context=dec_state.context)

  def _calc_transform(self, mlp_dec_state: AutoRegressiveDecoderState) -> tt.Tensor:
    h = tt.concatenate([mlp_dec_state.rnn_state.output(), mlp_dec_state.context])
    return self.transform.transform(h)

  def best_k(self, mlp_dec_state: AutoRegressiveDecoderState, k: numbers.Integral, normalize_scores: bool = False):
    h = self._calc_transform(mlp_dec_state)
    best_words, best_scores = self.scorer.best_k(h, k, normalize_scores=normalize_scores)
    return best_words, best_scores

  def sample(self, mlp_dec_state: AutoRegressiveDecoderState, n: numbers.Integral, temperature=1.0):
    h = self._calc_transform(mlp_dec_state)
    return self.scorer.sample(h, n)

  def calc_loss(self, mlp_dec_state, ref_action):
    return self.scorer.calc_loss(self._calc_transform(mlp_dec_state), ref_action)

import dynet as dy

import xnmt.batcher
import xnmt.linear
import xnmt.residual
from xnmt.bridge import CopyBridge
from xnmt.lstm import UniLSTMSeqTransducer
from xnmt.mlp import MLP
from xnmt.param_collection import ParamManager
from xnmt.persistence import serializable_init, Serializable, bare, Ref, Path

class Decoder(object):
  '''
  A template class to convert a prefix of previously generated words and
  a context vector into a probability distribution over possible next words.
  '''

  '''
  Document me
  '''

  def calc_loss(self, x, ref_action):
    raise NotImplementedError('calc_loss must be implemented in Decoder subclasses')

class MlpSoftmaxDecoderState(object):
  """A state holding all the information needed for MLPSoftmaxDecoder
  
  Args:
    rnn_state: a DyNet RNN state
    context: a DyNet expression
  """
  def __init__(self, rnn_state=None, context=None):
    self.rnn_state = rnn_state
    self.context = context

class MlpSoftmaxDecoder(Decoder, Serializable):
  """
  Standard MLP softmax decoder.

  Args:
    input_dim (int): input dimension
    trg_embed_dim (int): dimension of target embeddings
    input_feeding (bool): whether to activate input feeding
    rnn_layer (UniLSTMSeqTransducer): recurrent layer of the decoder
    mlp_layer (MLP): final prediction layer of the decoder
    bridge (Bridge): how to initialize decoder state
    label_smoothing (float): label smoothing value (if used, 0.1 is a reasonable value).
                             Label Smoothing is implemented with reference to Section 7 of the paper
                             "Rethinking the Inception Architecture for Computer Vision"
                             (https://arxiv.org/pdf/1512.00567.pdf)
  """

  # TODO: This should probably take a softmax object, which can be normal or class-factored, etc.
  # For now the default behavior is hard coded.

  yaml_tag = '!MlpSoftmaxDecoder'

  @serializable_init
  def __init__(self,
               input_dim=Ref("exp_global.default_layer_dim"),
               trg_embed_dim=Ref("exp_global.default_layer_dim"),
               input_feeding=True,
               rnn_layer=bare(UniLSTMSeqTransducer),
               mlp_layer=bare(MLP),
               bridge=bare(CopyBridge),
               label_smoothing=0.0):
    self.param_col = ParamManager.my_params(self)
    self.input_dim = input_dim
    self.label_smoothing = label_smoothing
    # Input feeding
    self.input_feeding = input_feeding
    rnn_input_dim = trg_embed_dim
    if input_feeding:
      rnn_input_dim += input_dim
    assert rnn_input_dim == rnn_layer.input_dim, "Wrong input dimension in RNN layer"
    # Bridge
    self.bridge = bridge

    # LSTM
    self.rnn_layer = rnn_layer

    # MLP
    self.mlp_layer = mlp_layer

  def shared_params(self):
    return [set([Path(".trg_embed_dim"), Path(".rnn_layer.input_dim")]),
            set([Path(".input_dim"), Path(".rnn_layer.decoder_input_dim")]),
            set([Path(".input_dim"), Path(".mlp_layer.input_dim")]),
            set([Path(".input_feeding"), Path(".rnn_layer.decoder_input_feeding")]),
            set([Path(".rnn_layer.layers"), Path(".bridge.dec_layers")]),
            set([Path(".rnn_layer.hidden_dim"), Path(".bridge.dec_dim")]),
            set([Path(".rnn_layer.hidden_dim"), Path(".mlp_layer.decoder_rnn_dim")])]

  def initial_state(self, enc_final_states, ss_expr):
    """Get the initial state of the decoder given the encoder final states.

    Args:
      enc_final_states: The encoder final states. Usually but not necessarily an :class:`xnmt.expression_sequence.ExpressionSequence`
    Returns:
      MlpSoftmaxDecoderState:
    """
    rnn_state = self.rnn_layer.initial_state()
    rnn_state = rnn_state.set_s(self.bridge.decoder_init(enc_final_states))
    zeros = dy.zeros(self.input_dim) if self.input_feeding else None
    rnn_state = rnn_state.add_input(dy.concatenate([ss_expr, zeros]))
    return MlpSoftmaxDecoderState(rnn_state=rnn_state, context=zeros)

  def add_input(self, mlp_dec_state, trg_embedding):
    """Add an input and update the state.
    
    Args:
      mlp_dec_state (MlpSoftmaxDecoderState): An object containing the current state.
      trg_embedding: The embedding of the word to input.
    Returns:
      The update MLP decoder state.
    """
    inp = trg_embedding
    if self.input_feeding:
      inp = dy.concatenate([inp, mlp_dec_state.context])
    return MlpSoftmaxDecoderState(rnn_state=mlp_dec_state.rnn_state.add_input(inp),
                                  context=mlp_dec_state.context)

  def get_scores(self, mlp_dec_state):
    """Get scores given a current state.

    Args:
      mlp_dec_state: An :class:`xnmt.decoder.MlpSoftmaxDecoderState` object.
    Returns:
      Scores over the vocabulary given this state.
    """
    return self.mlp_layer(dy.concatenate([mlp_dec_state.rnn_state.output(), mlp_dec_state.context]))

  def calc_loss(self, mlp_dec_state, ref_action):
    scores = self.get_scores(mlp_dec_state)

    if self.label_smoothing == 0.0:
      # single mode
      if not xnmt.batcher.is_batched(ref_action):
        return dy.pickneglogsoftmax(scores, ref_action)
      # minibatch mode
      else:
        return dy.pickneglogsoftmax_batch(scores, ref_action)

    else:
      log_prob = dy.log_softmax(scores)
      if not xnmt.batcher.is_batched(ref_action):
        pre_loss = -dy.pick(log_prob, ref_action)
      else:
        pre_loss = -dy.pick_batch(log_prob, ref_action)

      ls_loss = -dy.mean_elems(log_prob)
      loss = ((1 - self.label_smoothing) * pre_loss) + (self.label_smoothing * ls_loss)
      return loss

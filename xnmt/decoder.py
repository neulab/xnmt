from collections.abc import Sequence

import dynet as dy
from xnmt.serialize.serializable import Serializable, bare
from xnmt.serialize.tree_tools import Ref, Path
import xnmt.batcher
from xnmt.events import register_handler, handle_xnmt_event
import xnmt.linear
from xnmt.lstm import UniLSTMSeqTransducer
import xnmt.residual
from xnmt.bridge import CopyBridge
from xnmt.param_init import GlorotInitializer

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

class RnnDecoder(Decoder):
  @staticmethod
  def rnn_from_spec(spec, num_layers, input_dim, hidden_dim, exp_global, residual_to_output):
    decoder_type = spec.lower()
    if decoder_type == "lstm":
      return UniLSTMSeqTransducer(layers=num_layers, input_dim=input_dim,
                                  hidden_dim=hidden_dim, exp_global=exp_global)
    elif decoder_type == "residuallstm":
      return xnmt.residual.ResidualRNNBuilder(num_layers, input_dim, hidden_dim,
                                         exp_global.dynet_param_collection.param_col,
                                         residual_to_output)
    else:
      raise RuntimeError("Unknown decoder type {}".format(spec))

class MlpSoftmaxDecoderState(object):
  """A state holding all the information needed for MLPSoftmaxDecoder"""
  def __init__(self, rnn_state=None, context=None):
    self.rnn_state = rnn_state
    self.context = context

class MlpSoftmaxDecoder(RnnDecoder, Serializable):
  # TODO: This should probably take a softmax object, which can be normal or class-factored, etc.
  # For now the default behavior is hard coded.

  yaml_tag = u'!MlpSoftmaxDecoder'

  def __init__(self, exp_global=Ref(Path("exp_global")),
               rnn_layer=bare(UniLSTMSeqTransducer), input_dim=None,
               mlp_hidden_dim=None, trg_embed_dim=None, input_feeding=True,
               param_init_context=None, bias_init_context=None,
               param_init_output=None, bias_init_output=None,
               bridge=bare(CopyBridge), label_smoothing=0.0,
               vocab_projector=None, vocab_size = None, vocab = None,
               trg_reader = Ref(path=Path("model.trg_reader"), required=False)):
               # obsolete or moved into rnn_layer:
               #layers=1, lstm_dim=None,
               #dropout=None,
               #rnn_spec="lstm", residual_to_output=False,
               #param_init_lstm=None,
    register_handler(self)
    self.param_col = exp_global.dynet_param_collection.param_col
    # Define dim
    self.mlp_hidden_dim = mlp_hidden_dim = mlp_hidden_dim or exp_global.default_layer_dim
    trg_embed_dim  = trg_embed_dim or exp_global.default_layer_dim
    input_dim      = input_dim or exp_global.default_layer_dim
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

    #=== TODO: what does this do and why is it needed? how to preserve it with
    #===       the refactoring?
    #
    #if not isinstance(param_init_lstm, GlorotInitializer):
    #  raise NotImplementedError("For the decoder LSTM, only Glorot initialization is currently supported")
    #if getattr(param_init_lstm,"gain",1.0) != 1.0:
    #  for l in range(layers):
    #    for i in [0,1]:
    #      self.rnn_layer.param_collection().parameters_list()[3*l+i].scale(param_init_lstm.gain)
    #===========================================================================

    # MLP
    self.context_projector = xnmt.linear.Linear(input_dim  = input_dim + self.rnn_layer.hidden_dim,
                                                output_dim = mlp_hidden_dim,
                                                model = self.param_col,
                                                param_init = param_init_context or exp_global.param_init,
                                                bias_init = bias_init_context or exp_global.bias_init)
    self.vocab_size = self.choose_vocab_size(vocab_size, vocab, trg_reader)
    self.vocab_projector = vocab_projector or xnmt.linear.Linear(input_dim = self.mlp_hidden_dim,
                                                                 output_dim = self.vocab_size,
                                                                 model = self.param_col,
                                                                 param_init = param_init_output or exp_global.param_init,
                                                                 bias_init = bias_init_output or exp_global.bias_init)

  def choose_vocab_size(self, vocab_size, vocab, trg_reader):
    """Choose the vocab size for the embedder basd on the passed arguments

    This is done in order of priority of vocab_size, vocab, model+yaml_path
    """
    if vocab_size != None:
      return vocab_size
    elif vocab != None:
      return len(vocab)
    elif trg_reader == None or trg_reader.vocab == None:
      raise ValueError("Could not determine trg_embedder's size. Please set its vocab_size or vocab member explicitly, or specify the vocabulary of trg_reader ahead of time.")
    else:
      return len(trg_reader.vocab)

  def shared_params(self):
    # TODO: one idea to get proper dimensionality in the rnn_layer is to explicitly share these parameters
    #       -- but the rnn_layer would also somehow need to know if it's even being used inside a decoder
    # set([Path(".trg_embed_dim"), Path(".rnn_layer.input_dim")]),
    # set([Path(".input_dim"), Path(".rnn_layer._decoder_input_dim")]),
    # set([Path(".input_feeding"), Path(".rnn_layer._decoder_input_feeding")])
    return [set([Path(".rnn_layer.layers"), Path(".bridge.dec_layers")]),
            set([Path(".rnn_layer.hidden_dim"), Path(".bridge.dec_dim")])]

  def initial_state(self, enc_final_states, ss_expr):
    """Get the initial state of the decoder given the encoder final states.

    :param enc_final_states: The encoder final states.
    :returns: An MlpSoftmaxDecoderState
    """
    rnn_state = self.rnn_layer.initial_state()
    rnn_state = rnn_state.set_s(self.bridge.decoder_init(enc_final_states))
    zeros = dy.zeros(self.input_dim) if self.input_feeding else None
    rnn_state = rnn_state.add_input(dy.concatenate([ss_expr, zeros]))
    return MlpSoftmaxDecoderState(rnn_state=rnn_state, context=zeros)

  def add_input(self, mlp_dec_state, trg_embedding):
    """Add an input and update the state.

    :param mlp_dec_state: An MlpSoftmaxDecoderState object containing the current state.
    :param trg_embedding: The embedding of the word to input.
    :returns: The update MLP decoder state.
    """
    inp = trg_embedding
    if self.input_feeding:
      inp = dy.concatenate([inp, mlp_dec_state.context])
    return MlpSoftmaxDecoderState(rnn_state=mlp_dec_state.rnn_state.add_input(inp),
                                  context=mlp_dec_state.context)

  def get_scores(self, mlp_dec_state):
    """Get scores given a current state.

    :param mlp_dec_state: An MlpSoftmaxDecoderState object.
    :returns: Scores over the vocabulary given this state.
    """
    h_t = dy.tanh(self.context_projector(dy.concatenate([mlp_dec_state.rnn_state.output(), mlp_dec_state.context])))
    return self.vocab_projector(h_t)

  def calc_loss(self, mlp_dec_state, ref_action):
    """
        Label Smoothing is implemented with reference to Section 7 of the paper
        "Rethinking the Inception Architecture for Computer Vision"
        (https://arxiv.org/pdf/1512.00567.pdf)
        """
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

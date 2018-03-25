import dynet as dy

import xnmt.batcher
from xnmt.bridge import CopyBridge
from xnmt.events import register_xnmt_handler, handle_xnmt_event
import xnmt.linear
import xnmt.residual
from xnmt.param_init import GlorotInitializer, ZeroInitializer
from xnmt.param_collection import ParamManager
from xnmt.serialize.serializable import Serializable, bare, Ref, Path
from xnmt.serialize.serializer import serializable_init

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
  def rnn_from_spec(spec, num_layers, input_dim, hidden_dim, model, residual_to_output):
    decoder_type = spec.lower()
    if decoder_type == "lstm":
      return dy.CompactVanillaLSTMBuilder(num_layers, input_dim, hidden_dim, model)
    elif decoder_type == "residuallstm":
      return xnmt.residual.ResidualRNNBuilder(num_layers, input_dim, hidden_dim,
                                         model, residual_to_output)
    else:
      raise RuntimeError("Unknown decoder type {}".format(spec))

class MlpSoftmaxDecoderState(object):
  """A state holding all the information needed for MLPSoftmaxDecoder
  
  Args:
    rnn_state: a DyNet RNN state
    context: a DyNet expression
  """
  def __init__(self, rnn_state=None, context=None):
    self.rnn_state = rnn_state
    self.context = context

class MlpSoftmaxDecoder(RnnDecoder, Serializable):
  """
  Standard MLP softmax decoder.

  Args:
    layers (int): number of LSTM layers
    input_dim (int): input dimension
    lstm_dim (int): LSTM hidden dimension
    mlp_hidden_dim (int): MLP hidden dimension
    trg_embed_dim (int): dimension of target embeddings
    dropout (float): dropout probability for LSTM
    rnn_spec (str): 'lstm' or 'residuallstm'
    residual_to_output (bool): option passed on if rnn_spec == 'residuallstm'
    input_feeding (bool): whether to activate input feeding
    param_init_lstm (ParamInitializer): how to initialize LSTM weight matrices (currently, only :class:`xnmt.param_init.GlorotInitializer` is supported)
    param_init_context (ParamInitializer): how to initialize context weight matrices
    bias_init_context (ParamInitializer): how to initialize context bias vectors
    param_init_output (ParamInitializer): how to initialize output weight matrices
    bias_init_output (ParamInitializer): how to initialize output bias vectors
    bridge (Bridge): how to initialize decoder state
    label_smoothing (float): label smoothing value (if used, 0.1 is a reasonable value).
                             Label Smoothing is implemented with reference to Section 7 of the paper
                             "Rethinking the Inception Architecture for Computer Vision"
                             (https://arxiv.org/pdf/1512.00567.pdf)
    vocab_projector (Linear):
    vocab_size (int): vocab size or None
    vocab (Vocab): vocab or None
    trg_reader (InputReader): Model's trg_reader, if exists and unambiguous.
  """

  # TODO: This should probably take a softmax object, which can be normal or class-factored, etc.
  # For now the default behavior is hard coded.

  yaml_tag = '!MlpSoftmaxDecoder'

  @register_xnmt_handler
  @serializable_init
  def __init__(self,
               layers=1,
               input_dim=Ref(Path("exp_global.default_layer_dim")),
               lstm_dim=Ref(Path("exp_global.default_layer_dim")),
               mlp_hidden_dim=Ref(Path("exp_global.default_layer_dim")),
               trg_embed_dim=Ref(Path("exp_global.default_layer_dim")),
               dropout=Ref(Path("exp_global.dropout"), default=0.0),
               rnn_spec="lstm",
               residual_to_output=False,
               input_feeding=True,
               param_init_lstm=Ref(Path("exp_global.param_init"), default=bare(GlorotInitializer)),
               param_init_context=Ref(Path("exp_global.param_init"), default=bare(GlorotInitializer)),
               bias_init_context=Ref(Path("exp_global.bias_init"), default=bare(ZeroInitializer)),
               param_init_output=Ref(Path("exp_global.param_init"), default=bare(GlorotInitializer)),
               bias_init_output=Ref(Path("exp_global.bias_init"), default=bare(ZeroInitializer)),
               bridge=bare(CopyBridge),
               label_smoothing=0.0,
               vocab_projector=None,
               vocab_size = None,
               vocab = None,
               trg_reader = Ref(path=Path("model.trg_reader"), default=None)):
    self.param_col = ParamManager.my_subcollection(self)
    self.mlp_hidden_dim = mlp_hidden_dim = mlp_hidden_dim
    self.input_dim = input_dim
    self.label_smoothing = label_smoothing
    # Input feeding
    self.input_feeding = input_feeding
    self.lstm_dim = lstm_dim
    lstm_input = trg_embed_dim
    if input_feeding:
      lstm_input += input_dim
    # Bridge
    self.lstm_layers = layers
    self.bridge = bridge

    # LSTM
    self.fwd_lstm  = RnnDecoder.rnn_from_spec(spec       = rnn_spec,
                                              num_layers = layers,
                                              input_dim  = lstm_input,
                                              hidden_dim = lstm_dim,
                                              model = self.param_col,
                                              residual_to_output = residual_to_output)
    if not isinstance(param_init_lstm, GlorotInitializer): raise NotImplementedError("For the decoder LSTM, only Glorot initialization is currently supported")
    if getattr(param_init_lstm,"gain",1.0) != 1.0:
      for l in range(layers):
        for i in [0,1]:
          self.fwd_lstm.param_collection().parameters_list()[3*l+i].scale(param_init_lstm.gain)

    # MLP
    self.context_projector = xnmt.linear.Linear(input_dim  = input_dim + lstm_dim,
                                                output_dim = mlp_hidden_dim,
                                                param_init = param_init_context,
                                                bias_init = bias_init_context)
    self.vocab_size = self.choose_vocab_size(vocab_size, vocab, trg_reader)
    self.vocab_projector = vocab_projector or xnmt.linear.Linear(input_dim = self.mlp_hidden_dim,
                                                                 output_dim = self.vocab_size,
                                                                 param_init = param_init_output,
                                                                 bias_init = bias_init_output)
    # Dropout
    self.dropout = dropout

  def choose_vocab_size(self, vocab_size, vocab, trg_reader):
    """Choose the vocab size for the embedder basd on the passed arguments

    This is done in order of priority of vocab_size, vocab, model+yaml_path

    Args:
      vocab_size (int): vocab size or None
      vocab (Vocab): vocab or None
      yaml_path (Path): Path of this embedder in the component hierarchy. Automatically determined when deserializing the YAML model.
      trg_reader (InputReader): Model's trg_reader, if exists and unambiguous.
    
    Returns:
      int: chosen vocab size
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
    return [set([Path(".layers"), Path(".bridge.dec_layers")]),
            set([Path(".lstm_dim"), Path(".bridge.dec_dim")])]

  def initial_state(self, enc_final_states, ss_expr):
    """Get the initial state of the decoder given the encoder final states.

    Args:
      enc_final_states: The encoder final states. Usually but not necessarily an :class:`xnmt.expression_sequence.ExpressionSequence`
    Returns:
      MlpSoftmaxDecoderState:
    """
    rnn_state = self.fwd_lstm.initial_state()
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
    h_t = dy.tanh(self.context_projector(dy.concatenate([mlp_dec_state.rnn_state.output(), mlp_dec_state.context])))
    return self.vocab_projector(h_t)

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

  @handle_xnmt_event
  def on_set_train(self, val):
    self.fwd_lstm.set_dropout(self.dropout if val else 0.0)


import dynet as dy
import numpy as np
import functools

import xnmt.batcher
import xnmt.linear
import xnmt.residual
from xnmt.param_init import GlorotInitializer, ZeroInitializer
from xnmt import logger
from xnmt.bridge import CopyBridge
from xnmt.lstm import UniLSTMSeqTransducer
from xnmt.mlp import MLP
from xnmt.param_collection import ParamManager
from xnmt.persistence import serializable_init, Serializable, bare, Ref, Path
from xnmt.events import register_xnmt_handler, handle_xnmt_event

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
    return [{".trg_embed_dim", ".rnn_layer.input_dim"},
            {".input_dim", ".rnn_layer.decoder_input_dim"},
            {".input_dim", ".mlp_layer.input_dim"},
            {".input_feeding", ".rnn_layer.decoder_input_feeding"},
            {".rnn_layer.layers", ".bridge.dec_layers"},
            {".rnn_layer.hidden_dim", ".bridge.dec_dim"},
            {".rnn_layer.hidden_dim", ".mlp_layer.decoder_rnn_dim"}]

  def initial_state(self, enc_final_states, ss_expr):
    """Get the initial state of the decoder given the encoder final states.

    Args:
      enc_final_states: The encoder final states. Usually but not necessarily an :class:`xnmt.expression_sequence.ExpressionSequence`
      ss_expr: first input
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

  def get_scores_logsoftmax(self, mlp_dec_state):
    return dy.log_softmax(self.get_scores(mlp_dec_state))

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

class MlpSoftmaxLexiconDecoder(MlpSoftmaxDecoder, Serializable):
  yaml_tag = '!MlpSoftmaxLexiconDecoder'

  @register_xnmt_handler
  @serializable_init
  def __init__(self,
               input_dim=Ref("exp_global.default_layer_dim"),
               trg_embed_dim=Ref("exp_global.default_layer_dim"),
               input_feeding=True,
               rnn_layer=bare(UniLSTMSeqTransducer),
               mlp_layer=bare(MLP),
               bridge=bare(CopyBridge),
               label_smoothing=0.0,
               lexicon_file=None,
               src_vocab=Ref(Path("model.src_reader.vocab")),
               trg_vocab=Ref(Path("model.trg_reader.vocab")),
               attender=Ref(Path("model.attender")),
               lexicon_type='bias',
               lexicon_alpha=0.001,
               linear_projector=None,
               param_init_lin=Ref("exp_global.param_init", default=bare(GlorotInitializer)),
               bias_init_lin=Ref("exp_global.bias_init", default=bare(ZeroInitializer)),
               ):
    super().__init__(input_dim, trg_embed_dim, input_feeding, rnn_layer,
                     mlp_layer, bridge, label_smoothing)
    assert lexicon_file is not None
    self.lexicon_file = lexicon_file
    self.src_vocab = src_vocab
    self.trg_vocab = trg_vocab
    self.attender = attender
    self.lexicon_type = lexicon_type
    self.lexicon_alpha = lexicon_alpha

    self.linear_projector = self.add_serializable_component("linear_projector", linear_projector,
                                                             lambda: xnmt.linear.Linear(input_dim=input_dim,
                                                                                        output_dim=mlp_layer.output_dim))

    if self.lexicon_type == "linear":
      self.lexicon_method = self.linear
    elif self.lexicon_type == "bias":
      self.lexicon_method = self.bias
    else:
      raise ValueError("Unrecognized lexicon method:", lexicon_type, "can only choose between [bias, linear]")

  def load_lexicon(self):
    logger.info("Loading lexicon from file: " + self.lexicon_file)
    assert self.src_vocab.frozen
    assert self.trg_vocab.frozen
    lexicon = [{} for _ in range(len(self.src_vocab))]
    with open(self.lexicon_file, encoding='utf-8') as fp:
      for line in fp:
        try:
          trg, src, prob = line.rstrip().split()
        except:
          logger.warning("Failed to parse 'trg src prob' from:" + line.strip())
          continue
        trg_id = self.trg_vocab.convert(trg)
        src_id = self.src_vocab.convert(src)
        lexicon[src_id][trg_id] = float(prob)
    # Setting the rest of the weight to the unknown word
    for i in range(len(lexicon)):
      sum_prob = sum(lexicon[i].values())
      if sum_prob < 1.0:
        lexicon[i][self.trg_vocab.convert(self.trg_vocab.unk_token)] = 1.0 - sum_prob
    # Overriding special tokens
    src_unk_id = self.src_vocab.convert(self.src_vocab.unk_token)
    trg_unk_id = self.trg_vocab.convert(self.trg_vocab.unk_token)
    lexicon[self.src_vocab.SS] = {self.trg_vocab.SS: 1.0}
    lexicon[self.src_vocab.ES] = {self.trg_vocab.ES: 1.0}
    # TODO(philip30): Note sure if this is intended
    lexicon[src_unk_id] = {trg_unk_id: 1.0}
    return lexicon

  @handle_xnmt_event
  def on_new_epoch(self, training_task, *args, **kwargs):
    if hasattr(self, "lexicon_prob"):
      del self.lexicon_prob
    if not hasattr(self, "lexicon"):
      self.lexicon = self.load_lexicon()

  @handle_xnmt_event
  def on_start_sent(self, src):
    batch_size = len(src)
    col_size = len(src[0])

    idxs = [(x, j, i) for i in range(batch_size) for j in range(col_size) for x in self.lexicon[src[i][j]].keys()]
    idxs = tuple(map(list, list(zip(*idxs))))

    values = [x for i in range(batch_size) for j in range(col_size) for x in self.lexicon[src[i][j]].values()]
    self.lexicon_prob = dy.nobackprop(dy.sparse_inputTensor(idxs, values, (len(self.trg_vocab), col_size, batch_size), batched=True))
    
  def get_scores_logsoftmax(self, mlp_dec_state):
    score = super().get_scores(mlp_dec_state)
    lex_prob = self.lexicon_prob * self.attender.get_last_attention()
    # Note that the sum dim is only summing a tensor of 1 size in dim 1.
    # This is to make sure that the shape of the returned tensor matches the vanilla decoder
    return dy.sum_dim(self.lexicon_method(mlp_dec_state, score, lex_prob), [1])

  def linear(self, mlp_dec_state, score, lex_prob):
    coef = dy.logistic(self.linear_projector(mlp_dec_state.rnn_state.output()))
    return dy.log(dy.cmult(dy.softmax(score), coef) + dy.cmult((1-coef), lex_prob))

  def bias(self, mlp_dec_state, score, lex_prob):
    return dy.log_softmax(score + dy.log(lex_prob + self.lexicon_alpha))

  def calc_loss(self, mlp_dec_state, ref_action):
    logsoft = self.get_scores_logsoftmax(mlp_dec_state)
    if not xnmt.batcher.is_batched(ref_action):
      return -dy.pick(logsoft, ref_action)
    else:
      return -dy.pick_batch(logsoft, ref_action)


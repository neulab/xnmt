from typing import Any

import dynet as dy

import xnmt.batcher
import xnmt.residual
from xnmt.param_init import GlorotInitializer, ZeroInitializer
from xnmt import logger
from xnmt.bridge import Bridge, CopyBridge
from xnmt.lstm import UniLSTMSeqTransducer
from xnmt.param_collection import ParamManager
from xnmt.persistence import serializable_init, Serializable, bare, Ref, Path
from xnmt.events import register_xnmt_handler, handle_xnmt_event
from xnmt.transform import Linear, AuxNonLinear, NonLinear, Transform
from xnmt.scorer import Scorer, Softmax

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
    label_smoothing: label smoothing value (if used, 0.1 is a reasonable value).
                     Label Smoothing is implemented with reference to Section 7 of the paper
                     "Rethinking the Inception Architecture for Computer Vision"
                     (https://arxiv.org/pdf/1512.00567.pdf)
  """

  yaml_tag = '!AutoRegressiveDecoder'

  @serializable_init
  def __init__(self,
               input_dim: int = Ref("exp_global.default_layer_dim"),
               trg_embed_dim: int = Ref("exp_global.default_layer_dim"),
               input_feeding: bool = True,
               bridge: Bridge = bare(CopyBridge),
               rnn: UniLSTMSeqTransducer = bare(UniLSTMSeqTransducer),
               transform: Transform = bare(AuxNonLinear),
               scorer: Scorer = bare(Softmax),
               truncate_dec_batches: bool = Ref("exp_global.truncate_dec_batches", default=False)) -> None:
    self.param_col = ParamManager.my_params(self)
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
    rnn_state = rnn_state.set_s(self.bridge.decoder_init(enc_final_states))
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
    if self.truncate_dec_batches: rnn_state, inp = xnmt.batcher.truncate_batches(rnn_state, inp)
    return AutoRegressiveDecoderState(rnn_state=rnn_state.add_input(inp),
                                      context=mlp_dec_state.context)

  def _calc_transform(self, mlp_dec_state: AutoRegressiveDecoderState) -> dy.Expression:
    h = dy.concatenate([mlp_dec_state.rnn_state.output(), mlp_dec_state.context])
    return self.transform(h)

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

# TODO: This should be factored to simply use Softmax
# class AutoRegressiveLexiconDecoder(AutoRegressiveDecoder, Serializable):
#   yaml_tag = '!AutoRegressiveLexiconDecoder'
# 
#   @register_xnmt_handler
#   @serializable_init
#   def __init__(self,
#                input_dim=Ref("exp_global.default_dim"),
#                trg_embed_dim=Ref("exp_global.default_dim"),
#                input_feeding=True,
#                rnn=bare(UniLSTMSeqTransducer),
#                mlp=bare(AttentionalOutputMLP),
#                bridge=bare(CopyBridge),
#                label_smoothing=0.0,
#                lexicon_file=None,
#                src_vocab=Ref(Path("model.src_reader.vocab")),
#                trg_vocab=Ref(Path("model.trg_reader.vocab")),
#                attender=Ref(Path("model.attender")),
#                lexicon_type='bias',
#                lexicon_alpha=0.001,
#                linear_projector=None,
#                truncate_dec_batches: bool = Ref("exp_global.truncate_dec_batches", default=False),
#                param_init_lin=Ref("exp_global.param_init", default=bare(GlorotInitializer)),
#                bias_init_lin=Ref("exp_global.bias_init", default=bare(ZeroInitializer)),
#                ) -> None:
#     super().__init__(input_dim, trg_embed_dim, input_feeding, rnn,
#                      mlp, bridge, truncate_dec_batches, label_smoothing)
#     assert lexicon_file is not None
#     self.lexicon_file = lexicon_file
#     self.src_vocab = src_vocab
#     self.trg_vocab = trg_vocab
#     self.attender = attender
#     self.lexicon_type = lexicon_type
#     self.lexicon_alpha = lexicon_alpha
# 
#     self.linear_projector = self.add_serializable_component("linear_projector", linear_projector,
#                                                              lambda: xnmt.linear.Linear(input_dim=input_dim,
#                                                                                         output_dim=mlp.output_dim))
# 
#     if self.lexicon_type == "linear":
#       self.lexicon_method = self.linear
#     elif self.lexicon_type == "bias":
#       self.lexicon_method = self.bias
#     else:
#       raise ValueError("Unrecognized lexicon method:", lexicon_type, "can only choose between [bias, linear]")
# 
#   def load_lexicon(self):
#     logger.info("Loading lexicon from file: " + self.lexicon_file)
#     assert self.src_vocab.frozen
#     assert self.trg_vocab.frozen
#     lexicon = [{} for _ in range(len(self.src_vocab))]
#     with open(self.lexicon_file, encoding='utf-8') as fp:
#       for line in fp:
#         try:
#           trg, src, prob = line.rstrip().split()
#         except:
#           logger.warning("Failed to parse 'trg src prob' from:" + line.strip())
#           continue
#         trg_id = self.trg_vocab.convert(trg)
#         src_id = self.src_vocab.convert(src)
#         lexicon[src_id][trg_id] = float(prob)
#     # Setting the rest of the weight to the unknown word
#     for i in range(len(lexicon)):
#       sum_prob = sum(lexicon[i].values())
#       if sum_prob < 1.0:
#         lexicon[i][self.trg_vocab.convert(self.trg_vocab.unk_token)] = 1.0 - sum_prob
#     # Overriding special tokens
#     src_unk_id = self.src_vocab.convert(self.src_vocab.unk_token)
#     trg_unk_id = self.trg_vocab.convert(self.trg_vocab.unk_token)
#     lexicon[self.src_vocab.SS] = {self.trg_vocab.SS: 1.0}
#     lexicon[self.src_vocab.ES] = {self.trg_vocab.ES: 1.0}
#     # TODO(philip30): Note sure if this is intended
#     lexicon[src_unk_id] = {trg_unk_id: 1.0}
#     return lexicon
# 
#   @handle_xnmt_event
#   def on_new_epoch(self, training_task, *args, **kwargs):
#     if hasattr(self, "lexicon_prob"):
#       del self.lexicon_prob
#     if not hasattr(self, "lexicon"):
#       self.lexicon = self.load_lexicon()
# 
#   @handle_xnmt_event
#   def on_start_sent(self, src):
#     batch_size = len(src)
#     col_size = len(src[0])
# 
#     idxs = [(x, j, i) for i in range(batch_size) for j in range(col_size) for x in self.lexicon[src[i][j]].keys()]
#     idxs = tuple(map(list, list(zip(*idxs))))
# 
#     values = [x for i in range(batch_size) for j in range(col_size) for x in self.lexicon[src[i][j]].values()]
#     self.lexicon_prob = dy.nobackprop(dy.sparse_inputTensor(idxs, values, (len(self.trg_vocab), col_size, batch_size), batched=True))
#     
#   def calc_scores_logsoftmax(self, mlp_dec_state):
#     score = super().calc_scores(mlp_dec_state)
#     lex_prob = self.lexicon_prob * self.attender.get_last_attention()
#     # Note that the sum dim is only summing a tensor of 1 size in dim 1.
#     # This is to make sure that the shape of the returned tensor matches the vanilla decoder
#     return dy.sum_dim(self.lexicon_method(mlp_dec_state, score, lex_prob), [1])
# 
#   def linear(self, mlp_dec_state, score, lex_prob):
#     coef = dy.logistic(self.linear_projector(mlp_dec_state.rnn_state.output()))
#     return dy.log(dy.cmult(dy.softmax(score), coef) + dy.cmult((1-coef), lex_prob))
# 
#   def bias(self, mlp_dec_state, score, lex_prob):
#     return dy.log_softmax(score + dy.log(lex_prob + self.lexicon_alpha))
# 
#   def calc_loss(self, mlp_dec_state, ref_action):
#     logsoft = self.calc_scores_logsoftmax(mlp_dec_state)
#     if not xnmt.batcher.is_batched(ref_action):
#       return -dy.pick(logsoft, ref_action)
#     else:
#       return -dy.pick_batch(logsoft, ref_action)


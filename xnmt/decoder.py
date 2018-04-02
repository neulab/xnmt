import dynet as dy
from xnmt.serialize.serializable import Serializable, bare
from xnmt.serialize.tree_tools import Ref, Path
import xnmt.batcher
from xnmt.events import register_handler, handle_xnmt_event
import xnmt.linear
import xnmt.residual
from xnmt.bridge import CopyBridge
import numpy as np
from xnmt.vocab import Vocab
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
    exp_global (ExpGlobal): ExpGlobal object to acquire DyNet params and global settings. By default, references the experiment's top level exp_global object.
    layers (int): number of LSTM layers
    input_dim (int): input dimension; if None, use ``exp_global.default_layer_dim``
    lstm_dim (int): LSTM hidden dimension; if None, use ``exp_global.default_layer_dim``
    mlp_hidden_dim (int): MLP hidden dimension; if None, use ``exp_global.default_layer_dim``
    trg_embed_dim (int): dimension of target embeddings; if None, use ``exp_global.default_layer_dim``
    dropout (float): dropout probability for LSTM; if None, use exp_global.dropout
    rnn_spec (str): 'lstm' or 'residuallstm'
    residual_to_output (bool): option passed on if rnn_spec == 'residuallstm'
    input_feeding (bool): whether to activate input feeding
    param_init_lstm (ParamInitializer): how to initialize LSTM weight matrices (currently, only :class:`xnmt.param_init.GlorotInitializer` is supported); if None, use ``exp_global.param_init``
    param_init_context (ParamInitializer): how to initialize context weight matrices; if None, use ``exp_global.param_init``
    bias_init_context (ParamInitializer): how to initialize context bias vectors; if None, use ``exp_global.bias_init``
    param_init_output (ParamInitializer): how to initialize output weight matrices; if None, use ``exp_global.param_init``
    bias_init_output (ParamInitializer): how to initialize output bias vectors; if None, use ``exp_global.bias_init``
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

  def __init__(self, exp_global=Ref(Path("exp_global")), layers=1, input_dim=None, lstm_dim=None,
               mlp_hidden_dim=None, trg_embed_dim=None, dropout=None,
               rnn_spec="lstm", residual_to_output=False, input_feeding=True,
               param_init_lstm=None, param_init_context=None, bias_init_context=None,
               param_init_output=None, bias_init_output=None,
               bridge=bare(CopyBridge), label_smoothing=0.0,
               vocab_projector=None, vocab_size = None, vocab = None,
               trg_reader = Ref(path=Path("model.trg_reader"), required=False)):
    register_handler(self)
    self.param_col = exp_global.dynet_param_collection.param_col
    # Define dim
    lstm_dim       = lstm_dim or exp_global.default_layer_dim
    self.mlp_hidden_dim = mlp_hidden_dim = mlp_hidden_dim or exp_global.default_layer_dim
    trg_embed_dim  = trg_embed_dim or exp_global.default_layer_dim
    input_dim      = input_dim or exp_global.default_layer_dim
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
    param_init_lstm = param_init_lstm or exp_global.param_init
    if not isinstance(param_init_lstm, GlorotInitializer): raise NotImplementedError("For the decoder LSTM, only Glorot initialization is currently supported")
    if getattr(param_init_lstm,"gain",1.0) != 1.0:
      for l in range(layers):
        for i in [0,1]:
          self.fwd_lstm.param_collection().parameters_list()[3*l+i].scale(param_init_lstm.gain)
      
    # MLP
    self.context_projector = xnmt.linear.Linear(input_dim  = input_dim + lstm_dim,
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
    # Dropout
    self.dropout = dropout or exp_global.dropout

  def choose_vocab_size(self, vocab_size, vocab, trg_reader):
    """Choose the vocab size for the embedder basd on the passed arguments

    This is done in order of priority of vocab_size, vocab, model+yaml_path

    Args:
      vocab_size (int): vocab size or None
      vocab (Vocab): vocab or None
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

class OpenNonterm:
  def __init__(self, label=None, parent_state=None, is_sibling=False, sib_state=None):
    self.label = label
    self.parent_state = parent_state
    self.is_sibling = is_sibling
    self.sib_state = sib_state

class TreeDecoderState:
  """A state holding all the information needed for MLPSoftmaxDecoder"""
  def __init__(self, rnn_state=None, word_rnn_state=None, context=None, word_context=None,
               states=[], tree=None, open_nonterms=[], stop_action=False):
    self.rnn_state = rnn_state
    self.context = context
    self.word_rnn_state = word_rnn_state
    self.word_context = word_context
    # training time
    self.states = states
    self.tree = tree

    # decoding time
    self.open_nonterms = open_nonterms
    self.stop_action = stop_action

  def copy(self):
    open_nonterms_copy = []
    for n in self.open_nonterms:
      open_nonterms_copy.append(OpenNonterm(n.label, n.parent_state, n.is_sibling, n.sib_state))
    return TreeDecoderState(rnn_state=self.rnn_state, word_rnn_state=self.word_rnn_state, context=self.context, word_context=self.word_context,
                            open_nonterms=open_nonterms_copy, stop_action=self.stop_action)

class TreeHierDecoder(RnnDecoder, Serializable):
  # TreeHierDecoder final version
  yaml_tag = u'!TreeHierDecoder'
  def __init__(self, exp_global = Ref(Path("exp_global")),
               trg_reader = Ref(Path("model.trg_reader")),
               layers=1, input_dim=None, lstm_dim=None,
               mlp_hidden_dim=None, trg_embed_dim=None, dropout=None,
               rnn_spec="lstm", residual_to_output=False, input_feeding=True,
               bridge=bare(CopyBridge), start_nonterm='ROOT', bpe_stop=False):

    register_handler(self)
    self.param_col = exp_global.dynet_param_collection.param_col
    # best setups
    vocab_size = len(trg_reader.vocab)
    word_vocab_size = len(trg_reader.word_vocab)
    self.set_word_lstm = True
    self.start_nonterm = start_nonterm
    self.rule_label_smooth = -1
    self.rule_size = vocab_size

    self.bpe_stop = bpe_stop
    # Define dim
    lstm_dim       = lstm_dim or exp_global.default_layer_dim
    mlp_hidden_dim = mlp_hidden_dim or exp_global.default_layer_dim
    trg_embed_dim  = trg_embed_dim or exp_global.default_layer_dim
    input_dim      = input_dim or exp_global.default_layer_dim
    self.input_dim = input_dim
    # Input feeding
    self.input_feeding = input_feeding
    self.lstm_dim = lstm_dim
    self.trg_embed_dim = trg_embed_dim
    rule_lstm_input = trg_embed_dim
    word_lstm_input = trg_embed_dim
    if input_feeding:
      rule_lstm_input += input_dim
      word_lstm_input += input_dim

    # parent state + wordRNN output
    rule_lstm_input += lstm_dim*2
    # ruleRNN output
    word_lstm_input += lstm_dim

    self.rule_lstm_input = rule_lstm_input
    self.word_lstm_input = word_lstm_input
    # Bridge
    self.lstm_layers = layers
    self.bridge = bridge

    # LSTM
    self.fwd_lstm  = RnnDecoder.rnn_from_spec(spec       = rnn_spec,
                                              num_layers = layers,
                                              input_dim  = rule_lstm_input,
                                              hidden_dim = lstm_dim,
                                              model = self.param_col,
                                              residual_to_output = residual_to_output)

    self.word_lstm = RnnDecoder.rnn_from_spec(spec       = rnn_spec,
                                              num_layers = layers,
                                              input_dim  = word_lstm_input,
                                              hidden_dim = lstm_dim,
                                              model = self.param_col,
                                              residual_to_output = residual_to_output)
    # MLP
    self.rule_context_projector = xnmt.linear.Linear(input_dim=2*input_dim + 2*lstm_dim,
                                                output_dim=mlp_hidden_dim,
                                                model= self.param_col)
    self.word_context_projector = xnmt.linear.Linear(input_dim=2*input_dim + 2*lstm_dim,
                                                output_dim=mlp_hidden_dim,
                                                model=self.param_col)
    self.rule_vocab_projector = xnmt.linear.Linear(input_dim = mlp_hidden_dim,
                                         output_dim = vocab_size,
                                         model = self.param_col)
    self.word_vocab_projector = xnmt.linear.Linear(input_dim = mlp_hidden_dim,
                                         output_dim = word_vocab_size,
                                         model = self.param_col)
    # Dropout
    self.dropout = dropout or exp_global.dropout

  def shared_params(self):
    return [set([Path(".layers"), Path(".bridge.dec_layers")]),
            set([Path(".lstm_dim"), Path(".bridge.dec_dim")])]

  def initial_state(self, enc_final_states, ss_expr, decoding=False):
    """Get the initial state of the decoder given the encoder final states.

    :param enc_final_states: The encoder final states.
    :returns: An MlpSoftmaxDecoderState
    """
    init_state = self.bridge.decoder_init(enc_final_states)
    # init_state: [c, h]
    word_rnn_state = self.word_lstm.initial_state()
    word_rnn_state = word_rnn_state.set_s(init_state)
    zeros_word_rnn = dy.zeros(self.word_lstm_input - self.trg_embed_dim)
    word_rnn_state = word_rnn_state.add_input(dy.concatenate([ss_expr, zeros_word_rnn]))

    rnn_state = self.fwd_lstm.initial_state()
    rnn_state = rnn_state.set_s(init_state)
    zeros_rnn = dy.zeros(self.rule_lstm_input - self.trg_embed_dim)
    rnn_state = rnn_state.add_input(dy.concatenate([ss_expr, zeros_rnn]))

    self.decoding = decoding
    if decoding:
      zeros_lstm = dy.zeros(self.lstm_dim)
      return TreeDecoderState(rnn_state=rnn_state, context=zeros_rnn, word_rnn_state=word_rnn_state, word_context=zeros_word_rnn,  \
          open_nonterms=[OpenNonterm(self.start_nonterm, parent_state=zeros_lstm, sib_state=zeros_lstm)])
    else:
      batch_size = ss_expr.dim()[1]
      return TreeDecoderState(rnn_state=rnn_state, context=zeros_rnn, word_rnn_state=word_rnn_state, word_context=zeros_word_rnn, \
          states=np.array([dy.zeros((self.lstm_dim,), batch_size=batch_size)]))

  def add_input(self, tree_dec_state, trg, word_embedder, rule_embedder,
                trg_rule_vocab=None, word_vocab=None, tag_embedder=None):
    """Add an input and update the state.

    :param tree_dec_state: An TreeDecoderState object containing the current state.
    :param trg_embedding: The embedding of the word to input.
    :param trg: The data list of the target word, with the first element as the word index, the rest as timestep.
    :param trg_rule_vocab: RuleVocab object used at decoding time
    :returns: The update MLP decoder state.
    """
    word_rnn_state = tree_dec_state.word_rnn_state
    rnn_state = tree_dec_state.rnn_state
    if not self.decoding:
      # get parent states for this batch
      #batch_size = trg_embedding.dim()[1]
      #assert batch_size == 1
      states = tree_dec_state.states
      paren_tm1_states = tree_dec_state.states[trg.get_col(1)] # ((hidden_dim,), batch_size) * batch_size
      is_terminal = trg.get_col(3, batched=False)
      paren_tm1_state = paren_tm1_states[0]
      if is_terminal[0] == 0:
        # rule rnn
        rule_idx = trg.get_col(0)
        inp = rule_embedder.embed(rule_idx)
        if self.input_feeding:
          inp = dy.concatenate([inp, tree_dec_state.context])
        inp = dy.concatenate([inp, paren_tm1_state, word_rnn_state.output()])

        rnn_state = rnn_state.add_input(inp)
        states = np.append(states, rnn_state.output())
      else:
        # word rnn
        word_idx = trg.get_col(0)
        # if this is end of phrase append states list
        inp = word_embedder.embed(word_idx)
        if self.input_feeding:
          inp = dy.concatenate([inp, tree_dec_state.word_context])
        inp = dy.concatenate([inp, paren_tm1_state])
        word_rnn_state = word_rnn_state.add_input(inp)
        # update rule RNN
        rnn_inp = dy.concatenate([dy.zeros(self.rule_lstm_input - self.lstm_dim),
                                  word_rnn_state.output()])
        rnn_state = rnn_state.add_input(rnn_inp)
        # if this is end of phrase append states list
        if self.bpe_stop:
          word = word_vocab[word_idx[0]]
          if not word.endswith('@@'):
            states = np.append(states, rnn_state.output())
        else:
          if word_idx[0] == Vocab.ES:
            states = np.append(states, rnn_state.output())

      return TreeDecoderState(rnn_state=rnn_state, context=tree_dec_state.context, word_rnn_state=word_rnn_state, word_context=tree_dec_state.word_context, \
                           states=states)
    else:
      open_nonterms = tree_dec_state.open_nonterms[:]
      stop_action = tree_dec_state.stop_action
      if open_nonterms[-1].label == u'*':
        inp = word_embedder.embed(trg)
        if self.input_feeding:
          inp = dy.concatenate([inp, tree_dec_state.word_context])
        inp = dy.concatenate([inp, tree_dec_state.open_nonterms[-1].parent_state])
        word_rnn_state = word_rnn_state.add_input(inp)
        rnn_inp = dy.concatenate([dy.zeros(self.rule_lstm_input - self.lstm_dim),
                                 word_rnn_state.output()])
        rnn_state = rnn_state.add_input(rnn_inp)
        if self.bpe_stop:
          word = word_vocab[trg]
          #if word.endswith(u'\u2581'):
          if not word.endswith('@@'):
            open_nonterms.pop()
            stop_action = True
        else:
          if trg == Vocab.ES:
            open_nonterms.pop()
      else:
        inp = rule_embedder.embed(trg)
        if self.input_feeding:
          inp = dy.concatenate([inp, tree_dec_state.context])
        cur_nonterm = open_nonterms.pop()
        rule = trg_rule_vocab[trg]
        if cur_nonterm.label != rule.lhs:
          for c in cur_nonterm:
            print(c.label)
        assert cur_nonterm.label == rule.lhs, "the lhs of the current input rule %s does not match the next open nonterminal %s" % (rule.lhs, cur_nonterm.label)

        inp = dy.concatenate([inp, cur_nonterm.parent_state, word_rnn_state.output()])
        rnn_state = rnn_state.add_input(inp)
        # add rule to tree_dec_state.open_nonterms
        new_open_nonterms = []
        for rhs in rule.rhs:
          if rhs in rule.open_nonterms:
            new_open_nonterms.append(OpenNonterm(rhs, parent_state=rnn_state.output()))
        new_open_nonterms.reverse()
        open_nonterms.extend(new_open_nonterms)
      return TreeDecoderState(rnn_state=rnn_state, context=tree_dec_state.context, word_rnn_state=word_rnn_state, word_context=tree_dec_state.word_context,\
                              open_nonterms=open_nonterms, stop_action=stop_action)

  def get_scores(self, tree_dec_state, trg_rule_vocab, is_terminal, label_idx=-1, sample_len=False):
    """Get scores given a current state.

    :param mlp_dec_state: An MlpSoftmaxDecoderState object.
    :returns: Scores over the vocabulary given this state.
    """
    if is_terminal:
      inp = dy.concatenate([tree_dec_state.word_rnn_state.output(), tree_dec_state.word_context,
                            tree_dec_state.rnn_state.output(), tree_dec_state.context])
      h_t = dy.tanh(self.word_context_projector(inp))
      return self.word_vocab_projector(h_t), -1, None, None
    else:
      inp = dy.concatenate([tree_dec_state.rnn_state.output(), tree_dec_state.context,
                            tree_dec_state.word_rnn_state.output(), tree_dec_state.word_context])
      h_t = dy.tanh(self.rule_context_projector(inp))
    if self.rule_label_smooth > 0:
      proj = dy.cmult(self.rule_vocab_projector(h_t), dy.scalarInput(1. - self.rule_label_smooth)) \
             + dy.scalarInput(self.rule_label_smooth / float(self.rule_size))
    else:
      proj = self.rule_vocab_projector(h_t)
    if label_idx >= 0:
      # training
      return proj, -1, None, None
      #label = trg_rule_vocab.tag_vocab[label_idx]
      #valid_y_index = trg_rule_vocab.rule_index_with_lhs(label)
    else:
      valid_y_index = trg_rule_vocab.rule_index_with_lhs(tree_dec_state.open_nonterms[-1].label)
    if not valid_y_index:
      print('warning: no rule with lhs: {}'.format(tree_dec_state.open_nonterms[-1].label))
    valid_y_mask = np.ones((len(trg_rule_vocab),)) * (-1000)
    valid_y_mask[valid_y_index] = 0.

    return proj + dy.inputTensor(valid_y_mask), len(valid_y_index), None, None

  def calc_loss(self, tree_dec_state, ref_action, trg_rule_vocab):
    ref_word = ref_action.get_col(0)
    is_terminal = ref_action.get_col(3)[0]
    is_stop =ref_action.get_col(4)[0]
    #leaf_len = ref_action.get_col(5)[0]

    scores, valid_y_len, stop_prob, len_scores = self.get_scores(tree_dec_state, trg_rule_vocab,
                                                                 is_terminal, label_idx=1, sample_len=False)
    # single mode
    if not xnmt.batcher.is_batched(ref_action):
      word_loss = dy.pickneglogsoftmax(scores, ref_word)
    # minibatch mode
    else:
      word_loss = dy.pickneglogsoftmax_batch(scores, ref_word)
    return word_loss

  def set_train(self, val):
    self.fwd_lstm.set_dropout(self.dropout if val else 0.0)
    if self.set_word_lstm:
      self.word_lstm.set_dropout(self.dropout if val else 0.0)

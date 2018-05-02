import dynet as dy

from xnmt.param_collection import ParamManager
from xnmt.param_init import GlorotInitializer, ZeroInitializer
from xnmt.persistence import serializable_init, Serializable, Ref, bare
import xnmt.linear

class MLP(Serializable):
  """
  Multi-layer perceptron.

  Args:
    input_dim (int): input dimension
    hidden_dim (int): hidden dimension
    output_dim (int): output dimension; if ``yaml_path`` contains 'decoder', this argument will be ignored (and set via ``vocab_size``/``vocab``/``trg_reader`` instead)
    param_init_hidden (ParamInitializer): how to initialize hidden weight matrices
    bias_init_hidden (ParamInitializer): how to initialize hidden bias vectors
    param_init_output (ParamInitializer): how to initialize output weight matrices
    bias_init_output (ParamInitializer): how to initialize output bias vectors
    activation (str): One of ``tanh``, ``relu``, ``sigmoid``, ``elu``, ``selu``, ``asinh`` or ``identity``. Defaults to ``tanh``
    output_projector:
    yaml_path (str):
    vocab_size (int): vocab size or None; if not None and ``yaml_path`` contains 'decoder', this will overwrite ``output_dim``
    vocab (Vocab): vocab or None; if not None and ``yaml_path`` contains 'decoder', this will overwrite ``output_dim``
    trg_reader (InputReader): Model's trg_reader, if exists and unambiguous; if not None and ``yaml_path`` contains 'decoder', this will overwrite ``output_dim``
    decoder_rnn_dim (int): dimension of a decoder RNN that feeds into this MLP; if ``yaml_path`` contains 'decoder', this will be added to ``input_dim``
  """
  yaml_tag = '!MLP'

  @serializable_init
  def __init__(self,
               input_dim=Ref("exp_global.default_layer_dim"),
               hidden_dim=Ref("exp_global.default_layer_dim"),
               output_dim=Ref("exp_global.default_layer_dim", default=None),
               param_init_hidden=Ref("exp_global.param_init", default=bare(GlorotInitializer)),
               bias_init_hidden=Ref("exp_global.bias_init", default=bare(ZeroInitializer)),
               param_init_output=Ref("exp_global.param_init", default=bare(GlorotInitializer)),
               bias_init_output=Ref("exp_global.bias_init", default=bare(ZeroInitializer)),
               activation='tanh',
               hidden_layer=None,
               output_projector=None,
               yaml_path=None,
               vocab_size=None,
               vocab=None,
               trg_reader=Ref("model.trg_reader", default=None),
               decoder_rnn_dim=Ref("exp_global.default_layer_dim", default=None)):
    model = ParamManager.my_params(self)
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    if yaml_path is not None and "decoder" in yaml_path:
      self.input_dim += decoder_rnn_dim
      self.output_dim = self.choose_vocab_size(vocab_size, vocab, trg_reader)
      self.save_processed_arg("vocab_size", self.output_dim)

    self.hidden_layer = self.add_serializable_component("hidden_layer", hidden_layer,
                                                        lambda: xnmt.linear.Linear(input_dim=self.input_dim,
                                                                                   output_dim=self.hidden_dim,
                                                                                   param_init=param_init_hidden,
                                                                                   bias_init=bias_init_hidden))
    if activation == 'tanh':
      self.activation = dy.tanh
    elif activation == 'relu':
      self.activation = dy.rectify
    elif activation == 'sigmoid':
      self.activation = dy.sigmoid
    elif activation == 'elu':
      self.activation = dy.elu
    elif activation == 'selu':
      self.activation = dy.selu
    elif activation == 'asinh':
      self.activation = dy.asinh
    elif activation == 'identity':
      def identity(x): return x
      self.activation = identity
    else:
      raise ValueError('Unknown activation %s' % activation)

    self.output_projector = self.add_serializable_component("output_projector", output_projector,
                                                            lambda: output_projector or xnmt.linear.Linear(
                                                              input_dim=self.hidden_dim, output_dim=self.output_dim,
                                                              param_init=param_init_output, bias_init=bias_init_output))

  def __call__(self, input_expr):
    return self.output_projector(self.activation(self.hidden_layer(input_expr)))

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
    if vocab_size is not None:
      return vocab_size
    elif vocab is not None:
      return len(vocab)
    elif trg_reader is None or trg_reader.vocab is None:
      raise ValueError("Could not determine MLP's output size. Please set its vocab_size or vocab member explicitly, or specify the vocabulary of trg_reader ahead of time.")
    else:
      return len(trg_reader.vocab)


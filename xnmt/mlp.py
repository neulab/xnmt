from typing import Optional

import dynet as dy

from xnmt.param_init import ParamInitializer, GlorotInitializer, ZeroInitializer
from xnmt.persistence import serializable_init, Serializable, Ref, bare
from xnmt import linear, vocab, input_reader


class MLP(Serializable):
  """
  Multi-layer perceptron.

  Args:
    input_dim: input dimension
    hidden_dim: hidden dimension
    output_dim: output dimension
    param_init_hidden: how to initialize hidden weight matrices
    bias_init_hidden: how to initialize hidden bias vectors
    param_init_output: how to initialize output weight matrices
    bias_init_output: how to initialize output bias vectors
    activation: One of ``tanh``, ``relu``, ``sigmoid``, ``elu``, ``selu``, ``asinh`` or ``identity``.
    hidden_layer: hidden layer linear subcomponent (created automatically)
    output_projector: output layer linear subcomponent (created automatically)
  """
  yaml_tag = '!MLP'

  @serializable_init
  def __init__(self,
               input_dim: int = Ref("exp_global.default_layer_dim"),
               hidden_dim: int = Ref("exp_global.default_layer_dim"),
               output_dim: int = Ref("exp_global.default_layer_dim", default=None),
               param_init_hidden: ParamInitializer = Ref("exp_global.param_init", default=bare(GlorotInitializer)),
               bias_init_hidden: ParamInitializer = Ref("exp_global.bias_init", default=bare(ZeroInitializer)),
               param_init_output: ParamInitializer = Ref("exp_global.param_init", default=bare(GlorotInitializer)),
               bias_init_output: ParamInitializer = Ref("exp_global.bias_init", default=bare(ZeroInitializer)),
               activation: str = 'tanh',
               hidden_layer=None,
               output_projector=None):
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim

    self.hidden_layer = self.add_serializable_component("hidden_layer", hidden_layer,
                                                        lambda: linear.Linear(input_dim=self.input_dim,
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
      def identity(x):
        return x

      self.activation = identity
    else:
      raise ValueError('Unknown activation %s' % activation)

    self.output_projector = self.add_serializable_component("output_projector", output_projector,
                                                            lambda: output_projector or linear.Linear(
                                                              input_dim=self.hidden_dim, output_dim=self.output_dim,
                                                              param_init=param_init_output, bias_init=bias_init_output))

  def __call__(self, input_expr):
    return self.output_projector(self.activation(self.hidden_layer(input_expr)))


class OutputMLP(MLP, Serializable):
  """
  Multi-layer perceptron to be used as output layer with some convenience features.

  Args:
    input_dim: input dimension
    hidden_dim: hidden dimension
    output_dim: ignored
    param_init_hidden: how to initialize hidden weight matrices
    bias_init_hidden: how to initialize hidden bias vectors
    param_init_output: how to initialize output weight matrices
    bias_init_output: how to initialize output bias vectors
    activation: One of ``tanh``, ``relu``, ``sigmoid``, ``elu``, ``selu``, ``asinh`` or ``identity``.
    hidden_layer: hidden layer linear subcomponent (created automatically)
    output_projector: output layer linear subcomponent (created automatically)
    vocab_size: if not ``None`', this will be used as output dimension
    vocab: if not ``None`` and ``vocab_size`` was not specified, this will be used as output dimension
    trg_reader: Model's trg_reader, if exists and unambiguous; if not None and ``vocab_size``/``vocab`` were not
          specified, this will be used as output dimension
  """
  yaml_tag = '!OutputMLP'

  @serializable_init
  def __init__(self,
               input_dim: int = Ref("exp_global.default_layer_dim"),
               hidden_dim: int = Ref("exp_global.default_layer_dim"),
               output_dim: int = Ref("exp_global.default_layer_dim", default=None),
               param_init_hidden: ParamInitializer = Ref("exp_global.param_init", default=bare(GlorotInitializer)),
               bias_init_hidden: ParamInitializer = Ref("exp_global.bias_init", default=bare(ZeroInitializer)),
               param_init_output: ParamInitializer = Ref("exp_global.param_init", default=bare(GlorotInitializer)),
               bias_init_output: ParamInitializer = Ref("exp_global.bias_init", default=bare(ZeroInitializer)),
               activation: str = 'tanh',
               hidden_layer=None,
               output_projector=None,
               vocab_size: Optional[int] = None,
               vocab: Optional[vocab.Vocab] = None,
               trg_reader: Optional[input_reader.InputReader] = Ref("model.trg_reader", default=None)):
    output_dim = self._choose_vocab_size(vocab_size, vocab, trg_reader)
    self.save_processed_arg("vocab_size", output_dim)
    super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                     param_init_hidden=param_init_hidden, bias_init_hidden=bias_init_hidden,
                     param_init_output=param_init_output, bias_init_output=bias_init_output,
                     activation=activation, hidden_layer=hidden_layer, output_projector=output_projector)

  def _choose_vocab_size(self, vocab_size: Optional[int], vocab: Optional[vocab.Vocab],
                         trg_reader: Optional[input_reader.InputReader]) -> int:
    """Choose the vocab size for the embedder based on the passed arguments.

    This is done in order of priority of vocab_size, vocab, model

    Args:
      vocab_size: vocab size or None
      vocab: vocab or None
      trg_reader: Model's trg_reader, if exists and unambiguous.

    Returns:
      chosen vocab size
    """
    if vocab_size is not None:
      return vocab_size
    elif vocab is not None:
      return len(vocab)
    elif trg_reader is None or trg_reader.vocab is None:
      raise ValueError(
        "Could not determine MLP's output size. "
        "Please set its vocab_size or vocab member explicitly, or specify the vocabulary of trg_reader ahead of time.")
    else:
      return len(trg_reader.vocab)


class AttentionalOutputMLP(OutputMLP, Serializable):
  """
  Multi-layer perceptron used as output layer in an attentional sequence-to-sequence model.

  This class allows setting the input dimension automatically to RNN dim + context dim, otherwise behaves identical
  to its base class.

  Args:
    input_dim: input dimension
    hidden_dim: hidden dimension
    output_dim: ignored
    param_init_hidden: how to initialize hidden weight matrices
    bias_init_hidden: how to initialize hidden bias vectors
    param_init_output: how to initialize output weight matrices
    bias_init_output: how to initialize output bias vectors
    activation: One of ``tanh``, ``relu``, ``sigmoid``, ``elu``, ``selu``, ``asinh`` or ``identity``.
    hidden_layer: hidden layer linear subcomponent (created automatically)
    output_projector: output layer linear subcomponent (created automatically)
    vocab_size: if not ``None`', this will be used as output dimension
    vocab: if not ``None`` and ``vocab_size`` was not specified, this will be used as output dimension
    trg_reader: Model's trg_reader, if exists and unambiguous; if not None and ``vocab_size``/``vocab`` were not
          specified, this will be used as output dimension
    decoder_rnn_dim (int): dimension of a decoder RNN that feeds into this MLP; this will be added to ``input_dim``
  """
  yaml_tag = '!AttentionalOutputMLP'

  @serializable_init
  def __init__(self,
               input_dim: int = Ref("exp_global.default_layer_dim"),
               hidden_dim: int = Ref("exp_global.default_layer_dim"),
               output_dim: int = Ref("exp_global.default_layer_dim", default=None),
               param_init_hidden: ParamInitializer = Ref("exp_global.param_init", default=bare(GlorotInitializer)),
               bias_init_hidden: ParamInitializer = Ref("exp_global.bias_init", default=bare(ZeroInitializer)),
               param_init_output: ParamInitializer = Ref("exp_global.param_init", default=bare(GlorotInitializer)),
               bias_init_output: ParamInitializer = Ref("exp_global.bias_init", default=bare(ZeroInitializer)),
               activation: str = 'tanh',
               hidden_layer=None,
               output_projector=None,
               vocab_size: Optional[int] = None,
               vocab: Optional[vocab.Vocab] = None,
               trg_reader: Optional[input_reader.InputReader] = Ref("model.trg_reader", default=None),
               decoder_rnn_dim: int = Ref("exp_global.default_layer_dim", default=0)):
    original_input_dim = input_dim
    input_dim += decoder_rnn_dim
    super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, param_init_hidden=param_init_hidden,
                     bias_init_hidden=bias_init_hidden, param_init_output=param_init_output,
                     bias_init_output=bias_init_output, activation=activation, hidden_layer=hidden_layer,
                     output_projector=output_projector, vocab_size=vocab_size, vocab=vocab, trg_reader=trg_reader)
    self.save_processed_arg("input_dim", original_input_dim)

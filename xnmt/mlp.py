import dynet as dy

from xnmt.serialize.serializable import Serializable
from xnmt.serialize.tree_tools import Ref, Path
import xnmt.linear

class MLP(Serializable):
  """
  Multi-layer perceptron.

  Args:
    input_dim (int): input dimension; if None, use ``exp_global.default_layer_dim``
    hidden_dim (int): hidden dimension; if None, use ``exp_global.default_layer_dim``
    output_dim (int): output dimension; if None, use ``exp_global.default_layer_dim``
    param_init_hidden (ParamInitializer): how to initialize hidden weight matrices; if None, use ``exp_global.param_init``
    bias_init_hidden (ParamInitializer): how to initialize hidden bias vectors; if None, use ``exp_global.bias_init``
    param_init_output (ParamInitializer): how to initialize output weight matrices; if None, use ``exp_global.param_init``
    bias_init_output (ParamInitializer): how to initialize output bias vectors; if None, use ``exp_global.bias_init``
    output_projector: TODO
    vocab_size (int): vocab size or None; only relevant if MLP is used as a decoder component
    vocab (Vocab): vocab or None; only relevant if MLP is used as a decoder component
    trg_reader (InputReader): Model's trg_reader, if exists and unambiguous; only relevant if MLP is used as a decoder component
  """
  yaml_tag = '!MLP'

  def __init__(self, exp_global=Ref(Path("exp_global")),
               input_dim=None, hidden_dim=None, output_dim=None,
               param_init_hidden=None, bias_init_hidden=None,
               param_init_output=None, bias_init_output=None,
               output_projector=None,
               vocab_size=None, vocab=None,
               trg_reader=Ref(path=Path("model.trg_reader"), required=False),
               yaml_path=None, decoder_rnn_dim=None):
    model = exp_global.dynet_param_collection.param_col
    self.input_dim = input_dim or exp_global.default_layer_dim
    self.hidden_dim = hidden_dim or exp_global.default_layer_dim
    self.output_dim = output_dim or exp_global.default_layer_dim
    if yaml_path is not None and "decoder" in yaml_path:
      decoder_rnn_dim = decoder_rnn_dim or exp_global.default_layer_dim
      self.input_dim += decoder_rnn_dim
      self.output_dim = self.choose_vocab_size(vocab_size, vocab, trg_reader)

    self.hidden = xnmt.linear.Linear(
      self.input_dim, self.hidden_dim, model,
      param_init=param_init_hidden or exp_global.param_init,
      bias_init=bias_init_hidden or exp_global.bias_init)
    self.output = output_projector or xnmt.linear.Linear(
      self.hidden_dim, self.output_dim, model,
      param_init=param_init_output or exp_global.param_init,
      bias_init=bias_init_hidden or exp_global.bias_init)

  def __call__(self, input_expr):
    return self.output(dy.tanh(self.hidden(input_expr)))

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

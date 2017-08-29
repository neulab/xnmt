import dynet as dy
from encoder_state import FinalEncoderState, PseudoState
import lstm

class PyramidalRNNBuilder(object):
  """
  Builder for pyramidal RNNs that delegates to regular RNNs and wires them together.
  See https://arxiv.org/abs/1508.01211

  Every layer (except the first) reduces sequence length by factor 2.

      builder = PyramidalRNNBuilder(4, 128, 100, model, VanillaLSTMBuilder)
      [o1,o2,o3] = builder.transduce([i1,i2,i3])
  """
  def __init__(self, num_layers, input_dim, hidden_dim, model,
               downsampling_method="concat", reduce_factor=2):
    """
    :param num_layers: depth of the PyramidalRNN
    :param input_dim: size of the inputs
    :param hidden_dim: size of the outputs (and intermediate layer representations)
    :param model
    :param rnn_builder_factory: RNNBuilder subclass, e.g. VanillaLSTMBuilder
    :param downsampling_method: how to perform downsampling (concat|skip)
    :param reduce_factor: integer, or list of ints (different skip for each layer)
    """
    assert num_layers > 0
    assert hidden_dim % 2 == 0
    assert type(reduce_factor)==int or (type(reduce_factor)==list and len(reduce_factor)==num_layers-1)
    assert downsampling_method in ["concat", "skip"]
    self.builder_layers = []
    self.downsampling_method = downsampling_method
    self.reduce_factor = reduce_factor
    self.input_dim = input_dim
    f = lstm.CustomCompactLSTMBuilder(1, input_dim, hidden_dim / 2, model)
    b = lstm.CustomCompactLSTMBuilder(1, input_dim, hidden_dim / 2, model)
    self.builder_layers.append((f, b))
    for _ in range(num_layers - 1):
      layer_input_dim = hidden_dim if downsampling_method=="skip" else hidden_dim*reduce_factor
      f = lstm.CustomCompactLSTMBuilder(1, layer_input_dim, hidden_dim / 2, model)
      b = lstm.CustomCompactLSTMBuilder(1, layer_input_dim, hidden_dim / 2, model)
      self.builder_layers.append((f, b))

  def get_final_states(self):
    return self._final_states

  def _reduce_factor_for_layer(self, layer_i):
    if layer_i >= len(self.builder_layers)-1:
      return 1
    elif type(self.reduce_factor)==int:
      return self.reduce_factor
    else:
      return self.reduce_factor[layer_i]

  def whoami(self): return "PyramidalRNNBuilder"

  def set_dropout(self, p):
    for (fb, bb) in self.builder_layers:
      fb.set_dropout(p)
      bb.set_dropout(p)
  def disable_dropout(self):
    for (fb, bb) in self.builder_layers:
      fb.disable_dropout()
      bb.disable_dropout()

  def transduce(self, es):
    """
    returns the list of output Expressions obtained by adding the given inputs
    to the current state, one by one, to both the forward and backward RNNs,
    and concatenating.

    :param es: an ExpressionSequence
    """

    es = list(es)
    zero_pad = None
    batch_size = es[0].dim()[1]

    for layer_i, (fb, bb) in enumerate(self.builder_layers):
      reduce_factor = self._reduce_factor_for_layer(layer_i)
      while self.downsampling_method=="concat" and len(es) % reduce_factor != 0:
        if zero_pad is None:
          zero_pad = dy.zeros(dim=self.input_dim, batch_size=batch_size)
        es.append(zero_pad)
      fs = fb.initial_state().transduce(es)
      bs = bb.initial_state().transduce(reversed(es))
      if layer_i < len(self.builder_layers) - 1:
        if self.downsampling_method=="skip":
#          es = [dy.concatenate([f, b]) for f, b in zip(fs[::reduce_factor], bs[::reduce_factor][::-1])]
          es = [[f, b] for f, b in zip(fs[::reduce_factor], bs[::reduce_factor][::-1])]
        elif self.downsampling_method=="concat":
          es_len = len(es)
          es = []
          for i in range(0, es_len, reduce_factor):
            concat_states = []
            for j in range(reduce_factor):
              concat_states.append(fs[i+j])
              concat_states.append(bs[len(es)-reduce_factor+j-i])
#            es.append(dy.concatenate(concat_states))
            es.append(concat_states)
        else:
          raise RuntimeError("unknown downsampling_method %s" % self.downsampling_method)
      else:
        # concat final outputs
        es = [dy.concatenate([f, b]) for f, b in zip(fs, reversed(bs))]
    
    self._final_states = [FinalEncoderState(dy.concatenate([fb.get_final_states()[0].main_expr(),
                                                            bb.get_final_states()[0].main_expr()]),
                                            dy.concatenate([fb.get_final_states()[0].cell_expr(),
                                                            bb.get_final_states()[0].cell_expr()])) \
                          for (fb, bb) in self.builder_layers]
    
    return es

  def initial_state(self):
    return PseudoState(self)

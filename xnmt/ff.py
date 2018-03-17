import dynet as dy

from xnmt.transducer import SeqTransducer, FinalTransducerState
from xnmt.serialize.serializable import Serializable, Ref, Path
from xnmt.expression_sequence import ExpressionSequence

class FullyConnectedSeqTransducer(SeqTransducer, Serializable):
  yaml_tag = '!FullyConnectedSeqTransducer'
  def __init__(self, in_height, out_height, nonlinearity='linear', exp_global=Ref(Path("exp_global"))):
    """
    Args:
      in_height: input dimension of the affine transform
      out_height: output dimension of the affine transform
      nonlinearity: nonlinear activation function
    """
    model = exp_global.dynet_param_collection.param_col
    self.in_height = in_height
    self.out_height = out_height
    self.nonlinearity = nonlinearity

    normalInit=dy.NormalInitializer(0, 0.1)
    self.pW = model.add_parameters(dim = (self.out_height, self.in_height), init=normalInit)
    self.pb = model.add_parameters(dim = self.out_height)

  def get_final_states(self):
    return self._final_states

  def transduce(self, embed_sent):
    src = embed_sent.as_tensor()

    W = dy.parameter(self.pW)
    b = dy.parameter(self.pb)

    l1 = dy.affine_transform([b, W, src])
    output = l1
    if self.nonlinearity is 'linear':
      output = l1
    elif  self.nonlinearity is 'sigmoid':
      output = dy.logistic(l1)
    elif self.nonlinearity is 'tanh':
      output = 2*dy.logistic(l1) - 1
    elif self.nonlinearity is 'relu':
      output = dy.rectify(l1)
    output_seq = ExpressionSequence(expr_tensor=output)
    self._final_states = [FinalTransducerState(output_seq[-1])]
    return output_seq


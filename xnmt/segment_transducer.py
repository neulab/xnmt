import dynet
import linear
import model_globals
import embedder
import numpy
from serializer import Serializable

class SegmentTransducer(object):
  def __init__(self, hidden_dim, category):
    model = model_globals.dynet_param_collection.param_col
    category_size, category_embed = category["size"], category["embed_dim"]
    self.category_output = linear.Linear(hidden_dim, category_size, model)
    self.category_embedder = embedder.SimpleWordEmbedder(category_size, category_embed)

  def transduce(self):
    raise RuntimeError("Should call subclass of SegmentTransducer instead!")

  def discrete_log_likelihood(self, loss):
    raise RuntimeError("Should call subclass of SegmentTransducer instead!")

class LSTMSegmentTransducer(SegmentTransducer, Serializable):
  yaml_tag = u"!LSTMSegmentTransducer"

  def __init__(self, input_dim=None, hidden_dim=None, layers=None, category=None):
    super(LSTMSegmentTransducer, self).__init__(hidden_dim, category)
    model = model_globals.dynet_param_collection.param_col
    self.rnn = dynet.VanillaLSTMBuilder(layers, input_dim, hidden_dim, model)

  def transduce(self, inputs):
    encoding = self.rnn.initial_state().transduce(inputs)[-1]
    category_logsoftmax = dynet.log_softmax(self.category_output(encoding))
    # TODO change it with dynet
    p_category = dynet.exp(category_logsoftmax).npvalue()
    p_category /= p_category.sum()
    category = numpy.random.choice(len(p_category), p=p_category)
    return self.category_embedder.embed(category), category, category_logsoftmax

def sample_from_log(log_softmax):
  # TODO Use the dynet version after it is fixed
#  sample = log_softmax.tensor_value().categorical_sample_log_prob().as_numpy().transpose()
#  if len(sample.shape) > 1:
#    sample = numpy.squeeze(sample, axis=1)
  prob = dynet.exp(log_softmax).npvalue().transpose()
  sample = []
  if len(prob.shape) == 2:
    for p in prob:
      p /= p.sum()
      choice = numpy.random.choice(len(p), p=p)
      sample.append(choice)
    sample = numpy.array(sample, dtype=int)
  elif len(prob.shape) == 1:
    prob /= prob.sum()
    choice = numpy.random.choice(len(prob), p=prob)
    sample.append(choice)
  else:
    raise ValueError("Unexpected prob with shape:", prob.shape, "expect up to 2 dimensions only.")
  return numpy.array(sample, dtype=int)

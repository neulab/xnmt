import dynet
import linear
import model_globals
import embedder
import numpy
from serializer import Serializable

class SegmentTransducer(object):
  def __init__(self, hidden_dim, category):
    assert(category, "Missing field: category")
    model = model_globals.dynet_param_collection.param_col
    category_size, category_embed = category["size"], category["embed_dim"]
    self.category_output = linear.Linear(hidden_dim, category_size, model)
    self.category_embedder = embedder.SimpleWordEmbedder(category_size, category_embed)

  def __call__(self):
    raise RuntimeError("Should call subclass of SegmentTransducer instead!")

class LSTMSegmentTransducer(SegmentTransducer, Serializable):
  yaml_tag = u"!LSTMSegmentTransducer"

  def __init__(self, input_dim=None, hidden_dim=None, layers=None, category=None):
    super(LSTMSegmentTransducer, self).__init__(hidden_dim, category)
    model = model_globals.dynet_param_collection.param_col
    self.rnn = dynet.VanillaLSTMBuilder(layers, input_dim, hidden_dim, model)

  def transduce(self, inputs):
    encoding = self.rnn.initial_state().transduce(inputs)[-1]
    category_softmax = dynet.softmax(self.category_output(encoding))
    category = numpy.argmax(category_softmax.npvalue())
    return self.category_embedder.embed(category)


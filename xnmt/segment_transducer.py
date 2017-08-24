import _dynet as dy 
import linear
import model_globals
import embedder
import numpy

from model import HierarchicalModel
from serializer import Serializable
from decorators import recursive, recursive_assign
from reports import Reportable

class SegmentTransducer(HierarchicalModel, Serializable, Reportable):
  yaml_tag = "!SegmentTransducer"

  def __init__(self, encoder, transformer):
    self.encoder = encoder
    self.transformer = transformer

    self.register_hier_child(encoder)
    self.register_hier_child(transformer)

  @recursive
  def set_input_size(self, batch_size, input_len):
    pass

  @recursive
  def next_item(self):
    pass

  @recursive_assign
  def html_report(self, context):
    return context

  def transduce(self, inputs):
    return self.transformer.transform(self.encoder.transduce(inputs))

  def disc_ll(self):
    ''' Discrete Log Likelihood '''
    log_ll = dy.scalarInput(0.0)
    if hasattr(self.encoder, "disc_ll"):
      log_ll += self.encoder.disc_ll()
    if hasattr(self.transformer, "disc_ll"):
      log_ll += self.transformer.disc_ll()
    return log_ll

class SegmentTransformer(HierarchicalModel, Serializable):
  def transform(self, encodings):
    raise RuntimeError("Should call subclass of SegmentTransformer instead")

class TailSegmentTransformer(SegmentTransformer):
  yaml_tag = u"!TailSegmentTransformer"
  def transform(self, encodings):
    return encodings[-1]

class AverageSegmentTransformer(SegmentTransformer):
  yaml_tag = u"!AverageSegmentTransformer"
  def transform(self, encodings):
    return dy.average(encodings.as_list())

# TODO(philip30): Complete this class!
# To test, modify the segment transformer in examples/test_segmenting.yaml into this class!
class DownsamplingSegmentTransformer(SegmentTransformer):
  yaml_tag = u"!DownsamplingSegmentTransformer"

  def __init__(self, sample_size=None):
    self.sample_size = sample_size
    # TODO(philip30): Add the dynet parameters here!

  def transduce(self, inputs):
    # TODO(philip30): Complete me
    pass

class CategorySegmentTransformer(SegmentTransformer):
  yaml_tag = u"!CategorySegmentTransformer"

  def __init__(self, input_dim=None, category_dim=None, embed_dim=None):
    model = model_globals.dynet_param_collection.param_col
    self.category_output = linear.Linear(input_dim, category_dim, model)
    self.category_embedder = embedder.SimpleWordEmbedder(category_dim, embed_dim)
    self.train = True

  @recursive
  def set_input_size(self, batch_size, input_len):
    self.batch_size = batch_size
    self.input_len  = input_len
    # Log likelihood buffer
    self.ll_buffer = [dy.scalarInput(0.0) for _ in range(batch_size)]
    self.counter = 0

  @recursive
  def set_train(self, train):
    self.train = train

  @recursive
  def next_item(self):
    self.counter = (self.counter + 1) % self.batch_size

  def transform(self, encodings):
    encoding = encodings[-1]
    category_logsoftmax = dy.log_softmax(self.category_output(encoding))
    if self.train:
      category = category_logsoftmax.tensor_value().categorical_sample_log_prob().as_numpy()[0]
    else:
      category = category_logsoftmax.tensor_value().argmax().as_numpy().transpose()
    # Accumulating the log likelihood for the batch
    self.ll_buffer[self.counter] += dy.pick(category_logsoftmax, category)

    return self.category_embedder.embed(category)

  def disc_ll(self):
    try:
      return dy.concatenate_to_batch(self.ll_buffer)
    finally:
      # Make sure that the state is not used again after the log likelihood is requested
      del self.ll_buffer
      del self.batch_size
      del self.counter


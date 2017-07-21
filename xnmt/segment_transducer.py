import dynet
import linear
import model_globals
import embedder
import numpy
from serializer import Serializable

class SegmentTransducer(Serializable):
  yaml_tag = "!SegmentTransducer"

  def __init__(self, encoder, transformer):
    self.encoder = encoder
    self.transformer = transformer

  def set_input_size(self, batch_size, input_len):
    self.invoke_method_to_components("set_input_size", batch_size, input_len)

  def next_item(self):
    self.invoke_method_to_components("next_item")

  def invoke_method_to_components(self, method, *args, **kwargs):
    for component in (self.encoder, self.transformer):
      if hasattr(component, method):
        getattr(component, method)(*args, **kwargs)

  def transduce(self, inputs):
    return self.transformer.transform(self.encoder.transduce(inputs))

  def disc_ll(self):
    ''' Discrete Log Likelihood '''
    log_ll = dynet.scalarInput(0.0)
    if hasattr(self.encoder, "disc_ll"):
      log_ll += self.encoder.disc_ll()
    if hasattr(self.transformer, "disc_ll"):
      log_ll += self.transformer.disc_ll()
    return log_ll

  def set_train(self, train):
    pass

class SegmentTransformer(Serializable):
  def __init__(self):
    pass

  def transform(self, encodings):
    raise RuntimeError("Should call subclass of SegmentTransformer instead")

class TailSegmentTransformer(SegmentTransformer):
  yaml_tag = u"!TailSegmentTransformer"
  def transform(self, encodings):
    return encodings[-1]

class AverageSegmentTransformer(SegmentTransformer):
  yaml_tag = u"!AverageSegmentTransformer"
  def transform(self, encodings):
    return dynet.average(encodings.as_list())

class CategorySegmentTransformer(SegmentTransformer):
  yaml_tag = u"!CategorySegmentTransformer"

  def __init__(self, input_dim=None, category_dim=None, embed_dim=None):
    model = model_globals.dynet_param_collection.param_col
    self.category_output = linear.Linear(input_dim, category_dim, model)
    self.category_embedder = embedder.SimpleWordEmbedder(category_dim, embed_dim)
    self.train = True

  def set_input_size(self, batch_size, input_len):
    self.batch_size = batch_size
    self.input_len  = input_len
    # Log likelihood buffer
    self.ll_buffer = [dynet.scalarInput(0.0) for _ in range(batch_size)]
    self.counter = 0

  def set_train(self, train):
    self.train = train

  def next_item(self):
    self.counter = (self.counter + 1) % self.batch_size

  def transform(self, encodings):
    encoding = encodings[-1]
    category_logsoftmax = dynet.log_softmax(self.category_output(encoding))
    # TODO change it with dynet
    if self.train:
      category = category_logsoftmax.tensor_value().categorical_sample_log_prob().as_numpy()[0]
    else:
      category = category_logsoftmax.tensor_value().argmax().as_numpy().transpose()
    # Accumulating the log likelihood for the batch
    self.ll_buffer[self.counter] += dynet.pick(category_logsoftmax, category)

    return self.category_embedder.embed(category)

  def disc_ll(self):
    try:
      return dynet.concatenate_to_batch(self.ll_buffer)
    finally:
      # Make sure that the state is not used again after the log likelihood is requested
      del self.ll_buffer
      del self.batch_size
      del self.counter


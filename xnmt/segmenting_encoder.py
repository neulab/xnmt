import dynet as dy
import segment_transducer
import linear
import six
import numpy
import enum

# TODO Fix the sample
from segment_transducer import sample_from_log

class SegmentingAction(enum.Enum):
  READ = 0
  SEGMENT = 1
  DELETE = 2

class SegmentingEncoderBuilder(object):

  def __init__(self, input_dim=None, embed_encoder=None, segment_transducer=None, model=None):
    # The Embed Encoder transduces the embedding vectors to a sequence of vector
    embed_encoder_param = [embed_encoder["layers"], input_dim, embed_encoder["hidden_dim"], model]
    if embed_encoder["bidirectional"]:
      embed_encoder_param.append(dy.VanillaLSTMBuilder)
      self.embed_encoder = dy.BiRNNBuilder(*embed_encoder_param)
    else:
      self.embed_encoder = dy.VanillaLSTMBuilder(*embed_encoder_param)

    # The Segment Encoder decides whether to segment or not
    self.segment_transform = linear.Linear(embed_encoder["hidden_dim"], len(SegmentingAction), model)

    # The Segment transducer predict a category based on the collected vector
    self.segment_transducer = segment_transducer

  def transduce(self, src):
    num_batch = src[0].dim()[1]
    # Softmax + segment decision
    encodings = self.embed_encoder.transduce(src)
    segment_logsoftmaxes = [dy.log_softmax(self.segment_transform(fb)) for fb in encodings]
    # Segment decision 
    segment_decisions = [sample_from_log(log_softmax) for log_softmax in segment_logsoftmaxes]
    # The last segment decision should be equal to 1
    if len(segment_decisions) > 0:
      segment_decisions[-1] = numpy.ones(num_batch, dtype=int)
    # Buffer for output
    buffers = [[] for _ in range(num_batch)]
    outputs = [[] for _ in range(num_batch)]
    categories = [[] for _ in range(num_batch)]
    categories_logsoftmaxes = [[] for _ in range(num_batch)]
    # Loop through all the frames (word / item) in input.
    for j, (encoding, segment_decision) in enumerate(six.moves.zip(encodings, segment_decisions)):
      # For each decision in the batch
      for i, decision in enumerate(segment_decision):
        # Get the particular encoding for that batch item
        encoding_i = dy.pick_batch_elem(encoding, i)
        # Append the encoding for this item to the buffer
        buffers[i].append(encoding_i)
        # If segment for this particular input
        decision = int(decision)
        if decision == SegmentingAction.SEGMENT.value:
          category_embed, category, category_logsoftmax = self.segment_transducer.transduce(buffers[i])
          outputs[i].append(category_embed)
          categories[i].append(category)
          categories_logsoftmaxes[i].append(category_logsoftmax)
          buffers[i].clear()
        elif decision == SegmentingAction.DELETE.value:
          buffers[i].clear()

    # Pooling + creating a batch of them
    outputs = dy.concatenate_to_batch(list(six.moves.map(lambda xs: dy.average(xs), outputs)))
    # Retain some information of this passes
    self.segment_decisions = segment_decisions
    self.segment_logsoftmaxes = segment_logsoftmaxes
    self.categories = categories
    self.categories_logsoftmaxes = categories_logsoftmaxes
    # Return the encoded batch by the size of ((encode,), batch)
    return outputs

  def set_train(self, val):
    pass

  def receive_decoder_loss(self, loss):
    segment_logprob = None
    for log_softmax, segment_decision in six.moves.zip(self.segment_logsoftmaxes, self.segment_decisions):
      softmax_decision = dy.pick_batch(log_softmax, segment_decision)
      if not segment_logprob:
        segment_logprob = softmax_decision
      else:
        segment_logprob += softmax_decision

    category_logprob = []
    for category, category_log_softmax in six.moves.zip(self.categories, self.categories_logsoftmaxes):
      category_log_softmax = dy.concatenate_to_batch(category_log_softmax)
      category_logprob.append(dy.sum_batches(dy.pick_batch(category_log_softmax, category)))
    category_logprob = dy.concatenate_to_batch(category_logprob)

    return (segment_logprob + category_logprob) * (-loss)


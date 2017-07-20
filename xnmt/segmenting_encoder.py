import dynet as dy
import segment_transducer
import linear
import six
import numpy
import enum

class SegmentingAction(enum.Enum):
  READ = 0
  SEGMENT = 1
  DELETE = 2

class SegmentingEncoderBuilder(object):

  def __init__(self, embed_encoder=None, segment_transducer=None, model=None):
    # The Embed Encoder transduces the embedding vectors to a sequence of vector
    self.embed_encoder = embed_encoder

    # The Segment Encoder decides whether to segment or not
    self.segment_transform = linear.Linear(embed_encoder.hidden_dim, 3, model)

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
    self.segment_transducer.set_input_size(num_batch, len(encodings))

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
          transduce_output = self.segment_transducer.transduce(buffers[i])
          outputs[i].append(transduce_output)
          buffers[i].clear()
        elif decision == SegmentingAction.DELETE.value:
          buffers[i].clear()

        self.segment_transducer.next_item()

    # Pooling + creating a batch of them
    outputs = dy.concatenate_to_batch(list(six.moves.map(lambda xs: dy.average(xs), outputs)))
    # Retain some information of this passes
    self.segment_decisions = segment_decisions
    self.segment_logsoftmaxes = segment_logsoftmaxes
    # Return the encoded batch by the size of ((encode,), batch)
    return outputs

  def set_train(self, val):
    pass

  def calc_reinforce_loss(self, reward, lmbd):
    segment_logprob = None
    for log_softmax, segment_decision in six.moves.zip(self.segment_logsoftmaxes, self.segment_decisions):
      softmax_decision = dy.pick_batch(log_softmax, segment_decision)
      if not segment_logprob:
        segment_logprob = softmax_decision
      else:
        segment_logprob += softmax_decision

    # Segment Decision for the first item in the minibatch
    #print([x[0] for x in self.segment_decisions])

    return (segment_logprob + self.segment_transducer.disc_ll()) * reward * lmbd

def sample_from_log(log_softmax):
  # TODO Use the dynet version after it is fixed
#  sample = log_softmax.tensor_value().categorical_sample_log_prob().as_numpy().transpose()
#  if len(sample.shape) > 1:
#    sample = numpy.squeeze(sample, axis=1)
  prob = dy.exp(log_softmax).npvalue().transpose()
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

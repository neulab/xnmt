import dynet as dy
import segment_transducer
import linear
import six
import numpy

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
    self.segment_softmax = linear.Linear(embed_encoder["hidden_dim"], 2, model)

    # The Segment transducer predict a category based on the collected vector
    self.segment_transducer = segment_transducer

  def transduce(self, src):
    num_batch = src[0].dim()[1] if len(src) > 0 else 1
    # Softmax + segment decision
    encodings = self.embed_encoder.transduce(src)
    log_softmaxes = [dy.log_softmax(self.segment_softmax(fb)) for fb in encodings]
    # Segment decision
    def sample(log_softmax):
      sample = log_softmax.tensor_value().categorical_sample_log_prob().as_numpy().transpose()
      if len(sample.shape) > 1:
        sample = numpy.squeeze(sample, axis=1)
      return sample
    segment_decisions = [sample(log_softmax) for log_softmax in log_softmaxes]
    # The last segment decision should be equal to 1
    if len(segment_decisions) > 0:
      segment_decisions[-1] = numpy.ones(num_batch, dtype=int)
    # Buffer for output
    buffers = [[] for _ in range(num_batch)]
    outputs = [[] for _ in range(num_batch)]
    # Loop through all the frames (word / item) in input.
    for j, (encoding, segment_decision) in enumerate(six.moves.zip(encodings, segment_decisions)):
      # For each decision in the batch
      for i, decision in enumerate(segment_decision):
        # Get the particular encoding for that batch item
        encoding_i = dy.pick_batch_elem(encoding, i)
        # Append the encoding for this item to the buffer
        buffers[i].append(encoding_i)
        # If segment for this particular input
        if int(decision):
          outputs[i].append(self.segment_transducer.transduce(buffers[i]))
          buffers[i].clear()

    # Pooling + creating a batch of them
    outputs = dy.concatenate_to_batch(list(map(lambda xs: dy.emax(xs), outputs)))
    # Retain some information of this passes
    self.segment_decisions = segment_decisions
    self.log_softmaxes = log_softmaxes
    # Return the encoded batch by the size of ((encode,), batch)
    return outputs

  def set_train(self, val):
    pass

  def receive_decoder_loss(self, loss):
    segmentation_loss = []
    assert(len(self.log_softmaxes) == len(self.segment_decisions))
    for log_softmax, segment_decision in zip(self.log_softmaxes, self.segment_decisions):
      segmentation_loss.append(dy.sum_batches(-dy.pick_batch(log_softmax, segment_decision)))
    return loss + dy.average(segmentation_loss)


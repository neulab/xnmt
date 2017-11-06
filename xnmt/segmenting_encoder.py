from __future__ import print_function

import io
import six
import numpy
import dynet as dy

from enum import Enum
from xml.sax.saxutils import escape, unescape
from lxml import etree
from scipy.stats import poisson

import xnmt.linear as linear
import xnmt.expression_sequence as expression_sequence

from xnmt.events import register_handler, handle_xnmt_event
from xnmt.reports import Reportable
from xnmt.serializer import Serializable
from xnmt.transducer import SeqTransducer, FinalTransducerState
from xnmt.loss import LossBuilder

class SegmentingAction(Enum):
  """
  The enumeration of possible action.
  """
  READ = 0
  SEGMENT = 1
  DELETE = 2

class ScalarParam(Serializable):
  yaml_tag = u'!ScalarParam'

  def __init__(self, initial=0.1, warmup=0, grow=1, min_value=0.0, max_value=1.0):
    register_handler(self)
    self.value = initial
    self.warmup = warmup
    self.grow = grow
    self.min_value = min_value
    self.max_value = max_value

  def get_value(self, warmup_counter=None):
    if warmup_counter is None or warmup_counter > self.warmup:
      return self.value
    else:
      return 0.0

  def grow_param(self, warmup_counter=None):
    if warmup_counter is None or warmup_counter > self.warmup:
      self.value *= self.grow
      self.value = max(self.min_value, self.value)
      self.value = min(self.max_value, self.value)

  def __repr__(self):
    return str(self.value)

class SegmentingSeqTransducer(SeqTransducer, Serializable, Reportable):
  yaml_tag = u'!SegmentingSeqTransducer'

  def __init__(self, yaml_context, embed_encoder=None, segment_transducer=None, learn_segmentation=True,
               reinforcement_param=None, length_prior=3.5, learn_delete=False,
               length_prior_alpha=1.0, use_baseline=True, segmentation_warmup_counter=None):
    register_handler(self)
    model = yaml_context.dynet_param_collection.param_col
    # The Embed Encoder transduces the embedding vectors to a sequence of vector
    self.embed_encoder = embed_encoder
    # The Segment transducer predict a category based on the collected vector
    self.segment_transducer = segment_transducer
    # The Segment Encoder decides whether to segment or not
    self.segment_transform = linear.Linear(input_dim  = embed_encoder.hidden_dim,
                                           output_dim = 3 if learn_delete else 2,
                                           model=model)
    # The baseline linear regression model
    self.baseline = linear.Linear(input_dim = embed_encoder.hidden_dim,
                                  output_dim = 1,
                                  model = model)
    self.use_baseline = use_baseline
    # Whether to learn segmentation or not
    self.learn_segmentation = learn_segmentation
    # Whether to learn deletion or not
    self.learn_delete = learn_delete
    # Other Parameters
    self.length_prior = length_prior
    self.length_prior_alpha = length_prior_alpha
    self.lmbd = reinforcement_param

    # States of the object
    self.train = True
    self.warmup_counter = 0
    self.segmentation_warmup_counter = segmentation_warmup_counter
    # Register all the children 

  @handle_xnmt_event
  def on_start_sent(self, src=None):
    self._src = src

  def sample_segmentation(self, encodings, batch_size, src=None):
    lmbd = self.lmbd.get_value(self.warmup_counter)
    segment_logsoftmaxes = None
    if src is not None and \
       (self.segmentation_warmup_counter is None or self.warmup_counter >= self.segmentation_warmup_counter): # Indicates that prior segmentation is given
      segment_decisions = numpy.zeros((batch_size, len(encodings)), dtype=int)
      for i, sent in enumerate(src):
        if "segment" not in sent.annotation:
          raise ValueError("If segmentation is not learned, SegmentationTextReader should be used to read in the input.")
        segment_decisions[i][sent.annotation["segment"]] = 1
      segment_decisions = numpy.split(segment_decisions, len(encodings), 1)
    elif lmbd == 0: # Indicates that it is still warmup time
      randoms = numpy.random.poisson(lam=self.length_prior, size=batch_size * len(encodings))
      segment_decisions = numpy.zeros((batch_size, len(encodings)), dtype=int)
      idx = 0
      # Filling up the segmentation matrix based on the poisson distribution
      for decision in segment_decisions:
        current = randoms[idx]
        while current < len(decision):
          decision[current] = 1
          idx = (idx + 1) % len(randoms)
          current += randoms[idx]
        decision[-1] = 1
      segment_decisions = numpy.split(segment_decisions, len(encodings), 1)
    else:
      segment_logsoftmaxes = [dy.log_softmax(self.segment_transform(fb)) for fb in encodings]
      if self.train:
        # Sample from the softmax
        segment_decisions = [log_softmax.tensor_value().categorical_sample_log_prob().as_numpy()[0]
                             for log_softmax in segment_logsoftmaxes]
        if batch_size == 1:
          segment_decisions = list(six.moves.map(lambda x: numpy.array([x]), segment_decisions))
      else:
        segment_decisions = [log_softmax.tensor_value().argmax().as_numpy().transpose()
                             for log_softmax in segment_logsoftmaxes]
    return segment_decisions, segment_logsoftmaxes

  def __call__(self, embed_sent):
    batch_size = embed_sent[0].dim()[1]
    # Softmax + segment decision
    encodings = self.embed_encoder(embed_sent)
    if self.learn_segmentation:
      segment_decisions, segment_logsoftmaxes = self.sample_segmentation(encodings, batch_size)
    else:
      segment_decisions, segment_logsoftmaxes = self.sample_segmentation(encodings, batch_size, self._src)
    # Some checks
    assert len(encodings) == len(segment_decisions), \
           "Encoding={}, segment={}".format(len(encodings), len(segment_decisions))
    # The last segment decision should be equal to 1
    if len(segment_decisions) > 0:
      segment_decisions[-1] = numpy.ones(segment_decisions[-1].shape, dtype=int)
    # Buffer for output
    buffers = [[] for _ in range(batch_size)]
    outputs = [[] for _ in range(batch_size)]
    last_segment = [-1 for _ in range(batch_size)]
    length_prior = [0 for _ in range(batch_size)]
    length_div = [0 for _ in range(batch_size)]
    self.segment_transducer.set_input_size(batch_size, len(encodings))
    # Loop through all the frames (word / item) in input.
    for j, (encoding, segment_decision) in enumerate(six.moves.zip(encodings, segment_decisions)):
      # For each decision in the batch
      for i, decision in enumerate(segment_decision):
        # If segment for this particular input
        decision = int(decision)
        if decision == SegmentingAction.DELETE.value:
          continue
        # Get the particular encoding for that batch item
        encoding_i = dy.pick_batch_elem(encoding, i)
        # Append the encoding for this item to the buffer
        buffers[i].append(encoding_i)
        if decision == SegmentingAction.SEGMENT.value:
          expr_seq = expression_sequence.ExpressionSequence(expr_list=buffers[i])
          transduce_output = self.segment_transducer.transduce(expr_seq)
          outputs[i].append(transduce_output)
          buffers[i] = []
          # Calculate length prior
          length_prior[i] += numpy.log(poisson.pmf(j - last_segment[i], self.length_prior) + 1e-10)
          length_div[i] += 1
          last_segment[i] = j
        self.segment_transducer.next_item()
    length_prior = list(six.moves.map(lambda i: length_prior[i] / length_div[i],
                                      range(len(length_prior))))
    # Padding
    max_col = max(len(xs) for xs in outputs)
    P0 = dy.vecInput(self.segment_transducer.encoder.hidden_dim)
    def pad(xs):
      deficit = max_col - len(xs)
      if deficit > 0:
        xs.extend([P0 for _ in range(deficit)])
      return xs
    outputs = dy.concatenate_to_batch(list(six.moves.map(lambda xs: dy.concatenate_cols(pad(xs)), outputs)))
    self.segment_decisions = segment_decisions
    self.segment_logsoftmaxes = segment_logsoftmaxes
    # Packing output together
    if self.train and self.learn_segmentation:
      self.segment_length_prior = dy.inputTensor(length_prior, batched=True)
      if self.use_baseline:
        self.bs = list(six.moves.map(lambda x: self.baseline(dy.nobackprop(x)), encodings))
    if not self.train:
      self.set_report_input(segment_decisions)
    self._final_encoder_state = [FinalTransducerState(encodings[-1])]
    # Return the encoded batch by the size of [(encode,segment)] * batch_size
    return expression_sequence.ExpressionSequence(expr_tensor=outputs)

  @handle_xnmt_event
  def on_set_train(self, train):
    self.train = train

  def get_final_states(self):
    return self._final_encoder_state

  # TODO: handle as global event?
  def new_epoch(self):
    self.lmbd.grow_param(self.warmup_counter)
    self.warmup_counter += 1
    lmbd = self.lmbd.get_value(self.warmup_counter)
    if lmbd > 0.0:
      print("Now Lambda:", lmbd)

  @handle_xnmt_event
  def on_calc_additional_loss(self, reward):
    if not self.learn_segmentation:
      return None
    ret = LossBuilder()
    if self.length_prior_alpha > 0:
      reward += self.segment_length_prior * self.length_prior_alpha
    reward = dy.cdiv(reward - dy.mean_batches(reward), dy.std_batches(reward))
    # Baseline Loss
    if self.use_baseline:
      baseline_loss = []
      for i, baseline in enumerate(self.bs):
        baseline_loss.append(dy.squared_distance(reward, baseline))
      ret.add_loss("Baseline", dy.esum(baseline_loss))
    # Reinforce Loss
    lmbd = self.lmbd.get_value(self.warmup_counter)
    if lmbd > 0.0:
      reinforce_loss = []
      # Calculating the loss of the baseline and reinforce
      for i in range(len(self.segment_decisions)):
        ll = dy.pick_batch(self.segment_logsoftmaxes[i], self.segment_decisions[i])
        if self.use_baseline:
          r_i = reward - self.bs[i]
        else:
          r_i = reward
        reinforce_loss.append(dy.logistic(r_i) * ll)
      ret.add_loss("Reinforce", -dy.esum(reinforce_loss) * lmbd)
    # Total Loss
    return ret

  @handle_xnmt_event
  def on_html_report(self, context):
    segment_decision = self.get_report_input()[0]
    segment_decision = list(six.moves.map(lambda x: int(x[0]), segment_decision))
    src_words = list(six.moves.map(escape, self.get_report_resource("src_words")))
    main_content = context.xpath("//body/div[@name='main_content']")[0]
    # construct the sub element from string
    segmented = self.apply_segmentation(src_words, segment_decision)
    segmented = [(x if not delete else ("<font color='red'><del>" + x + "</del></font>")) for x, delete in segmented]
    if len(segmented) > 0:
      segment_html = "<p>Segmentation: " + ", ".join(segmented) + "</p>"
      main_content.insert(2, etree.fromstring(segment_html))

    return context

  @handle_xnmt_event
  def on_file_report(self):
    segment_decision = self.get_report_input()[0]
    segment_decision = list(six.moves.map(lambda x: int(x[0]), segment_decision))
    src_words = self.get_report_resource("src_words")
    segmented = self.apply_segmentation(src_words, segment_decision)
    segmented = [x for x, delete in segmented]
    if len(segmented) > 0:
      with io.open(self.get_report_path() + ".segment", encoding='utf-8', mode='w') as segmentation_file:
        print(" ".join(segmented), file=segmentation_file)

  def apply_segmentation(self, words, segmentation):
    assert(len(words) == len(segmentation))
    segmented = []
    temp = ""
    for decision, word in zip(segmentation, words):
      if decision == SegmentingAction.READ.value:
        temp += word
      elif decision == SegmentingAction.SEGMENT.value:
        temp += word
        segmented.append((temp, False))
        temp = ""
      else: # Case: DELETE
        if temp: segmented.append((temp, False))
        segmented.append((word, True))
        temp = ""
    if temp: segmented.append((temp, False))
    return segmented


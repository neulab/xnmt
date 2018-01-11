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
from xnmt.segment_transducer import TailWordSegmentTransformer

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

  def __init__(self, yaml_context, embed_encoder=None, segment_transducer=None, segment_encoder=None,
               learn_segmentation=True, reinforcement_param=None, length_prior=3.5, learn_delete=False,
               length_prior_alpha=1.0, use_baseline=True, segmentation_warmup_counter=None,
               epsilon_greedy_param=None, debug=False):
    '''
    reinforcement_param: the value of lambda in: \lambda * reinforce_loss
    epsilon_greedy_param: param for structural dropout. 30% means 70% sample from poisson and 30% from softmax
    '''
    register_handler(self)
    assert embed_encoder is not None
    assert segment_transducer is not None
    assert segment_encoder is not None
    model = yaml_context.dynet_param_collection.param_col
    # The Embed Encoder transduces the embedding vectors to a sequence of vector
    self.embed_encoder = embed_encoder
    # The Segment transducer predict a category based on the collected vector
    self.segment_transducer = segment_transducer
    # The segment Encoder will encode the whole segment
    self.segment_encoder = segment_encoder
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
    self.eps = epsilon_greedy_param
    self.encoder_hidden_dim = segment_transducer.encoder.hidden_dim
    self.debug = debug

    # States of the object
    self.train = False
    self.warmup_counter = 0
    self.segmentation_warmup_counter = segmentation_warmup_counter

  @handle_xnmt_event
  def on_start_sent(self, src=None):
    self._src = src

  # Sample from prior segmentation
  def sample_from_prior(self, encodings, batch_size, src):
    self.print_debug("sample_from_prior")
    segment_decisions = numpy.zeros((batch_size, len(encodings)), dtype=int)
    for i, sent in enumerate(src):
      if "segment" not in sent.annotation:
        raise ValueError("If segmentation is not learned, SegmentationTextReader should be used to read in the input.")
      segment_decisions[i][sent.annotation["segment"]] = 1
    return numpy.split(segment_decisions, len(encodings), 1)

  # Sample from poisson prior
  def sample_from_poisson(self, encodings, batch_size):
    self.print_debug("samplef_from_poisson")
    randoms = numpy.random.poisson(lam=self.length_prior, size=batch_size*len(encodings))
    segment_decisions = numpy.zeros((batch_size, len(encodings)), dtype=int)
    idx = 0
    # Filling up the segmentation matrix based on the poisson distribution
    for decision in segment_decisions:
      current = randoms[idx]
      while current < len(decision):
        decision[current] = 1
        idx += 1
        current += randoms[idx]
      decision[-1] = 1
    return numpy.split(segment_decisions, len(encodings), 1)

  # Sample from the softmax
  def sample_from_softmax(self, encodings, batch_size, segment_logsoftmaxes):
    # Sample from the softmax
    if self.train:
      self.print_debug("sample_from_softmax")
      segment_decisions = [log_softmax.tensor_value().categorical_sample_log_prob().as_numpy()[0]
                           for log_softmax in segment_logsoftmaxes]
      if batch_size == 1:
        segment_decisions = [numpy.array([x]) for x in segment_decisions]
    else:
      self.print_debug("argmax(softmax)")
      segment_decisions = [log_softmax.tensor_value().argmax().as_numpy().transpose()
                           for log_softmax in segment_logsoftmaxes]
    return segment_decisions

  # Indicates warmup time. So we shouldn't sample from softmax
  def is_segmentation_warmup(self):
    return self.segmentation_warmup_counter is not None and self.warmup_counter <= self.segmentation_warmup_counter

  def sample_segmentation(self, encodings, batch_size, src=None):
    lmbd = self.lmbd.get_value(self.warmup_counter)
    eps = self.eps.get_value(self.warmup_counter) if self.eps is not None else None
    segment_logsoftmaxes = [dy.log_softmax(self.segment_transform(fb)) for fb in encodings]
    # Flags
    is_presegment_provided = src is not None and len(src) != 0 and hasattr(src[0], "annotation")
    is_warmup = lmbd == 0 or self.is_segmentation_warmup()
    is_epsgreedy_triggered = eps is not None and numpy.random.random() > eps
    # Sample based on the criterion
    if self.learn_segmentation and not is_warmup and not self.train:
      # During testing always sample from softmax if it is not warmup
      segment_decisions = self.sample_from_softmax(encodings, batch_size, segment_logsoftmaxes)
    elif is_presegment_provided:
      segment_decisions = self.sample_from_prior(encodings, batch_size, src)
    elif is_warmup or is_epsgreedy_triggered:
      segment_decisions = self.sample_from_poisson(encodings, batch_size)
    else:
      segment_decisions = self.sample_from_softmax(encodings, batch_size, segment_logsoftmaxes)

    return segment_decisions, segment_logsoftmaxes

  def __call__(self, embed_sent):
    batch_size = embed_sent[0].dim()[1]
    # Softmax + segment decision
    encodings = self.embed_encoder(embed_sent)
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
    length_prior = [[] for _ in range(batch_size)]
    self.segment_transducer.set_input_size(batch_size, len(encodings))
    # Loop through all the frames (word / item) in input.
    for j, (embed, segment_decision) in enumerate(zip(embed_sent, segment_decisions)):
      # For each decision in the batch
      for i, decision in enumerate(segment_decision):
        # If segment for this particular input
        decision = int(decision)
        if decision == SegmentingAction.DELETE.value:
          continue
        # Get the particular encoding for that batch item
        embed_i = dy.pick_batch_elem(embed, i)
        # Append the encoding for this item to the buffer
        buffers[i].append(embed_i)
        if decision == SegmentingAction.SEGMENT.value:
          # Special case for TailWordSegmentTransformer only
          words = None
          if type(self.segment_transducer.transformer) == TailWordSegmentTransformer:
            words = self._src[i].words[last_segment[i]+1:j+1]
          # Reducing the [expression] -> expression
          expr_seq = expression_sequence.ExpressionSequence(expr_list=buffers[i])
          transduce_output = self.segment_transducer.transduce(expr_seq, words)
          outputs[i].append(transduce_output)
          buffers[i] = []
          # Calculate length prior
          length_prior[i].append(numpy.log(poisson.pmf(j-last_segment[i], self.length_prior) + 1e-10))
          last_segment[i] = j
        # Notify the segment transducer to process the next decision
        self.segment_transducer.next_item()
    # Calculate the actual length prior length
    length_prior = [numpy.sum(len_prior) for len_prior in length_prior]
    # Padding
    max_col = max(len(xs) for xs in outputs)
    P0 = dy.vecInput(self.encoder_hidden_dim)
    def pad(xs):
      deficit = max_col - len(xs)
      if deficit > 0:
        xs.extend([P0 for _ in range(deficit)])
      return xs
    outputs = dy.concatenate_to_batch([dy.concatenate_cols(pad(xs)) for xs in outputs])
    self.segment_decisions = segment_decisions
    self.segment_logsoftmaxes = segment_logsoftmaxes
    # Packing output together
    if self.train:
      if self.learn_segmentation:
        self.segment_length_prior = dy.inputTensor(length_prior, batched=True)
        if self.use_baseline:
          self.bs = [self.baseline(dy.nobackprop(enc)) for enc in encodings]
    else:
      # Rewrite segmentation
      self.set_report_resource("segmentation", self.segment_decisions)
      self.set_report_input(segment_decisions)

    # Return the encoded batch by the size of [(encode,segment)] * batch_size
    return self.segment_encoder(expression_sequence.ExpressionSequence(expr_tensor=outputs))

  @handle_xnmt_event
  def on_set_train(self, train):
    self.train = train

  def get_final_states(self):
    return self.segment_encoder.get_final_states()

  @handle_xnmt_event
  def on_new_epoch(self):
    self.lmbd.grow_param(self.warmup_counter)
    if self.eps is not None:
      self.eps.grow_param(self.warmup_counter)
    self.warmup_counter += 1
    lmbd = self.lmbd.get_value(self.warmup_counter)
    if lmbd > 0.0:
      print("Now Lambda:", lmbd)
    if self.eps is not None:
      print("Now epsilon greedy:", self.eps.get_value(self.warmup_counter))

  @handle_xnmt_event
  def on_calc_additional_loss(self, reward):
    if not self.learn_segmentation:
      return None
    ret = LossBuilder()
    if self.length_prior_alpha > 0:
      reward += self.segment_length_prior * self.length_prior_alpha
    # reward z-score normalization
    reward = dy.cdiv(reward-dy.mean_batches(reward), dy.std_batches(reward) + 1e-10)
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
    segment_decision = [int(x[0]) for x in segment_decision]
    src_words = [escape(x) for x in self.get_report_resource("src_words")]
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
    segment_decision = [int(x[0]) for x in segment_decision]
    src_words = self.get_report_resource("src_words")
    segmented = self.apply_segmentation(src_words, segment_decision)
    segmented = [x for x, delete in segmented]

    with io.open(self.get_report_path() + ".segment", encoding='utf-8', mode='w') as segmentation_file:
      if len(segmented) > 0:
        print(" ".join(segmented), file=segmentation_file)

    with io.open(self.get_report_path() + ".segdecision", encoding='utf-8', mode='w') as segmentation_file:
      for softmax in self.segment_logsoftmaxes:
        print(" ".join(["%.5f" % f for f in dy.exp(softmax).npvalue()]), file=segmentation_file)

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

  #### DEBUG
  def print_debug(self, *args, **kwargs):
    if self.debug:
      print(*args, **kwargs)

  def print_debug_once(self, *args, **kwargs):
    if not hasattr(self, "_debug_lock"):
      self._debug_lock = True
      self.print_debug(*args, **kwargs)

  def print_debug_unlock(self):
    if hasattr(self, "_debug_lock"):
      delattr(self, "_debug_lock")


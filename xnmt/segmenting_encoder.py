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

from xnmt.batcher import Mask
from xnmt.events import register_handler, handle_xnmt_event
from xnmt.reports import Reportable
from xnmt.serializer import Serializable
from xnmt.transducer import SeqTransducer, FinalTransducerState
from xnmt.loss import LossBuilder
from xnmt.segmenting_composer import TailWordSegmentTransformer, WordOnlySegmentTransformer
from xnmt.parameters import GeometricSequence

EPS = 1e-10

class SegmentingSeqTransducer(SeqTransducer, Serializable, Reportable):
  yaml_tag = u'!SegmentingSeqTransducer'

  def __init__(self, yaml_context,
               ## COMPONENTS
               embed_encoder=None, segment_composer=None, final_transducer=None,
               ## OPTIONS
               length_prior=3.3,
               length_prior_alpha=None, # GeometricSequence
               epsilon_greedy=None,     # GeometricSequence
               reinforce_scale=None,    # GeometricSequence
               confidence_penalty=None, # SegmentationConfidencePenalty
               # For segmentation warmup (Always use the poisson prior)
               segmentation_warmup=0,
               segmentation_warmup_counter=0,
               ## FLAGS
               learn_delete       = False,
               use_baseline       = True,
               z_normalization    = True,
               learn_segmentation = True,
               debug=False):
    register_handler(self)
    model = yaml_context.dynet_param_collection.param_col
    # Sanity check
    assert embed_encoder is not None
    assert segment_composer is not None
    assert final_transducer is not None
    # The Embed Encoder transduces the embedding vectors to a sequence of vector
    self.embed_encoder = embed_encoder
    if not hasattr(embed_encoder, "hidden_dim"):
      embed_encoder_dim = yaml_context.default_layer_dim
    else:
      embed_encoder_dim = embed_encoder.hidden_dim
    # The Segment transducer produced word embeddings based on sequence of character embeddings
    self.segment_composer = segment_composer
    # The final transducer
    self.final_transducer = final_transducer
    # Decision layer of segmentation
    self.segment_transform = linear.Linear(input_dim  = embed_encoder_dim,
                                           output_dim = 3 if learn_delete else 2,
                                           model=model)
    # The baseline linear regression model
    self.baseline = linear.Linear(input_dim = embed_encoder_dim,
                                  output_dim = 1,
                                  model = model)
    # Flags
    self.use_baseline = use_baseline
    self.learn_segmentation = learn_segmentation
    self.learn_delete = learn_delete
    self.z_normalization = z_normalization
    self.debug = debug
    # Fixed Parameters
    self.length_prior = length_prior
    self.segmentation_warmup = segmentation_warmup
    # Variable Parameters
    self.length_prior_alpha = length_prior_alpha
    self.lmbd = reinforce_scale
    self.eps = epsilon_greedy
    self.confidence_penalty = confidence_penalty
    # States of the object
    self.train = False
    self.segmentation_warmup_counter = segmentation_warmup_counter

  def __call__(self, embed_sent):
    batch_size = embed_sent[0].dim()[1]
    # Softmax + segment decision
    encodings = self.embed_encoder(embed_sent)
    mask = encodings.mask
    segment_decisions, segment_logsoftmaxes = self.sample_segmentation(encodings, batch_size)
    # Some checks
    assert len(encodings) == len(segment_decisions), \
           "Encoding={}, segment={}".format(len(encodings), len(segment_decisions))
    # Buffer for output
    buffers = [[] for _ in range(batch_size)]
    outputs = [[] for _ in range(batch_size)]
    last_segment = [-1 for _ in range(batch_size)]
    length_prior = [[] for _ in range(batch_size)]
    self.segment_composer.set_input_size(batch_size, len(encodings))
    # Loop through all the frames (word / item) in input.
    for j, (encoding, segment_decision) in enumerate(zip(encodings, segment_decisions)):
      # For each decision in the batch
      for i, decision in enumerate(segment_decision):
        # If segment for this particular input
        decision = int(decision)
        if decision == SegmentingAction.DELETE.value or \
           (mask is not None and mask.np_arr[i][j] == 1):
          continue
        # Get the particular encoding for that batch item
        enc_i = dy.pick_batch_elem(encoding, i)
        # Append the encoding for this item to the buffer
        buffers[i].append(enc_i)
        if decision == SegmentingAction.SEGMENT.value:
          # Special case for TailWordSegmentTransformer only
          words = None
          if type(self.segment_composer.transformer) == TailWordSegmentTransformer or\
             type(self.segment_composer.transformer) == WordOnlySegmentTransformer:
            words = self.src_sent[i].words[last_segment[i]+1:j+1]
          # Reducing the [expression] -> expression
          expr_seq = expression_sequence.ExpressionSequence(expr_list=buffers[i])
          transduce_output = self.segment_composer.transduce(expr_seq, words)
          outputs[i].append(transduce_output)
          buffers[i] = []
          # Calculate length prior
          length_prior[i].append(numpy.log(poisson.pmf(j-last_segment[i], self.length_prior) + EPS))
          last_segment[i] = j
        # Notify the segment transducer to process the next decision
        self.segment_composer.next_item()
    # Calculate the actual length prior length
    length_prior = [sum(len_prior) / len(len_prior) for len_prior in length_prior]
    # Padding
    outputs, masks = self.pad(outputs)
    self.segment_decisions = segment_decisions
    self.segment_logsoftmaxes = segment_logsoftmaxes
    # Packing output together
    if self.learn_segmentation:
      self.segment_length_prior = dy.inputTensor(length_prior, batched=True)
      if self.use_baseline:
        self.bs = [self.baseline(dy.nobackprop(enc)) for enc in encodings]
    if not self.train:
      # Rewrite segmentation
      self.set_report_resource("segmentation", self.segment_decisions)
      self.set_report_input(segment_decisions)

    # Return the encoded batch by the size of [(encode,segment)] * batch_size
    return self.final_transducer(expression_sequence.ExpressionSequence(expr_tensor=outputs, mask=masks))

  @handle_xnmt_event
  def on_start_sent(self, src=None):
    self.src_sent = src

  def pad(self, outputs):
    # Padding
    max_col = max(len(xs) for xs in outputs)
    P0 = dy.vecInput(outputs[0][0].dim()[0][0])
    masks = numpy.zeros((len(outputs), max_col), dtype=int)
    ret = []
    modified = False
    for xs, mask in zip(outputs, masks):
      deficit = max_col - len(xs)
      if deficit > 0:
        xs.extend([P0 for _ in range(deficit)])
        mask[-deficit:] = 1
        modified = True
      ret.append(dy.concatenate_cols(xs))
    mask = Mask(masks) if modified else None
    return dy.concatenate_to_batch(ret), mask

  def sample_segmentation(self, encodings, batch_size):
    lmbd = self.lmbd.value() if self.lmbd is not None else 0
    eps = self.eps.value() if self.eps is not None else None
    segment_logsoftmaxes = [dy.log_softmax(self.segment_transform(fb)) for fb in encodings]
    # Flags
    is_presegment_provided = len(self.src_sent) != 0 and hasattr(self.src_sent[0], "annotation")
    is_warmup = lmbd == 0 or self.is_segmentation_warmup()
    is_epsgreedy_triggered = eps is not None and numpy.random.random() <= eps
    # Sample based on the criterion
    if self.learn_segmentation and not is_warmup and not self.train:
      # During testing always sample from softmax if it is not warmup
      segment_decisions = self.sample_from_softmax(encodings, batch_size, segment_logsoftmaxes)
    elif is_presegment_provided:
      segment_decisions = self.sample_from_prior(encodings, batch_size)
    elif is_warmup or is_epsgreedy_triggered:
      segment_decisions = self.sample_from_poisson(encodings, batch_size)
    else:
      segment_decisions = self.sample_from_softmax(encodings, batch_size, segment_logsoftmaxes)
    segment_decisions = segment_decisions.transpose()

    # The last segment decision of an active components should be equal to 1
    if encodings.mask is not None:
      src = self.src_sent
      mask = [numpy.nonzero(m)[0] for m in encodings.mask.np_arr.transpose()]
      assert len(segment_decisions) == len(mask)
      for i in range(len(segment_decisions)):
        if len(mask[i]) != 0:
          segment_decisions[i-1][mask[i]] = 1
    segment_decisions[-1] = numpy.ones(segment_decisions[-1].shape, dtype=int)

    return segment_decisions, segment_logsoftmaxes

  # Sample from prior segmentation
  def sample_from_prior(self, encodings, batch_size):
    self.print_debug("sample_from_prior")
    segment_decisions = numpy.zeros((batch_size, len(encodings)), dtype=int)
    for segment_decision, sent in zip(segment_decisions, self.src_sent):
      segment_decision[sent.annotation["segment"]] = 1
    return segment_decisions

  # Sample from poisson prior
  def sample_from_poisson(self, encodings, batch_size):
    self.print_debug("sample_from_poisson")
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
    return segment_decisions

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
    return numpy.stack(segment_decisions, 1)

  # Indicates warmup time. So we shouldn't sample from softmax
  def is_segmentation_warmup(self):
    return self.segmentation_warmup_counter < self.segmentation_warmup

  @handle_xnmt_event
  def on_set_train(self, train):
    self.train = train

  def get_final_states(self):
    if hasattr(self.final_transducer, "get_final_states"):
      return self.final_transducer.get_final_states()
    else:
      return self.embed_encoder.get_final_states()

  @handle_xnmt_event
  def on_new_epoch(self):
    name = ["Epsilon Greedy Prob", "Reinforce Loss Weight", "Confidence Penalty Weight", "Length Prior Weight"]
    param = [self.eps, self.lmbd, self.confidence_penalty, self.length_prior_alpha]
    for n, p in zip(name, param):
      if p is not None:
        print(n + ":", p.value())

  @handle_xnmt_event
  def on_next_epoch(self):
    self.segmentation_warmup_counter += 1

  @handle_xnmt_event
  def on_calc_additional_loss(self, reward):
    if not self.learn_segmentation:
      return None
    ret = LossBuilder()
    alpha = self.length_prior_alpha.value()
    if alpha > 0:
      reward += self.segment_length_prior * alpha
    # reward z-score normalization
    if self.z_normalization:
      reward = dy.cdiv(reward-dy.mean_batches(reward) + EPS, dy.std_batches(reward) + EPS)
    # Baseline Loss
    if self.use_baseline:
      baseline_loss = []
      for i, baseline in enumerate(self.bs):
        baseline_loss.append(dy.squared_distance(reward, baseline))
      ret.add_loss("Baseline", dy.esum(baseline_loss))
    # Reinforce Loss
    lmbd = self.lmbd.value()
    if lmbd > 0.0:
      reinforce_loss = []
      # Calculating the loss of the baseline and reinforce
      for i in range(len(self.segment_decisions)):
        ll = dy.pick_batch(self.segment_logsoftmaxes[i], self.segment_decisions[i])
        if self.use_baseline:
          r_i = reward - self.bs[i]
        else:
          r_i = reward
        if self.z_normalization:
          r_i = dy.logistic(r_i)
        else:
          r_i = dy.exp(r_i)
        reinforce_loss.append(r_i * ll)
      loss = -dy.esum(reinforce_loss) * lmbd
      ret.add_loss("Reinforce", loss)
    if self.confidence_penalty:
      ls_loss = self.confidence_penalty(self.segment_logsoftmaxes)
      ret.add_loss("Confidence Penalty", ls_loss)
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

class SegmentingAction(Enum):
  """
  The enumeration of possible action.
  """
  READ = 0
  SEGMENT = 1
  DELETE = 2

class SegmentationConfidencePenalty(Serializable):
  ''' https://arxiv.org/pdf/1701.06548.pdf
      strength: the beta value
  '''
  yaml_tag = u"!SegmentationConfidencePenalty"

  def __init__(self, strength):
    self.strength = strength
    if strength.value() < 0:
      raise RuntimeError("Strength of label smoothing parameter should be >= 0")

  def __call__(self, logsoftmaxes):
    strength = self.strength.value()
    if strength == 0:
      return 0
    neg_entropy = []
    for logsoftmax in logsoftmaxes:
      neg_entropy.append(dy.cmult(dy.exp(logsoftmax), logsoftmax))

    return strength * dy.sum_elems(dy.esum(neg_entropy))

  def value(self):
    return self.strength.value()


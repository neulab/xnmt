from __future__ import print_function

import copy
import logging
logger = logging.getLogger('xnmt')
import numpy
import dynet as dy

from enum import Enum
from xml.sax.saxutils import escape, unescape
from lxml import etree
from scipy.stats import poisson

import xnmt.linear as linear
import xnmt.expression_sequence as expression_sequence

from xnmt.batcher import Mask
from xnmt.serialize.tree_tools import Ref, Path
from xnmt.events import register_handler, handle_xnmt_event
from xnmt.reports import Reportable
from xnmt.serialize.serializable import Serializable
from xnmt.transducer import SeqTransducer, FinalTransducerState
from xnmt.loss import LossBuilder
from xnmt.segmenting_composer import TailWordSegmentTransformer, WordOnlySegmentTransformer
from xnmt.hyper_parameters import GeometricSequence

EPS = 1e-10

class SegmentingSeqTransducer(SeqTransducer, Serializable, Reportable):
  yaml_tag = '!SegmentingSeqTransducer'

  def __init__(self, exp_global=Ref(Path("exp_global")),
               ## COMPONENTS
               embed_encoder=None, segment_composer=None, final_transducer=None,
               ## OPTIONS
               length_prior=3.3,
               length_prior_alpha=None, # GeometricSequence
               epsilon_greedy=None,     # GeometricSequence
               reinforce_scale=None,    # GeometricSequence
               confidence_penalty=None, # SegmentationConfidencePenalty
               print_sample_prob=0.01,
               # For segmentation warmup (Always use the poisson prior)
               segmentation_warmup=0,
               src_vocab = Ref(Path("model.src_reader.vocab")),
               trg_vocab = Ref(Path("model.trg_reader.vocab")),
               ## FLAGS
               learn_delete       = False,
               use_baseline       = True,
               z_normalization    = True,
               learn_segmentation = True,
               compose_char       = False,
               bow_loss           = False,
               print_sample=False):
    register_handler(self)
    model = exp_global.dynet_param_collection.param_col
    # Sanity check
    assert embed_encoder is not None
    assert segment_composer is not None
    assert final_transducer is not None
    # The Embed Encoder transduces the embedding vectors to a sequence of vector
    self.embed_encoder = embed_encoder
    self.src_vocab = src_vocab
    self.trg_vocab = trg_vocab
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
    self.bow_projector = linear.Linear(input_dim = self.segment_composer.hidden_dim,
                                       output_dim = len(trg_vocab),
                                       model = model)
    # Flags
    self.use_baseline = use_baseline
    self.bow_loss = bow_loss
    self.learn_segmentation = learn_segmentation
    self.learn_delete = learn_delete
    self.z_normalization = z_normalization
    self.compose_char = compose_char
    self.print_sample = print_sample
    self.print_sample_prob = print_sample_prob
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
    self.segmentation_warmup_counter = 1

  def __call__(self, embed_sent):
    batch_size = embed_sent[0].dim()[1]
    # Softmax + segment decision
    encodings = self.embed_encoder(embed_sent)
    enc_mask = copy.deepcopy(encodings.mask)
    segment_decisions, segment_logsoftmaxes = self.sample_segmentation(encodings, batch_size)
    # Some checks
    assert len(encodings) == len(segment_decisions), \
           "Encoding={}, segment={}".format(len(encodings), len(segment_decisions))
    # Buffer for output
    buffers = [[] for _ in range(batch_size)]
    outputs = [[] for _ in range(batch_size)]
    last_segment = [-1 for _ in range(batch_size)]
    length_prior_enabled = self.length_prior_alpha is not None and self.length_prior_alpha.value() > 0
    # input
    enc_inp = encodings if not self.compose_char else embed_sent
    # Loop through all the frames (word / item) in input.
    for j, (encoding, segment_decision) in enumerate(zip(enc_inp, segment_decisions)):
      # For each decision in the batch
      for i, decision in enumerate(segment_decision):
        # If segment for this particular input
        decision = int(decision)
        if decision == SegmentingAction.DELETE.value or \
           (enc_mask is not None and enc_mask.np_arr[i][j] == 1):
          continue
        # Get the particular encoding for that batch item
        enc_i = dy.pick_batch_elem(encoding, i)
        # Append the encoding for this item to the buffer
        buffers[i].append(enc_i)
        if decision == SegmentingAction.SEGMENT.value:
          # Reducing the [expression] -> expression
          expr_seq = expression_sequence.ExpressionSequence(expr_list=buffers[i])
          self.segment_composer.set_word_boundary(last_segment[i], j, self.src_sent[i])
          transduce_output = self.segment_composer.transduce(expr_seq)
          outputs[i].append(transduce_output)
          buffers[i] = []
          # Calculate length prior
          #if length_prior_enabled:
          #  length_prior[i] += numpy.log(poisson.pmf(j-last_segment[i], self.length_prior))
          last_segment[i] = j
    if length_prior_enabled:
      length_prior = [poisson.pmf(len(outputs[i]), len(self.src_sent[i]) / self.length_prior) for i in range(len(outputs))]
    # Flag out </s>
    if enc_mask is not None:
      for i, e_i in enumerate(enc_mask.np_arr):
        nonzero = numpy.nonzero(e_i)[0]
        if len(nonzero) != 0:
          eos_index = nonzero[0]-1
        else:
          eos_index = len(e_i)-1
        enc_mask.np_arr[i][eos_index] = 1
    else:
      np_arr = numpy.zeros((len(self.src_sent[0]), batch_size))
      np_arr[-1][:] = 1
      enc_mask = Mask(np_arr.transpose())
    # Packing output together
    outputs, masks = self.pad(outputs)
    self.segment_decisions = segment_decisions
    self.segment_logsoftmaxes = segment_logsoftmaxes
    self.encodings = encodings
    self.enc_mask = enc_mask
    self.outputs = dy.concatenate_to_batch(outputs)
    self.out_mask = masks
    if length_prior_enabled:
      self.segment_length_prior = dy.log(dy.inputTensor(length_prior, batched=True)+EPS)
    # Decide to print or not
    if self.print_sample:
      self.print_sample_triggered = (self.print_sample_prob-numpy.random.random()) >= 0
      if self.print_sample_triggered:
        self.print_sample_enc(outputs, enc_mask, masks)
    if not self.train:
      # Rewrite segmentation
      self.set_report_resource("segmentation", self.segment_decisions)
      self.set_report_input(segment_decisions)
    # Return the encoded batch by the size of [(encode,segment)] * batch_size
    return self.final_transducer(expression_sequence.ExpressionSequence(expr_tensor=self.outputs,
                                                                        mask=masks))

  @handle_xnmt_event
  def on_start_sent(self, src=None):
    self.src_sent = src
    self.segment_length_prior = None
    self.segment_decisions = None
    self.segment_logsoftmaxes = None
    self.bs = None

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
    return ret, mask

  def sample_segmentation(self, encodings, batch_size):
    lmbd = self.lmbd.value() if self.lmbd is not None else 0
    eps = self.eps.value() if self.eps is not None else None
    segment_logsoftmaxes = [dy.log_softmax(self.segment_transform(fb)) for fb in encodings]
    # Flags
    is_presegment_provided = len(self.src_sent) != 0 and self.src_sent[0].has_annotation("segment")
    is_warmup = self.is_segmentation_warmup()
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
      assert len(segment_decisions) == len(mask), \
             "Len(seg)={}, Len(mask)={}".format(len(segment_decisions), len(mask))
      for i in range(len(segment_decisions)):
        if len(mask[i]) != 0:
          segment_decisions[i-1][mask[i]] = 1
    segment_decisions[-1][:] = 1 # </s>
    #### DEBUG
    #print(segment_decisions.transpose()[0])
    #print(numpy.exp(numpy.array([list(log.npvalue().transpose()[0]) for log in segment_logsoftmaxes])))
    ####
    return segment_decisions, segment_logsoftmaxes

  # Sample from prior segmentation
  def sample_from_prior(self, encodings, batch_size):
    segment_decisions = numpy.zeros((batch_size, len(encodings)), dtype=int)
    for segment_decision, sent in zip(segment_decisions, self.src_sent):
      segment_decision[sent.annotation["segment"]] = 1
    return segment_decisions

  # Sample from poisson prior
  def sample_from_poisson(self, encodings, batch_size):
    randoms = numpy.random.poisson(lam=self.length_prior, size=batch_size*len(encodings))
    segment_decisions = numpy.zeros((batch_size, len(encodings)), dtype=int)
    if len(encodings) == 0 or batch_size == 0:
      return segment_decisions
    idx = 0
    # Filling up the segmentation matrix based on the poisson distribution
    if len(encodings) != 0:
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
    if self.train and self.learn_segmentation:
      segment_decisions = [log_softmax.tensor_value().categorical_sample_log_prob().as_numpy()[0]
                           for log_softmax in segment_logsoftmaxes]
      if batch_size == 1:
        segment_decisions = [numpy.array([x]) for x in segment_decisions]
    else:
      segment_decisions = [log_softmax.tensor_value().argmax().as_numpy().transpose()
                           for log_softmax in segment_logsoftmaxes]
    ret = numpy.stack(segment_decisions, 1)
    # Handling dynet argmax() inconsistency (it returns a lesser dimension for output of size 1)
    if len(ret.shape) == 3:
      ret = numpy.squeeze(ret, axis=2)
    return ret

  # Indicates warmup time. So we shouldn't sample from softmax
  def is_segmentation_warmup(self):
    return self.segmentation_warmup_counter <= self.segmentation_warmup

  @handle_xnmt_event
  def on_set_train(self, train):
    self.train = train
#
  def get_final_states(self):
    if hasattr(self.final_transducer, "get_final_states"):
      return self.final_transducer.get_final_states()
    else:
      return self.embed_encoder.get_final_states()

  @handle_xnmt_event
  def on_new_epoch(self, training_task, *args, **kwargs):
    self.segmentation_warmup_counter = training_task.training_state.epoch_num
    name = ["Epsilon Greedy Prob", "Reinforce Weight", "Confidence Penalty Weight", "Length Prior Weight",
            "Epoch Counter"]
    param = [self.eps, self.lmbd, self.confidence_penalty, self.length_prior_alpha, self.segmentation_warmup_counter]
    for n, p in zip(name, param):
      if p is not None:
        if hasattr(p, "value"):
          p = p.value()
        logger.debug(n + ": " + str(p))

  def bow_representation(self, trg):
    ret = []
    for trg_i in trg:
      if "one_hot_sum" not in trg_i:
        one_hot_sum = [0 for _ in range(len(self.trg_vocab))]
        for word in trg_i:
          one_hot_sum[word] += 1
        trg_i.one_hot_sum = one_hot_sum
      ret.append(dy.inputTensor(trg_i.one_hot_sum))
    return dy.concatenate_to_batch(ret)

  def RMSE(self, squared_distance, N):
    return dy.sqrt(dy.cdiv(squared_distance, dy.scalarInput(N))+EPS)

  @handle_xnmt_event
  def on_calc_additional_loss(self, src, trg, translator_loss, trg_words_counts):
    if not self.learn_segmentation or self.segment_decisions is None:
      return None
    ### Constructing Rewards
    # 1. Translator reward
    trans_reward = -(translator_loss.sum())
    trans_reward = dy.cdiv(trans_reward, dy.inputTensor(trg_words_counts, batched=True))
    reward = LossBuilder({"trans_reward": dy.nobackprop(trans_reward)})
    assert trans_reward.dim()[1] == len(self.src_sent)
    enc_mask = self.enc_mask.get_active_one_mask().transpose() if self.enc_mask is not None else None
    ret = LossBuilder()
    # 2. Length prior
    alpha = self.length_prior_alpha.value() if self.length_prior_alpha is not None else 0
    if alpha > 0:
      reward.add_loss("lp_reward", dy.nobackprop(self.segment_length_prior * alpha))
    # 3. Bag of words rewards + loss
    if self.bow_loss:
      bow_rep = self.bow_representation(trg)
      mask = None
      if self.out_mask is not None:
        mask = dy.inputTensor(self.out_mask.get_active_one_mask().transpose(), batched=True)

      p_rep = dy.softmax(self.bow_projector(self.outputs), d=0)
      if mask is not None:
        p_rep = dy.cmult(dy.transpose(p_rep), mask)
      if len(p_rep.dim()[0]) > 1:
        p_rep = dy.sum_dim(p_rep, d=[0])
      bow_loss = dy.squared_distance(p_rep, bow_rep)
      ret.add_loss("rmse_bow", bow_loss)
      reward.add_loss("bow_reward", dy.nobackprop(-bow_loss))
    reward = reward.sum(batch_sum=False)
    # reward z-score normalization
    if self.z_normalization:
      reward = dy.cdiv(reward-dy.mean_batches(reward), dy.std_batches(reward) + EPS)
    ## Baseline Loss
    if self.use_baseline:
      baseline_loss = []
      baseline_score = []
      for i, encoding in enumerate(self.encodings):
        baseline = self.baseline(dy.nobackprop(encoding))
        baseline_score.append(dy.nobackprop(baseline))
        loss = dy.squared_distance(reward, baseline)
        if enc_mask is not None:
          loss = dy.cmult(dy.inputTensor(enc_mask[i], batched=True), loss)
        baseline_loss.append(loss)
      ret.add_loss("baseline", dy.esum(baseline_loss))
    ## Reinforce Loss
    lmbd = self.lmbd.value()
    rewards = []
    lls = []
    reinforce_loss = []
    baseline = []
    if lmbd > 0.0:
      # Calculating the loss of the baseline and reinforce
      for i in range(len(self.segment_decisions)):
        # Log likelihood
        ll = dy.pick_batch(self.segment_logsoftmaxes[i], self.segment_decisions[i])
        if enc_mask is not None:
          ll = dy.cmult(dy.inputTensor(enc_mask[i], batched=True), ll)
        lls.append(ll)
        # reward
        if self.use_baseline:
          r_i = reward - baseline_score[i]
          baseline.append(baseline_score[i])
        else:
          r_i = reward
        rewards.append(r_i)
        # Loss
        reinforce_loss.append(-(r_i * ll))
      loss = dy.esum(reinforce_loss) * lmbd
      ret.add_loss("reinf", loss)
    if self.confidence_penalty:
      ls_loss = self.confidence_penalty(self.segment_logsoftmaxes, enc_mask)
      ret.add_loss("conf_pen", ls_loss)
    if self.print_sample and self.print_sample_triggered:
      self.print_sample_loss(rewards, reward, lls, reinforce_loss, baseline)
    # Total Loss
    return ret

  def print_sample_enc(self, outputs, enc_mask, out_mask):
    src = [self.src_vocab[w] for w in self.src_sent[0]]
    dec = self.segment_decisions.transpose()[0]
    out = []
    number_seg = 0
    if enc_mask is not None:
      self.last_masked = len(enc_mask.np_arr[0]) - numpy.count_nonzero(enc_mask.np_arr[0])
      dec = dec[:self.last_masked]
      src = src[:self.last_masked]
    else:
      self.last_masked = None
    number_seg = numpy.count_nonzero(dec)
    segmented = self.apply_segmentation(src, dec)
    segmented = ["SRC: "] + [x for x, delete in segmented]
    logger.debug(" ".join(segmented))
    if out_mask is not None:
      # Number of segment == 0 flag
      assert len(out_mask.np_arr[0]) - numpy.count_nonzero(out_mask.np_arr[0]) == number_seg+1

  # TODO: Fix if the baseline is none
  def print_sample_loss(self, rewards, trans_reward, ll, loss, baseline):
    if hasattr(self, "segment_length_prior") and self.segment_length_prior is not None:
      logger.debug("LP: " + str(self.segment_length_prior.npvalue()[0][0]))
    rw_val = trans_reward.npvalue()[0][0]
    logger.debug("RW: " +  str(rw_val))
    enc_mask = self.enc_mask
    if self.last_masked is not None:
      rewards = [x.npvalue()[0][0] for x in rewards[:self.last_masked]]
      ll = [x.npvalue()[0][0] for x in ll[:self.last_masked]]
      loss = [x.npvalue()[0][0] for x in loss[:self.last_masked]]
      baseline = [x.npvalue()[0][0] for x in baseline[:self.last_masked]]
    else:
      rewards = [x.npvalue()[0][0] for x in rewards]
      ll = [x.npvalue()[0][0] for x in ll]
      loss = [x.npvalue()[0][0] for x in loss]
      baseline = [x.npvalue()[0][0] for x in baseline]

    logger.debug("loss = -[ll * (rewards - baseline)]")
    for l, log, r, b in zip(loss, ll, rewards, baseline):
      logger.debug("%f = -[%f * (%f - %f)]" % (l, log, rw_val, b))

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
    logsoftmaxes = [x.npvalue() for x in self.segment_logsoftmaxes]

    with open(self.get_report_path() + ".segment", encoding='utf-8', mode='w') as segmentation_file:
      if len(segmented) > 0:
        print(" ".join(segmented), file=segmentation_file)

    if self.learn_segmentation:
      with open(self.get_report_path() + ".segdecision", encoding='utf-8', mode='w') as segmentation_file:
        for softmax in logsoftmaxes:
          print(" ".join(["%.5f" % f for f in numpy.exp(softmax)]), file=segmentation_file)

      with open(self.get_report_path() + ".segprob", encoding='utf-8', mode='w') as segmentation_file:
        logprob = 0
        for logsoftmax, decision in zip(logsoftmaxes, segment_decision):
          logprob += logsoftmax[decision]
        print(logprob, file=segmentation_file)

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
  yaml_tag = "!SegmentationConfidencePenalty"

  def __init__(self, strength):
    self.strength = strength
    if strength.value() < 0:
      raise RuntimeError("Strength of label smoothing parameter should be >= 0")

  def __call__(self, logsoftmaxes, mask):
    strength = self.strength.value()
    if strength < 1e-8:
      return None
    neg_entropy = []
    for i, logsoftmax in enumerate(logsoftmaxes):
      loss = dy.cmult(dy.exp(logsoftmax), logsoftmax)
      if mask is not None:
        loss = dy.cmult(dy.inputTensor(mask[i], batched=True), loss)
      neg_entropy.append(loss)

    return strength * dy.sum_elems(dy.esum(neg_entropy))

  def value(self):
    return self.strength.value()

  def __str__(self):
    return str(self.strength.value())

  def __repr__(self):
    return repr(self.strength.value())


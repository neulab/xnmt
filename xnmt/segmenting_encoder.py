import numpy
import dynet as dy

from enum import Enum
from xml.sax.saxutils import escape
from lxml import etree
from scipy.stats import poisson

import xnmt.linear as linear
import xnmt.expression_sequence as expression_sequence
from xnmt import logger
from xnmt.vocab import Vocab
from xnmt.batcher import Mask
from xnmt.events import register_xnmt_handler, handle_xnmt_event
from xnmt.persistence import serializable_init, Serializable
from xnmt.transducer import SeqTransducer
from xnmt.loss import LossBuilder
from xnmt.param_collection import ParamManager
from xnmt.persistence import Ref, bare, Path
from xnmt.constants import EPSILON

class SegmentingSeqTransducer(SeqTransducer, Serializable):
  yaml_tag = '!SegmentingSeqTransducer'

  @register_xnmt_handler
  @serializable_init
  def __init__(self,
               ## COMPONENTS
               embed_encoder=None, segment_composer=None, final_transducer=None, segment_transform=None, baseline=None,
               ## OPTIONS
               length_prior=3.3,
               length_prior_alpha=None, # GeometricSequence
               epsilon_greedy=None,     # GeometricSequence
               reinforce_scale=None,    # GeometricSequence
               confidence_penalty=None, # SegmentationConfidencePenalty
               learn_gold=None,
               print_sample_prob=0.01,
               src_vocab = Ref(Path("model.src_reader.vocab")),
               trg_vocab = Ref(Path("model.trg_reader.vocab")),
               ## FLAGS
               learn_delete         = False,
               use_baseline         = True,
               z_normalization      = False,
               learn_segmentation   = True,
               compose_char         = False,
               sample_during_search = False,
               exp_reward           = True,
               exp_logsoftmax       = False,
               print_sample         = False):
    model = ParamManager.my_params(self)
    # Sanity check
    assert embed_encoder is not None
    assert segment_composer is not None
    assert final_transducer is not None
    # The Embed Encoder transduces the embedding vectors to a sequence of vector
    self.embed_encoder = embed_encoder
    embed_encoder_dim = embed_encoder.hidden_dim
    # The Segment transducer produced word embeddings based on sequence of character embeddings
    self.segment_composer = segment_composer
    # The final transducer
    self.final_transducer = final_transducer
    # Decision layer of segmentation
    self.segment_transform = self.add_serializable_component("segment_transform", segment_transform,
                                                             lambda: linear.Linear(input_dim=embed_encoder_dim,
                                                                                   output_dim=3 if learn_delete else 2))
    # The baseline linear regression model
    self.baseline = self.add_serializable_component("baseline", baseline,
                                                    lambda: linear.Linear(input_dim=embed_encoder_dim, output_dim=1))
    self.src_vocab = src_vocab
    # Flags
    self.use_baseline = use_baseline
    self.learn_segmentation = learn_segmentation
    self.learn_delete = learn_delete
    self.z_normalization = z_normalization
    self.compose_char = compose_char
    self.print_sample = print_sample
    self.print_sample_prob = print_sample_prob
    self.exp_reward = exp_reward
    self.exp_logsoftmax = exp_logsoftmax
    self.sample_during_search = sample_during_search
    # Fixed Parameters
    self.length_prior = length_prior
    # Variable Parameters
    self.length_prior_alpha = length_prior_alpha
    self.lmbd = reinforce_scale
    self.eps = epsilon_greedy
    self.learn_gold = learn_gold
    self.confidence_penalty = confidence_penalty
    # States of the object
    self.train = False
    if learn_delete:
      raise NotImplementedError("Learn delete is not supported yet.")

  def __call__(self, embed_sent):
    batch_size = embed_sent[0].dim()[1]
    # Softmax + segment decision
    encodings = self.embed_encoder(embed_sent)
    segment_decisions, segment_logsoftmaxes = self.sample_segmentation(encodings, batch_size)
    # Length prior
    if self.length_prior_alpha is not None and self.length_prior_alpha.value() > 0:
      length_prior = [poisson.pmf(len(seg_dec), exp_len)+EPSILON \
                      for seg_dec, exp_len in zip(segment_decisions, self.expected_length)]
      self.segment_length_prior = dy.log(dy.inputTensor(length_prior, batched=True))
    # Composing segments
    compose_input = encodings if self.compose_char else embed_sent
    sent_encodings = dy.concatenate(compose_input.expr_list, d=1)
    outputs = [[] for _ in range(batch_size)]
    for i, decision in enumerate(segment_decisions):
      sequence = dy.pick_batch_elem(sent_encodings, i)
      src_sent = self.src_sent[i]
      lower_bound = 0
      for upper_bound in sorted(decision):
        expr_tensor = dy.pick_range(sequence, lower_bound, upper_bound+1, 1)
        expr_seq = expression_sequence.ExpressionSequence(expr_tensor=expr_tensor)
        self.segment_composer.set_word_boundary(lower_bound, upper_bound, src_sent)
        composed = self.segment_composer.transduce(expr_seq)
        outputs[i].append(composed)
        lower_bound = upper_bound+1
    # Padding
    outputs, segment_mask = self.pad(outputs)
    # Packing outputs
    self.segment_decisions = segment_decisions
    self.segment_logsoftmaxes = segment_logsoftmaxes
    self.encodings = encodings
    self.segment_mask = segment_mask
    self.outputs = dy.concatenate_to_batch(outputs)
    # Decide to print or not
    if self.print_sample:
      self.print_sample_triggered = (self.print_sample_prob-numpy.random.random()) >= 0
      if self.print_sample_triggered:
        self.print_sample_enc(outputs, self.enc_mask, segment_mask)
    if not self.train:
      # Rewrite segmentation
      self.segmentation = self.segment_decisions[0] 
    # Return the encoded batch by the size of [(encode,segment)] * batch_size
    return self.final_transducer(expression_sequence.ExpressionSequence(expr_tensor=self.outputs,
                                                                        mask=segment_mask))


  @handle_xnmt_event
  def on_start_sent(self, src=None):
    self.src_sent = src
    self.segment_length_prior = None
    self.segment_decisions = None
    self.segment_logsoftmaxes = None
    self.bs = None

    # Note we want to use the original length of the input
    # Refactor it better?
    self.src_length = [src.original_length for src in self.src_sent]
    self.expected_length = [src_len / self.length_prior for src_len in self.src_length]

  def pad(self, outputs):
    # Padding
    max_col = max(len(xs) for xs in outputs)
    P0 = dy.vecInput(outputs[0][0].dim()[0][0])
    masks = numpy.zeros((len(outputs), max_col), dtype=int)
    modified = False
    ret = []
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
    learn_gold = self.learn_gold.value() if self.learn_gold is not None else None
    segment_logsoftmaxes = [dy.log_softmax(self.segment_transform(fb)) for fb in encodings]
    # Flags
    is_presegment_provided = self.src_sent[0].has_annotation("segment")
    is_epsgreedy_triggered = eps is not None and numpy.random.random() <= eps
    is_not_learn_from_gold = learn_gold is not None and learn_gold == 0
    # Sample based on the criterion
    if self.learn_segmentation and not self.train and is_not_learn_from_gold:
      segment_decisions = self.sample_from_softmax(encodings, batch_size, segment_logsoftmaxes)
    elif is_presegment_provided:
      segment_decisions = self.sample_from_prior(encodings, batch_size)
    elif is_epsgreedy_triggered:
      segment_decisions = self.sample_from_poisson(encodings, batch_size)
    else:
      segment_decisions = self.sample_from_softmax(encodings, batch_size, segment_logsoftmaxes)
    # Masking + adding last dec must be 1
    ret = []
    if encodings.mask is None:
      enc_mask = numpy.ones((batch_size, len(encodings)))
    else:
      enc_mask = 1-encodings.mask.np_arr
    self.enc_mask = enc_mask

    if is_presegment_provided:
      ret = [set(dec) for dec in segment_decisions]
    else:
      for i in range(len(segment_decisions)):
        segment_decision = segment_decisions[i]
        src_length = self.src_length[i]
        mask = enc_mask[i]
        dec = set(filter(lambda j: mask[j] == 1, segment_decision))
        if len(dec) == 0 or src_length-1 not in dec:
          dec.add(src_length-1)
        ret.append(dec)
        mask[src_length-1] = 0
    return ret, segment_logsoftmaxes

  # Sample from prior segmentation
  def sample_from_prior(self, encodings, batch_size):
    #print("sample_from_prior")
    return [sent.annotation["segment"] for sent in self.src_sent]

  # Sample from poisson prior
  def sample_from_poisson(self, encodings, batch_size):
    assert len(encodings) != 0
    #print("sample_from_poisson")
    randoms = list(filter(lambda x: x > 0, numpy.random.poisson(lam=self.length_prior, size=batch_size*len(encodings))))
    segment_decisions = [[] for _ in range(batch_size)]
    idx = 0
    # Filling up the segmentation matrix based on the poisson distribution
    for decision in segment_decisions:
      current = randoms[idx]
      while current < len(encodings):
        decision.append(current)
        idx += 1
        current += randoms[idx]
    return segment_decisions

  # Sample from the softmax
  def sample_from_softmax(self, encodings, batch_size, segment_logsoftmaxes):
    # Sample from the softmax
    if self.train and self.learn_segmentation or self.sample_during_search:
      #print("sample_from_softmax")
      segment_decisions = [log_softmax.tensor_value().categorical_sample_log_prob().as_numpy()[0]
                           for log_softmax in segment_logsoftmaxes]
      if batch_size == 1:
        segment_decisions = [numpy.array([x]) for x in segment_decisions]
    else:
      #print("sample_from_argmax")
      segment_decisions = [log_softmax.tensor_value().argmax().as_numpy().transpose()
                           for log_softmax in segment_logsoftmaxes]
    ret = numpy.stack(segment_decisions, axis=1)
    ret = [numpy.where(line == SegmentingAction.SEGMENT.value)[0] for line in ret]
    return ret

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
    name = ["Epsilon Greedy Prob", "Reinforce Weight", "Confidence Penalty Weight", "Length Prior Weight",
            "Epoch Counter"]
    param = [self.eps, self.lmbd, self.confidence_penalty, self.length_prior_alpha]
    for n, p in zip(name, param):
      if p is not None:
        if hasattr(p, "value"):
          p = p.value()
        logger.debug(n + ": " + str(p))

  @handle_xnmt_event
  def on_calc_additional_loss(self, src, trg, translator_loss):
    if not self.learn_segmentation or self.segment_decisions is None:
      return None
    ### Constructing Rewards
    # 1. Translator reward
    trans_loss = translator_loss.get_nobackprop_loss()
    trg_counts = [t.original_length for t in trg]
    trans_loss["mle"] = dy.cdiv(trans_loss["mle"], dy.inputTensor(trg_counts, batched=True))
    trans_reward = -dy.esum(list(trans_loss.values()))
    reward = LossBuilder({"trans_reward": dy.nobackprop(trans_reward)})
    assert trans_reward.dim()[1] == len(self.src_sent)
    enc_mask = self.enc_mask.transpose()
    # Sanity check
    actual = len(self.segment_logsoftmaxes), len(src)
    assert enc_mask.shape == actual,\
        "expected %s != actual %s" % (str(enc_mask.shape), str(actual))
    ret = LossBuilder()
    # 2. Length prior
    alpha = self.length_prior_alpha.value() if self.length_prior_alpha is not None else 0
    if alpha > 0:
      reward.add_loss("lp_reward", dy.nobackprop(self.segment_length_prior * alpha))
    # reward z-score normalization
    reward = reward.sum(batch_sum=False)
    if self.exp_reward:
      reward = dy.exp(reward)
    if self.z_normalization:
      reward = dy.cdiv(reward-dy.mean_batches(reward), dy.std_batches(reward) + EPSILON)
    baseline_score = []
    ## Baseline Loss
    if self.use_baseline:
      baseline_loss = []
      baseline_score = []
      for i, encoding in enumerate(self.encodings):
        baseline = self.baseline(dy.nobackprop(encoding))
        baseline_score.append(dy.nobackprop(baseline))
        loss = dy.squared_distance(reward, baseline)
        baseline_loss.append(dy.cmult(dy.inputTensor(enc_mask[i], batched=True), loss))
      ret.add_loss("baseline", dy.esum(baseline_loss))
    if self.exp_logsoftmax:
      self.segment_logsoftmaxes = [dy.exp(logsoftmax) for logsoftmax in self.segment_logsoftmaxes]
    ## Reinforce Loss
    lmbd = self.lmbd.value()
    rewards = []
    lls = []
    reinforce_loss = []
    if lmbd > 0.0:
      # Calculating the loss of the baseline and reinforce
      for i in range(len(self.segment_logsoftmaxes)):
        # Log likelihood
        decision = [(1 if i in dec_set else 0) for dec_set in self.segment_decisions]
        ll = dy.pick_batch(self.segment_logsoftmaxes[i], decision)
        # reward
        r_i = reward - baseline_score[i] if self.use_baseline else reward
        # Loss
        loss = -r_i * ll
        rewards.append(r_i)
        lls.append(ll)
        reinforce_loss.append(dy.cmult(dy.inputTensor(enc_mask[i], batched=True), loss))
      loss = dy.esum(reinforce_loss) * lmbd
      ret.add_loss("reinf", loss)
    if self.confidence_penalty:
      ls_loss = self.confidence_penalty(self.segment_logsoftmaxes, enc_mask)
      ret.add_loss("conf_pen", ls_loss)
    if self.print_sample and self.print_sample_triggered:
      self.print_sample_loss(rewards, reward, lls, reinforce_loss, baseline_score, enc_mask)
    # Total Loss
    return ret

  def print_sample_enc(self, outputs, enc_mask, out_mask):
    src = [self.src_vocab[w] for w in self.src_sent[0]][:self.src_length[0]]
    dec = self.segment_decisions[0]
    out = []
    number_seg = 0
    segmented = self.apply_segmentation(src, dec)
    segmented = ["SRC: "] + segmented
    logger.debug(" ".join(segmented))

  def print_sample_loss(self, rewards, trans_reward, ll, loss, baseline, enc_mask):
    if loss[0].dim()[1] <= 1:
      return
    src = [self.src_vocab[w] for w in self.src_sent[0]]
    dec = numpy.zeros(len(ll))
    dec[list(self.segment_decisions[0])] = 1
    if hasattr(self, "segment_length_prior") and self.segment_length_prior is not None:
      logger.debug("LP: " + str(self.segment_length_prior.npvalue()[0][0]))
    rw_val = trans_reward.npvalue()[0][0]
    logger.debug("RW: " +  str(rw_val))
    rewards = [x.npvalue()[0][0] for x in rewards]
    ll = [x.npvalue()[0][0] for x in ll]
    loss = [x.npvalue()[0][0] for x in loss]
    if self.use_baseline:
      baseline = [x.npvalue()[0][0] for x in baseline]
    if enc_mask is not None:
      mask = enc_mask.transpose()[0]
    else:
      mask = [1 for _ in range(len(loss))]

    logger.debug("loss = -[ll * (rewards - baseline)]")
    for i, (l, log, r) in enumerate(zip(loss, ll, rewards)):
      if self.use_baseline:
        logger.debug("%f = -[%f * (%f - %f)] [m=%d] %s %d" % (l, log, rw_val, baseline[i], mask[i], src[i], dec[i]))
      else:
        logger.debug("%f = -[%f * %f] [m=%d] %s %d" % (l, log, rw_val, mask[i], src[i], dec[i]))

  @handle_xnmt_event
  def on_html_report(self, context):
    segment_decision = self.segmentation
    src_words = [escape(self.src_vocab[x]) for x in self.src_sent[0].words]
    main_content = context.xpath("//body/div[@name='main_content']")[0]
    # construct the sub element from string
    segmented = self.apply_segmentation(src_words, segment_decision)
    if len(segmented) > 0:
      segment_html = "<p>Segmentation: " + ", ".join(segmented) + "</p>"
      main_content.insert(2, etree.fromstring(segment_html))

    return context

  @handle_xnmt_event
  def on_file_report(self, report_path):
    segment_decision = self.segmentation
    src_words = [self.src_vocab[x] for x in self.src_sent[0].words]
    segmented = self.apply_segmentation(src_words, segment_decision)

    if self.learn_segmentation and self.segment_logsoftmaxes:
      logsoftmaxes = [x.npvalue() for x in self.segment_logsoftmaxes]
      with open(report_path + ".segdecision", encoding='utf-8', mode='w') as segmentation_file:
        for softmax in logsoftmaxes:
          print(" ".join(["%.5f" % f for f in numpy.exp(softmax)]), file=segmentation_file)

  @handle_xnmt_event
  def on_line_report(self, output_dict):
    logsoft = self.segment_logsoftmaxes
    if logsoft is None:
      return
    decision = lambda i: [(1 if i in dec_set else 0) for dec_set in self.segment_decisions]
    segmentation_prob = [dy.pick_batch(logsoft[i], decision(i)) for i in range(len(logsoft))]
    segmentation_prob = dy.pick_batch_elem(dy.esum(segmentation_prob), 0)
    output_dict["07segenc"] = segmentation_prob.scalar_value()

  def apply_segmentation(self, words, segmentation):
    segmented = []
    lower_bound = 0
    for j in sorted(segmentation):
      upper_bound = j+1
      segmented.append("".join(words[lower_bound:upper_bound]))
      lower_bound = upper_bound
    return segmented

class SegmentingAction(Enum):
  """
  The enumeration of possible action.
  """
  READ = 0
  SEGMENT = 1

class SegmentationConfidencePenalty(Serializable):
  ''' https://arxiv.org/pdf/1701.06548.pdf
      strength: the beta value
  '''
  yaml_tag = "!SegmentationConfidencePenalty"

  @serializable_init
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


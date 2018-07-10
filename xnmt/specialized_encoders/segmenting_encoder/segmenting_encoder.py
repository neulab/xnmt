import numpy as np
import dynet as dy
from typing import List

from enum import Enum
from xml.sax.saxutils import escape
from lxml import etree

from xnmt.transform import Linear
from xnmt.expression_sequence import ExpressionSequence

from xnmt import logger
from xnmt.vocab import Vocab
from xnmt.batcher import Mask
from xnmt.events import register_xnmt_handler, handle_xnmt_event
from xnmt.persistence import serializable_init, Serializable
from xnmt.transducer import SeqTransducer, FinalTransducerState
from xnmt.loss import FactoredLossExpr
from xnmt.param_collection import ParamManager
from xnmt.persistence import Ref, bare, Path
from xnmt.constants import EPSILON
from xnmt.rl.eps_greedy import EpsilonGreedy
from xnmt.specialized_encoders.segmenting_encoder.length_prior import LengthPrior

class SegmentingSeqTransducer(SeqTransducer, Serializable):
  yaml_tag = '!SegmentingSeqTransducer'

  @register_xnmt_handler
  @serializable_init
  def __init__(self,
               ## COMPONENTS
               embed_encoder=None, segment_composer=None,
               final_transducer=None, policy_learning=None,
               eps_greedy=None,
               length_prior=None,
               # GeometricSequence
               sample_during_search=True,
               confidence_penalty=None, # SegmentationConfidencePenalty
               src_vocab = Ref(Path("model.src_reader.vocab")),
               trg_vocab = Ref(Path("model.trg_reader.vocab")),
               embed_encoder_dim = Ref("exp_global.default_layer_dim")):
    model = ParamManager.my_params(self)
    # Sanity check
    assert embed_encoder is not None
    assert segment_composer is not None
    assert final_transducer is not None
    self.embed_encoder = embed_encoder
    # The Segment transducer produced word embeddings based on sequence of character embeddings
    self.segment_composer = segment_composer
    # The final transducer
    self.final_transducer = final_transducer
    # Policy learning
    self.policy_learning = policy_learning

    # Reference to the vocab
    self.src_vocab = src_vocab
    self.trg_vocab = trg_vocab
    # Fixed Parameters
    self.length_prior = length_prior
    # Variable Parameters
    self.eps_greedy = eps_greedy
    self.confidence_penalty = confidence_penalty
    self.sample_during_search = sample_during_search
    # States of the object
    self.train = False

  def transduce(self, embed_sent: ExpressionSequence) -> List[ExpressionSequence]:
    batch_size = embed_sent[0].dim()[1]
    actions = self.sample_segmentation(self.embed_encoder.transduce(embed_sent), batch_size)
    sample_size = len(actions)
    embeddings = dy.concatenate(embed_sent.expr_list, d=1)
    outputs = [[[] for _ in range(batch_size)] for _ in range(sample_size)]
    for i in range(batch_size):
      sequence = dy.pick_batch_elem(embeddings, i)
      # For each sampled segmentations
      for j, sample in enumerate(actions):
        lower_bound = 0
        # Read every 'segment' decision
        for upper_bound in sample[i]:
          char_sequence = ExpressionSequence(expr_tensor=dy.pick_range(sequence, lower_bound, upper_bound+1, 1))
          self.segment_composer.set_word_boundary(lower_bound, upper_bound, self.src_sent[i])
          composed = self.segment_composer.transduce(char_sequence)
          outputs[j][i].append(composed)
          lower_bound = upper_bound+1
    try:
      enc_outputs = []
      for batched_sampled_sentence in outputs:
        sampled_sentence, segment_mask = self.pad(batched_sampled_sentence)
        expr_seq = ExpressionSequence(expr_tensor=dy.concatenate_to_batch(sampled_sentence), mask=segment_mask)
        sent_context = self.final_transducer.transduce(expr_seq)
        self.final_states.append(self.final_transducer.get_final_states())
        enc_outputs.append(sent_context)
      return enc_outputs
    finally:
      self.word_embeddings = outputs

  @handle_xnmt_event
  def on_start_sent(self, src):
    self.src_sent = src
    self.final_states = []

  @handle_xnmt_event
  def on_set_train(self, train):
    self.train = train

  def pad(self, outputs):
    # Padding
    max_col = max(len(xs) for xs in outputs)
    P0 = dy.vecInput(outputs[0][0].dim()[0][0])
    masks = np.zeros((len(outputs), max_col), dtype=int)
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

  def sample_from_policy(self, encodings, batch_size):
    from_argmax = not self.train and not self.sample_during_search
    action_size = 1 if from_argmax else self.policy_learning.get_num_sample()
    actions = [[set() for _ in range(batch_size)] for _ in range(action_size)]
    for position, encoding in enumerate(encodings):
      action = self.policy_learning.sample_action(encoding, argmax=from_argmax)
      if batch_size == 1:
        action = [action]
      for i, sample in enumerate(action):
        for j in np.nonzero(sample)[0]:
          actions[i][j].add(position)
    try:
      return actions
    finally:
      self.sample_action = SampleAction.ARGMAX if from_argmax else SampleAction.SOFTMAX
    
  def sample_segmentation(self, encodings, batch_size):
    triggered = self.eps_greedy is not None and self.eps_greedy.is_triggered()
    if triggered:
      pass # Implement logic for eps_greedy here
    elif self.policy_learning is None:
      actions = self.sample_from_gold()
    else:
      actions = self.sample_from_policy(encodings, batch_size)
    # The last action needs to be 1
    for sample in actions:
      for i, sample_i in enumerate(sample):
        last_eos = self.src_sent[i].len_unpadded()
        if sample_i[-1] != last_eos:
          sample_i.append(last_eos)
    return actions 

  # Sample from prior segmentation
  def sample_from_gold(self):
    try:
      return [[sent.segment for sent in self.src_sent]]
    finally:
      self.sample_action = SampleAction.GOLD

  # Sample from poisson prior
  def sample_from_poisson(self, encodings, batch_size):
    assert len(encodings) != 0
    randoms = list(filter(lambda x: x > 0, np.random.poisson(lam=self.length_prior, size=batch_size*len(encodings))))
    segment_decisions = [[] for _ in range(batch_size)]
    idx = 0
    if len(randoms) == 0:
      randoms = [0]
    # Filling up the segmentation matrix based on the poisson distribution
    for decision in segment_decisions:
      current = randoms[idx]
      while current < len(encodings):
        decision.append(current)
        idx = (idx + 1) % len(randoms)
        current += max(randoms[idx], 1)
    try:
      return segment_decisions
    finally:
      self.sample_action = SampleAction.LP

  def get_final_states(self) -> List[List[FinalTransducerState]]:
    return self.final_states

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
    ret = FactoredLossExpr()
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

#  @handle_xnmt_event
#  def on_html_report(self, context):
#    segment_decision = self.segmentation
#    src_words = [escape(self.src_vocab[x]) for x in self.src_sent[0].words]
#    main_content = context.xpath("//body/div[@name='main_content']")[0]
#    # construct the sub element from string
#    segmented = self.apply_segmentation(src_words, segment_decision)
#    if len(segmented) > 0:
#      segment_html = "<p>Segmentation: " + ", ".join(segmented) + "</p>"
#      main_content.insert(2, etree.fromstring(segment_html))
#
#    return context
#
#  @handle_xnmt_event
#  def on_file_report(self, report_path):
#    segment_decision = self.segmentation
#    src_words = [self.src_vocab[x] for x in self.src_sent[0].words]
#    segmented = self.apply_segmentation(src_words, segment_decision)
#
#    if self.learn_segmentation and self.segment_logsoftmaxes:
#      logsoftmaxes = [x.npvalue() for x in self.segment_logsoftmaxes]
#      with open(report_path + ".segdecision", encoding='utf-8', mode='w') as segmentation_file:
#        for softmax in logsoftmaxes:
#          print(" ".join(["%.5f" % f for f in np.exp(softmax)]), file=segmentation_file)
#
#  @handle_xnmt_event
#  def on_line_report(self, output_dict):
#    logsoft = self.segment_logsoftmaxes
#    if logsoft is None:
#      return
#    decision = lambda i: [(1 if i in dec_set else 0) for dec_set in self.segment_decisions]
#    segmentation_prob = [dy.pick_batch(logsoft[i], decision(i)) for i in range(len(logsoft))]
#    segmentation_prob = dy.pick_batch_elem(dy.esum(segmentation_prob), 0)
#    output_dict["07segenc"] = segmentation_prob.scalar_value()

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

class SampleAction(Enum):
  SOFTMAX = 0
  ARGMAX = 1
  GOLD = 2
  LP = 3


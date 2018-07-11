import numpy as np
import dynet as dy

from typing import List
from enum import Enum

from xnmt import logger
from xnmt.batcher import Mask
from xnmt.events import register_xnmt_handler, handle_xnmt_event
from xnmt.expression_sequence import ExpressionSequence
from xnmt.persistence import serializable_init, Serializable
from xnmt.transducer import SeqTransducer, FinalTransducerState
from xnmt.loss import FactoredLossExpr

class SegmentingSeqTransducer(SeqTransducer, Serializable):
  yaml_tag = '!SegmentingSeqTransducer'

  @register_xnmt_handler
  @serializable_init
  def __init__(self, embed_encoder=None, segment_composer=None,
                     final_transducer=None, policy_learning=None,
                     length_prior=None, eps_greedy=None,
                     sample_during_search=False):
    self.embed_encoder = self.add_serializable_component("embed_encoder", embed_encoder, lambda: embed_encoder)
    self.segment_composer = self.add_serializable_component("segment_composer", segment_composer, lambda: segment_composer)
    self.final_transducer = self.add_serializable_component("final_transducer", final_transducer, lambda: final_transducer)
    self.policy_learning = self.add_serializable_component("policy_learning", policy_learning, lambda: policy_learning) if policy_learning is not None else None
    self.length_prior = self.add_serializable_component("length_prior", length_prior, lambda: length_prior) if length_prior is not None else None
    self.eps_greedy = self.add_serializable_component("eps_greedy", eps_greedy, lambda: eps_greedy) if eps_greedy is not None else None
    self.sample_during_search = sample_during_search

  def transduce(self, embed_sent: ExpressionSequence) -> List[ExpressionSequence]:
    batch_size = embed_sent[0].dim()[1]
    embed_encode = self.embed_encoder.transduce(embed_sent)
    actions = self.sample_segmentation(embed_encode, batch_size)
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
      self.compose_output = outputs
      self.segment_actions = actions

      if self.policy_learning:
        self.policy_learning.set_baseline_input(embed_encode)

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

  def sample_from_policy(self, encodings, batch_size, predefined_actions=None):
    from_argmax = not self.train and not self.sample_during_search
    sample_size = 1 if from_argmax else self.policy_learning.get_num_sample()
    actions = [[[] for _ in range(batch_size)] for _ in range(sample_size)]
    mask = encodings.mask.np_arr if encodings.mask else None
    # Callback to ensure all samples are ended with </s> being segmented
    def ensure_end_segment(sample, position):
      for i in range(len(sample)):
        last_eos = self.src_sent[i].len_unpadded()
        if position >= last_eos:
          sample[i] = 1
      return sample
    # Loop through all items in the sequence
    for position, encoding in enumerate(encodings):
      # Sample from softmax if we have no predefined action
      predefined = predefined_actions[position] if predefined_actions is not None else None
      action = self.policy_learning.sample_action(encoding,
                                                  argmax=from_argmax,
                                                  sample_pp=lambda x: ensure_end_segment(x, position),
                                                  predefined_actions=predefined)
      # Appending the "1" position if it has valid flags
      for i, sample in enumerate(action):
        for j in np.nonzero(sample)[0]:
          if mask is None or mask[j][position] == 0:
            actions[i][j].append(position)
    try:
      return actions
    finally:
      self.sample_action = SampleAction.ARGMAX if from_argmax else SampleAction.SOFTMAX
    
  def sample_segmentation(self, encodings, batch_size):
    if self.policy_learning is None:
      actions = self.sample_from_gold()
    else:
      actions = self.sample_from_policy(encodings, batch_size)
    return actions 

  # Sample from prior segmentation
  def sample_from_gold(self):
    try:
      return [[sent.segment for sent in self.src_sent]]
    finally:
      self.sample_action = SampleAction.GOLD

  def get_final_states(self) -> List[List[FinalTransducerState]]:
    return self.final_states

  @handle_xnmt_event
  def on_calc_additional_loss(self, trg, generator, generator_loss):
    assert hasattr(generator, "losses"), "Must support multi sample encoder from generator."
    if self.policy_learning is None:
      return None
    ### Calculate reward
    rewards = []
    trg_counts = dy.inputTensor([t.len_unpadded() for t in trg], batched=True)
    # Iterate through all samples
    for loss, actions in zip(generator.losses, self.compose_output):
      reward = FactoredLossExpr()
      # Adding all reward from the translator
      for key, value in loss.get_nobackprop_loss().items():
        if key == 'mle':
          reward.add_loss('mle', dy.cdiv(value, trg_counts))
        else:
          reward.add_loss('key', value)
      if self.length_prior is not None:
        reward.add_loss('seg_lp', self.length_prior.log_ll(sample_action))
      rewards.append(dy.esum(list(reward.expr_factors.values())))
    ### Calculate losses    
    return self.policy_learning.calc_loss(rewards)

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

class SampleAction(Enum):
  SOFTMAX = 0
  ARGMAX = 1
  GOLD = 2
  LP = 3


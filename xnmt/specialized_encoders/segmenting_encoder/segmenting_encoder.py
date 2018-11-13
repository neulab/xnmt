import numpy as np
import dynet as dy

from typing import List
from enum import Enum

from xnmt import logger
from xnmt.batchers import Mask
from xnmt.events import register_xnmt_handler, handle_xnmt_event
from xnmt.expression_seqs import ExpressionSequence
from xnmt.persistence import serializable_init, Serializable, Ref, bare
from xnmt.transducers.base import SeqTransducer, FinalTransducerState, IdentitySeqTransducer
from xnmt.losses import FactoredLossExpr
from xnmt.specialized_encoders.segmenting_encoder.priors import GoldInputPrior
from xnmt.reports import Reportable
from xnmt.transducers.recurrent import BiLSTMSeqTransducer
from xnmt.specialized_encoders.segmenting_encoder.segmenting_composer import SeqTransducerComposer, VocabBasedComposer

class SegmentingSeqTransducer(SeqTransducer, Serializable, Reportable):
  """
  A transducer that perform composition on smaller units (characters) into bigger units (words).
  This transducer will predict/use the segmentation discrete actions to compose the inputs.

  The composition function is defined by segment_composer. Please see segmenting_composer.py
  The final transducer is used to transduce the composed sequence. Usually it is a variant of RNN.

  ** USAGE
  This transducer is able to sample from several distribution.

  To segment from some predefined segmentations, please read the word corpus with CharFromWordTextReader.
  To learn the segmentation, please define the policy_learning. Please see rl/policy_learning.py
  To partly defined some segmentation using priors or gold input and learn from it, use the EpsilonGreedy with the proper priors. Please see priors.py.
  To sample from the policy instead doing argmax when doing inference please turn on the sample_during_search
  
  ** LEARNING
  By default it will use the policy gradient function to learn the network. The reward is composed by:

  REWARD = -sum(GENERATOR_LOSS) / len(TRG_SEQUENCE)

  Additional reward can be added by specifying the length prior. Please see length_prior.py

  ** REPORTING

  You can produce the predicted segmentation by using the SegmentationReporter in your inference configuration.
  This will produce one segmentation per line in {REPORT_PATH}.segment

  """
  yaml_tag = '!SegmentingSeqTransducer'

  @register_xnmt_handler
  @serializable_init
  def __init__(self, embed_encoder=bare(IdentitySeqTransducer),
                     segment_composer=bare(SeqTransducerComposer),
                     final_transducer=bare(BiLSTMSeqTransducer),
                     policy_learning=None,
                     length_prior=None,
                     eps_greedy=None,
                     sample_during_search=False,
                     reporter=None):
    self.embed_encoder = self.add_serializable_component("embed_encoder", embed_encoder, lambda: embed_encoder)
    self.segment_composer = self.add_serializable_component("segment_composer", segment_composer, lambda: segment_composer)
    self.final_transducer = self.add_serializable_component("final_transducer", final_transducer, lambda: final_transducer)
    self.policy_learning = self.add_serializable_component("policy_learning", policy_learning, lambda: policy_learning) if policy_learning is not None else None
    self.length_prior = self.add_serializable_component("length_prior", length_prior, lambda: length_prior) if length_prior is not None else None
    self.eps_greedy = self.add_serializable_component("eps_greedy", eps_greedy, lambda: eps_greedy) if eps_greedy is not None else None
    self.sample_during_search = sample_during_search
    self.reporter = reporter
    self.no_char_embed = issubclass(segment_composer.__class__, VocabBasedComposer)
    # Others
    self.segmenting_action = None
    self.compose_output = None
    self.segment_actions = None
    self.seg_size_unpadded = None
    self.src_sent = None
    self.reward = None
    self.train = None

  def shared_params(self):
    return [{".embed_encoder.hidden_dim",".policy_learning.policy_network.input_dim"},
            {".embed_encoder.hidden_dim",".policy_learning.baseline.input_dim"},
            {".segment_composer.hidden_dim", ".final_transducer.input_dim"}]

  def transduce(self, embed_sent: ExpressionSequence) -> List[ExpressionSequence]:
    batch_size = embed_sent[0].dim()[1]
    actions = self.sample_segmentation(embed_sent, batch_size)
    embeddings = dy.concatenate(embed_sent.expr_list, d=1)
    embeddings.value()
    #
    composed_words = []
    for i in range(batch_size):
      sequence = dy.pick_batch_elem(embeddings, i)
      # For each sampled segmentations
      lower_bound = 0
      for j, upper_bound in enumerate(actions[i]):
        if self.no_char_embed:
          char_sequence = []
        else:
          char_sequence = dy.pick_range(sequence, lower_bound, upper_bound+1, 1)
        composed_words.append((char_sequence, i, j, lower_bound, upper_bound+1))
        lower_bound = upper_bound+1
    outputs = self.segment_composer.compose(composed_words, batch_size)
    # Padding + return
    try:
      if self.length_prior:
        seg_size_unpadded = [len(outputs[i]) for i in range(batch_size)]
      sampled_sentence, segment_mask = self.pad(outputs)
      expr_seq = ExpressionSequence(expr_tensor=dy.concatenate_to_batch(sampled_sentence), mask=segment_mask)
      return self.final_transducer.transduce(expr_seq)
    finally:
      if self.length_prior:
        self.seg_size_unpadded = seg_size_unpadded
      self.compose_output = outputs
      self.segment_actions = actions
      if not self.train and self.is_reporting():
        if len(actions) == 1: # Support only AccuracyEvalTask
          self.report_sent_info({"segment_actions": actions})

  @handle_xnmt_event
  def on_calc_additional_loss(self, trg, generator, generator_loss):
    if self.policy_learning is None:
      return None
    reward = FactoredLossExpr()
    reward.add_loss("generator", -dy.inputTensor(generator_loss.value(), batched=True))
    if self.length_prior is not None:
      reward.add_loss('length_prior', self.length_prior.log_ll(self.seg_size_unpadded))
    reward_value = reward.value()
    if trg.batch_size() == 1:
      reward_value = [reward_value]
    reward_tensor = dy.inputTensor(reward_value, batched=True)
    ### Calculate losses
    try:
      return self.policy_learning.calc_loss(reward_tensor)
    finally:
      self.reward = reward
      if self.train and self.reporter is not None:
        self.reporter.report_process(self)

  @handle_xnmt_event
  def on_start_sent(self, src):
    self.src_sent = src
    self.segmenting_action = self.SegmentingAction.NONE

  @handle_xnmt_event
  def on_set_train(self, train):
    self.train = train
  
  def get_final_states(self) -> List[FinalTransducerState]:
    return self.final_transducer.get_final_states()
  
  def sample_segmentation(self, embed_sent, batch_size):
    if self.policy_learning is None: # Not Learning any policy
      if self.eps_greedy is not None:
        self.segmenting_action = self.SegmentingAction.PURE_SAMPLE
        actions = self.sample_from_prior()
      else:
        self.segmenting_action = self.SegmentingAction.GOLD
        actions = self.sample_from_gold()
    else: # Learning policy, with defined action or not
      predefined_actions = None
      seq_len = len(embed_sent)
      if self.eps_greedy and self.eps_greedy.is_triggered():
        self.segmenting_action = self.SegmentingAction.POLICY_SAMPLE
        predefined_actions = self.sparse_to_dense(self.sample_from_prior(), seq_len)
      else:
        self.segmenting_action = self.SegmentingAction.POLICY
      embed_encode = self.embed_encoder.transduce(embed_sent)
      actions = self.sample_from_policy(embed_encode, batch_size, predefined_actions)
    return actions 

  def sample_from_policy(self, encodings, batch_size, predefined_actions=None):
    from_argmax = not self.train and not self.sample_during_search
    actions = [[] for _ in range(batch_size)]
    mask = encodings.mask.np_arr if encodings.mask else None
    # Callback to ensure all samples are ended with </s> being segmented
    def ensure_end_segment(sample_batch, position):
      for i in range(len(sample_batch)):
        last_eos = self.src_sent[i].len_unpadded()
        if position >= last_eos:
          sample_batch[i] = 1
      return sample_batch
    # Loop through all items in the sequence
    for position, encoding in enumerate(encodings):
      # Sample from softmax if we have no predefined action
      predefined = predefined_actions[position] if predefined_actions is not None else None
      action = self.policy_learning.sample_action(encoding,
                                                  argmax=from_argmax,
                                                  sample_pp=lambda x: ensure_end_segment(x, position),
                                                  predefined_actions=predefined)
      # Appending the "1" position if it has valid flags
      for i in np.nonzero(action)[0]:
        if mask is None or mask[i][position] == 0:
          actions[i].append(position)
    return actions
 
  def sample_from_gold(self):
    return [sent.segment for sent in self.src_sent]

  def sample_from_prior(self):
    prior = self.eps_greedy.get_prior()
    batch_size = self.src_sent.batch_size()
    length_size = self.src_sent.sent_len()
    samples = prior.sample(batch_size, length_size)
    if issubclass(prior.__class__, GoldInputPrior):
      # Exception when the action is taken directly from the input
      actions = samples
    else:
      actions = []
      for src_sent, sample in zip(self.src_sent, samples):
        current, action = 0, []
        src_len = src_sent.len_unpadded()
        for j in range(len(sample)):
          current += sample[j]
          if current >= src_len:
            break
          action.append(current)
        if len(action) == 0 or action[-1] != src_len:
          action.append(src_len)
        actions.append(action)
    return actions

  def sparse_to_dense(self, actions, length):
    try:
      from xnmt.cython import xnmt_cython
    except:
      logger.error("BLEU evaluate fast requires xnmt cython installation step."
                   "please check the documentation.")
      raise RuntimeError()
    batch_dense = []
    for batch_action in actions:
      batch_dense.append(xnmt_cython.dense_from_sparse(batch_action, length))
    return np.array(batch_dense).transpose()

  def pad(self, outputs):
    # Padding
    max_col = max(len(xs) for xs in outputs)
    p0 = dy.vecInput(outputs[0][0].dim()[0][0])
    masks = np.zeros((len(outputs), max_col), dtype=int)
    modified = False
    ret = []
    for xs, mask in zip(outputs, masks):
      deficit = max_col - len(xs)
      if deficit > 0:
        xs.extend([p0 for _ in range(deficit)])
        mask[-deficit:] = 1
        modified = True
      ret.append(dy.concatenate_cols(xs))
    mask = Mask(masks) if modified else None
    return ret, mask
  
  class SegmentingAction(Enum):
    GOLD = 0
    POLICY = 1
    POLICY_SAMPLE = 2
    PURE_SAMPLE = 3
    NONE = 100

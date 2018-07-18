import numpy as np
import dynet as dy

from typing import List
from enum import Enum

from xnmt import logger
from xnmt.batcher import Mask
from xnmt.events import register_xnmt_handler, handle_xnmt_event
from xnmt.expression_sequence import ExpressionSequence
from xnmt.persistence import serializable_init, Serializable, Ref, bare
from xnmt.transducer import SeqTransducer, FinalTransducerState, IdentitySeqTransducer
from xnmt.loss import FactoredLossExpr
from xnmt.specialized_encoders.segmenting_encoder.priors import GoldInputPrior
from xnmt.reports import Reportable
from xnmt.lstm import BiLSTMSeqTransducer
from xnmt.specialized_encoders.segmenting_encoder.segmenting_composer import SeqTransducerComposer
from xnmt.compound_expr import CompoundSeqExpression

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
                     compute_report=Ref("exp_global.compute_report", default=False)):
    self.embed_encoder = self.add_serializable_component("embed_encoder", embed_encoder, lambda: embed_encoder)
    self.segment_composer = self.add_serializable_component("segment_composer", segment_composer, lambda: segment_composer)
    self.final_transducer = self.add_serializable_component("final_transducer", final_transducer, lambda: final_transducer)
    self.policy_learning = self.add_serializable_component("policy_learning", policy_learning, lambda: policy_learning) if policy_learning is not None else None
    self.length_prior = self.add_serializable_component("length_prior", length_prior, lambda: length_prior) if length_prior is not None else None
    self.eps_greedy = self.add_serializable_component("eps_greedy", eps_greedy, lambda: eps_greedy) if eps_greedy is not None else None
    self.sample_during_search = sample_during_search
    self.compute_report = compute_report

  def shared_params(self):
    return [{".embed_encoder.hidden_dim",".policy_learning.policy_network.input_dim"},
            {".embed_encoder.hidden_dim",".policy_learning.baseline.input_dim"},
            {".segment_composer.hidden_dim", ".final_transducer.input_dim"}]

  def transduce(self, embed_sent: ExpressionSequence) -> List[ExpressionSequence]:
    batch_size = embed_sent[0].dim()[1]
    actions = self.sample_segmentation(embed_sent, batch_size)
    sample_size = len(actions)
    embeddings = dy.concatenate(embed_sent.expr_list, d=1)
    embeddings.value()
    #
    composed_words = []
    for i in range(batch_size):
      sequence = dy.pick_batch_elem(embeddings, i)
      # For each sampled segmentations
      for j, sample in enumerate(actions):
        lower_bound = 0
        # Read every 'segment' decision
        for k, upper_bound in enumerate(sample[i]):
          char_sequence = dy.pick_range(sequence, lower_bound, upper_bound+1, 1)
          composed_words.append((dy.pick_range(sequence, lower_bound, upper_bound+1, 1), j, i, k, lower_bound, upper_bound+1))
          #self.segment_composer.set_word_boundary(lower_bound, upper_bound, self.src_sent[i])
          #composed = self.segment_composer.transduce(char_sequence)
          #outputs[j][i].append(composed)
          lower_bound = upper_bound+1
    outputs = self.segment_composer.compose(composed_words, sample_size, batch_size)
    # Padding + return
    try:
      if self.length_prior:
        seg_size_unpadded = [[len(outputs[i][j]) for j in range(batch_size)] for i in range(sample_size)]
      enc_outputs = []
      for batched_sampled_sentence in outputs:
        sampled_sentence, segment_mask = self.pad(batched_sampled_sentence)
        expr_seq = ExpressionSequence(expr_tensor=dy.concatenate_to_batch(sampled_sentence), mask=segment_mask)
        sent_context = self.final_transducer.transduce(expr_seq)
        self.final_states.append(self.final_transducer.get_final_states())
        enc_outputs.append(sent_context)
      return CompoundSeqExpression(enc_outputs)
    finally:
      if self.length_prior:
        self.seg_size_unpadded = seg_size_unpadded
      self.compose_output = outputs
      self.segment_actions = actions
      if not self.train and self.compute_report:
        self.add_sent_for_report({"segment_actions": actions})

  @handle_xnmt_event
  def on_calc_additional_loss(self, trg, generator, generator_loss):
    assert hasattr(generator, "losses"), "Must support multi sample encoder from generator."
    if self.policy_learning is None:
      return None
    ### Calculate reward
    rewards = []
    trg_counts = dy.inputTensor([t.len_unpadded() for t in trg], batched=True)
    # Iterate through all samples
    for i, (loss, actions) in enumerate(zip(generator.losses, self.compose_output)):
      reward = FactoredLossExpr()
      # Adding all reward from the translator
      for loss_key, loss_value in loss.get_nobackprop_loss().items():
        if loss_key == 'mle':
          reward.add_loss('mle', dy.cdiv(-loss_value, trg_counts))
        else:
          reward.add_loss(loss_key, -loss_value)
      if self.length_prior is not None:
        reward.add_loss('seg_lp', self.length_prior.log_ll(self.seg_size_unpadded[i]))
      rewards.append(dy.esum(list(reward.expr_factors.values())))
    ### Calculate losses    
    return self.policy_learning.calc_loss(rewards)

  @handle_xnmt_event
  def on_start_sent(self, src):
    self.src_sent = src
    self.final_states = []
    self.segmenting_action = self.SegmentingAction.NONE

  @handle_xnmt_event
  def on_set_train(self, train):
    self.train = train
  
  def get_final_states(self) -> List[List[FinalTransducerState]]:
    return self.final_states
  
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
    sample_size = 1 if from_argmax else self.policy_learning.sample
    sample_size = sample_size if predefined_actions is None else len(predefined_actions[0])
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
    return actions
 
  def sample_from_gold(self):
    return [[sent.segment for sent in self.src_sent]]

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
        if action[-1] != src_len:
          action.append(src_len)
        actions.append(action)
    # Return only 1 sample for each batc
    return [actions]

  def sparse_to_dense(self, actions, length):
    try:
      from xnmt.cython import xnmt_cython
    except:
      logger.error("BLEU evaluate fast requires xnmt cython installation step."
                   "please check the documentation.")
      raise RuntimeError()
    dense_actions = []
    for sample_actions in actions:
      batch_dense = []
      for batch_action in sample_actions:
        batch_dense.append(xnmt_cython.dense_from_sparse(batch_action, length))
      dense_actions.append(batch_dense)
    arr = np.array(dense_actions) # (sample, batch, length)
    return np.rollaxis(arr, 2)

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
  
  class SegmentingAction(Enum):
    GOLD = 0
    POLICY = 1
    POLICY_SAMPLE = 2
    PURE_SAMPLE = 3
    NONE = 100


import dynet as dy
import numpy as np

from enum import Enum

import xnmt.input_readers as input_readers
import xnmt.modelparts.embedders as embedders
import xnmt.modelparts.attenders as attenders
import xnmt.modelparts.decoders as decoders
import xnmt.inferences as inferences
import xnmt.transducers.recurrent as recurrent
import xnmt.transducers.base as transducers_base
import xnmt.events as events
import xnmt.event_trigger as event_trigger
import xnmt.expression_seqs as expr_seq
import xnmt.vocabs as vocabs
import xnmt.batchers as batchers
import xnmt.simultaneous.rewards as rewards

from xnmt.models.translators import DefaultTranslator, TranslatorOutput
from xnmt.persistence import bare, Serializable, serializable_init
from xnmt.reports import Reportable


class SimultaneousState(object):
  """
  The read/write state used to determine the state of the SimultaneousTranslator.
  """
  def __init__(self, model, encoder_state, context_state, output_embed, to_read=0, to_write=0, reset_attender=True):
    self.model = model
    self.encoder_state = encoder_state
    self.context_state = context_state
    self.output_embed = output_embed
    self.has_been_read = to_read
    self.has_been_written = to_write
    self.reset_attender = reset_attender
    
  def read(self, src):
    src_embed = self.model.src_embedder.embed(src[self.has_been_read])
    next_encoder_state = self.encoder_state.add_input(src_embed)
    return SimultaneousState(self.model, next_encoder_state, self.context_state,
                             self.output_embed, self.has_been_read+1, self.has_been_written, True)
  
  def calc_context(self, src_encoding, prev_word):
    # Generating h_t based on RNN(h_{t-1}, embed(e_{t-1}))
    if self.context_state is None:
      final_transducer_state = [transducers_base.FinalTransducerState(h, c) \
                              for h, c in zip(self.encoder_state.h(), self.encoder_state.c())]
      context_state = self.model.decoder.initial_state(final_transducer_state,
                                                       self.model.trg_embedder.embed(vocabs.Vocab.SS))
    else:
      context_state = self.model.decoder.add_input(self.context_state, self.model.trg_embedder.embed(prev_word))
    # Reset attender if there is a read action
    reset_attender = self.reset_attender
    if reset_attender:
      self.model.attender.init_sent(expr_seq.ExpressionSequence(expr_list=src_encoding))
      reset_attender = False
    # Calc context for decoding
    context_state.context = self.model.attender.calc_context(context_state.rnn_state.output())
    return SimultaneousState(self.model, self.encoder_state, context_state,
                             self.output_embed, self.has_been_read, self.has_been_written, reset_attender)
    
  def write(self, next_word):
    return SimultaneousState(self.model, self.encoder_state, self.context_state,
                             self.model.trg_embedder.embed(next_word), self.has_been_read,
                             self.has_been_written+1, self.reset_attender)


class SimultaneousTranslator(DefaultTranslator, Serializable, Reportable):
  yaml_tag = '!SimultaneousTranslator'
  
  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               src_reader: input_readers.InputReader,
               trg_reader: input_readers.InputReader,
               src_embedder: embedders.Embedder = bare(embedders.SimpleWordEmbedder),
               encoder: recurrent.UniLSTMSeqTransducer = bare(recurrent.UniLSTMSeqTransducer),
               attender: attenders.Attender = bare(attenders.MlpAttender),
               trg_embedder: embedders.Embedder = bare(embedders.SimpleWordEmbedder),
               decoder: decoders.Decoder = bare(decoders.AutoRegressiveDecoder),
               inference: inferences.AutoRegressiveInference = bare(inferences.AutoRegressiveInference),
               truncate_dec_batches: bool = False,
               policy_learning=None,
               freeze_decoder_param=False,
               max_generation=100) -> None:
    super().__init__(src_reader=src_reader,
                     trg_reader=trg_reader,
                     encoder=encoder,
                     attender=attender,
                     src_embedder=src_embedder,
                     trg_embedder=trg_embedder,
                     decoder=decoder,
                     inference=inference,
                     truncate_dec_batches=truncate_dec_batches)
    self.policy_learning = policy_learning
    self.actions = []
    self.outputs = []
    self.freeze_decoder_param = freeze_decoder_param
    self.max_generation = max_generation
    
  @events.handle_xnmt_event
  def on_set_train(self, train):
    self.train = train
    
  @events.handle_xnmt_event
  def on_start_sent(self, src_batch):
    self.src = src_batch
  
  def calc_nll(self, src_batch, trg_batch) -> dy.Expression:
    self.actions.clear()
    self.outputs.clear()
    event_trigger.start_sent(src_batch)
    batch_loss = []
    # For every item in the batch
    for src, trg in zip(src_batch, trg_batch):
      state = self.initial_state()
      src_len = src.sent_len()
      # Reading + Writing
      src_encoding = []
      loss_exprs = []
      now_action = []
      outputs = []
      prev_word = None
      while not self._stoping_criterions_met(state, trg, prev_word):
        action = self.next_action(state, src_len, len(src_encoding))
        if action == self.Action.READ:
          state = state.read(src)
          src_encoding.append(state.encoder_state.output())
        else:
          state = state.calc_context(src_encoding, prev_word)
          ground_truth = self._select_ground_truth(state, trg)
          loss_exprs.append(self.decoder.calc_loss(state.context_state, ground_truth))
          next_word = self._select_next_word(ground_truth)
          outputs.append(next_word)
          state = state.write(next_word)
          prev_word = next_word
        now_action.append(action.value)
      self.actions.append(now_action)
      self.outputs.append(outputs)
      # Accumulate loss
      batch_loss.append(dy.esum(loss_exprs))
    dy.forward(batch_loss)
    loss = dy.esum(batch_loss)
    return loss if not self.freeze_decoder_param else dy.nobackprop(loss)

  def _select_next_word(self, ref):
    if self.policy_learning is None:
      return ref
    else:
      return np.argmax(self.decoder.scorer.last_model_scores.npvalue())
    
  def _stoping_criterions_met(self, state, trg, prev_word):
    if self.policy_learning is None:
      return state.has_been_written >= trg.sent_len()
    else:
      return state.has_been_written >= self.max_generation or prev_word == vocabs.Vocab.ES
    
  def _select_ground_truth(self, state, trg):
    if trg.sent_len() <= state.has_been_written:
      return vocabs.Vocab.ES
    else:
      return trg[state.has_been_written]

  @events.handle_xnmt_event
  def on_calc_additional_loss(self, trg, generator, generator_loss):
    if self.policy_learning is None:
      return None
    reward = rewards.SimultaneousReward(self.src, trg, self.actions, self.outputs, self.trg_reader.vocab).calculate()
    return self.policy_learning.calc_loss(reward, only_final_reward=False)

  def next_action(self, state, src_len, enc_len):
    if self.policy_learning is None:
      if state.has_been_read < src_len:
        return self.Action.READ
      else:
        return self.Action.WRITE
    else:
      # Sanity Check here:
      force_action = [self.Action.READ.value] if enc_len == 0 else None # No writing at the beginning.
      force_action = [self.Action.WRITE.value] if enc_len == src_len else force_action # No reading at the end.
      # Compose inputs from 3 states
      encoder_state = state.encoder_state.output()
      enc_dim = encoder_state.dim()
      context_state = state.context_state.rnn_state.output() if state.context_state else dy.zeros(enc_dim[0], enc_dim[1])
      output_embed = state.output_embed if state.output_embed else dy.zeros(enc_dim[0], enc_dim[1])
      input_state = dy.concatenate([encoder_state, context_state, output_embed])
      # Sample / Calculate a single action
      action = self.policy_learning.sample_action(input_state, predefined_actions=force_action, argmax=not self.train)[0]
      return self.Action(action)
      
  class Action(Enum):
    READ = 0
    WRITE = 1
  
  def generate_search_output(self,
                             src,
                             search_strategy,
                             forced_trg_ids = None):
    """
    Takes in a batch of source sentences and outputs a list of search outputs.
    Args:
      src: The source sentences
      search_strategy: The strategy with which to perform the search
      forced_trg_ids: The target IDs to generate if performing forced decoding
    Returns:
      A list of search outputs including scores, etc.
    """

    if forced_trg_ids is not None:
      raise NotImplementedError("Forced decoding is not implemented for Simultaneous Translator.")
    event_trigger.start_sent(src)
    # Generating outputs
    search_outputs = search_strategy.generate_output(self, self.initial_state(),
                                                     src_length=None,
                                                     forced_trg_ids=None)
    return search_outputs
  
  def generate_one_step(self, current_word, current_state: SimultaneousState) -> TranslatorOutput:
    next_logsoftmax = self.decoder.calc_log_probs(current_state.context_state)
    return TranslatorOutput(current_state.context_state, next_logsoftmax, self.attender.get_last_attention())
  
  def initial_state(self):
    self.decoder.scorer.last_model_scores = None
    return SimultaneousState(self, self.encoder.initial_state(), None, None)
  

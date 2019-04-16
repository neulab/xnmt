import dynet as dy

from enum import Enum

import xnmt.input_readers as input_readers
import xnmt.modelparts.embedders as embedders
import xnmt.modelparts.attenders as attenders
import xnmt.modelparts.decoders as decoders
import xnmt.inferences as inferences
import xnmt.transducers.recurrent as recurrent
import xnmt.events as events
import xnmt.event_trigger as event_trigger
import xnmt.vocabs as vocabs
import xnmt.simultaneous.simult_rewards as rewards

from xnmt.models.translators.default import DefaultTranslator
from xnmt.persistence import bare, Serializable, serializable_init
from xnmt.reports import Reportable

from .simult_state import SimultaneousState

class SimultaneousTranslator(DefaultTranslator, Serializable, Reportable):
  yaml_tag = '!SimultaneousTranslator'
  
  class Action(Enum):
    READ = 0
    WRITE = 1
  
  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               src_reader: input_readers.InputReader,
               trg_reader: input_readers.InputReader,
               src_embedder: embedders.Embedder = bare(embedders.SimpleWordEmbedder),
               encoder: recurrent.UniLSTMSeqTransducer = bare(recurrent.UniLSTMSeqTransducer),
               attender: attenders.Attender = bare(attenders.MlpAttender),
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
                     decoder=decoder,
                     inference=inference,
                     truncate_dec_batches=truncate_dec_batches)
    self.policy_learning = policy_learning
    self.actions = []
    self.outputs = []
    self.freeze_decoder_param = freeze_decoder_param
    self.max_generation = max_generation
  
  def calc_nll(self, src_batch, trg_batch) -> dy.Expression:
    self.actions.clear()
    self.outputs.clear()
    event_trigger.start_sent(src_batch)
    batch_loss = []
    # For every item in the batch
    for src, trg in zip(src_batch, trg_batch):
      # Initial state with no read/write actions being taken
      current_dec_state, current_att_state = self._initial_state(src)
      src_len = src.sent_len()
      # Reading + Writing
      src_encoding = []
      loss_exprs = []
      now_action = []
      outputs = []
      # Simultaneous greedy search
      while not self._stoping_criterions_met(current_dec_state, trg):
        # Define action based on state
        action = self.next_action(current_dec_state, src_len, len(src_encoding))
        if action == self.Action.READ:
          # Reading + Encoding
          current_state = current_dec_state.read(src)
          src_encoding.append(current_dec_state.encoder_state.output())
        else:
          # Predicting next word
          current_state = current_dec_state.calc_context(src_encoding)
          current_output = self.add_input(current_dec_state.prev_written_word, current_dec_state, current_att_state)
          # Calculating losses
          ground_truth = self._select_ground_truth(current_dec_state, trg)
          loss_exprs.append(self.decoder.calc_loss(current_output.dec_state, ground_truth))
          # Use word from ref/model depeding on settings
          next_word = self._select_next_word(ground_truth, current_output.dec_state)
          # The produced words
          outputs.append(next_word)
          current_state = current_dec_state.write(next_word)
        now_action.append(action.value)
      self.actions.append(now_action)
      self.outputs.append(outputs)
      # Accumulate loss
      batch_loss.append(dy.esum(loss_exprs))
    dy.forward(batch_loss)
    loss = dy.esum(batch_loss)
    return loss if not self.freeze_decoder_param else dy.nobackprop(loss)

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
      context_state = state.context_state.as_vector() if state.context_state else dy.zeros(*enc_dim)
      output_embed = state.output_embed if state.output_embed else dy.zeros(*enc_dim)
      input_state = dy.concatenate([encoder_state, context_state, output_embed])
      # Sample / Calculate a single action
      action = self.policy_learning.sample_action(input_state, predefined_actions=force_action, argmax=not self.train)[0]
      return self.Action(action)

  @events.handle_xnmt_event
  def on_set_train(self, train):
    self.train = train
    
  @events.handle_xnmt_event
  def on_start_sent(self, src_batch):
    self.src = src_batch

  @events.handle_xnmt_event
  def on_calc_additional_loss(self, trg, generator, generator_loss):
    if self.policy_learning is None:
      return None
    reward = rewards.SimultaneousReward(self.src, trg, self.actions, self.outputs, self.trg_reader.vocab).calculate()
    return self.policy_learning.calc_loss(reward, only_final_reward=False)
  
  def _initial_state(self, src):
    att_state = self.attender.init_sent(self.encoder.initial_state())
    return SimultaneousState(self, self.encoder.initial_state(), None, None), att_state

  def _select_next_word(self, ref, state):
    if self.policy_learning is None:
      return ref
    else:
      best_words, _ = self.best_k(state, 1)
      return best_words[0]
    
  def _stoping_criterions_met(self, state, trg):
    if self.policy_learning is None:
      return state.has_been_written >= trg.sent_len()
    else:
      return state.has_been_written >= self.max_generation or \
             state.prev_written_word == vocabs.Vocab.ES
    
  def _select_ground_truth(self, state, trg):
    if trg.sent_len() <= state.has_been_written:
      return vocabs.Vocab.ES
    else:
      return trg[state.has_been_written]


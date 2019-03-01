import numbers

import xnmt.tensor_tools as tt
import xnmt.modelparts.decoders as decoders
import xnmt.transducers.recurrent as recurrent
import xnmt.transducers.base as transducers_base
import xnmt.expression_seqs as expr_seq
import xnmt.vocabs as vocabs

class SimultaneousState(decoders.AutoRegressiveDecoderState):
  """
  The read/write state used to determine the state of the SimultaneousTranslator.
  """
  def __init__(self,
               model,
               encoder_state: recurrent.UniLSTMState,
               context_state: decoders.AutoRegressiveDecoderState,
               output_embed: tt.Tensor,
               to_read:int = 0,
               to_write:int = 0,
               prev_written_word: numbers.Integral = None,
               reset_attender:bool = True):
    super().__init__(None, None)
    self.model = model
    self.encoder_state = encoder_state
    self.context_state = context_state
    self.output_embed = output_embed
    self.has_been_read = to_read
    self.has_been_written = to_write
    self.prev_written_word = prev_written_word
    self.reset_attender = reset_attender
    
  def read(self, src):
    src_embed = self.model.src_embedder.embed(src[self.has_been_read])
    next_encoder_state = self.encoder_state.add_input(src_embed)
    return SimultaneousState(self.model, next_encoder_state, self.context_state,
                             self.output_embed, self.has_been_read+1, self.has_been_written,
                             self.prev_written_word, True)
  
  def calc_context(self, src_encoding):
    # Generating h_t based on RNN(h_{t-1}, embed(e_{t-1}))
    if self.prev_written_word is None:
      final_transducer_state = [transducers_base.FinalTransducerState(h, c) \
                                for h, c in zip(self.encoder_state.h(), self.encoder_state.c())]
      context_state = self.model.decoder.initial_state(final_transducer_state,
                                                       vocabs.Vocab.SS)
    else:
      context_state = self.model.decoder.add_input(self.context_state, self.prev_written_word)
    # Reset attender if there is a read action
    reset_attender = self.reset_attender
    if reset_attender:
      self.model.attender.init_sent(expr_seq.ExpressionSequence(expr_list=src_encoding))
      reset_attender = False
    # Calc context for decoding
    context_state.context = self.model.attender.calc_context(context_state.rnn_state.output())
    return SimultaneousState(self.model, self.encoder_state, context_state,
                             self.output_embed, self.has_been_read, self.has_been_written,
                             self.prev_written_word,
                             reset_attender)
    
  def write(self, next_word):
    return SimultaneousState(self.model, self.encoder_state, self.context_state,
                             self.model.decoder.embedder.embed(next_word), self.has_been_read,
                             self.has_been_written+1,
                             next_word,
                             self.reset_attender)
 
  # These states are used for decoding
  def as_vector(self):
    return self.context_state.as_vector()
  
  @property
  def rnn_state(self):
    return self.context_state.rnn_state
  
  @property
  def context(self):
    return self.context_state.context

  @context.setter
  def context(self, value):
    self.context_state.context = value


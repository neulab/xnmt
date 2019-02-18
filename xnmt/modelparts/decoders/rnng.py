import numpy as np
import functools
import numbers
import dynet as dy
from typing import List

import xnmt.persistence as persistence
import xnmt.modelparts.decoders.base as decoders
import xnmt.modelparts.embedders as embedders
import xnmt.modelparts.scorers as scorers
import xnmt.modelparts.bridges as bridges
import xnmt.modelparts.transforms as transforms
import xnmt.transducers.recurrent as rec
import xnmt.batchers as batchers
import xnmt.sent as sent
import xnmt.input_readers as input_readers
import xnmt.transducers.char_compose.segmenting_composer as segment_composer
import xnmt.losses as losses

from xnmt.persistence import bare, Ref, Path


# RNNG Decoder state
class RNNGDecoderState(decoders.DecoderState):
  """A state holding all the information needed for RNNGDecoder
  
  Args:
    parser_state
    stack
    context
  """
  def __init__(self, stack, context, word_read=0, num_open_nt=0):
    self._stack = stack
    self._context = context
    self._word_read = word_read
    self._num_open_nt = num_open_nt

  # DecoderState interface
  def as_vector(self): return self.stack[-1].output()
  # Public accessible fields
  @property
  def context(self): return self._context
  @context.setter
  def context(self, value): self._context = value
  @property
  def stack(self): return self._stack
  @property
  def word_read(self): return self._word_read
  @property
  def num_open_nt(self): return self._num_open_nt
  

class RNNGStackState(object):
  def __init__(self, stack_content, stack_action=sent.RNNGAction.Type.NONE):
    self._content = stack_content
    self._action = stack_action
    
  def add_input(self, x):
    return RNNGStackState(self._content.add_input(x), self._action)
  
  def output(self):
    return self._content.output()
  
  @property
  def action(self):
    return self._action
  

class RNNGDecoder(decoders.Decoder, persistence.Serializable):
  yaml_tag = "!RNNGDecoder"
  RNNG_ACTION_SIZE = 6
  
  @persistence.serializable_init
  def __init__(self,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               head_composer: segment_composer.SequenceComposer = bare(segment_composer.DyerHeadComposer),
               rnn: rec.UniLSTMSeqTransducer = None,
               bridge: bridges.Bridge = bare(bridges.NoBridge),
               nt_embedder: embedders.SimpleWordEmbedder = None,
               edge_embedder: embedders.SimpleWordEmbedder = None,
               term_embedder: embedders.SimpleWordEmbedder = None,
               action_scorer: scorers.Softmax = None,
               nt_scorer: scorers.Softmax = None,
               term_scorer: scorers.Softmax = None,
               edge_scorer: scorers.Softmax = None,
               transform: transforms.Transform = bare(transforms.AuxNonLinear),
               ban_actions: List[numbers.Integral] = [1, 4],
               shift_from_enc: bool = True,
               max_open_nt: numbers.Integral = 100,
               graph_reader: input_readers.GraphReader = Ref(Path("model.trg_reader"))):
    self.input_dim = input_dim
    self.rnn = rnn
    self.bridge = bridge
    self.head_composer = head_composer
    self.nt_embedder = self.add_serializable_component("nt_embedder", nt_embedder,
                                                       lambda: embedders.SimpleWordEmbedder(
                                                         emb_dim=self.input_dim,
                                                         vocab_size=len(graph_reader.node_vocab)
                                                       ))
    self.edge_embedder = self.add_serializable_component("edge_embedder", edge_embedder,
                                                         lambda: embedders.SimpleWordEmbedder(
                                                           emb_dim=self.input_dim,
                                                           vocab_size=len(graph_reader.edge_vocab)
                                                         ))
    self.term_embedder = self.add_serializable_component("term_embedder", term_embedder,
                                                         lambda: embedders.SimpleWordEmbedder(
                                                           emb_dim=self.input_dim,
                                                           vocab_size=len(graph_reader.value_vocab)
                                                         ))
    self.transform = self.add_serializable_component("transform", transform, lambda: transform)
    self.action_scorer = self.add_serializable_component("action_scorer", action_scorer,
                                                         lambda: scorers.Softmax(
                                                         input_dim=input_dim,
                                                         vocab_size=RNNGDecoder.RNNG_ACTION_SIZE))
    self.nt_scorer = self.add_serializable_component("nt_scorer", nt_scorer,
                                                     lambda: scorers.Softmax(
                                                       input_dim=input_dim,
                                                       vocab_size=len(graph_reader.node_vocab)
                                                     ))
    self.term_scorer = self.add_serializable_component("term_scorer", term_scorer,
                                                       lambda: scorers.Softmax(
                                                         input_dim=input_dim,
                                                         vocab_size=len(graph_reader.value_vocab)
                                                       ))
    self.edge_scorer = self.add_serializable_component("edge_scorer", edge_scorer,
                                                       lambda: scorers.Softmax(
                                                         input_dim=input_dim,
                                                         vocab_size=len(graph_reader.edge_vocab)
                                                       ))
    self.ban_actions = ban_actions
    self.max_open_nt = max_open_nt
    self.shift_from_enc = shift_from_enc
  
  ### Decoder Interface
  def initial_state(self, enc_final_states, ss_expr):
    rnn_state = self.rnn.initial_state()
    rnn_s = self.bridge.decoder_init(enc_final_states)
    rnn_state = rnn_state.set_s(rnn_s)
    # This is important
    assert ss_expr.batch_size() == 1, "Currently, RNNG could not handle batch size > 1 in training and testing.\n" \
                                      "Please consider using autobatching."
    return RNNGDecoderState(stack=[RNNGStackState(rnn_state)], context=None)
 
  def add_input(self, dec_state: RNNGDecoderState, actions: List[sent.RNNGAction]):
    action = actions[0] if batchers.is_batched(actions) else actions
    action_type = action.action_type
    if action_type == sent.RNNGAction.Type.GEN:
      # Shifting the embedding of a word
      if self.shift_from_enc:
        # Feed in the decoder based on input string
        return self._perform_gen(dec_state, self.sent_enc[dec_state.word_read])
      else:
        # Feed in the decoder based on the previously generated output / oracle output
        return self._perform_gen(dec_state, self.term_embedder.embed(action.action_content))
    elif action_type == sent.RNNGAction.Type.REDUCE_LEFT or \
         action_type == sent.RNNGAction.Type.REDUCE_RIGHT:
      # Perform Reduce on Left direction or right direction
      return self._perform_reduce(dec_state,
                                  action == sent.RNNGAction.Type.REDUCE_LEFT,
                                  action.action_content)
    elif action_type == sent.RNNGAction.Type.NT:
      # Shifting the embedding of the NT's head
      return self._perform_nt(dec_state, action.action_content)
    elif action_type == sent.RNNGAction.Type.REDUCE_NT:
      return self._perform_reduce_nt(dec_state)
    elif action_type == sent.RNNGAction.Type.NONE:
      return dec_state
    else:
      raise  NotImplementedError("Unimplemented for action word:", action)
  
  def calc_loss(self, dec_state, ref_action):
    state = self._calc_transform(dec_state)
    action_batch = batchers.mark_as_batch([x.action_type.value for x in ref_action])
    loss = self.action_scorer.calc_loss(state, action_batch)
    # Aux Losses based on action content
    if ref_action == sent.RNNGAction.Type.NT:
      nt_batch = batchers.mark_as_batch([x.action_content for x in ref_action])
      loss += self.nt_scorer(state, nt_batch)
    elif ref_action == sent.RNNGAction.Type.GEN:
      term_batch = batchers.mark_as_batch([x.action_content for x in ref_action])
      loss += self.term_scorer(state, term_batch)
    # Total Loss
    return loss
  
  def best_k(self, dec_state, k, normalize_scores=False):
    final_state = self._calc_transform(dec_state)
    # p(a)
    action_logprob = self.action_scorer.calc_log_probs(final_state).npvalue()
    # p(nt|a == 'NT')
    action_logprob = np.array([action_logprob[i] for i in range(self.RNNG_ACTION_SIZE)])
    # RULING OUT INVALID ACTIONS
    rule_out = set(self.ban_actions)
    rule_out.add(sent.RNNGAction.Type.NONE.value)
    if len(dec_state.stack) <= 2:
      rule_out.add(sent.RNNGAction.Type.REDUCE_LEFT.value)
      rule_out.add(sent.RNNGAction.Type.REDUCE_RIGHT.value)
    if self.shift_from_enc:
      if dec_state.word_read >= len(self.sent_enc) :
        rule_out.add(sent.RNNGAction.Type.GEN.value)
    if dec_state.num_open_nt == 0:
      rule_out.add(sent.RNNGAction.Type.REDUCE_NT.value)
    if dec_state.num_open_nt > self.max_open_nt:
      rule_out.add(sent.RNNGAction.Type.NT.value)
    if len(rule_out) == len(action_logprob):
      rule_out.remove(sent.RNNGAction.Type.NONE.value)
    # Nulling out probability
    for action_value in rule_out:
      action_logprob[action_value] = -np.inf
    # Take out best action
    action_type = sent.RNNGAction.Type(np.argmax(action_logprob))
    best_score = action_logprob[action_type.value]
    if action_type == sent.RNNGAction.Type.NT:
      nt_logprob = self.nt_scorer.calc_log_probs(final_state).npvalue()
      return self._find_best_k(action_type, nt_logprob, k, best_score)
    elif action_type == sent.RNNGAction.Type.GEN:
      term_logprob = self.term_scorer.calc_log_probs(final_state).npvalue()
      return self._find_best_k(action_type, term_logprob, k, best_score)
    elif action_type == sent.RNNGAction.Type.REDUCE_LEFT or \
         action_type == sent.RNNGAction.Type.REDUCE_RIGHT:
      edge_logprob = self.edge_scorer.calc_log_probs(final_state).npvalue()
      return self._find_best_k(action_type, edge_logprob, k, best_score)
    else:
      best_action = sent.RNNGAction(action_type)
      return [best_action], [best_score]
  
  def _find_best_k(self, action_type, logprob, k, action_cond_prob):
    best_k = logprob.argsort()[max(-k, -len(logprob)+1):][::-1]
    actions = []
    scores = []
    for item in best_k:
      actions.append(sent.RNNGAction(action_type=action_type, action_content=item))
      scores.append(action_cond_prob + logprob[item])
    return actions, scores
    
  def sample(self, dec_state, n, temperature=1.0):
    raise NotImplementedError("Implement this function!")

  def init_sent(self, sent_enc):
    self.sent_enc = sent_enc

  def shared_params(self):
    return [{".embedder.emb_dim", ".rnn.input_dim"},
            {".input_dim", ".rnn.decoder_input_dim"},
            {".input_dim", ".transform.input_dim"},
            {".input_feeding", ".rnn.decoder_input_feeding"},
            {".rnn.layers", ".bridge.dec_layers"},
            {".rnn.hidden_dim", ".bridge.dec_dim"},
            {".rnn.hidden_dim", ".transform.aux_input_dim"},
            {".transform.output_dim", ".scorer.input_dim"}]

  @functools.lru_cache(maxsize=1)
  def eog_symbol(self):
    return sent.RNNGAction(sent.RNNGAction.Type.NONE)

  ### RNNGDecoder Modules
  def _calc_transform(self, dec_state):
    return self.transform.transform(dy.concatenate([dec_state.as_vector(), dec_state.context]))

  def _perform_gen(self, dec_state, word_encoding):
    h_i = dec_state.stack[-1].add_input(word_encoding)
    stack_i = [x for x in dec_state.stack] + [RNNGStackState(h_i)]
    return RNNGDecoderState(stack=stack_i,
                            context=dec_state.context,
                            word_read=dec_state.word_read+1,
                            num_open_nt=dec_state.num_open_nt)
  
  def _perform_reduce(self, dec_state, is_left, edge_id):
    children = dec_state.stack[-2:]
    if is_left: children = reversed(children)
    children = [child.output() for child in children]
    edge_embedding = self.edge_embedder.embed(batchers.mark_as_batch([edge_id]))
    children.append(edge_embedding)
    x_i = self.head_composer.transduce(children)
    h_i = dec_state.stack[-3].add_input(x_i)
    stack_i = dec_state.stack[:-2] + [RNNGStackState(h_i)]
    return RNNGDecoderState(stack=stack_i,
                            context=dec_state.context,
                            word_read=dec_state.word_read,
                            num_open_nt=dec_state.num_open_nt)
    
  def _perform_nt(self, dec_state, nt_id):
    x_i = self.nt_embedder.embed(nt_id)
    h_i = dec_state.stack[-1].add_input(x_i)
    stack_i = [x for x in dec_state.stack] + [RNNGStackState(h_i, sent.RNNGAction.Type.NT)]
    return RNNGDecoderState(stack=stack_i,
                            context=dec_state.context,
                            word_read=dec_state.word_read,
                            num_open_nt=dec_state.num_open_nt+1)
  
  def _perform_reduce_nt(self, dec_state):
    num_pop = 0
    while dec_state.stack[-(num_pop+1)].action != sent.RNNGAction.Type.REDUCE_NT:
      num_pop += 1
    children = dec_state.stack[-num_pop:]
    children = [child.output() for child in children]
    head_embedding = self.nt_embedder.embed(dec_state.stack[-(num_pop+1)].action.action_content)
    children.append(head_embedding)
    x_i = self.head_composer.transduce(children)
    h_i = dec_state.stack[-(num_pop+1)].add_input(x_i)
    stack_i = dec_state.stack[:-num_pop] + [RNNGStackState(h_i)]
    return RNNGDecoderState(stack=stack_i,
                            context=dec_state.context,
                            word_read=dec_state.word_read,
                            num_open_nt=dec_state.num_open_nt-1)

  def finish_generating(self, dec_output, dec_state):
    if type(dec_output) == np.ndarray or type(dec_output) == list:
      assert len(dec_output) == 1
    done = dec_state.num_open_nt == 0 and \
           len(dec_state.stack) == 2 and \
           dec_state.word_read == len(self.sent_enc)
    return [done]
 
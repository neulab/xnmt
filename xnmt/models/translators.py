import dynet as dy
import numpy as np
import collections
import itertools
from collections import namedtuple
from typing import Any, List, Optional, Sequence, Union

from xnmt.settings import settings
from xnmt import batchers, event_trigger, events, inferences, input_readers, losses, search_strategies, sent, vocabs
from xnmt.modelparts import attenders, decoders, embedders
from xnmt.models import base
from xnmt.transducers import recurrent, base as transducers_base
from xnmt.persistence import serializable_init, Serializable, bare

from xnmt.reports import Reportable

TranslatorOutput = namedtuple('TranslatorOutput', ['state', 'logsoftmax', 'attention'])

class AutoRegressiveTranslator(base.ConditionedModel, base.GeneratorModel):
  """
  A template class for auto-regressive translators.
  The core methods are calc_nll and generate / generate_one_step.
  The former is used during training, the latter for inference.
  Similarly during inference, a search strategy is used to generate an output sequence by repeatedly calling
  generate_one_step.
  """

  def calc_nll(self, src: Union[batchers.Batch, sent.Sentence], trg: Union[batchers.Batch, sent.Sentence]) -> dy.Expression:
    """
    Calculate the negative log likelihood, or similar value, of trg given src
    Args:
      src: The input
      trg: The output
    Return:
      The likelihood
    """
    raise NotImplementedError('must be implemented by subclasses')

  def generate(self,
               src: batchers.Batch,
               search_strategy: search_strategies.SearchStrategy,
               forced_trg_ids: batchers.Batch=None) -> Sequence[sent.Sentence]:
    raise NotImplementedError("must be implemented by subclasses")

  def generate_one_step(self, current_word: Any, current_state: decoders.AutoRegressiveDecoderState) -> TranslatorOutput:
    raise NotImplementedError("must be implemented by subclasses")

  def set_trg_vocab(self, trg_vocab: Optional[vocabs.Vocab] = None) -> None:
    """
    Set target vocab for generating outputs. If not specified, word IDs are generated instead.
    Args:
      trg_vocab: target vocab, or None to generate word IDs
    """
    self.trg_vocab = trg_vocab

  def get_nobp_state(self, state: decoders.AutoRegressiveDecoderState) -> dy.Expression:
    output_state = state.rnn_state.output()
    if type(output_state) == EnsembleListDelegate:
      for i in range(len(output_state)):
        output_state[i] = dy.nobackprop(output_state[i])
    else:
      output_state = dy.nobackprop(output_state)
    return output_state

class DefaultTranslator(AutoRegressiveTranslator, Serializable, Reportable):
  """
  A default translator based on attentional sequence-to-sequence models.
  Args:
    src_reader: A reader for the source side.
    trg_reader: A reader for the target side.
    src_embedder: A word embedder for the input language
    encoder: An encoder to generate encoded inputs
    attender: An attention module
    trg_embedder: A word embedder for the output language
    decoder: A decoder
    inference: The default inference strategy used for this model
  """

  yaml_tag = '!DefaultTranslator'

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               src_reader: input_readers.InputReader,
               trg_reader: input_readers.InputReader,
               src_embedder: embedders.Embedder = bare(embedders.SimpleWordEmbedder),
               encoder: transducers_base.SeqTransducer = bare(recurrent.BiLSTMSeqTransducer),
               attender: attenders.Attender = bare(attenders.MlpAttender),
               trg_embedder: embedders.Embedder = bare(embedders.SimpleWordEmbedder),
               decoder: decoders.Decoder = bare(decoders.AutoRegressiveDecoder),
               inference: inferences.AutoRegressiveInference = bare(inferences.AutoRegressiveInference),
               truncate_dec_batches: bool = False) -> None:
    super().__init__(src_reader=src_reader, trg_reader=trg_reader)
    self.src_embedder = src_embedder
    self.encoder = encoder
    self.attender = attender
    self.trg_embedder = trg_embedder
    self.decoder = decoder
    self.inference = inference
    self.truncate_dec_batches = truncate_dec_batches

  def shared_params(self):
    return [{".src_embedder.emb_dim", ".encoder.input_dim"},
            {".encoder.hidden_dim", ".attender.input_dim", ".decoder.input_dim"},
            {".attender.state_dim", ".decoder.rnn.hidden_dim"},
            {".trg_embedder.emb_dim", ".decoder.trg_embed_dim"}]

  def _encode_src(self, src: Union[batchers.Batch, sent.Sentence]):
    embeddings = self.src_embedder.embed_sent(src)
    encoding = self.encoder.transduce(embeddings)
    final_state = self.encoder.get_final_states()
    self.attender.init_sent(encoding)
    ss = batchers.mark_as_batch([vocabs.Vocab.SS] * src.batch_size()) if batchers.is_batched(src) else vocabs.Vocab.SS
    initial_state = self.decoder.initial_state(final_state, self.trg_embedder.embed(ss))
    return initial_state

  def calc_nll(self, src: Union[batchers.Batch, sent.Sentence], trg: Union[batchers.Batch, sent.Sentence]) -> dy.Expression:
    event_trigger.start_sent(src)
    if isinstance(src, batchers.CompoundBatch): src = src.batches[0]
    # Encode the sentence
    initial_state = self._encode_src(src)

    dec_state = initial_state
    trg_mask = trg.mask if batchers.is_batched(trg) else None
    cur_losses = []
    seq_len = trg.sent_len()

    if settings.CHECK_VALIDITY and batchers.is_batched(src):
      for j, single_trg in enumerate(trg):
        assert single_trg.sent_len() == seq_len # assert consistent length
        assert 1==len([i for i in range(seq_len) if (trg_mask is None or trg_mask.np_arr[j,i]==0) and single_trg[i]==vocabs.Vocab.ES]) # assert exactly one unmasked ES token

    input_word = None
    for i in range(seq_len):
      ref_word = DefaultTranslator._select_ref_words(trg, i, truncate_masked=self.truncate_dec_batches)
      if self.truncate_dec_batches and batchers.is_batched(ref_word):
        dec_state.rnn_state, ref_word = batchers.truncate_batches(dec_state.rnn_state, ref_word)

      if input_word is not None:
        dec_state = self.decoder.add_input(dec_state, self.trg_embedder.embed(input_word))
      rnn_output = dec_state.rnn_state.output()
      dec_state.context = self.attender.calc_context(rnn_output)
      word_loss = self.decoder.calc_loss(dec_state, ref_word)

      if not self.truncate_dec_batches and batchers.is_batched(src) and trg_mask is not None:
        word_loss = trg_mask.cmult_by_timestep_expr(word_loss, i, inverse=True)
      cur_losses.append(word_loss)
      input_word = ref_word

    if self.truncate_dec_batches:
      loss_expr = dy.esum([dy.sum_batches(wl) for wl in cur_losses])
    else:
      loss_expr = dy.esum(cur_losses)
    return loss_expr

  @staticmethod
  def _select_ref_words(sent, index, truncate_masked = False):
    if truncate_masked:
      mask = sent.mask if batchers.is_batched(sent) else None
      if not batchers.is_batched(sent):
        return sent[index]
      else:
        ret = []
        found_masked = False
        for (j, single_trg) in enumerate(sent):
          if mask is None or mask.np_arr[j, index] == 0 or np.sum(mask.np_arr[:, index]) == mask.np_arr.shape[0]:
            assert not found_masked, "sentences must be sorted by decreasing target length"
            ret.append(single_trg[index])
          else:
            found_masked = True
        return batchers.mark_as_batch(ret)
    else:
      if not batchers.is_batched(sent): return sent[index]
      else: return batchers.mark_as_batch([single_trg[index] for single_trg in sent])

  def generate_search_output(self,
                             src: batchers.Batch,
                             search_strategy: search_strategies.SearchStrategy,
                             forced_trg_ids: batchers.Batch=None) -> List[search_strategies.SearchOutput]:
    """
    Takes in a batch of source sentences and outputs a list of search outputs.
    Args:
      src: The source sentences
      search_strategy: The strategy with which to perform the search
      forced_trg_ids: The target IDs to generate if performing forced decoding
    Returns:
      A list of search outputs including scores, etc.
    """
    if src.batch_size()!=1:
      raise NotImplementedError("batched decoding not implemented for DefaultTranslator. "
                                "Specify inference batcher with batch size 1.")
    event_trigger.start_sent(src)
    if isinstance(src, batchers.CompoundBatch): src = src.batches[0]
    # Generating outputs
    cur_forced_trg = None
    src_sent = src[0]
    sent_mask = None
    if src.mask: sent_mask = batchers.Mask(np_arr=src.mask.np_arr[0:1])
    sent_batch = batchers.mark_as_batch([sent], mask=sent_mask)

    # Encode the sentence
    initial_state = self._encode_src(src)

    if forced_trg_ids is  not None: cur_forced_trg = forced_trg_ids[0]
    search_outputs = search_strategy.generate_output(self, initial_state,
                                                     src_length=[src_sent.sent_len()],
                                                     forced_trg_ids=cur_forced_trg)
    return search_outputs

  def generate(self,
               src: batchers.Batch,
               search_strategy: search_strategies.SearchStrategy,
               forced_trg_ids: batchers.Batch=None) -> Sequence[sent.Sentence]:
    """
    Takes in a batch of source sentences and outputs a list of search outputs.
    Args:
      src: The source sentences
      search_strategy: The strategy with which to perform the search
      forced_trg_ids: The target IDs to generate if performing forced decoding
    Returns:
      A list of search outputs including scores, etc.
    """
    assert src.batch_size() == 1
    search_outputs = self.generate_search_output(src, search_strategy, forced_trg_ids)
    if isinstance(src, batchers.CompoundBatch): src = src.batches[0]
    sorted_outputs = sorted(search_outputs, key=lambda x: x.score[0], reverse=True)
    assert len(sorted_outputs) >= 1
    outputs = []
    for curr_output in sorted_outputs:
      output_actions = [x for x in curr_output.word_ids[0]]
      attentions = [x for x in curr_output.attentions[0]]
      score = curr_output.score[0]
      out_sent = sent.SimpleSentence(idx=src[0].idx,
                                     words=output_actions,
                                     vocab=getattr(self.trg_reader, "vocab", None),
                                     output_procs=self.trg_reader.output_procs,
                                     score=score)
      if len(sorted_outputs) == 1:
        outputs.append(out_sent)
      else:
        outputs.append(sent.NbestSentence(base_sent=out_sent, nbest_id=src[0].idx))
    
    if self.is_reporting():
      attentions = np.concatenate([x.npvalue() for x in attentions], axis=1)
      self.report_sent_info({"attentions": attentions,
                             "src": src[0],
                             "output": outputs[0]})

    return outputs

  def generate_one_step(self, current_word: Any, current_state: decoders.AutoRegressiveDecoderState) -> TranslatorOutput:
    if current_word is not None:
      if type(current_word) == int:
        current_word = [current_word]
      if type(current_word) == list or type(current_word) == np.ndarray:
        current_word = batchers.mark_as_batch(current_word)
      current_word_embed = self.trg_embedder.embed(current_word)
      next_state = self.decoder.add_input(current_state, current_word_embed)
    else:
      next_state = current_state
    next_state.context = self.attender.calc_context(next_state.rnn_state.output())
    next_logsoftmax = self.decoder.calc_log_probs(next_state)
    return TranslatorOutput(next_state, next_logsoftmax, self.attender.get_last_attention())


class TransformerTranslator(AutoRegressiveTranslator, Serializable, Reportable):
  """
  A translator based on the transformer model.
  Args:
    src_reader (InputReader): A reader for the source side.
    src_embedder (Embedder): A word embedder for the input language
    encoder (TransformerEncoder): An encoder to generate encoded inputs
    trg_reader (InputReader): A reader for the target side.
    trg_embedder (Embedder): A word embedder for the output language
    decoder (TransformerDecoder): A decoder
    inference (AutoRegressiveInference): The default inference strategy used for this model
    input_dim (int):
  """

  yaml_tag = '!TransformerTranslator'

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self, src_reader, src_embedder, encoder, trg_reader, trg_embedder, decoder, inference=None, input_dim=512):
    super().__init__(src_reader=src_reader, trg_reader=trg_reader)
    self.src_embedder = src_embedder
    self.encoder = encoder
    self.trg_embedder = trg_embedder
    self.decoder = decoder
    self.input_dim = input_dim
    self.inference = inference
    self.scale_emb = self.input_dim ** 0.5
    self.max_input_len = 500
    self.initialize_position_encoding(self.max_input_len, input_dim)  # TODO: parametrize this

  def initialize_training_strategy(self, training_strategy):
    self.loss_calculator = training_strategy

  def make_attention_mask(self, source_block, target_block):
    mask = (target_block[:, None, :] <= 0) * (source_block[:, :, None] <= 0)
    # (batch, source_length, target_length)
    return mask

  def make_history_mask(self, block):
    batch, length = block.shape
    arange = np.arange(length)
    history_mask = (arange[None,] <= arange[:, None])[None,]
    history_mask = np.broadcast_to(history_mask, (batch, length, length))
    return history_mask

  def mask_embeddings(self, embeddings, mask):
    """
    We convert the embeddings of masked input sequence to zero vector
    """
    (embed_dim, _), _ = embeddings.dim()
    temp_mask = np.repeat(1. - mask[:, None, :], embed_dim, axis=1)
    temp_mask = dy.inputTensor(np.moveaxis(temp_mask, [1, 0, 2], [0, 2, 1]), batched=True)
    embeddings = dy.cmult(embeddings, temp_mask)
    return embeddings

  def initialize_position_encoding(self, length, n_units):
    # Implementation in the Google tensor2tensor repo
    channels = n_units
    position = np.arange(length, dtype='f')
    num_timescales = channels // 2
    log_timescale_increment = (np.log(10000. / 1.) / (float(num_timescales) - 1))
    inv_timescales = 1. * np.exp(np.arange(num_timescales).astype('f') * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)
    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.reshape(signal, [1, length, channels])
    self.position_encoding_block = np.transpose(signal, (0, 2, 1))

  def make_input_embedding(self, emb_block, length):
    if length > self.max_input_len:
      self.initialize_position_encoding(2 * length, self.input_dim)
      self.max_input_len = 2 * length
    emb_block = emb_block * self.scale_emb
    emb_block += dy.inputTensor(self.position_encoding_block[0, :, :length])
    return emb_block

  def sentence_block_embed(self, embed, x, mask):
    batch, length = x.shape
    x_mask = mask.reshape((batch * length,))
    _, units = embed.shape()  # According to updated Dynet
    e = dy.concatenate_cols([dy.zeros(units) if x_mask[j] == 1 else dy.lookup(embed, id_) for j, id_ in enumerate(x.reshape((batch * length,)))])
    e = dy.reshape(e, (units, length), batch_size=batch)
    return e

  def calc_loss(self, src, trg, infer_prediction=False):
    event_trigger.start_sent(src)
    if not batchers.is_batched(src):
      src = batchers.mark_as_batch([src])
    if not batchers.is_batched(trg):
      trg = batchers.mark_as_batch([trg])
    src_words = np.array([[vocabs.Vocab.SS] + x.words for x in src])
    batch_size, src_len = src_words.shape

    if isinstance(src.mask, type(None)):
      src_mask = np.zeros((batch_size, src_len), dtype=np.int)
    else:
      src_mask = np.concatenate([np.zeros((batch_size, 1), dtype=np.int), src.mask.np_arr.astype(np.int)], axis=1)

    src_embeddings = self.sentence_block_embed(self.src_embedder.embeddings, src_words, src_mask)
    src_embeddings = self.make_input_embedding(src_embeddings, src_len)

    trg_words = np.array(list(map(lambda x: [vocabs.Vocab.SS] + x.words[:-1], trg)))
    batch_size, trg_len = trg_words.shape

    if isinstance(trg.mask, type(None)):
      trg_mask = np.zeros((batch_size, trg_len), dtype=np.int)
    else:
      trg_mask = trg.mask.np_arr.astype(np.int)

    trg_embeddings = self.sentence_block_embed(self.trg_embedder.embeddings, trg_words, trg_mask)
    trg_embeddings = self.make_input_embedding(trg_embeddings, trg_len)

    xx_mask = self.make_attention_mask(src_mask, src_mask)
    xy_mask = self.make_attention_mask(trg_mask, src_mask)
    yy_mask = self.make_attention_mask(trg_mask, trg_mask)
    yy_mask *= self.make_history_mask(trg_mask)

    z_blocks = self.encoder.transduce(src_embeddings, xx_mask)
    h_block = self.decoder(trg_embeddings, z_blocks, xy_mask, yy_mask)

    if infer_prediction:
      y_len = h_block.dim()[0][1]
      last_col = dy.pick(h_block, dim=1, index=y_len - 1)
      logits = self.decoder.output(last_col)
      return logits

    ref_list = list(itertools.chain.from_iterable(map(lambda x: x.words, trg)))
    concat_t_block = (1 - trg_mask.ravel()).reshape(-1) * np.array(ref_list)
    loss = self.decoder.output_and_loss(h_block, concat_t_block)
    return losses.FactoredLossExpr({"mle": loss})

  def generate(self,
               src: batchers.Batch,
               search_strategy: search_strategies.SearchStrategy,
               forced_trg_ids: batchers.Batch=None) -> Sequence[sent.Sentence]:
    event_trigger.start_sent(src)
    if not batchers.is_batched(src):
      src = batchers.mark_as_batch([src])
    outputs = []

    trg = sent.SimpleSentence([0])

    if not batchers.is_batched(trg):
      trg = batchers.mark_as_batch([trg])

    output_actions = []
    score = 0.

    # TODO Fix this with generate_one_step and use the appropriate search_strategy
    self.max_len = 100 # This is a temporary hack
    for _ in range(self.max_len):
      dy.renew_cg(immediate_compute=settings.IMMEDIATE_COMPUTE, check_validity=settings.CHECK_VALIDITY)
      log_prob_tail = self.calc_loss(src, trg, loss_cal=None, infer_prediction=True)
      ys = np.argmax(log_prob_tail.npvalue(), axis=0).astype('i')
      if ys == vocabs.Vocab.ES:
        output_actions.append(ys)
        break
      output_actions.append(ys)
      trg = sent.SimpleSentence(words=output_actions + [0])
      if not batchers.is_batched(trg):
        trg = batchers.mark_as_batch([trg])

    # Append output to the outputs
    if hasattr(self, "trg_vocab") and self.trg_vocab is not None:
      outputs.append(sent.SimpleSentence(words=output_actions, vocab=self.trg_vocab))
    else:
      outputs.append((output_actions, score))

    return outputs

class EnsembleTranslator(AutoRegressiveTranslator, Serializable):
  """
  A translator that decodes from an ensemble of DefaultTranslator models.
  Args:
    models: A list of DefaultTranslator instances; for all models, their
      src_reader.vocab and trg_reader.vocab has to match (i.e., provide
      identical conversions to) those supplied to this class.
    src_reader: A reader for the source side.
    trg_reader: A reader for the target side.
    inference: The inference strategy used for this ensemble.
  """

  yaml_tag = '!EnsembleTranslator'

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               models: Sequence[DefaultTranslator],
               src_reader: input_readers.InputReader,
               trg_reader: input_readers.InputReader,
               inference: inferences.AutoRegressiveInference=bare(inferences.AutoRegressiveInference)) -> None:
    super().__init__(src_reader=src_reader, trg_reader=trg_reader)
    self.models = models
    self.inference = inference

    # perform checks to verify the models can logically be ensembled
    for i, model in enumerate(self.models):
      if hasattr(self.src_reader, "vocab") or hasattr(model.src_reader, "vocab"):
        assert self.src_reader.vocab.is_compatible(model.src_reader.vocab), \
          f"src_reader.vocab is not compatible with model {i}"
      assert self.trg_reader.vocab.is_compatible(model.trg_reader.vocab), \
        f"trg_reader.vocab is not compatible with model {i}"

    # proxy object used for generation, to avoid code duplication
    self._proxy = DefaultTranslator(
      self.src_reader,
      self.trg_reader,
      EnsembleListDelegate([model.src_embedder for model in self.models]),
      EnsembleListDelegate([model.encoder for model in self.models]),
      EnsembleListDelegate([model.attender for model in self.models]),
      EnsembleListDelegate([model.trg_embedder for model in self.models]),
      EnsembleDecoder([model.decoder for model in self.models])
    )

  def shared_params(self):
    shared = [params for model in self.models for params in model.shared_params()]
    return shared

  def set_trg_vocab(self, trg_vocab: Optional[vocabs.Vocab] = None) -> None:
    self._proxy.set_trg_vocab(trg_vocab=trg_vocab)

  def calc_nll(self, src: Union[batchers.Batch, sent.Sentence], trg: Union[batchers.Batch, sent.Sentence]) -> dy.Expression:
    sub_losses = collections.defaultdict(list)
    for model in self.models:
      for loss_name, loss in model.calc_nll(src, trg).expr_factors.items():
        sub_losses[loss_name].append(loss)
    model_loss = losses.FactoredLossExpr()
    for loss_name, losslist in sub_losses.items():
      # TODO: dy.average(losslist)  _or_  dy.esum(losslist) / len(self.models) ?
      #       -- might not be the same if not all models return all losses
      model_loss.add_loss(loss_name, dy.average(losslist))
    return model_loss

  def generate(self,
               src: batchers.Batch,
               search_strategy: search_strategies.SearchStrategy,
               forced_trg_ids: batchers.Batch=None) -> Sequence[sent.Sentence]:
    return self._proxy.generate(src, search_strategy, forced_trg_ids=forced_trg_ids)

class EnsembleListDelegate(object):
  """
  Auxiliary object to wrap a list of objects for ensembling.
  This class can wrap a list of objects that exist in parallel and do not need
  to interact with each other. The main functions of this class are:
  - All attribute access and function calls are delegated to the wrapped objects.
  - When wrapped objects return values, the list of all returned values is also
    wrapped in an EnsembleListDelegate object.
  - When EnsembleListDelegate objects are supplied as arguments, they are
    "unwrapped" so the i-th object receives the i-th element of the
    EnsembleListDelegate argument.
  """

  def __init__(self, objects):
    assert isinstance(objects, (tuple, list))
    self._objects = objects

  def __getitem__(self, key):
    return self._objects[key]

  def __setitem__(self, key, value):
    self._objects[key] = value

  def __iter__(self):
    return self._objects.__iter__()

  def __call__(self, *args, **kwargs):
    return self.__getattr__('__call__')(*args, **kwargs)

  def __len__(self):
    return len(self._objects)

  def __getattr__(self, attr):
    def unwrap(list_idx, args, kwargs):
      args = [arg if not isinstance(arg, EnsembleListDelegate) else arg[list_idx] \
              for arg in args]
      kwargs = {key: val if not isinstance(val, EnsembleListDelegate) else val[list_idx] \
                for key, val in kwargs.items()}
      return args, kwargs

    attrs = [getattr(obj, attr) for obj in self._objects]
    if callable(attrs[0]):
      def wrapper_func(*args, **kwargs):
        ret = []
        for i, attr_ in enumerate(attrs):
          args_i, kwargs_i = unwrap(i, args, kwargs)
          ret.append(attr_(*args_i, **kwargs_i))
        if all(val is None for val in ret):
          return None
        else:
          return EnsembleListDelegate(ret)
      return wrapper_func
    else:
      return EnsembleListDelegate(attrs)

  def __setattr__(self, attr, value):
    if not attr.startswith('_'):
      if isinstance(value, EnsembleListDelegate):
        for i, obj in enumerate(self._objects):
          setattr(obj, attr, value[i])
      else:
        for obj in self._objects:
          setattr(obj, attr, value)
    else:
      self.__dict__[attr] = value

  def __repr__(self):
    return "EnsembleListDelegate([" + ', '.join(repr(elem) for elem in self._objects) + "])"


class EnsembleDecoder(EnsembleListDelegate):
  """
  Auxiliary object to wrap a list of decoders for ensembling.
  This behaves like an EnsembleListDelegate, except that it overrides
  get_scores() to combine the individual decoder's scores.
  Currently only supports averaging.
  """
  def calc_log_probs(self, mlp_dec_states):
    scores = [obj.calc_log_probs(dec_state) for obj, dec_state in zip(self._objects, mlp_dec_states)]
    return dy.average(scores)

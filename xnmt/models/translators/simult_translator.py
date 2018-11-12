
import dynet as dy
import numpy as np

from typing import Union, List, Any

from xnmt import batchers, event_trigger, inferences, input_readers, search_strategies, sent
from xnmt.events import register_xnmt_handler
from xnmt.losses import FactoredLossExpr
from xnmt.models.translators import TranslatorOutput, AutoRegressiveTranslator
from xnmt.modelparts.attenders import Attender, MlpAttender
from xnmt.modelparts.embedders import Embedder, SimpleWordEmbedder
from xnmt.modelparts.decoders import Decoder, AutoRegressiveDecoder, AutoRegressiveDecoderState
from xnmt.persistence import serializable_init, Serializable, bare
from xnmt.reports import Reportable
from xnmt.settings import settings
from xnmt.transducers import base as transducers_base, UniLSTMSeqTransducer
from xnmt.vocabs import Vocab


class SimultaneuosTranslator(AutoRegressiveTranslator, Serializable, Reportable):
  yaml_tag = "!SimultaneuousTranslator"

  @register_xnmt_handler
  @serializable_init
  def __init__(self,
               src_reader: input_readers.InputReader,
               trg_reader: input_readers.InputReader,
               src_embedder: Embedder=bare(SimpleWordEmbedder),
               encoder: transducers_base.SeqTransducer=bare(UniLSTMSeqTransducer),
               attender: Attender=bare(MlpAttender),
               trg_embedder: Embedder=bare(SimpleWordEmbedder),
               decoder: Decoder=bare(AutoRegressiveDecoder),
               inference: inferences.AutoRegressiveInference=bare(inferences.AutoRegressiveInference),
               truncate_dec_batches:bool=False):
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

  def _encode_src(self,
                  src: Union[batchers.Batch, sent.Sentence]):
    embeddings = self.src_embedder.embed_sent(src)
    encoding = self.encoder.transduce(embeddings)
    final_state = self.encoder.get_final_states()
    self.attender.init_sent(encoding)
    ss = batchers.mark_as_batch([Vocab.SS] * src.batch_size()) if batchers.is_batched(src) else Vocab.SS
    initial_state = self.decoder.initial_state(final_state, self.trg_embedder.embed(ss))
    return initial_state

  def calc_nll(self,
               src: Union[batchers.Batch, sent.Sentence],
               trg: Union[batchers.Batch, sent.Sentence]) -> dy.Expression:
    event_trigger.start_sent(src)
    if isinstance(src, batchers.CompoundBatch): src = src.batches[0]
    # Encode the sentence
    initial_state = self._encode_src(src)

    dec_state = initial_state
    trg_mask = trg.mask
    losses = []
    seq_len = trg.sent_len()

    for 


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
      losses.append(word_loss)
      input_word = ref_word

    if self.truncate_dec_batches:
      loss_expr = dy.esum([dy.sum_batches(wl) for wl in losses])
    else:
      loss_expr = dy.esum(losses)
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
      return sent[index] if not batchers.is_batched(sent) else \
             batchers.mark_as_batch([single_trg[index] for single_trg in sent])

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
               forced_trg_ids: batchers.Batch=None):
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

  def generate_one_step(self,
                        current_word: Any,
                        current_state: AutoRegressiveDecoderState) -> TranslatorOutput:
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



import numpy as np
import itertools

from typing import Sequence

import xnmt
import xnmt.vocabs as vocabs
import xnmt.sent as sent
import xnmt.events as events
import xnmt.event_trigger as event_trigger
import xnmt.batchers as batchers
import xnmt.search_strategies as search_strategies
import xnmt.losses as losses

from xnmt.persistence import Serializable, serializable_init
from xnmt.reports import Reportable
from xnmt.settings import settings

from xnmt.models.translators import auto_regressive

if xnmt.backend_dynet:
  import dynet as dy

@xnmt.require_dynet
class TransformerTranslator(auto_regressive.AutoRegressiveTranslator, Serializable, Reportable):
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
    history_mask = arange[None,] <= arange[:, None][None,]
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
               search_strategy: search_strategies.SearchStrategy) -> Sequence[sent.Sentence]:
    event_trigger.start_sent(src)
    if not batchers.is_batched(src):
      src = batchers.mark_as_batch([src])
    outputs = []

    trg = sent.SimpleSentence([0])

    if not batchers.is_batched(trg):
      trg = batchers.mark_as_batch([trg])

    output_actions = []
    score = 0.

    # TODO Fix this with add_input / best_k and use the appropriate search_strategy
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


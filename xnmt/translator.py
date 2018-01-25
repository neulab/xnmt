from __future__ import division, generators

import six
import io
import xnmt.plot
import dynet as dy
import numpy as np
import itertools
# Reporting purposes
from lxml import etree
from simple_settings import settings
from xnmt.batcher import mark_as_batch, is_batched
from xnmt.events import register_xnmt_event_assign, handle_xnmt_event, register_handler
from xnmt.generator import GeneratorModel
from xnmt.input import SimpleSentenceInput
import xnmt.length_normalization
from xnmt.output import TextOutput
from xnmt.reports import Reportable
from xnmt.serialize.serializable import Serializable
from xnmt.search_strategy import BeamSearch, GreedySearch
import xnmt.serialize.serializer
from xnmt.serialize.tree_tools import Path
from xnmt.vocab import Vocab

class Translator(GeneratorModel):
  '''
  A template class implementing an end-to-end translator that can calculate a
  loss and generate translations.
  '''

  def calc_loss(self, src, trg):
    '''Calculate loss based on input-output pairs.

    :param src: The source, a sentence or a batch of sentences.
    :param trg: The target, a sentence or a batch of sentences.
    :returns: An expression representing the loss.
    '''
    raise NotImplementedError('calc_loss must be implemented for Translator subclasses')

  def set_trg_vocab(self, trg_vocab=None):
    """
    Set target vocab for generating outputs.

    :param trg_vocab: target vocab, or None to generate word ids
    """
    self.trg_vocab = trg_vocab

  def set_post_processor(self, post_processor):
    self.post_processor = post_processor

class DefaultTranslator(Translator, Serializable, Reportable):
  '''
  A default translator based on attentional sequence-to-sequence models.
  '''

  yaml_tag = u'!DefaultTranslator'

  def __init__(self, src_reader, trg_reader, src_embedder, encoder, attender, trg_embedder, decoder, inference=None):
    '''Constructor.

    :param src_reader: A reader for the source side.
    :param src_embedder: A word embedder for the input language
    :param encoder: An encoder to generate encoded inputs
    :param attender: An attention module
    :param trg_reader: A reader for the target side.
    :param trg_embedder: A word embedder for the output language
    :param decoder: A decoder
    :param inference: The default inference strategy used for this model
    '''
    register_handler(self)
    self.src_reader = src_reader
    self.trg_reader = trg_reader
    self.src_embedder = src_embedder
    self.encoder = encoder
    self.attender = attender
    self.trg_embedder = trg_embedder
    self.decoder = decoder
    self.inference = inference

  def shared_params(self):
    return [set([Path(".src_embedder.emb_dim"), Path(".encoder.input_dim")]),
            set([Path(".encoder.hidden_dim"), Path(".attender.input_dim"), Path(".decoder.input_dim")]),
            set([Path(".attender.state_dim"), Path(".decoder.lstm_dim")]),
            set([Path(".trg_embedder.emb_dim"), Path(".decoder.trg_embed_dim")])]

  def initialize_generator(self, **kwargs):
    if kwargs.get("len_norm_type", None) is None:
      len_norm = xnmt.length_normalization.NoNormalization()
    else:
      len_norm = xnmt.serialize.serializer.YamlSerializer().initialize_if_needed(kwargs["len_norm_type"])
    search_args = {}
    if kwargs.get("max_len", None) is not None: search_args["max_len"] = kwargs["max_len"]
    if kwargs.get("beam", None) is None:
      self.search_strategy = GreedySearch(**search_args)
    else:
      search_args["beam_size"] = kwargs.get("beam", 1)
      search_args["len_norm"] = len_norm
      self.search_strategy = BeamSearch(**search_args)
    self.report_path = kwargs.get("report_path", None)
    self.report_type = kwargs.get("report_type", None)

  def calc_loss(self, src, trg, loss_calculator):
    """
    :param src: source sequence (unbatched, or batched + padded)
    :param trg: target sequence (unbatched, or batched + padded); losses will be accumulated only if trg_mask[batch,pos]==0, or no mask is set
    :param loss_calculator:
    :returns: (possibly batched) loss expression
    """
    self.start_sent(src)
    embeddings = self.src_embedder.embed_sent(src)
    encodings = self.encoder(embeddings)
    self.attender.init_sent(encodings)
    # Initialize the hidden state from the encoder
    ss = mark_as_batch([Vocab.SS] * len(src)) if is_batched(src) else Vocab.SS
    dec_state = self.decoder.initial_state(self.encoder.get_final_states(), self.trg_embedder.embed(ss))
    return loss_calculator(self, dec_state, src, trg)

  def generate(self, src, idx, src_mask=None, forced_trg_ids=None):
    if not xnmt.batcher.is_batched(src):
      src = xnmt.batcher.mark_as_batch([src])
    else:
      assert src_mask is not None
    outputs = []
    for sents in src:
      self.start_sent(src)
      embeddings = self.src_embedder.embed_sent(src)
      encodings = self.encoder(embeddings)
      self.attender.init_sent(encodings)
      ss = mark_as_batch([Vocab.SS] * len(src)) if is_batched(src) else Vocab.SS
      dec_state = self.decoder.initial_state(self.encoder.get_final_states(), self.trg_embedder.embed(ss))
      output_actions, score = self.search_strategy.generate_output(self.decoder, self.attender, self.trg_embedder, dec_state, src_length=len(sents), forced_trg_ids=forced_trg_ids)
      # In case of reporting
      if self.report_path is not None:
        src_words = [self.reporting_src_vocab[w] for w in sents]
        trg_words = [self.trg_vocab[w] for w in output_actions]
        attentions = self.attender.attention_vecs
        self.set_report_input(idx, src_words, trg_words, attentions)
        self.set_report_resource("src_words", src_words)
        self.set_report_path('{}.{}'.format(self.report_path, str(idx)))
        self.generate_report(self.report_type)
      # Append output to the outputs
      outputs.append(TextOutput(actions=output_actions,
                                vocab=self.trg_vocab if hasattr(self, "trg_vocab") else None,
                                score=score))
    return outputs

  def set_reporting_src_vocab(self, src_vocab):
    """
    Sets source vocab for reporting purposes.
    """
    self.reporting_src_vocab = src_vocab

  @register_xnmt_event_assign
  def html_report(self, context=None):
    assert(context is None)
    idx, src, trg, att = self.get_report_input()
    path_to_report = self.get_report_path()
    html = etree.Element('html')
    head = etree.SubElement(html, 'head')
    title = etree.SubElement(head, 'title')
    body = etree.SubElement(html, 'body')
    report = etree.SubElement(body, 'h1')
    if idx is not None:
      title.text = report.text = 'Translation Report for Sentence %d' % (idx)
    else:
      title.text = report.text = 'Translation Report'
    main_content = etree.SubElement(body, 'div', name='main_content')

    # Generating main content
    captions = [u"Source Words", u"Target Words"]
    inputs = [src, trg]
    for caption, inp in six.moves.zip(captions, inputs):
      if inp is None: continue
      sent = ' '.join(inp)
      p = etree.SubElement(main_content, 'p')
      p.text = u"{}: {}".format(caption, sent)

    # Generating attention
    if not any([src is None, trg is None, att is None]):
      attention = etree.SubElement(main_content, 'p')
      att_text = etree.SubElement(attention, 'b')
      att_text.text = "Attention:"
      etree.SubElement(attention, 'br')
      attention_file = u"{}.attention.png".format(path_to_report)

      if type(att) == dy.Expression:
        attentions = att.npvalue()
      elif type(att) == list:
        attentions = np.concatenate([x.npvalue() for x in att], axis=1)
      elif type(att) != np.ndarray:
        raise RuntimeError("Illegal type for attentions in translator report: {}".format(type(attentions)))
      xnmt.plot.plot_attention(src, trg, attentions, file_name = attention_file)

    # return the parent context to be used as child context
    return html

  @handle_xnmt_event
  def on_file_report(self):
    idx, src, trg, att = self.get_report_input()
    path_to_report = self.get_report_path()

    # Writing attention matrix to text file
    if not any([src is None, trg is None, att is None]):
      attention_file = u"{}.attention.txt".format(path_to_report)

      if type(att) == dy.Expression:
        attentions = att.npvalue()
      elif type(att) == list:
        attentions = np.concatenate([x.npvalue() for x in att], axis=1)
      elif type(att) != np.ndarray:
        raise RuntimeError("Illegal type for attentions in translator report: {}".format(type(attentions)))
      assert attentions.shape == (len(src), len(trg))

      # Not sure why elsewhere in the code the attention matrix is line-indexed by src
      # and column-indexed by trg (the convention seems to be the other way around,
      # eg. in [Bahdanau, 2014] https://arxiv.org/pdf/1409.0473.pdf)
      # I expect attention matrices to sum to 1.0 line-wise.
      att_transpose = np.transpose(attentions)
      with io.open(attention_file, 'w', encoding='utf-8') as f:
        f.write('\t' + '\t'.join(src) + '\n')
        for i in range(len(trg)):
          coeffs = '\t'.join(map(str, list(att_transpose[i])))
          f.write('{}\t{}\n'.format(trg[i], coeffs))


class TransformerTranslator(Translator, Serializable, Reportable):
  yaml_tag = u'!TransformerTranslator'

  def __init__(self, src_reader, src_embedder, encoder, trg_reader, trg_embedder, decoder, inference=None, input_dim=512):
    '''Constructor.
    :param src_embedder: A word embedder for the input language
    :param encoder: An encoder to generate encoded inputs
    :param attender: An attention module
    :param trg_embedder: A word embedder for the output language
    :param decoder: A decoder
    '''
    register_handler(self)
    self.src_reader = src_reader
    self.src_embedder = src_embedder
    self.encoder = encoder
    self.trg_reader = trg_reader
    self.trg_embedder = trg_embedder
    self.decoder = decoder
    self.input_dim = input_dim
    self.inference = inference
    self.scale_emb = self.input_dim ** 0.5
    self.max_input_len = 500
    self.initialize_position_encoding(self.max_input_len, input_dim)  # TODO: parametrize this

  def initialize_generator(self, **kwargs):
    if kwargs.get("len_norm_type", None) is None:
      len_norm = xnmt.length_normalization.NoNormalization()
    else:
      len_norm = xnmt.serialize.serializer.YamlSerializer().initialize_object(kwargs["len_norm_type"])
    search_args = {}
    if kwargs.get("max_len", None) is not None:
      search_args["max_len"] = kwargs["max_len"]
      self.max_len = kwargs.get("max_len", 50)
    if kwargs.get("beam", None) is None:
      self.search_strategy = GreedySearch(**search_args)
    else:
      search_args["beam_size"] = kwargs.get("beam", 1)
      search_args["len_norm"] = len_norm
      # self.search_strategy = TransformerBeamSearch(**search_args)
    self.report_path = kwargs.get("report_path", None)
    self.report_type = kwargs.get("report_type", None)

  def initialize_training_strategy(self, training_strategy):
    self.loss_calculator = training_strategy

  def set_reporting_src_vocab(self, src_vocab):
    """
    Sets source vocab for reporting purposes.
    """
    self.reporting_src_vocab = src_vocab

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

  def calc_loss(self, src, trg, loss_cal=None, infer_prediction=False):
    if not xnmt.batcher.is_batched(src):
      src = xnmt.batcher.mark_as_batch([src])
    if not xnmt.batcher.is_batched(trg):
      trg = xnmt.batcher.mark_as_batch([trg])
    src_words = np.array([[Vocab.SS] + x.words for x in src])
    batch_size, src_len = src_words.shape

    if isinstance(src.mask, type(None)):
      src_mask = np.zeros((batch_size, src_len), dtype=np.int)
    else:
      src_mask = np.concatenate([np.zeros((batch_size, 1), dtype=np.int), src.mask.np_arr.astype(np.int)], axis=1)

    src_embeddings = self.sentence_block_embed(self.src_embedder.embeddings, src_words, src_mask)
    src_embeddings = self.make_input_embedding(src_embeddings, src_len)

    trg_words = np.array(list(map(lambda x: [Vocab.SS] + x.words[:-1], trg)))
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

    z_blocks = self.encoder(src_embeddings, xx_mask)
    h_block = self.decoder(trg_embeddings, z_blocks, xy_mask, yy_mask)

    if infer_prediction:
      y_len = h_block.dim()[0][1]
      last_col = dy.pick(h_block, dim=1, index=y_len - 1)
      logits = self.decoder.output(last_col)
      return logits

    ref_list = list(itertools.chain.from_iterable(map(lambda x: x.words, trg)))
    concat_t_block = (1 - trg_mask.ravel()).reshape(-1) * np.array(ref_list)
    loss = self.decoder.output_and_loss(h_block, concat_t_block)
    return loss

  def generate(self, src, idx, src_mask=None, forced_trg_ids=None):
    if not xnmt.batcher.is_batched(src):
      src = xnmt.batcher.mark_as_batch([src])
    else:
      assert src_mask is not None
    outputs = []

    trg = SimpleSentenceInput([0])

    if not xnmt.batcher.is_batched(trg):
      trg = xnmt.batcher.mark_as_batch([trg])

    output_actions = []
    score = 0.

    for _ in range(self.max_len):
      dy.renew_cg(immediate_compute=settings.IMMEDIATE_COMPUTE, check_validity=settings.CHECK_VALIDITY)
      log_prob_tail = self.calc_loss(src, trg, loss_cal=None, infer_prediction=True)
      ys = np.argmax(log_prob_tail.npvalue(), axis=0).astype('i')
      if ys == Vocab.ES:
        output_actions.append(ys)
        break
      output_actions.append(ys)
      trg = SimpleSentenceInput(output_actions + [0])
      if not xnmt.batcher.is_batched(trg):
        trg = xnmt.batcher.mark_as_batch([trg])

    # In case of reporting
    sents = src[0]
    if self.report_path is not None:
      src_words = [self.reporting_src_vocab[w] for w in sents]
      trg_words = [self.trg_vocab[w] for w in output_actions]
      self.set_report_input(idx, src_words, trg_words)
      self.set_report_resource("src_words", src_words)
      self.set_report_path('{}.{}'.format(self.report_path, str(idx)))
      self.generate_report(self.report_type)

    # Append output to the outputs
    if hasattr(self, "trg_vocab") and self.trg_vocab is not None:
      outputs.append(TextOutput(output_actions, self.trg_vocab))
    else:
      outputs.append((output_actions, score))

    return outputs

from __future__ import division, generators

import six
import plot
import os
import dynet as dy
import numpy as np

import xnmt.length_normalization
import xnmt.batcher

from xnmt.vocab import Vocab
from xnmt.serializer import Serializable, DependentInitParam
from xnmt.search_strategy import BeamSearch, GreedySearch
from xnmt.output import TextOutput
from xnmt.model import GeneratorModel
from xnmt.reports import Reportable
from xnmt.decorators import recursive_assign, recursive
import xnmt.serializer
from xnmt.batcher import mark_as_batch, is_batched
from xnmt.vocab import Vocab

# Reporting purposes
from lxml import etree

class Translator(GeneratorModel):
  '''
  A template class implementing an end-to-end translator that can calculate a
  loss and generate translations.
  '''

  def calc_loss(self, src, trg, src_mask=None, trg_mask=None):
    '''Calculate loss based on input-output pairs.

    :param src: The source, a sentence or a batch of sentences.
    :param trg: The target, a sentence or a batch of sentences.
    :param src_mask: A numpy array specifying the masking over the source, where rows are time steps, and columns are batch IDs.
    :param trg_mask: A numpy array specifying the masking over the target, where rows are time steps, and columns are batch IDs.
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

  def __init__(self, src_embedder, encoder, attender, trg_embedder, decoder):
    '''Constructor.

    :param src_embedder: A word embedder for the input language
    :param encoder: An encoder to generate encoded inputs
    :param attender: An attention module
    :param trg_embedder: A word embedder for the output language
    :param decoder: A decoder
    '''
    self.src_embedder = src_embedder
    self.encoder = encoder
    self.attender = attender
    self.trg_embedder = trg_embedder
    self.decoder = decoder

    self.register_hier_child(self.encoder)
    self.register_hier_child(self.decoder)
    self.register_hier_child(self.src_embedder)
    self.register_hier_child(self.trg_embedder)

  def shared_params(self):
    return [set(["src_embedder.emb_dim", "encoder.input_dim"]),
            set(["encoder.hidden_dim", "attender.input_dim", "decoder.input_dim"]),
            set(["attender.state_dim", "decoder.lstm_dim"]),
            set(["trg_embedder.emb_dim", "decoder.trg_embed_dim"])]

  def dependent_init_params(self):
    return [DependentInitParam(param_descr="src_embedder.vocab_size", value_fct=lambda: self.context.corpus_parser.src_reader.vocab_size()),
            DependentInitParam(param_descr="decoder.vocab_size", value_fct=lambda: self.context.corpus_parser.trg_reader.vocab_size()),
            DependentInitParam(param_descr="trg_embedder.vocab_size", value_fct=lambda: self.context.corpus_parser.trg_reader.vocab_size())]

  def initialize_generator(self, **kwargs):
    if kwargs.get("len_norm_type", None) is None: 
      len_norm = xnmt.length_normalization.NoNormalization()
    else:
      len_norm = xnmt.serializer.YamlSerializer().initialize_object(kwargs["len_norm_type"])
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

  def calc_loss(self, src, trg, src_mask=None, trg_mask=None, info=None):
    """
    :param src: source sequence (unbatched, or batched + padded)
    :param trg: target sequence (unbatched, or batched + padded)
    :param src_mask: binary mask denoting src padding, passed to embedder
    :param trg_mask: binary mask denoting trg padding; losses will be accumulated only if trg_mask[batch,pos]==0
    :returns: (possibly batched) loss expression
    """
    self.start_sent()
    embeddings = self.src_embedder.embed_sent(src, mask=src_mask)
    encodings = self.encoder.transduce(embeddings)
    self.attender.init_sent(encodings)
    # Initialize the hidden state from the encoder
    ss = mark_as_batch([Vocab.SS] * len(src)) if is_batched(src) else Vocab.SS
    self.decoder.initialize(self.encoder.get_final_states(), self.trg_embedder.embed(ss))
    losses = []

    seq_len = len(trg[0]) if xnmt.batcher.is_batched(src) else len(trg)
    if xnmt.batcher.is_batched(src):
      for j, single_trg in enumerate(trg):
        assert len(single_trg) == seq_len # assert consistent length
        assert 1==len([i for i in range(seq_len) if (trg_mask is None or trg_mask[j,i]==0) and single_trg[i]==Vocab.ES]) # assert exactly one unmasked ES token
    for i in range(seq_len):
      ref_word = trg[i] if not xnmt.batcher.is_batched(src) \
                      else xnmt.batcher.mark_as_batch([single_trg[i] for single_trg in trg])
 
      context = self.attender.calc_context(self.decoder.state.output())
      word_loss = self.decoder.calc_loss(context, ref_word)
      if xnmt.batcher.is_batched(src) and trg_mask is not None:
        mask_exp = dy.inputTensor((1.0 - trg_mask)[:,i:i+1].transpose(),batched=True)
        word_loss = word_loss * mask_exp
      losses.append(word_loss)
      if i < seq_len-1:
        self.decoder.add_input(self.trg_embedder.embed(ref_word))

    return dy.esum(losses)

  def generate(self, src, idx, src_mask=None, forced_trg_ids=None):
    if not xnmt.batcher.is_batched(src):
      src = xnmt.batcher.mark_as_batch([src])
    else:
      assert src_mask is not None
    outputs = []
    for sents in src:
      self.start_sent()
      embeddings = self.src_embedder.embed_sent(src, mask=src_mask)
      encodings = self.encoder.transduce(embeddings)
      self.attender.init_sent(encodings)
      ss = mark_as_batch([Vocab.SS] * len(src)) if is_batched(src) else Vocab.SS
      self.decoder.initialize(self.encoder.get_final_states(), self.trg_embedder.embed(ss))
      output_actions, score = self.search_strategy.generate_output(self.decoder, self.attender, self.trg_embedder, src_length=len(sents), forced_trg_ids=forced_trg_ids)
      # In case of reporting
      if self.report_path is not None:
        src_words = [self.reporting_src_vocab[w] for w in sents]
        trg_words = [self.trg_vocab[w] for w in output_actions[1:]]
        attentions = self.attender.attention_vecs
        self.set_report_input(idx, src_words, trg_words, attentions)
        self.set_report_resource("src_words", src_words)
        self.set_report_path('{}.{}'.format(self.report_path, str(idx)))
        self.generate_report(self.report_type)
      # Append output to the outputs
      if hasattr(self, "trg_vocab") and self.trg_vocab is not None:
        outputs.append(TextOutput(output_actions, self.trg_vocab))
      else:
        outputs.append((output_actions, score))
    return outputs

  def set_reporting_src_vocab(self, src_vocab):
    """
    Sets source vocab for reporting purposes.
    """
    self.reporting_src_vocab = src_vocab

  @recursive_assign
  def html_report(self, context=None):
    assert(context is None)
    idx, src, trg, att = self.get_report_input()
    path_to_report = self.get_report_path()
    filename_of_report = os.path.basename(path_to_report)
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
      att_mtr = etree.SubElement(attention, 'img', src="{}.attention.png".format(filename_of_report))
      attention_file = u"{}.attention.png".format(path_to_report)

      if type(att) == dy.Expression:
        attentions = att.npvalue()
      elif type(att) == list:
        attentions = np.concatenate([x.npvalue() for x in att], axis=1)
      elif type(att) != np.ndarray:
        raise RuntimeError("Illegal type for attentions in translator report: {}".format(type(attentions)))
      plot.plot_attention(src, trg, attentions, file_name = attention_file)

    # return the parent context to be used as child context
    return html

  @recursive
  def file_report(self):
    pass


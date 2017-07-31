from __future__ import division, generators

import dynet as dy
import numpy as np
import length_normalization
import decorators
import batcher
import six
import plot
import os

from vocab import Vocab
from serializer import Serializable, DependentInitParam
from search_strategy import BeamSearch
from embedder import SimpleWordEmbedder
from decoder import MlpSoftmaxDecoder
from output import TextOutput
from model import HierarchicalModel, GeneratorModel
from reports import HTMLReportable
from decorators import recursive_assign, recursive

# Reporting purposes
from lxml import etree

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

  def calc_reinforce_loss(self, reward):
    ''' Calculate reinforce loss based on the reward
    :param reward: The default is log likelihood (-1 * calc_loss).
    '''
    return None

  def set_vocabs(self, src_vocab, trg_vocab):
    self.src_vocab = src_vocab
    self.trg_vocab = trg_vocab

  def set_post_processor(self, post_processor):
    self.post_processor = post_processor

class DefaultTranslator(Translator, Serializable, HTMLReportable):
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
    super(DefaultTranslator, self).__init__()
    self.src_embedder = src_embedder
    self.encoder = encoder
    self.attender = attender
    self.trg_embedder = trg_embedder
    self.decoder = decoder

    self.register_hier_child(self.encoder)
    self.register_hier_child(self.decoder)

  def shared_params(self):
    return [set(["src_embedder.emb_dim", "encoder.input_dim"]),
            # TODO: encoder.hidden_dim may not always exist (e.g. for CNN encoders), need to deal with that case    
            set(["encoder.hidden_dim", "attender.input_dim", "decoder.input_dim"]),
            set(["attender.state_dim", "decoder.lstm_dim"]),
            set(["trg_embedder.emb_dim", "decoder.trg_embed_dim"])]

  def dependent_init_params(self):
    return [DependentInitParam(param_descr="src_embedder.vocab_size", value_fct=lambda: self.context["corpus_parser"].src_reader.vocab_size()),
            DependentInitParam(param_descr="decoder.vocab_size", value_fct=lambda: self.context["corpus_parser"].trg_reader.vocab_size()),
            DependentInitParam(param_descr="trg_embedder.vocab_size", value_fct=lambda: self.context["corpus_parser"].trg_reader.vocab_size())]

  def initialize(self, args):
      # Search Strategy
    len_norm_type   = getattr(length_normalization, args.len_norm_type)
    self.search_strategy = BeamSearch(b=args.beam, max_len=args.max_len, len_norm=len_norm_type(**args.len_norm_params))
    self.report_path = args.report_path

  def calc_loss(self, src, trg, info=None):
    embeddings = self.src_embedder.embed_sent(src)
    encodings = self.encoder.transduce(embeddings)
    self.attender.start_sent(encodings)
    self.decoder.initialize()
    self.decoder.add_input(self.trg_embedder.embed(0))  # XXX: HACK, need to initialize decoder better
    losses = []

    # single mode
    if not batcher.is_batched(src):
      for ref_word in trg:
        context = self.attender.calc_context(self.decoder.state.output())
        word_loss = self.decoder.calc_loss(context, ref_word)
        losses.append(word_loss)
        self.decoder.add_input(self.trg_embedder.embed(ref_word))

    # minibatch mode
    else:
      max_len = max([len(single_trg) for single_trg in trg])

      for i in range(max_len):
        ref_word = batcher.mark_as_batch([single_trg[i] if i < len(single_trg) else Vocab.ES for single_trg in trg])
        context = self.attender.calc_context(self.decoder.state.output())

        word_loss = self.decoder.calc_loss(context, ref_word)
        mask_exp = dy.inputVector([1 if i < len(single_trg) else 0 for single_trg in trg])
        mask_exp = dy.reshape(mask_exp, (1,), len(trg))
        word_loss = word_loss * mask_exp
        losses.append(word_loss)

        self.decoder.add_input(self.trg_embedder.embed(ref_word))

    return dy.esum(losses)

  def generate(self, src, idx):
    # Not including this as a default argument is a hack to get our documentation pipeline working
    search_strategy = self.search_strategy
    if search_strategy == None:
      search_strategy = BeamSearch(1, len_norm=NoNormalization())
    if not batcher.is_batched(src):
      src = batcher.mark_as_batch([src])
    outputs = []
    for sents in src:
      embeddings = self.src_embedder.embed_sent(src)
      encodings = self.encoder.transduce(embeddings)
      self.attender.start_sent(encodings)
      self.decoder.initialize()
      output_actions = search_strategy.generate_output(self.decoder, self.attender, self.trg_embedder, src_length=len(sents))
      # In case of reporting
      if self.report_path is not None:
        src_words = [self.src_vocab[w] for w in sents]
        trg_words = [self.trg_vocab[w] for w in output_actions[1:]]
        attentions = self.attender.attention_vecs
        self.set_html_input(idx, src_words, trg_words, attentions)
        self.set_html_path('{}.{}'.format(self.report_path, str(idx)))
      # Append output to the outputs
      outputs.append(TextOutput(output_actions, self.trg_vocab))
    return outputs

  @recursive_assign
  def html_report(self, context=None):
    assert(context is None)
    idx, src, trg, att = self.html_input
    path_to_report = self.html_path
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
    captions = ["Source Words", "Target Words"]
    inputs = [src, trg]
    for caption, inp in six.moves.zip(captions, inputs):
      if inp is None: continue
      sent = ' '.join(inp)
      p = etree.SubElement(main_content, 'p')
      p.text = "{}: {}".format(six.u(caption), six.u(sent))

    # Generating attention
    if not any([src is None, trg is None, att is None]):
      attention = etree.SubElement(main_content, 'p')
      att_text = etree.SubElement(attention, 'b')
      att_text.text = "Attention:"
      etree.SubElement(attention, 'br')
      att_mtr = etree.SubElement(attention, 'img', src="{}.attention.png".format(filename_of_report))
      attention_file = "{}.attention.png".format(path_to_report)

      if type(att) == dy.Expression:
        attentions = att.npvalue()
      elif type(att) == list:
        attentions = np.concatenate([x.npvalue() for x in att], axis=1)
      elif type(att) != np.ndarray:
        raise RuntimeError("Illegal type for attentions in translator report: {}".format(type(attentions)))
      plot.plot_attention(src, trg, attentions, file_name = attention_file)

    # return the parent context to be used as child context
    return html


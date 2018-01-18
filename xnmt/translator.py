from __future__ import division, generators

import six
import plot
import io
import dynet as dy
import numpy as np

import xnmt.length_normalization
import xnmt.batcher

from xnmt.vocab import Vocab
from xnmt.events import register_xnmt_event_assign, register_handler, handle_xnmt_event
from xnmt.generator import GeneratorModel
from xnmt.serializer import Serializable, DependentInitParam
from xnmt.search_strategy import BeamSearch, GreedySearch
from xnmt.output import TextOutput
from xnmt.reports import Reportable
from xnmt.loss import LossBuilder
import xnmt.serializer
from xnmt.batcher import mark_as_batch, is_batched

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

  def __init__(self, src_embedder, encoder, attender, trg_embedder, decoder,
               calc_global_fertility=False):
    '''Constructor.

    :param src_embedder: A word embedder for the input language
    :param encoder: An encoder to generate encoded inputs
    :param attender: An attention module
    :param trg_embedder: A word embedder for the output language
    :param decoder: A decoder
    '''
    register_handler(self)
    self.src_embedder = src_embedder
    self.encoder = encoder
    self.attender = attender
    self.trg_embedder = trg_embedder
    self.decoder = decoder
    self.calc_global_fertility = calc_global_fertility

  def shared_params(self):
    return [set(["src_embedder.emb_dim", "encoder.input_dim"]),
            set(["encoder.hidden_dim", "attender.input_dim", "decoder.input_dim"]),
            set(["attender.state_dim", "decoder.lstm_dim"]),
            set(["trg_embedder.emb_dim", "decoder.trg_embed_dim"])]

  def dependent_init_params(self):
    return [DependentInitParam(param_descr="src_embedder.vocab_size", value_fct=lambda: self.yaml_context.corpus_parser.src_reader.vocab_size()),
            DependentInitParam(param_descr="decoder.vocab_size", value_fct=lambda: self.yaml_context.corpus_parser.trg_reader.vocab_size()),
            DependentInitParam(param_descr="trg_embedder.vocab_size", value_fct=lambda: self.yaml_context.corpus_parser.trg_reader.vocab_size()),
            DependentInitParam(param_descr="src_embedder.vocab", value_fct=lambda: self.yaml_context.corpus_parser.src_reader.vocab),
            DependentInitParam(param_descr="trg_embedder.vocab", value_fct=lambda: self.yaml_context.corpus_parser.trg_reader.vocab)]

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
    self.print_fertility = kwargs.get("print_fertility", 0) == 1

  def initialize_training_strategy(self, training_strategy):
    self.loss_calculator = training_strategy

  @handle_xnmt_event
  def on_set_train(self, val):
    self.is_train = val

  def calc_loss(self, src, trg):
    """
    :param src: source sequence (unbatched, or batched + padded)
    :param trg: target sequence (unbatched, or batched + padded); losses will be accumulated only if trg_mask[batch,pos]==0, or no mask is set
    :returns: (possibly batched) loss expression
    """
    assert hasattr(self, "loss_calculator")
    self.start_sent(src)
    embeddings = self.src_embedder.embed_sent(src)
    encodings = self.encoder(embeddings)
    self.attender.init_sent(encodings)
    # Initialize the hidden state from the encoder
    ss = mark_as_batch([Vocab.SS] * len(src)) if is_batched(src) else Vocab.SS
    dec_state = self.decoder.initial_state(self.encoder.get_final_states(), self.trg_embedder.embed(ss))
    model_loss = self.loss_calculator(self, dec_state, src, trg)

    if self.is_train and self.calc_global_fertility:
      loss = LossBuilder()
      loss.add_loss("mle", model_loss)
      loss.add_loss("fertility", self.global_fertility(self.attender.attention_vecs))
      model_loss = loss

    return model_loss

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
        # Attentions
        attentions = self.attender.attention_vecs
        if type(attentions) == dy.Expression:
          attentions = attentions.npvalue()
        elif type(attentions) == list:
          attentions = np.concatenate([x.npvalue() for x in attentions], axis=1)
        elif type(attentions) != np.ndarray:
          raise RuntimeError("Illegal type for attentions in translator report: {}".format(type(attentions)))
        # Segmentation
        segment = self.get_report_resource("segmentation")
        if segment is not None:
          segment = list(six.moves.map(lambda x: int(x[0]), segment))
          src_inp = [x[0] for x in self.encoder.apply_segmentation(src_words, segment)]
        else:
          src_inp = src_words
        # Other Resources
        self.set_report_input(idx, src_inp, trg_words, attentions)
        self.set_report_resource("src_words", src_words)
        self.set_report_path('{}.{}'.format(self.report_path, str(idx)))
        self.generate_report(self.report_type)
      # Append output to the outputs
      outputs.append(TextOutput(actions=output_actions,
                                vocab=self.trg_vocab if hasattr(self, "trg_vocab") else None,
                                score=score))
    self.outputs = outputs
    return outputs

  def global_fertility(self, a):
    return dy.sum_elems(dy.square(1 - dy.esum(a)))

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
      plot.plot_attention(src, trg, att, file_name = attention_file)

    # return the parent context to be used as child context
    return html

  @handle_xnmt_event
  def on_file_report(self):
    idx, src, trg, attn = self.get_report_input()
    col_length = []
    col_length.append(max(len(x) for x in src))
    for word in trg:
      col_length.append(max(len(word), 6))
    with io.open(self.get_report_path() + ".att", encoding='utf-8', mode='w') as attn_file:
      for i in range(len(src)+1):
        if i == 0:
          words = trg + [""]
        else:
          words = ["%.4f" % (f) for f in attn[i-1]] + [src[i-1]]
        str_format = ""
        for length in col_length:
          str_format += "{:%ds}" % (length+2)
        print(str_format.format(*words), file=attn_file)


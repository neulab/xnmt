from __future__ import division, generators

import six
import plot
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
import xnmt.evaluator
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

  def __init__(self, src_embedder, encoder, attender, trg_embedder, decoder, loss_calculator=None):
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

    if loss_calculator is None:
      self.loss_calculator = TranslatorMLELoss()
    else:
      self.loss_calculator = loss_calculator

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
            DependentInitParam(param_descr="trg_embedder.vocab_size", value_fct=lambda: self.context.corpus_parser.trg_reader.vocab_size()),
            DependentInitParam(param_descr="src_embedder.vocab", value_fct=lambda: self.context.corpus_parser.src_reader.vocab),
            DependentInitParam(param_descr="trg_embedder.vocab", value_fct=lambda: self.context.corpus_parser.trg_reader.vocab)]

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

  def calc_loss(self, src, trg):
    """
    :param src: source sequence (unbatched, or batched + padded)
    :param trg: target sequence (unbatched, or batched + padded); losses will be accumulated only if trg_mask[batch,pos]==0, or no mask is set
    :returns: (possibly batched) loss expression
    """
    self.start_sent()
    embeddings = self.src_embedder.embed_sent(src)
    encodings = self.encoder.transduce(embeddings)
    self.attender.init_sent(encodings)
    # Initialize the hidden state from the encoder
    ss = mark_as_batch([Vocab.SS] * len(src)) if is_batched(src) else Vocab.SS
    dec_state = self.decoder.initial_state(self.encoder.get_final_states(), self.trg_embedder.embed(ss))
    return self.loss_calculator(self, dec_state, src, trg)

  def generate(self, src, idx, src_mask=None, forced_trg_ids=None):
    if not xnmt.batcher.is_batched(src):
      src = xnmt.batcher.mark_as_batch([src])
    else:
      assert src_mask is not None
    outputs = []
    for sents in src:
      self.start_sent()
      embeddings = self.src_embedder.embed_sent(src)
      encodings = self.encoder.transduce(embeddings)
      self.attender.init_sent(encodings)
      ss = mark_as_batch([Vocab.SS] * len(src)) if is_batched(src) else Vocab.SS
      dec_state = self.decoder.initial_state(self.encoder.get_final_states(), self.trg_embedder.embed(ss))
      output_actions, score = self.search_strategy.generate_output(self.decoder, self.attender, self.trg_embedder, dec_state, src_length=len(sents), forced_trg_ids=forced_trg_ids)
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
      plot.plot_attention(src, trg, attentions, file_name = attention_file)

    # return the parent context to be used as child context
    return html

  @recursive
  def file_report(self):
    pass

class TranslatorMLELoss(Serializable):
  yaml_tag = '!TranslatorMLELoss'

  def __call__(self, translator, dec_state, src, trg):
    trg_mask = trg.mask if xnmt.batcher.is_batched(trg) else None
    losses = []
    seq_len = len(trg[0]) if xnmt.batcher.is_batched(src) else len(trg)
    if xnmt.batcher.is_batched(src):
      for j, single_trg in enumerate(trg):
        assert len(single_trg) == seq_len # assert consistent length
        assert 1==len([i for i in range(seq_len) if (trg_mask is None or trg_mask.np_arr[j,i]==0) and single_trg[i]==Vocab.ES]) # assert exactly one unmasked ES token
    for i in range(seq_len):
      ref_word = trg[i] if not xnmt.batcher.is_batched(src) \
                      else xnmt.batcher.mark_as_batch([single_trg[i] for single_trg in trg])

      dec_state.context = translator.attender.calc_context(dec_state.rnn_state.output())
      word_loss = translator.decoder.calc_loss(dec_state, ref_word)
      if xnmt.batcher.is_batched(src) and trg_mask is not None:
        word_loss = trg_mask.cmult_by_timestep_expr(word_loss, i, inverse=True)
      losses.append(word_loss)
      if i < seq_len-1:
        dec_state = translator.decoder.add_input(dec_state, translator.trg_embedder.embed(ref_word))

    return dy.esum(losses)

class TranslatorReinforceLoss(Serializable):
  yaml_tag = '!TranslatorReinforceLoss'

  def __init__(self, evaluation_metric=None, sample_length=50):
    self.sample_length = sample_length
    if evaluation_metric is None:
      self.evaluation_metric = xnmt.evaluator.BLEUEvaluator(ngram=4, smooth=1)
    else:
      self.evaluation_metric = evaluation_metric

  def __call__(self, translator, dec_state, src, trg):
    # TODO: apply trg.mask ?
    samples = []
    logsofts = []
    done = [False for _ in range(len(trg))]
    for _ in range(self.sample_length):
      dec_state.context = translator.attender.calc_context(dec_state.rnn_state.output())
      logsoft = dy.log_softmax(translator.decoder.get_scores(dec_state))
      sample = logsoft.tensor_value().categorical_sample_log_prob().as_numpy()[0]
      # Keep track of previously sampled EOS
      sample = [sample_i if not done_i else Vocab.ES for sample_i, done_i in zip(sample, done)]
      # Appending and feeding in the decoder
      logsofts.append(logsoft)
      samples.append(sample)
      dec_state = translator.decoder.add_input(dec_state, translator.trg_embedder.embed(xnmt.batcher.mark_as_batch(sample)))
      # Check if we are done.
      done = list(six.moves.map(lambda x: x == Vocab.ES, sample))
      if all(done):
        break
    samples = np.stack(samples, axis=1).tolist()
    eval_score = []
    for trg_i, sample_i in zip(trg, samples):
      # Removing EOS
      try:
        idx = sample_i.index(Vocab.ES)
        sample_i = sample_i[:idx]
      except ValueError:
        pass
      try:
        idx = trg_i.words.index(Vocab.ES)
        trg_i.words = trg_i.words[:idx]
      except ValueError:
        pass
      # Calculate the evaluation score
      eval_score.append(self.evaluation_metric.evaluate_fast(trg_i.words, sample_i))

    return -dy.sum_elems(dy.cmult(dy.inputTensor(eval_score, batched=True), dy.esum(logsofts)))

# To be implemented
class TranslatorMinRiskLoss(Serializable):
  yaml_tag = 'TranslatorMinRiskLoss'


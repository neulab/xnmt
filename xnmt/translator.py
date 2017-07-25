from __future__ import division, generators

import dynet as dy
import length_normalization
from batcher import *
from search_strategy import *
from vocab import Vocab
from serializer import Serializable, DependentInitParam
from embedder import SimpleWordEmbedder
from decoder import MlpSoftmaxDecoder
from output import TextOutput
from model import recursive, HierarchicalModel, GeneratorModel

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

  def translate(self, src):
    '''Translate a particular sentence.

    :param src: The source, a sentence or a batch of sentences.
    :returns: A translated expression.
    '''
    raise NotImplementedError('translate must be implemented for Translator subclasses')

  @recursive
  def set_train(self, val):
    pass

class DefaultTranslator(Translator, Serializable):
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

  def calc_loss(self, src, trg, info=None):
    embeddings = self.src_embedder.embed_sent(src)
    encodings = self.encoder.transduce(embeddings)
    self.attender.start_sent(encodings)
    self.decoder.initialize()
    self.decoder.add_input(self.trg_embedder.embed(0))  # XXX: HACK, need to initialize decoder better
    losses = []

    # single mode
    if not Batcher.is_batched(src):
      for ref_word in trg:
        context = self.attender.calc_context(self.decoder.state.output())
        word_loss = self.decoder.calc_loss(context, ref_word)
        losses.append(word_loss)
        self.decoder.add_input(self.trg_embedder.embed(ref_word))

    # minibatch mode
    else:
      max_len = max([len(single_trg) for single_trg in trg])

      for i in range(max_len):
        ref_word = Batcher.mark_as_batch([single_trg[i] if i < len(single_trg) else Vocab.ES for single_trg in trg])
        context = self.attender.calc_context(self.decoder.state.output())

        word_loss = self.decoder.calc_loss(context, ref_word)
        mask_exp = dy.inputVector([1 if i < len(single_trg) else 0 for single_trg in trg])
        mask_exp = dy.reshape(mask_exp, (1,), len(trg))
        word_loss = dy.sum_batches(word_loss * mask_exp)
        losses.append(word_loss)

        self.decoder.add_input(self.trg_embedder.embed(ref_word))

    return dy.esum(losses)

  def generate(self, src):
    # Not including this as a default argument is a hack to get our documentation pipeline working
    search_strategy = self.search_strategy
    if search_strategy == None:
      search_strategy = BeamSearch(1, len_norm=NoNormalization())
    if not Batcher.is_batched(src):
      src = Batcher.mark_as_batch([src])
    outputs = []
    for sents in src:
      embeddings = self.src_embedder.embed_sent(src)
      encodings = self.encoder.transduce(embeddings)
      self.attender.start_sent(encodings)
      self.decoder.initialize()
      output_actions = search_strategy.generate_output(self.decoder, self.attender, self.trg_embedder, src_length=len(sents))
      #if report != None:
      #  report.trg_words = [trg_vocab[x] for x in output_actions[1:]] # The first token is the start token
      #  report.attentions = self.attender.attention_vecs
      outputs.append(TextOutput(output_actions, self.trg_vocab))
    return outputs


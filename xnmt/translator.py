from __future__ import division, generators

import dynet as dy
from batcher import *
from search_strategy import *
from vocab import Vocab
from serializer import Serializable, DependentInitParam, bare_component
from train_test_interface import TrainTestInterface
from embedder import SimpleWordEmbedder
from decoder import MlpSoftmaxDecoder

class Translator(TrainTestInterface):
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

  def set_train(self, val):
    for component in self.get_train_test_components():
      Translator.set_train_recursive(component, val)
  @staticmethod
  def set_train_recursive(component, val):
    component.set_train(val)
    for sub_component in component.get_train_test_components():
      Translator.set_train_recursive(sub_component, val)


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
    self.src_embedder = src_embedder
    self.encoder = encoder
    self.attender = attender
    self.trg_embedder = trg_embedder
    self.decoder = decoder
  
  def shared_params(self):
    return [
            set(["src_embedder.emb_dim", "encoder.input_dim"]),
            set(["encoder.hidden_dim", "attender.input_dim", "decoder.input_dim"]), # TODO: encoder.hidden_dim may not always exist (e.g. for CNN encoders), need to deal with that case
            set(["attender.state_dim", "decoder.lstm_dim"]),
            set(["trg_embedder.emb_dim", "decoder.trg_embed_dim"]),
            ]
  def dependent_init_params(self):
    return [
            DependentInitParam(param_descr="src_embedder.vocab_size", value_fct=lambda: self.context["corpus_parser"].src_reader.vocab_size()),
            DependentInitParam(param_descr="decoder.vocab_size", value_fct=lambda: self.context["corpus_parser"].trg_reader.vocab_size()),
            DependentInitParam(param_descr="trg_embedder.vocab_size", value_fct=lambda: self.context["corpus_parser"].trg_reader.vocab_size()),
            ]

  def get_train_test_components(self):
    return [self.encoder, self.decoder]

  def calc_loss(self, src, trg):
    embeddings = self.src_embedder.embed_sent(src)
    encodings = self.encoder.transduce(embeddings)
    self.attender.start_sent(encodings)
    self.decoder.initialize()
    self.decoder.add_input(self.trg_embedder.embed(0))  # XXX: HACK, need to initialize decoder better
    losses = []

    # single mode
    if not Batcher.is_batch_sent(src):
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

  def translate(self, src, search_strategy=None):
    # Not including this as a default argument is a hack to get our documentation pipeline working
    if search_strategy == None:
      search_strategy = BeamSearch(1, len_norm=NoNormalization())
    output = []
    if not Batcher.is_batch_sent(src):
      src = Batcher.mark_as_batch([src])
    for sents in src:
      embeddings = self.src_embedder.embed_sent(src)
      encodings = self.encoder.transduce(embeddings)
      self.attender.start_sent(encodings)
      self.decoder.initialize()
      output.append(search_strategy.generate_output(self.decoder, self.attender, self.trg_embedder, src_length=len(sents)))
    return output



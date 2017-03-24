import dynet as dy
from batcher import *
from search_strategy import *

class Translator:
  '''
  A template class implementing an end-to-end translator that can calculate a
  loss and generate translations.
  '''

  '''
  Calculate the loss of the input and output.
  '''

  def loss(self, x, y):
    raise NotImplementedError('loss must be implemented for Translator subclasses')

  '''
  Calculate the loss for a batch. By default, just iterate. Overload for better efficiency.
  '''

  def batch_loss(self, xs, ys):
    return dy.esum([self.loss(x, y) for x, y in zip(xs, ys)])


class DefaultTranslator(Translator):
  def __init__(self, encoder, attender, decoder):
    self.encoder = encoder
    self.attender = attender
    self.decoder = decoder

  def calc_loss(self, source, target):
    encodings = self.encoder.encode(source)
    self.attender.start_sentence(encodings)
    self.decoder.initialize()
    self.decoder.add_input(0)  # XXX: HACK, need to initialize decoder better
    losses = []

    # single mode
    if not Batcher.is_batch_sentence(source):
      for ref_word in target:
        context = self.attender.calc_context(self.decoder.state.output())
        word_loss = self.decoder.calc_loss(context, ref_word)
        losses.append(word_loss)
        self.decoder.add_input(ref_word)

    # minibatch mode
    else:
      max_len = max([len(single_target) for single_target in target])

      for i in range(max_len):
        ref_word = [single_target[i] if i < len(single_target) else single_target[len(single_target) - 1] \
                    for single_target in target]
        context = self.attender.calc_context(self.decoder.state.output())

        word_loss = self.decoder.calc_loss(context, ref_word)
        mask_exp = dy.inputVector([1 if i < len(single_target) else 0 for single_target in target])
        mask_exp = dy.reshape(mask_exp, (1,), len(target))
        word_loss = dy.sum_batches(word_loss * mask_exp)
        losses.append(word_loss)

        self.decoder.add_input(ref_word)

    return dy.esum(losses)

  def translate(self, source, search_strategy=BeamSearch(1, len_norm=NoNormalization())):
    output = []
    if not Batcher.is_batch_sentence(source):
      source = [source]
    for sentences in source:
      encodings = self.encoder.encode(sentences)
      self.attender.start_sentence(encodings)
      self.decoder.initialize()
      output.append(search_strategy.generate_output(self.decoder, self.attender, source_length=len(sentences)))
    return output


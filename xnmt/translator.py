import dynet as dy

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
    self.decoder.add_input(0) # XXX: HACK, need to initialize decoder better

    losses = []
    for ref_word in target:
      context = self.attender.calc_context(self.decoder.state.output())
      word_loss = self.decoder.calc_loss(context, ref_word)
      losses.append(word_loss)

    return dy.esum(losses)

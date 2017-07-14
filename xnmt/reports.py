import matplotlib.pyplot as plt

class DefaultTranslatorReport(object):
  
  def __init__(self):
    self.src_text = None
    self.trg_text = None
    self.src_words = None
    self.trg_words = None
    self.attentions = None

  def plot_attention(self, src_words, trg_words, attention_matrix, file_name):
    """This takes in source and target words and an attention matrix (in numpy format)
    and prints a visualization of this to a file.

    :param src_words: a list of words in the source
    :param trg_words: a list of target words
    :param attention_matrix: a two-dimensional numpy array of values between zero and one,
      where rows correspond to source words, and columns correspond to target words
    :param file_name: the name of the file to which we write the attention
    """
    pass

  def write_report(self, path_to_report, idx, src_vocab, trg_vocab):
    with open("{}.html".format(path_to_report)) as f:
      f.write("<html><head><title>Translation Report</title></head><body>\n")
      src_text, trg_text = None, None
      # Print Source text
      if self.src_text != None: f.write("<p><b>Source Text: </b> {}</p>".format(src_text))
      if self.src_words != None: f.write("<p><b>Source Words: </b> {}".format(' '.join(src_words)))
      if self.trg_text != None: f.write("<p><b>Target Text: </b> {}</p>".format(trg_text))
      if self.trg_words != None: f.write("<p><b>Target Words: </b> {}".format(' '.join(trg_words)))
      # Alignments
      if all([x != None for x in (self.src_words, self.trg_words, self.attentions)):
        pass

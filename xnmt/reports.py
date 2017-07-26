
from lxml import etree

class HTMLReportable(object):
  def html_report(self, parent_context=None):
    raise NotImplementedError()

  def set_html_input(self, *inputs):
    self.html_input = inputs

  def set_html_path(self, report_path):
    self.html_path = report_path

  def write_resources(self):
    pass

  def generate_html_report(self):
    html = etree.tostring(self.html_report(), encoding='unicode', pretty_print=True)
    self.write_resources()
    with open(self.html_path + '.html', 'w') as f:
      f.write(html)

#class DefaultTranslatorReport(object):
#
#  def __init__(self):
#    self.src_text = None
#    self.trg_text = None
#    self.src_words = None
#    self.trg_words = None
#    self.attentions = None
#
#  def write_report(self, path_to_report, idx=None):
#    filename_of_report = os.path.basename(path_to_report)
#    with open("{}.html".format(path_to_report), 'w') as f:
#      if idx != None:
#        f.write("<html><head><title>Translation Report for Sentence {}</title></head><body>\n".format(idx))
#        f.write("<h1>Translation Report for Sentence {}</h1>\n".format(idx))
#      else:
#        f.write("<html><head><title>Translation Report</title></head><body>\n")
#        f.write("<h1>Translation Report</h1>\n")
#      src_text, trg_text = None, None
#      # Print Source text
#      if self.src_text != None: f.write("<p><b>Source Text: </b> {}</p>\n".format(self.src_text))
#      if self.src_words != None: f.write("<p><b>Source Words: </b> {}</p>\n".format(' '.join(self.src_words)))
#      if self.trg_text != None: f.write("<p><b>Target Text: </b> {}</p>\n".format(self.trg_text))
#      if self.trg_words != None: f.write("<p><b>Target Words: </b> {}</p>\n".format(' '.join(self.trg_words)))
#      # Alignments
#      if self.src_words is not None and self.trg_words is not None and self.attentions is not None:
#        if type(self.attentions) == dy.Expression:
#          self.attentions = self.attentions.npvalue()
#        elif type(self.attentions) == list:
#          self.attentions = np.concatenate([x.npvalue() for x in self.attentions], axis=1)
#        elif type(self.attentions) != np.ndarray:
#          raise RuntimeError("Illegal type for attentions in translator report: {}".format(type(self.attentions)))
#        attention_file = "{}.attention.png".format(path_to_report)
#        DefaultTranslatorReport.plot_attention(self.src_words, self.trg_words, self.attentions, file_name = attention_file)
#        f.write("<p><b>Attention:</b><br/><img src=\"{}.attention.png\"/></p>\n".format(filename_of_report))
#
#      f.write("</body></html>")
#
#if __name__ == "__main__":
#
#  # temporary call to plot_attention
#  rep = DefaultTranslatorReport()
#  rep.src_words = ['The', 'cat', 'was', 'sitting', 'on', 'top', 'of', 'the', 'wardrobe', '.']
#  rep.trg_words = ['Le', 'chat', 'etait', 'assis', 'sur', 'le', 'dessus', 'de', 'l\'armoire', '.']
#  rep.attentions = np.random.rand(len(rep.src_words), len(rep.trg_words))
#  rep.write_report("examples/output/xnmt_translator_report", 1)

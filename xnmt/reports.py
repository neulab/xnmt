from lxml import etree

class HTMLReportable(object):
  def html_report(self, context=None):
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

#if __name__ == "__main__":
#
#  # temporary call to plot_attention
#  rep = DefaultTranslatorReport()
#  rep.src_words = ['The', 'cat', 'was', 'sitting', 'on', 'top', 'of', 'the', 'wardrobe', '.']
#  rep.trg_words = ['Le', 'chat', 'etait', 'assis', 'sur', 'le', 'dessus', 'de', 'l\'armoire', '.']
#  rep.attentions = np.random.rand(len(rep.src_words), len(rep.trg_words))
#  rep.write_report("examples/output/xnmt_translator_report", 1)

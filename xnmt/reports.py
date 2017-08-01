import io

from lxml import etree
from decorators import recursive, recursive_assign

class HTMLReportable(object):
  @recursive_assign
  def html_report(self, context=None):
    if context is None:
      raise NotImplementedError("Not implemented html_report for class:",
                                self.__class__.__name__)
    return context

  def set_html_input(self, *inputs):
    self.html_input = inputs

  @recursive
  def set_html_path(self, report_path):
    self.html_path = report_path

  @recursive
  def set_html_resource(self, key, value):
    if not hasattr(self, "html_reportable_resources"):
      self.html_reportable_resources = {}
    self.html_reportable_resources[key] = value

  @recursive
  def clear_html_resources(self):
    if hasattr(self, "html_reportable_resources"):
      self.html_reportable_resources.clear()

  def get_html_resource(self, key):
    return self.html_reportable_resources[key]

  def generate_html_report(self):
    html = etree.tostring(self.html_report(), encoding='unicode', pretty_print=True)
    with io.open(self.html_path + '.html', 'w', encoding='utf-8') as f:
      f.write(html)

#if __name__ == "__main__":
#
#  # temporary call to plot_attention
#  rep = DefaultTranslatorReport()
#  rep.src_words = ['The', 'cat', 'was', 'sitting', 'on', 'top', 'of', 'the', 'wardrobe', '.']
#  rep.trg_words = ['Le', 'chat', 'etait', 'assis', 'sur', 'le', 'dessus', 'de', 'l\'armoire', '.']
#  rep.attentions = np.random.rand(len(rep.src_words), len(rep.trg_words))
#  rep.write_report("examples/output/xnmt_translator_report", 1)

from lxml import etree

from xnmt.events import register_xnmt_event, register_xnmt_event_sum
from xnmt import logger

class Reportable(object):
  """ Template class for a Reportable Model """
  @register_xnmt_event
  def report_start(self, report_path, report_type):
    if report_type:
      report_type = [x.strip() for x in report_type.split(",")]
    self.report_type = report_type
    self.report_path = report_path
    self.notified = False

    if "line" in self.report_type:
      self.report_file = open(report_path + ".line", 'w', encoding='utf-8')
  
  def report_end(self):
    if "line" in self.report_type:
      self.report_file.close()

  @register_xnmt_event
  def report_item(self, i):
    if self.report_path is None:
      return
    report_path = '{}.{}'.format(self.report_path, str(i))
    for typ in self.report_type:
      if typ == "html":
        html_report = self.html_report(context=None)
        html = etree.tostring(html_report, encoding='unicode', pretty_print=True)
        with open(report_path + '.html', 'w', encoding='utf-8') as f:
          f.write(html)
      elif typ == "file":
        self.file_report(self.report_path)
      elif typ == "line":
        out = {}
        self.line_report(out)
        line_output = []
        if not self.notified:
          self.notified = True
          logger.info("Reporting line key: " + str(sorted(out.keys())))
        for key, value in sorted(out.items()):
          line_output.append(str(value))
        print(" ||| ".join(line_output), file=self.report_file)
      else:
        raise ValueError("Unknown report type:", typ)

  def html_report(self, context):
    pass

  @register_xnmt_event
  def file_report(self, report_path):
    pass

  @register_xnmt_event
  def line_report(self, output_dict):
    pass


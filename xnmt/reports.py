import io
import six

from lxml import etree
from xnmt.events import register_xnmt_event, register_xnmt_event_assign, handle_xnmt_event

class Reportable(object):
  @register_xnmt_event_assign
  def html_report(self, context=None):
    raise NotImplementedError()

  ### Getter + Setter for particular report py
  def set_report_input(self, *inputs):
    self.__report_input = inputs

  def get_report_input(self):
    return self.__report_input

  def get_report_path(self):
    return self.__report_path

  @register_xnmt_event
  def set_report_path(self, report_path):
    self.__report_path = report_path
  @handle_xnmt_event
  def on_set_report_path(self, report_path):
    self.__report_path = report_path

  @register_xnmt_event
  def set_report_resource(self, key, value):
    if not hasattr(self, "__reportable_resources"):
      self.__reportable_resources = {}
    self.__reportable_resources[key] = value
  @handle_xnmt_event
  def on_set_report_resource(self, key, value):
    if not hasattr(self, "__reportable_resources"):
      self.__reportable_resources = {}
    self.__reportable_resources[key] = value

  @register_xnmt_event
  def clear_report_resources(self):
    if hasattr(self, "clear_resources"):
      self.__reportable_resources.clear()
  @handle_xnmt_event
  def on_clear_report_resources(self):
    if hasattr(self, "clear_resources"):
      self.__reportable_resources.clear()

  def get_report_resource(self, key):
    return self.__reportable_resources.get(key, None)

  # Methods to generate report
  def generate_html_report(self):
    html_report = self.html_report(context=None)
    html = etree.tostring(html_report, encoding='unicode', pretty_print=True)
    with io.open(self.__report_path + '.html', 'w', encoding='utf-8') as f:
      f.write(html)

  def generate_file_report(self):
    self.file_report()
  
  @register_xnmt_event
  def file_report(self):
    pass

  ### Public acessible Methods
  def generate_report(self, report_type):
    if report_type:
      report_type = list(six.moves.map(lambda x: x.strip(), report_type.strip().split(",")))
    for typ in report_type:
      if typ == "html":
        self.generate_html_report()
      elif typ == "file":
        self.generate_file_report()
      else:
        raise ValueError("Unknown report type:", typ)


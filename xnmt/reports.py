import io
import six

from lxml import etree
from decorators import recursive, recursive_assign

class Reportable(object):
  @recursive_assign
  def html_report(self, context=None):
    if context is None:
      raise NotImplementedError("Not implemented html_report for class:",
                                self.__class__.__name__)
    return context

  @recursive
  def file_report(self):
    pass

  ### Getter + Setter for particular report py
  def set_report_input(self, *inputs):
    self.__report_input = inputs

  def get_report_input(self):
    return self.__report_input

  def get_report_path(self):
    return self.__report_path

  ### Methods that are applied recursively to the childs
  ### of HierarchicalModel
  @recursive
  def set_report_path(self, report_path):
    self.__report_path = report_path

  @recursive
  def set_report_resource(self, key, value):
    if not hasattr(self, "__reportable_resources"):
      self.__reportable_resources = {}
    self.__reportable_resources[key] = value

  @recursive
  def clear_report_resources(self):
    if hasattr(self, "clear_resources"):
      self.__reportable_resources.clear()

  def get_report_resource(self, key):
    return self.__reportable_resources.get(key, None)

  # Methods to generate report
  def generate_html_report(self):
    html_report = self.html_report()
    if html_report is None:
      raise RuntimeError("Some of the html_report of childs of HTMLReportable object have not been implemented.")
    html = etree.tostring(html_report, encoding='unicode', pretty_print=True)
    with io.open(self.__report_path + '.html', 'w', encoding='utf-8') as f:
      f.write(html)

  def generate_file_report(self):
    self.file_report()

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


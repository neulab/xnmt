import html

class HTMLReportable(object):
  ''' The interface of an reportable object '''

  def __init__(self, *reportable):
    self.content = []
    self.__reportable = reportable

  def report(self):
    ''' Driver method of recursive report '''
    self.__recursive_report(None)

  def __recursive_report(self, parent_context=None):
    if parent_context is None:
      parent_context = html.HTML()

    # self report + clear content after report
    self.html_report(context=parent_context)
    self.content.clear()

    # Traversal pre-order for the children
    for child in self.__reportable:
      child.recursive_report(parent_context)

  def register_report_content(self, *contents):
    self.content.extend(contents)

  def html_report(self, context):
    raise NotImplementedError("Not implemented html report for HTMLReportable interface.")


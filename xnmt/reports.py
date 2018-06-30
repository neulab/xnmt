import os
from lxml import etree

import numpy as np

from xnmt.events import register_xnmt_event_assign, handle_xnmt_event
import xnmt.plot
from xnmt.persistence import Serializable, serializable_init
from xnmt import vocab
import xnmt.output

class Reportable(object):

  def add_sent_for_report(self, sent_info):
    if not hasattr(self, "_sent_info_list"):
      self._sent_info_list = []
    self._sent_info_list.append(sent_info)

  @handle_xnmt_event
  def on_get_report_input(self, context={}):
    if len(context)>0:
      assert len(context) == len(self._sent_info_list)
    else:
      context = []
      for _ in range(len(self._sent_info_list)): context.append({})
    for context_i, sent_i in zip(context, self._sent_info_list):
      context_i.update(sent_i)
    self._sent_info_list.clear()
    return context

#   # TODO: document me
#
#   @register_xnmt_event_assign
#   def html_report(self, context=None):
#     raise NotImplementedError()
#
#   ### Getter + Setter for particular report py
#   def set_report_input(self, *inputs):
#     self.__report_input = inputs
#
#   def get_report_input(self):
#     return self.__report_input
#
#   def get_report_path(self):
#     return self.__report_path
#
#   @register_xnmt_event
#   def set_report_path(self, report_path):
#     self.__report_path = report_path
#   @handle_xnmt_event
#   def on_set_report_path(self, report_path):
#     self.__report_path = report_path
#
#   @register_xnmt_event
#   def set_report_resource(self, key, value):
#     if not hasattr(self, "__reportable_resources"):
#       self.__reportable_resources = {}
#     self.__reportable_resources[key] = value
#   @handle_xnmt_event
#   def on_set_report_resource(self, key, value):
#     if not hasattr(self, "__reportable_resources"):
#       self.__reportable_resources = {}
#     self.__reportable_resources[key] = value
#
#   @register_xnmt_event
#   def clear_report_resources(self):
#     if hasattr(self, "clear_resources"):
#       self.__reportable_resources.clear()
#   @handle_xnmt_event
#   def on_clear_report_resources(self):
#     if hasattr(self, "clear_resources"):
#       self.__reportable_resources.clear()
#
#   def get_report_resource(self, key):
#     return self.__reportable_resources.get(key, None)
#
#   # Methods to generate report
#   def generate_html_report(self):
#     html_report = self.html_report(context=None)
#     html = etree.tostring(html_report, encoding='unicode', pretty_print=True)
#     with open(self.__report_path + '.html', 'w', encoding='utf-8') as f:
#       f.write(html)
#
#   def generate_file_report(self):
#     self.file_report()
#
#   @register_xnmt_event
#   def file_report(self):
#     pass
#
#   ### Public acessible Methods
#   def generate_report(self, report_type):
#     if report_type:
#       report_type = [x.strip() for x in report_type.strip().split(",")]
#     for typ in report_type:
#       if typ == "html":
#         self.generate_html_report()
#       elif typ == "file":
#         self.generate_file_report()
#       else:
#         raise ValueError("Unknown report type:", typ)

class Reporter(object):
  def create_report(self, **kwargs):
    raise NotImplementedError("must be implemented by subclasses")
  def gather_and_create_reports(self):
    report_inputs = self.get_report_input(context={}) # TODO: might pass a custom dictionary that makes sure no values are overwritten
    for report_input in report_inputs:
      self.create_report(**report_input)
  @register_xnmt_event_assign
  def get_report_input(self, context={}):
    return context

class AttentionHtmlReporter(Reporter, Serializable):

  yaml_tag = "!AttentionHtmlReporter"

  @serializable_init
  def __init__(self, report_path: str):
    self.report_path = report_path

  def create_report(self, idx: int, src: xnmt.input.SimpleSentenceInput, src_vocab: vocab.Vocab,
                    output: xnmt.output.Output, attentions:np.ndarray, ** kwargs):
    src_str = " ".join([src_vocab.i2w[src_token] for src_token in src])
    trg_str = " ".join(output.readable_actions())
    html = etree.Element('html')
    head = etree.SubElement(html, 'head')
    title = etree.SubElement(head, 'title')
    body = etree.SubElement(html, 'body')
    report = etree.SubElement(body, 'h1')
    if idx is not None:
      title.text = report.text = f'Translation Report for Sentence {idx}'
    else:
      title.text = report.text = 'Translation Report'
    main_content = etree.SubElement(body, 'div', name='main_content')

    # Generating main content
    captions = ["Source Words", "Target Words"]
    inputs = [src_str, trg_str]
    for caption, sent in zip(captions, inputs):
      p = etree.SubElement(main_content, 'p')
      p.text = f"{caption}: {sent}"

    # Generating attention
    if not any([src is None, output is None, attentions is None]):
      attention = etree.SubElement(main_content, 'p')
      att_text = etree.SubElement(attention, 'b')
      att_text.text = "Attention:"
      etree.SubElement(attention, 'br')
      attention_file = f"{self.report_path}.attention.png"
      att_img = etree.SubElement(attention, 'img')
      att_img_src = f"{self.report_path}.attention.png"
      att_img.attrib['src'] = os.path.basename(att_img_src)
      att_img.attrib['alt'] = 'attention matrix'
      xnmt.plot.plot_attention(src_str.split(), trg_str.split(), attentions, file_name = attention_file)

    # return the parent context to be used as child context
    return html


class AttentionFileReporter(Reporter, Serializable):

  yaml_tag = "!AttentionFileReporter"

  @serializable_init
  def __init__(self, report_path: str):
    self.report_path = report_path

  def create_report(self, **kwargs):
    idx = kwargs.get("idx", -1)
    src = kwargs.get("src") # TODO: what type is this?
    trg = kwargs.get("trg") # TODO: what type is this?
    attn = kwargs.get("att") # TODO: what type is this?
    # idx, src, trg, att = self.get_report_input()
    assert attn.shape == (len(src), len(trg))
    col_length = []
    for word in trg:
      col_length.append(max(len(word), 6))
    col_length.append(max(len(x) for x in src))
    with open(self.get_report_path() + ".attention.txt", encoding='utf-8', mode='w') as attn_file:
      for i in range(len(src)+1):
        if i == 0:
          words = trg + [""]
        else:
          words = [f"{f:.4f}" for f in attn[i-1]] + [src[i-1]]
        str_format = ""
        for length in col_length:
          str_format += "{:%ds}" % (length+2)
        print(str_format.format(*words), file=attn_file)

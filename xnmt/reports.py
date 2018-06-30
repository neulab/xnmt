import os
from lxml import etree
from typing import Any, Dict
import numpy as np

from xnmt.events import register_xnmt_event_assign, handle_xnmt_event, register_xnmt_handler
import xnmt.plot
from xnmt.persistence import Serializable, serializable_init
from xnmt import vocab
import xnmt.output

class Reportable(object):
  """
  Base class for classes that contribute information to a report.

  Doing so requires the implementing class to do the following:
  - specify Reportable as base class
  - call this super class's __init__(), or do @register_xnmt_handler manually
  - call self.add_sent_for_report() for each sentence
  """

  @register_xnmt_handler
  def __init__(self) -> None:
    self._sent_info_list = []

  def add_sent_for_report(self, sent_info: Dict[str,Any]) -> None:
    """
    Add key/value pairs belonging to the current sentence for reporting.

    This should be called consistently for every sentence and in order.

    Args:
      sent_info: A dictionary of key/value pairs. The keys must match (be a subset of) the arguments in the reporter's
                 create_report() method, and the values must be of the corresponding types.
    """
    if not hasattr(self, "_sent_info_list"):
      self._sent_info_list = []
    self._sent_info_list.append(sent_info)

  @handle_xnmt_event
  def on_get_report_input(self, context={}):
    if len(context)>0:
      assert len(context) == len(self._sent_info_list)
    else:
      context = []
      if not hasattr(self, "_sent_info_list"):
        raise ValueError("Nothing to report. Make sure to enable compute_report.")
      for _ in range(len(self._sent_info_list)): context.append({})
    for context_i, sent_i in zip(context, self._sent_info_list):
      context_i.update(sent_i)
    self._sent_info_list.clear()
    return context

class Reporter(object):
  """
  A base class for a reporter that collects reportable information, formats it and writes it to disk.
  """
  def create_report(self, **kwargs) -> None:
    """
    Create the report.

    The reporter should specify the arguments it needs explicitly, and should **kwargs in addition to handle extra
    (unused) arguments without crashing.

    Args:
      **kwargs: additional arguments
    """
    raise NotImplementedError("must be implemented by subclasses")
  @register_xnmt_event_assign
  def get_report_input(self, context={}):
    return context

class AttentionHtmlReporter(Reporter, Serializable):
  """
  Reporter that writes attention matrices to HTML.

  Args:
    report_path: Prefix for path to write HTML and image files to (i.e. directory + filename-prefix)
  """

  yaml_tag = "!AttentionHtmlReporter"

  @serializable_init
  def __init__(self, report_path: str):
    self.report_path = report_path
    self.html_tree = etree.Element('html')
    head = etree.SubElement(self.html_tree, 'head')
    title = etree.SubElement(head, 'title')
    title.text = 'Translation Report'
    self.html_body = etree.SubElement(self.html_tree, 'body')

  def create_report(self, idx: int, src: xnmt.input.SimpleSentenceInput, src_vocab: vocab.Vocab,
                    trg_vocab: vocab.Vocab, output: xnmt.output.Output, attentions:np.ndarray, ** kwargs):
    """
    Create report.

    Args:
      idx: number of sentence
      src: source-side input
      src_vocab: source-side vocabulary
      trg_vocab: source-side vocabulary
      output: generated output
      attentions: attention matrices
      **kwargs: arguments to be ignored
    """
    src_str = " ".join([src_vocab.i2w[src_token] for src_token in src])
    trg_str = " ".join(output.readable_actions())
    report_div = etree.SubElement(self.html_body, 'div')
    report = etree.SubElement(report_div, 'h1')
    report.text = f'Translation Report for Sentence {idx}'
    main_content = etree.SubElement(report_div, 'div', name='main_content')

    # Generating main content
    captions = ["Source Words", "Target Words"]
    inputs = [src_str, trg_str]
    for caption, sent in zip(captions, inputs):
      p = etree.SubElement(main_content, 'p')
      p.text = f"{caption}: {sent}"

    # Generating attention
    attention = etree.SubElement(main_content, 'p')
    att_text = etree.SubElement(attention, 'b')
    att_text.text = "Attention:"
    etree.SubElement(attention, 'br')
    attention_file = f"{self.report_path}.attention.{idx}.png"
    att_img = etree.SubElement(attention, 'img')
    att_img.attrib['src'] = os.path.basename(attention_file)
    att_img.attrib['alt'] = 'attention matrix'
    xnmt.plot.plot_attention(src_str.split(), trg_str.split(), attentions, file_name = attention_file)

    html_str = etree.tostring(self.html_tree, encoding='unicode', pretty_print=True)
    with open(self.report_path + '.html', 'w', encoding='utf-8') as f:
      f.write(html_str)

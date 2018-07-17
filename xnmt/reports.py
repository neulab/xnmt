"""
Reports gather inputs, outputs, and intermediate computations in a nicely formatted way for convenient manual inspection.

To support reporting, the models providing the data to be reported must subclass ``Reportable`` and call
``self.add_sent_for_report(d)`` with key/value pairs containing the data to be reported at the appropriate times.
If this causes a computational overhead, the boolean ``compute_report`` field should queried and extra computations
skipped if this field is ``False``.

Next, a ``Reporter`` needs to be specified that supports reports based on the previously created key/value pairs.
Reporters are passed to inference classes, so it's possible e.g. to report only at the final test decoding, or specify
a special reporting inference object that only looks at a handful of sentences, etc.

Note that currently reporting is only supported at test-time, not at training time.
"""

import os
import numpy as np
from lxml import etree
from typing import Any, Dict, Tuple

import xnmt.plot
import xnmt.output

from xnmt import vocab, util
from xnmt.events import register_xnmt_event_assign, handle_xnmt_event, register_xnmt_handler
from xnmt.persistence import Serializable, serializable_init, Ref
from xnmt.settings import settings

class Reportable(object):
  """
  Base class for classes that contribute information to a report.

  Making an arbitrary class reportable requires to do the following:

  - specify ``Reportable`` as base class
  - call this super class's ``__init__()``, or do ``@register_xnmt_handler`` manually
  - call ``self.add_sent_for_report(d)`` for each sentence, where d is a dictionary containing info to pass on to the
    reporter
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
                 ``create_report()`` method, and the values must be of the corresponding types.
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

    The reporter should specify the arguments it needs explicitly, and should specify ``kwargs`` in addition to handle
    extra (unused) arguments without crashing.

    Args:
      **kwargs: additional arguments
    """
    raise NotImplementedError("must be implemented by subclasses")
  @register_xnmt_event_assign
  def get_report_input(self, context={}) -> dict:
    return context

class HtmlReporter(Reporter):
  """
  A base class for reporters that produce HTML outputs that takes care of some common functionality.

  Args:
    report_path: Prefix for path to write HTML and image files to (i.e. directory + filename-prefix)
  """
  def __init__(self, report_path: str = settings.DEFAULT_REPORT_PREFIX) -> None:
    self.report_path = report_path
    self.html_tree = etree.Element('html')
    head = etree.SubElement(self.html_tree, 'head')
    title = etree.SubElement(head, 'title')
    title.text = 'Translation Report'
    self.html_body = etree.SubElement(self.html_tree, 'body')

  def start_sent(self, idx: int) :
    report_div = etree.SubElement(self.html_body, 'div')
    report = etree.SubElement(report_div, 'h1')
    report.text = f'Translation Report for Sentence {idx}'
    main_content = etree.SubElement(report_div, 'div', name='main_content')
    return main_content

  def write_html_tree(self) -> None:
    html_str = etree.tostring(self.html_tree, encoding='unicode', pretty_print=True)
    html_file_name = self.report_path + '.html'
    util.make_parent_dir(html_file_name)
    with open(html_file_name, 'w', encoding='utf-8') as f:
      f.write(html_str)

  def add_sent_src_trg(self, main_content, output, src, src_vocab) -> Tuple[str,str]:
    src_is_speech = isinstance(src, xnmt.input.ArrayInput)
    if src_is_speech:
      src_str = ""
    else:
      src_str = " ".join([src_vocab.i2w[src_token] for src_token in src])
    trg_str = " ".join(output.readable_actions())
    captions = ["Source Words", "Target Words"]
    inputs = [src_str, trg_str]
    for caption, sent in zip(captions, inputs):
      p = etree.SubElement(main_content, 'p')
      p.text = f"{caption}: {sent}"
    return src_str, trg_str


class AttentionHtmlReporter(HtmlReporter, Serializable):
  """
  Reporter that writes attention matrices to HTML.

  Args:
    report_path: Prefix for path to write HTML and image files to (i.e. directory + filename-prefix)
  """

  yaml_tag = "!AttentionHtmlReporter"

  @serializable_init
  def __init__(self, report_path: str = settings.DEFAULT_REPORT_PREFIX):
    super().__init__(report_path=report_path)

  def create_report(self, idx: int, src: xnmt.input.Input, src_vocab: vocab.Vocab,
                    trg_vocab: vocab.Vocab, output: xnmt.output.Output, attentions:np.ndarray, **kwargs) -> None:
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
    main_content = self.start_sent(idx)
    src_str, trg_str = self.add_sent_src_trg(main_content, output, src, src_vocab)
    self.add_atts(attentions, main_content, src, src_str, trg_str, idx)
    self.write_html_tree()

  def add_atts(self, attentions, main_content, src, src_str, trg_str, idx):
    src_is_speech = isinstance(src, xnmt.input.ArrayInput)
    if src_is_speech:
      src_feat_file = f"{self.report_path}.src_feat.{idx}.png"
      xnmt.plot.plot_speech_features(src.get_array(), file_name=src_feat_file)
    attention = etree.SubElement(main_content, 'p')
    att_text = etree.SubElement(attention, 'b')
    att_text.text = "Attention:"
    etree.SubElement(attention, 'br')
    attention_file = f"{self.report_path}.attention.{idx}.png"
    table = etree.SubElement(attention, 'table')
    table_tr = etree.SubElement(table, 'tr')
    table_td1 = etree.SubElement(table_tr, 'td')
    table_td2 = etree.SubElement(table_tr, 'td')
    if src_is_speech:
      att_img = etree.SubElement(table_td1, 'img')
      att_img.attrib['src'] = os.path.basename(src_feat_file)
      att_img.attrib['alt'] = 'speech features'
    att_img = etree.SubElement(table_td2, 'img')
    att_img.attrib['src'] = os.path.basename(attention_file)
    att_img.attrib['alt'] = 'attention matrix'
    xnmt.plot.plot_attention(src_str.split(), trg_str.split(), attentions, file_name=attention_file)


class SegmentationReporter(Reporter, Serializable):
  """
  A reporter to be used with the segmenting encoder.

  """
  yaml_tag = "!SegmentationReporter"

  @serializable_init
  @register_xnmt_handler
  def __init__(self, report_path: str=settings.DEFAULT_REPORT_PREFIX,
               compute_report=Ref("exp_global.compute_report", default=False)):
    self.report_path = report_path
    self.compute_report = compute_report
    self.report_fp = None

  def create_report(self, segment_actions, src_vocab, src, **kwargs):
    if self.compute_report:
      if self.report_fp is None:
        self.report_fp = open(self.report_path + ".segment", "w")

      actions = segment_actions[0][0]
      src = [src_vocab[x] for x in src]
      words = []
      start = 0
      for end in actions:
        words.append("".join(str(src[start:end+1])))
        start = end+1
      print(" ".join(words), file=self.report_fp)

  @handle_xnmt_event
  def on_end_inference(self):
    if hasattr(self, "report_fp") and self.report_fp:
      self.report_fp.close()


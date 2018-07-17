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
from typing import Any, Dict, Optional, Tuple
import numpy as np

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
    if not hasattr(self, "_sent_info_list"):
      return context
    if len(context)>0:
      assert len(context) == len(self._sent_info_list)
    else:
      context = []
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

class ReferenceDiffReporter(Reporter, Serializable):
  """
  Reporter that uses the CharCut tool for nicely displayed difference highlighting between outputs and references.

  The stand-alone tool can be found at https://github.com/alardill/CharCut

  Args:
    match_size: min match size in characters (set < 3 e.g. for Japanese or Chinese)
    alt_norm: alternative normalization scheme: use only the candidate's length for normalization
    report_path: Path to write HTML files to
  """
  yaml_tag = "!ReferenceDiffReporter"
  @serializable_init
  @register_xnmt_handler
  def __init__(self, match_size: int = 3, alt_norm: bool = False, report_path: str = settings.DEFAULT_REPORT_PATH) \
          -> None:
    self.match_size = match_size
    self.alt_norm = alt_norm
    self.report_path = report_path
    self.hyp_sents, self.ref_sents, self.src_sents = [], [], []

  def create_report(self, src: xnmt.input.Input, src_vocab: vocab.Vocab, trg_vocab: vocab.Vocab,
                    output: xnmt.output.Output, output_proc: xnmt.output.OutputProcessor, reference: str = None,
                    **kwargs) -> None:
    trg_str = output.apply_post_processor(output_proc)
    src_is_speech = isinstance(src, xnmt.input.ArrayInput)
    if not src_is_speech:
      src_str = " ".join([src_vocab.i2w[src_token] for src_token in src])
      self.src_sents.append(src_str)
    self.hyp_sents.append(trg_str)
    self.ref_sents.append(reference)

  @handle_xnmt_event
  def on_end_inference(self):
    if self.hyp_sents:
      html_filename = f"{self.report_path}/charcut.html"
      util.make_parent_dir(html_filename)
      import xnmt.thirdparty.charcut.charcut as charcut
      args = util.ArgClass(html_output_file=html_filename, match_size=self.match_size, alt_norm=self.alt_norm)
      aligned_segs = charcut.load_input_segs(cand_segs=self.hyp_sents,
                                             ref_segs=self.ref_sents,
                                             src_segs=self.src_sents)
      charcut.run_on(aligned_segs, args)
      self.hyp_sents, self.ref_sents, self.src_sents = [], [], []

class CompareMtReporter(Reporter, Serializable):
  """
  Reporter that uses the compare-mt.py script to analyze and compare MT results.

  The stand-alone tool can be found at https://github.com/neubig/util-scripts

  Args:
    out2_file: A path to another system output. Add only if you want to compare outputs from two systems.
    train_file: A link to the training corpus target file
    train_counts: A link to the training word frequency counts as a tab-separated "word\\tfreq" file
    alpha: A smoothing coefficient to control how much the model focuses on low- and high-frequency events.
           1.0 should be fine most of the time.
    ngram: Maximum length of n-grams.
    sent_size: How many sentences to print.
    ngram_size: How many n-grams to print.

    report_path: Path to write report files to
  """
  yaml_tag = "!CompareMtReporter"
  @serializable_init
  @register_xnmt_handler
  def __init__(self, out2_file: Optional[str] = None, train_file: Optional[str] = None,
               train_counts: Optional[str] = None, alpha: float = 1.0, ngram: int = 4, ngram_size: int = 50,
               sent_size: int = 10, report_path: str = settings.DEFAULT_REPORT_PATH) -> None:
    self.out2_file = out2_file
    self.train_file = train_file
    self.train_counts = train_counts
    self.alpha = alpha
    self.ngram = ngram
    self.ngram_size = ngram_size
    self.sent_size = sent_size
    self.report_path = report_path
    self.hyp_sents, self.ref_sents = [], []

  def create_report(self, trg_vocab: vocab.Vocab, output: xnmt.output.Output, output_proc: xnmt.output.OutputProcessor,
                    reference: str, **kwargs) -> None:
    trg_str = output.apply_post_processor(output_proc)
    self.hyp_sents.append(trg_str)
    self.ref_sents.append(reference)

  @handle_xnmt_event
  def on_end_inference(self):
    if self.hyp_sents:
      ref_filename = f"{self.report_path}/tmp/compare-mt.ref"
      out_filename = f"{self.report_path}/tmp/compare-mt.out"
      util.make_parent_dir(out_filename)
      with open(ref_filename, "w") as fout:
        for l in self.ref_sents: fout.write(f"{l.strip()}\n")
      with open(out_filename, "w") as fout:
        for l in self.hyp_sents: fout.write(f"{l.strip()}\n")
      import xnmt.thirdparty.comparemt.compare_mt as compare_mt
      args = util.ArgClass(ref_file = ref_filename,
                           out_file = out_filename,
                           out2_file = self.out2_file,
                           train_file = self.train_file,
                           train_counts = self.train_counts,
                           alpha = self.alpha,
                           ngram = self.ngram,
                           ngram_size = self.ngram_size,
                           sent_size = self.sent_size)
      out_lines = compare_mt.main(args)
      report_filename = f"{self.report_path}/compare-mt.txt"
      util.make_parent_dir(report_filename)
      with open(report_filename, "w") as fout:
        for l in out_lines: fout.write(f"{l}\n")
      self.hyp_sents, self.ref_sents, self.src_sents = [], [], []


class HtmlReporter(Reporter):
  """
  A base class for reporters that produce HTML outputs that takes care of some common functionality.

  Args:
    report_name: prefix for report files
    report_path: Path to write HTML and image files to
  """
  def __init__(self, report_name: str, report_path: str = settings.DEFAULT_REPORT_PATH) -> None:
    self.report_name = report_name
    self.report_path = report_path
    self.html_tree = etree.Element('html')
    meta = etree.SubElement(self.html_tree, 'meta')
    meta.attrib['charset'] = 'UTF-8'
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
    html_file_name = f"{self.report_path}/{self.report_name}.html"
    util.make_parent_dir(html_file_name)
    with open(html_file_name, 'w', encoding='utf-8') as f:
      f.write(html_str)

  def add_sent_in_out(self, main_content, output, output_proc: xnmt.output.OutputProcessor, src, src_vocab,
                      reference: Optional[str]=None) -> Tuple[str, str]:
    src_is_speech = isinstance(src, xnmt.input.ArrayInput)
    if src_is_speech:
      src_tokens = []
    else:
      src_tokens = [src_vocab.i2w[src_token] for src_token in src]
    src_str = " ".join(src_tokens)
    trg_str = output.apply_post_processor(output_proc)
    captions, inputs = [], []
    if not src_is_speech:
      captions.append("Source Words")
      inputs.append(src_str)
    captions.append("Output Words")
    inputs.append(trg_str)
    if reference:
      captions.append("Reference Words")
      inputs.append(reference)
    for caption, sent in zip(captions, inputs):
      p = etree.SubElement(main_content, 'p')
      b = etree.SubElement(p, 'b')
      c = etree.SubElement(p, 'span')
      b.text = f"{caption}: "
      c.text = sent
    trg_tokens = output.readable_actions()
    return src_tokens, trg_tokens


class AttentionReporter(HtmlReporter, Serializable):
  """
  Reporter that writes attention matrices to HTML.

  Args:
    report_path: Path to write HTML and image files to
  """

  yaml_tag = "!AttentionReporter"

  @register_xnmt_handler
  @serializable_init
  def __init__(self, report_name: str = "attention", report_path: str = settings.DEFAULT_REPORT_PATH):
    super().__init__(report_name=report_name, report_path=report_path)

  def create_report(self, idx: int, src: xnmt.input.Input, src_vocab: vocab.Vocab,
                    trg_vocab: vocab.Vocab, output: xnmt.output.Output, output_proc: xnmt.output.OutputProcessor,
                    attentions: np.ndarray, reference: Optional[str] = None, **kwargs) -> None:
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
    src_tokens, trg_tokens = self.add_sent_in_out(main_content, output, output_proc, src, src_vocab, reference)
    self.add_atts(attentions, main_content, src, src_tokens, trg_tokens, idx)

  @handle_xnmt_event
  def on_end_inference(self):
    self.write_html_tree()

  def add_atts(self, attentions, main_content, src, src_tokens, trg_tokens, idx, desc="Attentions"):
    src_is_speech = isinstance(src, xnmt.input.ArrayInput)
    if src_is_speech:
      src_feat_file = f"{self.report_path}/img/attention.src_feat.{idx}.png"
      xnmt.plot.plot_speech_features(src.get_array(), file_name=src_feat_file)
    attention = etree.SubElement(main_content, 'p')
    att_text = etree.SubElement(attention, 'b')
    att_text.text = f"{desc}:"
    etree.SubElement(attention, 'br')
    attention_file = f"{self.report_path}/img/attention.{util.valid_filename(desc).lower()}.{idx}.png"
    table = etree.SubElement(attention, 'table')
    table_tr = etree.SubElement(table, 'tr')
    table_td1 = etree.SubElement(table_tr, 'td')
    table_td2 = etree.SubElement(table_tr, 'td')
    if src_is_speech:
      att_img = etree.SubElement(table_td1, 'img')
      att_img.attrib['src'] = "img/" + os.path.basename(src_feat_file)
      att_img.attrib['alt'] = 'speech features'
    att_img = etree.SubElement(table_td2, 'img')
    att_img.attrib['src'] = "img/" + os.path.basename(attention_file)
    att_img.attrib['alt'] = 'attention matrix'
    xnmt.plot.plot_attention(src_tokens, trg_tokens, attentions, file_name=attention_file)


class SegmentationReporter(Reporter, Serializable):
  """
  A reporter to be used with the segmenting encoder.

  Args:
    report_path: Path to write text files to
  """
  yaml_tag = "!SegmentationReporter"

  @serializable_init
  @register_xnmt_handler
  def __init__(self, report_path: str=settings.DEFAULT_REPORT_PATH):
    self.report_path = report_path
    self.report_fp = None

  def create_report(self, segment_actions, src_vocab, src, **kwargs):
    if self.report_fp is None:
      report_path = self.report_path + "/segment.txt"
      util.make_parent_dir(report_path)
      self.report_fp = open(report_path, "w")

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


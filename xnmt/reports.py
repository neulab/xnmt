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
import math
from typing import Any, Dict, List, Optional, Sequence, Union

from bs4 import BeautifulSoup as bs

import matplotlib
matplotlib.use('Agg')
import numpy as np

import xnmt.plot
import xnmt.output
import xnmt.input
from xnmt import vocab, util
from xnmt.events import register_xnmt_event_assign, handle_xnmt_event, register_xnmt_handler
from xnmt.persistence import Serializable, serializable_init
from xnmt.settings import settings
import xnmt.thirdparty.charcut.charcut as charcut

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
    self.html_contents = ["<html><meta charset='UTF-8' /><head><title>Translation Report</title></head><body>"]
    self.html_contents.append("""
      <style>
        body {font-family: sans-serif; font-size: 11pt;}
        table, td, th {border-spacing: 0;}
        th {padding: 10px;}
        td {padding: 5px;}
        th {border-top: solid black 2px; font-weight: normal;}
        .tophead {border-bottom: solid black 1px;}
        .src {font-style: oblique;}
        .trg {font-family: Consolas, monospace;}
        .del {font-weight: bold; color: #f00000;}
        .ins {font-weight: bold; color: #0040ff;}
        .shift {font-weight: bold;}
        .match {}
        .mainrow {border-top: solid black 1px; padding: 1em;}
        .midrow {border-bottom: dotted gray 1px;}
        .seghead {color: gray; text-align: right;}
        .score {font-family: Consolas, monospace; text-align: right; font-size: large;}
        .detail {font-size: xx-small; color: gray;}
      </style>
      <script>
        function enter(cls) {
          var elts = document.getElementsByClassName(cls);
          for (var i=0; i<elts.length; i++)
            elts[i].style.backgroundColor = "yellow";
        }
        function leave(cls) {
          var elts = document.getElementsByClassName(cls);
          for (var i=0; i<elts.length; i++)
            elts[i].style.backgroundColor = "transparent";
        }
      </script>
   """)

  def add_sent_heading(self, idx: int):
    self.html_contents.append(f"<h1>Translation Report for Sentence {idx}</h1>")
    self.html_contents.append("<table>")

  def finish_sent(self):
    self.html_contents.append("</table>")

  def finish_html_doc(self):
    self.html_contents.append("</body></html>")

  def write_html(self) -> None:
    html_str = "\n".join(self.html_contents)
    soup = bs(html_str, "lxml")
    pretty_html = soup.prettify()
    html_file_name = f"{self.report_path}/{self.report_name}.html"
    util.make_parent_dir(html_file_name)
    with open(html_file_name, 'w', encoding='utf-8') as f:
      f.write(pretty_html)

  def get_tokens(self, output=None, inp=None, inp_vocab=None) -> List[str]:
    assert output is None or (inp is None and inp_vocab is None)
    if output:
      return output.readable_actions()
    else:
      src_is_speech = isinstance(inp, xnmt.input.ArrayInput)
      if src_is_speech:
        src_tokens = []
      else:
        src_tokens = [inp_vocab.i2w[src_token] for src_token in inp]
      return src_tokens

  def get_strings(self, src_tokens, output, output_proc: xnmt.output.OutputProcessor):
    trg_str = output.apply_post_processor(output_proc)
    src_str = " ".join(src_tokens)
    return src_str, trg_str

  def add_fields_if_set(self, fields):
    html_ret = ""
    for key, val in fields.items():
      if val:
        html_ret += f"<tr><td class='seghead'>{key}:</td><td>{val}</td></tr>"
    if html_ret:
      self.html_contents.append(html_ret)

  def add_charcut_diff(self, trg_str, reference, match_size=3, alt_norm=False, mt_label="MT:", ref_label="Ref:"):
    aligned_segs = charcut.load_input_segs(cand_segs=[trg_str],
                                           ref_segs=[reference])
    styled_ops = [charcut.compare_segments(cand, ref, match_size)
                  for seg_id, _, _, cand, ref in aligned_segs]

    seg_scores = list(charcut.score_all(aligned_segs, styled_ops, alt_norm))
    # doc_cost = sum(cost for cost, _ in seg_scores)
    # doc_div = sum(div for _, div in seg_scores)

    self.html_contents.append(charcut.segs2html(aligned_segs[0], styled_ops[0], seg_scores[0], mt_label=mt_label,
                                                ref_label=ref_label, use_id_col=False))


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
      reference: reference string
      **kwargs: arguments to be ignored
    """
    self.add_sent_heading(idx)
    src_tokens = self.get_tokens(inp=src, inp_vocab=src_vocab)
    trg_tokens = self.get_tokens(output=output)
    src_str, trg_str = self.get_strings(src_tokens=src_tokens, output=output, output_proc=output_proc)
    self.add_charcut_diff(trg_str, reference)
    self.add_fields_if_set({"Src" : src_str})
    self.add_atts(attentions, src.get_array() if isinstance(src, xnmt.input.ArrayInput) else src_tokens,
                  trg_tokens, idx)
    self.finish_sent()

  @handle_xnmt_event
  def on_end_inference(self):
    self.finish_html_doc()
    self.write_html()

  def add_atts(self,
               attentions: np.ndarray,
               src_tokens: Union[Sequence[str], np.ndarray],
               trg_tokens: Sequence[str],
               idx: int,
               desc: str = "Attentions") -> None:
    """
    Add attention matrix to HTML code.

    Args:
      attentions: numpy array of dimensions (src_len x trg_len)
      src_tokens: list of strings (case of src text) or numpy array of dims (nfeat x speech_len) (case of src speech)
      trg_tokens: list of string tokens
      idx: sentence no
      desc: readable description
    """
    src_is_speech = isinstance(src_tokens, np.ndarray)
    size_x = math.log(len(trg_tokens)+2) * 3
    if src_is_speech:
      size_y = math.log(src_tokens.shape[1]+2)
    else:
      size_y = math.log(len(src_tokens)+2) * 3
    attention_file = f"{self.report_path}/img/attention.{util.valid_filename(desc).lower()}.{idx}.png"
    html_att = f'<tr><td class="seghead">{desc}:</td><td></td></tr>' \
               f'<tr><td colspan="2" align="left"><img src="img/{os.path.basename(attention_file)}" alt="attention matrix" /></td></tr>'
    xnmt.plot.plot_attention(src_words=src_tokens, trg_words=trg_tokens, attention_matrix=attentions,
                             file_name=attention_file, size_x=size_x, size_y=size_y)
    self.html_contents.append(html_att)


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


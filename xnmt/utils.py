import argparse
import os
import time
import math
import unicodedata
import string
import functools
import numbers
import errno
from typing import List, MutableMapping, Optional

import numpy as np

import xnmt
from xnmt import logger
from xnmt.settings import settings

def print_cg_conditional() -> None:
  if settings.PRINT_CG_ON_ERROR:
    if xnmt.backend_dynet:
      import dynet as dy
      dy.print_text_graphviz()
    else:
      logger.warning("CG printing not implemented with Torch backend")
      # TODO: print_cg_torch() works but needs a loss variable which we don't have access to here..

def print_cg_torch(loss_var, render=False) -> None:
  import torchviz
  from xnmt import param_collections
  dot = torchviz.make_dot(loss_var, dict(param_collections.ParamManager.global_collection().named_parameters()))
  if render:
    dot.render('computation_graph.gv')
  else:
    dot.save('computation_graph.gv')


def make_parent_dir(filename: str) -> None:
  if not os.path.exists(os.path.dirname(filename) or "."):
    try:
      os.makedirs(os.path.dirname(filename))
    except OSError as exc: # Guard against race condition
      if exc.errno != errno.EEXIST:
        raise


_valid_filename_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)

def valid_filename(filename: str, whitelist: str = _valid_filename_chars, replace: str = ' '):
  # from https://gist.github.com/wassname/1393c4a57cfcbf03641dbc31886123b8
  # replace spaces
  for r in replace:
    filename = filename.replace(r, '_')
  # keep only valid ascii chars
  cleaned_filename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode()
  # keep only whitelisted chars
  return ''.join(c for c in cleaned_filename if c in whitelist)

def format_time(seconds: numbers.Number) -> str:
  return "{}-{}".format(int(seconds) // 86400, time.strftime("%H:%M:%S", time.gmtime(seconds)))

def log_readable_and_tensorboard(template: str,
                                 args: MutableMapping,
                                 n_iter: numbers.Integral,
                                 fractional_epoch: numbers.Real,
                                 data_name: str,
                                 task_name: Optional[str] = None,
                                 **kwargs) -> None:
  log_args = dict(args)
  log_args["data_name"] = data_name
  log_args["epoch"] = fractional_epoch
  log_args["n_iter"] = n_iter
  log_args.update(kwargs)
  if task_name: log_args["task_name"] = task_name
  logger.info(template.format(**log_args), extra=log_args)

  if settings.USE_TENSORBOARD:
    from xnmt.tee import tensorboard_writer
    if tensorboard_writer.writer is not None:
      tensorboard_writer.add_scalars(f"{task_name}/{data_name}" if task_name else data_name,
                                     args,
                                     n_iter)

class RollingStatistic(object):
  """
  Efficient computation of rolling average and standard deviations.

  Code adopted from http://jonisalonen.com/2014/efficient-and-accurate-rolling-standard-deviation/
  """

  def __init__(self, window_size: numbers.Integral = 100) -> None:
    self.N = window_size
    self.average = None
    self.variance = None
    self.stddev = None
    self.vals = []

  def update(self, new: numbers.Real) -> None:
    self.vals.append(new)
    if len(self.vals) == self.N:
      self.average = np.average(self.vals)
      self.variance = np.var(self.vals)
      self.stddev = math.sqrt(self.variance)
    elif len(self.vals) == self.N+1:
      old = self.vals.pop(0)
      oldavg = self.average
      newavg = oldavg + (new - old) / self.N
      self.average = newavg
      self.variance += (new - old) * (new - newavg + old - oldavg) / (self.N - 1)
      try:
        self.stddev = math.sqrt(self.variance)
      except ValueError:
        # This happens in case of numerical issues in computation of rolling stddev, but we can easily resolve this
        # through full re-computation
        self.variance = np.var(self.vals)
        self.stddev = math.sqrt(self.variance)
    else:
      assert len(self.vals) < self.N

class ReportOnException(object):
  """
  Context manager that prints debug information when an exception occurs.

  Args:
    args: a dictionary containing debug info. Callable items are called, other items are passed to logger.error()
  """
  def __init__(self, args: dict) -> None:
    self.args = args
  def __enter__(self):
    return self
  def __exit__(self, et, ev, traceback):
    if et is not None: # exception occurred
      logger.error("------ Error Report ------")
      for key, val in self.args.items():
        logger.error(f"*** {key} ***")
        if callable(val):
          val()
        else:
          logger.error(str(val))

class SkipOutOfMemory(object):
  """
  Context manager that skips occasional out of memory exceptions.

  Args:
    active: whether out-of-memory errors should be skipped.
    max_ratio: don't skip if out-of-memory exceptions happen more often than this ratio.
    warn_msg: the message to log when out of memory errors are skipped.
  """
  def __init__(self, active: bool, max_ratio: float = 0.02, warn_msg: str = "Out of memory, skip minibatch.") -> None:
    self.active = active
    self.max_ratio = max_ratio
    self.warn_msg = warn_msg
    self.call_count = 0
    self.exc_count = 0
  def __enter__(self):
    self.call_count += 1
    return self
  def __exit__(self, et, ev, traceback):
    if self.active:
      is_oom = et==RuntimeError and "out of memory" in str(ev)
      if is_oom:
        self.exc_count += 1
        should_ignore = float(self.exc_count)/float(self.call_count+50) <= self.max_ratio
        if should_ignore:
          logger.warn(self.warn_msg)
          return True

class dummy_context_mgr(object):
  def __enter__(self):
    return None
  def __exit__(self, exc_type, exc_value, traceback):
    return False

class ArgClass(object):
  """
  A class that converts dictionary items to class attributes in order to support argparse-like configuration.

  Can be useful e.g. when integrating standalone-scripts into XNMT.
  """
  def __init__(self, **kwargs) -> None:
    for key in kwargs: setattr(self, key, kwargs[key])

@functools.lru_cache()
def cached_file_lines(file_name: str) -> List[str]:
  with open(file_name) as f:
    ret = f.readlines()
  return ret

def add_backend_argparse(argparser: argparse.ArgumentParser) -> None:
  if xnmt.backend_torch:
    argparser.add_argument("--gpu", action='store_true', help="use GPU acceleration")
    argparser.add_argument("--seed", type=int, help="set random seed for XNMT and DyNet/Torch backend")
  if xnmt.backend_dynet:
    argparser.add_argument("--dynet-mem", type=str)
    argparser.add_argument("--dynet-seed", type=int, help="set random seed for XNMT and DyNet/Torch backend")
    argparser.add_argument("--dynet-autobatch", type=int)
    argparser.add_argument("--dynet-devices", type=str)
    argparser.add_argument("--dynet-viz", action='store_true', help="use visualization")
    argparser.add_argument("--dynet-gpu", action='store_true', help="use GPU acceleration")
    argparser.add_argument("--dynet-gpu-ids", type=int)
    argparser.add_argument("--dynet-gpus", type=int)
    argparser.add_argument("--dynet-weight-decay", type=float)
    argparser.add_argument("--dynet-profiling", type=int)

def has_cython() -> bool:
  try:
    from xnmt.cython import xnmt_cython
    return True
  except:
    return False


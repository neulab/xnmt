import os
import time
import math
import unicodedata
import string
import functools

import numpy as np
import dynet as dy

from xnmt import logger, yaml_logger
from xnmt.settings import settings

def print_cg_conditional():
  if settings.PRINT_CG_ON_ERROR:
    dy.print_text_graphviz()

def make_parent_dir(filename):
  if not os.path.exists(os.path.dirname(filename) or "."):
    try:
      os.makedirs(os.path.dirname(filename))
    except OSError as exc: # Guard against race condition
      if exc.errno != os.errno.EEXIST:
        raise


_valid_filename_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)

def valid_filename(filename, whitelist=_valid_filename_chars, replace=' '):
  # from https://gist.github.com/wassname/1393c4a57cfcbf03641dbc31886123b8
  # replace spaces
  for r in replace:
    filename = filename.replace(r, '_')
  # keep only valid ascii chars
  cleaned_filename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode()
  # keep only whitelisted chars
  return ''.join(c for c in cleaned_filename if c in whitelist)

def format_time(seconds):
  return "{}-{}".format(int(seconds) // 86400,
                        time.strftime("%H:%M:%S", time.gmtime(seconds)))

def log_readable_and_tensorboard(template, args, n_iter, data_name, task_name=None, **kwargs):
  log_args = dict(args)
  log_args["data_name"] = data_name
  log_args["epoch"] = n_iter
  log_args.update(kwargs)
  if task_name: log_args["task_name"] = task_name
  logger.info(template.format(**log_args), extra=log_args)

  from xnmt.tee import tensorboard_writer
  tensorboard_writer.add_scalars(f"{task_name}/{data_name}" if task_name else data_name,
                                 args,
                                 n_iter)

class RollingStatistic(object):
  """
  Efficient computation of rolling average and standard deviations.

  Code adopted from http://jonisalonen.com/2014/efficient-and-accurate-rolling-standard-deviation/
  """

  def __init__(self, window_size=100):
    self.N = window_size
    self.average = None
    self.variance = None
    self.stddev = None
    self.vals = []

  def update(self, new):
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
      self.stddev = math.sqrt(self.variance)
    else:
      assert len(self.vals) < self.N

class ReportOnException(object):
  """
  Context manager that prints debug information when an exception occurs.

  Args:
    args: a dictionary containing debug info. Callable items are called, other items are passed to logger.error()
  """
  def __init__(self, args: dict):
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

class ArgClass(object):
  """
  A class that converts dictionary items to class attributes in order to support argparse-like configuration.

  Can be useful e.g. when integrating standalone-scripts into XNMT.
  """
  def __init__(self, **kwargs):
    for key in kwargs: setattr(self, key, kwargs[key])

@functools.lru_cache()
def cached_file_lines(file_name):
  with open(file_name) as f:
    ret = f.readlines()
  return ret

def add_dynet_argparse(argparser):
  argparser.add_argument("--dynet-mem", type=str)
  argparser.add_argument("--dynet-seed", type=int, help="set random seed for DyNet and XNMT.")
  argparser.add_argument("--dynet-autobatch", type=int)
  argparser.add_argument("--dynet-devices", type=str)
  argparser.add_argument("--dynet-viz", action='store_true', help="use visualization")
  argparser.add_argument("--dynet-gpu", action='store_true', help="use GPU acceleration")
  argparser.add_argument("--dynet-gpu-ids", type=int)
  argparser.add_argument("--dynet-gpus", type=int)
  argparser.add_argument("--dynet-weight-decay", type=float)
  argparser.add_argument("--dynet-profiling", type=int)

def has_cython():
  try:
    from xnmt.cython import xnmt_cython
    return True
  except:
    return False


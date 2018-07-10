import os
import time
import math

import numpy as np

from xnmt import logger, yaml_logger

def make_parent_dir(filename):
  if not os.path.exists(os.path.dirname(filename) or "."):
    try:
      os.makedirs(os.path.dirname(filename))
    except OSError as exc: # Guard against race condition
      if exc.errno != os.errno.EEXIST:
        raise

def format_time(seconds):
  return "{}-{}".format(int(seconds) // 86400,
                        time.strftime("%H:%M:%S", time.gmtime(seconds)))

def log_readable_and_structured(template, args, task_name=None):
  if task_name: args["task_name"] = task_name
  logger.info(template.format(**args), extra=args)
  yaml_logger.info(args)

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

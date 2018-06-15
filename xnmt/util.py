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

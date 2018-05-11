import os
from typing import TypeVar, Sequence, Union, Dict,List
import time

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

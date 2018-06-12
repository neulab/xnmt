import sys, os
import logging

import yaml

from xnmt.settings import settings
from xnmt.util import make_parent_dir
import xnmt.git_rev

STD_OUTPUT_LEVELNO = 35

class NoErrorFilter(logging.Filter):
  def filter(self, record):
    return not record.levelno in [logging.WARNING, logging.ERROR, logging.CRITICAL]

class ErrorOnlyFilter(logging.Filter):
  def filter(self, record):
    return record.levelno in [logging.WARNING, logging.ERROR, logging.CRITICAL]

class MainFormatter(logging.Formatter):
  def format(self, record):
    task_name = getattr(record, "task_name", None)
    if task_name:
      record.msg = f"[{record.task_name}] {record.msg}"
    if record.levelno in [logging.WARNING, logging.ERROR, logging.CRITICAL]:
      record.msg = "\n".join([f"{record.levelname}: {line}" for line in record.msg.split("\n")])
    return super().format(record)

class YamlFormatter(logging.Formatter):
  def format(self, record):
    record.msg = yaml.dump([record.msg]).rstrip()
    return super().format(record)


sys_std_out = sys.stdout
sys_std_err = sys.stderr
logging.addLevelName(STD_OUTPUT_LEVELNO, "STD_OUTPUT")

logger = logging.getLogger('xnmt')
logger.setLevel(min(logging._checkLevel(settings.LOG_LEVEL_CONSOLE), logging._checkLevel(settings.LOG_LEVEL_FILE)))

ch_out = logging.StreamHandler(sys_std_out)
ch_out.setLevel(settings.LOG_LEVEL_CONSOLE)
ch_out.addFilter(NoErrorFilter())
ch_out.setFormatter(MainFormatter())
ch_err = logging.StreamHandler(sys_std_err)
ch_err.setLevel(settings.LOG_LEVEL_CONSOLE)
ch_err.addFilter(ErrorOnlyFilter())
ch_err.setFormatter(MainFormatter())
logger.addHandler(ch_out)
logger.addHandler(ch_err)

yaml_logger = logging.getLogger("yaml")
yaml_logger.setLevel(logging.INFO)

_preamble_content = []
def log_preamble(log_line, level=logging.INFO):
  """
  Logs a message when no out_file is set. Once out_file is set, all preamble strings will be prepended to the out_file.
  Args:
    log_line: log message
    level: log level
  """
  _preamble_content.append(log_line)
  logger.log(level=level, msg=log_line)

def set_out_file(out_file):
  """
  Set the file to log to. Before calling this, logs are only passed to stdout/stderr.
  Args:
    out_file: file name
  """
  unset_out_file()
  make_parent_dir(out_file)
  with open(out_file, mode="w") as f_out:
    for line in _preamble_content:
      f_out.write(f"{line}\n")
  fh = logging.FileHandler(out_file, encoding="utf-8")
  fh.setLevel(settings.LOG_LEVEL_FILE)
  fh.setFormatter(MainFormatter())
  logger.addHandler(fh)
  yaml_fh = logging.FileHandler(f"{out_file}.yaml", mode='w', encoding="utf-8")
  yaml_fh.setLevel(logging.DEBUG)
  yaml_fh.setFormatter(YamlFormatter())
  yaml_fh.setLevel(logging.DEBUG)
  yaml_logger.addHandler(yaml_fh)

def unset_out_file():
  """
  Unset the file to log to.
  """
  for hdlr in list(logger.handlers):
    if isinstance(hdlr, logging.FileHandler):
      hdlr.close()
      logger.removeHandler(hdlr)
  for hdlr in list(yaml_logger.handlers):
    if isinstance(hdlr, logging.FileHandler):
      hdlr.close()
      yaml_logger.removeHandler(hdlr)

class Tee(object):
  def __init__(self, indent=0, error=False):
    self.logger = logger
    self.stdstream = sys_std_err if error else sys_std_out
    self.indent = indent
    self.error = error
    if error:
      sys.stderr = self
    else:
      sys.stdout = self

  def close(self):
    if self.error:
      sys.stderr = self.stdstream
    else:
      sys.stdout = self.stdstream

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()

  def write(self, data):
    if data.strip()!="":
      if self.error:
        self.logger.error(data.rstrip())
      else:
        self.logger.log(STD_OUTPUT_LEVELNO, data.rstrip())
      self.flush()

  def flush(self):
    self.stdstream.flush()

  def getvalue(self):
    return self.stdstream.getvalue()

def get_git_revision():
  if xnmt.git_rev.CUR_GIT_REVISION: return xnmt.git_rev.CUR_GIT_REVISION
  from subprocess import CalledProcessError, check_output
  try:
    command = 'git rev-parse --short HEAD'
    rev = check_output(command.split(u' '), cwd=os.path.dirname(__file__)).decode('ascii').strip()
  except (CalledProcessError, OSError):
    rev = None
  return rev

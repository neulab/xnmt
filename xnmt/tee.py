import sys, os
import logging
import numbers
from typing import Optional

import tensorboardX
import yaml

from xnmt.settings import settings
from xnmt import utils
import xnmt.git_rev

STD_OUTPUT_LEVELNO = 35

class NoErrorFilter(logging.Filter):
  def filter(self, record: logging.LogRecord) -> bool:
    return not record.levelno in [logging.WARNING, logging.ERROR, logging.CRITICAL]

class ErrorOnlyFilter(logging.Filter):
  def filter(self, record: logging.LogRecord) -> bool:
    return record.levelno in [logging.WARNING, logging.ERROR, logging.CRITICAL]

class MainFormatter(logging.Formatter):
  def format(self, record: logging.LogRecord) -> str:
    task_name = getattr(record, "task_name", None)
    if task_name:
      record.msg = f"[{record.task_name}] {record.msg}"
    if record.levelno in [logging.WARNING, logging.ERROR, logging.CRITICAL]:
      record.msg = "\n".join([f"{record.levelname}: {line}" for line in record.msg.split("\n")])
    return super().format(record)

class YamlFormatter(logging.Formatter):
  def format(self, record: logging.LogRecord) -> str:
    record.msg = yaml.dump([record.msg]).rstrip()
    return super().format(record)


sys_std_out = sys.stdout
sys_std_err = sys.stderr
logging.addLevelName(STD_OUTPUT_LEVELNO, "STD_OUTPUT")

logger = logging.getLogger('xnmt')
logger.setLevel(min(logging._checkLevel(settings.LOG_LEVEL_CONSOLE), logging._checkLevel(settings.LOG_LEVEL_FILE)))
logger_file = logging.getLogger('xnmt_file')
logger_file.setLevel(min(logging._checkLevel(settings.LOG_LEVEL_CONSOLE), logging._checkLevel(settings.LOG_LEVEL_FILE)))

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

class TensorboardCustomWriter(object):
  def __init__(self) -> None:
    self.out_file_name = None
    self.writer = None
    self.exp_name = None
  def set_out_file(self, out_file_name: str, exp_name: str) -> None:
    self.out_file_name = out_file_name
    self.exp_name = exp_name
    self.writer = tensorboardX.SummaryWriter(log_dir=f"{out_file_name}")
  def unset_out_file(self) -> None:
    self.out_file_name = None
  def add_scalars(self, name: str, *args, **kwargs):
    return self.writer.add_scalars(f"{self.exp_name}/{name}", *args, **kwargs)

tensorboard_writer = TensorboardCustomWriter()

_preamble_content = []
def log_preamble(log_line: str, level: numbers.Integral = logging.INFO) -> None:
  """
  Log a message when no out_file is set. Once out_file is set, all preamble strings will be prepended to the out_file.

  Args:
    log_line: log message
    level: log level
  """
  _preamble_content.append(log_line)
  logger.log(level=level, msg=log_line)

def set_out_file(out_file: str, exp_name: str) -> None:
  """
  Set the file to log to. Before calling this, logs are only passed to stdout/stderr.
  Args:
    out_file: file name
    exp_name: name of experiment
  """
  unset_out_file()
  utils.make_parent_dir(out_file)
  with open(out_file, mode="w") as f_out:
    for line in _preamble_content:
      f_out.write(f"{line}\n")
  fh = logging.FileHandler(out_file, encoding="utf-8")
  fh.setLevel(settings.LOG_LEVEL_FILE)
  fh.setFormatter(MainFormatter())
  logger.addHandler(fh)
  logger_file.addHandler(fh)
  yaml_fh = logging.FileHandler(f"{out_file}.yaml", mode='w', encoding="utf-8")
  yaml_fh.setLevel(logging.DEBUG)
  yaml_fh.setFormatter(YamlFormatter())
  yaml_fh.setLevel(logging.DEBUG)
  yaml_logger.addHandler(yaml_fh)
  tensorboard_writer.set_out_file(f"{out_file}.tb", exp_name=exp_name)

def unset_out_file() -> None:
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
  for hdlr in list(logger_file.handlers):
    hdlr.close()
    logger_file.removeHandler(hdlr)
  tensorboard_writer.unset_out_file()

class Tee(object):
  def __init__(self, indent: numbers.Integral = 0, error: bool = False) -> None:
    self.logger = logger
    self.stdstream = sys_std_err if error else sys_std_out
    self.indent = indent
    self.error = error
    if error:
      sys.stderr = self
    else:
      sys.stdout = self

  def close(self) -> None:
    if self.error:
      sys.stderr = self.stdstream
    else:
      sys.stdout = self.stdstream

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()

  def write(self, data: str) -> None:
    if data.strip()!="":
      if self.error:
        self.logger.error(data.rstrip())
      else:
        self.logger.log(STD_OUTPUT_LEVELNO, data.rstrip())
      self.flush()

  def flush(self) -> None:
    self.stdstream.flush()

  def getvalue(self):
    return self.stdstream.getvalue()

def get_git_revision() -> Optional[str]:
  if xnmt.git_rev.CUR_GIT_REVISION: return xnmt.git_rev.CUR_GIT_REVISION
  from subprocess import CalledProcessError, check_output
  try:
    command = 'git rev-parse --short HEAD'
    rev = check_output(command.split(u' '), cwd=os.path.dirname(__file__)).decode('ascii').strip()
  except (CalledProcessError, OSError):
    rev = None
  return rev

import sys, os
import logging

from simple_settings import settings
import yaml

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

def set_out_file(out_file):
  unset_out_file()
  dirname = os.path.dirname(out_file)
  if dirname and not os.path.exists(dirname):
    os.makedirs(dirname)  
  fh = logging.FileHandler(out_file, mode='w')
  fh.setLevel(settings.LOG_LEVEL_FILE)
  fh.setFormatter(MainFormatter())
  logger.addHandler(fh)
  yaml_fh = logging.FileHandler(f"{out_file}.yaml", mode='w')
  yaml_fh.setLevel(logging.DEBUG)
  yaml_fh.setFormatter(YamlFormatter())
  yaml_fh.setLevel(logging.DEBUG)
  yaml_logger.addHandler(yaml_fh)

def unset_out_file():
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

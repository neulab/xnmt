import sys, os
import logging

from simple_settings import settings

STD_OUTPUT_LEVELNO = 35

class NoErrorFilter(logging.Filter):
  def filter(self, record):
    return not record.levelno in [logging.WARNING, logging.ERROR, logging.CRITICAL]

class ErrorOnlyFilter(logging.Filter):
  def filter(self, record):
    return record.levelno in [logging.WARNING, logging.ERROR, logging.CRITICAL]

sys_std_out = sys.stdout
sys_std_err = sys.stderr
logging.addLevelName(STD_OUTPUT_LEVELNO, "STD_OUTPUT")

logger = logging.getLogger('xnmt')
logger.setLevel(min(logging._checkLevel(settings.LOG_LEVEL_CONSOLE), logging._checkLevel(settings.LOG_LEVEL_FILE)))

ch_out = logging.StreamHandler(sys_std_out)
ch_out.setLevel(settings.LOG_LEVEL_CONSOLE)
ch_out.addFilter(NoErrorFilter())
ch_err = logging.StreamHandler(sys_std_err)
ch_err.setLevel(settings.LOG_LEVEL_CONSOLE)
ch_err.addFilter(ErrorOnlyFilter())
logger.addHandler(ch_out)
logger.addHandler(ch_err)





def set_out_file(out_file):
  unset_out_file()
  dirname = os.path.dirname(out_file)
  if dirname and not os.path.exists(dirname):
    os.makedirs(dirname)  
  fh = logging.FileHandler(out_file)
  fh.setLevel(settings.LOG_LEVEL_FILE)
  logger.addHandler(fh)

def unset_out_file():
  for hdlr in list(logger.handlers):
    if isinstance(hdlr, logging.FileHandler):
      hdlr.close()
      logger.removeHandler(hdlr)

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
    if data!="\n":
      if self.error:
        self.logger.error(data.rstrip())
      else:
        self.logger.log(STD_OUTPUT_LEVELNO, data.rstrip())
      self.flush()

  def flush(self):
    self.stdstream.flush()

  def getvalue(self):
    return self.stdstream.getvalue()

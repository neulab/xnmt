import sys
import os
import io
import six

class Tee(object):
  """
  Emulates a standard output or error streams. Calls to write on that stream will result
  in printing to stdout as well as logging to a file.
  """

  def __init__(self, name, indent=0, error=False):
    dirname = os.path.dirname(name)
    if not os.path.exists(dirname):
      os.makedirs(dirname)
    self.file = io.open(name, 'wt', encoding='utf-8')
    self.stdstream = sys.stderr if error else sys.stdout
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
    self.file.close()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()

  def write(self, data):
    self.file.write(six.u(data))
    self.stdstream.write(" " * self.indent + data)
    self.flush()

  def flush(self):
    self.file.flush()
    self.stdstream.flush()

  def getvalue(self):
    return self.stdstream.getvalue()

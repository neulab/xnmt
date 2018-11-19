"""
Global settings that control the overall behavior of XNMT.

Currently, settings control the following:

* ``OVERWRITE_LOG``: whether logs should be overwritten (not overwriting helps when copy-pasting config files and forgetting to change the output location)
* ``IMMEDIATE_COMPUTE``: whether to execute DyNet in eager mode
* ``CHECK_VALIDITY``: configure xnmt and DyNet to perform checks of validity
* ``RESOURCE_WARNINGS``: whether to show resource warnings
* ``LOG_LEVEL_CONSOLE``: verbosity of console output (``DEBUG`` | ``INFO`` | ``WARNING`` | ``ERROR`` | ``CRITICAL``)
* ``LOG_LEVEL_FILE``: verbosity of file output (``DEBUG`` | ``INFO`` | ``WARNING`` | ``ERROR`` | ``CRITICAL``)
* ``DEFAULT_MOD_PATH``: default location to write models to
* ``DEFAULT_LOG_PATH``: default location to write out logs

There are several predefined configurations (``Standard``, ``Debug``, ``Unittest``), with ``Standard`` being used by
default. Settings are specified from the command line using ``--settings={standard|debug|unittest}`` and should not be
changed during execution.

It is possible to control individual settings by setting an environment variable of the same name, e.g. like this:
``OVERWRITE_LOG=1 python -m xnmt.xnmt_run_experiments my_config.yaml``

To specify a custom configuration, subclass ``settings.Standard`` accordinly and add an alias to ``settings._aliases``.
"""

import sys
import os

class Standard(object):
  """
  Standard configuration, used by default.
  """
  OVERWRITE_LOG = False
  IMMEDIATE_COMPUTE = False
  CHECK_VALIDITY = False
  RESOURCE_WARNINGS = False
  PRINT_CG_ON_ERROR = False
  LOG_LEVEL_CONSOLE = "INFO"
  LOG_LEVEL_FILE = "DEBUG"
  DEFAULT_MOD_PATH = "{EXP_DIR}/models/{EXP}.mod"
  DEFAULT_LOG_PATH = "{EXP_DIR}/logs/{EXP}.log"
  DEFAULT_REPORT_PATH = "{EXP_DIR}/reports/{EXP}"

class Debug(Standard):
  """
  Adds checks and verbosity to help debugging code or configuration files.
  """
  OVERWRITE_LOG = True
  IMMEDIATE_COMPUTE = True
  CHECK_VALIDITY = True
  RESOURCE_WARNINGS = True
  PRINT_CG_ON_ERROR = True
  LOG_LEVEL_CONSOLE = "DEBUG"
  LOG_LEVEL_FILE = "DEBUG"

class Unittest(Standard):
  """
  More checks and less verbosity, activated automatically when running the unit tests from the "test" package.
  """
  OVERWRITE_LOG = True
  RESOURCE_WARNINGS = True
  LOG_LEVEL_CONSOLE = "WARNING"
  DEFAULT_MOD_PATH = "test/tmp/{EXP}.mod"
  DEFAULT_LOG_PATH = "test/tmp/{EXP}.log"
  DEFAULT_REPORT_PATH = "test/tmp/{EXP}.report"

class SettingsAccessor(object):
  def __getattr__(self, item):
    if _active is None:
      _resolve_active_settings()
    return getattr(_active, item)

settings = SettingsAccessor()

def _resolve_active_settings() -> None:
  # use command line argument, if not given use environment var, if not given us 'standard'
  # overwrite with environment variables if present.
  global _active
  settings_alias = "standard"
  settings_alias = os.environ.get("XNMT_SETTINGS", default=settings_alias)
  for arg in sys.argv:
    if arg.startswith("--settings"):
      settings_alias = arg.split("=")[1]
  _active = _aliases[settings_alias]
  for key, val in os.environ.items():
    if hasattr(_active, key):
      setattr(_active, key, val)

_active = None

_aliases = {
  "settings.standard" : Standard,
  "standard": Standard,
  "settings.debug" : Debug,
  "debug": Debug,
  "settings.unittest" : Unittest,
  "unittest": Unittest,
}

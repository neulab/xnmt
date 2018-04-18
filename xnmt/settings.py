import sys
import os

class Standard(object):
  OVERWRITE_LOG = False
  IMMEDIATE_COMPUTE = False
  CHECK_VALIDITY = False
  RESOURCE_WARNINGS = False
  LOG_LEVEL_CONSOLE = "INFO"
  LOG_LEVEL_FILE = "DEBUG"
  DEFAULT_MOD_PATH = "{EXP_DIR}/models/{EXP}.mod"
  DEFAULT_LOG_PATH = "{EXP_DIR}/logs/{EXP}.log"

class Debug(Standard):
  OVERWRITE_LOG = True
  IMMEDIATE_COMPUTE = True
  CHECK_VALIDITY = True
  RESOURCE_WARNINGS = True
  LOG_LEVEL_CONSOLE = "DEBUG"
  LOG_LEVEL_FILE = "DEBUG"

class Unittest(Standard):
  OVERWRITE_LOG = True
  RESOURCE_WARNINGS = True
  LOG_LEVEL_CONSOLE = "WARNING"

class SettingsAccessor(object):
  def __getattr__(self, item):
    if _active is None:
      _resolve_active_settings()
    return getattr(_active, item)

settings = SettingsAccessor()

def _resolve_active_settings():
  # use command line argument, if not given use environment var, if not given us 'standard'
  global _active
  settings_alias = "standard"
  settings_alias = os.environ.get("XNMT_SETTINGS", default=settings_alias)
  for arg in sys.argv:
    if arg.startswith("--settings"):
      settings_alias = arg.split("=")[1]
  _active = _aliases[settings_alias]


_active = None

_aliases = {
  "settings.standard" : Standard,
  "standard": Standard,
  "settings.debug" : Debug,
  "debug": Debug,
  "settings.unittest" : Unittest,
  "unittest": Unittest,
}
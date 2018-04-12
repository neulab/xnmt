
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
    return getattr(_active, item)

settings = SettingsAccessor()

_active = Standard

_aliases = {
  "settings.standard" : Standard,
  "standard": Standard,
  "settings.debug" : Debug,
  "debug": Debug,
  "settings.unittest" : Unittest,
  "unittest": Unittest,
}

def activate(settings_alias):
  global _active
  _active = _aliases[settings_alias]
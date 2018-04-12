import dynet_config

dynet_config.set(random_seed=2)

import xnmt.settings
xnmt.settings.activate("unittest")
import xnmt.tee
xnmt.tee.update_level_console()

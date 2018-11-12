from collections import namedtuple

TranslatorOutput = namedtuple('TranslatorOutput', ['state', 'logsoftmax', 'attention'])

from xnmt.models.translators.auto_regressive import AutoRegressiveTranslator
from xnmt.models.translators.default_translator import DefaultTranslator
from xnmt.models.translators.ensemble_translator import EnsembleTranslator

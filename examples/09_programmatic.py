# It is also possible to configure model training using Python code rather than
# YAML config files. This is less convenient and usually not necessary, but there
# may be cases where the added flexibility is needed. This basically works by
# using XNMT as a library of components that are initialized and run in this
# config file.
#
# This demonstrates a standard model training, including set up of logging, model
# saving, etc.; models are saved into YAML files that can again be loaded using
# the standard YAML  way (examples/07_load_finetune.yaml) or the Python way
# (10_programmatic_load.py)
#
# To launch this, use ``python -m examples.09_programmatic``, making sure that XNMT
# setup.py has been run properly.


import os
import random

import numpy as np

from xnmt.attender import MlpAttender
from xnmt.batcher import SrcBatcher, InOrderBatcher
from xnmt.bridge import CopyBridge
from xnmt.decoder import AutoRegressiveDecoder
from xnmt.embedder import SimpleWordEmbedder
from xnmt.eval_task import LossEvalTask, AccuracyEvalTask
from xnmt.experiment import Experiment
from xnmt.inference import AutoRegressiveInference
from xnmt.input_reader import PlainTextReader
from xnmt.lstm import BiLSTMSeqTransducer, UniLSTMSeqTransducer
from xnmt.transform import AuxNonLinear
from xnmt.scorer import Softmax
from xnmt.optimizer import AdamTrainer
from xnmt.param_collection import ParamManager
from xnmt.persistence import save_to_file
import xnmt.tee
from xnmt.training_regimen import SimpleTrainingRegimen
from xnmt.translator import DefaultTranslator
from xnmt.vocab import Vocab

seed=13
random.seed(seed)
np.random.seed(seed)

EXP_DIR = os.path.dirname(__file__)
EXP = "programmatic"

model_file = f"{EXP_DIR}/models/{EXP}.mod"
log_file = f"{EXP_DIR}/logs/{EXP}.log"

xnmt.tee.set_out_file(log_file)

ParamManager.init_param_col()
ParamManager.param_col.model_file = model_file

src_vocab = Vocab(vocab_file="examples/data/head.ja.vocab")
trg_vocab = Vocab(vocab_file="examples/data/head.en.vocab")

batcher = SrcBatcher(batch_size=64)

inference = AutoRegressiveInference(batcher=InOrderBatcher(batch_size=1))

layer_dim = 512

model = DefaultTranslator(
  src_reader=PlainTextReader(vocab=src_vocab),
  trg_reader=PlainTextReader(vocab=trg_vocab),
  src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=len(src_vocab)),

  encoder=BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, layers=1),
  attender=MlpAttender(hidden_dim=layer_dim, state_dim=layer_dim, input_dim=layer_dim),
  trg_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=len(trg_vocab)),
  decoder=AutoRegressiveDecoder(input_dim=layer_dim,
                                rnn=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim,
                                                         decoder_input_dim=layer_dim, yaml_path="decoder"),
                                transform=AuxNonLinear(input_dim=layer_dim, output_dim=layer_dim,
                                                       aux_input_dim=layer_dim),
                                scorer=Softmax(vocab_size=len(trg_vocab), input_dim=layer_dim),
                                trg_embed_dim=layer_dim,
                                bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
  inference=inference
)

train = SimpleTrainingRegimen(
  name=f"{EXP}",
  model=model,
  batcher=batcher,
  trainer=AdamTrainer(alpha=0.001),
  run_for_epochs=2,
  src_file="examples/data/head.ja",
  trg_file="examples/data/head.en",
  dev_tasks=[LossEvalTask(src_file="examples/data/head.ja",
                          ref_file="examples/data/head.en",
                          model=model,
                          batcher=batcher)],
)

evaluate = [AccuracyEvalTask(eval_metrics="bleu,wer",
                             src_file="examples/data/head.ja",
                             ref_file="examples/data/head.en",
                             hyp_file=f"examples/output/{EXP}.test_hyp",
                             inference=inference,
                             model=model)]

standard_experiment = Experiment(
  model=model,
  train=train,
  evaluate=evaluate
)

# run experiment
standard_experiment(save_fct=lambda: save_to_file(model_file, standard_experiment))

exit()

import os

import xnmt.serialize.imports
from xnmt.attender import MlpAttender
from xnmt.batcher import SrcBatcher
from xnmt.bridge import CopyBridge
from xnmt.decoder import MlpSoftmaxDecoder
from xnmt.embedder import SimpleWordEmbedder
from xnmt.eval_task import LossEvalTask, AccuracyEvalTask
from xnmt.experiment import Experiment
from xnmt.inference import SimpleInference
from xnmt.input_reader import PlainTextReader
from xnmt.lstm import BiLSTMSeqTransducer
from xnmt.optimizer import AdamTrainer
from xnmt.param_collection import ParamManager
from xnmt.serialize.serializer import YamlSerializer
from xnmt.training_regimen import SimpleTrainingRegimen
from xnmt.translator import DefaultTranslator
from xnmt.vocab import Vocab

EXP_DIR = os.path.dirname(__file__)
EXP = "programmatic"

model_file = f"{EXP_DIR}/models/{EXP}.mod"

ParamManager.init_param_col()
ParamManager.param_col.model_file = model_file

src_vocab = Vocab(vocab_file="examples/data/head.ja.vocab")
trg_vocab = Vocab(vocab_file="examples/data/head.en.vocab")

batcher = SrcBatcher(batch_size=64)

inference = SimpleInference(batcher=batcher)

layer_dim = 512

model = DefaultTranslator(
  src_reader=PlainTextReader(vocab=src_vocab),
  trg_reader=PlainTextReader(vocab=trg_vocab),
  src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=len(src_vocab)),
  encoder=BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, layers=1),
  attender=MlpAttender(hidden_dim=layer_dim, state_dim=layer_dim, input_dim=layer_dim),
  trg_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=len(trg_vocab)),
  decoder=MlpSoftmaxDecoder(input_dim=layer_dim, lstm_dim=layer_dim, mlp_hidden_dim=layer_dim, trg_embed_dim=layer_dim,
                            vocab_size=len(trg_vocab),
                            bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
  inference=inference
)

train = SimpleTrainingRegimen(
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
standard_experiment(save_fct=lambda: YamlSerializer().save_to_file(model_file,
                                                                   standard_experiment,
                                                                   ParamManager.param_col))

exit()
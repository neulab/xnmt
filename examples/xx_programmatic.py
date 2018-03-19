import os

import xnmt.serialize.imports
from xnmt.attender import MlpAttender
from xnmt.batcher import SrcBatcher
from xnmt.bridge import CopyBridge
from xnmt.decoder import MlpSoftmaxDecoder
from xnmt.embedder import SimpleWordEmbedder
from xnmt.eval_task import LossEvalTask, AccuracyEvalTask
from xnmt.exp_global import ExpGlobal, PersistentParamCollection
from xnmt.experiment import Experiment
from xnmt.inference import SimpleInference
from xnmt.input_reader import PlainTextReader
from xnmt.lstm import BiLSTMSeqTransducer
from xnmt.optimizer import AdamTrainer
from xnmt.serialize.serializer import YamlSerializer
from xnmt.training_regimen import SimpleTrainingRegimen
from xnmt.translator import DefaultTranslator
from xnmt.vocab import Vocab

placeholders = {
  "EXP_DIR": os.path.dirname(__file__),
  "EXP": "standard"
}

model_file = "{EXP_DIR}/models/{EXP}.mod".format(**placeholders)

exp_global = ExpGlobal(
  dynet_param_collection=PersistentParamCollection(model_file=model_file))

src_vocab = Vocab(vocab_file="examples/data/head.ja.vocab")
trg_vocab = Vocab(vocab_file="examples/data/head.en.vocab")

batcher = SrcBatcher(batch_size=64)

inference = SimpleInference(batcher=batcher)

model = DefaultTranslator(
  src_reader=PlainTextReader(vocab=src_vocab),
  trg_reader=PlainTextReader(vocab=trg_vocab),
  src_embedder=SimpleWordEmbedder(exp_global=exp_global, vocab_size=len(src_vocab)),
  encoder=BiLSTMSeqTransducer(exp_global=exp_global, layers=1),
  attender=MlpAttender(exp_global=exp_global, hidden_dim=512, state_dim=512, input_dim=512),
  trg_embedder=SimpleWordEmbedder(exp_global=exp_global, vocab_size=len(trg_vocab)),
  decoder=MlpSoftmaxDecoder(exp_global=exp_global, vocab_size=len(trg_vocab),
                            bridge=CopyBridge(exp_global=exp_global, dec_layers=1)),
  inference=inference
)

train = SimpleTrainingRegimen(
  model=model,
  batcher=batcher,
  trainer=AdamTrainer(alpha=0.001, exp_global=exp_global),
  run_for_epochs=2,
  src_file="examples/data/head.ja",
  trg_file="examples/data/head.en",
  dev_tasks=[LossEvalTask(src_file="examples/data/head.ja",
                          ref_file="examples/data/head.en",
                          model=model,
                          batcher=batcher)],
  exp_global=exp_global
)

evaluate = [AccuracyEvalTask(eval_metrics="bleu,wer",
                             src_file="examples/data/head.ja",
                             ref_file="examples/data/head.en",
                             hyp_file="examples/output/{EXP}.test_hyp",
                             inference=inference,
                             model=model)]

standard_experiment = Experiment(
  exp_global=exp_global,
  model=model,
  train=train,
  evaluate=evaluate
)

# run experiment
standard_experiment(save_fct=lambda: YamlSerializer().save_to_file(model_file,
                                                                   standard_experiment,
                                                                   exp_global.dynet_param_collection))


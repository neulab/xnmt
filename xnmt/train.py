# coding: utf-8
from __future__ import division, print_function

import argparse
import sys
import six
from six.moves import range
from subprocess import Popen

import dynet as dy

# all Serializable objects must be imported here, otherwise we get in trouble with the
# YAML parser
import xnmt.batcher
from xnmt.embedder import *
from xnmt.attender import *
from xnmt.input import *
import xnmt.lstm
import xnmt.pyramidal
import xnmt.conv
import xnmt.ff
import xnmt.segment_transducer
import xnmt.residual
from xnmt.specialized_encoders import *
from xnmt.decoder import *
from xnmt.translator import *
from xnmt.retriever import *
from xnmt.training_corpus import *
from xnmt.loss_tracker import *
from xnmt.segmenting_encoder import *
from xnmt.loss import LossBuilder
from xnmt.model_context import ModelContext, PersistentParamCollection
from xnmt.training_strategy import TrainingStrategy, TrainingMLELoss
from xnmt.serializer import YamlSerializer, Serializable
import xnmt.xnmt_decode
import xnmt.xnmt_evaluate
from xnmt.evaluator import LossScore
import xnmt.optimizer

'''
This will be the main class to perform training.
'''

class TrainingRegimen(Serializable):
  yaml_tag = u'!TrainingRegimen'
  def __init__(self, corpus_parser, model_file, model, yaml_context=None, glob={},
               dev_every=0, batcher=None, training_strategy=None, save_num_checkpoints=1,
               pretrained_model_file="", src_format="text",
               trainer=None, lr_decay=1.0, lr_decay_times=3, attempts_before_lr_decay=1,
               dev_metrics="", schedule_metric="loss", restart_trainer=False,
               reload_command=None):
    """
    :param corpus_parser:
    :param model_file:
    :param model:
    :param yaml_context: (TODO: remove default value)
    :param dev_every (int): dev checkpoints every n sentences (0 for only after epoch)
    :param batcher: Type of batcher. Defaults to SrcBatcher of batch size 32.
    :param training_strategy:
    :param save_num_checkpoints (int): Save recent n best checkpoints
    :param pretrained_model_file: Path of pre-trained model file
    :param src_format: Format of input data: text/contvec
    :param trainer: Trainer object, default is SGD with learning rate 0.1
    :param lr_decay (float):
    :param lr_decay_times (int):  Early stopping after decaying learning rate a certain number of times
    :param attempts_before_lr_decay (int): apply LR decay after dev scores haven't improved over this many checkpoints
    :param dev_metrics: Comma-separated list of evaluation metrics (bleu/wer/cer)
    :param schedule_metric: determine learning schedule based on this dev_metric (loss/bleu/wer/cer)
    :param restart_trainer: Restart trainer (useful for Adam) and revert weights to best dev checkpoint when applying LR decay (https://arxiv.org/pdf/1706.09733.pdf)
    :param reload_command: Command to change the input data after each epoch.
                           --epoch EPOCH_NUM will be appended to the command.
                           To just reload the data after each epoch set the command to 'true'.
    """
    dy.renew_cg()

    # TODO: don't need to keep a dedicated args object any longer
    args = dict(dev_every=dev_every, batcher=batcher,
               corpus_parser=corpus_parser, training_strategy=training_strategy, model_file=model_file, save_num_checkpoints=save_num_checkpoints,
               pretrained_model_file=pretrained_model_file, src_format=src_format, default_layer_dim=glob.get("default_layer_dim", 512),
               trainer=trainer, lr_decay=lr_decay, lr_decay_times=lr_decay_times, attempts_before_lr_decay=attempts_before_lr_decay,
               dev_metrics=dev_metrics, schedule_metric=schedule_metric, restart_trainer=restart_trainer,reload_command=reload_command,
               dropout=glob.get("dropout", 0.0), weight_noise=glob.get("weight_noise", 0.0), model=model)
    self.args = args
    if yaml_context:
      self.model_context = yaml_context
    else:
      self.model_context = ModelContext()
      self.model_context.dynet_param_collection = PersistentParamCollection(self.args["model_file"], self.args["save_num_checkpoints"])


    if args["lr_decay"] > 1.0 or args["lr_decay"] <= 0.0:
      raise RuntimeError("illegal lr_decay, must satisfy: 0.0 < lr_decay <= 1.0")
    self.num_times_lr_decayed = 0
    self.early_stopping_reached = False
    self.cur_attempt = 0

    self.evaluators = [s.lower() for s in self.args["dev_metrics"].split(",") if s.strip()!=""]
    if self.args["schedule_metric"].lower() not in self.evaluators:
              self.evaluators.append(self.args["schedule_metric"].lower())
    if "loss" not in self.evaluators: self.evaluators.append("loss")

    if args["reload_command"] is not None:
        self._augmentation_handle = None
        self._augment_data_initial()

    # Initialize the serializer
    self.model_serializer = YamlSerializer()

    self.create_corpus_and_model()

    self.model.initialize_training_strategy(self.training_strategy)

    if self.args["batcher"] is None:
      self.batcher = SrcBatcher(32)
    else:
      self.batcher = self.args["batcher"]
    if args["src_format"] == "contvec":
      self.batcher.pad_token = np.zeros(self.model.src_embedder.emb_dim)
    self.pack_batches()
    self.logger = BatchLossTracker(args["dev_every"], self.total_train_sent)

    if args["trainer"] is None:
      self.trainer = xnmt.optimizer.SimpleSGDTrainer(self.model_context, 0.1)
    else:
      self.trainer = args["trainer"] 

  def dependent_init_params(self, initialized_subcomponents):
    return [DependentInitParam(param_descr="model.src_embedder.vocab_size", value_fct=lambda: initialized_subcomponents["corpus_parser"].src_reader.vocab_size()),
            DependentInitParam(param_descr="model.decoder.vocab_size", value_fct=lambda: initialized_subcomponents["corpus_parser"].trg_reader.vocab_size()),
            DependentInitParam(param_descr="model.trg_embedder.vocab_size", value_fct=lambda: initialized_subcomponents["corpus_parser"].trg_reader.vocab_size()),
            DependentInitParam(param_descr="model.src_embedder.vocab", value_fct=lambda: initialized_subcomponents["corpus_parser"].src_reader.vocab),
            DependentInitParam(param_descr="model.trg_embedder.vocab", value_fct=lambda: initialized_subcomponents["corpus_parser"].trg_reader.vocab)]

  def pack_batches(self):
    self.train_src, self.train_trg = \
      self.batcher.pack(self.corpus_parser.training_corpus.train_src_data, self.corpus_parser.training_corpus.train_trg_data)
    self.dev_src, self.dev_trg = \
      self.batcher.pack(self.corpus_parser.training_corpus.dev_src_data, self.corpus_parser.training_corpus.dev_trg_data)

  def create_corpus_and_model(self):
    self.corpus_parser = self.args["corpus_parser"]
    if not hasattr(self.corpus_parser.training_corpus, "train_src_data"): # TODO: not so pretty, needs refactoring
      self.corpus_parser._read_training_corpus(self.corpus_parser.training_corpus)
    self.total_train_sent = len(self.corpus_parser.training_corpus.train_src_data)
    self.model_context.default_layer_dim = self.args["default_layer_dim"]
    self.model_context.dropout = self.args["dropout"]
    self.model_context.weight_noise = self.args["weight_noise"]
    if not self.args["model"]:
      raise RuntimeError("No model specified!")
    self.model = self.args["model"]
    if self.args["training_strategy"]:
      self.training_strategy = self.args["training_strategy"]
    else:
      self.training_strategy = TrainingStrategy(TrainingMLELoss())
    if self.args.get("pretrained_model_file", None):
      self.model_context.dynet_param_collection.load_from_data_file(self.args["pretrained_model_file"] + '.data')
    
  def _augment_data_initial(self):
    augment_command = self.args["reload_command"]
    print('initial augmentation')
    if self._augmentation_handle is None:
      # first run
      self._augmentation_handle = Popen(augment_command + " --epoch 0", shell=True)
      self._augmentation_handle.wait()

  def _augment_data_next_epoch(self):
    augment_command = self.args["reload_command"]
    if self._augmentation_handle is None:
      # first run
      self._augmentation_handle = Popen(augment_command + " --epoch %d" % self.logger.epoch_num, shell=True)
      self._augmentation_handle.wait()
   
    self._augmentation_handle.poll()
    retcode = self._augmentation_handle.returncode
    if retcode is not None:
      if self.logger.epoch_num > 0:
        print('using reloaded data')
      # reload the data   
      self.corpus_parser._read_training_corpus(self.corpus_parser.training_corpus) # TODO: fix
      self.pack_batches()
      self.logger.total_train_sent = len(self.corpus_parser.training_corpus.train_src_data)
      # restart data generation
      self._augmentation_handle = Popen(augment_command + " --epoch %d" % self.logger.epoch_num, shell=True)
    else:
      print('new data set is not ready yet, using data from last epoch.')

  def run_epochs(self, num_epochs=None):
    epoch_i = 0
    while True:
      self.one_epoch()
      epoch_i += 1
      if self.early_stopping_reached or (num_epochs is not None and epoch_i >= num_epochs):
        break

  def one_epoch(self, update_weights=True):
    """
    :param update_weights: Whether to perform backward pass & update weights (useful for debugging)
    """

    self.logger.new_epoch()

    if self.args["reload_command"] is not None:
      self._augment_data_next_epoch()

    self.model.set_train(update_weights)
    order = list(range(0, len(self.train_src)))
    np.random.shuffle(order)
    for batch_num in order:
      src = self.train_src[batch_num]
      trg = self.train_trg[batch_num]

      # Loss calculation
      dy.renew_cg()
      loss_builder = LossBuilder()
      standard_loss = self.model.calc_loss(src, trg)

      if standard_loss.__class__ == LossBuilder:
        loss = None
        for loss_name, loss_expr in standard_loss.loss_nodes:
          loss_builder.add_loss(loss_name, loss_expr)
          loss = loss_expr if not loss else loss + loss_expr
        standard_loss = loss

      else:
        loss_builder.add_loss("loss", standard_loss)

      additional_loss = self.model.calc_additional_loss(dy.nobackprop(-standard_loss))
      if additional_loss != None:
        loss_builder.add_loss("additional_loss", additional_loss)

      # Log the loss sum
      loss_value = loss_builder.compute()
      self.logger.update_epoch_loss(src, trg, loss_builder)
      if update_weights:
        loss_value.backward()
        self.trainer.update()

      # Devel reporting
      self.logger.report_train_process()
      if self.logger.should_report_dev():
        self.dev_evaluation()

      self.model.new_epoch()

  def dev_evaluation(self, out_ext=".dev_hyp", ref_ext=".dev_ref", encoding='utf-8'):
    self.model.set_train(False)
    self.logger.new_dev()
    trg_words_cnt, loss_score = self.compute_dev_loss()
    schedule_metric = self.args["schedule_metric"].lower()

    eval_scores = {"loss" : loss_score}
    if len(list(filter(lambda e: e!="loss", self.evaluators))) > 0:
#       self.xnmt_decoder["src_file"] = self.corpus_parser.training_corpus.dev_src
#       self.xnmt_decoder["candidate_id_file"] = self.corpus_parser.training_corpus.dev_id_file
      trg_file = None
      if self.args["model_file"]:
        out_file = self.args["model_file"] + out_ext
        out_file_ref = self.args["model_file"] + ref_ext
        trg_file = out_file
      # Decoding + post_processing
#       xnmt.xnmt_decode.xnmt_decode(model_elements=(self.corpus_parser, self.model), **self.xnmt_decoder)
      self.xnmt_decoder(src_file = self.corpus_parser.training_corpus.dev_src,
                                   trg_file = trg_file,
                                   candidate_id_file = self.corpus_parser.training_corpus.dev_id_file,
                                   model_elements=(self.corpus_parser, self.model))
      output_processor = self.xnmt_decoder.get_output_processor() # TODO: hack, refactor
      # Copy Trg to Ref
      processed = []
      with io.open(self.corpus_parser.training_corpus.dev_trg, encoding=encoding) as fin:
        for line in fin:
          processed.append(output_processor.words_to_string(line.strip().split()) + u"\n")
      with io.open(out_file_ref, 'wt', encoding=encoding) as fout:
        for line in processed:
          fout.write(line)
      # Evaluation
      if self.args["model_file"]:
        self.evaluate_args["hyp_file"] = out_file
        self.evaluate_args["ref_file"] = out_file_ref
      for evaluator in self.evaluators:
        if evaluator=="loss": continue
        self.evaluate_args["evaluator"] = evaluator
        eval_score = xnmt.xnmt_evaluate.xnmt_evaluate(**self.evaluate_args)
        eval_scores[evaluator] = eval_score
    # Logging
    if schedule_metric == "loss":
      self.logger.set_dev_score(trg_words_cnt, loss_score)
    else:
      self.logger.set_dev_score(trg_words_cnt, eval_scores[schedule_metric])

    print("> Checkpoint")
    # print previously computed metrics
    for metric in self.evaluators:
      if metric != schedule_metric:
        self.logger.report_auxiliary_score(eval_scores[metric])
    
    # Write out the model if it's the best one
    if self.logger.report_dev_and_check_model(self.args["model_file"]):
      if self.args["model_file"] is not None:
        self.model_serializer.save_to_file(self.args["model_file"],
                                           self,
                                           self.model_context.dynet_param_collection)
      self.cur_attempt = 0
    else:
      # otherwise: learning rate decay / early stopping
      self.cur_attempt += 1
      if self.args["lr_decay"] < 1.0 and self.cur_attempt >= self.args["attempts_before_lr_decay"]:
        self.num_times_lr_decayed += 1
        if self.num_times_lr_decayed > self.args["lr_decay_times"]:
          print('  Early stopping')
          self.early_stopping_reached = True
        else:
          self.trainer.learning_rate *= self.args["lr_decay"]
          print('  new learning rate: %s' % self.trainer.learning_rate)
          if self.args["restart_trainer"]:
            print('  restarting trainer and reverting learned weights to best checkpoint..')
            self.trainer.restart()
            self.model_context.dynet_param_collection.revert_to_best_model()

    self.model.set_train(True)
    return

  def compute_dev_loss(self):
    loss_builder = LossBuilder()
    trg_words_cnt = 0
    for src, trg in zip(self.dev_src, self.dev_trg):
      dy.renew_cg()
      standard_loss = self.model.calc_loss(src, trg)
      loss_builder.add_loss("loss", standard_loss)
      trg_words_cnt += self.logger.count_trg_words(trg)
      loss_builder.compute()
    return trg_words_cnt, LossScore(loss_builder.sum() / trg_words_cnt)

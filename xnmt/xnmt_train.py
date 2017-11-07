# coding: utf-8
from __future__ import division, print_function

import argparse
import sys
import six
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
from xnmt.serialize_container import *
from xnmt.training_corpus import *
from xnmt.loss_tracker import *
from xnmt.segmenting_encoder import *
from xnmt.options import Option, OptionParser, general_options
from xnmt.loss import LossBuilder
from xnmt.model_context import ModelContext, PersistentParamCollection
from xnmt.training_strategy import TrainingStrategy, TrainingMLELoss
import xnmt.serializer
import xnmt.xnmt_decode
import xnmt.xnmt_evaluate
from xnmt.evaluator import LossScore
'''
This will be the main class to perform training.
'''

options = [
  Option("dynet-mem", int, required=False),
  Option("dynet-gpu-ids", int, required=False),
  Option("dynet-gpus", int, required=False),
  Option("dev_every", int, default_value=0, force_flag=True, help_str="dev checkpoints every n sentences (0 for only after epoch)"),
  Option("batcher", default_value=None, required=False, help_str="Type of batcher. Defaults to SrcBatcher of batch size 32."),
  Option("training_corpus"),
  Option("corpus_parser"),
  Option("training_strategy", required=False),
#  Option("train_filters", list, required=False, help_str="Specify filtering criteria for the training data"),
#  Option("dev_filters", list, required=False, help_str="Specify filtering criteria for the development data"),
  Option("model_file"),
  Option("save_num_checkpoints", int, default_value=1, help_str="Save recent n best checkpoints"),
  Option("pretrained_model_file", default_value="", help_str="Path of pre-trained model file"),
  Option("src_format", default_value="text", help_str="Format of input data: text/contvec"),
  Option("default_layer_dim", int, default_value=512, help_str="Default size to use for layers if not otherwise overridden"),
  Option("trainer", default_value="sgd"),
  Option("learning_rate", float, default_value=0.1),
  Option("momentum", float, default_value = 0.9),
  Option("lr_decay", float, default_value=1.0),
  Option("lr_decay_times", int, default_value=3, help_str="Early stopping after decaying learning rate a certain number of times"),
  Option("attempts_before_lr_decay", int, default_value=1, help_str="apply LR decay after dev scores haven't improved over this many checkpoints"),
  Option("dev_metrics", default_value="", help_str="Comma-separated list of evaluation metrics (bleu/wer/cer)"),
  Option("schedule_metric", default_value="loss", help_str="determine learning schedule based on this dev_metric (loss/bleu/wer/cer)"),
  Option("restart_trainer", bool, default_value=False, help_str="Restart trainer (useful for Adam) and revert weights to best dev checkpoint when applying LR decay (https://arxiv.org/pdf/1706.09733.pdf)"),
  Option("reload_command", default_value=None, required=False, help_str="Command to change the input data after each epoch. "
                                                                        "--epoch EPOCH_NUM will be appended to the command."
                                                                        "To just reload the data after each epoch set the command to 'true'."),
  Option("dropout", float, default_value=0.0),
  Option("weight_noise", float, default_value=0.0),
  Option("model", dict, default_value={}),
]

class XnmtTrainer(object):
  def __init__(self, args, need_deserialization=True, param_collection=None):
    """
    :param args: xnmt.options.Args instance corresponding to the options given above
    :param need_deserialization: Whether we need to invoke model_serializer.initialize_object on objects in args;
        This is usually the case when these have been deserialized from a YAML file, but not when instantiating XnmtTrainer manually.
    """
    dy.renew_cg()

    self.need_deserialization = need_deserialization
    self.args = args
    self.model_context = ModelContext()
    if param_collection:
      self.model_context.dynet_param_collection = param_collection
    else:
      self.model_context.dynet_param_collection = PersistentParamCollection(self.args.model_file, self.args.save_num_checkpoints)

    self.trainer = self.dynet_trainer_for_args(args, self.model_context)

    if args.lr_decay > 1.0 or args.lr_decay <= 0.0:
      raise RuntimeError("illegal lr_decay, must satisfy: 0.0 < lr_decay <= 1.0")
    self.num_times_lr_decayed = 0
    self.early_stopping_reached = False
    self.cur_attempt = 0

    self.evaluators = [s.lower() for s in self.args.dev_metrics.split(",") if s.strip()!=""]
    if self.args.schedule_metric.lower() not in self.evaluators:
              self.evaluators.append(self.args.schedule_metric.lower())
    if "loss" not in self.evaluators: self.evaluators.append("loss")

    if args.reload_command is not None:
        self._augmentation_handle = None
        self._augment_data_initial()

    # Initialize the serializer
    self.model_serializer = xnmt.serializer.YamlSerializer()

    if self.args.pretrained_model_file:
      self.load_corpus_and_model()
    else:
      self.create_corpus_and_model()

    self.model.initialize_training_strategy(self.training_strategy)

    if self.args.batcher is None:
      self.batcher = SrcBatcher(32)
    else:
      self.batcher = self.model_serializer.initialize_object(self.args.batcher) if self.need_deserialization else self.args.batcher
    if args.src_format == "contvec":
      self.batcher.pad_token = np.zeros(self.model.src_embedder.emb_dim)
    self.pack_batches()
    self.logger = BatchLossTracker(args.dev_every, self.total_train_sent)

  def pack_batches(self):
    self.train_src, self.train_trg = \
      self.batcher.pack(self.training_corpus.train_src_data, self.training_corpus.train_trg_data)
    self.dev_src, self.dev_trg = \
      self.batcher.pack(self.training_corpus.dev_src_data, self.training_corpus.dev_trg_data)

  def dynet_trainer_for_args(self, args, model_context):
    if args.trainer.lower() == "sgd":
      trainer = dy.SimpleSGDTrainer(model_context.dynet_param_collection.param_col, args.learning_rate)
    elif args.trainer.lower() == "adam":
      trainer = dy.AdamTrainer(model_context.dynet_param_collection.param_col, alpha = args.learning_rate)
    elif args.trainer.lower() == "msgd":
      trainer = dy.MomentumSGDTrainer(model_context.dynet_param_collection.param_col, args.learning_rate, mom = args.momentum)
    else:
      raise RuntimeError("Unknown trainer {}".format(args.trainer))
    return trainer

  def create_corpus_and_model(self):
    self.training_corpus = self.model_serializer.initialize_object(self.args.training_corpus) if self.need_deserialization else self.args.training_corpus
    self.corpus_parser = self.model_serializer.initialize_object(self.args.corpus_parser) if self.need_deserialization else self.args.corpus_parser
    self.corpus_parser.read_training_corpus(self.training_corpus)
    self.total_train_sent = len(self.training_corpus.train_src_data)
    self.model_context.corpus_parser = self.corpus_parser
    self.model_context.training_corpus = self.training_corpus
    self.model_context.default_layer_dim = self.args.default_layer_dim
    self.model_context.dropout = self.args.dropout
    self.model_context.weight_noise = self.args.weight_noise
    if not self.args.model:
      raise RuntimeError("No model specified!")
    self.model = self.model_serializer.initialize_object(self.args.model, self.model_context) if self.need_deserialization else self.args.model
    if self.args.training_strategy:
      self.training_strategy = self.model_serializer.initialize_object(self.args.training_strategy, self.model_context) if self.need_deserialization else self.args.training_strategy
    else:
      self.training_strategy = TrainingStrategy(TrainingMLELoss())

  def load_corpus_and_model(self):
    self.training_corpus = self.model_serializer.initialize_object(self.args.training_corpus) if self.need_deserialization else self.args.training_corpus
    corpus_parser, model, my_model_context = self.model_serializer.load_from_file(self.args.pretrained_model_file, self.model_context.dynet_param_collection)
    self.corpus_parser = self.model_serializer.initialize_object(corpus_parser) if self.need_deserialization else self.args.corpus_parser
    self.corpus_parser.read_training_corpus(self.training_corpus)
    self.model_context.update(my_model_context)
    self.total_train_sent = len(self.training_corpus.train_src_data)
    self.model_context.corpus_parser = self.corpus_parser
    self.model_context.training_corpus = self.training_corpus
    self.model = self.model_serializer.initialize_object(model, self.model_context) if self.need_deserialization else self.args.model
    self.model_context.dynet_param_collection.load_from_data_file(self.args.pretrained_model_file + '.data')
    if self.args.training_strategy:
      self.training_strategy = self.model_serializer.initialize_object(self.args.training_strategy, self.model_context) if self.need_deserialization else self.args.training_strategy
    else:
      self.training_strategy = TrainingStrategy(TrainingMLELoss())

  def _augment_data_initial(self):
    augment_command = self.args.reload_command
    print('initial augmentation')
    if self._augmentation_handle is None:
      # first run
      self._augmentation_handle = Popen(augment_command + " --epoch 0", shell=True)
      self._augmentation_handle.wait()

  def _augment_data_next_epoch(self):
    augment_command = self.args.reload_command
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
      self.corpus_parser.read_training_corpus(self.training_corpus)
      self.pack_batches()
      self.logger.total_train_sent = len(self.training_corpus.train_src_data)
      # restart data generation
      self._augmentation_handle = Popen(augment_command + " --epoch %d" % self.logger.epoch_num, shell=True)
    else:
      print('new data set is not ready yet, using data from last epoch.')

  def run_epoch(self, update_weights=True):
    """
    :param update_weights: Whether to perform backward pass & update weights (useful for debugging)
    """

    self.logger.new_epoch()
    self.model.new_epoch()

    if self.args.reload_command is not None:
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

  def dev_evaluation(self, out_ext=".dev_hyp", ref_ext=".dev_ref", encoding='utf-8'):
    self.model.set_train(False)
    self.logger.new_dev()
    trg_words_cnt, loss_score = self.compute_dev_loss()
    schedule_metric = self.args.schedule_metric.lower()

    eval_scores = {"loss" : loss_score}
    if len(list(filter(lambda e: e!="loss", self.evaluators))) > 0:
      self.decode_args.src_file = self.training_corpus.dev_src
      self.decode_args.candidate_id_file = self.training_corpus.dev_id_file
      if self.args.model_file:
        out_file = self.args.model_file + out_ext
        out_file_ref = self.args.model_file + ref_ext
        self.decode_args.trg_file = out_file
      # Decoding + post_processing
      xnmt.xnmt_decode.xnmt_decode(self.decode_args, model_elements=(self.corpus_parser, self.model))
      output_processor = xnmt.xnmt_decode.output_processor_for_spec(self.decode_args.post_process)
      # Copy Trg to Ref
      processed = []
      with io.open(self.training_corpus.dev_trg, encoding=encoding) as fin:
        for line in fin:
          processed.append(output_processor.words_to_string(line.strip().split()) + u"\n")
      with io.open(out_file_ref, 'wt', encoding=encoding) as fout:
        for line in processed:
          fout.write(line)
      # Evaluation
      if self.args.model_file:
        self.evaluate_args.hyp_file = out_file
        self.evaluate_args.ref_file = out_file_ref
      for evaluator in self.evaluators:
        if evaluator=="loss": continue
        self.evaluate_args.evaluator = evaluator
        eval_score = xnmt.xnmt_evaluate.xnmt_evaluate(self.evaluate_args)
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
    if self.logger.report_dev_and_check_model(self.args.model_file):
      if self.args.model_file is not None:
        self.model_serializer.save_to_file(self.args.model_file,
                                           SerializeContainer(self.corpus_parser, self.model, self.model_context),
                                           self.model_context.dynet_param_collection)
      self.cur_attempt = 0
    else:
      # otherwise: learning rate decay / early stopping
      self.cur_attempt += 1
      if self.args.lr_decay < 1.0 and self.cur_attempt >= self.args.attempts_before_lr_decay:
        self.num_times_lr_decayed += 1
        if self.num_times_lr_decayed > self.args.lr_decay_times:
          print('  Early stopping')
          self.early_stopping_reached = True
        else:
          self.trainer.learning_rate *= self.args.lr_decay
          print('  new learning rate: %s' % self.trainer.learning_rate)
          if self.args.restart_trainer:
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

if __name__ == "__main__":
  parser = OptionParser()
  parser.add_task("train", general_options + options)
  args = parser.args_from_command_line("train", sys.argv[1:])
  print("Starting xnmt-train:\nArguments: %r" % (args))

  xnmt_trainer = XnmtTrainer(args)

  while not xnmt_trainer.early_stopping_reached:
    xnmt_trainer.run_epoch()

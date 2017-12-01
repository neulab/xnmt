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
from xnmt.loss_calculator import LossCalculator, MLELoss
from xnmt.serializer import YamlSerializer, Serializable
import xnmt.xnmt_decode
import xnmt.xnmt_evaluate
from xnmt.evaluator import LossScore
import xnmt.optimizer
from xnmt.events import register_xnmt_event
'''
This will be the main class to perform training.
'''

class TrainingRegimen(Serializable):
  yaml_tag = u'!TrainingRegimen'
  def __init__(self, yaml_context, corpus_parser, model_file, model, glob={},
               dev_every=0, batcher=None, loss_calculator=None, 
               pretrained_model_file="", src_format="text", trainer=None, 
               run_for_epochs=None, lr_decay=1.0, lr_decay_times=3, attempts_before_lr_decay=1,
               dev_metrics="", schedule_metric="loss", restart_trainer=False,
               reload_command=None, dynet_profiling=0):
    """
    :param corpus_parser:
    :param model_file:
    :param model:
    :param yaml_context:
    :param dev_every (int): dev checkpoints every n sentences (0 for only after epoch)
    :param batcher: Type of batcher. Defaults to SrcBatcher of batch size 32.
    :param loss_calculator:
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
    assert yaml_context is not None
    self.yaml_context = yaml_context
    self.model_file = model_file

    if lr_decay > 1.0 or lr_decay <= 0.0:
      raise RuntimeError("illegal lr_decay, must satisfy: 0.0 < lr_decay <= 1.0")
    self.lr_decay = lr_decay
    self.attempts_before_lr_decay = attempts_before_lr_decay
    self.lr_decay_times = lr_decay_times
    self.restart_trainer = restart_trainer
    self.run_for_epochs = run_for_epochs
    
    self.early_stopping_reached = False
    # training state
    self.training_state = TrainingState()

    self.evaluators = [s.lower() for s in dev_metrics.split(",") if s.strip()!=""]
    if schedule_metric.lower() not in self.evaluators:
              self.evaluators.append(schedule_metric.lower())
    if "loss" not in self.evaluators: self.evaluators.append("loss")

    self.reload_command = reload_command
    if reload_command is not None:
        self._augmentation_handle = None
        self._augment_data_initial()

    self.model = model
    self.corpus_parser = corpus_parser
    self.loss_calculator = loss_calculator or LossCalculator(MLELoss())
    self.pretrained_model_file = pretrained_model_file
    if self.pretrained_model_file:
      self.yaml_context.dynet_param_collection.load_from_data_file(self.pretrained_model_file + '.data')

    self.batcher = batcher or SrcBatcher(32)
    if src_format == "contvec":
      self.batcher.pad_token = np.zeros(self.model.src_embedder.emb_dim)
    self.pack_batches()
    self.logger = BatchLossTracker(self, dev_every)

    self.trainer = trainer or xnmt.optimizer.SimpleSGDTrainer(self.yaml_context, 0.1)
       
    self.schedule_metric = schedule_metric.lower()
    self.dynet_profiling = dynet_profiling

  def dependent_init_params(self, initialized_subcomponents):
    return [DependentInitParam(param_descr="model.src_embedder.vocab_size", value_fct=lambda: initialized_subcomponents["corpus_parser"].src_reader.vocab_size()),
            DependentInitParam(param_descr="model.decoder.vocab_size", value_fct=lambda: initialized_subcomponents["corpus_parser"].trg_reader.vocab_size()),
            DependentInitParam(param_descr="model.trg_embedder.vocab_size", value_fct=lambda: initialized_subcomponents["corpus_parser"].trg_reader.vocab_size()),
            DependentInitParam(param_descr="model.src_embedder.vocab", value_fct=lambda: initialized_subcomponents["corpus_parser"].src_reader.vocab),
            DependentInitParam(param_descr="model.trg_embedder.vocab", value_fct=lambda: initialized_subcomponents["corpus_parser"].trg_reader.vocab)]

  def pack_batches(self):
    self.train_src, self.train_trg = \
      self.batcher.pack(self.corpus_parser.get_training_corpus().train_src_data, self.corpus_parser.get_training_corpus().train_trg_data)
    self.dev_src, self.dev_trg = \
      self.batcher.pack(self.corpus_parser.get_training_corpus().dev_src_data, self.corpus_parser.get_training_corpus().dev_trg_data)

  def _augment_data_initial(self):
    augment_command = self.reload_command
    print('initial augmentation')
    if self._augmentation_handle is None:
      # first run
      self._augmentation_handle = Popen(augment_command + " --epoch 0", shell=True)
      self._augmentation_handle.wait()

  def _augment_data_next_epoch(self):
    augment_command = self.reload_command
    if self._augmentation_handle is None:
      # first run
      self._augmentation_handle = Popen(augment_command + " --epoch %d" % self.training_state.epoch_num, shell=True)
      self._augmentation_handle.wait()
   
    self._augmentation_handle.poll()
    retcode = self._augmentation_handle.returncode
    if retcode is not None:
      if self.training_state.epoch_num > 0:
        print('using reloaded data')
      # reload the data   
      self.corpus_parser._read_training_corpus(self.corpus_parser.training_corpus) # TODO: fix
      # restart data generation
      self._augmentation_handle = Popen(augment_command + " --epoch %d" % self.training_state.epoch_num, shell=True)
    else:
      print('new data set is not ready yet, using data from last epoch.')

  @register_xnmt_event
  def new_epoch(self, training_regimen, num_sents):
    pass

  def should_stop_training(self):
    return self.early_stopping_reached \
      or self.training_state.epoch_num > self.run_for_epochs \
      or (self.training_state.epoch_num == self.run_for_epochs and self.training_state.steps_into_epoch >= self.cur_num_minibatches()-1)
  
  def cur_num_minibatches(self):
    return len(self.train_src)
  
  def cur_num_sentences(self):
    return len(self.corpus_parser.training_corpus.train_src_data)
  
  def advance_epoch(self):
    if self.reload_command is not None:
      self._augment_data_next_epoch()
    self.training_state.epoch_seed = random.randint(1,2147483647)
    random.seed(self.training_state.epoch_seed)
    np.random.seed(self.training_state.epoch_seed)
    self.pack_batches()
    self.training_state.epoch_num += 1
    self.training_state.steps_into_epoch = 0
    self.minibatch_order = list(range(0, self.cur_num_minibatches()))
    np.random.shuffle(self.minibatch_order)
    self.new_epoch(training_regimen=self, num_sents=self.cur_num_sentences())
  
  def next_minibatch(self):
    while True:
      self.advance_epoch()
      for batch_num in self.minibatch_order:
        src = self.train_src[batch_num]
        trg = self.train_trg[batch_num]
        yield src, trg
        self.training_state.steps_into_epoch += 1
  
  def training_step(self, src, trg):
    loss_builder = LossBuilder()
    standard_loss = self.model.calc_loss(src, trg, self.loss_calculator)
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

    loss_value = loss_builder.compute()
    self.logger.update_epoch_loss(src, trg, loss_builder)
    self.logger.report_train_process()

    return loss_value
    
  def update_weights(self, loss):
    if self.dynet_profiling > 0:
      dy.print_text_graphviz()
    loss.backward()
    self.trainer.update()
    
  def run_training(self, update_weights=True):
    self.model.set_train(update_weights)
    for src,trg in self.next_minibatch():
      dy.renew_cg()
      loss = self.training_step(src, trg)
      if update_weights: self.update_weights(loss)
      if self.checkpoint_needed():
        if update_weights: self.model.set_train(False)
        self.checkpoint()
        if update_weights: self.model.set_train(True)
      if self.should_stop_training(): break
    

  def checkpoint_needed(self):
    return self.logger.should_report_dev()

  def checkpoint(self, out_ext=".dev_hyp", ref_ext=".dev_ref", encoding='utf-8'):
    self.model.set_train(False)
    self.logger.new_dev()
    trg_words_cnt, loss_score = self.compute_dev_loss() # forced decoding loss

    eval_scores = {"loss" : loss_score}
    if len(list(filter(lambda e: e!="loss", self.evaluators))) > 0:
      self.decode_args["src_file"] = self.corpus_parser.training_corpus.dev_src
      self.decode_args["candidate_id_file"] = self.corpus_parser.training_corpus.dev_id_file
      if self.model_file:
        out_file = self.model_file + out_ext
        out_file_ref = self.model_file + ref_ext
        self.decode_args["trg_file"] = out_file
      # Decoding + post_processing
      xnmt.xnmt_decode.xnmt_decode(model_elements=(self.corpus_parser, self.model), **self.decode_args)
      output_processor = xnmt.xnmt_decode.output_processor_for_spec(self.decode_args.get("post_process", "none")) # TODO: hack, refactor
      # Copy Trg to Ref
      processed = []
      with io.open(self.corpus_parser.training_corpus.dev_trg, encoding=encoding) as fin:
        for line in fin:
          processed.append(output_processor.words_to_string(line.strip().split()) + u"\n")
      with io.open(out_file_ref, 'wt', encoding=encoding) as fout:
        for line in processed:
          fout.write(line)
      # Evaluation
      if self.model_file:
        self.evaluate_args["hyp_file"] = out_file
        self.evaluate_args["ref_file"] = out_file_ref
      for evaluator in self.evaluators:
        if evaluator=="loss": continue
        self.evaluate_args["evaluator"] = evaluator
        eval_score = xnmt.xnmt_evaluate.xnmt_evaluate(**self.evaluate_args)
        eval_scores[evaluator] = eval_score
    # Logging
    if self.schedule_metric == "loss":
      self.logger.set_dev_score(trg_words_cnt, loss_score)
    else:
      self.logger.set_dev_score(trg_words_cnt, eval_scores[self.schedule_metric])

    print("> Checkpoint")
    # print previously computed metrics
    for metric in self.evaluators:
      if metric != self.schedule_metric:
        self.logger.report_auxiliary_score(eval_scores[metric])
    
    # Write out the model if it's the best one
    if self.logger.report_dev_and_check_model(self.model_file):
      if self.model_file is not None:
        YamlSerializer().save_to_file(self.model_file,
                                           self,
                                           self.yaml_context.dynet_param_collection)
      self.training_state.cur_attempt = 0
    else:
      # otherwise: learning rate decay / early stopping
      self.training_state.cur_attempt += 1
      if self.lr_decay < 1.0 and self.training_state.cur_attempt >= self.attempts_before_lr_decay:
        self.training_state.num_times_lr_decayed += 1
        if self.training_state.num_times_lr_decayed > self.lr_decay_times:
          print('  Early stopping')
          self.early_stopping_reached = True
        else:
          self.trainer.learning_rate *= self.lr_decay
          print('  new learning rate: %s' % self.trainer.learning_rate)
          if self.restart_trainer:
            print('  restarting trainer and reverting learned weights to best checkpoint..')
            self.trainer.restart()
            self.yaml_context.dynet_param_collection.revert_to_best_model()

    self.model.set_train(True)
    return

  def compute_dev_loss(self):
    loss_builder = LossBuilder()
    trg_words_cnt = 0
    for src, trg in zip(self.dev_src, self.dev_trg):
      dy.renew_cg()
      standard_loss = self.model.calc_loss(src, trg, self.loss_calculator)
      loss_builder.add_loss("loss", standard_loss)
      trg_words_cnt += self.logger.count_trg_words(trg)
      loss_builder.compute()
    return trg_words_cnt, LossScore(loss_builder.sum() / trg_words_cnt)

class TrainingState(object):
  def __init__(self):
    self.num_times_lr_decayed = 0
    self.cur_attempt = 0
    self.epoch_num = 0
    self.steps_into_epoch = 0
    self.epoch_seed = random.randint(1,2147483647) # used to pack and shuffle minibatches; storing helps resuming crashed trainings
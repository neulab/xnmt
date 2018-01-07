import dynet as dy

from xnmt.serializer import Serializable
from xnmt.loss_calculator import LossCalculator, MLELoss
from xnmt.evaluator import LossScore

class EvalTask:
  '''
  An EvalTask is a task that does evaluation and returns one or more EvalScore objects.
  '''
  def eval(self):
    raise NotImplementedError("EvalTask.eval needs to be implemented in child classes")

class LossEvalTask(Serializable):
  '''
  A task that does evaluation of the loss function.
  '''

  yaml_tag = u'!LossEvalTask'
  
  def __init__(self, model, loss_calculator=None, src_file=None, trg_file=None):
    print("Calling loss eval task constructor")
    self.model = model
    self.loss_calculator = loss_calculator or LossCalculator(MLELoss())    
    self.src_file = src_file
    self.trg_file = trg_file
    self.src_data = None
    self.trg_data = None
    
  def eval(self):
    if self.src_data == None:
      self.src_data = list(self.model.src_reader.read_sents(self.src_file))
    if self.trg_data == None:
      self.trg_data = list(self.model.trg_reader.read_sents(self.trg_file))
    loss_val = 0
    trg_words_cnt = 0
    for src, trg in zip(self.src_data, self.trg_data):
      dy.renew_cg()
      standard_loss = self.model.calc_loss(src, trg, self.loss_calculator)
      trg_words_cnt += self.model.trg_reader.count_words(trg)
      loss_val += standard_loss.value()
    return LossScore(loss_val / trg_words_cnt)

class AccuracyEvalTask(Serializable):
  '''
  A task that does evaluation of some measure of accuracy.
  '''

  yaml_tag = u'!AccuracyEvalTask'
  
  def __init__(self, model):
    self.model = model
    
  def eval(self, src, ref=None):
    raise NotImplementedError("AccuracyEvalTask not implemented")

from serializer import Serializable

class BilingualTrainingCorpus(Serializable):
  """
  A structure containing training and development sets for bilingual training
  """
  
  yaml_tag = "!BilingualTrainingCorpus"
  def __init__(self, train_src, train_trg, dev_src, dev_trg):
    self.train_src = train_src
    self.train_trg = train_trg
    self.dev_src = dev_src
    self.dev_trg = dev_trg
  

from serializer import Serializable

class BilingualTrainingCorpus(Serializable):
  """
  A structure containing training and development sets for bilingual training
  """

  yaml_tag = "!BilingualTrainingCorpus"
  def __init__(self, train_src, train_trg, dev_src, dev_trg, train_id_file=None, dev_id_file=None):
    self.train_src = train_src
    self.train_trg = train_trg
    self.train_id_file = train_id_file
    self.dev_src = dev_src
    self.dev_trg = dev_trg
    self.dev_id_file = dev_id_file



class CompoundSeqExpression(object):
  """ A class that represent a list of Expression Sequence. """

  def __init__(self, exprseq_list):
    self.exprseq_list = exprseq_list

  def __iter__(self):
    return iter(self.exprseq_list)

  def __getitem__(self, idx):
    return self.exprseq_list[idx]


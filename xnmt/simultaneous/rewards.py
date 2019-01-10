class SimultaneousReward(object):
  def __init__(self, src_batch, trg_batch, actions, outputs, trg_vocab):
    self.src_batch = src_batch
    self.trg_batch = trg_batch
    self.trg_vocab = trg_vocab
    self.actions = actions
    self.outputs = outputs
    
  def calculate(self):
    for inp, ref, action, output in zip(self.src_batch, self.trg_batch, self.actions, self.outputs):
      print(action)
      print(output)
      print(len(self.trg_vocab.i2w))
      print(" ".join([self.trg_vocab[x] for x in output]))
      print("--------------")


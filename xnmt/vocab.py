class Vocab:
  '''
  Converts between strings and integer ids
  '''

  def __init__(self):
    self.w2i = {}
    self.i2w = []

  def convert(self, w):
    if w not in self.w2i:
      self.w2i[w] = len(self.i2w)
      self.i2w.append(w)
    return self.w2i[w]

  def __getitem__(self, i):
    return self.i2w[i]

  def __len__(self):
    return len(self.i2w)

class Vocab:
  '''
  Converts between strings and integer ids
  '''

  SS = 0
  ES = 1

  def __init__(self, i2w=None):
    if (i2w is not None):
      self.i2w = i2w
      self.w2i = {word: id for (id, word) in enumerate(self.i2w)}
      return
    self.w2i = {}
    self.i2w = []
    self.frozen = False
    self.unk_token = None
    self.w2i['<s>'] = self.SS
    self.w2i['</s>'] = self.ES
    self.i2w.append('<s>')
    self.i2w.append('</s>')
    self.serialize_params = [self.i2w]

  def convert(self, w):
    if w not in self.w2i:
      if self.frozen:
        assert self.unk_token != None, 'Attempt to convert an OOV in a frozen vocabulary with no UNK token set'
        return self.unk_token
      self.w2i[w] = len(self.i2w)
      self.i2w.append(w)
    return self.w2i[w]

  def __getitem__(self, i):
    return self.i2w[i]

  def __len__(self):
    return len(self.i2w)

  def freeze(self):
    self.frozen = True

  def set_unk(self, w):
    assert self.frozen, 'Attempt to call set_unk on a non-frozen dict'
    if w not in self.w2i:
      self.w2i[w] = len(self.i2w)
      self.i2w.append(w)
    self.unk_token = self.w2i[w]

  def get_serializing_params(self):
    return self.i2w

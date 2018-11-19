class SentenceStats(object):
  """
  to Populate the src and trg sents statistics.
  """

  def __init__(self) -> None:
    self.src_stat = {}
    self.trg_stat = {}
    self.max_pairs = 1000000
    self.num_pair = 0

  class SourceLengthStat:
    def __init__(self) -> None:
      self.num_sents = 0
      self.trg_len_distribution = {}

  class TargetLengthStat:
    def __init__(self) -> None:
      self.num_sents = 0

  def add_sent_pair_length(self, src_length, trg_length):
    src_len_stat = self.src_stat.get(src_length, self.SourceLengthStat())
    src_len_stat.num_sents += 1
    src_len_stat.trg_len_distribution[trg_length] = \
      src_len_stat.trg_len_distribution.get(trg_length, 0) + 1
    self.src_stat[src_length] = src_len_stat

    trg_len_stat = self.trg_stat.get(trg_length, self.TargetLengthStat())
    trg_len_stat.num_sents += 1
    self.trg_stat[trg_length] = trg_len_stat

  def populate_statistics(self, train_corpus_src, train_corpus_trg):
    self.num_pair = min(len(train_corpus_src), self.max_pairs)
    for sent_num, (src, trg) in enumerate(zip(train_corpus_src, train_corpus_trg)):
      self.add_sent_pair_length(len(src), len(trg))
      if sent_num > self.max_pairs:
        return

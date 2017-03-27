class SentenceStats:
  '''
  to Populate the source and target sentences statistics.
  '''

  def __init__(self):
      self.source_stat = {}
      self.target_stat = {}
      self.max_pairs = 1000000
      self.num_pair = 0

  class SourceLengthStat:
      def __init__(self):
          self.num_sentences = 0
          self.target_len_distribution = {}

  class TargetLengthStat:
      def __init__(self):
          self.num_sentences = 0

  def add_sentence_pair_length(self, source_length, target_length):
      source_len_stat = self.source_stat.get(source_length, self.SourceLengthStat())
      source_len_stat.num_sentences += 1
      source_len_stat.target_len_distribution[target_length] = \
          source_len_stat.target_len_distribution.get(target_length, 0) + 1
      self.source_stat[source_length] = source_len_stat

      target_len_stat = self.target_stat.get(target_length, self.TargetLengthStat())
      target_len_stat.num_sentences += 1
      self.target_stat[target_length] = target_len_stat

  def populate_statistics(self, train_corpus_source, train_corpus_target):
      self.num_pair = min(len(train_corpus_source), self.max_pairs)
      for sent_num, (src, tgt) in enumerate(zip(train_corpus_source, train_corpus_target)):
          self.add_sentence_pair_length(len(src), len(tgt))
          if sent_num > self.max_pairs:
              return
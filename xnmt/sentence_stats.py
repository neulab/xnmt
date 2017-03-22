class SentenceStats:
  '''
  Converts between strings and integer ids
  '''

  def __init__(self):
      self.sourceStat = {}
      self.targetStat = {}
      self.maxPairs = 1000000

  class SourceLengthStat:
      def __init__(self):
          self.num_sentences = 0
          self.tarLenDistribution = {}

  class TargetLengthStat:
      def __init__(self):
          self.num_sentences = 0

  def addSentencePairLength(self, sourceLength, targetLength):
      source_stat = self.sourceStat.get(sourceLength, self.SourceLengthStat())
      source_stat.num_sentences += 1
      source_stat.tarLenDistribution[targetLength] = source_stat.tarLenDistribution.get(targetLength, 0) + 1
      self.sourceStat[sourceLength] = source_stat

      target_stat = self.targetStat.get(targetLength, self.TargetLengthStat())
      target_stat.num_sentences += 1
      self.targetStat[targetLength] = target_stat

  def populateStatistics(self, train_corpus_source, train_corpus_target):
      for sent_num, (src, tgt) in enumerate(zip(train_corpus_source, train_corpus_target)):
          self.addSentencePairLength(len(src), len(tgt))
          if sent_num > self.maxPairs:
              return



from typing import Any, Sequence
import numbers

class LevenshteinAligner(object):
  # gap penalty:
  gapPenalty = -1.0
  gapSymbol = None

  # similarity function:
  def sim(self, word1: Any, word2: Any) -> numbers.Number:
    """
    Similarity function between two words

    Args:
      word1:
      word2:

    Returns:
      Similarity measure
    """
    if word1 == word2:
      return 0
    else:
      return -1

  def align(self, l1: Sequence, l2: Sequence) -> tuple:
    """
    Perform edit distance alignment between two lists.

    Args:
      l1: first sequence
      l2: second sequence

    Returns:
      a tuple (c, x, y, s), with:
        c = score
        x = l1 with gapSymbol inserted to indicate insertions
        y = l2 with gapSymbol inserted to indicate deletions
        s = list of same length as x and y, specifying correct words, substitutions,
            deletions, insertions as 'c', 's', 'd', 'i'
    """
    # compute matrix
    dp_matrix = [[0] * (len(l2) + 1) for _ in range((len(l1) + 1))]
    for i in range(len(l1) + 1):
      dp_matrix[i][0] = i * self.gapPenalty
    for j in range(len(l2) + 1):
      dp_matrix[0][j] = j * self.gapPenalty
    for i in range(0, len(l1)):
      for j in range(0, len(l2)):
        match = dp_matrix[i][j] + self.sim(l1[i], l2[j])
        delete = dp_matrix[i][j + 1] + self.gapPenalty
        insert = dp_matrix[i + 1][j] + self.gapPenalty
        dp_matrix[i + 1][j + 1] = max(match, delete, insert)
    c = dp_matrix[len(l1)][len(l2)]
    x = []
    y = []
    i = len(l1) - 1
    j = len(l2) - 1
    while i >= 0 and j >= 0:
      score = dp_matrix[i + 1][j + 1]
      score_diag = dp_matrix[i][j]
      score_up = dp_matrix[i + 1][j]
      score_left = dp_matrix[i][j + 1]
      if score == score_left + self.gapPenalty:
        x = [l1[i]] + x
        y = [self.gapSymbol] + y
        i -= 1
      elif score == score_up + self.gapPenalty:
        x = [self.gapSymbol] + x
        y = [l2[j]] + y
        j -= 1
      else:
        assert score == score_diag + self.sim(l1[i], l2[j])
        x = [l1[i]] + x
        y = [l2[j]] + y
        i -= 1
        j -= 1
    while i >= 0:
      x = [l1[i]] + x
      y = [self.gapSymbol] + y
      i -= 1
    while j >= 0:
      x = [self.gapSymbol] + x
      y = [l2[j]] + y
      j -= 1
    s = []
    assert len(x) == len(y)
    for i in range(len(x)):
      if x[i] is self.gapSymbol and y[i] is not self.gapSymbol:
        s.append('i')
      elif x[i] is not self.gapSymbol and y[i] is self.gapSymbol:
        s.append('d')
      elif self.sim(x[i], y[i]) >= 0:
        s.append('c')
      else:
        s.append('s')
    return c, x, y, s

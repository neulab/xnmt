"""
Script to turn words corpus into characters corpus.
"""

import sys

for line in sys.stdin:
  words = line.strip().split()
  line = []
  for word in words:
    line.extend([c for c in word])
  print(" ".join(line))


#!/usr/bin/env python3

"""
By: Philip Arthur

Script to generate CHARAGRAM vocabulary.
For example if we have 2 words corpus: ["ab", "deac"]
Then it wil produce the char n gram vocabulary.

a
b
c
d
e
ab
de
ea
ac
dea
eac
deac

This is useful to be used in CharNGramSegmentComposer.

Args:
  ngram - The size of the ngram.
  top - Prin only the top ngram.
"""



import sys
import argparse
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("ngram", type=int, default=4)
parser.add_argument("--top", type=int, default=-1)
args = parser.parse_args()

k = args.ngram
counts = Counter()
for line in sys.stdin:
  words = line.strip().split()
  for word in words:
    for i in range(len(word)):
      for j in range(i+1, min(i+k+1, len(word)+1)):
        counts[word[i:j]] += 1

for i, (key, count) in enumerate(sorted(counts.items(), key=lambda x: -x[1])):
  if args.top != -1:
    if i == args.top:
      break
  print(key)



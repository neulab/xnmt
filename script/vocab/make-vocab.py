#!/usr/bin/env python3

"""
Simple script to generate vocabulary that can be used in most of the xnmt.input_readers

--min_count Is a filter based on count of words that need to be at least min_count to appear in the vocab.
--char_vocab Is to treat words as characters.

"""


import sys
import argparse
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--min_count", type=int, default=1)
parser.add_argument("--char_vocab", action="store_true")
args = parser.parse_args()

all_words = Counter()
for line in sys.stdin:
  words = line.strip().split()
  chars = []
  if args.char_vocab:
    for word in words:
      chars.extend([c for c in word])
    words = chars
  all_words.update(words)

if args.min_count > 1:
  all_words = [key for key, value in all_words.items() if value >= args.min_count]
else:
  all_words = list(all_words.keys())

for word in sorted(all_words):
  print(word)


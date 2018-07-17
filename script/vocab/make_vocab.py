#!/usr/bin/env python3

"""
Simple script to generate vocabulary that can be used in most of the xnmt.input_readers

--min_count Is a filter based on count of words that need to be at least min_count to appear in the vocab.

"""


import sys
import argparse
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--min_count", type=int, default=1)
args = parser.parse_args()

all_words = Counter()
for line in sys.stdin:
  all_words.update(line.strip().split())

if args.min_count > 1:
  all_words = [key for key, value in all_words.items() if value >= args.min_count]
else:
  all_words = list(all_words.keys())

for word in sorted(all_words):
  print(word)


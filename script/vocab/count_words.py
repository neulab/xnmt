#!/usr/bin/env python3

"""
Script to print the count of the words appearing in the corpus in descending order.

python3 script/vocab/count-vocab.py < [CORPUS]

"""


import sys
from collections import Counter

counts = Counter()
for line in sys.stdin:
  counts.update(line.strip().split())

for key, count in sorted(counts.items(), key=lambda x: -x[1]):
  print(key, count)


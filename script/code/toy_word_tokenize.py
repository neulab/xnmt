#! /usr/bin/env python

import sys
from nltk import word_tokenize

'''
This snippet requires the NLTK library, licensed under the Apache 2.0
license. The purpose of this snippet is to demonstrate external
tokenizer functionality; NLTK is not required to use xnmt.
'''

for line in sys.stdin:
  words = word_tokenize(line.decode('utf-8').strip())
  words = [w.encode('utf-8') for w in words]
  sys.stdout.write(' '.join(words) + '\n')

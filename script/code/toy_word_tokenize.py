#! /usr/bin/env python

import argparse
import sys
from nltk import word_tokenize

'''
This snippet requires the NLTK library, licensed under the Apache 2.0
license. The purpose of this snippet is to demonstrate external
tokenizer functionality; NLTK is not required to use xnmt.
'''

def tokenize(stream, characters=False):
  if characters:
    for line in stream:
      tokenized_line = u''
      if isinstance(line, str):
        line = line.decode('utf-8')
      tokenized_line = u' '.join(list(line.strip())) + u'\n'
      sys.stdout.write(tokenized_line.encode('utf-8'))
  else:
    for line in stream:
      words = word_tokenize(line.decode('utf-8').strip())
      words = [w.encode('utf-8') for w in words]
      sys.stdout.write(' '.join(words) + '\n')

if __name__ == '__main__':
  argp = argparse.ArgumentParser()
  argp.add_argument('--characters')
  args = argp.parse_args()
  tokenize(sys.stdin, args.characters)

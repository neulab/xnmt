# coding: utf-8
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import sys

EPOCH = 1
TRAIN_TEST = 2
PPL = 3
WORDS = 4
WORDS_SEC = 6
TIME = 7

regex = r'Epoch ([.0-9]+): (train|test)_ppl=([.0-9]+) \(words=(\d+), (words/sec=([.0-9]{2,4}), )?time=([0-9:\-]+)\)'


def plot_ppl_against_epoch(log_file):

  data = defaultdict(list)

  with open(log_file, 'r') as f:
    for line in f.readlines():
      matches = re.search(regex, line.rstrip())
      if matches:
        if matches.group(TRAIN_TEST) == 'train':
          data['epoch'].append(matches.group(EPOCH))
          data['train_ppl'].append(matches.group(PPL))
        else:
          data['test_ppl'].append(matches.group(PPL))

  plt.figure(1)
  plt.xlabel('epoch')
  plt.ylabel('perplexity')
  plt.plot(data['epoch'], data['train_ppl'])
  plt.plot(data['epoch'], data['test_ppl'])
  plt.legend(['train_ppl', 'test_ppl'], loc='upper right')
  plt.grid(True)
  plt.savefig(log_file + '.png')
  plt.show()


if __name__ == '__main__':

  if len(sys.argv) != 2:
    print('USAGE: python analyze_log.py <log_file>')
    exit()

  plot_ppl_against_epoch(sys.argv[1])

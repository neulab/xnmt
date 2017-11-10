from __future__ import print_function

import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_file")
parser.add_argument("output_character")
parser.add_argument("output_segmentation")
args = parser.parse_args()

with open(args.input_file) as file_fp, \
     open(args.output_character, "w") as char_out, \
     open(args.output_segmentation, "w") as seg_out:
  for line in file_fp:
    ctr = 0
    line = line.strip().split()
    char_line = []
    seg_line = []
    for word in line:
      char_line.append(" ".join(word))
      ctr += len(word)
      seg_line.append(str(ctr-1))
    print(" ".join(char_line), file=char_out)
    print(" ".join(seg_line), file=seg_out)


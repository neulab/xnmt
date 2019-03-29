import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("conll_input")
parser.add_argument("surface_output")
parser.add_argument("edge_output")
parser.add_argument("nt_output")
args = parser.parse_args()

surface, edge, nt = set(), set(), set()
with open(args.conll_input) as fp:
  for line in fp:
    col = line.strip().split("\t")
    if len(col) <= 1: continue
    surface.add(col[1])
    edge.add(col[-1])
    nt.add(col[3])

def write_output(vocab, file_fp):
  with open(file_fp, "w") as fp:
    for word in sorted(vocab):
      print(word, file=fp)

write_output(surface, args.surface_output)
write_output(edge, args.edge_output)
write_output(nt, args.nt_output)

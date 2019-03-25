import argparse
import xnmt.vocabs as vocabs
import xnmt.input_readers as input_readers

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("surface_vocab_file")
parser.add_argument("nt_vocab_file")
parser.add_argument("edg_vocab_file")
args = parser.parse_args()

reader = input_readers.CoNLLToRNNGActionsReader(surface_vocab=vocabs.Vocab(vocab_file=args.surface_vocab_file),
                                                nt_vocab=vocabs.Vocab(vocab_file=args.nt_vocab_file),
                                                edg_vocab=vocabs.Vocab(vocab_file=args.edg_vocab_file))

for tree in reader.read_sents(args.input):
  print(str(tree) + " NONE()")

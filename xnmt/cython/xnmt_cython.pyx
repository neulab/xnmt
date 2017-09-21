from libcpp.vector cimport vector
from xnmt.vocab import Vocab

# djb2 hash
cdef extern from "src/functions.h" namespace "xnmt":
  double evaluate_bleu_sentence(vector[int] ref, vector[int] hyp, int ngram, int smooth, int eos_sym)

def bleu_sentence(int ngram, int smooth, list ref, list hyp):
  return evaluate_bleu_sentence(ref, hyp, ngram, smooth, Vocab.ES)

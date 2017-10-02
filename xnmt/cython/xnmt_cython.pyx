from libcpp.vector cimport vector

cdef extern from "src/functions.h" namespace "xnmt":
  double evaluate_bleu_sentence(vector[int] ref, vector[int] hyp, int ngram, int smooth)

def bleu_sentence(int ngram, int smooth, list ref, list hyp):
  return evaluate_bleu_sentence(ref, hyp, ngram, smooth)

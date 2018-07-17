from libcpp.vector cimport vector

cdef extern from "src/functions.h" namespace "xnmt":
  double evaluate_bleu_sentence(vector[int] ref, vector[int] hyp, int ngram, int smooth)
  vector[int] binary_dense_from_sparse(vector[int] sparse, int length)

def bleu_sentence(int ngram, int smooth, list ref, list hyp):
  return evaluate_bleu_sentence(ref, hyp, ngram, smooth)

def dense_from_sparse(list sparse, int length):
  return binary_dense_from_sparse(sparse, length)

#pragma once

#include <iostream>
#include <vector>

namespace xnmt {

double evaluate_bleu_sentence(const std::vector<int>& ref, const std::vector<int>& hyp,
                              int ngram=4, int smooth=1);

std::vector<int> binary_dense_from_sparse(const std::vector<int>& sparse_vector, int length);

}

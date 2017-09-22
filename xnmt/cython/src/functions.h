#pragma once

#include <iostream>

namespace xnmt {

double evaluate_bleu_sentence(const std::vector<int>& ref, const std::vector<int>& hyp,
                              int ngram=4, int smooth=1);

}

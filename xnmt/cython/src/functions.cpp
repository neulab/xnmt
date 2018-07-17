#include "functions.h"
#include <iostream>
#include <unordered_map>
#include <deque>
#include <vector>
#include <math.h>

using std::vector;
using std::unordered_map;
using std::deque;

namespace xnmt {

typedef std::vector<std::unordered_map<size_t, int>> NGramStats;

// https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector
size_t deque_hash (const deque<int>& vec) {
  std::size_t seed = vec.size();
  for(int i : vec) {
    seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
  return seed;
}

NGramStats calculate_stats(const vector<int>& sents, unsigned int ngram) {
  NGramStats stat;
  for (size_t i=0; i < ngram; ++i) {
    deque<int> buffer;
    unordered_map<size_t, int> current_map; 
    for (size_t j=0; j < sents.size(); ++j) {
      int word = sents[j];
      buffer.push_back(word);
      size_t size = buffer.size();
      if (size == i+1) {
        size_t hash = deque_hash(buffer);
        auto it = current_map.find(hash);
        if (it == current_map.end()) {
          current_map.insert(std::make_pair(hash, 1));
        } else {
          ++(it->second);
        }
        buffer.pop_front();
      } 
    }
    stat.push_back(current_map);
  }
  return stat;
}

double evaluate_bleu_sentence(const vector<int>& ref, const vector<int>& hyp,
                              int ngram, int smooth) {
  NGramStats ref_stat = calculate_stats(ref, ngram);
  NGramStats hyp_stat = calculate_stats(hyp, ngram);
  double log_precision = 0;
  double log_bp = 0;
  for (int i=0; i < ngram; ++i) {
    auto ref_stat_i = ref_stat[i];
    auto hyp_stat_i = hyp_stat[i];
    int tp = 0;
    int denom = 0;
    for (auto ref_it=ref_stat_i.begin(); ref_it != ref_stat_i.end(); ++ref_it) {
      int ref_count = ref_it->second;
      int hyp_count = 0;
      auto hyp_it = hyp_stat_i.find(ref_it->first);
      if (hyp_it != hyp_stat_i.end()) {
        hyp_count = hyp_it->second;
      }
      tp += std::min(ref_count, hyp_count);
    }
    for (auto hyp_it=hyp_stat_i.begin(); hyp_it != hyp_stat_i.end(); ++hyp_it) {
      denom += hyp_it->second;
    }
    int s = i == 0 ? 0 : smooth;
    double lp = log((static_cast<double>(tp + s)) / (denom + s));
    log_precision += lp;
  }
  int len_hyp = hyp.size();
  int len_ref = ref.size();
  if (len_ref != 0 and len_hyp < len_ref) {
    log_bp = 1 - (static_cast<double>(len_ref) / len_hyp);
  } else {
    log_bp = 0;
  }
  return exp(log_precision / ngram + log_bp);
}

vector<int> binary_dense_from_sparse(const std::vector<int>& sparse_batch, int length) {
  vector<int> ret(length);
  for (int i : sparse_batch) {
    ret[i] = 1;
  }
  return ret;
}


}  // namespace xnmt

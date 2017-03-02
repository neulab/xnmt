import numpy as np
from collections import defaultdict, Counter
import math


class Evaluator(object):
    # Doc to be added
    def __init__(self, ref_corpus, can_corpus, ngram=4):
        """

        :param ref_corpus:
        :param can_corpus:
        :param ngram:
        """
        self.ngram = ngram
        #self.weights = (1/ngram) * np.ones(ngram, dtype=np.float32)
        self.reference_corpus = ref_corpus
        self.candidate_corpus = can_corpus

    # Doc to be added
    def BLEU(self):
        """

        :return:
        """
        assert(len(self.reference_corpus) == len(self.candidate_corpus)), \
            "Length of Reference Corpus and Candidate Corpus should be same"

        # Modified Precision Score
        clipped_ngram_count = Counter()
        candidate_ngram_count = Counter()

        # Brevity Penalty variables
        word_counter = Counter()

        for ref_sent, can_sent in zip(self.reference_corpus, self.candidate_corpus):
            word_counter['reference'] += len(ref_sent.split())
            word_counter['candidate'] += len(can_sent.split())

            clip_count_dict, full_count_dict = self.modified_precision(ref_sent, can_sent)

            for ngram_type in full_count_dict:
                if ngram_type in clip_count_dict:
                    clipped_ngram_count[ngram_type] += sum(clip_count_dict[ngram_type].values())
                else:
                    clipped_ngram_count[ngram_type] += 0. # This line may not be required

                candidate_ngram_count[ngram_type] += sum(full_count_dict[ngram_type].values())

        log_precision_score = 0.
        num_keys = len(candidate_ngram_count)
        weights = (1 / num_keys) * np.ones(num_keys, dtype=np.float32)

        # Precision Score Calculation
        for ngram_type in candidate_ngram_count:
            if clipped_ngram_count[ngram_type] > 0:
                log_precision_score += weights[ngram_type-1] * math.log(clipped_ngram_count[ngram_type] / candidate_ngram_count[ngram_type])

        precision_score = math.exp(log_precision_score)

        # Brevity Penalty Score
        brevity_penalty_score = self.brevity_penalty(word_counter['reference'], word_counter['candidate'])

        # BLEU Score
        bleu_score = brevity_penalty_score * precision_score
        return bleu_score

    # Doc to be added
    def brevity_penalty(self, r, c):
        """

        :param r:
        :param c:
        :return:
        """
        penalty = 1.
        if c <= r:
            penalty = np.exp(1. - (r / c))
        return penalty

    # Doc to be added
    def extract_ngrams(self, string):
        """
        Extracts ngram counts from the input string
        :param string: string for which the ngram is to be computed
        :return: a Counter object containing ngram counts
        """

        ngram_count = defaultdict(Counter)
        tokens = string.split()
        num_words = len(tokens)

        for i, first_token in enumerate(tokens[0: num_words]):
            for j in range(0, self.ngram):
                outer_range = i + j + 1
                ngram_type = j + 1
                if outer_range <= num_words:
                    ngram_tuple = tuple(tokens[i: outer_range])
                    ngram_count[ngram_type][ngram_tuple] += 1

        return ngram_count

    # Doc to be added
    def modified_precision(self, reference_sentence, candidate_sentence):
        """

        :param reference_sentence:
        :param candidate_sentence:
        :return:
        """

        clipped_ngram_count = defaultdict(Counter)

        reference_ngram_count = self.extract_ngrams(reference_sentence)
        candidate_ngram_count = self.extract_ngrams(candidate_sentence)

        for ngram_type in candidate_ngram_count:
            clipped_ngram_count[ngram_type] = candidate_ngram_count[ngram_type] & reference_ngram_count[ngram_type]

        return clipped_ngram_count, candidate_ngram_count


if __name__ == "__main__":

    candidate1 = "It is a guide to action which ensures that the military always obeys the commands of the party"
    reference1 = "It is a guide to action that ensures that the military will forever heed Party commands"

    candidate2 = "the the the the the the the"
    reference2 = "the cat is on the mat"

    candidate_corpus = [candidate2]
    reference_corpus = [reference2]

    obj = Evaluator(reference_corpus, candidate_corpus, ngram=4)

    print(obj.BLEU())

    # Test for NLTK
    # print(bleu_score.corpus_bleu([[reference2]], [candidate2]))
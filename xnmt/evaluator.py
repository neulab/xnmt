import numpy as np
from collections import defaultdict, Counter
import math
import warnings


class Evaluator:
    """
    A class to evaluate the quality of output.
    """

    def evaluate(self, ref, hyp):
        """
        Calculate the quality of output given a references.
        :param ref:
        :param hyp:
        :return:
        """
        raise NotImplementedError('evaluate must be implemented in Evaluator subclasses')


class BLEUEvaluator(Evaluator):
    # Doc to be added
    def __init__(self, ngram=4):
        """
        :param ref_corpus:
        :param can_corpus:
        :param ngram:
        """
        self.ngram = ngram
        self.weights = (1/ngram) * np.ones(ngram, dtype=np.float32)
        self.reference_corpus = None
        self.candidate_corpus = None

    # Doc to be added
    def evaluate(self, ref, hyp):
        """

        :rtype: object
        :param ref:
        :param hyp:
        :return:
        :return:
        """
        self.reference_corpus = ref
        self.candidate_corpus = hyp

        assert(len(self.reference_corpus) == len(self.candidate_corpus)), \
            "Length of Reference Corpus and Candidate Corpus should be same"

        # Modified Precision Score
        clipped_ngram_count = Counter()
        candidate_ngram_count = Counter()

        # Brevity Penalty variables
        word_counter = Counter()

        for ref_sent, can_sent in zip(self.reference_corpus, self.candidate_corpus):
            word_counter['reference'] += len(ref_sent)
            word_counter['candidate'] += len(can_sent)

            clip_count_dict, full_count_dict = self.modified_precision(ref_sent, can_sent)

            for ngram_type in full_count_dict:
                if ngram_type in clip_count_dict:
                    clipped_ngram_count[ngram_type] += sum(clip_count_dict[ngram_type].values())
                else:
                    clipped_ngram_count[ngram_type] += 0. # This line may not be required

                candidate_ngram_count[ngram_type] += sum(full_count_dict[ngram_type].values())

        # Edge case
        # Return 0 if there are no matching n-grams
        # If there are no unigrams, return BLEU score of 0
        # No need to check for higher order n-grams
        if clipped_ngram_count[1] == 0:
            return 0.

        log_precision_score = 0.
        # Precision Score Calculation
        for ngram_type in range(1, self.ngram+1):
            if clipped_ngram_count[ngram_type] == 0:
                warning_msg = "Count of {}-gram is 0. Will lead to incorrect BLEU scores".format(ngram_type)
                warnings.warn(warning_msg)
                break
            else:
                log_precision_score += self.weights[ngram_type-1] * \
                                       math.log(clipped_ngram_count[ngram_type] / candidate_ngram_count[ngram_type])

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

        # If candidate sentence length is 0 (empty), return 0.
        if c == 0:
            return 0.
        elif c <= r:
            penalty = np.exp(1. - (r / c))
        return penalty

    # Doc to be added
    def extract_ngrams(self, tokens):
        """
        Extracts ngram counts from the input string
        :param tokens: tokens of string for which the ngram is to be computed
        :return: a Counter object containing ngram counts
        """

        ngram_count = defaultdict(Counter)
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

    # Example 1
    reference1 = "It is a guide to action that ensures that the military will forever heed Party commands".split()
    candidate1 = "It is a guide to action which ensures that the military always obeys the commands of the party".split()

    obj = BLEUEvaluator(ngram=4)
    print("xnmt bleu score :"), print(obj.evaluate([reference1], [candidate1]))
    # print("nltk BLEU scores"), print(corpus_bleu([[reference1]], [candidate1]))

    # Example 2
    reference2 = "the cat is on the mat".split()
    candidate2 = "the the the the the the the".split()

    # Generates a warning because of no 2-grams and beyond
    obj = BLEUEvaluator(ngram=4)
    print("xnmt bleu score :"), print(obj.evaluate([reference2], [candidate2]))
    # print("nltk BLEU scores"), print(corpus_bleu([[reference2]], [candidate2]))

    # Example 3 (candidate1 + candidate3)
    reference3 = "he was interested in world history because he read the book".split()
    candidate3 = "he read the book because he was interested in world history".split()
    obj = BLEUEvaluator(ngram=4)
    print("xnmt bleu score :"), print(obj.evaluate([reference1, reference3], [candidate1, candidate3]))
    # print("nltk BLEU scores"), print(corpus_bleu([[reference1], [reference3]],
    #                                              [candidate1, candidate3]))

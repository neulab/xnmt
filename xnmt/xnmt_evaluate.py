import argparse
from evaluator import BLEUEvaluator


def read_data(loc_):
    """Reads the lines in the file specified in loc_ and return the list after inserting the tokens
    """
    data = list()
    with open(loc_) as fp:
        for line in fp:
            t = line.split()
            data.append(t)
    return data
    

def xnmt_evaluate(args):
    """"Returns the BLEU score of the hyp sentences using reference target sentences
    """
    
    ref_corpus = read_data(args.ref_file)
    hyp_corpus = read_data(args.target_file)

    model = BLEUEvaluator(ngram=4)
    bleu_score = model.evaluate(ref_corpus, hyp_corpus)
    
    return bleu_score
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("This script performs evaluation using BLEU score metric \
                                     between the reference and candidate (hypothesis) target files ")

    parser.add_argument('ref_file',
                        type=str,
                        help='path of the reference file')

    parser.add_argument('target_file',
                        type=str,
                        help='path of the hypothesis target file')
    
    args = parser.parse_args()
    
    bleu_score = xnmt_evaluate(args)
    print("BLEU Score = {}".format(bleu_score))

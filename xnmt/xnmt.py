# coding: utf-8

import dynet as dy
from serializer import JSONSerializer
import argparse
from input import *
from output import *
from encoder import *
from model_params import *
from decoder import *
from translator import *
from serializer import *
'''
This will be the main class to perform decoding.
'''

def xnmt_decode(args, search_strategy=BeamSearch(1, len_norm=NoNormalization())):
  model = dy.Model()
  model_serializer = JSONSerializer()
  model_params = model_serializer.load_from_file(args.model, model)
  # Perform decoding

  source_vocab = Vocab(model_params.source_vocab)
  input_reader = PlainTextReader(source_vocab)
  input_reader.freeze()
  source_corpus = input_reader.read_file(args.source_file)

  output_generator = PlainTextOutput()
  target_vocab = Vocab(model_params.target_vocab)
  output_generator.load_vocab(target_vocab)

  translator = DefaultTranslator(model_params.encoder, model_params.attender, model_params.decoder)

  for src in source_corpus:
    token_string = translator.translate(src, search_strategy)
    print output_generator.process(token_string)


if __name__ == "__main__":
  # Parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str)
  parser.add_argument('source_file')
  parser.add_argument('target_file')
  args = parser.parse_args()
  # Load model
  xnmt_decode(args)


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
import sys
from options import OptionParser, Option


'''
This will be the main class to perform decoding.
'''

options = [
  Option("model_file", force_flag=True, required=True, help="pretrained (saved) model path"),
  Option("source_file", help="path of input source file to be translated"),
  Option("target_file", help="path of file where expected target translatons will be written"),
  Option("input_type", default_value="word")
]


def xnmt_decode(args, search_strategy=BeamSearch(1, len_norm=NoNormalization())):
  model = dy.Model()
  model_serializer = JSONSerializer()
  model_params = model_serializer.load_from_file(args.model_file, model)
  # Perform decoding

  source_vocab = Vocab(model_params.source_vocab)
  input_reader = InputReader.create_input_reader(args.input_type, source_vocab)
  input_reader.freeze()
  source_corpus = input_reader.read_file(args.source_file)

  output_generator = PlainTextOutput()
  target_vocab = Vocab(model_params.target_vocab)
  output_generator.load_vocab(target_vocab)

  translator = DefaultTranslator(model_params.encoder, model_params.attender, model_params.decoder)

  with open(args.target_file, 'w') as fp:  # Saving the translated output to a target file
    for src in source_corpus:
      token_string = translator.translate(src, search_strategy)
      target_sentence = output_generator.process(token_string)[0]

      if isinstance(target_sentence, unicode):
        target_sentence = target_sentence.encode('utf8')

      else:  # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
        target_sentence = unicode(target_sentence, 'utf8', errors='ignore').encode('utf8')

      fp.write(target_sentence + u'\n')



if __name__ == "__main__":
  # Parse arguments
  parser = OptionParser()
  parser.add_task("decode", options)
  args = parser.args_from_command_line("decode", sys.argv[1:])
  # Load model
  xnmt_decode(args)


# coding: utf-8

from output import *
from serializer import *
import codecs
import sys
from options import OptionParser, Option
from io import open

'''
This will be the main class to perform decoding.
'''

options = [
  Option("model_file", force_flag=True, required=True, help="pretrained (saved) model path"),
  Option("source_file", help="path of input source file to be translated"),
  Option("target_file", help="path of file where expected target translatons will be written"),
  Option("input_format", default_value="text", help="format of input data: text/contvec"),
  Option("post_process", default_value="none", help="post-processing of translation outputs: none/join-char/join-bpe"),
  Option("beam", int, default_value=1),
  Option("max_len", int, default_value=100),
]


def xnmt_decode(args, model_elements=None):
  """
  :param model_elements: If None, the model will be loaded from args.model_file. If set, should
  equal (source_vocab, target_vocab, translator).
  """
  if model_elements is None:
    model = dy.Model()
    model_serializer = JSONSerializer()
    model_params = model_serializer.load_from_file(args.model_file, model)

    source_vocab = Vocab(model_params.source_vocab)
    target_vocab = Vocab(model_params.target_vocab)

    translator = DefaultTranslator(model_params.encoder, model_params.attender, model_params.decoder)

  else:
    source_vocab, target_vocab, translator = model_elements

  input_reader = InputReader.create_input_reader(args.input_format, source_vocab)
  input_reader.freeze()

  if args.post_process=="none":
    output_generator = PlainTextOutput()
  elif args.post_process=="join-char":
    output_generator = JoinedCharTextOutput()
  elif args.post_process=="join-bpe":
    output_generator = JoinedBPETextOutput()
  else:
    raise RuntimeError("Unkonwn postprocessing argument {}".format(args.postprocess)) 
  output_generator.load_vocab(target_vocab)

  source_corpus = input_reader.read_file(args.source_file)
  
  search_strategy=BeamSearch(b=args.beam, max_len=args.max_len, len_norm=NoNormalization())

  # Perform decoding

  with open(args.target_file, 'wb') as fp:  # Saving the translated output to a target file
    for src in source_corpus:
      dy.renew_cg()
      token_string = translator.translate(src, search_strategy)
      target_sentence = output_generator.process(token_string)[0]

      if isinstance(target_sentence, unicode):
        target_sentence = target_sentence.encode('utf-8', errors='ignore')

      else:  # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
        #target_sentence = unicode(target_sentence, 'utf-8', errors='ignore').encode('utf-8', errors='ignore')
        target_sentence = target_sentence.decode('utf-8', errors='ignore').encode('utf-8', errors='ignore')

      fp.write(target_sentence + '\n')


if __name__ == "__main__":
  # Parse arguments
  parser = OptionParser()
  parser.add_task("decode", options)
  args = parser.args_from_command_line("decode", sys.argv[1:])
  # Load model
  xnmt_decode(args)


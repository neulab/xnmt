# coding: utf-8

from output import *
from serializer import *
import codecs
import sys
from search_strategy import *
from options import OptionParser, Option
from io import open
import length_normalization

'''
This will be the main class to perform decoding.
'''

options = [
  Option("dynet-mem", int, required=False),
  Option("dynet-gpu-ids", int, required=False),
  Option("model_file", force_flag=True, required=True, help_str="pretrained (saved) model path"),
  Option("src_file", help_str="path of input src file to be translated"),
  Option("trg_file", help_str="path of file where expected trg translatons will be written"),
  Option("max_src_len", int, required=False, help_str="Remove sentences from data to decode that are longer than this on the source side"),
  Option("input_format", default_value="text", help_str="format of input data: text/contvec"),
  Option("post_process", default_value="none", help_str="post-processing of translation outputs: none/join-char/join-bpe"),
  Option("beam", int, default_value=1),
  Option("max_len", int, default_value=100),
  Option("len_norm_type", str, default_value="NoNormalization"),
  Option("len_norm_params", dict, default_value={}),
]

NO_DECODING_ATTEMPTED = u"@@NO_DECODING_ATTEMPTED@@"

def xnmt_decode(args, model_elements=None):
  """
  :param model_elements: If None, the model will be loaded from args.model_file. If set, should
  equal (corpus_parser, translator).
  """
  if model_elements is None:
    raise RuntimeError("xnmt_decode with model_element=None needs to be updated to run with the new YamlSerializer")
    model = dy.Model()
    model_serializer = JSONSerializer()
    model_params = model_serializer.load_from_file(args.model_file, model)

    src_vocab = Vocab(model_params.src_vocab)
    trg_vocab = Vocab(model_params.trg_vocab)

    translator = DefaultTranslator(model_params.input_embedder, model_params.encoder, model_params.attender, model_params.output_embedder, model_params.decoder)

  else:
    corpus_parser, translator = model_elements

  if args.post_process=="none":
    output_generator = PlainTextOutput()
  elif args.post_process=="join-char":
    output_generator = JoinedCharTextOutput()
  elif args.post_process=="join-bpe":
    output_generator = JoinedBPETextOutput()
  else:
    raise RuntimeError("Unknown postprocessing argument {}".format(args.postprocess)) 
  output_generator.load_vocab(corpus_parser.trg_reader.vocab)

  src_corpus = corpus_parser.src_reader.read_file(args.src_file)
  
  len_norm_type = getattr(length_normalization, args.len_norm_type)
  search_strategy=BeamSearch(b=args.beam, max_len=args.max_len, len_norm=len_norm_type(**args.len_norm_params))

  # Perform decoding

  translator.set_train(False)
  with open(args.trg_file, 'wb') as fp:  # Saving the translated output to a trg file
    for src in src_corpus:
      if args.max_src_len is not None and len(src) > args.max_src_len:
        trg_sent = NO_DECODING_ATTEMPTED
      else:
        dy.renew_cg()
        token_string = translator.translate(src, search_strategy)
        trg_sent = output_generator.process(token_string)[0]

      assert isinstance(trg_sent, unicode), "Expected unicode as translator output, got %s" % type(trg_sent)
      trg_sent = trg_sent.encode('utf-8', errors='ignore')

      fp.write(trg_sent + '\n')


if __name__ == "__main__":
  # Parse arguments
  parser = OptionParser()
  parser.add_task("decode", options)
  args = parser.args_from_command_line("decode", sys.argv[1:])
  # Load model
  xnmt_decode(args)


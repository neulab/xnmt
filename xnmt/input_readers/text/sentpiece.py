from xnmt import output
from xnmt.sent import SimpleSentence
from xnmt.vocabs import Vocab
from xnmt.input_readers.text.base import BaseTextReader
from xnmt.persistence import Serializable, serializable_init
from xnmt.events import register_xnmt_handler, handle_xnmt_event


class SentencePieceTextReader(BaseTextReader, Serializable):
  """
  Read in text and segment it with sentencepiece. Optionally perform sampling
  for subword regularization, only at training time.
  https://arxiv.org/pdf/1804.10959.pdf
  """
  yaml_tag = '!SentencePieceTextReader'

  @register_xnmt_handler
  @serializable_init
  def __init__(self, model_file, sample_train=False, subword_sample=-1, alpha=0.1, vocab=None,
               output_proc=[output.JoinPieceTextOutputProcessor]):
    """
    Args:
      model_file: The sentence piece model file
      sample_train: On the training set, sample outputs
      subword_sample: The "l" parameter for subword regularization, how many sentences to sample
      alpha: The "alpha" parameter for subword regularization, how much to smooth the distribution
      vocab: The vocabulary
      output_proc: output processors to revert the created sentences back to a readable string
    """
    import sentencepiece as spm
    self.subword_model = spm.SentencePieceProcessor()
    self.subword_model.Load(model_file)
    self.sample_train = sample_train
    self.subword_sample = subword_sample
    self.alpha = alpha
    self.vocab = vocab
    self.train = False
    self.output_procs = output.OutputProcessor.get_output_processor(output_proc)

  @handle_xnmt_event
  def on_set_train(self, val):
    self.train = val

  def read_sent(self, line, idx):
    if self.sample_train and self.train:
      words = self.subword_model.SampleEncodeAsPieces(line.strip(), self.subword_sample, self.alpha)
    else:
      words = self.subword_model.EncodeAsPieces(line.strip())
    words = [w.decode('utf-8') for w in words]
    return SimpleSentence(idx=idx,
                          words=[self.vocab.convert(word) for word in words] + [self.vocab.convert(Vocab.ES_STR)],
                          vocab=self.vocab,
                          output_procs=self.output_procs)

  def vocab_size(self):
    return len(self.vocab)

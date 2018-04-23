import dynet as dy
import numpy as np
import itertools

# Reporting purposes
from simple_settings import settings

import xnmt.batcher
from xnmt.linear import Linear
from xnmt.batcher import mark_as_batch, is_batched
from xnmt.decoder import MlpSoftmaxDecoder
from xnmt.embedder import SimpleWordEmbedder
from xnmt.generator import GeneratorModel
from xnmt.inference import SimpleInference
from xnmt.input import SimpleSentenceInput
from xnmt.loss import LossBuilder
from xnmt.lstm import BiLSTMSeqTransducer
from xnmt.output import TextOutput
from xnmt.vocab import Vocab
from xnmt.persistence import Ref, bare, Path, Serializable

class BOWPredictor(GeneratorModel, Serializable):
  yaml_tag = u'!BOWPredictor'

  def __init__(self, exp_global=Ref(Path("exp_global")),
               src_reader=None, trg_reader=None,
               src_embedder=bare(SimpleWordEmbedder),
               encoder=bare(BiLSTMSeqTransducer),
               inference=bare(SimpleInference),
               encoder_hidden_dim=None,
               weigh_loss=False,
               src_vocab=Ref(Path("model.src_reader.vocab")),
               trg_vocab=Ref(Path("model.trg_reader.vocab"))):
    register_handler(self)
    self.src_reader = src_reader
    self.trg_reader = trg_reader
    self.src_embedder = src_embedder
    self.encoder = encoder
    self.inference = inference
    self.src_vocab = src_vocab
    self.trg_vocab = trg_vocab
    self.weigh_loss = weigh_loss
    encoder_hidden_dim = encoder_hidden_dim or exp_global.default_layer_dim

    self.bow_projector = Linear(encoder_hidden_dim,
                                len(self.trg_reader.vocab),
                                exp_global.dynet_param_collection.param_col)

  def set_post_processor(self, post_processor):
    self.post_processor = post_processor

  def set_trg_vocab(self, trg_vocab=None):
    self.trg_vocab = trg_vocab

  def get_primary_loss(self):
    return "bow"

  def initialize_generator(self, **kwargs):
    self.report_path = kwargs.get("report_path", None)
    self.report_type = kwargs.get("report_type", None)

  def calc_loss(self, src, trg, loss_calc=False, infer_prediction=False, *args, **kwargs):
    self.start_sent(src)
    embeddings = self.src_embedder.embed_sent(src)
    encodings = self.encoder(embeddings)

    encoding_tensor = encodings.as_tensor()
    bow_prediction = dy.logistic(self.bow_projector(encoding_tensor))
    if encodings.mask is not None:
      mask = dy.transpose(dy.inputTensor(encodings.mask.get_active_one_mask().transpose(), batched=True))
      bow_prediction = dy.cmult(bow_prediction, mask)
    if len(bow_prediction.dim()[0]) != 1:
      bow_prediction_sum = dy.sum_dim(bow_prediction, [1])
    else:
      bow_prediction_sum = bow_prediction
    if infer_prediction:
      return np.round(bow_prediction_sum.npvalue().transpose()).astype(int)
    else:
      assert trg is not None
      # Turn TRG into vectors
      trg_bow = []
      #   Needs to be fixed and use sparse_inputTensor instead
      #    bow = [trg[i].annotation["bow"] for i in range(len(trg))]
      #    key = [list(bow[i].keys()) for i in range(len(bow))]
      #    val = [list(bow[i].values()) for i in range(len(bow))]
      #    tensor = dy.sparse_inputTensor(key, val, shape=(len(self.trg_reader.vocab), len(trg)), batched=True)
      for i in range(len(trg)):
        bow_vct = [0 for _ in range(len(self.trg_reader.vocab))]
        for key, val in trg[i].annotation["bow"].items():
          bow_vct[key] = val
        trg_bow.append(bow_vct)
      trg_bow = dy.inputTensor(np.asarray(trg_bow).transpose(), batched=True)
      # Calculate loss
      loss = self.weighted_sd(bow_prediction_sum, trg_bow)
      return LossBuilder({"bow": loss})

  def weighted_sd(self, v1, v2):
    if self.weigh_loss:
      raise NotImplementedError()
    else:
      return dy.squared_distance(v1, v2)

  def generate(self, src, idx, src_mask=None, forced_trg_ids=None):
    self.start_sent(src)
    if not xnmt.batcher.is_batched(src):
      src = xnmt.batcher.mark_as_batch([src])
    else:
      assert src_mask is not None
    outputs = []

    for sents in src:
      dy.renew_cg(immediate_compute=settings.IMMEDIATE_COMPUTE, check_validity=settings.CHECK_VALIDITY)
      predicted_bow = self.calc_loss(src, trg=None, infer_prediction=True)
      output_actions = []
      for key in np.nonzero(predicted_bow)[0]:
        output_actions.extend([key] * predicted_bow[key])
      # Append output to the outputs
      outputs.append(TextOutput(actions=output_actions, vocab=self.trg_vocab))

    return outputs

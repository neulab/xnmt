import dynet as dy
import numpy as np
import itertools

# Reporting purposes
from lxml import etree
from simple_settings import settings

from xnmt.attender import MlpAttender
from xnmt.batcher import mark_as_batch, is_batched
from xnmt.decoder import MlpSoftmaxDecoder
from xnmt.embedder import SimpleWordEmbedder
from xnmt.events import register_xnmt_event_assign, handle_xnmt_event, register_handler
from xnmt.generator import GeneratorModel
from xnmt.hyper_parameters import multiply_weight
from xnmt.inference import SimpleInference
from xnmt.input import SimpleSentenceInput
import xnmt.length_normalization
from xnmt.loss import LossBuilder
from xnmt.lstm import BiLSTMSeqTransducer
from xnmt.output import TextOutput
import xnmt.plot
from xnmt.reports import Reportable
from xnmt.serialize.serializable import Serializable, bare
from xnmt.search_strategy import BeamSearch, GreedySearch
import xnmt.serialize.serializer
from xnmt.serialize.tree_tools import Ref, Path
from xnmt.vocab import Vocab

class BOWPredictor(GeneratorModel, Serializable):
  yaml_tag = u'!BOWPredictor'

  def __init__(self, exp_global=Ref(Path("exp_global")),
               src_reader=None, trg_reader=None,
               src_embedder=bare(SimpleWordEmbedder),
               encoder=bare(BiLSTMSeqTransducer),
               inference=bare(SimpleInference), encoder_hidden_dim=None):
    register_handler(self)
    self.src_reader = src_reader
    self.trg_reader = trg_reader
    self.src_embedder = src_embedder
    self.encoder = encoder
    self.inference = inference
    encoder_hidden_dim = encoder_hidden_dim or exp_global.default_layer_dim

    self.bow_projector = xnmt.linear.Linear(encoder_hidden_dim,
                                            len(self.trg_reader.vocab),
                                            exp_global.dynet_param_collection.param_col)
  
  def set_trg_vocab(self, trg_vocab=None):
    self.trg_vocab = trg_vocab

  def get_primary_loss(self):
    return "bow"

  def initialize_generator(self, **kwargs):
    self.report_path = kwargs.get("report_path", None)
    self.report_type = kwargs.get("report_type", None)

  def calc_loss(self, src, trg,  *args, **kwargs):
    self.start_sent(src)
    embeddings = self.src_embedder.embed_sent(src)
    encodings = self.encoder(embeddings)

    # Turn TRG into vectors
    trg_bow = []
    for i in range(len(trg)):
      bow_vct = [0 for _ in range(len(self.trg_reader.vocab))]
      for key, val in trg[i].annotation["bow"].items():
        bow_vct[key] = val
      trg_bow.append(bow_vct)
    trg_bow = dy.inputTensor(np.asarray(trg_bow).transpose(), batched=True)

#   Needs to be fixed and use sparse_inputTensor instead
#    bow = [trg[i].annotation["bow"] for i in range(len(trg))]
#    key = [list(bow[i].keys()) for i in range(len(bow))]
#    val = [list(bow[i].values()) for i in range(len(bow))]
#    tensor = dy.sparse_inputTensor(key, val, shape=(len(self.trg_reader.vocab), len(trg)), batched=True)

    encoding_tensor = encodings.as_tensor()
    # Double transpose operation seems silly, but this is to prevent dynet deleting one dimension if sequence length = 1
    bow_prediction = dy.transpose(dy.transpose(dy.rectify(self.bow_projector(encoding_tensor))))
    if encodings.mask is not None:
      mask = dy.transpose(dy.inputTensor(encodings.mask.get_active_one_mask().transpose(), batched=True))
      bow_prediction = dy.cmult(bow_prediction, mask)
    bow_prediction = dy.sum_dim(bow_prediction, [1])
    loss = dy.squared_distance(bow_prediction, trg_bow)

    return LossBuilder({"bow": loss})


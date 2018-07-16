import dynet as dy
from typing import List

from xnmt.embedder import Embedder
from xnmt.expression_sequence import ExpressionSequence
from xnmt.param_collection import ParamManager
from xnmt.param_init import ParamInitializer, GlorotInitializer, ZeroInitializer
from xnmt.persistence import Serializable
from xnmt.persistence import serializable_init, Serializable, bare, Ref
from xnmt.transducer import SeqTransducer, FinalTransducerState

class PositionEmbedder(Embedder, Serializable):

  yaml_tag = '!PositionEmbedder'

  @serializable_init
  def __init__(self, max_pos: int, emb_dim: int = Ref("exp_global.default_layer_dim"),
               param_init: ParamInitializer = Ref("exp_global.param_init", default=bare(GlorotInitializer))):
    """
    max_pos: largest embedded position
    emb_dim: embedding size
    param_init: how to initialize embedding matrix
    """
    self.max_pos = max_pos
    self.emb_dim = emb_dim
    param_collection = ParamManager.my_params(self)
    param_init = param_init
    dim = (self.emb_dim, max_pos)
    self.embeddings = param_collection.add_parameters(dim, init=param_init.initializer(dim, is_lookup=True))

  def embed(self, word): raise NotImplementedError("Position-embedding for individual words not implemented yet.")
  def embed_sent(self, sent_len):
    embeddings = dy.strided_select(dy.parameter(self.embeddings), [1,1], [0,0], [self.emb_dim, sent_len])
    return ExpressionSequence(expr_tensor=embeddings, mask=None)

# Note: alternatively, this could wrap "PositionEmbedder", but it seems to me
#       that PositionEmbedder is probably not necessary in the first place, so
#       it probably makes more sense to have this as a SeqTransducer that
#       adds positional embeddings to an input
class PositionalSeqTransducer(SeqTransducer, Serializable):
  yaml_tag = '!PositionalSeqTransducer'

  @serializable_init
  def __init__(self,
               max_pos: int,
               op: str = 'sum',
               emb_type: str = 'param',
               input_dim: int = Ref("exp_global.default_layer_dim"),
               param_init: ParamInitializer = Ref("exp_global.param_init", default=bare(GlorotInitializer))):
    """
    max_pos: largest embedded position
    op: how to combine positional encodings with the original encodings, can be "sum" or "concat"
    type: what type of embddings to use, "param"=parameterized (others, such as the trigonometric embeddings are todo)
    input_dim: embedding size
    param_init: how to initialize embedding matrix
    """
    self.max_pos = max_pos
    self.input_dim = input_dim
    self.op = op
    self.emb_type = emb_type
    param_init = param_init
    dim = (self.input_dim, max_pos)
    param_collection = ParamManager.my_params(self)
    self.embedder = param_collection.add_parameters(dim, init=param_init.initializer(dim, is_lookup=True))

  def get_final_states(self) -> List[FinalTransducerState]:
    return self._final_states

  def transduce(self, src: ExpressionSequence) -> ExpressionSequence:
    sent_len = len(src)
    embeddings = dy.strided_select(dy.parameter(self.embedder), [1,1], [0,0], [self.input_dim, sent_len])
    if self.op == 'sum':
      output = embeddings + src.as_tensor()
    elif self.op == 'concat':
      output = dy.concatenate([embeddings, src.as_tensor()])
    else:
      raise ValueError(f'Illegal op {op} in PositionalTransducer (options are "sum"/"concat")')
    output_seq = ExpressionSequence(expr_tensor=output, mask=src.mask)
    self._final_states = [FinalTransducerState(output_seq[-1])]
    return output_seq
